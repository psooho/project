import logging
import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarksConnections

logger = logging.getLogger("landmarks")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
)
_landmarker = vision.FaceLandmarker.create_from_options(_options)
# MediaPipe의 IMAGE 모드 태스크는 스레드 안전하지 않다. 요청은 스레드풀에서 처리되므로
# 전역 인스턴스를 여러 스레드가 동시에 건드리지 않도록 직렬화한다.
_landmarker_lock = threading.Lock()

# 검출 재시도 시 덧댈 여백 비율 (0 = 원본 그대로 먼저 시도).
_DETECT_PAD_RATIOS = (0.0, 0.25, 0.5)


def _unique_indices(connections) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for connection in connections:
        for idx in (connection.start, connection.end):
            if idx not in seen:
                seen.add(idx)
                ordered.append(idx)
    return ordered


def _ring_indices(connections) -> list[int]:
    """Order a closed-loop connection set (e.g. face oval) into a walkable ring
    so it can be filled as a polygon."""
    adjacency: dict[int, set[int]] = {}
    for connection in connections:
        adjacency.setdefault(connection.start, set()).add(connection.end)
        adjacency.setdefault(connection.end, set()).add(connection.start)

    start = connections[0].start
    ring = [start]
    prev, current = None, start
    while True:
        neighbors = adjacency[current] - ({prev} if prev is not None else set())
        next_node = next(iter(neighbors))
        if next_node == start:
            break
        ring.append(next_node)
        prev, current = current, next_node
    return ring


FACE_OVAL_RING = _ring_indices(FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL)
LEFT_EYE_IDX = _unique_indices(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE)
RIGHT_EYE_IDX = _unique_indices(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE)
LEFT_EYEBROW_IDX = _unique_indices(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW)
RIGHT_EYEBROW_IDX = _unique_indices(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW)
EYEBROW_IDX = LEFT_EYEBROW_IDX + RIGHT_EYEBROW_IDX
NOSE_IDX = _unique_indices(FaceLandmarksConnections.FACE_LANDMARKS_NOSE)
LIPS_IDX = _unique_indices(FaceLandmarksConnections.FACE_LANDMARKS_LIPS)
EYES_IDX = LEFT_EYE_IDX + RIGHT_EYE_IDX


def _detect_once(image_rgb: np.ndarray) -> np.ndarray | None:
    height, width = image_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
    with _landmarker_lock:
        result = _landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    points = result.face_landmarks[0]
    return np.array([[p.x * width, p.y * height] for p in points], dtype=np.float32)


def detect_landmarks(image_rgb: np.ndarray) -> np.ndarray | None:
    """Detect the first face's landmarks. Returns pixel-space (N, 2) points, or
    None when no face is found.

    턱·입이 프레임 밖으로 잘린 클로즈업 사진은 MediaPipe가 얼굴을 못 찾는 경우가 많다.
    그럴 때는 가장자리를 복제한 여백을 덧대서 다시 시도한다 — 실제 실패 사진으로
    확인해보니 흰색 여백은 인위적인 경계가 생겨 여전히 실패하고, 여백이 너무 크면
    얼굴이 상대적으로 작아져 또 실패해서, 복제 여백 25~50%가 잘 통했다."""
    height, width = image_rgb.shape[:2]

    for pad_ratio in _DETECT_PAD_RATIOS:
        if pad_ratio == 0:
            points = _detect_once(image_rgb)
        else:
            pad_y, pad_x = int(height * pad_ratio), int(width * pad_ratio)
            padded = cv2.copyMakeBorder(image_rgb, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
            points = _detect_once(padded)
            if points is not None:
                # 여백을 뺀 원본 좌표계로 되돌린다 (잘린 턱 등은 음수/범위 밖일 수 있고,
                # 기하 계산에만 쓰이므로 그대로 두는 게 맞다).
                points = points - np.array([pad_x, pad_y], dtype=np.float32)
                logger.info("face found only after %.0f%% edge padding", pad_ratio * 100)
        if points is not None:
            return points

    logger.info("no face detected in %dx%d image", width, height)
    return None
