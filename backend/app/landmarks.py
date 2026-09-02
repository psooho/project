from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarksConnections

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
)
_landmarker = vision.FaceLandmarker.create_from_options(_options)


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


def detect_landmarks(image_rgb: np.ndarray) -> np.ndarray | None:
    """Detect the first face's landmarks. Returns pixel-space (N, 2) points, or
    None when no face is found."""
    height, width = image_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
    result = _landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    points = result.face_landmarks[0]
    return np.array([[p.x * width, p.y * height] for p in points], dtype=np.float32)
