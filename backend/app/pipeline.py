from __future__ import annotations

import cv2
import numpy as np

from .landmarks import (
    EYEBROW_IDX,
    FACE_OVAL_RING,
    LEFT_EYEBROW_IDX,
    RIGHT_EYEBROW_IDX,
    detect_landmarks,
)

# 정렬 기준이 되는 캔버스: 두 눈썹을 항상 같은 위치·거리·수평으로 맞춰서
# 전/후 사진의 각도(회전)와 크기(스케일)를 동시에 정규화한다 (PRD 6.2, 6.3).
# 눈이 아니라 눈썹을 기준으로 삼은 이유: 모발이식 병원에서는 눈썹이 헤어라인과 함께
# 핵심적으로 노출·비교되는 부위라, 눈썹 자체가 수평이 되는 게 더 중요하다.
# 위쪽 여백을 넉넉히 둬서(눈썹 기준 위로 CANVAS_HEIGHT의 절반 이상) 헤어라인이
# 캔버스 밖으로 잘려나가지 않게 한다.
CANVAS_WIDTH = 900
CANVAS_HEIGHT = 1300
TARGET_LEFT_BROW = np.array([370.0, 550.0], dtype=np.float32)
TARGET_RIGHT_BROW = np.array([530.0, 550.0], dtype=np.float32)

# 눈썹 아래로 이 픽셀만큼 여유를 두고 그 아래(눈·코·입·볼·턱)만 모자이크한다.
# 눈썹 위쪽(이마~헤어라인)은 그대로 노출된다 (PRD 6.5).
BROW_EXPOSURE_PADDING_PX = 18
MOSAIC_DOWNSCALE_FACTOR = 0.06

# 회전 보정이 이 각도를 넘으면 정렬 신뢰도가 낮다고 보고 경고만 남긴다 (PRD 6.2).
MAX_TRUSTED_ROTATION_DEG = 45.0


def _landmark_center(points: np.ndarray, indices: list[int]) -> np.ndarray:
    return points[indices].mean(axis=0)


def _screen_ordered_brows(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MediaPipe의 LEFT/RIGHT 눈썹 라벨은 화면 기준이 아니라 피사체 본인의 해부학적
    좌/우를 가리킨다 (셀카가 아닌 일반 정면 사진에서는 반대로 나타남). 라벨을 믿는 대신
    사진에 실제로 찍힌 x좌표로 화면상 왼쪽/오른쪽을 정해야, 엉뚱하게 180도 가까이
    돌아가는 정렬(상하 반전)을 막을 수 있다."""
    brow_a = _landmark_center(points, LEFT_EYEBROW_IDX)
    brow_b = _landmark_center(points, RIGHT_EYEBROW_IDX)
    return (brow_a, brow_b) if brow_a[0] <= brow_b[0] else (brow_b, brow_a)


def _align_to_canonical(image_bgr: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, str | None]:
    screen_left_brow, screen_right_brow = _screen_ordered_brows(points)
    src = np.array([screen_left_brow, screen_right_brow], dtype=np.float32)
    dst = np.array([TARGET_LEFT_BROW, TARGET_RIGHT_BROW], dtype=np.float32)

    transform, _ = cv2.estimateAffinePartial2D(src, dst)

    warning = None
    rotation_deg = float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
    if abs(rotation_deg) > MAX_TRUSTED_ROTATION_DEG:
        warning = "얼굴 각도가 너무 기울어져 있어 정렬 결과의 신뢰도가 낮을 수 있습니다."

    aligned = cv2.warpAffine(
        image_bgr,
        transform,
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    aligned_points = np.hstack([points, ones]) @ transform.T
    return aligned, aligned_points, warning


def _face_oval_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    ring = points[FACE_OVAL_RING].astype(np.int32)
    cv2.fillPoly(mask, [ring], 255)
    return mask


def _match_color(
    reference: np.ndarray,
    reference_points: np.ndarray,
    target: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    """target의 밝기/색감을 reference(기본값: 전 사진)에 맞춘다 (PRD 6.4).
    얼굴 영역만으로 통계를 내어, 캔버스 여백(흰색 패딩)이 통계를 왜곡하지 않게 한다."""
    reference_mask = _face_oval_mask(reference.shape[:2], reference_points)
    target_mask = _face_oval_mask(target.shape[:2], target_points)

    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    result_lab = target_lab.copy()

    for channel in range(3):
        reference_values = reference_lab[:, :, channel][reference_mask > 0]
        target_values = target_lab[:, :, channel][target_mask > 0]
        if reference_values.size == 0 or target_values.size == 0:
            continue

        reference_mean, reference_std = reference_values.mean(), reference_values.std() + 1e-6
        target_mean, target_std = target_values.mean(), target_values.std() + 1e-6

        channel_data = target_lab[:, :, channel]
        result_lab[:, :, channel] = (channel_data - target_mean) / target_std * reference_std + reference_mean

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


def _pixelate(image: np.ndarray, factor: float) -> np.ndarray:
    height, width = image.shape[:2]
    small_size = (max(1, int(width * factor)), max(1, int(height * factor)))
    small = cv2.resize(image, small_size, interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def _apply_face_mosaic(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    face_mask = _face_oval_mask(image.shape[:2], points)

    brow_bottom_y = int(points[EYEBROW_IDX][:, 1].max() + BROW_EXPOSURE_PADDING_PX)
    brow_bottom_y = max(0, min(brow_bottom_y, image.shape[0]))

    mosaic_mask = face_mask.copy()
    mosaic_mask[:brow_bottom_y, :] = 0  # 눈썹 위(이마·헤어라인)는 모자이크 대상에서 제외

    pixelated = _pixelate(image, MOSAIC_DOWNSCALE_FACTOR)
    result = image.copy()
    hide = mosaic_mask > 0
    result[hide] = pixelated[hide]
    return result


def process_pair(before_bgr: np.ndarray, after_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    warnings: list[str] = []

    before_points = detect_landmarks(cv2.cvtColor(before_bgr, cv2.COLOR_BGR2RGB))
    after_points = detect_landmarks(cv2.cvtColor(after_bgr, cv2.COLOR_BGR2RGB))

    if before_points is None:
        warnings.append("전 사진에서 얼굴을 찾지 못해 원본을 그대로 반환합니다.")
    if after_points is None:
        warnings.append("후 사진에서 얼굴을 찾지 못해 원본을 그대로 반환합니다.")
    if before_points is None or after_points is None:
        return before_bgr, after_bgr, warnings

    aligned_before, before_canonical_points, before_warning = _align_to_canonical(before_bgr, before_points)
    aligned_after, after_canonical_points, after_warning = _align_to_canonical(after_bgr, after_points)
    if before_warning:
        warnings.append(f"전 사진: {before_warning}")
    if after_warning:
        warnings.append(f"후 사진: {after_warning}")

    matched_after = _match_color(aligned_before, before_canonical_points, aligned_after, after_canonical_points)

    # TODO: 모자이크(PRD 6.5)는 정렬 결과를 먼저 확인하기 위해 잠시 비활성화했다.
    # _apply_face_mosaic(image, points)를 다시 연결하면 된다.
    return aligned_before, matched_after, warnings
