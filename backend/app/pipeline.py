from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger("pipeline")

from .landmarks import (
    EYEBROW_IDX,
    EYES_IDX,
    FACE_OVAL_RING,
    LEFT_EYEBROW_IDX,
    NOSE_IDX,
    RIGHT_EYEBROW_IDX,
    detect_landmarks,
)

# 정렬 기준이 되는 캔버스: 두 눈썹을 항상 같은 위치·거리·수평으로 맞춰서
# 전/후 사진의 각도(회전)와 크기(스케일)를 동시에 정규화한다 (PRD 6.2, 6.3).
# 눈이 아니라 눈썹을 기준으로 삼은 이유: 모발이식 병원에서는 눈썹이 헤어라인과 함께
# 핵심적으로 노출·비교되는 부위라, 눈썹 자체가 수평이 되는 게 더 중요하다.
# 측면 사진은 두 눈썹 사이의 화면상 거리가 정면보다 훨씬 짧게 찍혀서, 같은 거리로
# 맞추려다 보면 확대(줌인) 배율이 정면보다 훨씬 커진다. 그만큼 위/옆 여백을 크게
# 잡아둬야 확대된 헤어라인·뒤통수까지도 캔버스 밖으로 밀려나지 않는다.
# (크롭 최소 배율(MIN_CROP_SCALE) 같은 걸로 타협하면 패딩이 남아 테두리가 다시
# 삐뚤어지므로, 캔버스를 넉넉히 키워서 패딩 없는 크롭만으로 해결한다.)
CANVAS_WIDTH = 1800
CANVAS_HEIGHT = 1700
TARGET_LEFT_BROW = np.array([820.0, 750.0], dtype=np.float32)
TARGET_RIGHT_BROW = np.array([980.0, 750.0], dtype=np.float32)
CROP_ANCHOR = ((TARGET_LEFT_BROW[0] + TARGET_RIGHT_BROW[0]) / 2, TARGET_LEFT_BROW[1])

# 측면 사진에서 "얼굴 앞쪽"(코·턱이 있는 방향) 여백이 "뒤쪽"(귀 방향)보다 필요 이상으로
# 넓게 나오는 경향이 있어, 그쪽만 줄이고 반대쪽(귀 뒤쪽)은 오히려 조금 더 넓힌다.
# 뒤쪽을 넓히는 건 패딩 없는 크롭 범위를 넘어설 수 있어(흰 여백이 살짝 보일 수 있음),
# 필요하다고 확인된 만큼만 적용한다. 정면 사진처럼 좌우가 비슷하면 이 보정 자체를
# 건너뛴다 (자세한 조건은 _adjust_side_margins 참고).
FRONT_MARGIN_TRIM_RATIO = 0.5
BACK_MARGIN_EXTEND_RATIO = 0.93
SIDE_ASYMMETRY_THRESHOLD = 0.2

# 눈썹 아래 경계선. 양수면 눈썹 아래로 여유를 두고(그만큼 눈이 노출될 수 있음),
# 음수면 눈썹 하단을 살짝 파고든다(그만큼 눈썹 일부가 가려질 수 있음). 눈은 확실히
# 가리되 눈썹은 온전히 남기는 게 목표라, 0에 가깝게 살짝만 파고드는 값으로 잡는다.
BROW_EXPOSURE_PADDING_PX = -4
# 블러 강도 — 값이 클수록 더 강하게 흐려진다. 최대한 강하게 요청받아 크게 잡았다.
MOSAIC_BLUR_KERNEL_RATIO = 0.35
# 블러 영역 경계를 얼마나 부드럽게 퍼뜨릴지 — 값이 클수록 그라데이션 폭이 넓어진다.
MOSAIC_FEATHER_RATIO = 0.04
# 얼굴 윤곽(FACE_OVAL) 폴리곤은 팽창이 아니라 오히려 살짝 안쪽으로 줄인다(침식).
# 모발이식 환자는 헤어라인이 후퇴해 있는 경우가 많아, FACE_OVAL이 실제보다 관자놀이
# 쪽으로 넓게 잡히기 쉽다 — 그대로 쓰면 구레나룻·헤어라인이 가려진다.
MOSAIC_OVAL_ERODE_RATIO = 0.02
# 눈·코는 얼굴 윤곽에 기대지 않고 각자의 랜드마크로 직접 가린다 — 윤곽을 안쪽으로
# 줄였으니 이 부위는 이걸로 확실히 보장한다. 여유는 작게(구레나룻 침범 방지).
MOSAIC_FEATURE_DILATE_RATIO = 0.045

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


def _align_to_canonical(
    image_bgr: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    screen_left_brow, screen_right_brow = _screen_ordered_brows(points)
    src = np.array([screen_left_brow, screen_right_brow], dtype=np.float32)
    dst = np.array([TARGET_LEFT_BROW, TARGET_RIGHT_BROW], dtype=np.float32)

    transform, _ = cv2.estimateAffinePartial2D(src, dst)

    warning = None
    rotation_deg = float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
    zoom_scale = float(np.hypot(transform[0, 0], transform[1, 0]))
    source_interbrow_px = float(np.linalg.norm(screen_right_brow - screen_left_brow))
    logger.info(
        "align: interbrow=%.1fpx rotation=%.1fdeg zoom=%.2fx", source_interbrow_px, rotation_deg, zoom_scale
    )
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

    # 회전 때문에 원본 사진이 캔버스 네 귀퉁이를 다 못 덮는 부분(흰 패딩 쐐기)을
    # 나중에 크롭으로 잘라내기 위해, 원본이 실제로 덮는 영역을 마스크로 같이 만든다.
    source_coverage = np.full(image_bgr.shape[:2], 255, dtype=np.uint8)
    valid_mask = cv2.warpAffine(
        source_coverage,
        transform,
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    aligned_points = np.hstack([points, ones]) @ transform.T
    return aligned, aligned_points, valid_mask, warning


def _max_valid_margins(valid_mask: np.ndarray, anchor: tuple[float, float]) -> tuple[int, int, int, int]:
    """anchor(눈썹 중앙)를 포함하면서 흰 패딩을 전혀 포함하지 않는 사각형
    (left, top, right, bottom)을 구한다.

    독립적으로 4변을 순서대로 넓혀나가는 좌표하강 방식은 진동하면서 얇은 조각으로
    수렴해버리는 문제가 있었다 (가로를 넓히면 세로가 좁아지고, 그 좁아진 세로 때문에
    다시 가로가 과하게 넓어지는 식). 대신 두 단계로 나눈다:
    1) anchor를 지나는 얇은 십자선으로 네 방향의 실제 여유 "비율(형태)"을 먼저 잰다.
    2) 그 형태(가로세로 비율)는 유지한 채, 사각형 네 모서리가 전부 유효해질 때까지
       하나의 배율로만 균일하게 줄인다. 방향별 형태가 이미 실제 여유를 반영하고
       있어서, 예전(캔버스 중심 대칭 가정) 방식과 달리 세로가 좁다고 가로까지
       똑같이 좁아지지 않는다."""
    anchor_x, anchor_y = anchor
    height, width = valid_mask.shape
    ax, ay = int(round(anchor_x)), int(round(anchor_y))

    def region_valid(l: int, t: int, r: int, b: int) -> bool:
        if r <= l or b <= t:
            return False
        return bool(valid_mask[t:b, l:r].min() > 0)

    def search(max_dist: float, test) -> float:
        lo, hi = 0.0, max_dist
        for _ in range(15):
            mid = (lo + hi) / 2
            if test(mid):
                lo = mid
            else:
                hi = mid
        return lo

    raw_left = search(anchor_x, lambda d: region_valid(int(ax - d), ay, ax + 1, ay + 1))
    raw_right = search(width - anchor_x, lambda d: region_valid(ax, ay, int(ax + d), ay + 1))
    raw_top = search(anchor_y, lambda d: region_valid(ax, int(ay - d), ax + 1, ay))
    raw_bottom = search(height - anchor_y, lambda d: region_valid(ax, ay, ax + 1, int(ay + d)))

    def box_at(t: float) -> tuple[int, int, int, int]:
        return (
            int(ax - raw_left * t),
            int(ay - raw_top * t),
            int(ax + raw_right * t),
            int(ay + raw_bottom * t),
        )

    scale = 1.0 if region_valid(*box_at(1.0)) else search(1.0, lambda t: region_valid(*box_at(t)))
    return box_at(scale)


def _adjust_side_margins(
    box: tuple[int, int, int, int],
    points_list: list[np.ndarray],
    anchor_x: float,
    front_trim_ratio: float,
    back_extend_ratio: float,
    canvas_width: int,
) -> tuple[int, int, int, int]:
    """측면 사진처럼 얼굴 윤곽(FACE_OVAL)이 좌우로 뚜렷하게 비대칭일 때만 적용한다.
    앵커에서 덜 뻗어나간 쪽을 "얼굴 앞쪽"으로 보고, 그쪽 여백은 front_trim_ratio만큼
    줄이고 반대쪽(귀 뒤쪽)은 back_extend_ratio만큼 넓힌다. 뒤쪽을 넓히는 건 캔버스의
    흰 여백 영역까지 쓸 수 있어 안전하게 슬라이스된다.
    (처음엔 "더 멀리 뻗은 쪽 = 앞쪽"으로 가정했는데 실제로는 반대였다 — 옆모습에서는
    볼·턱 실루엣이 넓게 퍼지는 귀 쪽 윤곽이 코 쪽보다 오히려 더 멀리 뻗는다.)

    정면 사진은 좌우 윤곽이 거의 대칭이라, 아주 작은 차이만으로 한쪽을 "앞쪽"으로
    판정해버리면 오히려 없던 비대칭을 만들어낸다. ASYMMETRY_THRESHOLD 이상 차이날
    때만 보정하고, 정면처럼 비슷하면 좌우를 그대로 둔다."""
    ax = anchor_x
    left_reach = max(ax - points[FACE_OVAL_RING][:, 0].min() for points in points_list)
    right_reach = max(points[FACE_OVAL_RING][:, 0].max() - ax for points in points_list)

    left, top, right, bottom = box
    longer, shorter = max(left_reach, right_reach), min(left_reach, right_reach)
    if longer == 0 or (longer - shorter) / longer < SIDE_ASYMMETRY_THRESHOLD:
        return box

    if left_reach <= right_reach:
        left = int(ax - (ax - left) * front_trim_ratio)
        right = min(canvas_width, int(ax + (right - ax) * back_extend_ratio))
    else:
        right = int(ax + (right - ax) * front_trim_ratio)
        left = max(0, int(ax - (ax - left) * back_extend_ratio))
    return left, top, right, bottom


def _crop_to_box(image: np.ndarray, points: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    left, top, right, bottom = box
    cropped = image[top:bottom, left:right]
    adjusted_points = points.copy()
    adjusted_points[:, 0] = points[:, 0] - left
    adjusted_points[:, 1] = points[:, 1] - top
    return cropped, adjusted_points


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


def _odd_kernel_size(shape: tuple[int, int], ratio: float) -> int:
    height, width = shape
    kernel = int(min(width, height) * ratio)
    kernel = kernel + 1 if kernel % 2 == 0 else kernel  # GaussianBlur는 홀수 커널만 허용
    return max(3, kernel)


def _blur(image: np.ndarray, kernel_ratio: float) -> np.ndarray:
    kernel = _odd_kernel_size(image.shape[:2], kernel_ratio)
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def _feature_hull_mask(shape: tuple[int, int], points: np.ndarray, indices: list[int], dilate_ratio: float) -> np.ndarray:
    """주어진 랜드마크 인덱스들의 볼록 껍질(convex hull)을 채우고 살짝 팽창시킨 마스크.
    눈·코처럼 특정 부위를 얼굴 윤곽 정확도와 무관하게 직접, 확실하게 가릴 때 쓴다."""
    mask = np.zeros(shape, dtype=np.uint8)
    hull = cv2.convexHull(points[indices].astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    kernel = _odd_kernel_size(shape, dilate_ratio)
    return cv2.dilate(mask, np.ones((kernel, kernel), np.uint8))


def _build_mosaic_mask(image_shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    """눈썹 아래로 얼굴 윤곽(FACE_OVAL, = 헤어라인 안쪽)을 기본 모자이크 영역으로 삼되,
    윤곽을 살짝 안쪽으로 줄여서 구레나룻·후퇴한 헤어라인을 침범하지 않게 한다. 그 대신
    눈·코는 얼굴 윤곽 정확도와 무관하게 각자의 랜드마크로 직접 확실히 가린다."""
    face_mask = _face_oval_mask(image_shape, points)

    # 모발이식 환자는 헤어라인이 후퇴해 있어 FACE_OVAL이 실제보다 관자놀이 쪽으로
    # 넓게 잡히기 쉽다. 팽창 대신 침식시켜 구레나룻·헤어라인을 침범하지 않게 한다.
    erode_kernel = _odd_kernel_size(face_mask.shape, MOSAIC_OVAL_ERODE_RATIO)
    face_mask = cv2.erode(face_mask, np.ones((erode_kernel, erode_kernel), np.uint8))

    brow_bottom_y = int(points[EYEBROW_IDX][:, 1].max() + BROW_EXPOSURE_PADDING_PX)
    brow_bottom_y = max(0, min(brow_bottom_y, image_shape[0]))

    mosaic_mask = face_mask.copy()
    mosaic_mask[:brow_bottom_y, :] = 0  # 눈썹 위(이마·헤어라인)는 모자이크 대상에서 제외

    # 윤곽을 안쪽으로 줄인 만큼 눈·코가 덜 덮일 수 있으니, 각자의 랜드마크로 직접 보강한다.
    for indices in (EYES_IDX, NOSE_IDX):
        feature_mask = _feature_hull_mask(image_shape, points, indices, MOSAIC_FEATURE_DILATE_RATIO)
        mosaic_mask = np.maximum(mosaic_mask, feature_mask)

    # 보강한 눈 마스크가 눈썹 경계선 위로 번지지 않도록 마지막에 한 번 더 제외시킨다.
    mosaic_mask[:brow_bottom_y, :] = 0
    return mosaic_mask


def _apply_face_mosaic(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    mosaic_mask = _build_mosaic_mask(image.shape[:2], points)

    # 마스크 자체를 블러 처리해서 0~255 사이의 부드러운 경계(그라데이션)로 만든 뒤,
    # 그 값을 알파값 삼아 원본과 블러 이미지를 섞는다 — 마스크를 그대로 써서
    # 딱 잘라 바꾸면 경계가 뚜렷하게 티가 난다.
    feather_kernel = _odd_kernel_size(mosaic_mask.shape, MOSAIC_FEATHER_RATIO)
    alpha = cv2.GaussianBlur(mosaic_mask, (feather_kernel, feather_kernel), 0).astype(np.float32) / 255.0
    alpha = alpha[:, :, None]

    blurred = _blur(image, MOSAIC_BLUR_KERNEL_RATIO)
    blended = image.astype(np.float32) * (1 - alpha) + blurred.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def process_pair(
    before_bgr: np.ndarray, after_bgr: np.ndarray, apply_mosaic: bool = False
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    warnings: list[str] = []

    before_points = detect_landmarks(cv2.cvtColor(before_bgr, cv2.COLOR_BGR2RGB))
    after_points = detect_landmarks(cv2.cvtColor(after_bgr, cv2.COLOR_BGR2RGB))

    if before_points is None:
        warnings.append("전 사진에서 얼굴을 찾지 못해 원본을 그대로 반환합니다.")
    if after_points is None:
        warnings.append("후 사진에서 얼굴을 찾지 못해 원본을 그대로 반환합니다.")
    if before_points is None or after_points is None:
        return before_bgr, after_bgr, warnings

    aligned_before, before_canonical_points, before_valid_mask, before_warning = _align_to_canonical(
        before_bgr, before_points
    )
    aligned_after, after_canonical_points, after_valid_mask, after_warning = _align_to_canonical(
        after_bgr, after_points
    )
    if before_warning:
        warnings.append(f"전 사진: {before_warning}")
    if after_warning:
        warnings.append(f"후 사진: {after_warning}")

    # 회전으로 생긴 흰 패딩 쐐기가 안 보이도록, 두 사진 다 패딩이 전혀 없는 영역까지만
    # 남기고 잘라낸다. 가로/세로 각 방향을 독립적으로 넓힌 뒤, 두 사진 모두에게
    # 안전한(패딩이 안 걸리는) 교집합 영역으로 맞춘다 — 그래야 두 결과의 크기가 같아진다.
    before_box = _max_valid_margins(before_valid_mask, CROP_ANCHOR)
    after_box = _max_valid_margins(after_valid_mask, CROP_ANCHOR)
    final_box = (
        max(before_box[0], after_box[0]),
        max(before_box[1], after_box[1]),
        min(before_box[2], after_box[2]),
        min(before_box[3], after_box[3]),
    )
    final_box = _adjust_side_margins(
        final_box,
        [before_canonical_points, after_canonical_points],
        CROP_ANCHOR[0],
        FRONT_MARGIN_TRIM_RATIO,
        BACK_MARGIN_EXTEND_RATIO,
        CANVAS_WIDTH,
    )
    logger.info("crop: before=%s after=%s -> using=%s (canvas=%dx%d)", before_box, after_box, final_box, CANVAS_WIDTH, CANVAS_HEIGHT)
    aligned_before, before_canonical_points = _crop_to_box(aligned_before, before_canonical_points, final_box)
    aligned_after, after_canonical_points = _crop_to_box(aligned_after, after_canonical_points, final_box)

    matched_after = _match_color(aligned_before, before_canonical_points, aligned_after, after_canonical_points)

    if apply_mosaic:
        aligned_before = _apply_face_mosaic(aligned_before, before_canonical_points)
        matched_after = _apply_face_mosaic(matched_after, after_canonical_points)

    return aligned_before, matched_after, warnings
