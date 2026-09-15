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
    LIPS_IDX,
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
# 얼굴 세로 높이를 700px에 맞춰 얹으므로, 고정 틀(측면은 폭이 얼굴 높이의 1.7배)이
# 잘리지 않을 만큼 넉넉해야 한다. 좁으면 틀 좌표가 캔버스 밖으로 나가 크롭이 깨진다.
CANVAS_WIDTH = 2400
CANVAS_HEIGHT = 2200
TARGET_LEFT_BROW = np.array([1160.0, 900.0], dtype=np.float32)
TARGET_RIGHT_BROW = np.array([1240.0, 900.0], dtype=np.float32)
CROP_ANCHOR = ((TARGET_LEFT_BROW[0] + TARGET_RIGHT_BROW[0]) / 2, TARGET_LEFT_BROW[1])

# 측면 사진에서 "얼굴 앞쪽"(코·턱이 있는 방향) 여백이 "뒤쪽"(귀 방향)보다 필요 이상으로
# 넓게 나오는 경향이 있어, 그쪽만 줄이고 반대쪽(귀 뒤쪽)은 오히려 조금 더 넓힌다.
# 뒤쪽을 넓히는 건 패딩 없는 크롭 범위를 넘어설 수 있어(흰 여백이 살짝 보일 수 있음),
# 필요하다고 확인된 만큼만 적용한다. 정면 사진처럼 좌우가 비슷하면 이 보정 자체를
# 건너뛴다 (자세한 조건은 _adjust_side_margins 참고).
FRONT_MARGIN_TRIM_RATIO = 0.6
BACK_MARGIN_EXTEND_RATIO = 0.93
SIDE_ASYMMETRY_THRESHOLD = 0.2

# 정면 사진은 위 앞/뒤 보정을 건너뛰는데, 그러면 원본이 살짝 치우쳐 찍혔을 때 화면 왼쪽
# 여백이 넓어 보인다. 정면일 때는 화면 왼쪽(보는 사람 기준) 여백만 이만큼으로 줄인다.
FRONTAL_LEFT_TRIM_RATIO = 0.90

# 눈썹 아래 경계선. 양수면 눈썹 아래로 여유를 두고(그만큼 눈이 노출될 수 있음),
# 음수면 눈썹 하단을 살짝 파고든다(그만큼 눈썹 일부가 가려질 수 있음). 눈은 확실히
# 가리되 눈썹은 온전히 남기는 게 목표라, 0에 가깝게 살짝만 파고드는 값으로 잡는다.
BROW_EXPOSURE_PADDING_PX = -4
# 블러 강도 — 값이 클수록 더 강하게 흐려진다. 최대한 강하게 요청받아 크게 잡았다.
MOSAIC_BLUR_KERNEL_RATIO = 0.35
# 블러 영역 경계를 얼마나 부드럽게 퍼뜨릴지 — 값이 클수록 그라데이션 폭이 넓어진다.
MOSAIC_FEATHER_RATIO = 0.04
# 얼굴 윤곽(FACE_OVAL)을 "좌우로만" 안쪽으로 당기는 폭(반지름 비율). 모발이식 환자는
# 헤어라인이 후퇴해 있는 경우가 많아 FACE_OVAL이 실제 헤어라인보다 관자놀이 쪽으로
# 넓게 잡히고, 그대로 쓰면 구레나룻이 가려진다. 사방으로 균일하게 깎으면(예전 방식)
# 구레나룻을 피하려 세게 깎을수록 턱·볼 커버리지까지 같이 사라지므로, 세로는 건드리지
# 않고 가로로만 당긴다.
MOSAIC_SIDE_TRIM_RATIO = 0.12
# 눈·코는 얼굴 윤곽에 기대지 않고 각자의 랜드마크로 직접 가린다 — 윤곽을 옆으로
# 당겼으니 이 부위는 이걸로 확실히 보장한다.
MOSAIC_FEATURE_DILATE_RATIO = 0.06
# 윤곽을 당기면서 눈·코 마스크와 얼굴 마스크 사이에 생기는 틈을 메우는 정도(닫기 연산).
MOSAIC_CLOSE_RATIO = 0.05

# 측면 사진에서 고개가 앞으로 숙여져 보이는 문제를 지각적으로 보정하는 각도.
# 눈썹 중심선은 변환상 이미 정확히 수평이라, 숙여져 보이는 건 2D 회전으로는 없앨 수 없는
# 고개 각도(pitch) 탓이다. 그래서 앞쪽(코 방향)을 이만큼 더 들어올려 상쇄한다.
# 정면 사진은 이미 자연스러우므로 좌우가 뚜렷하게 비대칭일 때(측면)만 적용한다.
FRONT_LIFT_DEG = 3.2

# 회전 보정이 이 각도를 넘으면 정렬 신뢰도가 낮다고 보고 경고만 남긴다 (PRD 6.2).
MAX_TRUSTED_ROTATION_DEG = 45.0

# --- 고정 틀 ------------------------------------------------------------------
# 결과물의 여백을 사진마다 똑같이 맞추기 위한 틀. 눈썹 간격(캔버스에서 160px)의 배수로
# 적는다 — 눈썹은 이미 캔버스 고정 위치에 놓이므로, 이 배수가 곧 "얼굴 대비 여백"이 된다.
#
# 예전에는 "흰 여백이 전혀 없는 최대 사각형"으로 잘랐다. 그러면 원본에 남아 있는 영역에
# 따라 결과가 달라져서, 실측해보니 같은 정면 사진끼리도 오른쪽 여백이 1.0~1.7배, 아래가
# 0.35~1.1배로 흔들렸다. 직원들이 "사진마다 여백이 제각각"이라고 지적한 게 이것이다.
#
# 값은 실제 사진 20장(정면 11, 측면 9)의 분포를 보고, 대부분의 사진이 원본만으로 틀을
# 채울 수 있는 선에서 잡았다. 모자라는 사진만 가장자리를 늘여 메운다 — 넉넉하게 잡을수록
# 여백은 시원하지만 늘인 자국이 보이므로, 실측 최솟값 바로 위에서 멈춘다.
FRONTAL_FRAME = (0.81, 0.81, 1.11, 0.33)   # (앞/왼쪽, 뒤/오른쪽, 위, 아래)
SIDE_FRAME = (0.42, 1.30, 0.81, 0.49)      # 측면은 뒤통수 쪽을 훨씬 넓게 잡는다

# 크기를 맞추는 기준은 "얼굴 세로 높이"다. 눈썹 간격을 쓰면 측면에서 어긋난다 — 고개를
# 돌린 각도만큼 눈썹 사이가 원근으로 짧아지기 때문이다(실측: 얼굴 높이 대비 눈썹 간격이
# 정면은 1.80배인데 측면은 2.95배). 전/후의 고개 각도가 조금만 달라도 배율이 틀어져서,
# 한쪽만 귀가 잘리거나 여백이 달라졌다. 세로 거리는 좌우 회전에 영향받지 않는다.
# 실측으로 전/후 어긋남이 6.5% -> 2.6%로 줄었다.
#
# 목표 높이를 크게 잡으면 결과물 해상도가 그만큼 올라간다. 원본이 대부분 이보다 크므로
# 대개 축소만 일어난다 (예전에는 눈썹 160px에 맞추느라 0.15~0.7배로 줄여 버렸다).
TARGET_FACE_HEIGHT = 700.0
# 고정 틀을 원본이 이 비율 이상 못 덮으면 알려준다. 모자란 만큼은 가장자리를 늘여 메우는데,
# 조금이면 티가 안 나지만 넓어지면 번진 자국이 보인다 — 그때는 더 넓게 찍는 게 답이다.
FRAME_FILL_WARN_RATIO = 0.03

# --- 수동 2점 모드 -----------------------------------------------------------
# 이마·헤어라인만 크게 찍은 클로즈업은 눈·코·입이 프레임 밖이라 MediaPipe가 얼굴을
# 아예 못 찾는다(실측: 이런 사진 16장 중 15장 실패, 검출 임계값을 0.01까지 낮추고
# 여백을 덧대 재시도해도 마찬가지). 그럴 때 사용자가 두 사진에서 같은 지점 두 곳을
# 직접 찍어주면, 정렬에 필요한 건 원래 점 두 개뿐이므로 나머지 처리는 그대로 돌아간다.

# 회전 때문에 사진이 캔버스 밖으로 밀려나지 않도록 두는 여유.
MANUAL_FIT_SAFETY = 0.95
# 점을 3개 이상 찍으면 전/후의 점들이 서로 맞는지 검사할 수 있다 (2개는 어떻게 찍든 항상
# 정확히 맞아떨어져서 검사가 불가능하다). 남는 오차가 점들이 퍼진 범위의 이 비율을 넘으면
# 서로 다른 자리를 찍은 것으로 보고 경고한다.
MANUAL_RESIDUAL_WARN_RATIO = 0.06
# 기준점(눈썹) 아래로 반드시 남겨둘 여백 — 기준점 사이 간격 대비 비율. 눈썹은 찍은 두 점
# 아래로도 두께만큼 더 내려오는데, 사진을 키우면 그 두께도 같이 커진다. 이마 클로즈업은
# 눈썹이 사진 맨 아래에 걸려 있어서 이 여백을 확보하지 않으면 배율을 조금만 바꿔도 잘린다.
ANCHOR_BOTTOM_MARGIN_RATIO = 0.10
# 사진 밖까지 넓힐 수 있는 한계 (크롭 높이 대비). 없는 부분은 가장자리를 늘여 채우는데,
# 넓게 늘이면 세로 줄무늬가 눈에 띈다. 눈썹을 살리는 데 필요한 만큼만 얇게 쓴다.
ANCHOR_BOTTOM_MAX_STRETCH = 0.04
# 색 보정 통계를 낼 피부 픽셀의 YCrCb 범위. 자동 모드는 얼굴 윤곽 안쪽으로 범위를
# 잡지만 수동 모드에는 윤곽이 없다. 프레임 전체를 쓰면 이식으로 늘어난 머리숱만큼
# 후 사진이 어두워져, 그걸 상쇄하려고 엉뚱하게 밝아진다. 그래서 피부만 골라 쓴다.
SKIN_YCRCB_LOW = (0, 133, 77)
SKIN_YCRCB_HIGH = (255, 173, 127)
# 피부로 잡힌 픽셀이 이 비율보다 적으면 신뢰할 수 없다고 보고 프레임 전체로 되돌린다.
SKIN_MIN_COVERAGE = 0.05


def _landmark_center(points: np.ndarray, indices: list[int]) -> np.ndarray:
    return points[indices].mean(axis=0)


def _screen_ordered_brows(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MediaPipe의 LEFT/RIGHT 눈썹 라벨은 화면 기준이 아니라 피사체 본인의 해부학적
    좌/우를 가리킨다 (셀카가 아닌 일반 정면 사진에서는 반대로 나타남). 라벨을 믿는 대신
    사진에 실제로 찍힌 x좌표로 화면상 왼쪽/오른쪽을 정해야, 엉뚱하게 180도 가까이
    돌아가는 정렬(상하 반전)을 막을 수 있다."""
    brow_a = _landmark_center(points, LEFT_EYEBROW_IDX)
    brow_b = _landmark_center(points, RIGHT_EYEBROW_IDX)
    return _screen_ordered_pair(brow_a, brow_b)


def _screen_ordered_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """두 점을 화면상 왼쪽·오른쪽 순으로 놓는다. 수동 2점 모드에서도 같은 규칙을 써야
    전/후 사진이 서로 뒤집혀 정렬되는 일이 없다."""
    return (a, b) if a[0] <= b[0] else (b, a)


def _front_is_right(points: np.ndarray) -> bool | None:
    """코가 향하는 쪽("앞쪽")이 화면 오른쪽인지 판정한다. 좌우가 비슷하면(정면) None.

    옆모습에서는 볼·턱 실루엣이 넓게 퍼지는 귀 쪽 윤곽이 코 쪽보다 더 멀리 뻗으므로,
    윤곽이 "덜" 뻗은 쪽이 앞(코) 방향이다 (_adjust_side_margins와 같은 기준)."""
    brow_mid_x = float(points[EYEBROW_IDX][:, 0].mean())
    oval_x = points[FACE_OVAL_RING][:, 0]
    left_reach = brow_mid_x - float(oval_x.min())
    right_reach = float(oval_x.max()) - brow_mid_x

    longer, shorter = max(left_reach, right_reach), min(left_reach, right_reach)
    if longer <= 0 or (longer - shorter) / longer < SIDE_ASYMMETRY_THRESHOLD:
        return None
    return right_reach < left_reach


def _lift_front(transform: np.ndarray, points: np.ndarray | None) -> np.ndarray:
    """앞쪽(코 방향)을 FRONT_LIFT_DEG만큼 들어올리도록 회전을 덧붙인다.
    getRotationMatrix2D의 양수 각도는 화면상 반시계 방향이라 오른쪽이 올라간다."""
    if points is None:
        # 수동 2점 모드: 얼굴 윤곽이 없어 앞/뒤를 판정할 수 없으므로 보정하지 않는다.
        return transform
    front_is_right = _front_is_right(points)
    if front_is_right is None or FRONT_LIFT_DEG == 0:
        return transform

    center = (TARGET_LEFT_BROW + TARGET_RIGHT_BROW) / 2
    angle = FRONT_LIFT_DEG if front_is_right else -FRONT_LIFT_DEG
    rotation = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)

    stacked = np.vstack([rotation, [0, 0, 1]]) @ np.vstack([transform, [0, 0, 1]])
    return stacked[:2].astype(np.float32)


def _face_height(points: np.ndarray) -> float:
    """얼굴 윤곽의 세로 길이. 좌우로 고개를 돌려도 거의 변하지 않아 크기 기준으로 쓴다."""
    oval_y = points[FACE_OVAL_RING][:, 1]
    return max(1.0, float(oval_y.max() - oval_y.min()))


def _brow_transform(points: np.ndarray) -> np.ndarray:
    """자동 모드의 정렬 변환.

    각도는 두 눈썹을 잇는 선에서, 크기는 얼굴 세로 높이에서 따로 가져온다. 예전에는 두
    눈썹을 캔버스의 고정 두 점에 그대로 얹어서 각도와 크기를 한꺼번에 정했는데, 그러면
    측면에서 고개 각도에 따라 눈썹 간격이 달라지는 게 그대로 배율 오차가 됐다."""
    screen_left_brow, screen_right_brow = _screen_ordered_brows(points)
    direction = np.asarray(screen_right_brow, dtype=np.float64) - np.asarray(
        screen_left_brow, dtype=np.float64
    )
    angle = np.arctan2(direction[1], direction[0])
    scale = TARGET_FACE_HEIGHT / _face_height(points)

    cos, sin = np.cos(-angle) * scale, np.sin(-angle) * scale
    linear = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)
    brow_mid = (np.asarray(screen_left_brow, dtype=np.float64)
                + np.asarray(screen_right_brow, dtype=np.float64)) / 2
    offset = np.asarray(CROP_ANCHOR, dtype=np.float64) - linear @ brow_mid

    transform = np.hstack([linear, offset.reshape(2, 1)]).astype(np.float32)
    return _lift_front(transform, points)


def _align_to_canonical(
    image_bgr: np.ndarray,
    transform: np.ndarray,
    points: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, str | None]:
    """이미 계산된 변환으로 사진을 캔버스에 얹는다. points는 있으면 같이 변환해서
    돌려주고, 없으면(수동 기준점 모드) None."""
    warning = None
    rotation_deg = float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
    zoom_scale = float(np.hypot(transform[0, 0], transform[1, 0]))
    logger.info("align: rotation=%.1fdeg zoom=%.2fx", rotation_deg, zoom_scale)
    if abs(rotation_deg) > MAX_TRUSTED_ROTATION_DEG:
        warning = "얼굴 각도가 너무 기울어져 있어 정렬 결과의 신뢰도가 낮을 수 있습니다."

    aligned = cv2.warpAffine(
        image_bgr,
        transform,
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        flags=cv2.INTER_LINEAR,
        # 아래 여백을 확보하느라 크롭이 사진 밖으로 조금 넘어갈 수 있다. 흰색으로 두면
        # 흰 띠가 생기지만, 가장자리를 늘여 채우면 피부가 살짝 번진 정도로 보인다.
        borderMode=cv2.BORDER_REPLICATE,
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

    aligned_points = None if points is None else _apply_transform(points, transform)
    return aligned, aligned_points, valid_mask, warning


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    return np.hstack([points.astype(np.float32), ones]) @ transform.T


def _fit_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """대응점들로부터 회전+배율+이동(유사변환)을 최소제곱으로 구한다 (Umeyama).

    점이 2개면 정확해와 같고, 3개 이상이면 클릭 오차가 서로 상쇄돼 더 정확해진다.
    늘리거나 비트는(어파인) 변환은 얼굴을 일그러뜨리므로 쓰지 않는다."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_centered, dst_centered = src - src_mean, dst - dst_mean

    covariance = (dst_centered.T @ src_centered) / len(src)
    u, singular, vt = np.linalg.svd(covariance)
    # 반사(거울상)로 뒤집히지 않도록 행렬식 부호를 바로잡는다.
    correction = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ correction @ vt

    variance = (src_centered ** 2).sum() / len(src)
    scale = singular.sum() / variance if variance > 0 else 1.0

    linear = scale * rotation
    return np.hstack([linear, (dst_mean - linear @ src_mean).reshape(2, 1)]).astype(np.float32)


def _fit_residual(src: np.ndarray, dst: np.ndarray, transform: np.ndarray) -> float:
    """맞춘 뒤에도 남는 어긋남의 크기(픽셀). 점이 2개면 항상 0이다."""
    mapped = _apply_transform(np.asarray(src, dtype=np.float32), transform)
    return float(np.sqrt(((mapped - np.asarray(dst, dtype=np.float32)) ** 2).sum(axis=1).mean()))


def _corners(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)


def _manual_transforms(
    before_anchors: np.ndarray,
    after_anchors: np.ndarray,
    before_shape: tuple[int, int],
    after_shape: tuple[int, int],
    level: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """수동 기준점 모드에서 전/후 사진 각각의 변환을 구한다.

    두 단계로 나눈다. (1) 후 사진을 전 사진에 맞추는 변환 — 여기서만 기준점을 쓰고,
    점이 많을수록 정확해진다. (2) 화면 배치 — 두 사진이 다 들어오도록 확대해 캔버스
    가운데 놓는다.

    level=True면 기준점들을 잇는 선(가장 왼쪽 점 → 가장 오른쪽 점)이 수평이 되게 돌린다.
    눈썹 위치를 자동으로 찾은 경우처럼 "이 점들이 눈썹"이라는 걸 아는 때만 켠다. 사용자가
    직접 찍은 점은 무엇인지 알 수 없어 방향을 강제할 근거가 없다 — 예전에 무조건 수평으로
    맞췄더니, "두 점을 멀리 벌리라"는 안내를 따라 헤어라인 위와 눈썹처럼 세로로 찍은 경우
    사진이 80도 넘게 누워버렸다.

    자동 모드처럼 기준점을 캔버스의 고정 좌표에 박지 않는 이유: 사용자가 어디를 찍을지
    알 수 없어서, 눈썹 양 끝처럼 멀리 떨어진 두 점을 고정 간격으로 줄이면 사진이 심하게
    축소돼 해상도가 뭉개진다."""
    after_to_before = _fit_similarity(after_anchors, before_anchors)
    residual = _fit_residual(after_anchors, before_anchors, after_to_before)

    rotation = np.eye(2)
    if level:
        ordered = np.asarray(before_anchors, dtype=np.float64)
        ordered = ordered[np.argsort(ordered[:, 0])]
        direction = ordered[-1] - ordered[0]
        angle = np.arctan2(direction[1], direction[0])
        cos, sin = np.cos(-angle), np.sin(-angle)
        rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)
        logger.info("manual: 눈썹선 %.1f도 -> 수평", np.degrees(angle))

    # 두 사진의 네 귀퉁이를 모두 전 사진 좌표계로 모아 필요한 크기를 잰다.
    spread = np.vstack([
        _corners(before_shape),
        _apply_transform(_corners(after_shape), after_to_before),
    ]) @ rotation.T
    extent = spread.max(axis=0) - spread.min(axis=0)
    scale = MANUAL_FIT_SAFETY * min(CANVAS_WIDTH / extent[0], CANVAS_HEIGHT / extent[1])

    linear = scale * rotation
    center = (spread.max(axis=0) + spread.min(axis=0)) / 2 * scale
    offset = np.array([CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2]) - center
    base = np.hstack([linear, offset.reshape(2, 1)]).astype(np.float32)

    return base, _compose(base, after_to_before), residual


def _scale_about(pivot: np.ndarray, factor: float) -> np.ndarray:
    """pivot을 고정한 채 factor배로 키우거나 줄이는 변환."""
    linear = float(factor) * np.eye(2)
    offset = np.asarray(pivot, dtype=np.float64) * (1.0 - float(factor))
    return np.hstack([linear, offset.reshape(2, 1)]).astype(np.float32)


def _compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """두 어파인 변환을 잇는다 (inner를 적용한 뒤 outer)."""
    stacked = np.vstack([outer, [0, 0, 1]]) @ np.vstack([inner, [0, 0, 1]])
    return stacked[:2].astype(np.float32)


def _skin_mask(image_bgr: np.ndarray) -> np.ndarray:
    """수동 2점 모드에서 색 보정 통계를 낼 피부 영역.

    처음엔 기준점 주변 타원을 썼는데, 눈썹을 찍으면 타원이 사진 아래 가장자리에 걸쳐
    좁은 영역만 잡히고 그 안의 색 분포가 전/후에서 크게 달라져 보정이 1.8배까지 증폭됐다
    (결과물이 새파래짐). 피부는 전/후 모두 넓게 잡히고 이식으로 바뀌는 머리카락이
    빠지므로 통계가 훨씬 안정적이다."""
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, SKIN_YCRCB_LOW, SKIN_YCRCB_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    if (mask > 0).mean() < SKIN_MIN_COVERAGE:
        logger.info("skin mask too small (%.1f%%), using whole frame", 100 * (mask > 0).mean())
        return np.full(image_bgr.shape[:2], 255, dtype=np.uint8)
    return mask


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


def _fixed_frame_box(front_is_right: bool | None) -> tuple[int, int, int, int]:
    """여백이 항상 같도록 정해진 크기로 자를 영역을 돌려준다.

    front_is_right가 None이면 정면, 아니면 측면이다. 측면은 코가 향하는 쪽을 "앞"으로
    보고 그쪽을 좁게, 뒤통수 쪽을 넓게 잡는다."""
    front, back, top, bottom = FRONTAL_FRAME if front_is_right is None else SIDE_FRAME
    left, right = (back, front) if front_is_right else (front, back)

    center_x, brow_y = CROP_ANCHOR
    # 캔버스를 벗어나면 잘라낸다. 음수 좌표로 슬라이스하면 파이썬이 뒤에서부터 세어
    # 엉뚱한 영역을 집는다 (예전에 여백을 넓혔다가 크롭이 깨진 적이 있다).
    return (
        max(0, int(round(center_x - left * TARGET_FACE_HEIGHT))),
        max(0, int(round(brow_y - top * TARGET_FACE_HEIGHT))),
        min(CANVAS_WIDTH, int(round(center_x + right * TARGET_FACE_HEIGHT))),
        min(CANVAS_HEIGHT, int(round(brow_y + bottom * TARGET_FACE_HEIGHT))),
    )


def _adjust_side_margins(
    box: tuple[int, int, int, int],
    points_list: list[np.ndarray | None],
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
    때만 앞/뒤 보정을 하고, 정면일 때는 화면 왼쪽 여백만 조금 줄인다."""
    if any(points is None for points in points_list):
        # 수동 2점 모드: 얼굴 윤곽이 없어 앞/뒤도 정면 여부도 판정할 수 없다. 근거 없이
        # 한쪽을 깎으면 오히려 치우치므로 여백은 그대로 둔다.
        return box

    ax = anchor_x
    left_reach = max(ax - points[FACE_OVAL_RING][:, 0].min() for points in points_list)
    right_reach = max(points[FACE_OVAL_RING][:, 0].max() - ax for points in points_list)

    left, top, right, bottom = box
    longer, shorter = max(left_reach, right_reach), min(left_reach, right_reach)
    if longer == 0 or (longer - shorter) / longer < SIDE_ASYMMETRY_THRESHOLD:
        # 정면: 앞/뒤 구분이 무의미하므로 그 보정은 하지 않고, 화면 왼쪽 여백만 살짝 줄인다.
        return int(ax - (ax - left) * FRONTAL_LEFT_TRIM_RATIO), top, right, bottom

    if left_reach <= right_reach:
        left = int(ax - (ax - left) * front_trim_ratio)
        right = min(canvas_width, int(ax + (right - ax) * back_extend_ratio))
    else:
        right = int(ax + (right - ax) * front_trim_ratio)
        left = max(0, int(ax - (ax - left) * back_extend_ratio))
    return left, top, right, bottom


def _crop_to_box(image: np.ndarray, points: np.ndarray | None, box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray | None]:
    left, top, right, bottom = box
    return image[top:bottom, left:right], _shift_points(points, box)


def _shift_points(points: np.ndarray | None, box: tuple[int, int, int, int]) -> np.ndarray | None:
    if points is None:
        return None
    shifted = points.copy()
    shifted[:, 0] = points[:, 0] - box[0]
    shifted[:, 1] = points[:, 1] - box[1]
    return shifted


def _face_oval_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    ring = points[FACE_OVAL_RING].astype(np.int32)
    cv2.fillPoly(mask, [ring], 255)
    return mask


def _match_color(
    reference: np.ndarray,
    reference_mask: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    match_contrast: bool = True,
    brightness_strength: float = 1.0,
    tone_strength: float = 1.0,
) -> np.ndarray:
    """target의 밝기/색감을 reference(기본값: 전 사진)에 맞춘다 (PRD 6.4).
    얼굴 영역만으로 통계를 내어, 캔버스 여백(흰색 패딩)이 통계를 왜곡하지 않게 한다.
    brightness_strength/tone_strength로 밝기(L)와 색감(a·b) 보정의 세기를 따로 정한다.
    0이면 원본 그대로, 1이면 완전히 맞추고, 그 사이는 비례해서 섞는다. 보정이 원본과 너무
    달라 보일 때 끄고 켜는 것만으로는 거칠어서, 중간 세기를 고를 수 있게 열어뒀다.
    LAB 색공간이라 밝기와 색이 애초에 분리돼 있어 서로 간섭하지 않는다.

    match_contrast=False면 평균만 맞추고 표준편차(대비)는 건드리지 않는다. 통계 범위가
    전/후에서 정확히 같은 부위라는 보장이 없는 수동 2점 모드용이다 — 표준편차까지 맞추면
    작은 표본 차이가 색 전체를 증폭시켜 사진이 망가진다."""
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    result_lab = target_lab.copy()

    for channel in range(3):
        # LAB의 0번은 밝기(L), 1·2번은 색감(a·b)이다.
        strength = brightness_strength if channel == 0 else tone_strength
        if strength <= 0:
            continue
        reference_values = reference_lab[:, :, channel][reference_mask > 0]
        target_values = target_lab[:, :, channel][target_mask > 0]
        if reference_values.size == 0 or target_values.size == 0:
            continue

        reference_mean, reference_std = reference_values.mean(), reference_values.std() + 1e-6
        target_mean, target_std = target_values.mean(), target_values.std() + 1e-6
        gain = reference_std / target_std if match_contrast else 1.0

        channel_data = target_lab[:, :, channel]
        adjusted = (channel_data - target_mean) * gain + reference_mean
        # 세기 1은 예전 동작과 완전히 같아야 하므로 섞는 계산을 거치지 않는다.
        result_lab[:, :, channel] = (
            adjusted if strength == 1.0 else channel_data + (adjusted - channel_data) * strength
        )

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
    윤곽을 좌우로만 안쪽으로 당겨서 구레나룻·후퇴한 헤어라인을 침범하지 않게 한다.
    눈·코는 얼굴 윤곽 정확도와 무관하게 각자의 랜드마크로 직접 확실히 가리고, 마지막에
    닫기 연산으로 얼굴 마스크와 이어붙여 빈틈이 남지 않게 한다."""
    face_mask = _face_oval_mask(image_shape, points)

    # 좌우로만 당긴다. 사방으로 균일하게 깎으면 구레나룻을 피하려 세게 깎을수록
    # 턱·볼 커버리지까지 같이 사라진다.
    trim = max(1, int(min(image_shape) * MOSAIC_SIDE_TRIM_RATIO))
    face_mask = cv2.erode(face_mask, np.ones((1, 2 * trim + 1), np.uint8))

    brow_bottom_y = int(points[EYEBROW_IDX][:, 1].max() + BROW_EXPOSURE_PADDING_PX)
    brow_bottom_y = max(0, min(brow_bottom_y, image_shape[0]))

    mosaic_mask = face_mask.copy()
    mosaic_mask[:brow_bottom_y, :] = 0  # 눈썹 위(이마·헤어라인)는 모자이크 대상에서 제외

    # 윤곽을 옆으로 당긴 만큼 눈·코가 덜 덮일 수 있으니, 각자의 랜드마크로 직접 보강한다.
    # 눈과 코를 따로 껍질을 씌우면 그 사이(볼)가 섬처럼 비므로, 눈·코·입을 한 덩어리로
    # 묶어 하나의 볼록 껍질을 씌운다 — 사이가 통째로 채워진다.
    feature_mask = _feature_hull_mask(
        image_shape, points, EYES_IDX + NOSE_IDX + LIPS_IDX, MOSAIC_FEATURE_DILATE_RATIO
    )
    mosaic_mask = np.maximum(mosaic_mask, feature_mask)

    # 눈·코 마스크가 얼굴 마스크와 떨어져 섬처럼 남으면 블러가 끊겨 보인다.
    # 닫기(close)로 그 사이 틈을 메워 하나로 이어준다.
    close_kernel = _odd_kernel_size(image_shape, MOSAIC_CLOSE_RATIO)
    mosaic_mask = cv2.morphologyEx(
        mosaic_mask, cv2.MORPH_CLOSE, np.ones((close_kernel, close_kernel), np.uint8)
    )

    # 보강·연결 과정에서 눈썹 경계선 위로 번지지 않도록 마지막에 한 번 더 제외시킨다.
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


ManualPoints = list[tuple[float, float]]


def _manual_anchors(points: ManualPoints, shape: tuple[int, int]) -> np.ndarray:
    """정규화 좌표(0~1)로 받은 기준점들을 픽셀 좌표로 바꾼다.

    프론트엔드가 화면에 줄여 띄운 사진 위에서 클릭을 받으므로, 픽셀이 아니라 비율로
    주고받아야 표시 크기와 원본 크기가 달라도 어긋나지 않는다. 찍은 순서가 곧 전/후의
    대응 관계이므로 정렬하지 않고 그대로 둔다."""
    height, width = shape
    return np.array([[x * width, y * height] for x, y in points], dtype=np.float32)


def process_pair(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    apply_mosaic: bool = False,
    manual_points: tuple[ManualPoints, ManualPoints] | None = None,
    scale_adjust: float = 1.0,
    level_points: bool = False,
    brightness_strength: float = 1.0,
    tone_strength: float = 1.0,
    preview_width: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """전/후 사진을 같은 크기·각도·밝기로 맞춘다.

    manual_points를 주면 얼굴 검출을 건너뛰고 그 점들을 정렬 기준으로 쓴다. 이마·헤어라인
    클로즈업처럼 눈·코·입이 프레임 밖이라 얼굴 검출이 원리적으로 불가능한 사진용이다.
    찍은 순서가 전/후의 대응 관계이므로 개수가 같아야 하고, 한쪽만 받지는 않는다.

    brightness_strength/tone_strength는 밝기·색감 보정의 세기다 (0이면 원본, 1이면 완전히
    맞춤).

    preview_width를 주면 결과를 그 폭으로 줄여서 돌려준다. 슬라이더를 끄는 동안 실시간으로
    보여주기 위한 것이다 — 색 보정이 픽셀 수에 비례해서(전체 214ms 중 144ms) 줄이면 그만큼
    빨라지고, 전송량도 596KB에서 크게 줄어든다. 손을 떼면 원래 크기로 다시 받는다.

    level_points는 기준점들을 잇는 선을 수평으로 맞출지 여부다. 눈썹 위치를 자동으로 찾은
    경우처럼 그 점들이 무엇인지 알 때만 켠다.

    scale_adjust는 후 사진만 그만큼 더 키우거나 줄인다. 두 사진의 고개 각도가 다르면
    2D 변환으로는 어떤 기준점을 써도 한쪽이 틀어진다 — 눈썹을 맞추면 두상이 안 맞고
    두상을 맞추면 눈썹이 안 맞는다(실측: 같은 두 장에서 가로 기준 1.12배, 세로 기준
    0.89배). 자동으로 정할 수 없는 값이라 사람이 보고 미세조정할 수 있게 열어둔다."""
    warnings: list[str] = []

    if manual_points is not None:
        before_manual, after_manual = manual_points
        before_anchors = _manual_anchors(before_manual, before_bgr.shape[:2])
        after_anchors = _manual_anchors(after_manual, after_bgr.shape[:2])
        before_points = after_points = None
        before_transform, after_transform, residual = _manual_transforms(
            before_anchors, after_anchors, before_bgr.shape[:2], after_bgr.shape[:2], level_points
        )
        # 점이 3개 이상일 때만 의미가 있다 — 2개는 어떻게 찍든 오차가 0으로 떨어진다.
        spread = float(np.linalg.norm(before_anchors.max(axis=0) - before_anchors.min(axis=0)))
        if len(before_anchors) > 2 and spread > 0 and residual > spread * MANUAL_RESIDUAL_WARN_RATIO:
            warnings.append(
                "전/후 사진의 기준점이 서로 잘 맞지 않습니다. 같은 자리를 찍었는지 확인해주세요."
            )
        logger.info("manual: %d points, residual=%.1fpx (spread=%.1fpx)", len(before_anchors), residual, spread)
        # 미세조정은 정렬된 기준점 자리를 중심으로 한다 — 그래야 크기만 변하고 위치는 그대로다.
        alignment_pivot = _apply_transform(
            before_anchors.mean(axis=0).reshape(1, 2), before_transform
        )[0].astype(np.float64)
        if apply_mosaic:
            warnings.append(
                "직접 지정 모드에서는 얼굴 위치를 알 수 없어 모자이크를 적용하지 않았습니다."
            )
            apply_mosaic = False
    else:
        before_points = detect_landmarks(cv2.cvtColor(before_bgr, cv2.COLOR_BGR2RGB))
        after_points = detect_landmarks(cv2.cvtColor(after_bgr, cv2.COLOR_BGR2RGB))

        if before_points is None:
            warnings.append("전 사진에서 얼굴을 찾지 못해 원본을 그대로 반환합니다.")
        if after_points is None:
            warnings.append("후 사진에서 얼굴을 찾지 못해 원본을 그대로 반환합니다.")
        if before_points is None or after_points is None:
            return before_bgr, after_bgr, warnings

        before_transform = _brow_transform(before_points)
        after_transform = _brow_transform(after_points)
        alignment_pivot = np.array(CROP_ANCHOR, dtype=np.float64)

    aligned_before, before_canonical_points, before_valid_mask, before_warning = _align_to_canonical(
        before_bgr, before_transform, before_points
    )
    aligned_after, after_canonical_points, after_valid_mask, after_warning = _align_to_canonical(
        after_bgr, after_transform, after_points
    )
    before_alignment_points = (
        _apply_transform(before_anchors, before_transform) if manual_points is not None else None
    )
    after_alignment_points = (
        _apply_transform(after_anchors, after_transform) if manual_points is not None else None
    )
    if before_warning:
        warnings.append(f"전 사진: {before_warning}")
    if after_warning:
        warnings.append(f"후 사진: {after_warning}")

    # 크롭이 반드시 품고 있어야 할 기준점. 자동 모드는 눈썹 중앙(캔버스에 고정)이고,
    # 수동 모드는 사진을 캔버스 한가운데 맞춰뒀으므로 캔버스 중심을 쓴다.
    crop_anchor = (
        CROP_ANCHOR if manual_points is None else (CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2)
    )

    # 여백이 사진마다 같도록 정해진 틀로 자른다. 자동 모드는 얼굴 윤곽으로 정면/측면을
    # 가릴 수 있어 각각의 틀을 쓴다. 기준점만 있는 모드는 그 판정을 할 수 없으므로
    # 예전처럼 "흰 여백이 없는 최대 사각형"으로 남겨둔다.
    if manual_points is None:
        facing = _front_is_right(before_canonical_points)
        final_box = _fixed_frame_box(facing)
        before_box = after_box = final_box
        logger.info("frame: %s 고정 틀 %s", "정면" if facing is None else "측면", final_box)

        # 원본이 틀을 얼마나 덮는지 확인한다. 모자란 부분은 가장자리를 늘여 메우므로
        # 결과는 나오지만, 넓으면 번진 자국이 보인다.
        left, top, right, bottom = final_box
        for name, mask in (("전", before_valid_mask), ("후", after_valid_mask)):
            window = mask[max(0, top) : bottom, max(0, left) : right]
            covered = float((window > 0).mean()) if window.size else 0.0
            outside = (right - left) * (bottom - top) - window.size
            missing = 1.0 - (covered * window.size) / max(1, (right - left) * (bottom - top))
            if missing > FRAME_FILL_WARN_RATIO:
                warnings.append(
                    f"{name} 사진이 기준 틀보다 좁아 가장자리 {missing:.0%}를 늘여 채웠습니다. "
                    "더 넓게 찍으면 깨끗해집니다."
                )
            logger.info("frame: %s 사진 덮는 비율 %.1f%% (밖 %d px)", name, 100 * (1 - missing), max(0, outside))
    else:
        before_box = _max_valid_margins(before_valid_mask, crop_anchor)
        after_box = _max_valid_margins(after_valid_mask, crop_anchor)
        final_box = (
            max(before_box[0], after_box[0]),
            max(before_box[1], after_box[1]),
            min(before_box[2], after_box[2]),
            min(before_box[3], after_box[3]),
        )
        final_box = _adjust_side_margins(
            final_box,
            [before_canonical_points, after_canonical_points],
            crop_anchor[0],
            FRONT_MARGIN_TRIM_RATIO,
            BACK_MARGIN_EXTEND_RATIO,
            CANVAS_WIDTH,
        )
    if abs(scale_adjust - 1.0) > 1e-6:
        # 배율은 두 사진의 "상대" 크기다. 그래서 후 사진을 줄이는 대신 전 사진을 그만큼
        # 키워도 결과는 같은데, 축소는 부작용이 두 가지 있다 — 사진이 캔버스를 덜 덮어
        # 크롭 하단이 올라오고, 남은 내용도 기준점 쪽으로 끌려 내려온다. 둘이 겹쳐서
        # 눈썹이 잘렸다(실측: 80%에서 여유 17px, 70%에서 2.8px). 그래서 어느 쪽이든
        # "키우는" 방향으로만 적용한다.
        #
        # 확대 기준점은 크롭의 "아래 + 얼굴 앞쪽" 모서리다. 그 바깥은 어차피 잘려 나가는
        # 영역이라, 확대해도 안쪽(위·뒤통수 쪽)으로만 커지고 눈썹과 얼굴 앞 여백은 제자리에
        # 남는다. 예전에는 가로 기준점이 눈썹 중앙이라 얼굴 앞쪽이 밖으로 밀려났다
        # (실측: 130%에서 앞 여백이 19.9% -> 13.7%로 줄었다).
        #
        # 앞쪽이 어느 쪽인지는 기준점 위치로 정한다 — 눈썹은 머리 앞쪽에 있으므로,
        # 기준점이 더 가까이 붙어 있는 가장자리가 얼굴 앞이다.
        pivot_x = alignment_pivot[0]
        if before_alignment_points is not None and after_alignment_points is not None:
            anchors_x = np.concatenate(
                [before_alignment_points[:, 0], after_alignment_points[:, 0]]
            )
            to_left = float(anchors_x.min()) - final_box[0]
            to_right = final_box[2] - float(anchors_x.max())
            pivot_x = float(final_box[0] if to_left <= to_right else final_box[2])
        pivot = np.array([pivot_x, float(final_box[3])], dtype=np.float64)
        if scale_adjust > 1.0:
            after_transform = _compose(_scale_about(pivot, scale_adjust), after_transform)
            aligned_after, after_canonical_points, after_valid_mask, after_warning = _align_to_canonical(
                after_bgr, after_transform, after_points
            )
            after_box = _max_valid_margins(after_valid_mask, crop_anchor)
        else:
            before_transform = _compose(_scale_about(pivot, 1.0 / scale_adjust), before_transform)
            aligned_before, before_canonical_points, before_valid_mask, before_warning = _align_to_canonical(
                before_bgr, before_transform, before_points
            )
            before_box = _max_valid_margins(before_valid_mask, crop_anchor)

        # 기준점은 수동/자동제안 모드에만 있다. 자동(얼굴 검출) 모드에서는 없으므로
        # 건드리지 않는다 — 예전에 조건 없이 참조해서 크기 조절이 500 에러로 죽었다.
        if manual_points is not None:
            before_alignment_points = _apply_transform(before_anchors, before_transform)
            after_alignment_points = _apply_transform(after_anchors, after_transform)

        # 자동 모드는 틀이 고정이라 배율을 바꿔도 크롭을 다시 잡지 않는다. 다시 잡으면
        # 사진마다 여백이 달라져서 고정 틀을 쓰는 의미가 없어진다.
        if manual_points is not None:
            final_box = (
                max(before_box[0], after_box[0]),
                max(before_box[1], after_box[1]),
                min(before_box[2], after_box[2]),
                min(before_box[3], after_box[3]),
            )
            final_box = _adjust_side_margins(
                final_box,
                [before_canonical_points, after_canonical_points],
                crop_anchor[0],
                FRONT_MARGIN_TRIM_RATIO,
                BACK_MARGIN_EXTEND_RATIO,
                CANVAS_WIDTH,
            )
        logger.info(
            "scale adjust: x%.3f applied to %s (pivot y=%.0f)",
            scale_adjust, "after" if scale_adjust > 1 else "before", pivot[1],
        )

    # 기준점(눈썹) 아래 여백을 확보한다. 크롭은 원래 "흰 여백이 전혀 없는 최대 사각형"인데,
    # 사진이 눈썹 바로 아래에서 끝나면 그 규칙만으로는 눈썹이 경계에 걸린다. 필요한 만큼만
    # 아래로 넓히고, 모자란 부분은 가장자리를 늘여 채운다.
    anchor_sets = [p for p in (before_alignment_points, after_alignment_points) if p is not None]
    if anchor_sets:
        lowest = max(float(p[:, 1].max()) for p in anchor_sets)
        spread = max(float(np.linalg.norm(p[1] - p[0])) for p in anchor_sets)
        needed = lowest + spread * ANCHOR_BOTTOM_MARGIN_RATIO
        limit = final_box[3] + (final_box[3] - final_box[1]) * ANCHOR_BOTTOM_MAX_STRETCH
        if needed > final_box[3]:
            widened = min(int(round(min(needed, limit))), CANVAS_HEIGHT)
            logger.info(
                "crop: 눈썹 아래 여백 확보 %d -> %d (필요 %d)", final_box[3], widened, int(needed)
            )
            final_box = (final_box[0], final_box[1], final_box[2], widened)

    logger.info("crop: before=%s after=%s -> using=%s (canvas=%dx%d)", before_box, after_box, final_box, CANVAS_WIDTH, CANVAS_HEIGHT)
    aligned_before, before_canonical_points = _crop_to_box(aligned_before, before_canonical_points, final_box)
    aligned_after, after_canonical_points = _crop_to_box(aligned_after, after_canonical_points, final_box)

    # 미리보기는 여기서 줄인다. 정렬·크롭까지 끝난 뒤라 결과 구도는 원본과 같고, 뒤따르는
    # 색 보정만 작은 이미지에서 돌아간다.
    if preview_width and aligned_before.shape[1] > preview_width:
        ratio = preview_width / aligned_before.shape[1]
        size = (preview_width, max(1, round(aligned_before.shape[0] * ratio)))
        aligned_before = cv2.resize(aligned_before, size, interpolation=cv2.INTER_AREA)
        aligned_after = cv2.resize(aligned_after, size, interpolation=cv2.INTER_AREA)
        if before_canonical_points is not None:
            before_canonical_points = before_canonical_points * ratio
        if after_canonical_points is not None:
            after_canonical_points = after_canonical_points * ratio

    if manual_points is None:
        before_color_mask = _face_oval_mask(aligned_before.shape[:2], before_canonical_points)
        after_color_mask = _face_oval_mask(aligned_after.shape[:2], after_canonical_points)
    else:
        before_color_mask = _skin_mask(aligned_before)
        after_color_mask = _skin_mask(aligned_after)
    matched_after = _match_color(
        aligned_before,
        before_color_mask,
        aligned_after,
        after_color_mask,
        match_contrast=manual_points is None,
        brightness_strength=brightness_strength,
        tone_strength=tone_strength,
    )

    if apply_mosaic:
        aligned_before = _apply_face_mosaic(aligned_before, before_canonical_points)
        matched_after = _apply_face_mosaic(matched_after, after_canonical_points)

    return aligned_before, matched_after, warnings
