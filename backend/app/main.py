import base64
import json
import logging
import os
import secrets
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from .animation import build_crossfade_gif
from .pipeline import (
    BRUSH_MAX_RADIUS_RATIO,
    BRUSH_MIN_RADIUS_RATIO,
    MAX_INPUT_SIDE,
    BrushStroke,
    UnusableResult,
    mosaic_only,
    process_photos,
)
from .suggest import can_level, is_available as suggest_available, matched_points

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

# 공용 암호. 환경변수로 넣으면 그 암호를 아는 사람만 쓸 수 있고, 안 넣으면(로컬 개발)
# 인증 없이 그냥 동작한다. 코드에 암호를 박지 않으려고 환경변수로 뺐다.
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

if APP_PASSWORD and not APP_PASSWORD.isascii():
    # 브라우저마다 비ASCII 헤더 처리가 달라 로그인이 실패할 수 있다. 관리자가 원인을
    # 바로 알 수 있도록 시작할 때 경고를 남긴다.
    logger.warning("APP_PASSWORD에 영문·숫자 외 문자가 있습니다. 브라우저에 따라 로그인이 실패할 수 있습니다.")

app = FastAPI(title="전후사진 변환 API")


def _password_matches(candidate: str) -> bool:
    """헤더로 받은 암호가 설정된 암호와 같은지 확인한다.

    문자열을 compare_digest에 그대로 넘기면 한글 등 비ASCII 암호에서 TypeError가 나므로
    바이트로 비교한다. 또 HTTP 헤더 값은 latin-1로 디코딩되기 때문에, 클라이언트가 UTF-8
    바이트를 그대로 실어 보낸 경우 원래 바이트를 되살려서도 한 번 더 비교한다.
    """
    expected = APP_PASSWORD.encode("utf-8")

    attempts = [candidate.encode("utf-8")]
    try:
        attempts.append(candidate.encode("latin-1"))
    except UnicodeEncodeError:
        pass

    # 타이밍 공격을 피하려고 단순 == 대신 compare_digest를 쓴다.
    return any(secrets.compare_digest(attempt, expected) for attempt in attempts)


def require_password(x_app_password: str = Header(default="")) -> None:
    if not APP_PASSWORD:
        return
    if not _password_matches(x_app_password):
        raise HTTPException(status_code=401, detail="암호가 올바르지 않습니다.")

# 개발 중에는 프론트엔드가 5173에서 따로 뜨므로 CORS가 필요하다. 배포 시에는 아래에서
# 빌드된 프론트엔드를 같은 오리진으로 서빙하므로 이 설정은 쓰이지 않는다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


# 크기는 _upload_to_bgr에서 직접 검사한다. PIL 기본 한계(1억 8천만 화소)는 파일을 열자마자
# 거절해서, 줄여서 풀면 충분히 쓸 수 있는 큰 JPEG까지 막아버린다.
Image.MAX_IMAGE_PIXELS = None
# 줄여서 풀 수 없는 형식(PNG 등)은 이 화소 수까지만 받는다. 통째로 풀어야 하고, 풀고 나서도
# 변환 과정에서 복사본이 서너 벌 생긴다 (2,500만 화소면 한 벌에 75MB).
MAX_UNREDUCIBLE_PIXELS = 25_000_000


def _upload_to_bgr(data: bytes, side: str) -> np.ndarray:
    """업로드된 파일을 BGR 배열로 푼다. 열 수 없으면 이유를 담아 400을 낸다.

    side는 "전"/"후" — 사용자가 어느 사진을 바꿔야 하는지 알 수 있게 문구에 넣는다."""
    try:
        image = Image.open(BytesIO(data))
        width, height = image.size
        longest = max(width, height)
        if longest > MAX_INPUT_SIDE:
            # 여기서 줄이지 않으면 통째로 푼 뒤에야 줄일 수 있다. 4725x29451 스크롤 캡처는
            # 푸는 데만 1,479MB를 써서, 1Gi인 운영 서버에서는 인스턴스가 죽는다.
            # JPEG는 헤더만 읽은 이 시점에 1/2~1/8 크기로 바로 풀 수 있다 (draft). 다른
            # 형식은 draft가 아무 일도 안 하므로 크기가 그대로 남고, 아래 검사에 걸린다.
            ratio = MAX_INPUT_SIDE / longest
            image.draft("RGB", (max(1, int(width * ratio)), max(1, int(height * ratio))))
            if image.size[0] * image.size[1] > MAX_UNREDUCIBLE_PIXELS:
                raise HTTPException(
                    status_code=400,
                    detail=f"{side} 사진이 너무 큽니다 ({width}x{height}). "
                    "크기를 줄이거나 JPG로 저장해서 다시 올려주세요.",
                )
        # 휴대폰 사진은 픽셀을 눕혀서 저장하고 "돌려서 보여라"는 EXIF 태그만 붙이는 경우가
        # 많다. PIL은 그 태그를 자동 적용하지 않으므로, 그대로 쓰면 옆으로 누운 얼굴이
        # 들어가 얼굴 검출에 실패한다.
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (HTTPException, MemoryError):
        # 메모리 부족은 파일 탓이 아니다. "손상된 파일"로 안내하면 엉뚱한 걸 고치게 된다.
        raise
    except Exception as exc:
        # 사진이 아닌 파일, 중간에 잘린 파일, 서버가 모르는 형식(HEIC 등). 사용자의 파일
        # 문제라 500이 아니라 400으로, 무엇을 하면 되는지 알려준다. 형식별 예외가
        # 제각각이라(UnidentifiedImageError, OSError, SyntaxError, struct.error...)
        # 넓게 잡되, 원인은 로그에 남긴다.
        logger.warning("%s 사진을 열 수 없음 (%d bytes): %r", side, len(data), exc)
        raise HTTPException(
            status_code=400,
            detail=f"{side} 사진 파일을 열 수 없습니다. 사진이 아니거나 손상된 파일입니다. "
            "JPG로 다시 저장해서 올려주세요.",
        ) from exc
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _bgr_to_data_url(image_bgr: np.ndarray, quality: int = 92) -> str:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/auth")
def auth_status(x_app_password: str = Header(default="")) -> dict[str, bool]:
    """프론트엔드가 "암호가 필요한 서버인지", "지금 가진 암호가 맞는지"를 확인하는 용도."""
    return {
        "required": bool(APP_PASSWORD),
        "valid": not APP_PASSWORD or _password_matches(x_app_password),
    }


# 기준점이 이보다 좁게 모여 있으면 정렬 기준으로 쓸 수 없다. 간격이 좁을수록 각도·배율
# 오차가 그만큼 증폭되고, 거의 같은 자리면 변환 자체가 성립하지 않는다.
MIN_MANUAL_SPREAD = 0.05
# 점을 더 찍을수록 정확해지지만 이 이상은 실익이 없고 화면만 복잡해진다.
MAX_MANUAL_POINTS = 6
# 후 사진 크기 미세조정의 허용 범위. 이 밖으로 나갈 만큼 어긋났다면 미세조정으로 덮을
# 문제가 아니라 기준점이나 사진 자체를 다시 봐야 한다.
MIN_SCALE_ADJUST = 0.6
MAX_SCALE_ADJUST = 1.6
# 밝기·색감 보정 세기의 허용 범위. 0이면 원본, 1이면 완전히 맞춤, 그 이상은 과보정이다.
MAX_MATCH_STRENGTH = 1.5
# 슬라이더를 끄는 동안 보내는 미리보기의 폭과 품질. 화면에서 보기엔 충분하면서 응답이
# 훨씬 가볍다 (전체 크기 596KB -> 이 크기면 수십 KB).
PREVIEW_WIDTH = 800
PREVIEW_JPEG_QUALITY = 80
# 붓 자국의 상한. 막아두지 않으면 마우스를 오래 끄는 것만으로 끝없이 커진 요청이 올라온다.
MAX_BRUSH_STROKES = 200
MAX_BRUSH_POINTS = 2000


PAIR_SIDES = ("before", "after")
# 경과 사진 3장. 전 사진이 기준이고 나머지를 거기에 맞춘다.
SET_SIDES = ("before", "middle", "after")
SIDE_NAMES = {"before": "전", "middle": "중간", "after": "후"}


def _parse_strokes(
    raw: str, field: str, side_keys: tuple[str, ...] = PAIR_SIDES
) -> list[list[BrushStroke]] | None:
    """붓으로 칠한 자국을 검증한다. 돌려주는 목록은 side_keys 순서를 따른다.

    형식: {"before": [{"radius": 0.03, "points": [[x, y], ...]}, ...], "middle": [...], "after": [...]}
    좌표는 사진 크기 대비 0~1 비율, 반지름은 짧은 변 대비 비율이다. 빠진 쪽은 빈 목록이다."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
        sides: list[list[BrushStroke]] = []
        for side in side_keys:
            raw_strokes = data.get(side) or []
            if len(raw_strokes) > MAX_BRUSH_STROKES:
                raise ValueError(f"{side}: 칠한 자국이 너무 많습니다 ({len(raw_strokes)}개)")
            strokes = []
            for stroke in raw_strokes:
                radius = float(stroke["radius"])
                if not BRUSH_MIN_RADIUS_RATIO <= radius <= BRUSH_MAX_RADIUS_RATIO:
                    raise ValueError(f"{side}: 붓 크기가 범위를 벗어났습니다 ({radius})")
                points = [(float(x), float(y)) for x, y in stroke["points"]]
                if not 1 <= len(points) <= MAX_BRUSH_POINTS:
                    raise ValueError(f"{side}: 자국 하나의 점 개수가 잘못됐습니다 ({len(points)}개)")
                if any(not (0.0 <= v <= 1.0) for point in points for v in point):
                    raise ValueError(f"{side}: 칠한 자리가 사진 범위를 벗어났습니다")
                strokes.append(BrushStroke(points=tuple(points), radius=radius))
            sides.append(strokes)
        return sides
    except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"{field}가 올바르지 않습니다: {exc}") from exc


def _parse_manual_points(raw: str, side_keys: tuple[str, ...] = PAIR_SIDES) -> list[list] | None:
    """프론트엔드가 보낸 수동 기준점을 검증한다. 돌려주는 목록은 side_keys 순서를 따른다.

    형식: {"before": [[x, y], ...], "middle": [...], "after": [[x, y], ...]}  (각각 2~6개)
    좌표는 사진 크기 대비 0~1 비율이다. 화면에 줄여 띄운 사진 위에서 클릭을 받으므로
    픽셀이 아니라 비율로 주고받아야 표시 크기와 원본 크기가 달라도 어긋나지 않는다.
    찍은 순서가 사진끼리의 대응 관계이므로 개수가 모두 같아야 한다."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
        sides = {}
        for side in side_keys:
            if side not in data:
                raise ValueError(f"{SIDE_NAMES[side]} 사진의 기준점이 없습니다")
            points = [(float(x), float(y)) for x, y in data[side]]
            if not 2 <= len(points) <= MAX_MANUAL_POINTS:
                raise ValueError(f"{side}: 점은 2~{MAX_MANUAL_POINTS}개여야 합니다 (지금 {len(points)}개)")
            if any(not (0.0 <= v <= 1.0) for point in points for v in point):
                raise ValueError(f"{side}: 좌표가 사진 범위를 벗어났습니다")
            xs = [x for x, _ in points]
            ys = [y for _, y in points]
            spread = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
            if spread < MIN_MANUAL_SPREAD:
                raise ValueError(f"{side}: 점들이 너무 가까이 모여 있습니다")
            sides[side] = points
        if len({len(points) for points in sides.values()}) > 1:
            names = "/".join(SIDE_NAMES[side] for side in side_keys)
            raise ValueError(f"{names} 사진의 점 개수가 다릅니다")
        return [sides[side] for side in side_keys]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"기준점이 올바르지 않습니다: {exc}") from exc


@app.post("/api/process", dependencies=[Depends(require_password)])
async def process(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    # 경과 사진의 중간 시점. 있으면 전/중간/후 세 장을 한 틀에 맞춘다.
    middle: UploadFile | None = File(None),
    mosaic: bool = Form(False),
    points: str = Form(""),
    scale_adjust: float = Form(1.0),
    middle_scale_adjust: float = Form(1.0),
    level: bool = Form(False),
    brightness_strength: float = Form(1.0),
    tone_strength: float = Form(1.0),
    preview: bool = Form(False),
    result_strokes: str = Form(""),
) -> dict:
    side_keys = SET_SIDES if middle is not None else PAIR_SIDES
    manual_points = _parse_manual_points(points, side_keys)
    result = _parse_strokes(result_strokes, "결과에 칠한 자국", side_keys)
    if result:
        logger.info("strokes: 결과에 칠한 자국 %s", "/".join(str(len(s)) for s in result))
    for name, value in (("밝기", brightness_strength), ("색감", tone_strength)):
        if not 0.0 <= value <= MAX_MATCH_STRENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"{name} 보정 세기는 0~{MAX_MATCH_STRENGTH:.0%} 사이여야 합니다.",
            )
    for value in (scale_adjust, middle_scale_adjust):
        if not MIN_SCALE_ADJUST <= value <= MAX_SCALE_ADJUST:
            raise HTTPException(
                status_code=400,
                detail=f"크기 조정은 {MIN_SCALE_ADJUST:.0%}~{MAX_SCALE_ADJUST:.0%} 사이여야 합니다.",
            )
    uploads = {"before": before, "middle": middle, "after": after}
    images = [_upload_to_bgr(await uploads[side].read(), SIDE_NAMES[side]) for side in side_keys]
    scale_adjusts = [middle_scale_adjust, scale_adjust] if middle is not None else [scale_adjust]

    try:
        results, warnings = await run_in_threadpool(
            process_photos,
            images,
            mosaic,
            manual_points,
            scale_adjusts,
            level,
            brightness_strength,
            tone_strength,
            PREVIEW_WIDTH if preview else 0,
            result,
        )
    except UnusableResult as exc:
        # 잘못된 입력이지 서버 오류가 아니다. 500 대신 이유를 돌려준다.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    quality = PREVIEW_JPEG_QUALITY if preview else 92
    response: dict = {side: _bgr_to_data_url(image, quality) for side, image in zip(side_keys, results)}
    response["warnings"] = warnings
    return response


@app.post("/api/mosaic", dependencies=[Depends(require_password)])
async def mosaic(
    before: UploadFile = File(...),
    after: UploadFile | None = File(None),
    face: bool = Form(True),
    strokes: str = Form(""),
) -> dict:
    """정렬하지 않고 모자이크만 입힌다. 사진은 원래 크기·구도 그대로 돌려준다.

    전/후를 맞출 필요 없이 가리기만 하면 되는 경우를 위한 길이다. 사진 한 장만 보내도 된다."""
    painted = _parse_strokes(strokes, "칠한 자국")
    before_strokes, after_strokes = painted if painted else ([], [])

    warnings: list[str] = []
    result: dict[str, str | None | list[str]] = {"before": None, "after": None}
    for key, side, upload, side_strokes in (
        ("before", "전", before, before_strokes),
        ("after", "후", after, after_strokes),
    ):
        if upload is None:
            continue
        image = _upload_to_bgr(await upload.read(), side)
        mosaicked, warning = await run_in_threadpool(mosaic_only, image, side_strokes, face)
        if warning:
            warnings.append(f"{side} 사진: {warning}")
        result[key] = _bgr_to_data_url(mosaicked)
    result["warnings"] = warnings
    return result


@app.post("/api/suggest-points", dependencies=[Depends(require_password)])
async def suggest_points(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    middle: UploadFile | None = File(None),
) -> dict:
    """전(·중간)·후 사진에서 정렬 기준점을 제안한다 (얼굴 검출이 실패했을 때의 출발점).

    눈썹뿐 아니라 눈·코·귀까지 보이는 지점을 모두 찾아, 모든 사진에 공통으로 있는 것만
    짝지어 돌려준다. 지점이 멀리 퍼질수록 배율·각도가 정확해진다.

    어디까지나 제안이라 실패해도 오류로 처리하지 않는다 — 화면은 빈 상태로 두고 사용자가
    직접 찍으면 된다. 그래서 available=false나 points=null도 정상 응답이다."""
    side_keys = SET_SIDES if middle is not None else PAIR_SIDES
    empty: dict = {side: None for side in side_keys} | {"labels": [], "level": False}
    if not suggest_available():
        return {"available": False} | empty

    uploads = {"before": before, "middle": middle, "after": after}
    images = [await uploads[side].read() for side in side_keys]
    result = await run_in_threadpool(matched_points, images)
    if result is None:
        return {"available": True} | empty

    points_list, labels = result
    return {
        "available": True,
        **dict(zip(side_keys, points_list)),
        "labels": labels,
        # 양쪽 눈썹이 다 잡혔을 때만 수평을 맞출 수 있다 (자세한 이유는 suggest.can_level).
        "level": can_level(labels),
    }


@app.post("/api/gif", dependencies=[Depends(require_password)])
async def make_gif(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    middle: UploadFile | None = File(None),
) -> Response:
    """전→(중간→)후로 스르륵 넘어가는 GIF를 만든다.

    이미 변환이 끝난 사진들을 그대로 받는다. 원본을 다시 받아 정렬·보정을 다시 돌리면
    시간이 오래 걸리고, 화면에 보이는 결과(모자이크 적용 여부 포함)와 달라질 수 있다."""
    images = []
    ordered = [("전", before), ("중간", middle), ("후", after)]
    for side, upload in [(side, upload) for side, upload in ordered if upload is not None]:
        # 정상이라면 이 앱이 만든 결과 JPEG가 온다. 그 밖의 것은 이유를 알려주고 거절한다.
        try:
            image = Image.open(BytesIO(await upload.read()))
            if image.size[0] * image.size[1] > MAX_UNREDUCIBLE_PIXELS:
                raise ValueError(f"too large: {image.size}")
            image.load()
        except Exception as exc:
            logger.warning("GIF: %s 사진을 열 수 없음: %r", side, exc)
            raise HTTPException(
                status_code=400,
                detail=f"GIF를 만들 {side} 사진을 열 수 없습니다. 변환을 다시 해주세요.",
            ) from exc
        images.append(image)

    gif_bytes = await run_in_threadpool(build_crossfade_gif, images)
    return Response(content=gif_bytes, media_type="image/gif")


# 빌드된 프론트엔드를 같은 서버에서 서빙한다 — 배포가 컨테이너 하나로 끝나고, 프론트엔드가
# 상대경로로 API를 부를 수 있어 접속 주소가 무엇이든 그대로 동작한다.
# 반드시 /api 라우트를 모두 등록한 뒤에 마운트해야 "/"가 그것들을 가리지 않는다.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
    logger.info("serving frontend from %s", FRONTEND_DIST)
else:
    logger.info("no frontend build at %s (dev mode: run vite separately)", FRONTEND_DIST)
