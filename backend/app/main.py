import base64
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
from PIL import Image, ImageOps

from .animation import build_crossfade_gif
from .pipeline import process_pair

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


def _upload_to_bgr(data: bytes) -> np.ndarray:
    # 휴대폰 사진은 픽셀을 눕혀서 저장하고 "돌려서 보여라"는 EXIF 태그만 붙이는 경우가
    # 많다. PIL은 그 태그를 자동 적용하지 않으므로, 그대로 쓰면 옆으로 누운 얼굴이
    # 들어가 얼굴 검출에 실패한다.
    image = ImageOps.exif_transpose(Image.open(BytesIO(data))).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _bgr_to_data_url(image_bgr: np.ndarray) -> str:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=92)
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


@app.post("/api/process", dependencies=[Depends(require_password)])
async def process(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    mosaic: bool = Form(False),
) -> dict:
    before_bgr = _upload_to_bgr(await before.read())
    after_bgr = _upload_to_bgr(await after.read())

    result_before, result_after, warnings = await run_in_threadpool(
        process_pair, before_bgr, after_bgr, mosaic
    )

    return {
        "before": _bgr_to_data_url(result_before),
        "after": _bgr_to_data_url(result_after),
        "warnings": warnings,
    }


@app.post("/api/gif", dependencies=[Depends(require_password)])
async def make_gif(before: UploadFile = File(...), after: UploadFile = File(...)) -> Response:
    """전→후로 스르륵 넘어가는 GIF를 만든다.

    이미 변환이 끝난 두 장을 그대로 받는다. 원본을 다시 받아 정렬·보정을 다시 돌리면
    시간이 오래 걸리고, 화면에 보이는 결과(모자이크 적용 여부 포함)와 달라질 수 있다."""
    before_image = Image.open(BytesIO(await before.read()))
    after_image = Image.open(BytesIO(await after.read()))

    gif_bytes = await run_in_threadpool(build_crossfade_gif, before_image, after_image)
    return Response(content=gif_bytes, media_type="image/gif")


# 빌드된 프론트엔드를 같은 서버에서 서빙한다 — 배포가 컨테이너 하나로 끝나고, 프론트엔드가
# 상대경로로 API를 부를 수 있어 접속 주소가 무엇이든 그대로 동작한다.
# 반드시 /api 라우트를 모두 등록한 뒤에 마운트해야 "/"가 그것들을 가리지 않는다.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
    logger.info("serving frontend from %s", FRONTEND_DIST)
else:
    logger.info("no frontend build at %s (dev mode: run vite separately)", FRONTEND_DIST)
