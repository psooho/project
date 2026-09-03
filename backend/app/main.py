import base64
import logging
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from .pipeline import process_pair

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

app = FastAPI(title="전후사진 변환 API")

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


@app.post("/api/process")
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


# 빌드된 프론트엔드를 같은 서버에서 서빙한다 — 배포가 컨테이너 하나로 끝나고, 프론트엔드가
# 상대경로로 API를 부를 수 있어 접속 주소가 무엇이든 그대로 동작한다.
# 반드시 /api 라우트를 모두 등록한 뒤에 마운트해야 "/"가 그것들을 가리지 않는다.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
    logger.info("serving frontend from %s", FRONTEND_DIST)
else:
    logger.info("no frontend build at %s (dev mode: run vite separately)", FRONTEND_DIST)
