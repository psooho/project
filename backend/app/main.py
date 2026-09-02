import base64
from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="전후사진 변환 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


# TODO: 얼굴 랜드마크 기반 정렬(각도·크기 정규화), 밝기/색감 자동 보정,
# 헤어라인·눈썹만 남기고 나머지를 가리는 모자이크 처리를 이 파이프라인에 추가한다 (PRD 6.2~6.5).
# 지금은 두 사진을 그대로 반환하는 통과(pass-through) 구현이다.
def process_pair(before: Image.Image, after: Image.Image) -> tuple[Image.Image, Image.Image]:
    return before, after


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/process")
async def process(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
) -> dict[str, str]:
    before_image = Image.open(BytesIO(await before.read()))
    after_image = Image.open(BytesIO(await after.read()))

    processed_before, processed_after = process_pair(before_image, after_image)

    return {
        "before": _to_data_url(processed_before),
        "after": _to_data_url(processed_after),
    }
