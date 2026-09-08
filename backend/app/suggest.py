"""얼굴 검출이 안 되는 클로즈업 사진에서 눈썹 양 끝 위치를 제안한다.

이마·헤어라인만 크게 찍은 사진은 눈·코·입이 프레임 밖이라 MediaPipe가 얼굴을 못 찾는다
(실측: 16장 중 15장 실패). 그런 사진도 비전 모델은 "눈썹"을 알아본다. 실측 결과 16장 모두
눈썹 양 끝을 찾았고, 좌표는 사람이 클릭하는 정밀도(표준편차 2~7픽셀)에 견줄 만했다.

어디까지나 "제안"이다. 결과는 화면에 점으로 표시되고, 사용자가 보고 고칠 수 있다.
자동으로 변환까지 진행하지는 않는다 — 틀렸을 때 아무도 모르는 상황을 만들지 않기 위해서다.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import statistics
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from PIL import Image

logger = logging.getLogger("suggest")

MODEL = "gemini-3.8-flash"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
TIMEOUT_SECONDS = 60

# 모델에 보낼 때 줄이는 폭. 좌표는 비율로 받으므로 원본 크기와 무관하게 되돌릴 수 있고,
# 줄여 보내면 요금과 응답 시간이 함께 줄어든다.
SEND_MAX_WIDTH = 1024

# 한 사진에 몇 번 물어볼지. 동시에 보내므로 횟수를 늘려도 기다리는 시간은 늘지 않는다.
SAMPLE_COUNT = 5
# 중앙값에서 이 비율(사진 폭 대비) 안에 들어오면 "같은 답"으로 본다.
AGREEMENT_LIMIT = 0.03
# 이만큼은 서로 일치해야 제안한다. 과반이 못 모이면 사진 자체가 애매한 것으로 본다.
MIN_AGREEING = 3

PROMPT = """이 사진은 이마와 헤어라인을 크게 찍은 클로즈업이라 얼굴 아래쪽은 잘려 있을 수 있어.
보이는 눈썹에서 아래 두 지점을 찍어줘. 눈썹이 여러 개 보이면 가장 크게 보이는 것 하나만.
- brow_end_left: 그 눈썹에서 화면 왼쪽 끝
- brow_end_right: 그 눈썹에서 화면 오른쪽 끝
눈썹이 전혀 안 보이면 빈 배열 []을 출력해. 반드시 JSON만:
[{"label":"brow_end_left","point":[y,x]},{"label":"brow_end_right","point":[y,x]}]
point는 [y, x] 순서, 0~1000 정규화."""


def _api_key() -> str:
    """환경변수를 먼저 보고, 없으면 개발 편의를 위해 프로젝트 루트의 .env를 읽는다."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def is_available() -> bool:
    return bool(_api_key())


def _encode(image_bytes: bytes) -> str:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    if image.width > SEND_MAX_WIDTH:
        height = round(image.height * SEND_MAX_WIDTH / image.width)
        image = image.resize((SEND_MAX_WIDTH, height), Image.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _ask_once(encoded: str, key: str) -> list[tuple[float, float]] | None:
    """눈썹 양 끝을 0~1 비율 좌표로 돌려준다. 못 찾으면 None."""
    body = json.dumps(
        {
            "contents": [
                {
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": encoded}},
                        {"text": PROMPT},
                    ]
                }
            ],
            # 같은 사진에는 같은 답이 나오도록 고정한다.
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        ENDPOINT, data=body, headers={"x-goog-api-key": key, "Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.info("suggest: 호출 실패 (%s)", exc)
        return None

    try:
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        logger.info("suggest: 예상과 다른 응답")
        return None

    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return None
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    found: dict[str, tuple[float, float]] = {}
    for item in items:
        label, point = item.get("label"), item.get("point")
        if label in ("brow_end_left", "brow_end_right") and point and len(point) == 2:
            y, x = point
            found[label] = (float(x) / 1000, float(y) / 1000)
    if len(found) != 2:
        return None
    if any(not (0.0 <= v <= 1.0) for pair in found.values() for v in pair):
        return None
    return [found["brow_end_left"], found["brow_end_right"]]


def suggest_brow_points(image_bytes: bytes) -> list[tuple[float, float]] | None:
    """눈썹 양 끝을 제안한다. 자신 없으면 None을 돌려주고 화면은 빈 상태로 둔다.

    여러 번 물어 중앙값을 쓴다. 이 모델은 추론 모델이라 temperature를 0으로 둬도 답이
    매번 같지는 않은데, 실측해보니 대부분은 아주 촘촘하고(표준편차 2~7픽셀) 가끔 한 번씩
    엉뚱한 곳을 찍는다 — 8번 중 1번이 163에서 32로 튀는 식이다. 평균이나 두 번의 일치
    검사는 그 한 번에 통째로 휘둘리지만, 중앙값은 영향을 받지 않는다.

    그러고도 과반이 모이지 않으면(좌우로 두 사람이 붙은 사진처럼 매번 다른 얼굴을 고르는
    경우) 제안하지 않는다 — 틀린 걸 자신 있게 내놓느니 비워두는 게 낫다."""
    key = _api_key()
    if not key:
        return None

    encoded = _encode(image_bytes)
    # 질문들은 서로 독립이라 동시에 보낸다. 순서대로 보내면 횟수만큼 기다려야 한다.
    with ThreadPoolExecutor(max_workers=SAMPLE_COUNT) as pool:
        answers = [a for a in pool.map(lambda _: _ask_once(encoded, key), range(SAMPLE_COUNT)) if a]
    if len(answers) < MIN_AGREEING:
        logger.info("suggest: 응답 %d개뿐이라 제안하지 않음", len(answers))
        return None

    median = [
        (
            statistics.median(answer[index][0] for answer in answers),
            statistics.median(answer[index][1] for answer in answers),
        )
        for index in (0, 1)
    ]

    agreeing = sum(
        1
        for answer in answers
        if all(
            max(abs(answer[i][0] - median[i][0]), abs(answer[i][1] - median[i][1])) <= AGREEMENT_LIMIT
            for i in (0, 1)
        )
    )
    if agreeing < MIN_AGREEING:
        logger.info("suggest: %d/%d만 일치해 제안하지 않음", agreeing, len(answers))
        return None

    logger.info("suggest: %d/%d 일치", agreeing, len(answers))
    return median
