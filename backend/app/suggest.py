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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# 한 지점을 채택하려면 최소 몇 번은 같은 자리로 나와야 하는지. 몇 개를 모았든 과반은
# 되어야 하고, 아무리 적어도 2번은 나와야 한다 — 한 번뿐인 답은 이상치와 구별할 수 없다.
def _needed(collected: int) -> int:
    return max(2, (collected + 1) // 2)


# 응답이 이보다 적으면 판단할 근거가 없다고 보고 포기한다.
MIN_ANSWERS = 3
# 이만큼 모이면 나머지는 기다리지 않는다. 응답 시간이 들쭉날쭉해서(중앙 7~14초, 최대 22초)
# 전부 기다리면 항상 제일 느린 하나에 발이 묶인다.
ENOUGH_ANSWERS = 4

# 쓸 수 있는 기준점 목록. 순서는 결과를 정렬할 때의 우선순위이기도 하다.
#
# 고르는 기준은 하나다 — 수술로 변하지 않을 것. 그래서 헤어라인·머리카락은 빠져 있다
# (이마축소 모발이식에서 바로 그 부분이 내려온다). 눈썹 산등성이(brow_peak)도 뺐는데,
# "가장 높은 지점"이 실측에서 47~115픽셀까지 흔들렸다 — 완만한 곡선의 꼭대기는 사람도
# 모델도 짚기 어렵다.
LANDMARKS = [
    ("left_brow_outer", "화면 왼쪽 눈썹의 바깥(화면 왼쪽) 끝"),
    ("left_brow_inner", "화면 왼쪽 눈썹의 안쪽(코 쪽) 끝"),
    ("right_brow_inner", "화면 오른쪽 눈썹의 안쪽(코 쪽) 끝"),
    ("right_brow_outer", "화면 오른쪽 눈썹의 바깥(화면 오른쪽) 끝"),
    ("left_eye_outer", "화면 왼쪽 눈의 눈꼬리"),
    ("right_eye_outer", "화면 오른쪽 눈의 눈꼬리"),
    ("nose_bridge", "두 눈 사이 콧대 위"),
    ("ear_top", "귀가 머리에 붙는 맨 위 지점"),
]

# 정렬의 수평 기준이 되는 지점들. 눈썹은 좌우가 거의 같은 높이라, 이 점들을 잇는 선을
# 수평으로 맞추면 사진이 똑바로 선다. 자동(MediaPipe) 모드가 두 눈썹으로 하는 것과 같다.
BROW_LANDMARKS = ("left_brow_outer", "left_brow_inner", "right_brow_inner", "right_brow_outer")

PROMPT = """이마와 헤어라인을 크게 찍은 클로즈업 사진이야. 얼굴 아래쪽은 잘려 있을 수 있어.
아래 지점 중 사진에 **또렷하게 보이는 것만** 골라서 위치를 찍어줘. 안 보이거나 확실하지
않으면 그 항목은 빼. 없는 걸 지어내지 마.
왼쪽·오른쪽은 **보는 사람 기준(화면 기준)**이야. 눈썹이 둘 다 보이면 넷을 다 찍어주고,
하나만 보이면 그게 화면 어느 쪽에 있는지에 따라 left_ 또는 right_ 두 개만 찍어줘.
머리카락과 헤어라인은 절대 쓰지 마 — 수술로 바뀌는 부위라 기준이 될 수 없어.
얼굴이 여러 개 보이면 가장 크게 나온 얼굴 하나만 써.

{items}

반드시 JSON 배열만 출력해:
[{{"label":"brow_inner","point":[y,x]}}, ...]
point는 [y, x] 순서, 0~1000으로 정규화한 값이야.""".format(
    items="\n".join(f"- {name}: {desc}" for name, desc in LANDMARKS)
)


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


def _ask_once(encoded: str, key: str) -> dict[str, tuple[float, float]] | None:
    """보이는 기준점들을 {이름: (x, y)} 형태로 돌려준다. 좌표는 0~1 비율. 실패하면 None."""
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

    known = {name for name, _ in LANDMARKS}
    found: dict[str, tuple[float, float]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        label, point = item.get("label"), item.get("point")
        if label not in known or not point or len(point) != 2:
            continue
        try:
            y, x = float(point[0]) / 1000, float(point[1]) / 1000
        except (TypeError, ValueError):
            continue
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            found[label] = (x, y)
    return found or None


def suggest_landmarks(image_bytes: bytes) -> dict[str, tuple[float, float]]:
    """사진에서 믿을 만한 기준점들을 찾는다. 하나도 없으면 빈 딕셔너리.

    여러 번 물어 지점별로 중앙값을 쓴다. 이 모델은 추론 모델이라 temperature를 0으로 둬도
    답이 매번 같지는 않은데, 실측해보니 대부분은 아주 촘촘하고(표준편차 2~7픽셀) 가끔 한
    번씩 엉뚱한 곳을 찍는다 — 8번 중 1번이 163에서 32로 튀는 식이다. 평균이나 두 번의
    일치 검사는 그 한 번에 통째로 휘둘리지만, 중앙값은 영향을 받지 않는다.

    지점마다 따로 판정한다. 눈썹은 또렷한데 귀는 흐릿한 사진이 흔해서, 하나가 흔들린다고
    나머지까지 버릴 이유가 없다. 반대로 흔들리는 지점을 억지로 끼워 넣으면 오히려
    부정확해지므로(점 개수 실측에서 확인) 과반이 모이지 않은 지점은 뺀다."""
    key = _api_key()
    if not key:
        return {}

    encoded = _encode(image_bytes)
    # 질문들은 서로 독립이라 동시에 보낸다. 순서대로 보내면 횟수만큼 기다려야 한다.
    # 충분히 모이면 남은 것은 기다리지 않고 넘어간다.
    pool = ThreadPoolExecutor(max_workers=SAMPLE_COUNT)
    futures = [pool.submit(_ask_once, encoded, key) for _ in range(SAMPLE_COUNT)]
    answers: list[dict[str, tuple[float, float]]] = []
    try:
        for future in as_completed(futures):
            answer = future.result()
            if answer:
                answers.append(answer)
            if len(answers) >= ENOUGH_ANSWERS:
                break
    finally:
        for future in futures:
            future.cancel()
        # 이미 떠난 요청은 백그라운드에서 끝나게 두고 기다리지 않는다.
        pool.shutdown(wait=False)

    if len(answers) < MIN_ANSWERS:
        logger.info("suggest: 응답 %d개뿐이라 제안하지 않음", len(answers))
        return {}

    needed = _needed(len(answers))
    result: dict[str, tuple[float, float]] = {}
    for name, _ in LANDMARKS:
        seen = [answer[name] for answer in answers if name in answer]
        if len(seen) < needed:
            continue
        median = (
            statistics.median(x for x, _ in seen),
            statistics.median(y for _, y in seen),
        )
        agreeing = sum(
            1 for x, y in seen if max(abs(x - median[0]), abs(y - median[1])) <= AGREEMENT_LIMIT
        )
        if agreeing >= needed:
            result[name] = median
        else:
            logger.info("suggest: %s는 %d/%d만 일치해 제외", name, agreeing, len(seen))

    logger.info("suggest: %s (응답 %d회)", list(result) or "찾은 지점 없음", len(answers))
    return result


def can_level(labels: list[str]) -> bool:
    """찾은 지점들로 사진을 똑바로 세울 수 있는지.

    양쪽 눈썹이 다 있을 때만 가능하다. 두 눈썹은 좌우 같은 높이라 그 선이 곧 수평이지만,
    눈썹 하나의 기울기는 고개가 기울어서인지 옆으로 돌아서인지 구분할 수 없다. 실제로
    측면 사진에서 한쪽 눈썹만 보고 수평을 맞췄더니 사진이 돌아가면서 흰 여백이 생기고,
    그걸 잘라내느라 눈썹이 통째로 프레임 밖으로 나갔다."""
    return any(l.startswith("left_brow") for l in labels) and any(
        l.startswith("right_brow") for l in labels
    )


def matched_points(
    images: list[bytes],
) -> tuple[list[list[tuple[float, float]]], list[str]] | None:
    """모든 사진(전·중간·후)에서 공통으로 찾은 기준점만 짝지어 돌려준다.

    돌려주는 것: (사진 순서대로 각 사진의 점 목록, 지점 이름)

    한쪽에만 있는 지점은 대응이 성립하지 않아 쓸 수 없다. 이마 클로즈업에서는 눈썹 양 끝만
    남는 경우가 대부분이지만, 귀처럼 더 멀리 있는 지점이 모두에 보이면 기준선이 그만큼
    길어져 배율·각도가 정확해진다 — 간격을 2배로 벌리면 오차가 절반이 된다."""
    with ThreadPoolExecutor(max_workers=len(images)) as pool:
        found = list(pool.map(suggest_landmarks, images))

    common = [name for name, _ in LANDMARKS if all(name in landmarks for landmarks in found)]
    if len(common) < 2:
        logger.info("suggest: 공통 지점이 %d개뿐이라 제안하지 않음", len(common))
        return None
    logger.info("suggest: 공통 기준점 %s", common)
    return [[landmarks[name] for name in common] for landmarks in found], common
