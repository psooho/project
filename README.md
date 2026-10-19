# 전후사진 변환

모발이식 전/후 사진을 같은 크기·각도·밝기로 맞춰주는 웹앱. 얼굴 모자이크도 넣을 수 있다.

- 운영 주소: https://photo-yvt3rcfwla-du.a.run.app (병원 공용 암호 필요)
- 무엇을 왜 만드는지는 [PRD.md](PRD.md) 참고

## 다른 컴퓨터에서 이어서 작업하기

git에 들어 있지 않은 것이 네 가지 있다 — 아래 순서대로 채우면 된다.

### 1. 받기

```bash
git clone https://github.com/psooho/project.git
```

필요한 것: Python 3.12, Node 22 이상, (배포하려면) gcloud CLI.

### 2. 백엔드 준비

```bash
python -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt
```

맥·리눅스라면 `backend/.venv/bin/python`이다.

### 3. 얼굴 검출 모델 내려받기

용량 때문에 git에 없다. `backend/models/` 안에 받는다.

```bash
curl -L -o backend/models/face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
```

### 4. 프론트엔드 준비

```bash
npm --prefix frontend ci
```

### 5. 환경변수

저장소 루트에 `.env`를 만든다. 값은 운영 서버에 들어 있는 것을 그대로 쓰면 된다.

```bash
gcloud run services describe photo --region asia-northeast3 --format="value(spec.template.spec.containers[0].env)"
```

```
GEMINI_API_KEY=<위에서 확인한 값>
```

`APP_PASSWORD`는 **로컬에서는 넣지 않는다.** 비어 있으면 암호 없이 바로 쓸 수 있고, 넣으면 로그인 화면이 뜬다.

`GEMINI_API_KEY`가 없어도 앱은 돈다 — 눈·코·입이 안 보이는 클로즈업에서 기준점을 자동으로 찾아주는 기능만 꺼진다.

## 실행

```bash
backend/.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir backend --port 8000
```

백엔드만 띄우면 `frontend/dist`를 같이 서빙한다 (배포와 같은 구성). 한 번은 `npm --prefix frontend run build`를 해둬야 한다.

화면을 고치는 중이라면 프론트엔드를 따로 띄우는 쪽이 편하다. `/api`는 8000번으로 넘어간다.

```bash
npm --prefix frontend run dev
```

## 테스트 사진

저장소에 없다. 환자 사진이라 넣지 않는다. 손에 있는 사진으로 확인하면 되고, 이런 경우를 꼭 섞어서 본다.

- **정면**과 **측면** 한 쌍씩 — 결과 크기가 각각 1134×1008, 1204×910으로 일정해야 한다
- **이마·헤어라인만 크게 찍은 클로즈업** — 눈·코·입이 없어 얼굴 검출이 원리적으로 안 된다. 기준점을 직접 찍거나 Gemini가 찾아주는 경로로 빠진다
- **휴대폰 원본 크기(12MP 안팎)** — 느려지는 문제가 여기서만 드러난다

## 배포

```bash
gcloud run deploy photo --source . --region asia-northeast3 --concurrency 2
```

- **`--concurrency 2`를 빼지 말 것.** 요청 하나가 최대 약 266MB를 쓰고 인스턴스는 1Gi라, 기본값(80)이면 동시에 몇 장만 들어와도 메모리 초과로 인스턴스가 죽는다. 2면 사용량이 한도의 65%에 머문다.
- **`--set-env-vars`를 쓰지 말 것.** 기존 환경변수를 전부 덮어써서 `APP_PASSWORD`와 `GEMINI_API_KEY`가 날아간다. 바꿔야 하면 `--update-env-vars`.
- 배포 후에는 리비전이 바뀌었는지, 기동 로그에 에러가 없는지 본다.

```bash
gcloud run services describe photo --region asia-northeast3 --format="value(status.latestReadyRevisionName,spec.template.spec.containerConcurrency)"
gcloud logging read 'resource.labels.revision_name="<리비전>" AND severity>=ERROR' --freshness=30m
```

## 운영 중인 것

| 항목 | 값 |
|---|---|
| 서비스 | Cloud Run `photo` (프로젝트 `banalps`, asia-northeast3) |
| 메모리 / 동시 처리 | 1Gi / 2 |
| 환경변수 | `APP_PASSWORD`, `GEMINI_API_KEY` |
| 알림 | 서버 오류(5xx)·메모리 초과 시 psooho@gmail.com으로 메일 |

## 손대기 전에 알아두면 좋은 것

- **결과를 눈으로 확인한다.** 이 앱은 기하 계산이 머리로는 맞아 보여도 결과 사진이 틀린 경우가 반복해서 나왔다. 코드를 고쳤으면 사진을 열어보거나 픽셀을 재서 확인한다.
- **자동 모드와 직접 지정 모드는 쓰는 변수가 다르다.** 한쪽에만 있는 값을 조건 없이 참조하면 다른 쪽에서 죽는다. 실제로 크기 슬라이더가 이것 때문에 500 에러를 냈다.
- **전/후 결과는 항상 같은 픽셀 크기여야 한다.**
- **윈도우 콘솔이 cp949라** 테스트 스크립트에서 한글을 찍으면 깨진다. 숫자나 영문으로 찍는 게 편하다.
- **`cv2.imwrite`는 비ASCII 경로에서 조용히 실패한다.** 한글 파일명으로 저장하지 말 것.
- 변경분을 검토하는 서브에이전트가 있다 — `.claude/agents/change-reviewer.md`.
