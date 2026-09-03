# 1단계: 프론트엔드 빌드
FROM node:22-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# 2단계: 백엔드 + 빌드된 프론트엔드
FROM python:3.12-slim
WORKDIR /app

# opencv/mediapipe가 요구하는 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# 얼굴 랜드마크 모델은 용량 때문에 git에 없으므로 빌드 때 받는다 (backend/models/README.md 참고)
RUN mkdir -p backend/models && curl -fsSL -o backend/models/face_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

COPY backend/ ./backend/
COPY --from=frontend /build/dist ./frontend/dist

# 배포 플랫폼이 지정하는 포트를 따른다 (없으면 8000)
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT}"]
