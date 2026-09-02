# 모델 파일

`face_landmarker.task`는 용량(약 3.7MB) 때문에 git에 커밋하지 않는다. 아래 명령으로 받는다.

```bash
curl -L -o face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
```

이 폴더(`backend/models/`)에 저장하면 된다. 한 번 받아두면 이후 서버 실행 시 외부 통신 없이 오프라인으로 동작한다.
