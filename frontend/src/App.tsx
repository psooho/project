import { useState } from "react";
import "./App.css";
import { PhotoDropzone } from "./components/PhotoDropzone";

type Status = "idle" | "processing" | "done";

interface Results {
  before: Blob;
  after: Blob;
}

function loadImage(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

// TODO: 이 파이프라인에 얼굴 랜드마크 기반 정렬(각도·크기), 밝기/색감 보정, 헤어라인·눈썹만
// 남기는 모자이크 처리를 추가한다 (PRD 6.2~6.5). 지금은 원본을 그대로 캔버스에 그려 내보낸다.
async function processPhoto(file: File): Promise<Blob> {
  const img = await loadImage(file);
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("캔버스를 생성할 수 없습니다.");
  ctx.drawImage(img, 0, 0);
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("이미지 변환에 실패했습니다."))),
      "image/jpeg",
      0.92,
    );
  });
}

function withJpgName(originalName: string, prefix: string) {
  const base = originalName.replace(/\.[^./\\]+$/, "");
  return `${prefix}_${base}.jpg`;
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function App() {
  const [beforePhoto, setBeforePhoto] = useState<File | null>(null);
  const [afterPhoto, setAfterPhoto] = useState<File | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [results, setResults] = useState<Results | null>(null);
  const [error, setError] = useState<string | null>(null);

  function handleBeforeChange(file: File | null) {
    setBeforePhoto(file);
    setStatus("idle");
    setResults(null);
    setError(null);
  }

  function handleAfterChange(file: File | null) {
    setAfterPhoto(file);
    setStatus("idle");
    setResults(null);
    setError(null);
  }

  async function handleTransform() {
    if (!beforePhoto || !afterPhoto) return;
    setStatus("processing");
    setError(null);
    try {
      const [before, after] = await Promise.all([processPhoto(beforePhoto), processPhoto(afterPhoto)]);
      setResults({ before, after });
      setStatus("done");
    } catch {
      setError("변환 중 문제가 발생했습니다. 다시 시도해주세요.");
      setStatus("idle");
    }
  }

  function handleDownload() {
    if (!results || !beforePhoto || !afterPhoto) return;
    downloadBlob(results.before, withJpgName(beforePhoto.name, "before"));
    downloadBlob(results.after, withJpgName(afterPhoto.name, "after"));
  }

  const canTransform = Boolean(beforePhoto && afterPhoto) && status === "idle";

  return (
    <div className="page">
      <header className="page-header">
        <h1>전후사진 변환</h1>
        <p>전/후 사진을 넣고 변환한 뒤 다운로드하세요.</p>
      </header>

      <section className="photo-grid">
        <PhotoDropzone label="전 (Before)" file={beforePhoto} onChange={handleBeforeChange} />
        <PhotoDropzone label="후 (After)" file={afterPhoto} onChange={handleAfterChange} />
      </section>

      <div className="actions">
        {status === "done" ? (
          <button type="button" className="submit-btn" onClick={handleDownload}>
            다운로드
          </button>
        ) : (
          <button type="button" className="submit-btn" disabled={!canTransform} onClick={handleTransform}>
            {status === "processing" ? "처리 중..." : "사진 변환"}
          </button>
        )}
        {error && <span className="error-note">{error}</span>}
      </div>
    </div>
  );
}

export default App;
