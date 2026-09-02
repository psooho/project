import { useState } from "react";
import "./App.css";
import { PhotoDropzone } from "./components/PhotoDropzone";

type Status = "idle" | "processing" | "done";
type MosaicStatus = "idle" | "processing" | "done";

interface Results {
  before: Blob;
  after: Blob;
  warnings: string[];
}

const API_BASE_URL = "http://localhost:8000";

// 실제 정렬(각도·크기)/밝기·색감 보정/헤어라인·눈썹만 남기는 모자이크 처리는
// 백엔드 파이프라인(backend/app/main.py의 process_pair, PRD 6.2~6.5)에서 이뤄진다.
async function processPair(before: File, after: File, mosaic: boolean): Promise<Results> {
  const formData = new FormData();
  formData.append("before", before);
  formData.append("after", after);
  formData.append("mosaic", String(mosaic));

  const res = await fetch(`${API_BASE_URL}/api/process`, { method: "POST", body: formData });
  if (!res.ok) throw new Error("서버 처리에 실패했습니다.");

  const data: { before: string; after: string; warnings?: string[] } = await res.json();
  const [beforeBlob, afterBlob] = await Promise.all([
    fetch(data.before).then((r) => r.blob()),
    fetch(data.after).then((r) => r.blob()),
  ]);
  return { before: beforeBlob, after: afterBlob, warnings: data.warnings ?? [] };
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
  const [mosaicStatus, setMosaicStatus] = useState<MosaicStatus>("idle");
  const [results, setResults] = useState<Results | null>(null);
  const [resultUrls, setResultUrls] = useState<{ before: string; after: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  function resetResults() {
    if (resultUrls) {
      URL.revokeObjectURL(resultUrls.before);
      URL.revokeObjectURL(resultUrls.after);
    }
    setResultUrls(null);
    setResults(null);
    setMosaicStatus("idle");
  }

  function applyResults(processed: Results) {
    setResults(processed);
    setResultUrls((prev) => {
      if (prev) {
        URL.revokeObjectURL(prev.before);
        URL.revokeObjectURL(prev.after);
      }
      return {
        before: URL.createObjectURL(processed.before),
        after: URL.createObjectURL(processed.after),
      };
    });
  }

  function handleBeforeChange(file: File | null) {
    setBeforePhoto(file);
    setStatus("idle");
    resetResults();
    setError(null);
  }

  function handleAfterChange(file: File | null) {
    setAfterPhoto(file);
    setStatus("idle");
    resetResults();
    setError(null);
  }

  async function handleTransform() {
    if (!beforePhoto || !afterPhoto) return;
    setStatus("processing");
    setMosaicStatus("idle");
    setError(null);
    try {
      const processed = await processPair(beforePhoto, afterPhoto, false);
      applyResults(processed);
      setStatus("done");
    } catch {
      setError("변환 중 문제가 발생했습니다. 다시 시도해주세요.");
      setStatus("idle");
    }
  }

  async function handleApplyMosaic() {
    if (!beforePhoto || !afterPhoto || mosaicStatus !== "idle") return;
    setMosaicStatus("processing");
    setError(null);
    try {
      const processed = await processPair(beforePhoto, afterPhoto, true);
      applyResults(processed);
      setMosaicStatus("done");
    } catch {
      setError("모자이크 처리 중 문제가 발생했습니다. 다시 시도해주세요.");
      setMosaicStatus("idle");
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
        <PhotoDropzone
          label="전 (Before)"
          file={beforePhoto}
          onChange={handleBeforeChange}
          resultUrl={resultUrls?.before}
        />
        <PhotoDropzone
          label="후 (After)"
          file={afterPhoto}
          onChange={handleAfterChange}
          resultUrl={resultUrls?.after}
        />
      </section>

      {results && results.warnings.length > 0 && (
        <div className="warning-banner">
          {results.warnings.map((warning) => (
            <p key={warning}>⚠ {warning}</p>
          ))}
        </div>
      )}

      <div className="actions">
        {status === "done" ? (
          <>
            <button type="button" className="submit-btn" onClick={handleDownload}>
              다운로드
            </button>
            <button
              type="button"
              className="submit-btn submit-btn--secondary"
              disabled={mosaicStatus !== "idle"}
              onClick={handleApplyMosaic}
            >
              {mosaicStatus === "processing"
                ? "모자이크 처리 중..."
                : mosaicStatus === "done"
                  ? "모자이크 적용됨"
                  : "모자이크"}
            </button>
          </>
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
