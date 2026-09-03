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

interface ResultSet {
  results: Results;
  urls: { before: string; after: string };
}

function toResultSet(results: Results): ResultSet {
  return {
    results,
    urls: {
      before: URL.createObjectURL(results.before),
      after: URL.createObjectURL(results.after),
    },
  };
}

function revokeResultSet(set: ResultSet | null) {
  if (!set) return;
  URL.revokeObjectURL(set.urls.before);
  URL.revokeObjectURL(set.urls.after);
}

// 상대경로로 호출한다. 배포 환경에서는 백엔드가 이 프론트엔드를 같이 서빙해서 같은
// 오리진이고, 개발 중에는 vite.config.ts의 프록시가 백엔드로 넘겨준다.
// (예전엔 "http://localhost:8000"을 박아뒀는데, 그러면 다른 PC에서 열었을 때
// localhost가 그 PC 자신을 가리켜서 무조건 실패한다.)
const API_BASE_URL = "";

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
  const [baseResult, setBaseResult] = useState<ResultSet | null>(null);
  const [mosaicResult, setMosaicResult] = useState<ResultSet | null>(null);
  const [showingMosaic, setShowingMosaic] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const displayed = showingMosaic && mosaicResult ? mosaicResult : baseResult;

  function resetResults() {
    revokeResultSet(baseResult);
    revokeResultSet(mosaicResult);
    setBaseResult(null);
    setMosaicResult(null);
    setShowingMosaic(false);
    setMosaicStatus("idle");
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
    setError(null);
    try {
      const processed = await processPair(beforePhoto, afterPhoto, false);
      revokeResultSet(baseResult);
      revokeResultSet(mosaicResult);
      setBaseResult(toResultSet(processed));
      setMosaicResult(null);
      setShowingMosaic(false);
      setMosaicStatus("idle");
      setStatus("done");
    } catch {
      setError("변환 중 문제가 발생했습니다. 다시 시도해주세요.");
      setStatus("idle");
    }
  }

  async function handleApplyMosaic() {
    if (!beforePhoto || !afterPhoto) return;
    // 이미 한 번 계산해둔 모자이크 결과가 있으면 다시 요청하지 않고 바로 보여준다.
    if (mosaicResult) {
      setShowingMosaic(true);
      return;
    }
    setMosaicStatus("processing");
    setError(null);
    try {
      const processed = await processPair(beforePhoto, afterPhoto, true);
      setMosaicResult(toResultSet(processed));
      setShowingMosaic(true);
      setMosaicStatus("done");
    } catch {
      setError("모자이크 처리 중 문제가 발생했습니다. 다시 시도해주세요.");
      setMosaicStatus("idle");
    }
  }

  function handleCancelMosaic() {
    setShowingMosaic(false);
  }

  function handleDownload() {
    if (!displayed || !beforePhoto || !afterPhoto) return;
    downloadBlob(displayed.results.before, withJpgName(beforePhoto.name, "before"));
    downloadBlob(displayed.results.after, withJpgName(afterPhoto.name, "after"));
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
          resultUrl={displayed?.urls.before}
        />
        <PhotoDropzone
          label="후 (After)"
          file={afterPhoto}
          onChange={handleAfterChange}
          resultUrl={displayed?.urls.after}
        />
      </section>

      {displayed && displayed.results.warnings.length > 0 && (
        <div className="warning-banner">
          {displayed.results.warnings.map((warning) => (
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
            {showingMosaic ? (
              <button type="button" className="submit-btn submit-btn--secondary" onClick={handleCancelMosaic}>
                모자이크 취소
              </button>
            ) : (
              <button
                type="button"
                className="submit-btn submit-btn--secondary"
                disabled={mosaicStatus === "processing"}
                onClick={handleApplyMosaic}
              >
                {mosaicStatus === "processing" ? "모자이크 처리 중..." : "모자이크"}
              </button>
            )}
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
