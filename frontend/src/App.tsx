import { useState } from "react";
import "./App.css";
import { clearStoredPassword, getStoredPassword } from "./components/PasswordGate";
import { PhotoDropzone } from "./components/PhotoDropzone";
import { PointPicker } from "./components/PointPicker";
import type { NormalizedPoint } from "./components/PointPicker";

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
type ManualPoints = { before: NormalizedPoint[]; after: NormalizedPoint[] };

/** 백엔드가 "얼굴을 못 찾았다"고 알려주는 경고인지. 이때만 직접 지정 화면을 권한다. */
function isFaceMissing(warnings: string[]) {
  return warnings.some((warning) => warning.includes("얼굴을 찾지 못해"));
}

async function processPair(
  before: File,
  after: File,
  mosaic: boolean,
  manual?: ManualPoints,
): Promise<Results> {
  const formData = new FormData();
  formData.append("before", before);
  formData.append("after", after);
  formData.append("mosaic", String(mosaic));
  if (manual) {
    formData.append(
      "points",
      JSON.stringify({
        before: manual.before.map((p) => [p.x, p.y]),
        after: manual.after.map((p) => [p.x, p.y]),
      }),
    );
  }

  const res = await fetch(`${API_BASE_URL}/api/process`, {
    method: "POST",
    headers: { "X-App-Password": getStoredPassword() },
    body: formData,
  });
  if (res.status === 401) {
    // 암호가 바뀐 경우. 저장된 암호를 지우고 새로고침하면 다시 입력 화면이 뜬다.
    clearStoredPassword();
    window.location.reload();
    throw new Error("암호가 변경되었습니다.");
  }
  if (!res.ok) throw new Error("서버 처리에 실패했습니다.");

  const data: { before: string; after: string; warnings?: string[] } = await res.json();
  const [beforeBlob, afterBlob] = await Promise.all([
    fetch(data.before).then((r) => r.blob()),
    fetch(data.after).then((r) => r.blob()),
  ]);
  return { before: beforeBlob, after: afterBlob, warnings: data.warnings ?? [] };
}

/** 얼굴 검출이 안 되는 사진에서 눈썹 양 끝 위치를 제안받는다.
 *  실패해도 오류로 다루지 않는다 — 제안이 없으면 사용자가 직접 찍으면 된다. */
async function fetchSuggestedPoints(before: File, after: File) {
  const formData = new FormData();
  formData.append("before", before);
  formData.append("after", after);
  const res = await fetch(`${API_BASE_URL}/api/suggest-points`, {
    method: "POST",
    headers: { "X-App-Password": getStoredPassword() },
    body: formData,
  });
  if (!res.ok) return null;
  const data: {
    available: boolean;
    before: [number, number][] | null;
    after: [number, number][] | null;
  } = await res.json();
  if (!data.available || !data.before || !data.after) return null;
  const toPoints = (pairs: [number, number][]) => pairs.map(([x, y]) => ({ x, y }));
  return { before: toPoints(data.before), after: toPoints(data.after) };
}

/** 이미 변환이 끝난 두 장을 보내서 전→후로 스르륵 넘어가는 GIF를 받아온다. */
async function buildGif(before: Blob, after: Blob): Promise<Blob> {
  const formData = new FormData();
  formData.append("before", before, "before.jpg");
  formData.append("after", after, "after.jpg");

  const res = await fetch(`${API_BASE_URL}/api/gif`, {
    method: "POST",
    headers: { "X-App-Password": getStoredPassword() },
    body: formData,
  });
  if (res.status === 401) {
    clearStoredPassword();
    window.location.reload();
    throw new Error("암호가 변경되었습니다.");
  }
  if (!res.ok) throw new Error("GIF 생성에 실패했습니다.");
  return res.blob();
}

function withJpgName(originalName: string, prefix: string) {
  const base = originalName.replace(/\.[^./\\]+$/, "");
  return `${prefix}_${base}.jpg`;
}

function withGifName(originalName: string) {
  const base = originalName.replace(/\.[^./\\]+$/, "");
  return `${base}_전후.gif`;
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
  const [gifBusy, setGifBusy] = useState(false);
  const [gifResult, setGifResult] = useState<{ blob: Blob; url: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  // 얼굴 검출이 실패한 사진(이마·헤어라인 클로즈업 등)에서 기준점을 직접 찍는 모드
  const [picking, setPicking] = useState(false);
  const [beforePoints, setBeforePoints] = useState<NormalizedPoint[]>([]);
  const [afterPoints, setAfterPoints] = useState<NormalizedPoint[]>([]);
  const [usedManual, setUsedManual] = useState(false);
  // 직접 지정으로 변환하기 직전의 결과. "변환 취소"로 되돌릴 수 있게 들고 있는다.
  const [preManualResult, setPreManualResult] = useState<ResultSet | null>(null);
  const [suggesting, setSuggesting] = useState(false);
  const [suggestFailed, setSuggestFailed] = useState(false);
  // 눈썹을 자동으로 찾아 맞춘 결과인지. 사용자가 그 사실을 알아야 검토할 수 있다.
  const [autoAligned, setAutoAligned] = useState(false);

  const displayed = showingMosaic && mosaicResult ? mosaicResult : baseResult;

  /** 만들어 둔 GIF는 지금 화면에 보이는 사진 기준이라, 사진이나 모자이크 상태가
   *  바뀌면 더 이상 맞지 않는다. 그럴 때 지워서 다시 만들게 한다. */
  function clearGif() {
    setGifResult((prev) => {
      if (prev) URL.revokeObjectURL(prev.url);
      return null;
    });
  }

  function resetResults() {
    revokeResultSet(baseResult);
    revokeResultSet(mosaicResult);
    revokeResultSet(preManualResult);
    setBaseResult(null);
    setMosaicResult(null);
    setPreManualResult(null);
    setShowingMosaic(false);
    setMosaicStatus("idle");
    setUsedManual(false);
    setAutoAligned(false);
    clearGif();
  }

  function resetPicking() {
    setPicking(false);
    setBeforePoints([]);
    setAfterPoints([]);
    setSuggestFailed(false);
  }

  /** 찍기 화면을 열면서 눈썹 위치를 미리 채워둔다. 사용자는 확인하고 고치기만 하면 된다. */
  async function handleStartPicking() {
    setPicking(true);
    setSuggestFailed(false);
    if (!beforePhoto || !afterPhoto) return;
    // 이미 찍어둔 점이 있으면(변환 취소로 돌아온 경우) 덮어쓰지 않는다.
    if (beforePoints.length || afterPoints.length) return;
    setSuggesting(true);
    try {
      const suggested = await fetchSuggestedPoints(beforePhoto, afterPhoto);
      if (suggested) {
        setBeforePoints(suggested.before);
        setAfterPoints(suggested.after);
      } else {
        setSuggestFailed(true);
      }
    } catch {
      setSuggestFailed(true);
    } finally {
      setSuggesting(false);
    }
  }

  function handleBeforeChange(file: File | null) {
    setBeforePhoto(file);
    setStatus("idle");
    resetResults();
    resetPicking();
    setError(null);
  }

  function handleAfterChange(file: File | null) {
    setAfterPhoto(file);
    setStatus("idle");
    resetResults();
    resetPicking();
    setError(null);
  }

  async function handleTransform(manual?: ManualPoints) {
    if (!beforePhoto || !afterPhoto) return;
    setStatus("processing");
    setError(null);
    setSuggestFailed(false);
    try {
      const first = await processPair(beforePhoto, afterPhoto, false, manual);

      // 얼굴을 못 찾았으면 눈썹 위치를 자동으로 받아 한 번 더 맞춰본다. 사용자가 점을
      // 찍지 않아도 변환 한 번으로 끝나고, 결과가 어긋나면 "변환 취소"로 고칠 수 있다.
      let auto: { points: ManualPoints; result: Results } | null = null;
      if (!manual && isFaceMissing(first.warnings)) {
        setSuggesting(true);
        try {
          const suggested = await fetchSuggestedPoints(beforePhoto, afterPhoto);
          if (suggested) {
            auto = {
              points: suggested,
              result: await processPair(beforePhoto, afterPhoto, false, suggested),
            };
          } else {
            setSuggestFailed(true);
          }
        } catch {
          setSuggestFailed(true);
        } finally {
          setSuggesting(false);
        }
      }

      const shown = auto ? auto.result : first;
      const points = manual ?? auto?.points;

      revokeResultSet(mosaicResult);
      if (auto) {
        // 되돌리면 방금의 "얼굴 못 찾음" 상태로 돌아간다.
        revokeResultSet(baseResult);
        revokeResultSet(preManualResult);
        setPreManualResult(toResultSet(first));
        setBeforePoints(auto.points.before);
        setAfterPoints(auto.points.after);
      } else if (manual) {
        // 되돌릴 수 있도록 직전 결과는 해제하지 않고 보관한다.
        revokeResultSet(preManualResult);
        setPreManualResult(baseResult);
      } else {
        revokeResultSet(baseResult);
        revokeResultSet(preManualResult);
        setPreManualResult(null);
      }

      setBaseResult(toResultSet(shown));
      setMosaicResult(null);
      setShowingMosaic(false);
      setMosaicStatus("idle");
      setUsedManual(Boolean(points));
      setAutoAligned(Boolean(auto));
      clearGif();
      setStatus("done");
      if (points) setPicking(false);
      else if (!isFaceMissing(shown.warnings)) resetPicking();
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
      clearGif();
      return;
    }
    setMosaicStatus("processing");
    setError(null);
    try {
      const processed = await processPair(beforePhoto, afterPhoto, true);
      setMosaicResult(toResultSet(processed));
      setShowingMosaic(true);
      clearGif();
      setMosaicStatus("done");
    } catch {
      setError("모자이크 처리 중 문제가 발생했습니다. 다시 시도해주세요.");
      setMosaicStatus("idle");
    }
  }

  /** 직접 지정 변환을 물리고, 찍어둔 점은 그대로 둔 채 다시 찍기 화면으로 돌아간다. */
  function handleUndoManual() {
    if (!preManualResult) return;
    revokeResultSet(baseResult);
    revokeResultSet(mosaicResult);
    setBaseResult(preManualResult);
    setPreManualResult(null);
    setMosaicResult(null);
    setShowingMosaic(false);
    setMosaicStatus("idle");
    setUsedManual(false);
    setAutoAligned(false);
    clearGif();
    setPicking(true);
  }

  function handleCancelMosaic() {
    setShowingMosaic(false);
    clearGif();
  }

  function handleDownload() {
    if (!displayed || !beforePhoto || !afterPhoto) return;
    downloadBlob(displayed.results.before, withJpgName(beforePhoto.name, "before"));
    downloadBlob(displayed.results.after, withJpgName(afterPhoto.name, "after"));
  }

  async function handleMakeGif() {
    if (!displayed || gifBusy) return;
    setGifBusy(true);
    setError(null);
    try {
      // 화면에 보이는 버전을 그대로 쓴다 — 모자이크를 적용해 뒀다면 GIF도 모자이크본이 된다.
      const blob = await buildGif(displayed.results.before, displayed.results.after);
      clearGif();
      setGifResult({ blob, url: URL.createObjectURL(blob) });
    } catch {
      setError("GIF 만들기에 실패했습니다. 다시 시도해주세요.");
    } finally {
      setGifBusy(false);
    }
  }

  function handleDownloadGif() {
    if (!gifResult || !beforePhoto) return;
    downloadBlob(gifResult.blob, withGifName(beforePhoto.name));
  }

  const canTransform = Boolean(beforePhoto && afterPhoto) && status === "idle";
  const faceMissing = Boolean(displayed && isFaceMissing(displayed.results.warnings));
  // 찍은 순서가 전/후의 대응 관계라 개수가 같아야 한다.
  const pointsReady =
    beforePoints.length >= 2 && beforePoints.length === afterPoints.length;
  const pointCountMismatch =
    beforePoints.length >= 2 && afterPoints.length >= 2 && beforePoints.length !== afterPoints.length;

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

      {gifResult && (
        <section className="gif-preview">
          <div className="gif-preview-label">GIF 미리보기</div>
          <img src={gifResult.url} alt="전후 전환 GIF 미리보기" />
          <p className="gif-preview-note">
            실제로 저장되는 파일입니다 ({Math.round(gifResult.blob.size / 1024).toLocaleString()}KB)
          </p>
        </section>
      )}

      {autoAligned && !picking && (
        <div className="auto-banner">
          <p>
            얼굴을 찾지 못해 <strong>눈썹 위치를 자동으로 찾아</strong> 크기와 각도를 맞췄습니다.
            결과를 확인해주세요 — 어긋났으면 <strong>변환 취소</strong>를 눌러 직접 고칠 수 있습니다.
          </p>
        </div>
      )}

      {displayed && displayed.results.warnings.length > 0 && (
        <div className="warning-banner">
          {displayed.results.warnings.map((warning) => (
            <p key={warning}>⚠ {warning}</p>
          ))}
          {faceMissing && !picking && (
            <p className="warning-action">
              이마·헤어라인만 크게 찍은 사진은 눈·코·입이 없어 얼굴을 자동으로 찾을 수 없습니다.
              기준점을 직접 찍으면 맞출 수 있습니다.{" "}
              <button type="button" className="link-btn" onClick={handleStartPicking}>
                직접 찍기
              </button>
            </p>
          )}
        </div>
      )}

      {picking && beforePhoto && afterPhoto && (
        <section className="picker-section">
          <h2 className="picker-title">기준점 지정</h2>
          {suggesting && <p className="picker-notice">눈썹 위치를 찾는 중입니다...</p>}
          {!suggesting && !suggestFailed && beforePoints.length > 0 && (
            <p className="picker-notice picker-notice--ok">
              눈썹 양 끝을 자동으로 찍어뒀습니다. <strong>맞는지 확인하고</strong> 어긋났으면
              점을 눌러 지운 뒤 다시 찍어주세요.
            </p>
          )}
          {suggestFailed && (
            <p className="picker-notice">
              눈썹 위치를 자동으로 찾지 못했습니다. 직접 찍어주세요.
            </p>
          )}
          <p className="picker-guide">
            두 사진에서 <strong>같은 자리</strong>를 <strong>같은 순서로</strong> 클릭해주세요.
            최소 2곳이면 되고, <strong>최대한 멀리 떨어진</strong> 지점을 고르는 게 가장 효과가
            큽니다 — 간격을 2배로 벌리면 오차가 절반으로 줄어듭니다. 가로·세로·대각선 아무
            방향이나 괜찮습니다. 사진은 원래 방향 그대로 두고, 후 사진만 전 사진에 맞춰 돌립니다.
          </p>
          <p className="picker-guide">
            뚜렷하게 알아볼 수 있는 지점이 더 있으면 3~4곳까지 찍어도 좋습니다. 그러면 서로
            어긋난 곳이 있는지도 검사해드립니다. 다만 <strong>애매한 자리를 억지로 늘리면 오히려
            부정확</strong>해지니, 확실한 곳만 찍으세요. 찍은 점을 다시 누르면 그 점만 지워집니다.
          </p>
          <div className="picker-grid">
            <PointPicker label="전 (Before)" file={beforePhoto} points={beforePoints} onChange={setBeforePoints} />
            <PointPicker label="후 (After)" file={afterPhoto} points={afterPoints} onChange={setAfterPoints} />
          </div>
          <div className="actions">
            {pointCountMismatch && (
              <span className="error-note">
                전 {beforePoints.length}곳, 후 {afterPoints.length}곳 — 개수를 같게 맞춰주세요.
              </span>
            )}
            <button
              type="button"
              className="submit-btn"
              disabled={!pointsReady || status === "processing"}
              onClick={() => handleTransform({ before: beforePoints, after: afterPoints })}
            >
              {status === "processing" ? "처리 중..." : "이 지점으로 변환"}
            </button>
            <button type="button" className="submit-btn submit-btn--secondary" onClick={resetPicking}>
              취소
            </button>
          </div>
        </section>
      )}

      <div className="actions">
        {status === "done" ? (
          <>
            <button type="button" className="submit-btn" onClick={handleDownload}>
              다운로드
            </button>
            {usedManual && preManualResult && (
              <button type="button" className="submit-btn submit-btn--secondary" onClick={handleUndoManual}>
                변환 취소
              </button>
            )}
            {usedManual ? null : showingMosaic ? (
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
            {gifResult ? (
              <button type="button" className="submit-btn submit-btn--secondary" onClick={handleDownloadGif}>
                GIF 다운로드
              </button>
            ) : (
              <button
                type="button"
                className="submit-btn submit-btn--secondary"
                disabled={gifBusy}
                onClick={handleMakeGif}
              >
                {gifBusy ? "GIF 만드는 중..." : "GIF 만들기"}
              </button>
            )}
          </>
        ) : (
          <button type="button" className="submit-btn" disabled={!canTransform} onClick={() => handleTransform()}>
            {suggesting ? "눈썹 위치를 찾는 중..." : status === "processing" ? "처리 중..." : "사진 변환"}
          </button>
        )}
        {error && <span className="error-note">{error}</span>}
      </div>
    </div>
  );
}

export default App;
