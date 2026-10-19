import { useEffect, useRef, useState } from "react";
import "./App.css";
import { BrushCanvas, EMPTY_STROKES, hasStrokes } from "./components/BrushCanvas";
import type { BrushStroke, StrokeSides } from "./components/BrushCanvas";
import { clearStoredPassword, getStoredPassword } from "./components/PasswordGate";
import { PhotoDropzone } from "./components/PhotoDropzone";
import { AdjustSlider } from "./components/AdjustSlider";
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

/** 서버가 이유를 알려줬으면 그걸 쓰고, 아니면 준비된 문구를 쓴다. */
/** 서버에 폼을 보낸다. 응답을 못 받은 경우와 암호가 바뀐 경우를 여기서 한 번에 처리한다.
 *
 *  서버가 죽었거나 연결이 끊기면 fetch는 응답 없이 실패하고, 브라우저가 "Failed to fetch"
 *  (사파리는 "Load failed") 같은 영어 문구를 던진다. 그게 그대로 화면에 떴었다. */
async function postForm(path: string, formData: FormData): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      headers: { "X-App-Password": getStoredPassword() },
      body: formData,
    });
  } catch {
    throw new Error("서버와 연결이 끊겼습니다. 잠시 후 다시 시도해주세요.");
  }
  if (res.status === 401) {
    // 암호가 바뀐 경우. 저장된 암호를 지우고 새로고침하면 다시 입력 화면이 뜬다.
    clearStoredPassword();
    window.location.reload();
    throw new Error("암호가 변경되었습니다.");
  }
  return res;
}

/** 서버가 이유를 주지 않은 실패를 상황에 맞는 말로. 운영 서버(Cloud Run)는 요청이 너무
 *  크면 413을, 인스턴스가 죽거나 재시작 중이면 502~504를 앱 대신 직접 돌려준다. */
function failureMessage(status: number) {
  if (status === 413) return "사진 용량이 너무 큽니다. 크기를 줄여서 다시 올려주세요.";
  if (status >= 502 && status <= 504) return "서버가 잠시 응답하지 않습니다. 잠시 후 다시 시도해주세요.";
  return "처리 중 예상치 못한 문제가 생겼습니다. 다시 시도해주세요.";
}

function messageOf(cause: unknown, fallback: string) {
  return cause instanceof Error && cause.message ? cause.message : fallback;
}

/** 백엔드가 "얼굴을 못 찾았다"고 알려주는 경고인지. 이때만 직접 지정 화면을 권한다. */
function isFaceMissing(warnings: string[]) {
  return warnings.some((warning) => warning.includes("얼굴을 찾지 못해"));
}

/** 후 사진 크기 미세조정. 두 사진의 고개 각도가 다르면 2D 변환으로는 어떤 기준점을 써도
 *  한쪽이 틀어지므로, 마지막은 사람 눈으로 맞춘다. */
const SCALE_STEP = 0.02;
const SCALE_MIN = 0.6;
const SCALE_MAX = 1.6;

/** 밝기·색감 보정 세기. 0이면 원본, 100%면 전 사진에 완전히 맞춘다. */
const MATCH_STEP = 0.1;
const MATCH_MIN = 0;
const MATCH_MAX = 1.5;

async function processPair(
  before: File,
  after: File,
  mosaic: boolean,
  manual?: ManualPoints,
  scaleAdjust = 1,
  level = false,
  brightnessStrength = 1,
  toneStrength = 1,
  preview = false,
  strokes?: StrokeSides,
): Promise<Results> {
  const formData = new FormData();
  formData.append("before", before);
  formData.append("after", after);
  formData.append("mosaic", String(mosaic));
  formData.append("scale_adjust", String(scaleAdjust));
  // 자동으로 찾은 점은 눈썹인 걸 알기 때문에 그 선을 수평으로 맞춘다. 직접 찍은 점은
  // 무엇인지 알 수 없어 방향을 강제하지 않는다.
  formData.append("level", String(level));
  formData.append("brightness_strength", String(brightnessStrength));
  formData.append("tone_strength", String(toneStrength));
  // 끄는 도중에는 작게 줄인 결과를 받는다 (서버 계산 181ms -> 60ms, 전송량도 크게 줄어든다).
  formData.append("preview", String(preview));
  // 결과 위에 칠한 자국. 크롭까지 끝난 뒤에 입혀 보고 칠한 자리에 그대로 남는다.
  if (hasStrokes(strokes)) formData.append("result_strokes", JSON.stringify(strokes));
  if (manual) {
    formData.append(
      "points",
      JSON.stringify({
        before: manual.before.map((p) => [p.x, p.y]),
        after: manual.after.map((p) => [p.x, p.y]),
      }),
    );
  }

  const res = await postForm("/api/process", formData);
  if (!res.ok) {
    // 서버가 왜 안 되는지 알려주는 경우가 있다 (기준점이 서로 안 맞는다, 파일이 깨졌다 등).
    // 그걸 그대로 보여줘야 사용자가 무엇을 고쳐야 할지 안다.
    const detail = await res
      .json()
      .then((body: { detail?: string }) => body.detail)
      .catch(() => undefined);
    throw new Error(detail || failureMessage(res.status));
  }

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
    labels?: string[];
    level?: boolean;
  } = await res.json();
  if (!data.available || !data.before || !data.after) return null;
  const toPoints = (pairs: [number, number][]) => pairs.map(([x, y]) => ({ x, y }));
  return {
    before: toPoints(data.before),
    after: toPoints(data.after),
    labels: data.labels ?? [],
    level: data.level ?? false,
  };
}

/** 정렬 없이 모자이크만 입혀서 돌려받는다. 사진 한 장만 보내도 된다. */
/** 정렬 없이 모자이크만 입혀서 돌려받는다. 사진 한 장씩 부른다 — 두 장을 한 번에 보내면
 *  둘 다 끝날 때까지 화면에 아무 변화가 없어서, 큰 사진에서는 멈춘 것처럼 보인다. */
async function mosaicOne(
  photo: File,
  face: boolean,
  strokes: BrushStroke[],
): Promise<{ blob: Blob; warnings: string[] }> {
  const formData = new FormData();
  formData.append("before", photo);
  formData.append("face", String(face));
  formData.append("strokes", JSON.stringify({ before: strokes, after: [] }));

  const res = await postForm("/api/mosaic", formData);
  if (!res.ok) {
    const detail = await res
      .json()
      .then((body: { detail?: string }) => body.detail)
      .catch(() => undefined);
    throw new Error(detail || failureMessage(res.status));
  }
  const data: { before: string; warnings?: string[] } = await res.json();
  const blob = await fetch(data.before).then((r) => r.blob());
  return { blob, warnings: data.warnings ?? [] };
}

/** 어디를 기준으로 맞췄는지 사람이 읽을 수 있게. 결과를 검토하려면 알아야 한다. */
const LANDMARK_NAMES: Record<string, string> = {
  brow_inner: "눈썹 안쪽",
  brow_outer: "눈썹 바깥쪽",
  eye_inner: "눈앞머리",
  eye_outer: "눈꼬리",
  nose_bridge: "콧대",
  ear_top: "귀 위",
  ear_lobe: "귓불",
};

/** 이미 변환이 끝난 두 장을 보내서 전→후로 스르륵 넘어가는 GIF를 받아온다. */
async function buildGif(before: Blob, after: Blob): Promise<Blob> {
  const formData = new FormData();
  formData.append("before", before, "before.jpg");
  formData.append("after", after, "after.jpg");

  const res = await postForm("/api/gif", formData);
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
  const [autoLabels, setAutoLabels] = useState<string[]>([]);
  // 밝기·색감·크기 조절값. 셋 다 같은 방식으로 다시 계산하므로 한 덩어리로 다룬다.
  const [adjust, setAdjust] = useState({ brightness: 1, tone: 1, scale: 1 });
  const [adjustBusy, setAdjustBusy] = useState(false);
  // 계산 중에 슬라이더를 또 움직이면 여기 담아뒀다가 끝난 뒤 한 번만 더 계산한다.
  const pendingAdjust = useRef<{ values: typeof adjust; preview: boolean } | null>(null);
  // 지금 화면의 결과를 만들 때 쓴 기준점. 크기만 조정할 때 다시 찾지 않고 그대로 쓴다
  // (자동 탐색은 15~20초가 걸리고 호출 요금도 든다).
  const [lastPoints, setLastPoints] = useState<ManualPoints | undefined>(undefined);
  // 자동으로 찾아준 점. 지금 쓰는 점이 이것과 같으면 눈썹인 걸 아는 셈이라 수평을 맞춘다.
  const [suggestedPoints, setSuggestedPoints] = useState<ManualPoints | null>(null);
  // 양쪽 눈썹이 다 잡힌 경우에만 수평을 맞출 수 있다 (백엔드가 판단해서 알려준다).
  const [suggestedLevel, setSuggestedLevel] = useState(false);

  // 붓으로 칠한 자국. 원본에 칠한 것과 결과에 칠한 것은 들어가는 시점이 달라 따로 둔다.
  const [sourceStrokes, setSourceStrokes] = useState<StrokeSides>(EMPTY_STROKES);
  const [resultStrokes, setResultStrokes] = useState<StrokeSides>(EMPTY_STROKES);
  // 지금 어느 사진 위에 칠하는 중인지. "source"는 원본, "result"는 변환 결과.
  const [painting, setPainting] = useState<null | "source" | "result">(null);
  // 칠하기를 취소했을 때 되돌릴 자국.
  const [strokesBackup, setStrokesBackup] = useState<StrokeSides>(EMPTY_STROKES);
  const [brushRadius, setBrushRadius] = useState(0.05);
  const [maskFace, setMaskFace] = useState(true);
  const [mosaicOnlyBusy, setMosaicOnlyBusy] = useState(false);
  // 칠하기 화면에 띄울 "자동 모자이크까지 들어간" 미리보기. 칠하기 전에 얼굴이 이미
  // 가려진 모습을 봐야, 어디를 더 칠해야 하는지 알 수 있다.
  type MosaicImage = { blob: Blob; url: string } | null;
  const [facePreview, setFacePreview] = useState<{ before: MosaicImage; after: MosaicImage }>({
    before: null,
    after: null,
  });
  const [facePreviewBusy, setFacePreviewBusy] = useState(false);
  const [facePreviewNotice, setFacePreviewNotice] = useState<string | null>(null);
  // 모자이크를 만드는 동안의 진행 상황. 사진 한 장이 한 단계다.
  const [mosaicProgress, setMosaicProgress] = useState<{
    done: number;
    total: number;
    label: string;
  } | null>(null);

  const displayed = showingMosaic && mosaicResult ? mosaicResult : baseResult;

  // 칠하기 화면에 띄울 원본 미리보기. 파일이 바뀔 때만 다시 만들고 쓰고 나면 해제한다.
  const [sourceUrls, setSourceUrls] = useState<{ before: string | null; after: string | null }>({
    before: null,
    after: null,
  });
  useEffect(() => {
    const before = beforePhoto ? URL.createObjectURL(beforePhoto) : null;
    const after = afterPhoto ? URL.createObjectURL(afterPhoto) : null;
    setSourceUrls({ before, after });
    return () => {
      if (before) URL.revokeObjectURL(before);
      if (after) URL.revokeObjectURL(after);
    };
  }, [beforePhoto, afterPhoto]);

  /** 변환 요청에 같이 보낼, 결과 위에 칠한 자국. */
  function strokePayload() {
    return resultStrokes;
  }

  /** 지금 쓰는 기준점이 자동으로 찾아준 것 그대로인지 (사용자가 손대면 false). */
  function isSuggested(points?: ManualPoints) {
    if (!points || !suggestedPoints) return false;
    const same = (a: NormalizedPoint[], b: NormalizedPoint[]) =>
      a.length === b.length && a.every((p, i) => p.x === b[i].x && p.y === b[i].y);
    return (
      suggestedLevel &&
      same(points.before, suggestedPoints.before) &&
      same(points.after, suggestedPoints.after)
    );
  }

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
    setAdjust({ brightness: 1, tone: 1, scale: 1 });
    pendingAdjust.current = null;
    setLastPoints(undefined);
    setSuggestedPoints(null);
    setSuggestedLevel(false);
    // 칠한 자국은 그 사진 위의 좌표라, 사진이 바뀌면 더 이상 맞지 않는다.
    setSourceStrokes(EMPTY_STROKES);
    setResultStrokes(EMPTY_STROKES);
    setPainting(null);
    clearFacePreview();
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
        setSuggestedPoints({ before: suggested.before, after: suggested.after });
        setSuggestedLevel(suggested.level);
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
      const first = await processPair(
        beforePhoto, afterPhoto, false, manual, 1, isSuggested(manual),
        adjust.brightness, adjust.tone, false, strokePayload(),
      );

      // 얼굴을 못 찾았으면 눈썹 위치를 자동으로 받아 한 번 더 맞춰본다. 사용자가 점을
      // 찍지 않아도 변환 한 번으로 끝나고, 결과가 어긋나면 "변환 취소"로 고칠 수 있다.
      let auto: { points: ManualPoints; result: Results } | null = null;
      if (!manual && isFaceMissing(first.warnings)) {
        setSuggesting(true);
        try {
          const suggested = await fetchSuggestedPoints(beforePhoto, afterPhoto);
          if (suggested) {
            const points = { before: suggested.before, after: suggested.after };
            auto = {
              points,
              result: await processPair(
                beforePhoto, afterPhoto, false, points, 1, suggested.level,
                adjust.brightness, adjust.tone, false, strokePayload(),
              ),
            };
            setSuggestedPoints(points);
            setSuggestedLevel(suggested.level);
            setAutoLabels(suggested.labels);
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
      setLastPoints(points);
      setAdjust({ brightness: 1, tone: 1, scale: 1 });
      if (!auto) setAutoLabels([]);
      clearGif();
      setStatus("done");
      if (points) setPicking(false);
      else if (!isFaceMissing(shown.warnings)) resetPicking();
    } catch (cause) {
      setError(messageOf(cause, "변환 중 문제가 발생했습니다. 다시 시도해주세요."));
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
      const processed = await processPair(
        beforePhoto, afterPhoto, true, lastPoints, adjust.scale, isSuggested(lastPoints),
        adjust.brightness, adjust.tone, false, strokePayload(),
      );
      setMosaicResult(toResultSet(processed));
      setShowingMosaic(true);
      clearGif();
      setMosaicStatus("done");
    } catch (cause) {
      setError(messageOf(cause, "모자이크 처리 중 문제가 발생했습니다. 다시 시도해주세요."));
      setMosaicStatus("idle");
    }
  }

  /** 밝기·색감·크기를 다시 계산한다. 정렬 기준점은 그대로 두므로 눈썹을 다시 찾지 않는다.
   *
   *  끄는 도중(preview=true)에는 작게 줄인 결과를 받아 실시간으로 보여주고, 손을 뗐을 때
   *  원본 크기로 한 번 더 받아 확정한다. 요청은 한 번에 하나씩만 나간다 — 처리 중에 또
   *  움직이면 마지막 값만 담아뒀다가 끝난 뒤 이어서 계산한다. */
  async function applyAdjust(next: Partial<typeof adjust>, preview = false) {
    if (!beforePhoto || !afterPhoto || status !== "done") return;
    const values = { ...adjust, ...next };
    setAdjust(values);
    if (adjustBusy) {
      pendingAdjust.current = { values, preview };
      return;
    }
    setAdjustBusy(true);
    setError(null);
    let current = { values, preview };
    try {
      for (;;) {
        const processed = await processPair(
          beforePhoto, afterPhoto, false, lastPoints, current.values.scale,
          isSuggested(lastPoints), current.values.brightness, current.values.tone,
          current.preview, strokePayload(),
        );
        // 직전 결과의 URL은 함수형 갱신으로 해제한다. 미리보기가 빠르게 이어지는 동안
        // 오래된 값을 붙잡고 있으면 URL이 새어나간다.
        setBaseResult((previous) => {
          revokeResultSet(previous);
          return toResultSet(processed);
        });
        setMosaicResult((previous) => {
          revokeResultSet(previous);
          return null;
        });
        setShowingMosaic(false);
        setMosaicStatus("idle");
        clearGif();
        if (!pendingAdjust.current) break;
        current = pendingAdjust.current;
        pendingAdjust.current = null;
      }
    } catch (cause) {
      setError(messageOf(cause, "조절에 실패했습니다. 다시 시도해주세요."));
    } finally {
      pendingAdjust.current = null;
      setAdjustBusy(false);
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

  /** 결과를 지금 값 그대로 다시 계산한다. 결과에 칠한 자국을 바꾼 뒤 반영할 때 쓴다. */
  async function recomputeResult() {
    if (!beforePhoto || !afterPhoto) return;
    const useMosaic = showingMosaic;
    setAdjustBusy(true);
    setError(null);
    try {
      const processed = await processPair(
        beforePhoto, afterPhoto, useMosaic, lastPoints, adjust.scale, isSuggested(lastPoints),
        adjust.brightness, adjust.tone, false, strokePayload(),
      );
      const next = toResultSet(processed);
      if (showingMosaic) {
        setMosaicResult((previous) => {
          revokeResultSet(previous);
          return next;
        });
      } else {
        setBaseResult((previous) => {
          revokeResultSet(previous);
          return next;
        });
        // 모자이크본은 옛 자국으로 만든 것이라 더 이상 맞지 않는다.
        setMosaicResult((previous) => {
          revokeResultSet(previous);
          return null;
        });
        setMosaicStatus("idle");
      }
      clearGif();
    } catch (cause) {
      setError(messageOf(cause, "모자이크 적용에 실패했습니다. 다시 시도해주세요."));
    } finally {
      setAdjustBusy(false);
    }
  }

  function clearFacePreview() {
    setFacePreview((previous) => {
      if (previous.before) URL.revokeObjectURL(previous.before.url);
      if (previous.after) URL.revokeObjectURL(previous.after.url);
      return { before: null, after: null };
    });
    setFacePreviewNotice(null);
    setMosaicProgress(null);
  }

  /** 사진을 한 장씩 모자이크하면서, 끝나는 대로 화면에 반영하고 진행 상황을 갱신한다.
   *
   *  큰 사진은 한 장에 2초 남짓 걸린다. 두 장을 한 번에 기다리게 하면 멈춘 것처럼 보여서,
   *  먼저 끝난 사진이라도 바로 보여준다. */
  async function buildMosaics(face: boolean, strokes: StrokeSides) {
    const jobs: { side: "before" | "after"; label: string; photo: File; strokes: BrushStroke[] }[] = [];
    if (beforePhoto) jobs.push({ side: "before", label: "전", photo: beforePhoto, strokes: strokes.before });
    if (afterPhoto) jobs.push({ side: "after", label: "후", photo: afterPhoto, strokes: strokes.after });

    const made: { before: Blob | null; after: Blob | null } = { before: null, after: null };
    const notices: string[] = [];
    for (const [index, job] of jobs.entries()) {
      setMosaicProgress({ done: index, total: jobs.length, label: `${job.label} 사진` });
      const { blob, warnings } = await mosaicOne(job.photo, face, job.strokes);
      made[job.side] = blob;
      notices.push(...warnings.map((w) => `${job.label} 사진: ${w}`));
      const url = URL.createObjectURL(blob);
      setFacePreview((previous) => {
        const old = previous[job.side];
        if (old) URL.revokeObjectURL(old.url);
        return { ...previous, [job.side]: { blob, url } };
      });
    }
    setMosaicProgress({ done: jobs.length, total: jobs.length, label: "" });
    return { made, notices };
  }

  /** 얼굴 자동 모자이크만 넣은 미리보기를 받아온다. 끄면 원본을 그대로 쓴다.
   *
   *  실패해도 칠하기는 할 수 있어야 하므로 오류로 막지 않고 원본 위에 칠하게 둔다. */
  async function refreshFacePreview(face: boolean) {
    if (!beforePhoto) return;
    if (!face) {
      clearFacePreview();
      return;
    }
    setFacePreviewBusy(true);
    setFacePreviewNotice(null);
    try {
      const { notices } = await buildMosaics(true, EMPTY_STROKES);
      if (notices.length > 0) setFacePreviewNotice(notices.join(" "));
    } catch {
      setFacePreviewNotice("자동 모자이크를 넣지 못했습니다. 가릴 곳을 직접 칠해주세요.");
    } finally {
      setFacePreviewBusy(false);
      setMosaicProgress(null);
    }
  }

  /** 칠하기 화면으로 들어간다. 취소하면 들어오기 전 상태로 되돌린다. */
  function startPainting(where: "source" | "result") {
    setStrokesBackup(where === "source" ? sourceStrokes : resultStrokes);
    setPainting(where);
    setError(null);
    if (where === "source") void refreshFacePreview(maskFace);
  }

  function cancelPainting() {
    if (painting === "source") setSourceStrokes(strokesBackup);
    else if (painting === "result") setResultStrokes(strokesBackup);
    setPainting(null);
    clearFacePreview();
  }

  /** 얼굴 자동 모자이크와 칠한 자국을 원본에 한 번에 입혀서 바로 내려받는다.
   *
   *  화면의 붓 자국은 브라우저가 그린 미리보기라, 저장할 파일은 여기서 서버에 한 번 더
   *  요청해 원본 화질로 만든다. */
  async function handleDownloadMosaic() {
    if (!beforePhoto) return;
    setMosaicOnlyBusy(true);
    setError(null);
    try {
      // 들어올 때 만든 것이 곧 저장할 파일이다 — 더 칠한 게 없으면 다시 만들지 않는다.
      const ready = facePreview.before && maskFace && !hasStrokes(sourceStrokes);
      const files = ready
        ? { before: facePreview.before!.blob, after: facePreview.after?.blob ?? null }
        : (await buildMosaics(maskFace, sourceStrokes)).made;
      if (files.before) downloadBlob(files.before, withJpgName(beforePhoto.name, "mosaic"));
      if (files.after && afterPhoto) {
        downloadBlob(files.after, withJpgName(afterPhoto.name, "mosaic"));
      }
    } catch (cause) {
      setError(messageOf(cause, "모자이크 저장에 실패했습니다. 다시 시도해주세요."));
    } finally {
      setMosaicOnlyBusy(false);
      setMosaicProgress(null);
    }
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
            얼굴을 찾지 못해{" "}
            <strong>
              {autoLabels.length > 0
                ? `${autoLabels.map((l) => LANDMARK_NAMES[l] ?? l).join(" · ")} ${autoLabels.length}곳을 자동으로 찾아`
                : "기준점을 자동으로 찾아"}
            </strong>{" "}
            크기와 각도를 맞췄습니다. 결과를 확인해주세요 — 어긋났으면{" "}
            <strong>변환 취소</strong>를 눌러 직접 고칠 수 있습니다.
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

      {painting && (
        <section className="brush-section">
          <h2 className="picker-title">
            {painting === "source" ? "모자이크" : "모자이크 더 칠하기 (결과)"}
          </h2>
          <p className="picker-guide">
            더 가리고 싶은 곳을 마우스로 끌어서 칠해주세요. 칠한 자리는 바로 흐려져 보입니다.
            {painting === "source"
              ? " 저장할 파일은 다운로드를 누를 때 원본 화질로 다시 만듭니다."
              : " '결과에 적용'을 누르면 최종본을 만들고, 그 뒤 밝기·색감·크기를 조절해도 자국은 남습니다."}
          </p>
          <div className="brush-controls">
            <label className="brush-size">
              붓 크기
              <input
                type="range"
                min={1}
                max={20}
                step={1}
                value={Math.round(brushRadius * 200)}
                onChange={(event) => setBrushRadius(Number(event.target.value) / 200)}
              />
              <span>{Math.round(brushRadius * 200)}</span>
            </label>
            {painting === "source" && (
              <label className="brush-face">
                <input
                  type="checkbox"
                  checked={maskFace}
                  disabled={facePreviewBusy}
                  onChange={(event) => {
                    setMaskFace(event.target.checked);
                    void refreshFacePreview(event.target.checked);
                  }}
                />
                얼굴(눈·코·입)도 자동으로 가리기
              </label>
            )}
          </div>
          {mosaicProgress && (
            <div className="progress" role="status" aria-live="polite">
              <div className="progress-track">
                <div
                  className={`progress-fill${
                    mosaicProgress.done < mosaicProgress.total ? " progress-fill--working" : ""
                  }`}
                  style={{ width: `${(mosaicProgress.done / mosaicProgress.total) * 100}%` }}
                />
              </div>
              <span className="progress-text">
                {mosaicProgress.done < mosaicProgress.total
                  ? `${mosaicProgress.label}에서 얼굴을 찾아 가리는 중... (${mosaicProgress.done}/${mosaicProgress.total}장 완료)`
                  : `${mosaicProgress.total}장 모두 완료`}
              </span>
            </div>
          )}
          {painting === "source" && !facePreviewBusy && facePreviewNotice && (
            <p className="picker-notice">{facePreviewNotice}</p>
          )}
          {painting === "source" && !facePreviewBusy && !facePreviewNotice && facePreview.before && (
            <p className="picker-notice picker-notice--ok">
              얼굴은 자동으로 가려뒀습니다. <strong>더 가릴 곳만</strong> 칠해주세요.
            </p>
          )}
          <div className="picker-grid">
            {painting === "source" ? (
              <>
                {sourceUrls.before && (
                  <BrushCanvas
                    label="전 (Before)"
                    src={facePreview.before?.url ?? sourceUrls.before}
                    radius={brushRadius}
                    strokes={sourceStrokes.before}
                    onChange={(update) =>
                      setSourceStrokes((prev) => ({ ...prev, before: update(prev.before) }))
                    }
                  />
                )}
                {sourceUrls.after && (
                  <BrushCanvas
                    label="후 (After)"
                    src={facePreview.after?.url ?? sourceUrls.after}
                    radius={brushRadius}
                    strokes={sourceStrokes.after}
                    onChange={(update) =>
                      setSourceStrokes((prev) => ({ ...prev, after: update(prev.after) }))
                    }
                  />
                )}
              </>
            ) : (
              displayed && (
                <>
                  <BrushCanvas
                    label="전 (Before)"
                    src={displayed.urls.before}
                    radius={brushRadius}
                    strokes={resultStrokes.before}
                    onChange={(update) =>
                      setResultStrokes((prev) => ({ ...prev, before: update(prev.before) }))
                    }
                  />
                  <BrushCanvas
                    label="후 (After)"
                    src={displayed.urls.after}
                    radius={brushRadius}
                    strokes={resultStrokes.after}
                    onChange={(update) =>
                      setResultStrokes((prev) => ({ ...prev, after: update(prev.after) }))
                    }
                  />
                </>
              )
            )}
          </div>
          <div className="actions">
            {painting === "source" ? (
              <button
                type="button"
                className="submit-btn"
                disabled={mosaicOnlyBusy || facePreviewBusy}
                onClick={handleDownloadMosaic}
              >
                {mosaicOnlyBusy ? "만드는 중..." : "다운로드"}
              </button>
            ) : (
              <button
                type="button"
                className="submit-btn"
                disabled={adjustBusy}
                onClick={async () => {
                  setPainting(null);
                  await recomputeResult();
                }}
              >
                {adjustBusy ? "적용 중..." : "결과에 적용"}
              </button>
            )}
            <button type="button" className="submit-btn submit-btn--secondary" onClick={cancelPainting}>
              {painting === "source" ? "닫기" : "취소"}
            </button>
          </div>
        </section>
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

      {status === "done" && (
        <section className="adjust-panel">
          <AdjustSlider
            label="밝기 보정"
            value={adjust.brightness}
            min={MATCH_MIN}
            max={MATCH_MAX}
            step={MATCH_STEP}
            busy={adjustBusy}
            hint="0%면 원본, 100%면 전 사진에 맞춤"
            onPreview={(value) => applyAdjust({ brightness: value }, true)}
            onCommit={(value) => applyAdjust({ brightness: value })}
          />
          <AdjustSlider
            label="색감 보정"
            value={adjust.tone}
            min={MATCH_MIN}
            max={MATCH_MAX}
            step={MATCH_STEP}
            busy={adjustBusy}
            hint="두 사진의 색이 실제로 다를 때만 효과가 보입니다"
            onPreview={(value) => applyAdjust({ tone: value }, true)}
            onCommit={(value) => applyAdjust({ tone: value })}
          />
          <AdjustSlider
            label="후 사진 크기"
            value={adjust.scale}
            min={SCALE_MIN}
            max={SCALE_MAX}
            step={SCALE_STEP}
            busy={adjustBusy}
            hint="두상 크기가 안 맞으면 끌어서 맞춰주세요"
            onPreview={(value) => applyAdjust({ scale: value }, true)}
            onCommit={(value) => applyAdjust({ scale: value })}
          />
        </section>
      )}

      <div className="actions">
        {painting ? null : status === "done" ? (
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
            <button
              type="button"
              className="submit-btn submit-btn--secondary"
              disabled={adjustBusy}
              onClick={() => startPainting("result")}
            >
              {hasStrokes(resultStrokes) ? "모자이크 다시 칠하기" : "모자이크 더 칠하기"}
            </button>
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
          <>
            <button type="button" className="submit-btn" disabled={!canTransform} onClick={() => handleTransform()}>
              {suggesting ? "눈썹 위치를 찾는 중..." : status === "processing" ? "처리 중..." : "사진 변환"}
            </button>
            {beforePhoto && !painting && (
              <button
                type="button"
                className="submit-btn submit-btn--secondary"
                disabled={status === "processing"}
                onClick={() => startPainting("source")}
              >
                모자이크만 하기
              </button>
            )}
          </>
        )}
        {error && <span className="error-note">{error}</span>}
      </div>
    </div>
  );
}

export default App;
