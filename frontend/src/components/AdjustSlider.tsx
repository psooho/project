import { useState } from "react";

interface AdjustSliderProps {
  label: string;
  /** 실제로 적용된 값 (0~1.5 같은 비율). */
  value: number;
  min: number;
  max: number;
  step: number;
  busy: boolean;
  hint?: string;
  /** 끄는 도중 계속 불린다. 가벼운 미리보기용. */
  onPreview: (value: number) => void;
  /** 손을 뗐을 때 한 번 불린다. 원본 크기로 확정한다. */
  onCommit: (value: number) => void;
}

/** 끌어서 조절하는 슬라이더.
 *
 *  끄는 동안에는 작게 줄인 미리보기를 계속 받아 실시간으로 보여주고, 손을 뗐을 때 원본
 *  크기로 한 번 더 받아 확정한다. 요청은 한 번에 하나씩만 나가도록 부모가 직렬화한다. */
export function AdjustSlider({
  label,
  value,
  min,
  max,
  step,
  busy,
  hint,
  onPreview,
  onCommit,
}: AdjustSliderProps) {
  const [dragging, setDragging] = useState<number | null>(null);
  const shown = dragging ?? value;
  const toPercent = (v: number) => Math.round(v * 100);

  function commit() {
    // 움직이지 않았으면(그냥 클릭했으면) dragging이 없다. 그때만 건너뛴다.
    if (dragging === null) return;
    const next = dragging;
    setDragging(null);
    // 값이 그대로여도 반드시 보낸다 — 끄는 동안 받은 건 작게 줄인 미리보기라,
    // 여기서 원본 크기로 다시 받아야 다운로드·GIF에 쓸 수 있다.
    onCommit(next);
  }

  return (
    <div className="adjust">
      <span className="adjust-label">{label}</span>
      <input
        type="range"
        className="adjust-range"
        min={toPercent(min)}
        max={toPercent(max)}
        step={toPercent(step)}
        value={toPercent(shown)}
        onChange={(event) => {
          const next = Number(event.target.value) / 100;
          setDragging(next);
          onPreview(next);
        }}
        // 마우스·터치·키보드 어느 쪽으로 움직였든 "끝난 시점"에 적용한다.
        onPointerUp={commit}
        onKeyUp={commit}
        onBlur={commit}
      />
      <span className={`adjust-value${shown !== 1 ? " adjust-value--changed" : ""}`}>
        {busy ? "..." : `${toPercent(shown)}%`}
      </span>
      {value !== 1 && (
        <button type="button" className="adjust-reset" disabled={busy} onClick={() => onCommit(1)}>
          되돌리기
        </button>
      )}
      {hint && <span className="adjust-hint">{hint}</span>}
    </div>
  );
}
