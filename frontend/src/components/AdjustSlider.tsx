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
  onCommit: (value: number) => void;
}

/** 끌어서 조절하는 슬라이더.
 *
 *  끄는 동안에는 숫자만 따라 움직이고, 손을 뗐을 때 한 번만 실제로 적용한다. 한 번 계산에
 *  2초쯤 걸려서, 움직일 때마다 보내면 요청이 밀린다. */
export function AdjustSlider({ label, value, min, max, step, busy, hint, onCommit }: AdjustSliderProps) {
  const [dragging, setDragging] = useState<number | null>(null);
  const shown = dragging ?? value;
  const toPercent = (v: number) => Math.round(v * 100);

  function commit() {
    if (dragging === null) return;
    const next = dragging;
    setDragging(null);
    if (next !== value) onCommit(next);
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
        onChange={(event) => setDragging(Number(event.target.value) / 100)}
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
