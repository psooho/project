import { useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

/** 붓으로 한 번 그은 자국. 좌표는 사진 크기 대비 0~1 비율, 반지름은 짧은 변 대비 비율이다.
 *  화면에 줄여 띄운 사진 위에서 받으므로 픽셀로 다루면 표시 크기에 따라 어긋난다. */
export interface BrushStroke {
  radius: number;
  points: [number, number][];
}

export interface StrokeSides {
  before: BrushStroke[];
  after: BrushStroke[];
}

export const EMPTY_STROKES: StrokeSides = { before: [], after: [] };

export function hasStrokes(sides: StrokeSides | null | undefined) {
  return Boolean(sides && (sides.before.length > 0 || sides.after.length > 0));
}

interface BrushCanvasProps {
  label: string;
  src: string;
  /** 짧은 변 대비 붓 반지름 비율. 두 사진이 같은 붓을 쓰도록 부모가 들고 있는다. */
  radius: number;
  strokes: BrushStroke[];
  onChange: (update: (previous: BrushStroke[]) => BrushStroke[]) => void;
}

/** 끌면서 지나간 자리가 멀면 중간 점을 채운다. 빠르게 끌 때 점 사이가 벌어져 자국이
 *  끊기는 걸 막는다. 서버는 점 사이를 선으로 이어 칠하므로 이건 화면 미리보기용이다. */
function fill(from: [number, number], to: [number, number], step: number): [number, number][] {
  const distance = Math.hypot(to[0] - from[0], to[1] - from[1]);
  const count = Math.floor(distance / step);
  const points: [number, number][] = [];
  for (let i = 1; i <= count; i += 1) {
    const t = i / (count + 1);
    points.push([from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t]);
  }
  points.push(to);
  return points;
}

export function BrushCanvas({ label, src, radius, strokes, onChange }: BrushCanvasProps) {
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const painting = useRef(false);
  const [size, setSize] = useState({ width: 0, height: 0 });

  // 캔버스는 사진이 화면에 그려진 크기에 정확히 맞춰야 좌표가 어긋나지 않는다.
  useEffect(() => {
    const image = imageRef.current;
    if (!image) return;
    const measure = () => setSize({ width: image.clientWidth, height: image.clientHeight });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(image);
    return () => observer.disconnect();
  }, [src]);

  // 칠한 자리를 실제로 흐리게 그려서 보여준다. 최종 결과는 서버가 만들지만, 어디가
  // 가려지는지는 여기서 바로 보여야 칠할 양을 가늠할 수 있다.
  useEffect(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image || !size.width || !size.height) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, size.width, size.height);
    if (strokes.length === 0) return;

    const shorter = Math.min(size.width, size.height);
    context.save();
    context.beginPath();
    for (const stroke of strokes) {
      const r = Math.max(1, stroke.radius * shorter);
      for (const [x, y] of stroke.points) {
        const px = x * size.width;
        const py = y * size.height;
        context.moveTo(px + r, py);
        context.arc(px, py, r, 0, Math.PI * 2);
      }
    }
    context.clip();
    context.filter = `blur(${Math.max(4, radius * shorter * 0.8)}px)`;
    context.drawImage(image, 0, 0, size.width, size.height);
    context.restore();
  }, [strokes, size, radius]);

  function at(event: ReactPointerEvent<HTMLCanvasElement>): [number, number] {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    return [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))];
  }

  function start(event: ReactPointerEvent<HTMLCanvasElement>) {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    painting.current = true;
    const point = at(event);
    onChange((previous) => [...previous, { radius, points: [point] }]);
  }

  function extend(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!painting.current) return;
    const point = at(event);
    onChange((previous) => {
      if (previous.length === 0) return previous;
      const last = previous[previous.length - 1];
      const from = last.points[last.points.length - 1];
      const step = Math.max(last.radius / 2, 0.004);
      const added = fill(from, point, step);
      return [...previous.slice(0, -1), { ...last, points: [...last.points, ...added] }];
    });
  }

  function stop() {
    painting.current = false;
  }

  return (
    <div className="brush">
      <div className="brush-header">
        <span className="brush-label">{label}</span>
        <span className="brush-count">{strokes.length}번 칠함</span>
        <button
          type="button"
          className="link-btn"
          disabled={strokes.length === 0}
          onClick={() => onChange((previous) => previous.slice(0, -1))}
        >
          한 번 취소
        </button>
        <button
          type="button"
          className="link-btn"
          disabled={strokes.length === 0}
          onClick={() => onChange(() => [])}
        >
          모두 지우기
        </button>
      </div>
      <div className="brush-stage">
        <img ref={imageRef} src={src} alt={`${label} 모자이크 칠하기`} className="brush-image" />
        <canvas
          ref={canvasRef}
          className="brush-canvas"
          width={size.width}
          height={size.height}
          style={{ width: size.width, height: size.height }}
          onPointerDown={start}
          onPointerMove={extend}
          onPointerUp={stop}
          onPointerCancel={stop}
        />
      </div>
    </div>
  );
}
