import { useEffect, useState } from "react";
import type { MouseEvent } from "react";

/** 사진 크기 대비 0~1 비율 좌표. 화면에 줄여 띄운 사진 위에서 클릭을 받으므로,
 *  픽셀이 아니라 비율로 다뤄야 표시 크기와 원본 크기가 달라도 어긋나지 않는다. */
export interface NormalizedPoint {
  x: number;
  y: number;
}

/** 이 이상은 정확도가 거의 안 늘고 화면만 복잡해진다 (백엔드도 같은 값으로 막는다). */
export const MAX_POINTS = 6;

interface PointPickerProps {
  label: string;
  file: File;
  points: NormalizedPoint[];
  /** 이전 값을 받아 새 값을 돌려주는 형태로 넘긴다. 빠르게 두 번 클릭하면 두 핸들러가
   *  같은(오래된) points를 보고 계산해서 첫 점이 사라지는 문제가 있었다. */
  onChange: (update: (previous: NormalizedPoint[]) => NormalizedPoint[]) => void;
}

export function PointPicker({ label, file, points, onChange }: PointPickerProps) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    const objectUrl = URL.createObjectURL(file);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  function handleClick(event: MouseEvent<HTMLImageElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const point = {
      x: (event.clientX - rect.left) / rect.width,
      y: (event.clientY - rect.top) / rect.height,
    };
    // 다 채운 뒤 또 누르면 처음부터 다시 찍는 것으로 본다.
    onChange((previous) => (previous.length >= MAX_POINTS ? [point] : [...previous, point]));
  }

  return (
    <div className="picker">
      <div className="picker-header">
        <span className="picker-label">{label}</span>
        <span className={`picker-count${points.length >= 2 ? " picker-count--done" : ""}`}>
          {points.length}개
        </span>
      </div>
      <div className="picker-stage">
        {url && (
          <img src={url} alt={`${label} 기준점 지정`} className="picker-image" onClick={handleClick} />
        )}
        {points.map((point, index) => (
          <span
            key={index}
            className="picker-dot"
            style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }}
          >
            {index + 1}
          </span>
        ))}
      </div>
      {points.length > 0 && (
        <button type="button" className="picker-reset" onClick={() => onChange(() => [])}>
          다시 찍기
        </button>
      )}
    </div>
  );
}
