import { useRef, useState } from "react";
import type { DragEvent } from "react";

// HEIC는 받지 않는다. 서버가 풀 수 없고, 윈도우 크롬은 미리보기조차 못 띄워서 기준점을
// 찍을 수도 없다. 받는 형식에서 빼두면 아이폰 사파리는 올릴 때 JPG로 바꿔서 보낸다.
const ACCEPTED_TYPES = ["image/jpeg", "image/png"];

interface PhotoDropzoneProps {
  label: string;
  file: File | null;
  onChange: (file: File | null) => void;
  /** 변환이 끝난 결과 미리보기 URL. 있으면 원본 대신 이걸 보여준다. */
  resultUrl?: string | null;
}

function isAcceptedFile(file: File) {
  if (ACCEPTED_TYPES.includes(file.type)) return true;
  return /\.(jpe?g|png)$/i.test(file.name);
}

function isHeic(file: File) {
  return /^image\/hei[cf]$/.test(file.type) || /\.hei[cf]$/i.test(file.name);
}

export function PhotoDropzone({ label, file, onChange, resultUrl }: PhotoDropzoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const previewUrl = resultUrl ?? (file ? URL.createObjectURL(file) : null);

  function handleFiles(files: FileList | null) {
    const picked = files?.[0];
    if (!picked) return;
    if (!isAcceptedFile(picked)) {
      setError(
        isHeic(picked)
          ? "HEIC(아이폰 기본 형식)는 열 수 없습니다. JPG로 바꾸거나 카카오톡으로 받아서 올려주세요."
          : "JPG, PNG 파일만 업로드할 수 있습니다.",
      );
      return;
    }
    setError(null);
    onChange(picked);
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragOver(false);
    handleFiles(e.dataTransfer.files);
  }

  return (
    <div className="dropzone-wrap">
      <div className="dropzone-label">{label}</div>
      <div
        className={`dropzone${isDragOver ? " dropzone--drag" : ""}${file ? " dropzone--filled" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        role="button"
        tabIndex={0}
      >
        {previewUrl ? (
          <>
            <img src={previewUrl} alt={`${label} 미리보기`} className="dropzone-preview" />
            {resultUrl && <span className="dropzone-badge">변환 결과</span>}
            <button
              type="button"
              className="dropzone-remove"
              onClick={(e) => {
                e.stopPropagation();
                onChange(null);
                setError(null);
                if (inputRef.current) inputRef.current.value = "";
              }}
            >
              제거
            </button>
          </>
        ) : (
          <div className="dropzone-placeholder">
            <span className="dropzone-icon">＋</span>
            <span>클릭하거나 파일을 끌어다 놓으세요</span>
            <span className="dropzone-hint">JPG · PNG</span>
          </div>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png,image/jpeg,image/png"
        hidden
        onChange={(e) => handleFiles(e.target.files)}
      />
      {error && <div className="dropzone-error">{error}</div>}
      {file && <div className="dropzone-filename">{file.name}</div>}
    </div>
  );
}
