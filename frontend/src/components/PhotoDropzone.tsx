import { useRef, useState } from "react";
import type { DragEvent } from "react";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/heic", "image/heif"];

interface PhotoDropzoneProps {
  label: string;
  file: File | null;
  onChange: (file: File | null) => void;
}

function isAcceptedFile(file: File) {
  if (ACCEPTED_TYPES.includes(file.type)) return true;
  return /\.(jpe?g|png|heic|heif)$/i.test(file.name);
}

export function PhotoDropzone({ label, file, onChange }: PhotoDropzoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const previewUrl = file ? URL.createObjectURL(file) : null;

  function handleFiles(files: FileList | null) {
    const picked = files?.[0];
    if (!picked) return;
    if (!isAcceptedFile(picked)) {
      setError("JPG, PNG, HEIC 파일만 업로드할 수 있습니다.");
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
            <span className="dropzone-hint">JPG · PNG · HEIC</span>
          </div>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.heic,.heif,image/*"
        hidden
        onChange={(e) => handleFiles(e.target.files)}
      />
      {error && <div className="dropzone-error">{error}</div>}
      {file && <div className="dropzone-filename">{file.name}</div>}
    </div>
  );
}
