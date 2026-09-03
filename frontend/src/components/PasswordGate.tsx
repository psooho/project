import { useEffect, useState } from "react";

const STORAGE_KEY = "app-password";

export function getStoredPassword(): string {
  try {
    return localStorage.getItem(STORAGE_KEY) ?? "";
  } catch {
    return ""; // 시크릿 모드 등 localStorage를 못 쓰는 경우
  }
}

export function clearStoredPassword() {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* 저장소를 못 써도 동작에는 문제 없다 */
  }
}

async function checkPassword(password: string): Promise<{ required: boolean; valid: boolean }> {
  const res = await fetch("/api/auth", { headers: { "X-App-Password": password } });
  if (!res.ok) throw new Error("서버에 연결할 수 없습니다.");
  return res.json();
}

interface PasswordGateProps {
  children: React.ReactNode;
}

/** 암호가 설정된 서버라면, 맞는 암호를 넣기 전까지 앱을 가린다.
 *  한 번 통과하면 브라우저에 저장되므로 다음부터는 묻지 않는다. */
export function PasswordGate({ children }: PasswordGateProps) {
  const [unlocked, setUnlocked] = useState(false);
  const [checking, setChecking] = useState(true);
  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    checkPassword(getStoredPassword())
      .then(({ valid }) => setUnlocked(valid))
      .catch(() => setUnlocked(false))
      .finally(() => setChecking(false));
  }, []);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const { valid } = await checkPassword(input);
      if (!valid) {
        setError("암호가 올바르지 않습니다.");
        return;
      }
      try {
        localStorage.setItem(STORAGE_KEY, input);
      } catch {
        /* 저장을 못 해도 이번 세션에서는 쓸 수 있게 통과시킨다 */
      }
      setUnlocked(true);
    } catch {
      setError("서버에 연결할 수 없습니다.");
    } finally {
      setSubmitting(false);
    }
  }

  if (checking) return null;
  if (unlocked) return <>{children}</>;

  return (
    <div className="gate">
      <form className="gate-box" onSubmit={handleSubmit}>
        <h1>전후사진 변환</h1>
        <p>병원 공용 암호를 입력하세요. 이 브라우저에서는 한 번만 입력하면 됩니다.</p>
        <input
          type="password"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="암호"
          autoFocus
        />
        <button type="submit" className="submit-btn" disabled={submitting || !input}>
          {submitting ? "확인 중..." : "들어가기"}
        </button>
        {error && <span className="error-note">{error}</span>}
      </form>
    </div>
  );
}
