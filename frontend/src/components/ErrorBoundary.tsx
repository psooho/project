import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  failed: boolean;
}

/** 화면을 그리다 예외가 나면 React는 화면 전체를 지워 흰 화면만 남긴다. 직원은 무슨 일인지도,
 *  무엇을 해야 하는지도 알 수 없다. 그 대신 안내와 새로고침 버튼을 보여준다.
 *
 *  서버 요청 실패처럼 이벤트 처리 중에 난 오류는 여기로 오지 않는다 — 그건 각 화면이 직접
 *  문구를 띄운다. 여기는 그리는 도중에 난, 예상하지 못한 오류용이다. */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { failed: false };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("화면 오류", error, info.componentStack);
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div className="gate">
        <div className="gate-box">
          <h1>화면에 문제가 생겼습니다</h1>
          <p>새로고침하면 다시 쓸 수 있습니다. 올려둔 사진은 다시 넣어주세요.</p>
          <button type="button" className="submit-btn" onClick={() => window.location.reload()}>
            새로고침
          </button>
        </div>
      </div>
    );
  }
}
