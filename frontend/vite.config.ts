import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // 배포 시에는 백엔드가 프론트엔드까지 같이 서빙해서 같은 오리진이 되므로, 프론트엔드는
  // 항상 상대경로(/api/...)로 호출한다. 개발 중에만 이 프록시가 백엔드로 넘겨준다.
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
})
