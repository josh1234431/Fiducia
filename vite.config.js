import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Electron loads the built bundle from the filesystem, so assets must be
  // referenced relatively rather than from the server root.
  base: './',
  server: {
    // Electron loads 127.0.0.1 explicitly; bare "localhost" can resolve to
    // IPv6 ::1 only, leaving the IPv4 address refusing connections.
    host: '127.0.0.1',
    port: 5273,
    strictPort: true,
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'chrome128',
  },
})
