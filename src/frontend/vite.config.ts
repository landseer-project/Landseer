import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

export default defineConfig({
  server: {
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/health': { target: 'http://localhost:8000', changeOrigin: true },
      '/info': { target: 'http://localhost:8000', changeOrigin: true },
      '/pipeline': { target: 'http://localhost:8000', changeOrigin: true },
      '/pipelines': { target: 'http://localhost:8000', changeOrigin: true },
      '/workflows': { target: 'http://localhost:8000', changeOrigin: true },
      '/tasks': { target: 'http://localhost:8000', changeOrigin: true },
      '/progress': { target: 'http://localhost:8000', changeOrigin: true },
      '/scheduler': { target: 'http://localhost:8000', changeOrigin: true },
      '/workers': { target: 'http://localhost:8000', changeOrigin: true },
      '/tools': { target: 'http://localhost:8000', changeOrigin: true },
      '/registry': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(process.cwd(), 'src'),
    },
  },
  plugins: [react()],
});

