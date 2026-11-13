import { defineConfig } from 'vite';
import { resolve } from 'node:path';

export default defineConfig({
  base: './',
  build: {
    rollupOptions: {
      input: {
        world: resolve(import.meta.dirname, 'index.html'),
        bench: resolve(import.meta.dirname, 'bench.html')
      }
    }
  }
});
