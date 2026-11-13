import { defineConfig } from 'vite';
import { resolve } from 'node:path';
import { spawn } from 'node:child_process';

// During `vite dev`, repaint the pre-painted sprite sheets (scripts/build-sprites.mjs) whenever the sprite code
// in src/shared changes, then reload the page so it picks up the new images. Changes that arrive while a
// repaint is running are folded into one more run after it.
function spriteSheets() {
  return {
    name: 'emberwood-sprite-sheets',
    apply: 'serve',
    configureServer(server) {
      const shared = resolve(import.meta.dirname, 'src', 'shared');
      let running = false, again = false, timer = null;
      const run = () => {
        if (running) { again = true; return; }
        running = true;
        const child = spawn(process.execPath, [resolve(import.meta.dirname, 'scripts', 'build-sprites.mjs')], { stdio: 'inherit' });
        child.on('exit', (code) => {
          running = false;
          if (again) { again = false; run(); return; }
          if (code === 0) server.ws.send({ type: 'full-reload' });
        });
      };
      // (paths compared without case on Windows, where the drive letter's case varies)
      const norm = (p) => (process.platform === 'win32' ? resolve(p).toLowerCase() : resolve(p));
      server.watcher.on('change', (file) => {
        if (!norm(file).startsWith(norm(shared))) return;
        clearTimeout(timer);
        timer = setTimeout(run, 300);
      });
    }
  };
}

export default defineConfig({
  base: './',
  plugins: [spriteSheets()],
  build: {
    rollupOptions: {
      input: {
        world: resolve(import.meta.dirname, 'index.html'),
        bench: resolve(import.meta.dirname, 'bench.html')
      }
    }
  }
});
