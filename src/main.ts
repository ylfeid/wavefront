import { Simulation } from './sim';
import { Renderer, type ColorMode } from './renderer';
import { scenes, type Scene } from './presets';

type Tool = 'fire' | 'wall' | 'erase';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const MAX_MOMENTUM = 2.2; // keep well below the grid's Nyquist limit of π
const DRAG_TO_MOMENTUM = 0.025;

async function init() {
  if (!navigator.gpu) return fail();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return fail('No suitable GPU adapter was found.');
  const device = await adapter.requestDevice();
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') fail(`The GPU device was lost: ${info.message}`);
  });

  const params = new URLSearchParams(location.search);
  const requested = Number(params.get('n'));
  const n = [256, 512, 1024].includes(requested) ? requested : 512;

  const sim = new Simulation(device, n);
  const simCanvas = $<HTMLCanvasElement>('sim');
  const overlay = $<HTMLCanvasElement>('overlay');
  const ctx = overlay.getContext('2d')!;
  const renderer = new Renderer(device, simCanvas, sim);
  new App(sim, renderer, simCanvas, overlay, ctx).start(params.get('scene'));
}

function fail(reason?: string) {
  $('unsupported').hidden = false;
  if (reason) $('unsupported-reason').textContent = reason;
}

class App {
  private scene: Scene = scenes[0];
  private tool: Tool = 'fire';
  private colorMode: ColorMode = 'probability';
  private paused = false;
  private speed = 6;
  private exposure = 1;
  private sigma = 14;
  private brush = 6;

  private drag: { x0: number; y0: number; x: number; y: number; shift: boolean } | null = null;
  private pointer: { x: number; y: number } | null = null;
  private painting: { x: number; y: number; erase: boolean } | null = null;
  private pings: { x: number; y: number; start: number }[] = [];
  private frame = 0;
  private peak = 0;
  private smoothedPeak = 1e-3;
  private snapshotRequested = false;
  private stopped = false;

  constructor(
    private readonly sim: Simulation,
    private readonly renderer: Renderer,
    private readonly simCanvas: HTMLCanvasElement,
    private readonly overlay: HTMLCanvasElement,
    private readonly ctx: CanvasRenderingContext2D,
  ) {}

  start(sceneId: string | null) {
    this.buildSceneButtons();
    this.bindControls();
    this.bindPointer();
    this.bindKeys();
    this.observeSize();
    this.setTool('fire');
    this.setColorMode('probability');
    this.loadScene(scenes.find((s) => s.id === sceneId) ?? scenes[0]);
    this.sim.device.lost.then(() => (this.stopped = true));
    requestAnimationFrame(this.tick);
  }

  private buildSceneButtons() {
    const container = $('scenes');
    for (const scene of scenes) {
      const button = document.createElement('button');
      button.textContent = scene.name;
      button.dataset.scene = scene.id;
      button.addEventListener('click', () => this.loadScene(scene));
      container.append(button);
    }
  }

  private loadScene(scene: Scene) {
    this.scene = scene;
    const { sim } = this;
    scene.potential(sim.potential, sim.n, sim.wallV);
    this.setAbsorb(!scene.closed);
    sim.markPotentialDirty();
    sim.addPacket(scene.packet(sim.n), true);
    $('blurb').textContent = scene.blurb;
    for (const b of $('scenes').querySelectorAll<HTMLButtonElement>('button')) {
      b.classList.toggle('active', b.dataset.scene === scene.id);
    }
    const url = new URL(location.href);
    url.searchParams.set('scene', scene.id);
    history.replaceState(null, '', url);
  }

  private reset() {
    this.loadScene(this.scene);
  }

  private clearWalls() {
    this.sim.potential.fill(0);
    this.sim.markPotentialDirty();
  }

  private bindControls() {
    for (const b of $('tools').querySelectorAll<HTMLButtonElement>('button')) {
      b.addEventListener('click', () => this.setTool(b.dataset.tool as Tool));
    }
    for (const b of $('color-mode').querySelectorAll<HTMLButtonElement>('button')) {
      b.addEventListener('click', () => this.setColorMode(b.dataset.mode as ColorMode));
    }
    $('btn-play').addEventListener('click', () => this.setPaused(!this.paused));
    $('btn-reset').addEventListener('click', () => this.reset());
    $('btn-measure').addEventListener('click', () => this.measure());
    $('btn-clear').addEventListener('click', () => this.clearWalls());
    $<HTMLInputElement>('in-absorb').addEventListener('change', (e) =>
      this.setAbsorb((e.target as HTMLInputElement).checked),
    );

    this.slider('in-speed', 'out-speed', (v) => {
      this.speed = v;
      return `${v}×`;
    });
    this.slider('in-exposure', 'out-exposure', (v) => {
      this.exposure = 2 ** v;
      return `${this.exposure.toFixed(2)}×`;
    });
    this.slider('in-sigma', 'out-sigma', (v) => {
      this.sigma = v;
      return `σ ${v}`;
    });
    this.slider('in-brush', 'out-brush', (v) => {
      this.brush = v;
      return `${v}`;
    });
  }

  private slider(inputId: string, outputId: string, apply: (v: number) => string) {
    const input = $<HTMLInputElement>(inputId);
    const output = $<HTMLOutputElement>(outputId);
    const update = () => (output.textContent = apply(Number(input.value)));
    input.addEventListener('input', update);
    update();
  }

  private setTool(tool: Tool) {
    this.tool = tool;
    for (const b of $('tools').querySelectorAll<HTMLButtonElement>('button')) {
      b.classList.toggle('active', b.dataset.tool === tool);
    }
    $('tool-hint').textContent = {
      fire: 'Drag to aim and set speed, click for a particle at rest. Shift adds to the current wave.',
      wall: 'Paint walls. Right-drag erases.',
      erase: 'Drag to erase walls and barriers.',
    }[tool];
  }

  private setColorMode(mode: ColorMode) {
    this.colorMode = mode;
    for (const b of $('color-mode').querySelectorAll<HTMLButtonElement>('button')) {
      b.classList.toggle('active', b.dataset.mode === mode);
    }
  }

  private setPaused(paused: boolean) {
    this.paused = paused;
    $('btn-play').textContent = paused ? 'Play' : 'Pause';
    $('stat-paused').hidden = !paused;
  }

  private setAbsorb(absorb: boolean) {
    this.sim.absorb = absorb;
    $<HTMLInputElement>('in-absorb').checked = absorb;
    this.sim.markPotentialDirty();
  }

  private async measure() {
    const hit = await this.sim.measure(Math.max(3, this.sim.n / 170));
    if (hit) this.pings.push({ ...hit, start: performance.now() });
  }

  private bindKeys() {
    window.addEventListener('keydown', (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.target instanceof HTMLInputElement && e.target.type !== 'range' && e.target.type !== 'checkbox') return;
      const key = e.key.toLowerCase();
      const actions: Record<string, () => void> = {
        ' ': () => this.setPaused(!this.paused),
        r: () => this.reset(),
        m: () => this.measure(),
        c: () => this.clearWalls(),
        p: () => this.setColorMode(this.colorMode === 'phase' ? 'probability' : 'phase'),
        s: () => (this.snapshotRequested = true),
        '1': () => this.setTool('fire'),
        '2': () => this.setTool('wall'),
        '3': () => this.setTool('erase'),
      };
      const action = actions[key];
      if (action) {
        e.preventDefault();
        action();
      }
    });
  }

  private toGrid(e: PointerEvent) {
    const rect = this.overlay.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * this.sim.n,
      y: ((e.clientY - rect.top) / rect.height) * this.sim.n,
    };
  }

  private bindPointer() {
    const el = this.overlay;
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    el.addEventListener('pointerdown', (e) => {
      el.setPointerCapture(e.pointerId);
      const p = this.toGrid(e);
      const erase = this.tool === 'erase' || (this.tool === 'wall' && e.button === 2);
      if (this.tool === 'fire' && e.button === 0) {
        this.drag = { x0: p.x, y0: p.y, x: p.x, y: p.y, shift: e.shiftKey };
      } else if (this.tool !== 'fire' || e.button === 2) {
        this.painting = { ...p, erase: erase || (this.tool === 'fire' && e.button === 2) };
        this.paint(p.x, p.y, p.x, p.y, this.painting.erase);
      }
    });
    el.addEventListener('pointermove', (e) => {
      const p = this.toGrid(e);
      this.pointer = p;
      if (this.drag) {
        this.drag.x = p.x;
        this.drag.y = p.y;
      }
      if (this.painting) {
        this.paint(this.painting.x, this.painting.y, p.x, p.y, this.painting.erase);
        this.painting.x = p.x;
        this.painting.y = p.y;
      }
    });
    const end = (e: PointerEvent) => {
      if (this.drag) {
        const d = this.drag;
        let kx = (d.x - d.x0) * DRAG_TO_MOMENTUM;
        let ky = (d.y - d.y0) * DRAG_TO_MOMENTUM;
        const k = Math.hypot(kx, ky);
        if (k > MAX_MOMENTUM) {
          kx *= MAX_MOMENTUM / k;
          ky *= MAX_MOMENTUM / k;
        }
        const sigma = this.sigma * (this.sim.n / 512);
        this.sim.addPacket({ x: d.x0, y: d.y0, kx, ky, sx: sigma, sy: sigma }, !(d.shift || e.shiftKey));
        this.drag = null;
      }
      this.painting = null;
    };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    el.addEventListener('pointerleave', () => (this.pointer = null));
  }

  private paint(x0: number, y0: number, x1: number, y1: number, erase: boolean) {
    const { n, potential, wallV } = this.sim;
    const r = this.brush * (n / 512);
    const steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / (r * 0.5)));
    for (let s = 0; s <= steps; s++) {
      const cx = x0 + ((x1 - x0) * s) / steps;
      const cy = y0 + ((y1 - y0) * s) / steps;
      const xmin = Math.max(0, Math.floor(cx - r));
      const xmax = Math.min(n - 1, Math.ceil(cx + r));
      const ymin = Math.max(0, Math.floor(cy - r));
      const ymax = Math.min(n - 1, Math.ceil(cy + r));
      for (let y = ymin; y <= ymax; y++) {
        for (let x = xmin; x <= xmax; x++) {
          if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) potential[y * n + x] = erase ? 0 : wallV;
        }
      }
    }
    this.sim.markPotentialDirty();
  }

  private observeSize() {
    const stage = $('stage');
    const viewport = $('viewport');
    const resize = () => {
      const style = getComputedStyle(stage);
      const padX = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
      const padY = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
      const narrow = window.innerWidth <= 820;
      const w = stage.clientWidth - padX;
      const h = narrow ? w : stage.clientHeight - padY;
      const size = Math.max(200, Math.floor(Math.min(w, h)));
      viewport.style.width = viewport.style.height = `${size}px`;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      for (const canvas of [this.simCanvas, this.overlay]) {
        canvas.width = Math.round(size * dpr);
        canvas.height = Math.round(size * dpr);
      }
    };
    new ResizeObserver(resize).observe(stage);
    window.addEventListener('resize', resize);
    resize();
  }

  private tick = () => {
    if (this.stopped) return;
    this.frameUpdate();
    requestAnimationFrame(this.tick);
  };

  private frameUpdate() {
    const { sim } = this;
    sim.sync();
    const encoder = sim.device.createCommandEncoder();
    if (!this.paused) sim.step(encoder, this.speed);

    // Auto-exposure, eased so it doesn't flicker.
    const s = 16 * (sim.n / 512);
    const reference = 1 / (2 * Math.PI * s * s); // peak density of a fresh default-sized packet
    const target = Math.max(this.peak, reference * 0.01);
    this.smoothedPeak += (target - this.smoothedPeak) * 0.08;
    this.renderer.draw(encoder, (this.exposure * 3.5) / this.smoothedPeak, this.colorMode);
    sim.device.queue.submit([encoder.finish()]);

    if (this.snapshotRequested) {
      this.snapshotRequested = false;
      this.saveSnapshot();
    }

    if (++this.frame % 6 === 0) {
      sim.readStats().then((stats) => {
        if (!stats) return;
        this.peak = stats.peak;
        $('stat-norm').textContent = `${(stats.norm * 100).toFixed(1)}%`;
      });
    }
    $('stat-time').textContent = sim.time.toFixed(0);

    this.drawOverlay();
  }

  private saveSnapshot() {
    this.simCanvas.toBlob((blob) => {
      if (!blob) return;
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `wavefront-${this.scene.id}.png`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }, 'image/png');
  }

  private drawOverlay() {
    const { ctx, overlay } = this;
    const scale = overlay.width / this.sim.n;
    const dpr = overlay.width / overlay.getBoundingClientRect().width || 1;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (this.drag) {
      const d = this.drag;
      const x0 = d.x0 * scale;
      const y0 = d.y0 * scale;
      const x1 = d.x * scale;
      const y1 = d.y * scale;
      const sigma = this.sigma * (this.sim.n / 512) * scale;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
      ctx.lineWidth = 2 * dpr;
      ctx.beginPath();
      ctx.arc(x0, y0, sigma * 2, 0, Math.PI * 2);
      ctx.setLineDash([4 * dpr, 4 * dpr]);
      ctx.stroke();
      ctx.setLineDash([]);
      const len = Math.hypot(x1 - x0, y1 - y0);
      if (len > 4 * dpr) {
        const ang = Math.atan2(y1 - y0, x1 - x0);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - 12 * dpr * Math.cos(ang - 0.4), y1 - 12 * dpr * Math.sin(ang - 0.4));
        ctx.lineTo(x1 - 12 * dpr * Math.cos(ang + 0.4), y1 - 12 * dpr * Math.sin(ang + 0.4));
        ctx.closePath();
        ctx.fill();
        const k = Math.min(Math.hypot(d.x - d.x0, d.y - d.y0) * DRAG_TO_MOMENTUM, MAX_MOMENTUM);
        ctx.font = `${12 * dpr}px ui-monospace, Menlo, monospace`;
        ctx.fillText(`k = ${k.toFixed(2)}`, x1 + 10 * dpr, y1 - 10 * dpr);
      }
    }

    if (this.pointer && (this.tool !== 'fire' || this.painting)) {
      const r = this.brush * (this.sim.n / 512) * scale;
      ctx.strokeStyle = this.tool === 'erase' || this.painting?.erase ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.7)';
      ctx.lineWidth = 1.5 * dpr;
      ctx.beginPath();
      ctx.arc(this.pointer.x * scale, this.pointer.y * scale, r, 0, Math.PI * 2);
      ctx.stroke();
    }

    const now = performance.now();
    this.pings = this.pings.filter((p) => now - p.start < 1200);
    for (const p of this.pings) {
      const t = (now - p.start) / 1200;
      ctx.strokeStyle = `rgba(255, 255, 255, ${1 - t})`;
      ctx.lineWidth = 2 * dpr;
      ctx.beginPath();
      ctx.arc(p.x * scale, p.y * scale, (6 + t * 40) * dpr, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}

init().catch((err) => fail(String(err?.message ?? err)));
