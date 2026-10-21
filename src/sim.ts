import fftSource from './shaders/fft.wgsl?raw';
import multiplySource from './shaders/multiply.wgsl?raw';
import operatorsSource from './shaders/operators.wgsl?raw';
import packetSource from './shaders/packet.wgsl?raw';
import normSource from './shaders/norm.wgsl?raw';

// Positions and widths in grid cells, momentum in radians per cell.
export interface Packet {
  x: number;
  y: number;
  kx: number;
  ky: number;
  sx: number;
  sy: number;
}

const NORM_GROUPS = 256;

// Strang split-step Fourier integrator for i dψ/dt = (-½∇² + V)ψ:
//   ψ <- e^{-iV dt/2} IFFT(e^{-ik² dt/2} FFT(e^{-iV dt/2} ψ))
export class Simulation {
  readonly n: number;
  readonly dt = 0.5;
  readonly wallV = 8;
  readonly potential: Float32Array;

  time = 0;
  absorb = true;

  readonly device: GPUDevice;
  readonly psi: GPUBuffer;
  readonly potentialBuffer: GPUBuffer;
  private readonly scratch: GPUBuffer;
  private readonly kinetic: GPUBuffer;
  private readonly potentialFactor: GPUBuffer;

  private readonly fftPipeline: GPUComputePipeline;
  private readonly multiplyPipeline: GPUComputePipeline;
  private readonly buildPotentialPipeline: GPUComputePipeline;
  private readonly buildKineticPipeline: GPUComputePipeline;
  private readonly packetPipeline: GPUComputePipeline;
  private readonly normPipeline: GPUComputePipeline;

  private readonly forwardPasses: GPUBindGroup[] = [];
  private readonly inversePasses: GPUBindGroup[] = [];
  private readonly applyPotential: GPUBindGroup;
  private readonly applyKinetic: GPUBindGroup;
  private readonly operatorsGroup: GPUBindGroup;
  private readonly operatorsUniform: GPUBuffer;
  private readonly packetGroup: GPUBindGroup;
  private readonly packetUniform: GPUBuffer;
  private readonly normGroup: GPUBindGroup;
  private readonly normPartials: GPUBuffer;
  private readonly normReadback: GPUBuffer;
  private readonly psiReadback: GPUBuffer;
  private normPending = false;
  private psiPending = false;
  private potentialDirty = true;

  constructor(device: GPUDevice, n: number) {
    if (n & (n - 1)) throw new Error('Grid size must be a power of two');
    this.device = device;
    this.n = n;
    this.potential = new Float32Array(n * n);

    const cells = n * n;
    const complexBytes = cells * 8;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.psi = device.createBuffer({ label: 'psi', size: complexBytes, usage: storage });
    this.scratch = device.createBuffer({ label: 'fft scratch', size: complexBytes, usage: storage });
    this.kinetic = device.createBuffer({ label: 'kinetic factor', size: complexBytes, usage: storage });
    this.potentialFactor = device.createBuffer({ label: 'potential factor', size: complexBytes, usage: storage });
    this.potentialBuffer = device.createBuffer({ label: 'potential', size: cells * 4, usage: storage });

    const constants = { N: n };
    const compute = (label: string, code: string, layout: GPUBindGroupLayout, entryPoint = 'main') =>
      device.createComputePipeline({
        label,
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: device.createShaderModule({ label, code }), entryPoint, constants },
      });

    const C = GPUShaderStage.COMPUTE;
    const uniform = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: C, buffer: { type: 'uniform' } });
    const read = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: C, buffer: { type: 'read-only-storage' } });
    const write = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: C, buffer: { type: 'storage' } });
    const layout = (...entries: GPUBindGroupLayoutEntry[]) => device.createBindGroupLayout({ entries });

    // 2·log2(N) passes is even, so the result always lands back in psi.
    const fftLayout = layout(uniform(0), read(1), write(2));
    this.fftPipeline = compute('fft', fftSource, fftLayout);
    const log2n = Math.log2(n);
    for (const sign of [-1, 1]) {
      const passes = sign < 0 ? this.forwardPasses : this.inversePasses;
      let q = 0;
      for (let axis = 0; axis < 2; axis++) {
        for (let s = 0; s < log2n; s++, q++) {
          const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          const data = new ArrayBuffer(16);
          new Uint32Array(data, 0, 2).set([1 << s, axis]);
          new Float32Array(data, 8, 1)[0] = sign;
          device.queue.writeBuffer(params, 0, data);
          const [src, dst] = q % 2 === 0 ? [this.psi, this.scratch] : [this.scratch, this.psi];
          passes.push(
            device.createBindGroup({
              layout: fftLayout,
              entries: [
                { binding: 0, resource: { buffer: params } },
                { binding: 1, resource: { buffer: src } },
                { binding: 2, resource: { buffer: dst } },
              ],
            }),
          );
        }
      }
    }

    const multiplyLayout = layout(write(0), read(1));
    this.multiplyPipeline = compute('multiply', multiplySource, multiplyLayout);
    const multiplyGroup = (factor: GPUBuffer) =>
      device.createBindGroup({
        layout: multiplyLayout,
        entries: [
          { binding: 0, resource: { buffer: this.psi } },
          { binding: 1, resource: { buffer: factor } },
        ],
      });
    this.applyPotential = multiplyGroup(this.potentialFactor);
    this.applyKinetic = multiplyGroup(this.kinetic);

    const operatorsLayout = layout(uniform(0), read(1), write(2), write(3));
    this.buildPotentialPipeline = compute('build potential', operatorsSource, operatorsLayout, 'buildPotential');
    this.buildKineticPipeline = compute('build kinetic', operatorsSource, operatorsLayout, 'buildKinetic');
    this.operatorsUniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.operatorsGroup = device.createBindGroup({
      layout: operatorsLayout,
      entries: [
        { binding: 0, resource: { buffer: this.operatorsUniform } },
        { binding: 1, resource: { buffer: this.potentialBuffer } },
        { binding: 2, resource: { buffer: this.potentialFactor } },
        { binding: 3, resource: { buffer: this.kinetic } },
      ],
    });

    const packetLayout = layout(uniform(0), write(1));
    this.packetPipeline = compute('packet', packetSource, packetLayout);
    this.packetUniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.packetGroup = device.createBindGroup({
      layout: packetLayout,
      entries: [
        { binding: 0, resource: { buffer: this.packetUniform } },
        { binding: 1, resource: { buffer: this.psi } },
      ],
    });

    const normLayout = layout(read(0), write(1));
    this.normPipeline = compute('norm', normSource, normLayout);
    this.normPartials = device.createBuffer({ size: NORM_GROUPS * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.normReadback = device.createBuffer({ size: NORM_GROUPS * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.normGroup = device.createBindGroup({
      layout: normLayout,
      entries: [
        { binding: 0, resource: { buffer: this.psi } },
        { binding: 1, resource: { buffer: this.normPartials } },
      ],
    });
    this.psiReadback = device.createBuffer({ size: complexBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    this.rebuildOperators(true);
  }

  markPotentialDirty() {
    this.potentialDirty = true;
  }

  private rebuildOperators(includeKinetic: boolean) {
    const { device, n } = this;
    device.queue.writeBuffer(this.potentialBuffer, 0, this.potential);
    const data = new ArrayBuffer(32);
    const f = new Float32Array(data);
    const u = new Uint32Array(data);
    f[0] = this.dt;
    u[1] = this.absorb ? 1 : 0;
    f[2] = Math.max(16, n * 0.06); // absorbing layer width
    f[3] = 0.05; // damping per half-step at the outer edge
    f[4] = this.wallV;
    device.queue.writeBuffer(this.operatorsUniform, 0, data);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setBindGroup(0, this.operatorsGroup);
    pass.setPipeline(this.buildPotentialPipeline);
    pass.dispatchWorkgroups(Math.ceil(n / 64), n);
    if (includeKinetic) {
      pass.setPipeline(this.buildKineticPipeline);
      pass.dispatchWorkgroups(Math.ceil(n / 64), n);
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
    this.potentialDirty = false;
  }

  addPacket(p: Packet, replace: boolean) {
    const { device, n } = this;
    // Unit norm for a lone packet; a superposition splits it evenly.
    let amp = 1 / Math.sqrt(2 * Math.PI * p.sx * p.sy);
    let keep = 0;
    if (!replace) {
      keep = Math.SQRT1_2;
      amp *= Math.SQRT1_2;
    } else {
      this.time = 0;
    }
    device.queue.writeBuffer(this.packetUniform, 0, new Float32Array([p.x, p.y, p.kx, p.ky, p.sx, p.sy, amp, keep]));
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.packetPipeline);
    pass.setBindGroup(0, this.packetGroup);
    pass.dispatchWorkgroups(Math.ceil(n / 64), n);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  sync() {
    if (this.potentialDirty) this.rebuildOperators(false);
  }

  step(encoder: GPUCommandEncoder, steps: number) {
    this.sync();
    const n = this.n;
    const pass = encoder.beginComputePass({ label: 'split-step' });
    const multiply = (group: GPUBindGroup) => {
      pass.setPipeline(this.multiplyPipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(n / 64), n);
    };
    const fft = (groups: GPUBindGroup[]) => {
      pass.setPipeline(this.fftPipeline);
      for (const group of groups) {
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(Math.ceil(n / 2 / 64), n);
      }
    };
    for (let s = 0; s < steps; s++) {
      multiply(this.applyPotential);
      fft(this.forwardPasses);
      multiply(this.applyKinetic);
      fft(this.inversePasses);
      multiply(this.applyPotential);
    }
    pass.end();
    this.time += steps * this.dt;
  }

  async readStats(): Promise<{ norm: number; peak: number } | null> {
    if (this.normPending) return null;
    this.normPending = true;
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.normPipeline);
    pass.setBindGroup(0, this.normGroup);
    pass.dispatchWorkgroups(NORM_GROUPS);
    pass.end();
    encoder.copyBufferToBuffer(this.normPartials, 0, this.normReadback, 0, NORM_GROUPS * 8);
    this.device.queue.submit([encoder.finish()]);
    try {
      await this.normReadback.mapAsync(GPUMapMode.READ);
      const partials = new Float32Array(this.normReadback.getMappedRange());
      let norm = 0;
      let peak = 0;
      for (let i = 0; i < partials.length; i += 2) {
        norm += partials[i];
        peak = Math.max(peak, partials[i + 1]);
      }
      this.normReadback.unmap();
      return { norm, peak };
    } catch {
      // The map was rejected (for example, the device was lost); the buffer is not
      // mapped in that case, and this read instance returns without data.
      return null;
    } finally {
      this.normPending = false;
    }
  }

  async readPsi(): Promise<Float32Array | null> {
    if (this.psiPending) return null;
    this.psiPending = true;
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.psi, 0, this.psiReadback, 0, this.psi.size);
    this.device.queue.submit([encoder.finish()]);
    try {
      await this.psiReadback.mapAsync(GPUMapMode.READ);
      const copy = new Float32Array(this.psiReadback.getMappedRange()).slice();
      this.psiReadback.unmap();
      return copy;
    } catch {
      return null;
    } finally {
      this.psiPending = false;
    }
  }

  // Sample a position from |ψ|² and collapse onto a narrow packet there.
  async measure(width: number): Promise<{ x: number; y: number } | null> {
    const psi = await this.readPsi();
    if (!psi) return null;
    let total = 0;
    for (let i = 0; i < psi.length; i += 2) total += psi[i] * psi[i] + psi[i + 1] * psi[i + 1];
    if (!(total > 1e-9)) return null;

    let target = Math.random() * total;
    let cell = 0;
    for (let i = 0; i < psi.length; i += 2, cell++) {
      target -= psi[i] * psi[i] + psi[i + 1] * psi[i + 1];
      if (target <= 0) break;
    }
    const x = cell % this.n;
    const y = Math.floor(cell / this.n);
    const time = this.time;
    this.addPacket({ x, y, kx: 0, ky: 0, sx: width, sy: width }, true);
    this.time = time;
    return { x, y };
  }
}
