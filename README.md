# Wavefront

A 2D quantum mechanics sandbox in the browser, solves the time-dependent Schrödinger equation on a 512×512 grid in real time using WebGPU compute shaders.

**[Try it in your browser](https://ylfeid.github.io/wavefront/)** (needs WebGPU)

Scenes: double slit, single slit, tunneling, harmonic trap, stadium billiard, scattering, crystal lattice, empty sandbox.

![Scattering, lattice diffraction, stadium billiard, single slit](docs/scenes.png)

## Controls

| Input | Action |
| --- | --- |
| Drag (Fire) | Launch a packet; the drag sets direction and momentum |
| Click (Fire) | Place a packet at rest |
| Shift + drag | Add a packet on top of the current wavefunction |
| Drag (Wall / Erase) | Paint or erase walls; right-drag erases |
| Space | Pause / play |
| M | Measure position |
| R | Reset scene |
| C | Clear walls |
| 1 / 2 / 3 | Fire / Wall / Erase |
| P | Toggle probability / phase colouring |
| S | Save screenshot |

URL parameters: `?scene=<id>` opens a scene directly, `?n=256|512|1024` sets the grid size.

## Method

Units are ħ = m = 1 with a grid spacing of 1, so the equation is

$$
i\,\frac{\partial \psi}{\partial t} = \left(-\tfrac{1}{2}\nabla^2 + V\right)\psi
$$

Each step uses the Strang split-step Fourier method:

$$
\psi(t + \Delta t) \approx e^{-iV\Delta t/2}\;\mathcal{F}^{-1}\!\left[e^{-i k^2 \Delta t/2}\;\mathcal{F}\!\left[e^{-iV\Delta t/2}\,\psi(t)\right]\right]
$$

All three factors are pure phases, so the scheme is unitary and stable for any Δt.

Implementation notes:

- The FFT is a radix-2 Stockham transform (`src/shaders/fft.wgsl`), run as log₂N passes over rows and then log₂N over columns, ping-ponging between two buffers.
- Both phase factors are precomputed into buffers and rebuilt only when the potential changes. The inverse FFT's 1/N² is folded into the kinetic factor.
- The FFT makes the domain periodic, so a damping layer at the border absorbs outgoing waves. With absorbing edges off, the border becomes a hard wall.
- A reduction shader computes the total probability and peak density each few frames. The peak drives auto-exposure.
- The fragment shader reads ψ directly from the simulation's storage buffer.

Scenes are defined in `src/presets.ts` as a potential plus an initial Gaussian packet. In the harmonic trap the packet is a coherent state (width 1/√(2ω), speed ωr), so it orbits without spreading. The tunneling barrier (V = 0.6) is higher than the packet's mean energy (about 0.55).

## Building

Requirements:

- [Node.js](https://nodejs.org/) 22.12 or newer (includes npm)
- A browser with WebGPU to run it: a recent Chrome or Edge, Safari 26+, or a recent Firefox

Clone and install dependencies:

```bash
git clone https://github.com/ylfeid/wavefront.git
cd wavefront
npm install
```

Start the dev server, then open the URL it prints (usually http://localhost:5173). Edits reload automatically.

```bash
npm run dev
```

Make a production build. This type-checks the code and writes the site to `dist/`:

```bash
npm run build
```

Serve that build locally to check it (usually at http://localhost:4173):

```bash
npm run preview
```

`dist/` is plain static files with relative paths, so it can be uploaded to any static host, including under a subpath. Run `npm run typecheck` to type-check without building.

## Deploying

`.github/workflows/deploy.yml` builds and publishes to GitHub Pages on pushes to `main`. Enable it under Settings → Pages → Source: GitHub Actions.

## License

MIT
