# AGENTS.md

## Project Overview

GPU Calculator Pro — a **static, client-side** GPU requirements calculator for AI/ML workloads (LLM inference). No backend, no build step, no package manager. Everything runs in the browser; libraries load from public CDNs (both pages: Tailwind; index.html also loads ECharts + anime.js).

**Shared style**: both pages follow the calculator page's (SelfHostLLM) visual language — dark `deep-charcoal` background, `rgba(17,24,39,0.75)` glass panels, blue accent `#3b82f6` (Tailwind color `accent` on index.html; `--color-accent` in `selfhost-llm.css`), `gradient-text` blue→emerald. Keep new UI consistent with these tokens.

## Architecture / File Map

Two coexisting front-ends share the `data/` catalogs:

- `index.html` + `main.js` — "GPU Calculator Pro" page. Single `GPUCalculator` class (`main.js:4`) holds state, the hardcoded `models`/`gpus` maps, and drives live calculations, ECharts memory breakdown, recommended-GPU cards/table views, and GPU/LLM catalog explorer pages.
- `calculator.html` + `selfhost-llm.js` + `selfhost-llm.css` (+ `styles/selfhost-extras.css`) — "SelfHostLLM" calculator page. Plain functions (no class); manages URL query-param state (`updateURL`/`loadFromURL`) and augments the GPU `<select>` at runtime from `data/GPUs.json` (`augmentCalculatorGPUOptionsFromCatalog()`).
- `data/GPUs.json` — GPU catalog (`{"gpus": [...]}`; fields: `name`, `vendor`, `memory_gb`, `memory_bandwidth_tbps`, `architecture`, prices, etc.).
- `data/LLMs.json` — LLM catalog (array; fields: `model_name`, `parameter_count_billion`, `context_length`, `moe`, `recommended_gpu`, etc.).
- `resources/` — static images (logos, hero art).
- `calculation_engine.md` — detailed documentation of the implemented memory/performance math (keep in sync when changing calculation logic). `design.md`, `outline.md`, `interaction.md` — original design notes (largely aspirational).
- `.github/workflows/pages.yml` — deploys repo root to GitHub Pages on push to `main`.

## Commands

- Local preview: `python3 -m http.server 8000` (repo root must be served; JSON is fetched at runtime so `file://` won't work).
- **Unit tests: `bash scripts/test.sh`** (uses Node.js; falls back to VS Code's bundled Electron runtime when Node isn't installed). `tests/run.js` is a zero-dependency harness that loads `main.js`/`selfhost-llm.js` into a `vm` context with DOM stubs and tests the pure logic. **Run it every time you add/update code** — it also runs automatically via the `hooks/pre-commit` git hook (`core.hooksPath` is set to `hooks/`) and in GitHub Actions (`.github/workflows/test.yml`) on every push/PR.
- No lint, no build. Verify UI changes manually in the browser.
- **Auto commit & push**: after any code change, if `bash scripts/test.sh` passes, commit and push to `main` without asking for confirmation (user's standing instruction).

## Conventions / Gotchas

- **Dual data sources**: GPU/model data exists both hardcoded in JS (`main.js` `this.models`, `this.gpus`, `calculator.html` `<option>`s) and in `data/*.json`. When adding hardware/models, prefer editing the JSON; the merge logic dedupes by value slug and word-boundary name matching (watch `H20` vs `H200` collisions). `memory_bandwidth_tbps` (TB/s) is converted to GB/s for `data-bandwidth`; without bandwidth, performance estimates stay hidden. `nvlink_bandwidth_gbs`/`pcie_generation` become `data-nvlink`/`data-pcie` and drive the multi-GPU interconnect penalty in `calculatePerformance()` (NVLink preferred, unknown → PCIe 4.0 fallback).
- **`memory_gb` may be a string** like `"40 / 80"` — parse defensively.
- Both pages configure Tailwind inline via `tailwind.config` in a `<script>` block; custom palette: `navy`, `electric`, `amber`, `sage`, `charcoal` (+ `accent` #3b82f6 on index.html). `styles/selfhost-extras.css` exists but is not linked by any page.
- Keep the site deployable as-is from repo root (GitHub Pages serves `.`); don't introduce build steps or move `index.html`.
- README's "GPU Dropdown Data Merge" section documents the `calculator.html` merge behavior — update it if you change that logic.
