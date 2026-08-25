# GPU Calculator Pro

A static, client-side GPU requirements calculator for AI/ML workloads (LLM inference). Configure model parameters and instantly see VRAM needs, performance estimates, and recommended GPUs. No backend, no build step — libraries load from public CDNs.

## Pages
- **`index.html` — GPU Calculator Pro**: model/GPU configuration with a live memory breakdown chart (weights, KV cache, activation, overhead), performance estimates, and recommended GPUs (Cards ↔ Table).
- **`calculator.html` — SelfHostLLM**: VRAM fit and concurrency calculator with MoE offloading support, interconnect-aware multi-GPU performance estimates, a performance scenario table (context × GPU count sweep with CSV export), and shareable URL state.

## Features
- 36 open-source models (Qwen 3, DeepSeek, GLM, Kimi, Mistral, Llama, …) and 34 GPUs (RTX 40 series through H200/B200, MI300X, …) in `data/*.json`
- Extended quantization: FP32, FP16/BF16, INT8/FP8, INT4/FP4/MXFP4, INT2, plus GGUF Q8_0–Q2_K
- Context length, concurrency, and batch controls; MoE active/total parameter handling
- Memory breakdown chart and bandwidth-based tokens/sec estimates, benchmark-validated against published TinyChat/AWQ, TensorRT-LLM, GigaGPU, and OCI measurements
- Multi-GPU scaling with NVLink/PCIe interconnect penalty
- **GPU Explorer** and **Open Source Models** catalogs: compact grouped cards (by vendor/organization), sortable table view, live search, and sort dropdowns

## Project Structure
- `index.html` + `main.js` — GPU Calculator Pro page (`GPUCalculator` class: state, catalogs, charts, recommendations)
- `calculator.html` + `selfhost-llm.js` + `selfhost-llm.css` — SelfHostLLM page (plain functions, URL query-param state)
- `data/GPUs.json`, `data/LLMs.json` — hardware and model catalogs (edit these to add entries)
- `tests/run.js` + `scripts/test.sh` — zero-dependency unit test harness
- `hooks/pre-commit` — runs the tests on every commit
- `resources/` — static assets (images)
- `calculation_engine.md` — detailed documentation of the memory/performance math; `design.md`, `outline.md`, `interaction.md` — original design notes

## Getting Started
Requirements: modern desktop or mobile browser. No build step needed.

Local preview (Python) — the repo root must be served; the JSON catalogs are fetched at runtime so `file://` won't work:

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/` (GPU Calculator Pro) or `http://localhost:8000/calculator.html` (SelfHostLLM) in your browser.

## Testing

Unit tests live in `tests/run.js` (zero-dependency harness; loads the browser scripts into a `vm` context with DOM stubs). Run them every time you add or update code:

```bash
bash scripts/test.sh   # uses Node.js, or VS Code's Electron runtime as a fallback
```

Tests also run automatically on every commit (pre-commit hook in `hooks/`, enabled via `git config core.hooksPath hooks`) and on every push/PR in GitHub Actions (`.github/workflows/test.yml`).

The suite includes validation against published batch-1 decode benchmarks (TinyChat/AWQ, NVIDIA TensorRT-LLM, GigaGPU, Oracle OCI): each case asserts the calculator's estimate lands within a documented tolerance of the measured tokens/sec.

## Usage Tips
- **SelfHostLLM page** (`calculator.html`): pick a model preset (or enter parameters/memory directly), choose quantization, context, GPU and count; results, the scenario table, and the shareable URL update live.
- **GPU Calculator Pro page** (`index.html`): the catalog sections support search, sort dropdowns, vendor/architecture/memory filters, and a Cards ↔ Table toggle; card view groups by vendor/organization.
- The parameter count and all estimates update live as you change inputs.

## Deployment (GitHub Pages)
1. Create a GitHub repository and push this project.
2. In GitHub, go to `Settings` → `Pages`.
3. Under “Build and deployment”, set:
   - Source: `Deploy from a branch`
   - Branch: `main` (or your default branch)
   - Folder: `/ (root)`
4. Save. Your site will be published at `https://<your-username>.github.io/<repo-name>/`.

Notes:
- Ensure `index.html` is at the repo root.
- External libraries are loaded via HTTPS CDNs.
- Commit the `resources/` folder so images appear.

## Customization
- **Add or adjust models**: edit `data/LLMs.json` (preferred). Fields include `model_name`, `parameter_count_billion`, `num_layers`, `hidden_size`, `context_length`, `moe`, `precision_supported`, `quantization_types`, `recommended_gpu`. Providing `num_layers`/`hidden_size` makes KV-cache estimates exact instead of heuristic.
- **Add or adjust GPUs**: edit `data/GPUs.json` — see the "GPU Dropdown Data Merge" section below for required fields.
- **Legacy hardcoded data**: `main.js` still contains `this.models`/`this.gpus` maps used as fallbacks by the index page; prefer the JSON catalogs for anything new.
- **Calculation logic**: see `calculation_engine.md` for the documented formulas and constants before tweaking them.
- Adjust layout and styling directly in `index.html` / `calculator.html` (Tailwind classes are configured inline).

## GPU Dropdown Data Merge

The calculator’s “GPU Model” dropdown merges two sources at runtime to keep things flexible:

- Hardcoded options in `calculator.html` remain as-is (e.g., A100, H100).
- Missing models from `data/GPUs.json` are appended on page load by `selfhost-llm.js`.

Implementation details:
- `selfhost-llm.js` → `augmentCalculatorGPUOptionsFromCatalog()`
  - Fetches `data/GPUs.json` and appends any GPUs not already present.
  - Dedupes using both the option `value` slug and word-boundary matching to avoid collisions (e.g., `H20` vs `H200`).
  - Groups appended options under existing optgroups based on vendor and name (e.g., `NVIDIA RTX 40 Series`, `NVIDIA Professional`, `AMD Instinct (Datacenter)`, `AMD Radeon`).
  - Sets `data-vram` from `memory_gb` and `data-bandwidth` from `memory_bandwidth_tbps` (converted to GB/s).
  - Sets `data-nvlink` from `nvlink_bandwidth_gbs` and `data-pcie` from `pcie_generation`; these drive the multi-GPU interconnect (PCIe/NVLink) communication penalty in `calculatePerformance()`.
- `window.onload` awaits the augmentation so merged options are available before initialization and URL preselection.
- `selfhost-llm.js` → `getGPUBandwidth(gpuModel)`
  - Reads `data-bandwidth` from the selected option first.
  - Falls back to a static map for known models.

How to add GPUs:
- Edit `data/GPUs.json` and include at minimum:
  - `name` (e.g., `NVIDIA H20`), `vendor`, `memory_gb` (number or string like `"40 / 80"`).
  - `memory_bandwidth_tbps` (number or numeric string in TB/s). This enables performance estimates for the new GPU.
- Alternatively, add an `<option>` directly in `calculator.html`. The merge logic avoids duplicates if the same GPU exists in JSON.

Notes:
- If bandwidth is missing for a new GPU, performance estimates may remain hidden until `data-bandwidth` is available (via JSON or static map).
- You can adjust optgroup mapping or insertion order in `augmentCalculatorGPUOptionsFromCatalog()` if you want specific placement.

## License
Unlicensed by default. Add a `LICENSE` file if you plan to open-source under specific terms.