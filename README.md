# GPU Calculator Pro

A static, client-side GPU requirements calculator for AI/ML workloads. Configure model parameters and instantly see VRAM needs, performance estimates, and recommended GPUs. No backend required.

## Features
- Model selection including Qwen 3 series and DeepSeek variants
- Quantization options (`fp32`, `fp16`, `bf16`, `fp8`, `int8`)
- Context length, concurrency, and batch controls
- Memory breakdown chart (weights, KV cache, activation, overhead)
- Performance estimates (tokens/sec, bandwidth utilization, per-request speed)
- Recommended GPUs with a view toggle (Cards ↔ Table)

## Project Structure
- `index.html` — UI markup and Tailwind-based layout
- `main.js` — Calculation engine and interactive behaviors
- `resources/` — Static assets (images)
- `design.md`, `outline.md`, `interaction.md`, `calculation_engine.md` — Notes and docs

## Getting Started
Requirements: modern desktop or mobile browser. No build step needed.

Local preview (Python):

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/` in your browser.

## Usage Tips
- Use the left panel to configure model, quantization, context, concurrency, and batch size.
- The “Recommended GPUs” panel supports a sliding toggle to switch between a card view and a table view.
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
- Add or adjust model specs in `main.js` under the `models` map.
- Tweak GPU database entries in `main.js` (`this.gpus`) to reflect hardware you care about.
- Adjust layout and styling directly in `index.html` (Tailwind classes are configured inline).

## GPU Dropdown Data Merge

The calculator’s “GPU Model” dropdown merges two sources at runtime to keep things flexible:

- Hardcoded options in `calculator.html` remain as-is (e.g., A100, H100).
- Missing models from `data/GPUs.json` are appended on page load by `selfhost-llm.js`.

Implementation details:
- `selfhost-llm.js` → `augmentCalculatorGPUOptionsFromCatalog()`
  - Fetches `data/GPUs.json` and appends any GPUs not already present.
  - Dedupes using both the option `value` slug and word-boundary matching to avoid collisions (e.g., `H20` vs `H200`).
  - Groups appended options under existing optgroups based on vendor and name (e.g., `NVIDIA RTX 40 Series`, `NVIDIA Professional`, `AMD Radeon`).
  - Sets `data-vram` from `memory_gb` and `data-bandwidth` from `memory_bandwidth_tbps` (converted to GB/s).
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