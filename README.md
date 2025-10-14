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

## License
Unlicensed by default. Add a `LICENSE` file if you plan to open-source under specific terms.