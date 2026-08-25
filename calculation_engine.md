# GPU Calculator — Calculation Engine

This document describes the calculation logic **as implemented** in the two
coexisting front-ends. Both share the `data/` catalogs but have separate
engines:

- **SelfHostLLM page** (`calculator.html` + `selfhost-llm.js`) — VRAM fit,
  concurrency, tokens/sec with interconnect-aware multi-GPU scaling, and the
  performance scenario table.
- **GPU Calculator Pro page** (`index.html` + `main.js`) — memory breakdown
  (weights / KV cache / activation / overhead), recommended GPUs, and catalog
  explorers.

All constants referenced below live at the top of `selfhost-llm.js` or inline
in `main.js`.

---

## 1. Shared foundations

### 1.1 Quantization

The calculator page expresses precision as a **ratio relative to FP16** (FP16
= 1.0), used to scale the model's FP16 memory footprint:

| Ratio | Precision | Notes |
|------:|-----------|-------|
| 2 | FP32 | |
| 1 | FP16 / BF16 | baseline |
| 0.53125 | Q8_0 | GGUF, 8.5 bits/weight |
| 0.5 | INT8 / FP8 | |
| 0.41015625 | Q6_K | GGUF, 6.5625 bits/weight |
| 0.34375 | Q5_K_M | GGUF, 5.5 bits/weight |
| 0.3 | Q4_K_M | GGUF, 4.8 bits/weight |
| 0.25 | INT4 / FP4 / MXFP4 | |
| 0.21484375 | Q3_K_M | GGUF, 3.4375 bits/weight |
| 0.16015625 | Q2_K | GGUF, 2.5625 bits/weight |
| 0.125 | INT2 | |

The index page uses **bytes per value** instead
(`main.js:bytesPerValueForPrecision`): fp32 = 4, fp16/bf16 = 2, fp8/int8 = 1,
int4/fp4 = 0.5, int2 = 0.25.

KV cache is always stored at 2 bytes/value (fp16/bf16,
`KV_BYTES_PER_VALUE`), regardless of weight quantization.

### 1.2 Model architecture resolution

KV cache and tensor-parallel sizing need `num_layers` and `hidden_size`.
Resolution order (`selfhost-llm.js:computeKVCacheGB` / `resolveModelArchitecture`):

1. **Catalog match** — the selected `<option>` is matched against
   `data/LLMs.json` (`findLLMConfigFromSelectedOption`: normalized direct
   match, then substring match).
2. **Parameter-count heuristic** (`heuristicArchitecture`):

| Params | Layers | Hidden |
|--------|-------:|-------:|
| ≤ 4B | 28 | 3072 |
| ≤ 8B | 32 | 4096 |
| ≤ 15B | 40 | 4096 |
| ≤ 35B | 64 | 5120 |
| ≤ 75B | 80 | 8192 |
| ≤ 130B | 88 | 12288 |
| larger / unknown | 60 / 32 | 7168 / 4096 |

The index page has an equivalent fallback ladder inside
`estimateLLMMemoryForPrecision` (32/40/80/96/120 layers, 4096–12288 hidden).

---

## 2. SelfHostLLM engine (`selfhost-llm.js`)

### 2.1 Model memory

`resolveModelMemory()` supports three input methods:

- **Preset** — `data-memory` attribute on the `<option>` (FP16 GB, i.e.
  2 GB per billion params). MoE presets also carry `data-active-memory`.
- **Parameters** — `params_B × GB_PER_BILLION_PARAMS` (2 GB/B, FP16).
- **Direct memory** — user-entered GB.

MoE handling: `deriveModelParamsB()` derives full vs. active parameters from
the catalog (`moe.num_experts` / `moe.active_experts`, `activeB = fullB ×
active/total`) or from label suffixes like `235B-A22B`. When **MoE offloading**
is enabled, the active-expert footprint is used for VRAM fit and performance;
otherwise the total footprint is used.

### 2.2 VRAM budget (`calculate()`)

```
totalVRAM            = gpuCount × vramPerGpu
adjustedModelMemory  = modelMemory × quantizationRatio
kvCachePerRequest    = ctx × num_layers × 2 (K+V) × hidden_size × 2 bytes
                       × (1 + kvCacheOverhead%)
availableMemory      = totalVRAM − systemOverhead − adjustedModelMemory
maxConcurrentRequests = availableMemory / kvCachePerRequest
```

`systemOverhead` (GB) and `kvCacheOverhead` (%) are user inputs.

### 2.3 Tokens/sec (`calculatePerformance()`)

Decode is memory-bandwidth-bound, so throughput scales with bandwidth divided
by the bytes read per token (the adjusted model memory):

```
baseSpeed = (totalBandwidthGBps / modelMemoryGB)
            × sizeEfficiency
            × quantKernelEff
            × contextImpact
            × multiGpuScaling
tokensPerSecond = baseSpeed × PERF_CONSERVATIVE_FACTOR (0.6)
```

**Model-size efficiency** (larger models achieve a smaller fraction of peak
bandwidth): ≤7B → 0.85, ≤30B → 0.7, ≤70B → 0.5, >70B → 0.3.

**Quantization kernel efficiency** — the footprint reduction is already
credited via `modelMemory`; these factors model dequant overhead eating part
of the bandwidth win. Calibrated against TinyChat/AWQ batch-1 measurements
(MIT HAN Lab): INT4 nets ~2–2.9× FP16, not the naive 4×.

| Ratio range | Factor | Applies to |
|-------------|-------:|------------|
| ≤ 0.25 | 0.6 | INT4 / FP4 / MXFP4 |
| ≤ 0.30 | 0.65 | ~5-bit (Q4_K_M, Q5_K_M) |
| ≤ 0.50 | 0.9 | INT8 / FP8 / Q8_0 / Q6_K |
| ≤ 0.75 | 0.95 | ~12-bit |
| > 0.75 | 1.0 | FP16 / BF16 / FP32 |

**Context impact** (attention cost grows with context): <8K → 1.0,
≥8K → 0.85, ≥32K → 0.6, ≥128K → 0.3.

**Multi-GPU scaling**:

```
multiGpuScaling = (0.85 + 0.15 / gpuCount) × commFactor
```

`commFactor` comes from `tensorParallelCommFactor()` — tensor-parallel decode
interleaves memory-bound weight reads with 2 all-reduces per layer
(attention + MLP), each moving a hidden-size fp16 activation per token:

```
commBytes/token = 2 × num_layers × hidden_size × 2 bytes
tMem  = modelMemory / totalBandwidth          # memory-bound time per token
tComm = commBytes / interconnectBW + latency  # interconnect time per token
commFactor = tMem / (tMem + tComm)            # 1.0 = no penalty
```

**Interconnect resolution** (`getInterconnect()`): NVLink preferred
(`data-nvlink`, GB/s), then PCIe by generation (`data-pcie`: 3.0 → 13,
4.0 → 25, 5.0 → 50, 6.0 → 100 GB/s, x16 at ~80% of nominal), unknown → PCIe
4.0. Per-all-reduce latency: 30 µs PCIe / 3 µs NVLink — latency dominates for
the small per-token messages, which is why PCIe multi-GPU loses ~20–30% to
NVLink (calibrated against GigaGPU and arXiv 2503.06433 measurements).

**GPU bandwidth** (`getGPUBandwidth()`): reads `data-bandwidth` (GB/s) from
the selected `<option>` first — populated from `data/GPUs.json`
(`memory_bandwidth_tbps × 1024`) by
`augmentCalculatorGPUOptionsFromCatalog()` — then falls back to a static map
(RTX 4090: 1008, A100-80: 2000, H100: 3000, H200: 4915, MI300X: 5325, …).
Returns 0 (performance section hidden) when bandwidth is unknown.

**Ratings**: ≥100 tps Excellent, ≥50 Good, ≥25 Moderate, ≥10 Slow.

### 2.4 Scenario table

`buildPerformanceScenarioTable()` sweeps the five standard context tiers
(8K/16K/32K/64K/128K) × GPU counts around the current selection (±5, extended
until at least 10 viable rows) using the same memory/performance math, and
exports to CSV (`buildScenarioCSV`).

---

## 3. GPU Calculator Pro engine (`main.js`)

### 3.1 Memory breakdown (`estimateLLMMemoryForPrecision()`)

```
weightsGB    = params × bytesPerValue / 2^30
kvCacheGB    = 2 × layers × hidden × ctx × concurrency × bytesPerValue / 2^30
               × (1 + kvOverheadPercent/100)
activationGB = batch × ctx × hidden × bytesPerValue × actFactor / 2^30
               # actFactor = 1.2 when hidden ≥ 8192 or layers ≥ 80, else 1.0
totalGB      = (weights + kvCache + activation) × 1.3
```

The 1.3 multiplier covers framework/runtime overhead (known limitation: it is
hardcoded, not driven by the system-overhead input). Layers/hidden come from
the catalog when available, else the size ladder in §1.2.

### 3.2 Tokens/sec (`estimateTokensPerSecondForPair()`)

Two paths:

1. **Reference scaling** — when the model carries a measured throughput on a
   reference GPU: `refTps × (gpuBandwidth / refBandwidth)`.
2. **Heuristic** — `bandwidthGBps / (totalMemoryGB × efficiency)` with
   precision-dependent efficiency (`efficiencyFactorForPrecision`):
   int8/fp8 → 0.85, int4/fp4/int2 → 0.9, fp16/bf16/fp32 → 0.7.

Both clamp to ≥ 1 tok/s.

### 3.3 GPU recommendations

`getGPURecommendations()` filters the catalog (GPUs whose VRAM fits the total
with headroom), estimates per-GPU throughput via §3.2, and renders cards or a
sortable table. Bandwidth parsing (`getBandwidthGBps`) accepts
`memory_bandwidth_tbps` (TB/s × 1024), legacy `bandwidth_tbps`, or raw GB/s;
`getMemoryGB` parses numbers and strings like `"40 / 80"` (takes the max).

---

## 4. Catalogs and validation

- `data/GPUs.json` — 34 GPUs; `name`, `vendor`, `memory_gb`,
  `memory_bandwidth_tbps`, `architecture`, `nvlink_bandwidth_gbs`,
  `pcie_generation`, prices, etc.
- `data/LLMs.json` — 36 models; `model_name`, `parameter_count_billion`,
  `num_layers`, `hidden_size`, `context_length`, `moe`
  (`enabled`/`num_experts`/`active_experts`), `precision_supported`,
  `quantization_types`, `recommended_gpu`, etc.

`tests/run.js` validates both catalogs (unique names, required fields,
expected 2025–2026 entries) and includes **benchmark-validation tests**: the
performance model is asserted against published batch-1 decode measurements
(TinyChat/AWQ on A100/RTX 4090, TensorRT-LLM, GigaGPU on H100, Oracle OCI
4×A100 NVLink) within documented tolerances.

## 5. Known limitations (preserved behaviors)

- The 1.3 total-memory overhead in `main.js` is hardcoded rather than using
  the system-overhead input.
- The performance model estimates **batch-1 decode**; prefill and large-batch
  serving are out of scope.
- Throughput heuristics ignore attention-variant details (GQA/MQA/MLA) except
  where catalog `num_layers`/`hidden_size` capture them.
