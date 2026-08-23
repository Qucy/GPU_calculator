// selfhost-llm.js — logic for the SelfHostLLM calculator page (calculator.html only).
// Plain top-level functions (no modules); the functions listed below are called
// from inline handlers in calculator.html and must remain global:
//   updateURL, loadFromURL, updateGPUSpecs, updateModelInputMethod,
//   updateModelSelection, updateContextInputMethod, calculate,
//   closeShareDialog, closeExplanationDialog, closePerformanceExplanation,
//   showHowCalculated, showPerformanceExplanation,
//   copyScenarioTable, downloadScenarioTable

// ============================================================================
// Constants
// ============================================================================

// --- Performance heuristics ---
// Model size efficiency factor (larger models are less efficient)
const EFFICIENCY_SMALL_MODEL = 0.85;   // <= 7B params
const EFFICIENCY_MEDIUM_MODEL = 0.7;   // <= 30B params
const EFFICIENCY_LARGE_MODEL = 0.5;    // <= 70B params
const EFFICIENCY_XLARGE_MODEL = 0.3;   // > 70B params

// Quantization kernel efficiency (fraction of the bandwidth win actually
// realized; the footprint reduction is credited separately via model size).
// Calibrated on TinyChat/AWQ batch=1 data (MIT HAN Lab, 2024): INT4 nets
// ~2-2.9x vs FP16, INT8 ~1.5-2x — not the naive 4x/2x bandwidth ratios.
const QUANT_KERNEL_EFF_INT4 = 0.6;   // quantization <= 0.25 (INT4/MXFP4) -> ~2.4x FP16
const QUANT_KERNEL_EFF_5BIT = 0.65;  // quantization <= 0.30
const QUANT_KERNEL_EFF_INT8 = 0.9;   // quantization <= 0.50 (INT8/FP8) -> ~1.8x FP16
const QUANT_KERNEL_EFF_12BIT = 0.95; // quantization <= 0.75

// Context length thresholds (tokens) and their speed impact
const CONTEXT_THRESHOLD_MEDIUM = 8192;    // 8K
const CONTEXT_THRESHOLD_LARGE = 32768;    // 32K
const CONTEXT_THRESHOLD_XL = 131072;      // 128K
const CONTEXT_IMPACT_MEDIUM = 0.85;  // >= 8K
const CONTEXT_IMPACT_LARGE = 0.6;    // >= 32K
const CONTEXT_IMPACT_XL = 0.3;       // >= 128K

// Multi-GPU scaling (not perfect linear scaling): base + spread / gpuCount
const MULTI_GPU_SCALING_BASE = 0.85;
const MULTI_GPU_SCALING_SPREAD = 0.15;

// --- Interconnect (multi-GPU tensor-parallel communication) ---
// Effective per-GPU interconnect bandwidth (GB/s) for tensor-parallel
// all-reduce traffic. PCIe: x16 at ~80% of nominal throughput.
const PCIE_BANDWIDTH_GBPS = { '3.0': 13, '4.0': 25, '5.0': 50, '6.0': 100 };
// Conservative fallback when the GPU's interconnect is unknown (PCIe 4.0 x16)
const DEFAULT_INTERCONNECT_GBPS = PCIE_BANDWIDTH_GBPS['4.0'];
// Tensor-parallel decode does 2 all-reduces per layer (attention + MLP outputs),
// each moving a hidden-size fp16/bf16 activation vector per token.
const TP_ALLREDUCES_PER_LAYER = 2;
const TP_ACTIVATION_BYTES = 2;
// Per-all-reduce latency (microseconds): dominates for the small per-token
// messages, especially over PCIe. PCIe value includes ring/protocol overhead;
// calibrated so 2x RTX 4090 (PCIe 4.0) on 70B INT4 shows ~22% comm penalty —
// consistent with the measured 25-30% (GigaGPU) and 8xA100 PCIe reaching
// ~60-70% of NVLink throughput (Seesaw, arXiv 2503.06433).
const TP_ALLREDUCE_LATENCY_US = { pcie: 30, nvlink: 3 };

const PERF_CONSERVATIVE_FACTOR = 0.6;  // conservative estimate for datacenter GPUs

// Performance rating thresholds (tokens/sec)
const RATING_EXCELLENT_TPS = 100;
const RATING_GOOD_TPS = 50;
const RATING_MODERATE_TPS = 25;
const RATING_SLOW_TPS = 10;

// --- Memory/model heuristics ---
const GB_PER_BILLION_PARAMS = 2;  // rough estimate: 2GB per billion parameters in FP16
const KV_BYTES_PER_VALUE = 2;     // KV stored typically in fp16/bf16 regardless of weight quantization

// --- Scenario table ---
const SCENARIO_CONTEXTS = [8192, 16384, 32768, 65536, 131072];  // 8K, 16K, 32K, 64K, 128K
const SCENARIO_GPU_COUNT_WINDOW = 5;     // explore ±5 around the current GPU count
const SCENARIO_MIN_ROWS = 10;            // guarantee at least this many recommendations
const SCENARIO_EXTEND_SAFETY_CAP = 200;  // loop safety cap while extending GPU counts
const SCENARIO_NUMERIC_SORT_KEYS = new Set(['gpuCount', 'context', 'maxConcurrent', 'tokensPerSecNum', 'genTimeNum']);

// ============================================================================
// DOM / state helpers
// ============================================================================

const el = (id) => document.getElementById(id);

// Shared mutable state (replaces the old window.__* globals; internal to this file)
const state = {
    moe: {
        isSelected: false,        // selected preset is a MoE model
        offloadingEnabled: false, // MoE offloading toggle
        activeMemory: null,       // GB, active experts only
        totalMemory: null         // GB, full model
    },
    scenario: {
        rows: [],                              // currently rendered rows (used by CSV export)
        baseRows: [],                          // rows before filtering/sorting
        sortState: { key: null, dir: 'asc' },
        sortingBound: false,                   // idempotent one-time binding flags
        filtersBound: false
    }
};

// ============================================================================
// URL state (query-param sync)
// ============================================================================

// URL parameter management
function updateURL() {
    const params = new URLSearchParams();

    // GPU Configuration
    const gpuType = el('gpu-type').value;
    if (gpuType) params.set('gpu', gpuType);
    params.set('gpu_count', el('gpu-count').value);
    params.set('sys_overhead', el('system-overhead').value);

    // Model Configuration
    const modelInputType = document.querySelector('input[name="model-input-type"]:checked').value;
    params.set('model_type', modelInputType);

    if (modelInputType === 'preset') {
        params.set('model', el('model-preset').value);
    } else if (modelInputType === 'parameters') {
        params.set('model_params', el('model-parameters').value);
    } else if (modelInputType === 'memory') {
        params.set('model_memory', el('model-memory-input').value);
    }

    // Quantization
    params.set('quant', el('quantization').value);

    // Context Configuration
    const contextInputType = document.querySelector('input[name="context-input-type"]:checked').value;
    params.set('context_type', contextInputType);

    if (contextInputType === 'preset') {
        params.set('context', el('context-preset').value);
    } else {
        params.set('context_custom', el('context-custom').value);
    }

    // KV Cache
    params.set('kv_cache', el('kv-cache-overhead').value);

    // Update URL without reloading page
    const newURL = window.location.pathname + '?' + params.toString();
    window.history.replaceState({}, '', newURL);
}

function loadFromURL() {
    const params = new URLSearchParams(window.location.search);

    // GPU Configuration
    if (params.has('gpu')) {
        el('gpu-type').value = params.get('gpu');
        updateGPUSpecs();
    }
    if (params.has('gpu_count')) {
        el('gpu-count').value = params.get('gpu_count');
    }
    if (params.has('sys_overhead')) {
        el('system-overhead').value = params.get('sys_overhead');
    }

    // Model Configuration
    if (params.has('model_type')) {
        const modelType = params.get('model_type');
        document.querySelector(`input[name="model-input-type"][value="${modelType}"]`).checked = true;
        updateModelInputMethod();

        if (modelType === 'preset' && params.has('model')) {
            el('model-preset').value = params.get('model');
        } else if (modelType === 'parameters' && params.has('model_params')) {
            el('model-parameters').value = params.get('model_params');
        } else if (modelType === 'memory' && params.has('model_memory')) {
            el('model-memory-input').value = params.get('model_memory');
        }
    }

    // Quantization
    if (params.has('quant')) {
        el('quantization').value = params.get('quant');
    }

    // Context Configuration
    if (params.has('context_type')) {
        const contextType = params.get('context_type');
        document.querySelector(`input[name="context-input-type"][value="${contextType}"]`).checked = true;
        updateContextInputMethod();

        if (contextType === 'preset' && params.has('context')) {
            el('context-preset').value = params.get('context');
        } else if (contextType === 'custom' && params.has('context_custom')) {
            el('context-custom').value = params.get('context_custom');
        }
    }

    // KV Cache
    if (params.has('kv_cache')) {
        el('kv-cache-overhead').value = params.get('kv_cache');
    }

    // Calculate after loading
    calculate();
}

// ============================================================================
// Bandwidth lookup
// ============================================================================

// Get GPU memory bandwidth based on model
function getGPUBandwidth(gpuModel) {
    if (!gpuModel) return 0;

    // Prefer bandwidth from the selected option's data attribute if available.
    // NOTE: this reads the currently SELECTED option regardless of gpuModel.
    const select = el('gpu-type');
    if (select && select.selectedIndex >= 0) {
        const selectedOption = select.options[select.selectedIndex];
        const bwAttr = selectedOption ? selectedOption.getAttribute('data-bandwidth') : null;
        if (bwAttr && !isNaN(Number(bwAttr))) {
            return Number(bwAttr);
        }
    }

    const bandwidthMap = {
        // RTX 40 Series
        'rtx4090': 1008,
        'rtx4080': 736,
        'rtx4070ti': 504,
        'rtx4070': 504,
        'rtx4060ti': 288,
        'rtx4060ti8': 288,

        // RTX 30 Series
        'rtx3090ti': 936,
        'rtx3090': 936,
        'rtx3080ti': 912,
        'rtx3080': 760,

        // NVIDIA Professional
        'a100': 1600,  // 40GB variant
        'a100-80': 2000,  // 80GB variant
        'h100': 3000,
        'v100': 900,
        'rtx6000': 960,  // RTX 6000 Ada
        'l40s': 864,
        'l40': 864,
        'l4': 300,
        't4': 320,
        'h200': 4915,
        'h20': 4096,

        // AMD Radeon
        'rx7900xtx': 960,
        'rx7900xt': 800,

        // AMD Instinct
        'mi300x': 5325,
        'mi250x': 3200
    };

    return bandwidthMap[gpuModel] || 0;
}

// ============================================================================
// Performance math
// ============================================================================

// Resolve the effective interconnect for the currently selected GPU option.
// Prefers NVLink (data-nvlink, GB/s) over PCIe (data-pcie, generation string).
// Unknown interconnect falls back to PCIe 4.0 x16 (worst common case).
function getInterconnect() {
    const select = el('gpu-type');
    const opt = (select && select.selectedIndex >= 0) ? select.options[select.selectedIndex] : null;
    const nvlink = opt ? parseFloat(opt.getAttribute('data-nvlink')) : NaN;
    if (nvlink > 0) return { bandwidth: nvlink, kind: 'nvlink' };
    const pcieGen = opt ? opt.getAttribute('data-pcie') : null;
    const pcieBw = PCIE_BANDWIDTH_GBPS[String(pcieGen)];
    if (pcieBw) return { bandwidth: pcieBw, kind: 'pcie' };
    return { bandwidth: DEFAULT_INTERCONNECT_GBPS, kind: 'pcie' };
}

// Resolve model architecture (layers, hidden size) for TP communication sizing:
// catalog match first, then the parameter-count heuristic.
function resolveModelArchitecture(modelParams) {
    const modelSelect = el('model-preset');
    const opt = (modelSelect && modelSelect.selectedIndex >= 0) ? modelSelect.options[modelSelect.selectedIndex] : null;
    const catalog = opt ? findLLMConfigFromSelectedOption(opt) : null;
    if (catalog && catalog.num_layers && catalog.hidden_size) {
        return { num_layers: catalog.num_layers, hidden_size: catalog.hidden_size };
    }
    const label = (opt && opt.textContent) ? opt.textContent : `${modelParams}B`;
    return heuristicArchitecture({ textContent: label });
}

// Multi-GPU communication overhead: tensor-parallel decode interleaves memory-
// bound weight reads with per-layer all-reduces over the interconnect.
// Returns a factor in (0, 1] (1 = no comm penalty).
function tensorParallelCommFactor(modelMemory, totalBandwidthGBps, gpuCount, arch, interconnect) {
    if (gpuCount <= 1) return 1.0;
    const commBytesPerToken = TP_ALLREDUCES_PER_LAYER * arch.num_layers * arch.hidden_size * TP_ACTIVATION_BYTES;
    const latencyS = (TP_ALLREDUCE_LATENCY_US[interconnect.kind] / 1e6) * TP_ALLREDUCES_PER_LAYER * arch.num_layers;
    const tMem = modelMemory / totalBandwidthGBps;                                  // s/token, memory-bound part
    const tComm = (commBytesPerToken / (1024 ** 3)) / interconnect.bandwidth + latencyS; // s/token, comm part
    return tMem / (tMem + tComm);
}

// Calculate performance estimate
function calculatePerformance(modelMemory, quantization, contextLength, gpuModel, gpuCount) {
    const bandwidth = getGPUBandwidth(gpuModel) * gpuCount;
    if (!bandwidth) return null;

    // Get model parameters from preset if available
    let modelParams = 7;
    const modelInputType = document.querySelector('input[name="model-input-type"]:checked').value;
    if (modelInputType === 'preset') {
        const modelSelect = el('model-preset');
        modelParams = parseFloat(modelSelect.value) || 7;
    } else if (modelInputType === 'parameters') {
        modelParams = parseFloat(el('model-parameters').value) || 7;
    }

    // Model size efficiency factor (larger models are less efficient)
    let efficiency;
    if (modelParams <= 7) {
        efficiency = EFFICIENCY_SMALL_MODEL;
    } else if (modelParams <= 30) {
        efficiency = EFFICIENCY_MEDIUM_MODEL;
    } else if (modelParams <= 70) {
        efficiency = EFFICIENCY_LARGE_MODEL;
    } else {
        efficiency = EFFICIENCY_XLARGE_MODEL;
    }

    // Quantization kernel efficiency. The smaller footprint is already credited
    // via modelMemory (e.g. INT4 = 1/4 the bytes of FP16); these factors model
    // the dequant/compute overhead that eats part of that bandwidth win.
    // Calibrated against TinyChat/AWQ batch=1 measurements (MIT HAN Lab):
    // Llama-2-7B/Llama-3-8B INT4 run ~2-2.9x FP16 (not the naive 4x), INT8 ~1.5-2x.
    let quantKernelEff = 1.0;
    if (quantization <= 0.25) {
        quantKernelEff = QUANT_KERNEL_EFF_INT4;
    } else if (quantization <= 0.3) {
        quantKernelEff = QUANT_KERNEL_EFF_5BIT;
    } else if (quantization <= 0.5) {
        quantKernelEff = QUANT_KERNEL_EFF_INT8;
    } else if (quantization <= 0.75) {
        quantKernelEff = QUANT_KERNEL_EFF_12BIT;
    }

    // Context length impact
    let contextImpact = 1.0;
    if (contextLength >= CONTEXT_THRESHOLD_XL) {
        contextImpact = CONTEXT_IMPACT_XL;
    } else if (contextLength >= CONTEXT_THRESHOLD_LARGE) {
        contextImpact = CONTEXT_IMPACT_LARGE;
    } else if (contextLength >= CONTEXT_THRESHOLD_MEDIUM) {
        contextImpact = CONTEXT_IMPACT_MEDIUM;
    }

    // Multi-GPU scaling: fixed heuristic for imperfect sharding, times a
    // communication factor derived from the GPU's interconnect (PCIe/NVLink).
    let multiGpuScaling = 1.0;
    let commFactor = 1.0;
    let interconnect = null;
    if (gpuCount > 1) {
        interconnect = getInterconnect();
        commFactor = tensorParallelCommFactor(modelMemory, bandwidth, gpuCount,
            resolveModelArchitecture(modelParams), interconnect);
        multiGpuScaling = (MULTI_GPU_SCALING_BASE + (MULTI_GPU_SCALING_SPREAD / gpuCount)) * commFactor;
    }

    // Calculate tokens per second
    // Formula: (bandwidth / model_memory_gb) * efficiency * quant_kernel_eff * context_impact * scaling
    const baseSpeed = (bandwidth / modelMemory) * efficiency * quantKernelEff * contextImpact * multiGpuScaling;

    // Apply realistic scaling factor
    const tokensPerSecond = baseSpeed * PERF_CONSERVATIVE_FACTOR; // Conservative estimate for datacenter GPUs

    return {
        tokensPerSecond: tokensPerSecond,
        bandwidth: bandwidth,
        efficiency: efficiency,
        quantKernelEff: quantKernelEff,
        contextImpact: contextImpact,
        multiGpuScaling: multiGpuScaling,
        commFactor: commFactor,          // 1.0 = no communication penalty
        interconnect: interconnect       // { bandwidth (GB/s), kind: 'pcie'|'nvlink' } or null (single GPU)
    };
}

// ============================================================================
// GPU catalog augmentation (extend the GPU <select> from data/GPUs.json)
// ============================================================================

async function augmentCalculatorGPUOptionsFromCatalog() {
    const select = el('gpu-type');
    if (!select) return;

    // Helper to parse memory_gb values like number or "40 / 80" (takes the MAX)
    const parseMemoryGB = (v) => {
        if (v == null) return null;
        if (typeof v === 'number' && !isNaN(v)) return Math.round(Number(v));
        const s = String(v);
        const nums = s.match(/[\d.]+/g);
        if (!nums || nums.length === 0) return null;
        const vals = nums.map(n => Number(n)).filter(n => !isNaN(n));
        if (vals.length === 0) return null;
        return Math.round(Math.max(...vals));
    };

    const baseName = (name) => String(name || '').replace(/^(NVIDIA|AMD|Huawei|Baidu|Alibaba|Biren)\s+/i, '').trim();
    const escapeRegex = (s) => String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const existsByBaseNameOrSlug = (base, slug) => {
        const b = String(base || '').toLowerCase();
        const s = String(slug || '').toLowerCase();
        const re = new RegExp(`(^|[^\\w])${escapeRegex(b)}([^\\w]|$)`, 'i');
        return Array.from(select.querySelectorAll('option')).some(o => {
            const val = (o.value || '').toLowerCase();
            if (val === s) return true;
            const t = (o.textContent || '').toLowerCase();
            return re.test(t);
        });
    };
    const ensureGroup = (label) => {
        const groups = Array.from(select.querySelectorAll('optgroup'));
        let group = groups.find(g => String(g.label).toLowerCase() === String(label).toLowerCase());
        if (!group) {
            group = document.createElement('optgroup');
            group.label = label;
            select.appendChild(group);
        }
        return group;
    };
    const groupLabelFor = (gpu) => {
        const vendor = String(gpu.vendor || '').trim().toLowerCase();
        const name = baseName(gpu.name || '');
        if (/^rtx\s*50/i.test(name)) return 'NVIDIA RTX 50 Series';
        if (/^rtx\s*40/i.test(name)) return 'NVIDIA RTX 40 Series';
        if (/^rtx\s*30/i.test(name)) return 'NVIDIA RTX 30 Series';
        if (vendor === 'nvidia') return 'NVIDIA Professional';
        if (vendor === 'amd') return 'AMD Radeon';
        return gpu.vendor ? gpu.vendor : 'Other Accelerators';
    };
    const slugFrom = (name) => baseName(name).toLowerCase().replace(/[^\w]+/g, '-');

    try {
        const res = await fetch('data/GPUs.json');
        if (!res.ok) return;
        const data = await res.json();
        const gpus = Array.isArray(data?.gpus) ? data.gpus : (Array.isArray(data) ? data : []);
        if (!gpus || gpus.length === 0) return;

        // Batch DOM updates per optgroup to reduce reflows
        const groupElements = new Map();
        // Collect options per group so we can SORT before appending
        const optionsByGroup = new Map();
        const getGroup = (label) => {
            let g = groupElements.get(label);
            if (!g) {
                g = ensureGroup(label);
                groupElements.set(label, g);
            }
            return g;
        };
        const addOptionToGroup = (label, opt) => {
            let list = optionsByGroup.get(label);
            if (!list) {
                list = [];
                optionsByGroup.set(label, list);
                // Ensure the group element exists ahead of time
                getGroup(label);
            }
            list.push(opt);
        };
        const parseModelNumber = (valOrText) => {
            const s = String(valOrText || '');
            const m = s.match(/(\d{3,4})/);
            return m ? parseInt(m[1], 10) : null;
        };

        gpus.forEach(gpu => {
            const base = baseName(gpu.name || '');
            if (!base) return;
            // Skip if a matching base name already exists in any option label
            const newSlug = slugFrom(gpu.name || '');
            if (existsByBaseNameOrSlug(base, newSlug)) return;
            const memGB = parseMemoryGB(gpu.memory_gb);
            if (!memGB || isNaN(memGB) || memGB <= 0) return;
            const groupLabel = groupLabelFor(gpu);
            const opt = document.createElement('option');
            opt.value = newSlug;
            opt.setAttribute('data-vram', String(memGB));
            // Attach memory bandwidth in GB/s if available
            const tbpsRaw = gpu.memory_bandwidth_tbps;
            let tbps = null;
            if (typeof tbpsRaw === 'number' && !isNaN(tbpsRaw)) {
                tbps = tbpsRaw;
            } else if (tbpsRaw != null) {
                const n = parseFloat(String(tbpsRaw).replace(/[^\d.]/g, ''));
                tbps = isNaN(n) ? null : n;
            }
            if (tbps && tbps > 0) {
                const gbps = Math.round(tbps * 1024);
                opt.setAttribute('data-bandwidth', String(gbps));
            }
            // Attach interconnect info for multi-GPU communication modeling
            const nvlink = Number(gpu.nvlink_bandwidth_gbs);
            if (nvlink > 0) opt.setAttribute('data-nvlink', String(nvlink));
            if (gpu.pcie_generation != null && gpu.pcie_generation !== '') {
                opt.setAttribute('data-pcie', String(gpu.pcie_generation));
            }
            opt.textContent = `${base} (${memGB}GB VRAM)`;
            addOptionToGroup(groupLabel, opt);
        });

        // Sort and flush options to their optgroups in one pass
        optionsByGroup.forEach((list, label) => {
            const group = getGroup(label);
            const frag = document.createDocumentFragment();
            // For RTX series, sort by model number ascending (e.g., 3060→3090)
            const isRTXSeries = /NVIDIA RTX (30|40|50) Series/i.test(label);
            const sorted = list.slice().sort((a, b) => {
                if (isRTXSeries) {
                    const an = parseModelNumber(a.value || a.textContent);
                    const bn = parseModelNumber(b.value || b.textContent);
                    if (an != null && bn != null) return an - bn;
                }
                // Fallback lexicographic by text
                const at = String(a.textContent || '');
                const bt = String(b.textContent || '');
                return at.localeCompare(bt);
            });
            sorted.forEach(opt => frag.appendChild(opt));
            group.appendChild(frag);
        });
    } catch (e) {
        // Silently ignore catalog errors to avoid disrupting existing flow
        console.warn('GPU catalog load failed:', e?.message || e);
    }
}

// ============================================================================
// Input-method toggles (GPU/model/context UI switching)
// ============================================================================

function updateGPUSpecs() {
    const select = el('gpu-type');
    const vramInput = el('vram-per-gpu');

    if (select.value) {
        const selectedOption = select.options[select.selectedIndex];
        const vram = selectedOption.getAttribute('data-vram');
        vramInput.value = vram;
    } else {
        vramInput.value = '';
    }

    calculate();
    updateURL();
}

function updateModelInputMethod() {
    const inputType = document.querySelector('input[name="model-input-type"]:checked').value;

    // Toggle visibility using hidden class
    el('model-preset-group').classList.toggle('hidden', inputType !== 'preset');
    el('model-parameters-group').classList.toggle('hidden', inputType !== 'parameters');
    el('model-memory-group').classList.toggle('hidden', inputType !== 'memory');

    calculate();
}

function updateModelSelection() {
    const modelSelect = el('model-preset');
    // Do not auto-change quantization on model selection; keep user's choice
    calculate();
    updateURL();
}

function updateContextInputMethod() {
    const inputType = document.querySelector('input[name="context-input-type"]:checked').value;

    // Toggle visibility using hidden class
    el('context-preset-group').classList.toggle('hidden', inputType !== 'preset');
    el('context-custom-group').classList.toggle('hidden', inputType !== 'custom');

    calculate();
}

// ============================================================================
// Master calculate
// ============================================================================

// Read all raw form inputs used by calculate()
function readInputs() {
    return {
        gpuCount: parseInt(el('gpu-count').value) || 1,
        vramPerGpu: parseFloat(el('vram-per-gpu').value) || 0,
        systemOverhead: parseFloat(el('system-overhead').value) || 2,
        modelInputType: document.querySelector('input[name="model-input-type"]:checked').value,
        contextInputType: document.querySelector('input[name="context-input-type"]:checked').value,
        quantization: parseFloat(el('quantization').value),
        kvCacheOverhead: parseFloat(el('kv-cache-overhead').value) / 100,
        gpuType: el('gpu-type').value
    };
}

// Get model memory based on input method; caches MoE context in state.moe
function resolveModelMemory(modelInputType) {
    let modelMemory;

    if (modelInputType === 'preset') {
        const modelSelect = el('model-preset');
        const selectedOption = modelSelect.options[modelSelect.selectedIndex];
        const activeMemoryAttr = selectedOption.getAttribute('data-active-memory');
        const totalMemoryAttr = selectedOption.getAttribute('data-memory');
        const isMoE = !!activeMemoryAttr;
        const offloadingEnabled = !!el('moe-offloading') && el('moe-offloading').checked;
        // For VRAM fit: choose active vs total based on offloading toggle
        if (isMoE) {
            modelMemory = offloadingEnabled ? parseFloat(activeMemoryAttr) : (parseFloat(totalMemoryAttr) || 14);
        } else {
            modelMemory = parseFloat(totalMemoryAttr) || 14;
        }
        // Cache MoE context for performance and notes
        state.moe.isSelected = isMoE;
        state.moe.offloadingEnabled = offloadingEnabled;
        state.moe.activeMemory = activeMemoryAttr ? parseFloat(activeMemoryAttr) : null;
        state.moe.totalMemory = totalMemoryAttr ? parseFloat(totalMemoryAttr) : null;
    } else if (modelInputType === 'parameters') {
        const paramCount = parseFloat(el('model-parameters').value) || 7;
        modelMemory = paramCount * GB_PER_BILLION_PARAMS;
    } else if (modelInputType === 'memory') {
        modelMemory = parseFloat(el('model-memory-input').value) || 14;
    }

    return modelMemory;
}

// Get context length based on input method
function resolveContextLength(contextInputType) {
    if (contextInputType === 'preset') {
        return parseInt(el('context-preset').value) || 4096;
    }
    return parseInt(el('context-custom').value) || 4096;
}

// Compute KV cache per request using the architecture-driven formula
function resolveKVCachePerRequest(contextLength, modelMemory, kvCacheOverhead) {
    const modelSelect = el('model-preset');
    const selectedOption = modelSelect ? modelSelect.options[modelSelect.selectedIndex] : null;
    if (selectedOption) {
        return computeKVCacheGB(contextLength, selectedOption, KV_BYTES_PER_VALUE, kvCacheOverhead);
    }
    // Fallback if not using preset
    return computeKVCacheGB(contextLength, { textContent: `${modelMemory / GB_PER_BILLION_PARAMS}B` }, KV_BYTES_PER_VALUE, kvCacheOverhead);
}

// Write the memory/results metrics and any capability warnings to the DOM
function renderResults({ totalVRAM, adjustedModelMemory, kvCachePerRequest, availableMemory, maxConcurrentRequests, effectiveContext }) {
    el('total-vram').textContent = totalVRAM.toFixed(1) + ' GB';
    el('model-memory').textContent = adjustedModelMemory.toFixed(1) + ' GB';
    el('kv-cache-memory').textContent = kvCachePerRequest.toFixed(2) + ' GB';
    el('available-memory').textContent = Math.max(0, availableMemory).toFixed(1) + ' GB';
    el('concurrent-requests').textContent = Math.max(0, maxConcurrentRequests).toFixed(2);
    el('effective-context').textContent = effectiveContext.toLocaleString() + ' tokens';

    // Show warnings for insufficient capability
    const warningsDiv = el('warnings');
    warningsDiv.innerHTML = '';
    const cannotServeSingleRequest = maxConcurrentRequests < 1;
    const modelExceedsVRAM = availableMemory < 0;

    if (modelExceedsVRAM || cannotServeSingleRequest) {
        const suggestions = [
            'Reduce model memory via stronger quantization (e.g., INT4/FP8)',
            'Choose a smaller parameter model or MoE with lower active memory',
            'Lower the context length to shrink KV cache per request',
            'Add more GPUs or use a GPU with more VRAM'
        ];
        const title = '⚠ Current GPU does not meet the minimum requirements to serve this model';
        const actionsHTML = `<div class="warning-actions">${suggestions.map(s => `• ${s}`).join('<br>')}</div>`;
        warningsDiv.innerHTML = `<div class="warning"><div class="warning-title">${title}</div>${actionsHTML}</div>`;
    }
}

// Update performance metrics: tokens/sec, generation time, rating
function renderPerformanceMetrics(perf, tokensPerSecNum) {
    const tokensPerSec = perf.tokensPerSecond.toFixed(2);
    el('tokens-per-second').textContent = `${tokensPerSec} tokens/sec`;

    // Generation time for 100 tokens
    const genTime = tokensPerSecNum > 0 ? (100 / tokensPerSecNum).toFixed(1) : 'N/A';
    el('generation-time').textContent = `${genTime} seconds`;

    // Performance rating
    let rating = '';
    let ratingClass = '';
    if (tokensPerSecNum > RATING_EXCELLENT_TPS) {
        rating = '🟢 Excellent';
        ratingClass = 'excellent';
    } else if (tokensPerSecNum > RATING_GOOD_TPS) {
        rating = '🟢 Good';
        ratingClass = 'good';
    } else if (tokensPerSecNum > RATING_MODERATE_TPS) {
        rating = '🟡 Moderate';
        ratingClass = 'moderate';
    } else if (tokensPerSecNum > RATING_SLOW_TPS) {
        rating = '🟡 Slow';
        ratingClass = 'slow';
    } else {
        rating = '🔴 Very Slow';
        ratingClass = 'very-slow';
    }

    const ratingElement = el('performance-rating');
    if (ratingElement) {
        ratingElement.textContent = rating;
        ratingElement.className = `metric-value ${ratingClass}`;
    }
}

// Update the performance tips/notes block
function renderPerformanceNotes(inputs, contextLength, tokensPerSecNum, perf) {
    const notesDiv = el('performance-notes');
    if (!notesDiv) return;

    let notes = [];
    // Multi-GPU interconnect note (below MoE note)
    if (perf && inputs.gpuCount > 1 && perf.interconnect) {
        const penalty = Math.round((1 - perf.commFactor) * 100);
        const label = perf.interconnect.kind === 'nvlink'
            ? `NVLink (~${perf.interconnect.bandwidth} GB/s)`
            : `PCIe (~${perf.interconnect.bandwidth} GB/s)`;
        notes.push(`• Multi-GPU over ${label}: ~${penalty}% of throughput lost to tensor-parallel communication` +
            (perf.interconnect.kind === 'pcie' && penalty >= 15 ? ' — NVLink GPUs would reduce this penalty' : ''));
    }
    // MoE mode note at the top
    if (state.moe.isSelected) {
        const hasActive = typeof state.moe.activeMemory === 'number' && !isNaN(state.moe.activeMemory);
        const hasTotal = typeof state.moe.totalMemory === 'number' && !isNaN(state.moe.totalMemory);
        const totalGB = hasTotal ? (state.moe.totalMemory * inputs.quantization).toFixed(1) : null;
        const activeGB = hasActive ? (state.moe.activeMemory * inputs.quantization).toFixed(1) : null;
        const moeLine = state.moe.offloadingEnabled
            ? (activeGB ? `• MoE offloading ON: VRAM and performance use active experts (~${activeGB} GB)` : `• MoE offloading ON: Using active experts for calculations`)
            : (totalGB ? `• MoE offloading OFF: VRAM and performance use full model (~${totalGB} GB)` : `• MoE offloading OFF: Using full model size for calculations`);
        notes.push(moeLine);
    }

    if (tokensPerSecNum < RATING_MODERATE_TPS) {
        notes.push('• Consider stronger quantization (INT4) for better speed');
    }
    if (contextLength > CONTEXT_THRESHOLD_LARGE && tokensPerSecNum < RATING_GOOD_TPS) {
        notes.push('• Reduce context length for faster generation');
    }
    if (inputs.gpuCount === 1 && tokensPerSecNum < 30) {
        notes.push('• Consider adding more GPUs for better performance');
    }
    if (tokensPerSecNum > RATING_GOOD_TPS) {
        notes.push('• Performance should be smooth for most use cases');
    }

    const defaultNotes = [
        '• Use INT4/FP8 where acceptable to improve speed',
        '• Shorter context reduces KV cache size and boosts throughput',
        '• Higher memory bandwidth GPUs deliver more tokens/sec'
    ];

    const tips = notes.length > 0 ? notes : defaultNotes;
    notesDiv.innerHTML = `<h4>Performance Tips:</h4>${tips.join('<br>')}`;
}

// Calculate and display performance if GPU is selected
function renderPerformance(inputs, modelMemory, contextLength, availableMemory) {
    const performanceSection = el('performance-section');

    if (!(inputs.gpuType && availableMemory >= 0 && performanceSection)) {
        if (performanceSection) {
            performanceSection.style.display = 'none';
        }
        return;
    }

    // Align MoE offloading behavior: when ON, use active experts; when OFF, use total
    let perfModelMemoryBase = modelMemory;
    if (state.moe.isSelected) {
        const hasActive = typeof state.moe.activeMemory === 'number' && !isNaN(state.moe.activeMemory);
        const hasTotal = typeof state.moe.totalMemory === 'number' && !isNaN(state.moe.totalMemory);
        if (state.moe.offloadingEnabled) {
            perfModelMemoryBase = hasActive ? state.moe.activeMemory : modelMemory;
        } else {
            perfModelMemoryBase = hasTotal ? state.moe.totalMemory : modelMemory;
        }
    }
    const perfMemoryAdjusted = perfModelMemoryBase * inputs.quantization;
    const perf = calculatePerformance(perfMemoryAdjusted, inputs.quantization, contextLength, inputs.gpuType, inputs.gpuCount);

    if (!perf) {
        performanceSection.style.display = 'none';
        return;
    }

    performanceSection.style.display = 'block';
    const tokensPerSecNum = parseFloat(perf.tokensPerSecond.toFixed(2));
    renderPerformanceMetrics(perf, tokensPerSecNum);
    renderPerformanceNotes(inputs, contextLength, tokensPerSecNum, perf);
}

function calculate() {
    const inputs = readInputs();
    const modelMemory = resolveModelMemory(inputs.modelInputType);
    const contextLength = resolveContextLength(inputs.contextInputType);

    // Calculate memory requirements
    const totalVRAM = inputs.gpuCount * inputs.vramPerGpu;
    const adjustedModelMemory = modelMemory * inputs.quantization;
    const kvCachePerRequest = resolveKVCachePerRequest(contextLength, modelMemory, inputs.kvCacheOverhead);
    const availableMemory = totalVRAM - inputs.systemOverhead - adjustedModelMemory;

    // Calculate concurrent requests
    const maxConcurrentRequests = availableMemory / kvCachePerRequest;
    const effectiveContext = contextLength;

    // Update results (metrics + warnings)
    renderResults({
        totalVRAM,
        adjustedModelMemory,
        kvCachePerRequest,
        availableMemory,
        maxConcurrentRequests,
        effectiveContext
    });

    // Update performance section
    renderPerformance(inputs, modelMemory, contextLength, availableMemory);

    if (totalVRAM > 0) {
        el('results').classList.remove('hidden');
    }

    // Update URL with current configuration
    updateURL();

    // Build/update the performance scenarios table
    buildPerformanceScenarioTable();
}

// ============================================================================
// Scenario table
// ============================================================================

// Format a context length as "8K"-style short label
const toK = (n) => {
    if (n >= 1024) return `${Math.round(n / 1024)}K`;
    return String(n);
};

// Read and validate the base inputs for the scenario table.
// Returns null when the section should stay/be hidden (essential selections missing).
function readScenarioContext() {
    const section = el('scenario-table-section');
    const table = el('scenario-table');
    const tbody = table ? table.querySelector('tbody') : null;
    if (!section || !tbody) return null;

    const gpuTypeEl = el('gpu-type');
    const gpuType = gpuTypeEl ? gpuTypeEl.value : '';
    const vramPerGpu = parseFloat(el('vram-per-gpu').value) || 0;
    const systemOverhead = parseFloat(el('system-overhead').value) || 2;
    const quantization = parseFloat(el('quantization').value);
    const kvCacheOverhead = parseFloat(el('kv-cache-overhead').value) / 100;

    const modelInputType = document.querySelector('input[name="model-input-type"]:checked').value;
    const modelSelect = el('model-preset');
    const selectedModelOption = modelSelect && modelSelect.selectedIndex >= 0 ? modelSelect.options[modelSelect.selectedIndex] : null;

    // If essential selections are missing, hide the section
    if (!gpuType || vramPerGpu <= 0 || !selectedModelOption) {
        section.style.display = 'none';
        return null;
    }

    // Determine MoE and memory base consistent with offloading toggle
    const activeMemoryAttr = selectedModelOption.getAttribute('data-active-memory');
    const totalMemoryAttr = selectedModelOption.getAttribute('data-memory');
    const isMoE = !!activeMemoryAttr;
    const offloadingEnabled = !!el('moe-offloading') && el('moe-offloading').checked;

    let perfModelMemoryBase;
    if (isMoE) {
        perfModelMemoryBase = offloadingEnabled
            ? (parseFloat(activeMemoryAttr) || parseFloat(totalMemoryAttr) || 14)
            : (parseFloat(totalMemoryAttr) || parseFloat(activeMemoryAttr) || 14);
    } else {
        perfModelMemoryBase = parseFloat(totalMemoryAttr) || 14;
    }
    const perfMemoryAdjusted = perfModelMemoryBase * quantization;

    // Presentation fields
    const modelLabel = (selectedModelOption.textContent || selectedModelOption.innerText || '').trim();
    const gpuLabelOption = gpuTypeEl.options[gpuTypeEl.selectedIndex];
    const gpuLabel = (gpuLabelOption && (gpuLabelOption.textContent || gpuLabelOption.innerText) || '').trim();
    const quantLabelMap = { '1.0': 'FP16/BF16', '0.5': 'INT8/FP8', '0.25': 'INT4/MXFP4', '0.125': 'INT2' };
    const quantLabel = quantLabelMap[String(quantization)] || `${quantization}x`;

    // Compose GPU counts and contexts to explore (min context 8K)
    const currentCount = parseInt(el('gpu-count').value) || 1;

    // Include selected context length plus standard tiers
    // (kept for parity; the explored contexts are the fixed SCENARIO_CONTEXTS set)
    let selectedContextLength;
    const contextInputType = document.querySelector('input[name="context-input-type"]:checked').value;
    if (contextInputType === 'preset') {
        selectedContextLength = parseInt(el('context-preset').value) || 4096;
    } else {
        selectedContextLength = parseInt(el('context-custom').value) || 4096;
    }

    return {
        section, table, tbody,
        gpuType, vramPerGpu, systemOverhead, quantization, kvCacheOverhead,
        modelInputType, selectedModelOption, perfMemoryAdjusted,
        modelLabel, gpuLabel, quantLabel, currentCount, selectedContextLength
    };
}

// Derive full vs active parameter counts (billions) for the selected model
function deriveModelParamsB(selectedModelOption, modelInputType) {
    let fullB = null;
    let activeB = null;
    const label = (selectedModelOption.textContent || selectedModelOption.innerText || '').trim();
    // Try catalog first
    const cfg = findLLMConfigFromSelectedOption(selectedModelOption);
    if (cfg && typeof cfg.parameter_count_billion === 'number') {
        fullB = Number(cfg.parameter_count_billion);
        const moe = cfg.moe || {};
        if (moe.enabled && typeof moe.num_experts === 'number' && typeof moe.active_experts === 'number' && moe.num_experts > 0) {
            activeB = fullB * (moe.active_experts / moe.num_experts);
        }
    }
    // Fallback: parse from label (supports e.g., "235B-A22B" or "1T-A32B")
    if (fullB == null) {
        const mB = label.match(/(\d+(?:\.\d+)?)\s*B/i);
        const mT = label.match(/(\d+(?:\.\d+)?)\s*T/i);
        if (mT) fullB = Number(mT[1]) * 1000;
        else if (mB) fullB = Number(mB[1]);
    }
    if (activeB == null) {
        const a = label.match(/-A(\d+(?:\.\d+)?)B/i);
        if (a) activeB = Number(a[1]);
    }
    // If user is in parameters mode, override with user-provided value
    if (modelInputType === 'parameters') {
        const mpEl = el('model-parameters');
        const v = mpEl ? parseFloat(mpEl.value) : NaN;
        if (!isNaN(v)) {
            fullB = v;
            activeB = v;
        }
    }
    return { fullB, activeB };
}

// Format "full / active" params display (or just whichever is available)
function formatParamsDisplay(fullB, activeB) {
    const formatB = (n) => {
        if (n == null || !isFinite(n)) return '';
        const isInt = Math.abs(n - Math.round(n)) < 1e-9;
        return isInt ? String(Math.round(n)) : n.toFixed(1);
    };
    const f = formatB(fullB);
    const a = formatB(activeB);
    if (f && a && f !== a) return `${f} / ${a}`;
    return f || a || '';
}

// Build GPU counts around the current selection: ±5 window, clipped at 1
function buildScenarioGPUCounts(currentCount) {
    const gpuCounts = [];
    const startCount = Math.max(1, currentCount - SCENARIO_GPU_COUNT_WINDOW);
    const endCount = currentCount + SCENARIO_GPU_COUNT_WINDOW;
    for (let c = startCount; c <= endCount; c++) {
        gpuCounts.push(c);
    }
    // De-duplicate and sort (in case currentCount < 11 and we later expand)
    const seenCounts = new Set();
    return gpuCounts.filter(c => {
        if (seenCounts.has(c)) return false;
        seenCounts.add(c);
        return true;
    }).sort((a, b) => a - b);
}

// Build one scenario row, or null when the scenario is not runnable.
// Filter: must fit model and at least 1 request, context >= 8K, and tps > 0.
// paramsDisplay: when non-null the row carries modelParamsB; extension rows
// (below the minimum-row guarantee) intentionally omit it by passing null.
function buildScenarioRow(base, gc, ctx, paramsDisplay) {
    const totalVRAM = gc * base.vramPerGpu;
    const availableMemory = totalVRAM - base.systemOverhead - base.perfMemoryAdjusted;
    // KV cache per request for this context
    const kvPerReq = computeKVCacheGB(ctx, base.selectedModelOption, KV_BYTES_PER_VALUE, base.kvCacheOverhead);
    const maxReqRaw = availableMemory / kvPerReq;
    const maxReq = Math.max(0, maxReqRaw);
    const perf = calculatePerformance(base.perfMemoryAdjusted, base.quantization, ctx, base.gpuType, gc);
    const tpsNum = perf ? Number(perf.tokensPerSecond) : 0;
    const genTimeNum = tpsNum > 0 ? (100 / tpsNum) : Infinity;
    const runnable = (availableMemory >= kvPerReq) && (maxReqRaw >= 1) && (ctx >= CONTEXT_THRESHOLD_MEDIUM) && (tpsNum > 0);
    if (!runnable) return null;
    const row = {
        model: base.modelLabel,
        gpu: base.gpuLabel || base.gpuType,
        gpuCount: gc,
        quant: base.quantLabel,
        context: ctx,
        maxConcurrent: maxReq.toFixed(2),
        tokensPerSec: tpsNum.toFixed(2),
        tokensPerSecNum: tpsNum,
        genTime: Number.isFinite(genTimeNum) ? `${genTimeNum.toFixed(1)} s` : 'N/A',
        genTimeNum: genTimeNum
    };
    if (paramsDisplay !== null) {
        row.modelParamsB = paramsDisplay;
    }
    return row;
}

// Build all scenario rows, extending GPU counts upward until >= SCENARIO_MIN_ROWS
function buildScenarioRows(base) {
    const { fullB: modelParamsFullB, activeB: modelParamsActiveB } = deriveModelParamsB(base.selectedModelOption, base.modelInputType);
    // Use derived display for model params (full / active if available)
    const paramsDisplay = formatParamsDisplay(modelParamsFullB, modelParamsActiveB);

    const baseCounts = buildScenarioGPUCounts(base.currentCount);

    const rows = [];
    baseCounts.forEach(gc => {
        SCENARIO_CONTEXTS.forEach(ctx => {
            const row = buildScenarioRow(base, gc, ctx, paramsDisplay);
            if (row) rows.push(row);
        });
    });

    // Guarantee at least 10 recommendations by extending counts upward if needed
    if (rows.length < SCENARIO_MIN_ROWS) {
        let gc = (baseCounts.length > 0 ? baseCounts[baseCounts.length - 1] + 1 : base.currentCount + 1);
        let safety = 0;
        while (rows.length < SCENARIO_MIN_ROWS && safety < SCENARIO_EXTEND_SAFETY_CAP) {
            for (const ctx of SCENARIO_CONTEXTS) {
                // Extension rows omit modelParamsB
                const row = buildScenarioRow(base, gc, ctx, null);
                if (row) {
                    rows.push(row);
                    if (rows.length >= SCENARIO_MIN_ROWS) break;
                }
            }
            gc++;
            safety++;
        }
    }

    return rows;
}

// Render scenario rows into the table body.
// Persists rows for copy/download and keeps them in sync with sort.
function renderScenarioRows(tbody, currentRows) {
    state.scenario.rows = currentRows;
    tbody.innerHTML = currentRows.map(r => (
        `<tr>
            <td class="py-1 pr-3">${r.model}</td>
            <td class="py-1 pr-3">${r.gpu}</td>
            <td class="py-1 pr-3">${r.gpuCount}</td>
            <td class="py-1 pr-3">${r.quant}</td>
            <td class="py-1 pr-3">${toK(r.context)} tokens</td>
            <td class="py-1 pr-3">${r.maxConcurrent}</td>
            <td class="py-1 pr-3">${r.tokensPerSec}</td>
            <td class="py-1 pr-3">${r.genTime}</td>
        </tr>`
    )).join('');
}

// Filtering
function applyScenarioFilters(baseRows) {
    const ctxSelEl = el('scenario-filter-context');
    const minTpsEl = el('scenario-filter-min-tps');
    const ctxSel = ctxSelEl ? ctxSelEl.value : 'all';
    const minTps = minTpsEl ? Number(minTpsEl.value) : NaN;
    return baseRows.filter(r => {
        if (ctxSel !== 'all' && r.context !== Number(ctxSel)) return false;
        if (!isNaN(minTps) && r.tokensPerSecNum < minTps) return false;
        return true;
    });
}

// Value extractor for sorting (shared by sortRows)
const scenarioSortValue = (row, key) => {
    if (key === 'tokensPerSec') return row.tokensPerSecNum || Number(row.tokensPerSec) || 0;
    return SCENARIO_NUMERIC_SORT_KEYS.has(key) ? Number(row[key]) : String(row[key] || '').toLowerCase();
};

// Sorting
function sortScenarioRows(rowsToSort) {
    const prev = state.scenario.sortState || { key: null, dir: 'asc' };
    const key = prev.key;
    const dir = prev.dir;
    if (!key) return rowsToSort;
    const copy = [...rowsToSort];
    copy.sort((a, b) => {
        const va = scenarioSortValue(a, key);
        const vb = scenarioSortValue(b, key);
        if (typeof va === 'number' && typeof vb === 'number' && !isNaN(va) && !isNaN(vb)) {
            return dir === 'asc' ? va - vb : vb - va;
        }
        const cmp = String(va).localeCompare(String(vb));
        return dir === 'asc' ? cmp : -cmp;
    });
    return copy;
}

// Bind sortable headers (idempotent — runs once)
function bindScenarioSorting(table, tbody) {
    if (state.scenario.sortingBound) return;
    const headerCells = table.querySelectorAll('thead th[data-key]');
    headerCells.forEach(th => {
        th.addEventListener('click', () => {
            const key = th.getAttribute('data-key');
            const prev = state.scenario.sortState || { key: null, dir: 'asc' };
            const dir = prev.key === key && prev.dir === 'asc' ? 'desc' : 'asc';
            state.scenario.sortState = { key, dir };
            const base = Array.isArray(state.scenario.baseRows) ? state.scenario.baseRows : [];
            const filtered = applyScenarioFilters(base);
            const sorted = sortScenarioRows(filtered);
            renderScenarioRows(tbody, sorted);

            // Update header sort indicators
            headerCells.forEach(h => { h.classList.remove('sorted-asc', 'sorted-desc'); h.removeAttribute('aria-sort'); });
            th.classList.add(dir === 'asc' ? 'sorted-asc' : 'sorted-desc');
            th.setAttribute('aria-sort', dir);
        });
    });
    state.scenario.sortingBound = true;
}

// Bind filter controls (idempotent — runs once)
function bindScenarioFilters(tbody) {
    if (state.scenario.filtersBound) return;
    const ctxSelEl = el('scenario-filter-context');
    const minTpsEl = el('scenario-filter-min-tps');
    const reapply = () => {
        const base = Array.isArray(state.scenario.baseRows) ? state.scenario.baseRows : [];
        const filtered = applyScenarioFilters(base);
        const sorted = sortScenarioRows(filtered);
        renderScenarioRows(tbody, sorted);
    };
    if (ctxSelEl) ctxSelEl.addEventListener('change', reapply);
    if (minTpsEl) minTpsEl.addEventListener('input', reapply);
    state.scenario.filtersBound = true;
}

// Build a scenario table showing performance across GPU counts and context lengths
function buildPerformanceScenarioTable() {
    const base = readScenarioContext();
    if (!base) return;

    const rows = buildScenarioRows(base);

    // Base rows before filtering/sorting
    state.scenario.baseRows = rows;

    const filteredRows = applyScenarioFilters(rows);
    const finalRows = sortScenarioRows(filteredRows);
    renderScenarioRows(base.tbody, finalRows);

    bindScenarioSorting(base.table, base.tbody);
    bindScenarioFilters(base.tbody);

    base.section.style.display = rows.length > 0 ? 'block' : 'none';
}

// ============================================================================
// CSV export
// ============================================================================

// Build the scenario table CSV string; returns null when there are no rows
function buildScenarioCSV() {
    const rows = Array.isArray(state.scenario.rows) ? state.scenario.rows : [];
    if (rows.length === 0) return null;
    const headers = ['Model','Model Parameters (B)','GPU','Number of GPUs','Quantization','Context Length','Max Concurrent Requests','Tokens per Second','Time for 100 Tokens (s)'];
    const esc = (v) => {
        const s = v == null ? '' : String(v);
        const escaped = s.replace(/"/g, '""');
        return /[",\n]/.test(escaped) ? `"${escaped}"` : escaped;
    };
    const fmtCtx = (n) => {
        const num = Number(n);
        if (!isFinite(num)) return '';
        return num >= 1024 ? `${Math.round(num / 1024)}k` : String(num);
    };
    const csvRows = [headers.join(',')].concat(rows.map(r => [
        esc(r.model),
        esc(r.modelParamsB ?? ''),
        esc(r.gpu),
        esc(r.gpuCount),
        esc(r.quant),
        esc(fmtCtx(r.context)),
        esc(r.maxConcurrent),
        esc(r.tokensPerSec),
        esc(Number.isFinite(r.genTimeNum) ? r.genTimeNum.toFixed(1) : '')
    ].join(',')));
    return csvRows.join('\n');
}

// Copy scenario table as CSV to clipboard
function copyScenarioTable() {
    const csv = buildScenarioCSV();
    if (csv === null) return;
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(csv).catch(() => {});
    }
}

// Download scenario table as CSV
function downloadScenarioTable() {
    const csv = buildScenarioCSV();
    if (csv === null) return;
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'performance_scenarios.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// ============================================================================
// ASCII art
// ============================================================================

// NOTE: currently dormant — no #ascii-art markup on this page; kept intentionally.
// ASCII Art Style - Circuit Board (GPU version)
const asciiArt = `▓█████ ▓█████  ██▓      █████▒██░ ██  ▒█████    ██████ ▄▄▄█████▓ ██▓     ██▓     ███▄ ▄███▓
▒██    ▒██   ▀ ▓██▒    ▓██   ▒▓██░ ██▒▒██▒  ██▒▒██    ▒ ▓  ██▒ ▓▒▓██▒    ▓██▒    ▓██▒▀█▀ ██▒
▒▓██▄   ▒███   ▓██░    ▒████ ░▒██▀▀██░▒██░  ██▒▒▓██▄    ▒ ▓██░ ▒░▓██░    ▓██░    ▓██    ▓██░
▒██  ▀█▄ ▒▓█  ▄ ▒██▄    ░▓█▒  ░░▓█ ░██ ▒██   ██░ ▒   ██▒░ ▓██▓ ░ ▒██▄    ▒██▄    ▒██    ▒██ 
░██▄▄▄▄██░▒████▒░██████▒░▒█░   ░▓█▒░██▓░ ████▓▒░▒██████▒▒  ▒██▒ ░ ░██████▒░██████▒▒██▒   ░██▒
 ▓█   ▓██▒░ ▒░ ░░ ▒░▓  ░ ▒ ░    ▒ ░░▒░▒░ ▒░▒░▒░ ▒ ▒▓▒ ▒ ░  ▒ ░░   ░ ▒░▓  ░░ ▒░▓  ░░ ▒░   ░  ░`;

// Display ASCII art on page load
// NOTE: currently dormant — no markup on this page; kept intentionally.
function displayAsciiArt() {
    const asciiElement = el('ascii-art');
    if (asciiElement) {
        asciiElement.textContent = asciiArt;
    }
}

// ============================================================================
// Dialogs
// ============================================================================

// NOTE: currently dormant — no #shareDialog/#shareUrl markup on this page; kept intentionally.
// Share dialog functions
function showShareDialog() {
    const dialog = el('shareDialog');
    const overlay = el('overlay');
    const urlContainer = el('shareUrl');

    urlContainer.textContent = window.location.href;

    dialog.classList.add('active');
    overlay.classList.add('active');
}

// NOTE: currently dormant — no markup on this page; kept intentionally.
function closeShareDialog() {
    const dialog = el('shareDialog');
    const overlay = el('overlay');

    dialog.classList.remove('active');
    overlay.classList.remove('active');
}

// NOTE: currently dormant — no markup on this page; kept intentionally.
// Relies on the implicit global `event`; do not convert this file to strict mode/modules.
function copyShareLink() {
    const urlText = el('shareUrl').textContent;

    navigator.clipboard.writeText(urlText).then(() => {
        const copyButton = event.target;
        const originalText = copyButton.textContent;
        copyButton.textContent = '✅ Copied!';

        setTimeout(() => {
            copyButton.textContent = originalText;
        }, 2000);
    }).catch(err => {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = urlText;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);

        const copyButton = event.target;
        const originalText = copyButton.textContent;
        copyButton.textContent = '✅ Copied!';

        setTimeout(() => {
            copyButton.textContent = originalText;
        }, 2000);
    });
}

// Show explanation dialog
function showHowCalculated(event) {
    event.preventDefault();
    const dialog = el('explanationDialog');
    const overlay = el('overlay');

    dialog.classList.add('active');
    overlay.classList.add('active');
    overlay.onclick = closeExplanationDialog;
}

// Close explanation dialog
function closeExplanationDialog() {
    const dialog = el('explanationDialog');
    const overlay = el('overlay');

    dialog.classList.remove('active');
    overlay.classList.remove('active');
    overlay.onclick = closeShareDialog;
}

// Show performance explanation dialog
function showPerformanceExplanation(event) {
    event.preventDefault();
    const dialog = el('performanceExplanationDialog');
    const overlay = el('overlay');

    dialog.classList.add('active');
    overlay.classList.add('active');
    overlay.onclick = closePerformanceExplanation;
}

// Close performance explanation dialog
function closePerformanceExplanation() {
    const dialog = el('performanceExplanationDialog');
    const overlay = el('overlay');

    dialog.classList.remove('active');
    overlay.classList.remove('active');
    overlay.onclick = closeShareDialog;
}

// ============================================================================
// Escape-key handling
// ============================================================================

// Close dialog on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeShareDialog();
        closeExplanationDialog();
        closePerformanceExplanation();
    }
});

// ============================================================================
// Init (window.onload)
// ============================================================================

// Initialize on page load
window.onload = async function() {
    displayAsciiArt();
    // Augment GPU dropdown with any missing catalog entries
    await augmentCalculatorGPUOptionsFromCatalog();

    // Load LLM catalog for architecture details (layers, hidden size)
    try {
        await loadLLMCatalog();
    } catch (e) {
        console.warn('Could not load LLM catalog', e);
    }

    // First check if we have URL parameters
    if (window.location.search) {
        loadFromURL();
    } else {
        // Default initialization
        updateModelInputMethod();
        updateContextInputMethod();
        updateGPUSpecs();
        calculate();
    }

    // NOTE: currently dormant — no #theme-toggle markup on this page; kept intentionally.
    // Theme toggle for calculator page
    const themeToggle = el('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const body = document.body;
            const iconSpan = themeToggle.querySelector('span');
            const isDark = body.classList.contains('bg-deep-charcoal') || body.classList.contains('bg-charcoal');

            if (isDark) {
                body.classList.remove('bg-deep-charcoal', 'bg-charcoal', 'text-soft-gray');
                body.classList.add('bg-gray-100', 'text-gray-900', 'light');
                if (iconSpan) iconSpan.textContent = '☀️';
            } else {
                body.classList.remove('bg-gray-100', 'text-gray-900', 'light');
                body.classList.add('bg-deep-charcoal', 'text-soft-gray');
                if (iconSpan) iconSpan.textContent = '🌙';
            }
        });
    }
};

// ============================================================================
// LLM catalog loading and KV-cache math
// ============================================================================

// --- LLM Catalog Loading and Lookup ---
let _llmCatalog = null;

async function loadLLMCatalog() {
    try {
        const res = await fetch('data/LLMs.json');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        _llmCatalog = await res.json();
    } catch (e) {
        console.warn('Failed to fetch data/LLMs.json:', e);
        _llmCatalog = null;
    }
}

function normalizeModelName(s) {
    return String(s || '')
        .toLowerCase()
        .replace(/\(.*?\)/g, '') // drop parentheticals
        .replace(/distilled|instruct|base|small|medium|large|chat|oss/gi, '') // drop suffixes
        .replace(/\s+/g, '-')
        .replace(/[^a-z0-9\-x]/g, '')
        .replace(/--+/g, '-')
        .trim();
}

function findLLMConfigFromSelectedOption(optionEl) {
    const label = optionEl.textContent || optionEl.innerText || '';
    const norm = normalizeModelName(label);
    if (Array.isArray(_llmCatalog)) {
        // direct match first
        const direct = _llmCatalog.find(m => normalizeModelName(m.model_name) === norm);
        if (direct) return direct;
        // try contains both ways
        const contains = _llmCatalog.find(m => {
            const mn = normalizeModelName(m.model_name);
            return mn.includes(norm) || norm.includes(mn);
        });
        if (contains) return contains;
    }
    return null;
}

function heuristicArchitecture(optionEl) {
    // Fallback heuristics based on parameter count in label
    const label = optionEl.textContent || optionEl.innerText || '';
    const m = label.match(/(\d+(?:\.\d+)?)\s*B/i);
    const paramB = m ? parseFloat(m[1]) : null;

    // Rough defaults
    if (!paramB) {
        return { num_layers: 32, hidden_size: 4096 };
    }
    if (paramB <= 4) return { num_layers: 28, hidden_size: 3072 };
    if (paramB <= 8) return { num_layers: 32, hidden_size: 4096 };
    if (paramB <= 15) return { num_layers: 40, hidden_size: 4096 };
    if (paramB <= 35) return { num_layers: 64, hidden_size: 5120 };
    if (paramB <= 75) return { num_layers: 80, hidden_size: 8192 };
    if (paramB <= 130) return { num_layers: 88, hidden_size: 12288 };
    // very large
    return { num_layers: 60, hidden_size: 7168 };
}

function computeKVCacheGB(contextLength, optionEl, bytesPerElem, overheadFraction) {
    const catalog = findLLMConfigFromSelectedOption(optionEl);
    const arch = catalog ? { num_layers: catalog.num_layers, hidden_size: catalog.hidden_size } : heuristicArchitecture(optionEl);
    const L = Math.max(1, parseInt(arch.num_layers || 0) || 32);
    const H = Math.max(1, parseInt(arch.hidden_size || 0) || 4096);
    const bytesPerElement = bytesPerElem || KV_BYTES_PER_VALUE; // fp16/bf16 typical for KV
    const overhead = Math.max(0, overheadFraction || 0);

    // KV bytes = context_len × L × 2 (K+V) × H × bytes_per_elem
    const kvBytes = contextLength * L * 2 * H * bytesPerElement;
    const kvGB = kvBytes / (1024 ** 3);
    return kvGB * (1 + overhead);
}
