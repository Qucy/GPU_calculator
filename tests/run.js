// Zero-dependency unit tests for main.js and selfhost-llm.js.
//
// Run with:  node tests/run.js        (or: bash scripts/test.sh)
//
// The browser scripts are evaluated inside a `vm` context with a minimal DOM
// stub, so no browser, npm install, or test framework is required. Top-level
// side effects of both files are limited to addEventListener/window.onload
// assignments, which the stubs absorb without executing any page logic.

const fs = require('fs');
const path = require('path');
const vm = require('vm');

// ---------------------------------------------------------------------------
// Tiny test harness
// ---------------------------------------------------------------------------

let passed = 0;
let failed = 0;

function test(name, fn) {
    try {
        fn();
        passed++;
        console.log(`ok   - ${name}`);
    } catch (e) {
        failed++;
        console.error(`FAIL - ${name}\n       ${e.message}`);
    }
}

function assert(cond, msg) {
    if (!cond) throw new Error(msg || 'assertion failed');
}

function assertEqual(actual, expected, msg) {
    if (actual !== expected) {
        throw new Error(`${msg || 'assertEqual'}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
    }
}

function assertClose(actual, expected, eps = 1e-6, msg) {
    if (!(Math.abs(actual - expected) <= eps)) {
        throw new Error(`${msg || 'assertClose'}: expected ~${expected}, got ${actual}`);
    }
}

// ---------------------------------------------------------------------------
// Minimal DOM stubs
// ---------------------------------------------------------------------------

function makeEl(overrides = {}) {
    return Object.assign({
        value: '',
        textContent: '',
        innerText: '',
        selectedIndex: -1,
        options: [],
        style: {},
        classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
        getAttribute() { return null; },
        setAttribute() {},
        addEventListener() {},
        appendChild() {},
        querySelector() { return null; },
        querySelectorAll() { return []; }
    }, overrides);
}

// Load a browser script into an isolated vm context and return the context.
// `byId` / `byQuery` are mutable registries tests use to plug in fake elements.
function loadScript(file, epilogue = '') {
    const byId = {};
    const byQuery = {};
    const documentStub = {
        getElementById: (id) => byId[id] || null,
        querySelector: (sel) => byQuery[sel] || null,
        querySelectorAll: () => [],
        addEventListener: () => {},
        createElement: () => makeEl(),
        body: makeEl()
    };
    const windowStub = {
        addEventListener: () => {},
        location: { search: '', pathname: '/' }
    };
    const sandbox = {
        console,
        URLSearchParams,
        document: documentStub,
        window: windowStub,
        history: { replaceState() {} },
        location: windowStub.location,
        navigator: { clipboard: { writeText: async () => {} } },
        fetch: async () => { throw new Error('fetch is not available in unit tests'); },
        setTimeout,
        clearTimeout
    };
    const src = fs.readFileSync(path.join(__dirname, '..', file), 'utf8');
    vm.createContext(sandbox);
    vm.runInContext(src + '\n' + epilogue, sandbox, { filename: file });
    return { sandbox, byId, byQuery };
}

// ---------------------------------------------------------------------------
// selfhost-llm.js (calculator page logic)
// ---------------------------------------------------------------------------

const self = loadScript(
    'selfhost-llm.js',
    'globalThis.__test = { state, setCatalog: (c) => { _llmCatalog = c; } };'
);
const S = self.sandbox;

// --- getGPUBandwidth ---

test('getGPUBandwidth: falls back to static map when no select element', () => {
    assertEqual(S.getGPUBandwidth('rtx4090'), 1008);
    assertEqual(S.getGPUBandwidth('h100'), 3000);
    assertEqual(S.getGPUBandwidth('h200'), 4915);
});

test('getGPUBandwidth: unknown / empty model returns 0', () => {
    assertEqual(S.getGPUBandwidth('no-such-gpu'), 0);
    assertEqual(S.getGPUBandwidth(''), 0);
});

test('getGPUBandwidth: prefers selected option data-bandwidth over the map', () => {
    self.byId['gpu-type'] = makeEl({
        selectedIndex: 0,
        options: [makeEl({ getAttribute: (k) => (k === 'data-bandwidth' ? '4800' : null) })]
    });
    // Even though 'rtx4090' maps to 1008, the selected option wins.
    assertEqual(S.getGPUBandwidth('rtx4090'), 4800);
    delete self.byId['gpu-type'];
});

// --- calculatePerformance ---

// attrs: extra data-* attributes for the GPU option (e.g. data-nvlink/data-pcie);
// modelLabel: textContent of the selected model option (drives arch heuristics).
function stubPerformanceDOM(modelParams = '7', gpuBandwidth = '1008', attrs = {}, modelLabel = null) {
    self.byQuery['input[name="model-input-type"]:checked'] = makeEl({ value: 'preset' });
    const modelOpt = makeEl({ textContent: modelLabel || `${modelParams}B` });
    self.byId['model-preset'] = makeEl({ value: modelParams, selectedIndex: 0, options: [modelOpt] });
    self.byId['gpu-type'] = makeEl({
        selectedIndex: 0,
        options: [makeEl({
            getAttribute: (k) => (k === 'data-bandwidth' ? gpuBandwidth : (attrs[k] ?? null))
        })]
    });
}

test('calculatePerformance: 7B fp16 on 1008 GB/s matches the documented formula', () => {
    stubPerformanceDOM('7', '1008');
    // (1008 / 14) * 0.85 * 1.0 * 1.0 * 1.0 * 0.6
    const perf = S.calculatePerformance(14, 1, 4096, 'rtx4090', 1);
    assertClose(perf.tokensPerSecond, 36.72, 1e-9);
    assertEqual(perf.bandwidth, 1008);
    assertEqual(perf.efficiency, 0.85);
});

test('calculatePerformance: efficiency tiers by model size', () => {
    stubPerformanceDOM('30', '1008');
    assertEqual(S.calculatePerformance(14, 1, 4096, 'x', 1).efficiency, 0.7);
    stubPerformanceDOM('70', '1008');
    assertEqual(S.calculatePerformance(14, 1, 4096, 'x', 1).efficiency, 0.5);
    stubPerformanceDOM('100', '1008');
    assertEqual(S.calculatePerformance(14, 1, 4096, 'x', 1).efficiency, 0.3);
});

test('calculatePerformance: quantization kernel efficiency, context impact, multi-GPU scaling', () => {
    stubPerformanceDOM('7', '1008');
    // Kernel efficiency (<1): the footprint shrink is already credited via modelMemory
    assertEqual(S.calculatePerformance(14, 0.25, 4096, 'x', 1).quantKernelEff, 0.6);
    assertEqual(S.calculatePerformance(14, 0.5, 4096, 'x', 1).quantKernelEff, 0.9);
    // New tiers: Q2_K/Q3_K_M (<=0.25) -> 0.6, Q4_K_M (0.3) -> 0.65,
    // Q5_K_M/Q6_K (<=0.5) -> 0.9, Q8_0 (<=0.75) -> 0.95, FP32 (2.0) -> 1.0
    assertEqual(S.calculatePerformance(14, 0.16015625, 4096, 'x', 1).quantKernelEff, 0.6);
    assertEqual(S.calculatePerformance(14, 0.3, 4096, 'x', 1).quantKernelEff, 0.65);
    assertEqual(S.calculatePerformance(14, 0.34375, 4096, 'x', 1).quantKernelEff, 0.9);
    assertEqual(S.calculatePerformance(14, 0.53125, 4096, 'x', 1).quantKernelEff, 0.95);
    assertEqual(S.calculatePerformance(14, 2.0, 4096, 'x', 1).quantKernelEff, 1.0);
    assertEqual(S.calculatePerformance(14, 1, 131072, 'x', 1).contextImpact, 0.3);
    const multi = S.calculatePerformance(14, 1, 4096, 'x', 4);
    assertEqual(multi.bandwidth, 4032);
    // heuristic scaling (0.85 + 0.15/4) further reduced by the PCIe comm penalty
    assert(multi.commFactor > 0 && multi.commFactor < 1, 'commFactor in (0,1)');
    assertClose(multi.multiGpuScaling, (0.85 + 0.15 / 4) * multi.commFactor, 1e-12);
    // single GPU: no comm penalty
    const single = S.calculatePerformance(14, 1, 4096, 'x', 1);
    assertEqual(single.commFactor, 1.0);
    assertEqual(single.interconnect, null);
});

test('calculatePerformance: returns null when bandwidth is unavailable', () => {
    stubPerformanceDOM('7', '1008');
    delete self.byId['gpu-type']; // no data-bandwidth, unknown slug -> 0
    assertEqual(S.calculatePerformance(14, 1, 4096, 'no-such-gpu', 1), null);
});

// --- Interconnect (PCIe / NVLink) modeling ---

test('getInterconnect: prefers NVLink, then PCIe generation, then conservative default', () => {
    // NVLink wins over PCIe when both are present
    self.byId['gpu-type'] = makeEl({
        selectedIndex: 0,
        options: [makeEl({ getAttribute: (k) => ({ 'data-nvlink': '600', 'data-pcie': '4.0' }[k] ?? null) })]
    });
    assertEqual(S.getInterconnect().kind, 'nvlink');
    assertEqual(S.getInterconnect().bandwidth, 600);
    // PCIe only
    self.byId['gpu-type'] = makeEl({
        selectedIndex: 0,
        options: [makeEl({ getAttribute: (k) => (k === 'data-pcie' ? '5.0' : null) })]
    });
    assertEqual(S.getInterconnect().kind, 'pcie');
    assertEqual(S.getInterconnect().bandwidth, 50);
    // Unknown -> conservative PCIe 4.0 x16 default
    self.byId['gpu-type'] = makeEl({ selectedIndex: 0, options: [makeEl()] });
    assertEqual(S.getInterconnect().kind, 'pcie');
    assertEqual(S.getInterconnect().bandwidth, 25);
    delete self.byId['gpu-type'];
});

test('tensorParallelCommFactor: single GPU no penalty, NVLink beats PCIe', () => {
    const arch = { num_layers: 80, hidden_size: 8192 }; // 70B-class
    assertEqual(S.tensorParallelCommFactor(140, 8000, 1, arch, { bandwidth: 600, kind: 'nvlink' }), 1.0);
    const nv = S.tensorParallelCommFactor(140, 8000, 4, arch, { bandwidth: 600, kind: 'nvlink' });
    const pc = S.tensorParallelCommFactor(140, 8000, 4, arch, { bandwidth: 25, kind: 'pcie' });
    assert(nv > pc, `NVLink (${nv}) should beat PCIe (${pc})`);
    assert(nv > 0.9, `NVLink should be near-lossless for 70B-class (got ${nv})`);
    // 2x RTX 4090 (PCIe 4.0, no NVLink) serving 70B INT4: measured all-reduce
    // eats ~25-30% of decode throughput (GigaGPU benchmark, 2026)
    const c4090 = S.tensorParallelCommFactor(35, 2016, 2, arch, { bandwidth: 25, kind: 'pcie' });
    assert(c4090 > 0.65 && c4090 < 0.9,
        `2x4090 PCIe comm factor ${c4090.toFixed(3)} should imply a ~10-35% penalty`);
});

test('calculatePerformance: NVLink multi-GPU beats PCIe multi-GPU by a realistic margin', () => {
    stubPerformanceDOM('70', '2000', { 'data-nvlink': '600' }, 'Llama 3 70B');
    const nvlinkPerf = S.calculatePerformance(140, 1, 4096, 'a100-80', 4);
    stubPerformanceDOM('70', '2000', { 'data-pcie': '4.0' }, 'Llama 3 70B');
    const pciePerf = S.calculatePerformance(140, 1, 4096, 'a100-80', 4);
    assert(nvlinkPerf.tokensPerSecond > pciePerf.tokensPerSecond,
        `NVLink ${nvlinkPerf.tokensPerSecond} should beat PCIe ${pciePerf.tokensPerSecond}`);
    // Seesaw (arXiv 2503.06433): 8x A100 PCIe reaches ~60% of SXM+NVLink for
    // 70B-class models (batched, x8 links); batch=1 over x16 should land higher.
    const ratio = pciePerf.tokensPerSecond / nvlinkPerf.tokensPerSecond;
    assert(ratio > 0.5 && ratio < 0.95, `PCIe/NVLink ratio ${ratio.toFixed(2)} should be in (0.5, 0.95)`);
});

// --- Real-world benchmark validation ---------------------------------------
// Published batch=1 decode measurements vs. this calculator's estimate.
// The model is a bandwidth-based heuristic; each case asserts the estimate
// lands within a documented tolerance factor of the observation.
//
// Sources:
//   [tinychat] MIT HAN Lab AWQ / TinyChat, batch=1 per-token decode latency,
//              https://github.com/mit-han-lab/llm-awq/blob/main/tinychat/README.md
//   [trtllm]   NVIDIA TensorRT-LLM blog, H100 vs A100 (Nov 2023, BS=1),
//              https://nvidia.github.io/TensorRT-LLM/blogs/H100vsA100.html
//   [gigagpu]  GigaGPU vLLM benchmarks (batch=1 sustained),
//              https://gigagpu.com/rtx-4090-24gb-llama-3-8b-benchmark/
//   [oci]      Oracle OCI Llama-3-70B benchmark, ~30.5 tok/s per request at
//              concurrency=1 (hardware undisclosed — order-of-magnitude check),
//              https://docs.oracle.com/en-us/iaas/Content/generative-ai/benchmark-meta-llama-3-70b-instruct.htm

const BENCHMARKS = [
    { name: 'Llama-3-8B FP16 @ A100 80GB [tinychat]', params: '8', label: 'Llama 3 8B', mem: 14, quant: 1, bw: '2000', gpus: 1, observed: 81, tol: [0.5, 2.0] },
    { name: 'Llama-2-7B FP16 @ A100 80GB [tinychat]', params: '7', label: 'Llama 2 7B', mem: 14, quant: 1, bw: '2000', gpus: 1, observed: 93, tol: [0.5, 2.0] },
    { name: 'Llama-3-8B FP16 @ RTX 4090 [tinychat]', params: '8', label: 'Llama 3 8B', mem: 14, quant: 1, bw: '1008', gpus: 1, observed: 59, tol: [0.5, 2.0] },
    { name: 'Llama-2-7B FP16 @ RTX 4090 [tinychat]', params: '7', label: 'Llama 2 7B', mem: 14, quant: 1, bw: '1008', gpus: 1, observed: 65, tol: [0.5, 2.0] },
    { name: 'GPT-J 6B FP16 @ A100 BS=1 [trtllm]', params: '6', label: 'GPT-J 6B', mem: 12, quant: 1, bw: '2000', gpus: 1, observed: 111, tol: [0.5, 2.0] },
    { name: 'Llama-3-8B INT4-AWQ @ A100 [tinychat]', params: '8', label: 'Llama 3 8B', mem: 14 * 0.25, quant: 0.25, bw: '2000', gpus: 1, observed: 159, tol: [0.5, 2.0] },
    // INT4 on consumer cards runs compute-bound at ~3.5GB footprint; the
    // bandwidth model underestimates there -> wider tolerance.
    { name: 'Llama-3-8B INT4-AWQ @ RTX 4090 [tinychat]', params: '8', label: 'Llama 3 8B', mem: 14 * 0.25, quant: 0.25, bw: '1008', gpus: 1, observed: 157, tol: [0.4, 2.5] },
    { name: 'Llama-2-7B INT4-AWQ @ RTX 4090 [tinychat]', params: '7', label: 'Llama 2 7B', mem: 14 * 0.25, quant: 0.25, bw: '1008', gpus: 1, observed: 189, tol: [0.4, 2.5] },
    { name: 'Llama-2-13B INT4-AWQ @ RTX 4090 [tinychat]', params: '13', label: 'Llama 2 13B', mem: 26 * 0.25, quant: 0.25, bw: '1008', gpus: 1, observed: 109, tol: [0.33, 3.0] },
    { name: 'Llama-3.1-8B FP8 @ H100 batch=1 [gigagpu]', params: '8', label: 'Llama 3.1 8B', mem: 14 * 0.5, quant: 0.5, bw: '3350', gpus: 1, observed: 330, tol: [0.4, 2.5] },
    // Multi-GPU absolute check (OCI hides its hardware; sanity band only)
    { name: 'Llama-3-70B FP16 @ 4x A100 80GB NVLink vs OCI ~30 tok/s', params: '70', label: 'Llama 3 70B', mem: 140, quant: 1, bw: '2000', gpus: 4, attrs: { 'data-nvlink': '600' }, observed: 30, tol: [0.3, 3.0] },
];

for (const b of BENCHMARKS) {
    test(`benchmark: ${b.name} ~= ${b.observed} tok/s`, () => {
        stubPerformanceDOM(b.params, b.bw, b.attrs || {}, b.label);
        const perf = S.calculatePerformance(b.mem, b.quant, 4096, 'bench-gpu', b.gpus);
        const ratio = perf.tokensPerSecond / b.observed;
        assert(ratio >= b.tol[0] && ratio <= b.tol[1],
            `estimate ${perf.tokensPerSecond.toFixed(1)} tok/s vs observed ${b.observed} ` +
            `(ratio ${ratio.toFixed(2)}, allowed ${b.tol[0]}-${b.tol[1]})`);
    });
}

// --- Model name normalization / LLM catalog matching ---

test('normalizeModelName: lowercase, dashes, strips parentheticals and suffixes', () => {
    assertEqual(S.normalizeModelName('Llama3 70B'), 'llama3-70b');
    // "(Instruct)" and the word "Instruct" are both dropped.
    assertEqual(S.normalizeModelName('Foo 7B (Chat)'), S.normalizeModelName('Foo 7B chat'));
});

test('findLLMConfigFromSelectedOption: direct match, then contains match', () => {
    S.__test.setCatalog([{ model_name: 'Qwen3 32B', num_layers: 64 }]);
    const direct = S.findLLMConfigFromSelectedOption(makeEl({ textContent: 'Qwen3 32B' }));
    assertEqual(direct.num_layers, 64);
    const contains = S.findLLMConfigFromSelectedOption(makeEl({ textContent: 'Qwen3 32B (256K context)' }));
    assertEqual(contains.num_layers, 64);
    assertEqual(S.findLLMConfigFromSelectedOption(makeEl({ textContent: 'Unknown Model 9B' })), null);
    S.__test.setCatalog(null);
});

// --- heuristicArchitecture ---

test('heuristicArchitecture: parameter-count tiers', () => {
    const at = (label) => S.heuristicArchitecture(makeEl({ textContent: label }));
    assertEqual(at('Some 3B model').num_layers, 28);
    assertEqual(at('Some 3B model').hidden_size, 3072);
    assertEqual(at('7B').hidden_size, 4096);
    assertEqual(at('13B').num_layers, 40);
    assertEqual(at('70B').num_layers, 80);
    assertEqual(at('70B').hidden_size, 8192);
    assertEqual(at('110B').num_layers, 88);
    assertEqual(at('500B').num_layers, 60);
    // No parameter count in label -> safe default.
    assertEqual(at('mystery model').num_layers, 32);
});

// --- computeKVCacheGB ---

test('computeKVCacheGB: heuristic 7B, 8K context, 20% overhead = 4.8 GB', () => {
    S.__test.setCatalog(null);
    const kv = S.computeKVCacheGB(8192, makeEl({ textContent: '7B' }), 2, 0.2);
    assertClose(kv, 4.8, 1e-9);
});

test('computeKVCacheGB: prefers catalog layers/hidden_size when matched', () => {
    S.__test.setCatalog([{ model_name: 'Custom 7B', num_layers: 40, hidden_size: 5120 }]);
    // 8192 * 40 * 2 * 5120 * 2 / 2^30 = 6.25
    const kv = S.computeKVCacheGB(8192, makeEl({ textContent: 'Custom 7B' }), 2, 0);
    assertClose(kv, 6.25, 1e-9);
    S.__test.setCatalog(null);
});

// --- buildScenarioCSV ---

test('buildScenarioCSV: header, quoting, and context formatting', () => {
    S.__test.state.scenario.rows = [{
        model: 'Model, One',
        modelParamsB: 7,
        gpu: 'H100',
        gpuCount: 2,
        quant: 'FP16',
        context: 8192,
        maxConcurrent: 10,
        tokensPerSec: '50',
        genTimeNum: 2
    }];
    const csv = S.buildScenarioCSV();
    const lines = csv.split('\n');
    assert(lines[0].startsWith('Model,Model Parameters (B),GPU,'), 'header row');
    assertEqual(lines[1], '"Model, One",7,H100,2,FP16,8k,10,50,2.0');
    S.__test.state.scenario.rows = [];
    assertEqual(S.buildScenarioCSV(), null);
});

// ---------------------------------------------------------------------------
// main.js (GPUCalculator class)
// ---------------------------------------------------------------------------

const main = loadScript('main.js', 'globalThis.__GPUCalculator = GPUCalculator;');
const GPUCalculator = main.sandbox.__GPUCalculator;
// The constructor calls this.init(), which binds DOM/events; stub it out so we
// can unit-test pure methods in isolation.
GPUCalculator.prototype.init = function () {};
const app = new GPUCalculator();

// --- quantizationFactors ---

test('quantizationFactors: fp32=4, fp16/bf16=2, fp8/int8=1, fp4/int4=0.5, int2=0.25', () => {
    assertEqual(app.quantizationFactors['fp32'], 4);
    assertEqual(app.quantizationFactors['fp16'], 2);
    assertEqual(app.quantizationFactors['bf16'], 2);
    assertEqual(app.quantizationFactors['fp8'], 1);
    assertEqual(app.quantizationFactors['int8'], 1);
    assertEqual(app.quantizationFactors['fp4'], 0.5);
    assertEqual(app.quantizationFactors['int4'], 0.5);
    assertEqual(app.quantizationFactors['int2'], 0.25);
});

// --- bytesPerValueForPrecision ---

test('bytesPerValueForPrecision: fp32=4, fp16/bf16=2, fp8/int8=1, int4/fp4=0.5, int2=0.25', () => {
    assertEqual(app.bytesPerValueForPrecision('fp32'), 4);
    assertEqual(app.bytesPerValueForPrecision('fp16'), 2);
    assertEqual(app.bytesPerValueForPrecision('bf16'), 2);
    assertEqual(app.bytesPerValueForPrecision('fp8'), 1);
    assertEqual(app.bytesPerValueForPrecision('int8'), 1);
    assertEqual(app.bytesPerValueForPrecision('int4'), 0.5);
    assertEqual(app.bytesPerValueForPrecision('fp4'), 0.5);
    assertEqual(app.bytesPerValueForPrecision('int2'), 0.25);
});

// --- getMemoryGB (defensive parsing of "40 / 80" style values) ---

test('getMemoryGB: numbers, numeric strings, and "40 / 80" take the max', () => {
    assertEqual(app.getMemoryGB({ memory_gb: 80 }), 80);
    assertEqual(app.getMemoryGB({ memory_gb: '24' }), 24);
    assertEqual(app.getMemoryGB({ memory_gb: '40 / 80' }), 80);
    assertEqual(app.getMemoryGB({ memory_gb: null }), 0);
    assertEqual(app.getMemoryGB(null), 0);
});

// --- getBandwidthGBps (TB/s -> GB/s conversion) ---

test('getBandwidthGBps: memory_bandwidth_tbps, legacy bandwidth_tbps, raw GB/s', () => {
    assertEqual(app.getBandwidthGBps({ memory_bandwidth_tbps: 2.0 }), 2048);
    assertClose(app.getBandwidthGBps({ bandwidth_tbps: '3.35' }), 3430.4, 1e-9);
    assertEqual(app.getBandwidthGBps({ bandwidth: 900 }), 900);
    assertEqual(app.getBandwidthGBps({}), 0);
    assertEqual(app.getBandwidthGBps(null), 0);
});

// --- efficiencyFactorForPrecision ---

test('efficiencyFactorForPrecision: int8/fp8=0.85, int4/fp4/int2=0.9, default=0.7', () => {
    assertEqual(app.efficiencyFactorForPrecision('int8'), 0.85);
    assertEqual(app.efficiencyFactorForPrecision('fp8'), 0.85);
    assertEqual(app.efficiencyFactorForPrecision('int4'), 0.9);
    assertEqual(app.efficiencyFactorForPrecision('fp4'), 0.9);
    assertEqual(app.efficiencyFactorForPrecision('int2'), 0.9);
    assertEqual(app.efficiencyFactorForPrecision('fp16'), 0.7);
});

// --- estimateTokensPerSecondForPair ---

test('estimateTokensPerSecondForPair: reference scaling and heuristic paths', () => {
    // Reference path: 700 tps @ 1000 GB/s scaled to 2000 GB/s.
    assertEqual(app.estimateTokensPerSecondForPair(14, { bandwidth: 2000 }, 'fp16', 700, 1000), 1400);
    // Heuristic path: round(1008 / (14 * 0.85)) = 85.
    assertEqual(app.estimateTokensPerSecondForPair(14, { bandwidth: 1008 }, 'int8'), 85);
});

// --- estimateLLMMemoryForPrecision (70B bf16, 8K context) ---

test('estimateLLMMemoryForPrecision: weights/KV/activation/overhead breakdown', () => {
    const m = { parameter_count_billion: 70, num_layers: 80, hidden_size: 8192, context_length: 8192 };
    const est = app.estimateLLMMemoryForPrecision(m, 'bf16', 8192, 1, 1);
    assertClose(est.weightsGB, 70e9 * 2 / (1024 ** 3), 1e-6, 'weights');
    assertClose(est.cacheGB, 20, 1e-9, 'KV cache');
    assertClose(est.activationGB, 0.15, 1e-9, 'activation (1.2x large-model factor)');
    // Preserved behavior: overhead is a hardcoded 30%.
    assertClose(est.totalGB, (est.weightsGB + est.cacheGB + est.activationGB) * 1.3, 1e-9, 'total');
});

// --- sortValues ---

test('sortValues: numeric, numeric-string, alpha, and null ordering', () => {
    assert(app.sortValues(2, 10, 'asc') < 0);
    assert(app.sortValues(2, 10, 'desc') > 0);
    assert(app.sortValues('10', '2', 'asc') > 0, 'numeric strings compare numerically');
    assert(app.sortValues('a', 'b', 'asc') < 0);
    assert(app.sortValues(null, 1, 'asc') > 0, 'nulls sort last (asc)');
    assert(app.sortValues(1, null, 'desc') < 0, 'nulls sort last (desc)');
    assertEqual(app.sortValues(null, null), 0);
});

// --- catalog search filtering (GPU) ---

test('applyGPUFilters: search query matches name/vendor/architecture, case-insensitive', () => {
    app.gpuCatalogData = [
        { name: 'H100', vendor: 'NVIDIA', architecture: 'Hopper' },
        { name: 'MI300X', vendor: 'AMD', architecture: 'CDNA 3' },
        { name: 'Gaudi 3', vendor: 'Intel', architecture: 'Gaudi' }
    ];
    app.gpuFilters = { vendor: '', architecture: '', memory: '' };

    app.gpuSearchQuery = 'h100';
    app.applyGPUFilters();
    assertEqual(app.filteredGpuData.map(g => g.name).join(','), 'H100', 'name match');

    app.gpuSearchQuery = 'amd';
    app.applyGPUFilters();
    assertEqual(app.filteredGpuData.map(g => g.name).join(','), 'MI300X', 'vendor match');

    app.gpuSearchQuery = 'cdna';
    app.applyGPUFilters();
    assertEqual(app.filteredGpuData.map(g => g.name).join(','), 'MI300X', 'architecture match');

    app.gpuSearchQuery = 'zzz-no-match';
    app.applyGPUFilters();
    assertEqual(app.filteredGpuData.length, 0, 'no match -> empty');

    app.gpuSearchQuery = '';
    app.applyGPUFilters();
    assertEqual(app.filteredGpuData.length, 3, 'empty query -> all');
});

// --- catalog search filtering (LLM) ---

test('applyLLMFilters: search query matches model name/organization, case-insensitive', () => {
    app.llms = [
        { model_name: 'Llama 3.1 70B', organization: 'Meta' },
        { model_name: 'DeepSeek-V3', organization: 'DeepSeek' }
    ];
    app.llmFilters = { size: '', type: '', license: '' };

    app.llmSearchQuery = 'deepseek';
    app.applyLLMFilters();
    assertEqual(app.filteredLlmData.map(m => m.model_name).join(','), 'DeepSeek-V3', 'name match');

    app.llmSearchQuery = 'meta';
    app.applyLLMFilters();
    assertEqual(app.filteredLlmData.map(m => m.model_name).join(','), 'Llama 3.1 70B', 'organization match');

    app.llmSearchQuery = 'zzz-no-match';
    app.applyLLMFilters();
    assertEqual(app.filteredLlmData.length, 0, 'no match -> empty');

    app.llmSearchQuery = '';
    app.applyLLMFilters();
    assertEqual(app.filteredLlmData.length, 2, 'empty query -> all');
});

// --- card view sorting ---

test('card view sort: GPUs by memory desc, LLMs by params desc', () => {
    const gpus = app.sortGPUList(
        [{ name: 'a', memory_gb: 24 }, { name: 'b', memory_gb: 80 }, { name: 'c', memory_gb: 48 }],
        'memory_gb', 'desc'
    );
    assertEqual(gpus.map(g => g.name).join(','), 'b,c,a');

    const llms = app.sortLLMList(
        [{ model_name: 'x', parameter_count_billion: 7 }, { model_name: 'y', parameter_count_billion: 70 }],
        'params_b', 'desc'
    );
    assertEqual(llms.map(m => m.model_name).join(','), 'y,x');
});

// --- chooseDefaultPrecisionForLLM ---

test('chooseDefaultPrecisionForLLM: prefers fp16 > bf16 > fp32 > int8 > fp8', () => {
    assertEqual(app.chooseDefaultPrecisionForLLM({ precision_supported: ['BF16', 'FP8'], quantization_types: ['INT8'] }), 'bf16');
    assertEqual(app.chooseDefaultPrecisionForLLM({ precision_supported: ['FP16'] }), 'fp16');
    assertEqual(app.chooseDefaultPrecisionForLLM({}), 'fp16');
});

// --- resolveGPUByName (fuzzy catalog match) ---

test('resolveGPUByName: exact, substring, and miss', () => {
    app.gpuCatalogData = [{ name: 'NVIDIA H100 80GB' }, { name: 'NVIDIA H20' }];
    assertEqual(app.resolveGPUByName('NVIDIA H100 80GB').name, 'NVIDIA H100 80GB');
    assertEqual(app.resolveGPUByName('h20').name, 'NVIDIA H20');
    assertEqual(app.resolveGPUByName('no-such-gpu'), null);
});

// --- formatNumber (locks current behavior; decimals arg is a known no-op) ---

test('formatNumber: thousands separators, decimals argument ignored (preserved)', () => {
    assertEqual(app.formatNumber(1234567), '1,234,567');
    assertEqual(app.formatNumber(0), '0');
});

// ---------------------------------------------------------------------------
// Catalog data validation (data/GPUs.json, data/LLMs.json, calculator.html)
// ---------------------------------------------------------------------------

const gpuCatalog = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data', 'GPUs.json'), 'utf8')).gpus;
const llmCatalog = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data', 'LLMs.json'), 'utf8'));
const calculatorHtml = fs.readFileSync(path.join(__dirname, '..', 'calculator.html'), 'utf8');

test('GPUs.json: unique names, required fields on every entry', () => {
    const names = gpuCatalog.map(g => g.name);
    assertEqual(new Set(names).size, names.length, 'duplicate GPU names');
    for (const g of gpuCatalog) {
        assert(g.vendor, `${g.name}: missing vendor`);
        assert(g.memory_gb != null, `${g.name}: missing memory_gb`);
        assert(g.memory_bandwidth_tbps != null, `${g.name}: missing memory_bandwidth_tbps (performance estimates stay hidden without it)`);
    }
});

test('GPUs.json: 2025-2026 additions present with expected specs', () => {
    const byName = Object.fromEntries(gpuCatalog.map(g => [g.name, g]));
    const expected = [
        ['NVIDIA B200', 192, 8.0, 1800],
        ['NVIDIA B300', 288, 8.0, 1800],
        ['NVIDIA RTX PRO 6000 Blackwell', 96, 1.792, null],
        ['AMD Instinct MI300X', 192, 5.3, 896],
        ['AMD Instinct MI325X', 256, 6.0, 896],
        ['AMD Instinct MI355X', 288, 8.0, 896],
        ['Intel Gaudi 3', 128, 3.67, null],
    ];
    for (const [name, mem, tbps, nvlink] of expected) {
        const g = byName[name];
        assert(g, `${name} missing from GPUs.json`);
        assertEqual(Number(g.memory_gb), mem, `${name} memory_gb`);
        assertClose(Number(g.memory_bandwidth_tbps), tbps, 1e-9, `${name} bandwidth`);
        assertEqual(g.nvlink_bandwidth_gbs ?? null, nvlink, `${name} interconnect`);
    }
});

test('LLMs.json: unique names, required fields on every entry', () => {
    const names = llmCatalog.map(m => m.model_name);
    assertEqual(new Set(names).size, names.length, 'duplicate model names');
    for (const m of llmCatalog) {
        assert(typeof m.parameter_count_billion === 'number' && m.parameter_count_billion > 0,
            `${m.model_name}: bad parameter_count_billion`);
        assert(typeof m.context_length === 'number' && m.context_length > 0,
            `${m.model_name}: bad context_length`);
        assert(m.moe && typeof m.moe.enabled === 'boolean', `${m.model_name}: missing moe block`);
    }
});

test('LLMs.json: 2025-2026 additions present with expected sizes', () => {
    const byName = Object.fromEntries(llmCatalog.map(m => [m.model_name, m]));
    const expected = [
        ['DeepSeek V4 Pro', 1600, true],
        ['DeepSeek V4 Flash', 284, true],
        ['GLM-5.2', 744, true],
        ['Kimi K3', 2800, true],
        ['MiniMax M3', 428, true],
        ['Qwen3-VL-235B-A22B', 235, true],
        ['Gemma 4 31B', 31, false],
        ['Mistral Large 3', 675, true],
    ];
    for (const [name, paramsB, isMoE] of expected) {
        const m = byName[name];
        assert(m, `${name} missing from LLMs.json`);
        assertEqual(m.parameter_count_billion, paramsB, `${name} params`);
        assertEqual(m.moe.enabled, isMoE, `${name} moe.enabled`);
    }
    // Kimi K3: 16 of 896 experts active per token (Moonshot model card)
    assertEqual(byName['Kimi K3'].moe.num_experts, 896);
    assertEqual(byName['Kimi K3'].moe.active_experts, 16);
});

test('calculator.html dropdown: new models present, data-memory = 2x params (FP16 GB)', () => {
    // [unique label fragment, total params B, active params B (null for dense)]
    const cases = [
        ['V4 Pro 1.6T', 1600, 49],
        ['V4 Flash 284B', 284, 13],
        ['GLM-5.2', 744, 40],
        ['Kimi K3', 2800, 50],
        ['MiniMax M3', 428, 23],
        ['Qwen3-VL', 235, 22],
        ['Large 3 675B', 675, 41],
        ['Gemma 4 31B', 31, null],
    ];
    for (const [frag, totalB, activeB] of cases) {
        const esc = frag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const re = new RegExp(`<option value="(\\d+(?:\\.\\d+)?)" data-memory="(\\d+(?:\\.\\d+)?)"[^>]*>[^<]*${esc}`);
        const match = calculatorHtml.match(re);
        assert(match, `dropdown option not found for ${JSON.stringify(frag)}`);
        assertEqual(parseFloat(match[2]), totalB * 2, `${frag} data-memory should be ${totalB * 2}`);
        if (activeB != null) {
            assertEqual(parseFloat(match[1]), activeB, `${frag} value (active B) should be ${activeB}`);
        }
    }
});

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
