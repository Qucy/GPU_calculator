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

function stubPerformanceDOM(modelParams = '7', gpuBandwidth = '1008') {
    self.byQuery['input[name="model-input-type"]:checked'] = makeEl({ value: 'preset' });
    self.byId['model-preset'] = makeEl({ value: modelParams });
    self.byId['gpu-type'] = makeEl({
        selectedIndex: 0,
        options: [makeEl({ getAttribute: (k) => (k === 'data-bandwidth' ? gpuBandwidth : null) })]
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

test('calculatePerformance: quantization boost, context impact, multi-GPU scaling', () => {
    stubPerformanceDOM('7', '1008');
    assertEqual(S.calculatePerformance(14, 0.25, 4096, 'x', 1).quantBoost, 2.5);
    assertEqual(S.calculatePerformance(14, 1, 131072, 'x', 1).contextImpact, 0.3);
    const multi = S.calculatePerformance(14, 1, 4096, 'x', 4);
    assertEqual(multi.bandwidth, 4032);
    assertClose(multi.multiGpuScaling, 0.85 + 0.15 / 4, 1e-12);
});

test('calculatePerformance: returns null when bandwidth is unavailable', () => {
    stubPerformanceDOM('7', '1008');
    delete self.byId['gpu-type']; // no data-bandwidth, unknown slug -> 0
    assertEqual(S.calculatePerformance(14, 1, 4096, 'no-such-gpu', 1), null);
});

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

// --- bytesPerValueForPrecision ---

test('bytesPerValueForPrecision: fp32=4, fp16/bf16=2, fp8/int8/int4=1', () => {
    assertEqual(app.bytesPerValueForPrecision('fp32'), 4);
    assertEqual(app.bytesPerValueForPrecision('fp16'), 2);
    assertEqual(app.bytesPerValueForPrecision('bf16'), 2);
    assertEqual(app.bytesPerValueForPrecision('fp8'), 1);
    assertEqual(app.bytesPerValueForPrecision('int8'), 1);
    assertEqual(app.bytesPerValueForPrecision('int4'), 1);
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

test('efficiencyFactorForPrecision: int8/fp8=0.85, int4=0.9, default=0.7', () => {
    assertEqual(app.efficiencyFactorForPrecision('int8'), 0.85);
    assertEqual(app.efficiencyFactorForPrecision('fp8'), 0.85);
    assertEqual(app.efficiencyFactorForPrecision('int4'), 0.9);
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
// Summary
// ---------------------------------------------------------------------------

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
