// GPU Calculator Pro - Main JavaScript Engine
// Advanced GPU requirements calculator with real-time updates and visual effects

// ===== Tunable constants (hoisted from inline literals; values unchanged) =====
const RECOMMENDATION_LIMIT = 5;          // max GPU recommendations shown
const CONCURRENCY_MAX = 1000;            // concurrency input clamp
const GPU_COUNT_MAX = 128;               // GPU count input clamp
const EFFICIENCY_INT8_FP8 = 0.85;        // throughput efficiency for int8/fp8
const EFFICIENCY_INT4 = 0.9;             // throughput efficiency for int4
const EFFICIENCY_DEFAULT = 0.7;          // throughput efficiency for fp16/bf16 and fallback
const ANIMATE_VALUE_MS = 800;            // anime.js duration for metric counters
const SCROLL_REVEAL_MS = 800;            // anime.js duration for scroll-in animations
const HOVER_LIFT_MS = 300;               // anime.js duration for hover lift effects
const PAGE_SLIDE_OUT_MS = 250;           // anime.js duration for SPA page exit
const PAGE_SLIDE_IN_MS = 300;            // anime.js duration for SPA page entrance

// ===== Catalog table column definitions =====
// align: 'left' | 'right' (maps to text-left / text-right header classes)
const GPU_TABLE_COLUMNS = [
    { key: 'name', label: 'GPU', align: 'left' },
    { key: 'vendor', label: 'Vendor', align: 'left' },
    { key: 'architecture', label: 'Architecture', align: 'left' },
    { key: 'process_node', label: 'Process', align: 'left' },
    { key: 'memory_gb', label: 'Memory (GB)', align: 'right' },
    { key: 'memory_type', label: 'Memory Type', align: 'left' },
    { key: 'memory_bandwidth_tbps', label: 'Bandwidth (TB/s)', align: 'right' },
    { key: 'fp32_tflops', label: 'FP32 TFLOPs', align: 'right' },
    { key: 'int8_tops', label: 'INT8 TOPS', align: 'right' },
    { key: 'tdp_w', label: 'TDP (W)', align: 'right' },
    { key: 'price_usd', label: 'Price (USD)', align: 'right' },
    { key: 'fp16_tflops', label: 'FP16 TFLOPs', align: 'right' },
    { key: 'nvlink_bandwidth_gbs', label: 'NVLink (GB/s)', align: 'right' },
    { key: 'pcie_generation', label: 'PCIe Gen', align: 'left' },
    { key: 'release_year', label: 'Release', align: 'right' },
    { key: 'mig_support', label: 'MIG', align: 'left' },
    { key: 'transformer_engine', label: 'Transformer Engine', align: 'left' },
    { key: 'cuda_cores', label: 'CUDA Cores', align: 'right' },
    { key: 'tensor_cores', label: 'Tensor Cores', align: 'left' },
    { key: 'rt_cores', label: 'RT Cores', align: 'left' },
    { key: 'price_rmb', label: 'Price (RMB)', align: 'right' }
];

const LLM_TABLE_COLUMNS = [
    { key: 'model_name', label: 'Model', align: 'left' },
    { key: 'release_date', label: 'Release', align: 'left' },
    { key: 'params_b', label: 'Params (B)', align: 'right' },
    { key: 'context_length', label: 'Context', align: 'right' },
    { key: 'architecture_type', label: 'Architecture', align: 'left' },
    { key: 'num_layers', label: 'Layers', align: 'right' },
    { key: 'hidden_size', label: 'Hidden', align: 'right' },
    { key: 'num_attention_heads', label: 'Heads', align: 'right' },
    { key: 'vocab_size', label: 'Vocab', align: 'right' },
    { key: 'organization', label: 'Organization', align: 'left' },
    { key: 'precision_supported', label: 'Precision', align: 'left' },
    { key: 'quantization_types', label: 'Quantization', align: 'left' },
    { key: 'moe_summary', label: 'MoE', align: 'left' },
    { key: 'serving_frameworks', label: 'Serving', align: 'left' },
    { key: 'recommended_gpu', label: 'Recommended GPU', align: 'left' },
    { key: 'throughput_tokens_per_sec_per_gpu', label: 'Throughput', align: 'right' },
    { key: 'memory_footprint_gb', label: 'Memory (GB)', align: 'right' },
    { key: 'license', label: 'License', align: 'left' },
    { key: 'source_links', label: 'Sources', align: 'left' }
];

class GPUCalculator {
    // ===== Section: Constructor (state and static data) =====
    constructor() {
        this.models = {
            'qwen-ds-7b': { params: 7.0e9, layers: 32, hiddenDim: 4096, heads: 32 },
            'qwen-ds-14b': { params: 14.0e9, layers: 40, hiddenDim: 5120, heads: 40 },
            'qwen-ds-32b': { params: 32.0e9, layers: 80, hiddenDim: 8192, heads: 64 },
            // Qwen 3 series
            'qwen3-7b': { params: 7.0e9, layers: 32, hiddenDim: 4096, heads: 32 },
            'qwen3-14b': { params: 14.0e9, layers: 40, hiddenDim: 5120, heads: 40 },
            'qwen3-32b': { params: 32.0e9, layers: 80, hiddenDim: 8192, heads: 64 },
            // DeepSeek V3 (671B)
            'deepseek-v3': { params: 671.0e9, layers: 120, hiddenDim: 12288, heads: 120 }
        };

        this.quantizationFactors = {
            'fp32': 4,
            'fp16': 2,
            'bf16': 2,
            'fp8': 1,
            'int8': 1,
            'fp4': 0.5,
            'int4': 0.5,
            'int2': 0.25
        };

        this.gpus = [
            // NOTE: gpus[0] is the reference GPU for calculatePerformance() — keep H200 first.
            { name: 'H200', vram: 141, bandwidth: 4800, price: 35000, cloudPrice: 6.50 },
            { name: 'B300', vram: 288, bandwidth: 8000, price: 55000, cloudPrice: 9.00 },
            { name: 'B200', vram: 192, bandwidth: 8000, price: 40000, cloudPrice: 8.00 },
            { name: 'H100', vram: 80, bandwidth: 3350, price: 30000, cloudPrice: 5.50 },
            { name: 'H20', vram: 96, bandwidth: 2000, price: 25000, cloudPrice: 4.50 },
            { name: 'A100', vram: 80, bandwidth: 2039, price: 15000, cloudPrice: 3.50 },
            { name: 'V100', vram: 32, bandwidth: 900, price: 8000, cloudPrice: 1.50 },
            { name: 'L40', vram: 48, bandwidth: 846, price: 3500, cloudPrice: 1.80 }
        ];

        // JSON-driven catalogs
        this.gpuCatalogData = []; // Loaded from data/GPUs.json
        this.llms = [];           // Loaded from data/LLMs.json

        // Infrastructure defaults for deployment sizing
        this.infra = {
            gpusPerServer: 8,     // typical DGX/enterprise servers
            serversPerRack: 8,    // rough estimate for 42U racks
            memUtilizationMax: 0.8, // target VRAM utilization per GPU
            overheadFactor: 1.3     // 30% safety overhead applied across sums
        };

        this.currentConfig = {
            model: 'qwen-ds-7b',
            quantization: 'fp16',
            contextLength: 4096,
            concurrency: 1,
            batchSize: 1,
            customParams: null,
            gpuCount: 1,
            sysOverheadPercent: 30,
            kvOverheadPercent: 0,
            // New hardware-related inputs
            selectedGPUModelName: '',
            vramPerGPU: null,
            sysOverheadGB: 2
        };

        this.memoryChart = null;
        this.gpuViewMode = 'cards'; // Calculator recommendations view: 'cards' or 'table'
        this.gpuCatalogViewMode = 'cards'; // GPU Explorer page
        this.llmCatalogViewMode = 'cards'; // Open Source Models page
        // Catalog sort states
        this.gpuCatalogSort = { key: 'name', dir: 'asc' };
        this.llmCatalogSort = { key: 'model_name', dir: 'asc' };

        // Filter states
        this.gpuFilters = {
            vendor: '',
            architecture: '',
            memory: ''
        };
        this.llmFilters = {
            size: '',
            type: '',
            license: ''
        };
        this.filteredGpuData = [];
        this.filteredLlmData = [];

        // SPA-style page navigation
        this.pageOrder = ['gpu', 'models', 'calculator'];
        this.currentPage = 'gpu';
        this.init();
    }

    // ===== Section: Init =====
    init() {
        this.setupEventListeners();
        this.initializeAnimations();
        this.initializeMemoryChart();
        this.updateCalculations();

        // Load datasets and render catalogs
        Promise.all([this.loadGPUData(), this.loadLLMData()])
            .then(() => this.initializeCatalogs(true))
            .catch(() => this.initializeCatalogs(false));

        // Initialize SPA page navigation
        this.initPageNavigation();
    }

    // Shared post-load setup for both the success and failure paths of catalog loading
    initializeCatalogs(dataLoaded) {
        // Initialize filtered data with all records (empty arrays if loading failed)
        this.filteredGpuData = dataLoaded ? [...(this.gpuCatalogData || [])] : [];
        this.filteredLlmData = dataLoaded ? [...(this.llms || [])] : [];

        this.renderGPUCatalog('gpu-catalog', this.gpuCatalogViewMode);
        this.renderLLMCatalog('llm-catalog', this.llmCatalogViewMode);
        // Set initial full-width layout for both catalogs
        this.setFullWidthLayout();
        if (dataLoaded) {
            // Populate filter options after data is loaded
            this.populateFilterOptions();
        }
        // Populate GPU model select in calculator (falls back to built-in list)
        this.populateGPUModelSelect();
        // Apply URL parameters after data and selects are ready
        this.loadFromURL();
    }

    // ===== Section: Event binding =====
    setupEventListeners() {
        // Model selection
        const modelSelect = this.el('model-select');
        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                this.currentConfig.model = e.target.value;
                this.toggleCustomModel();
                // Clear custom params when not using custom model
                if (this.currentConfig.model !== 'custom') {
                    this.currentConfig.customParams = null;
                    const customInput = this.el('custom-params');
                    if (customInput) customInput.value = '';
                }
                this.updateCalculations();
            });
        }

        // Custom model parameters
        const customParamsInput = this.el('custom-params');
        if (customParamsInput) {
            customParamsInput.addEventListener('input', (e) => {
                this.currentConfig.customParams = parseFloat(e.target.value) * 1e9;
                this.updateCalculations();
            });
        }

        // Quantization selection
        document.querySelectorAll('input[name="quantization"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.currentConfig.quantization = e.target.value;
                this.updateQuantizationUI();
                this.updateCalculations();
            });
        });

        // Context length slider
        const contextSlider = this.el('context-slider');
        if (contextSlider) {
            contextSlider.addEventListener('input', (e) => {
                this.currentConfig.contextLength = parseInt(e.target.value);
                const ctxValEl = this.el('context-value');
                if (ctxValEl) ctxValEl.textContent = this.formatNumber(this.currentConfig.contextLength);
                this.updateCalculations();
            });
        }

        // Context presets
        document.querySelectorAll('.context-preset').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const val = parseInt(btn.getAttribute('data-context'));
                this.currentConfig.contextLength = val;
                if (contextSlider) contextSlider.value = val;
                const ctxValEl = this.el('context-value');
                if (ctxValEl) ctxValEl.textContent = this.formatNumber(val);
                this.updateCalculations();
            });
        });

        // KV Cache Overhead slider
        const kvOverheadSlider = this.el('kv-overhead-slider');
        const kvOverheadValueEl = this.el('kv-overhead-value');
        if (kvOverheadSlider && kvOverheadValueEl) {
            kvOverheadSlider.addEventListener('input', (e) => {
                const pct = Math.max(0, Math.min(100, parseInt(e.target.value)));
                this.currentConfig.kvOverheadPercent = pct;
                kvOverheadValueEl.textContent = `${pct}%`;
                this.updateCalculations();
            });
        }

        // Concurrency controls
        const concInc = this.el('concurrency-inc');
        const concDec = this.el('concurrency-dec');
        const concInput = this.el('concurrency-input');
        if (concInc && concInput) {
            concInc.addEventListener('click', () => {
                const value = Math.min(CONCURRENCY_MAX, parseInt(concInput.value) + 1);
                concInput.value = value;
                this.currentConfig.concurrency = value;
                this.updateCalculations();
            });
        }
        if (concDec && concInput) {
            concDec.addEventListener('click', () => {
                const value = Math.max(1, parseInt(concInput.value) - 1);
                concInput.value = value;
                this.currentConfig.concurrency = value;
                this.updateCalculations();
            });
        }
        if (concInput) {
            concInput.addEventListener('input', (e) => {
                const value = Math.max(1, Math.min(CONCURRENCY_MAX, parseInt(e.target.value) || 1));
                e.target.value = value;
                this.currentConfig.concurrency = value;
                this.updateCalculations();
            });
        }

        // Batch size slider
        const batchSlider = this.el('batch-slider');
        if (batchSlider) {
            batchSlider.addEventListener('input', (e) => {
                this.currentConfig.batchSize = parseInt(e.target.value);
                const batchValEl = this.el('batch-value');
                if (batchValEl) batchValEl.textContent = this.currentConfig.batchSize;
                this.updateCalculations();
            });
        }

        // Batch presets
        document.querySelectorAll('.batch-preset').forEach(btn => {
            btn.addEventListener('click', () => {
                const val = parseInt(btn.getAttribute('data-batch'));
                this.currentConfig.batchSize = val;
                if (batchSlider) batchSlider.value = val;
                const batchValEl = this.el('batch-value');
                if (batchValEl) batchValEl.textContent = String(val);
                this.updateCalculations();
            });
        });

        // GPU count controls
        const gpuCountInput = this.el('gpu-count-input');
        const gpuCountInc = this.el('gpu-count-inc');
        const gpuCountDec = this.el('gpu-count-dec');
        if (gpuCountInput && gpuCountInc && gpuCountDec) {
            gpuCountInc.addEventListener('click', () => {
                const value = Math.min(GPU_COUNT_MAX, parseInt(gpuCountInput.value) + 1);
                gpuCountInput.value = value;
                this.currentConfig.gpuCount = value;
                this.updateCalculations();
            });
            gpuCountDec.addEventListener('click', () => {
                const value = Math.max(1, parseInt(gpuCountInput.value) - 1);
                gpuCountInput.value = value;
                this.currentConfig.gpuCount = value;
                this.updateCalculations();
            });
            gpuCountInput.addEventListener('input', (e) => {
                const value = Math.max(1, Math.min(GPU_COUNT_MAX, parseInt(e.target.value) || 1));
                e.target.value = value;
                this.currentConfig.gpuCount = value;
                this.updateCalculations();
            });
        }

        // GPU model select
        const gpuModelSelect = this.el('gpu-model-select');
        if (gpuModelSelect) {
            gpuModelSelect.addEventListener('change', (e) => {
                const name = e.target.value || '';
                this.currentConfig.selectedGPUModelName = name;
                // When a GPU is selected, set VRAM per GPU automatically if available
                const catalogGPU = this.resolveGPUByName(name);
                const memoryGB = this.getMemoryGB(catalogGPU);
                const vramInput = this.el('vram-per-gpu');
                if (vramInput && memoryGB) {
                    vramInput.value = memoryGB;
                    this.currentConfig.vramPerGPU = memoryGB;
                }
                this.updateCalculations();
            });
        }

        // VRAM per GPU override
        const vramPerGPUInput = this.el('vram-per-gpu');
        if (vramPerGPUInput) {
            vramPerGPUInput.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                this.currentConfig.vramPerGPU = isNaN(val) ? null : val;
                this.updateCalculations();
            });
        }

        // System overhead GB
        const sysOverheadGBInput = this.el('sys-overhead-gb');
        if (sysOverheadGBInput) {
            sysOverheadGBInput.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                this.currentConfig.sysOverheadGB = isNaN(val) ? 0 : val;
                this.updateCalculations();
            });
        }

        // System overhead slider
        const overheadSlider = this.el('sys-overhead-slider');
        const overheadValueEl = this.el('sys-overhead-value');
        if (overheadSlider && overheadValueEl) {
            overheadSlider.addEventListener('input', (e) => {
                const pct = Math.max(0, Math.min(100, parseInt(e.target.value)));
                this.currentConfig.sysOverheadPercent = pct;
                overheadValueEl.textContent = `${pct}%`;
                this.updateCalculations();
            });
        }

        // Theme toggle
        const themeToggle = this.el('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                this.toggleTheme();
            });
        }

        // Header nav -> SPA tab switching
        const navLinks = document.querySelectorAll('header nav a[href^="#"]');
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = (link.getAttribute('href') || '').replace('#', '');
                this.switchPage(targetId);
            });
        });

        // GPU recommendations view toggle (sliding switch)
        this.setupViewToggle('gpu-view-toggle', 'gpu-view-knob', 'gpuViewMode', () => {
            this.updateCalculations();
        });

        // GPU catalog toggle (GPU page)
        this.setupViewToggle('gpu-catalog-view-toggle', 'gpu-catalog-view-knob', 'gpuCatalogViewMode', () => {
            this.renderGPUCatalog('gpu-catalog', this.gpuCatalogViewMode);
            // Make both table and cards view take full width
            this.setWrapperFullWidth('gpu-catalog');
        });

        // LLM catalog toggle (Open Source Models page)
        this.setupViewToggle('llm-catalog-view-toggle', 'llm-catalog-view-knob', 'llmCatalogViewMode', () => {
            this.renderLLMCatalog('llm-catalog', this.llmCatalogViewMode);
            // Make both table and cards view take full width
            this.setWrapperFullWidth('llm-catalog');
        });

        // GPU Filter event listeners
        const gpuVendorFilter = this.el('gpu-vendor-filter');
        const gpuArchFilter = this.el('gpu-architecture-filter');
        const gpuMemoryFilter = this.el('gpu-memory-filter');
        const gpuClearFilters = this.el('gpu-clear-filters');

        if (gpuVendorFilter) {
            gpuVendorFilter.addEventListener('change', (e) => {
                this.gpuFilters.vendor = e.target.value;
                this.applyGPUFilters();
            });
        }
        if (gpuArchFilter) {
            gpuArchFilter.addEventListener('change', (e) => {
                this.gpuFilters.architecture = e.target.value;
                this.applyGPUFilters();
            });
        }
        if (gpuMemoryFilter) {
            gpuMemoryFilter.addEventListener('change', (e) => {
                this.gpuFilters.memory = e.target.value;
                this.applyGPUFilters();
            });
        }
        if (gpuClearFilters) {
            gpuClearFilters.addEventListener('click', () => {
                this.clearGPUFilters();
            });
        }

        // LLM Filter event listeners
        const llmSizeFilter = this.el('llm-size-filter');
        const llmTypeFilter = this.el('llm-type-filter');
        const llmLicenseFilter = this.el('llm-license-filter');
        const llmClearFilters = this.el('llm-clear-filters');

        if (llmSizeFilter) {
            llmSizeFilter.addEventListener('change', (e) => {
                this.llmFilters.size = e.target.value;
                this.applyLLMFilters();
            });
        }
        if (llmTypeFilter) {
            llmTypeFilter.addEventListener('change', (e) => {
                this.llmFilters.type = e.target.value;
                this.applyLLMFilters();
            });
        }
        if (llmLicenseFilter) {
            llmLicenseFilter.addEventListener('change', (e) => {
                this.llmFilters.license = e.target.value;
                this.applyLLMFilters();
            });
        }
        if (llmClearFilters) {
            llmClearFilters.addEventListener('click', () => {
                this.clearLLMFilters();
            });
        }
    }

    // Shared wiring for the sliding cards/table view toggles.
    // modeKey is the name of the view-mode property on this instance
    // ('gpuViewMode', 'gpuCatalogViewMode', or 'llmCatalogViewMode').
    setupViewToggle(toggleId, knobId, modeKey, onChange) {
        const toggleBtn = this.el(toggleId);
        const toggleKnob = this.el(knobId);
        if (!toggleBtn || !toggleKnob) return;

        const setTogglePosition = () => {
            const isCards = this[modeKey] === 'cards';
            toggleBtn.classList.toggle('justify-start', isCards);
            toggleBtn.classList.toggle('justify-end', !isCards);
            toggleBtn.setAttribute('aria-pressed', isCards ? 'false' : 'true');
        };

        // Initialize position
        setTogglePosition();

        toggleBtn.addEventListener('click', () => {
            this[modeKey] = this[modeKey] === 'cards' ? 'table' : 'cards';
            setTogglePosition();
            onChange();
        });
    }

    // ===== Section: UI helpers =====
    el(id) {
        return document.getElementById(id);
    }

    safeSet(id, value) {
        const element = this.el(id);
        if (element) element.textContent = value;
    }

    toggleCustomModel() {
        const customDiv = this.el('custom-model-params');
        const customInput = this.el('custom-params');
        if (this.currentConfig.model === 'custom') {
            if (customDiv) customDiv.classList.remove('hidden');
            // Initialize default custom params to 7B if unset
            if (!this.currentConfig.customParams) {
                const defaultB = 7.0;
                this.currentConfig.customParams = defaultB * 1e9;
                if (customInput) customInput.value = defaultB;
            }
        } else {
            if (customDiv) customDiv.classList.add('hidden');
        }
    }

    updateQuantizationUI() {
        document.querySelectorAll('.quantization-option').forEach(option => {
            const radio = option.querySelector('input[type="radio"]');
            const indicator = option.querySelector('.w-2');
            const border = option.querySelector('.w-4');

            if (radio.value === this.currentConfig.quantization) {
                border.classList.add('border-accent');
                border.classList.remove('border-soft-gray/50');
                indicator.classList.add('scale-100');
                indicator.classList.remove('scale-0');
            } else {
                border.classList.remove('border-accent');
                border.classList.add('border-soft-gray/50');
                indicator.classList.remove('scale-100');
                indicator.classList.add('scale-0');
            }
        });
    }

    // ===== Section: Calculation engine =====
    // NOTE: the memory/architecture math below is intentionally mirrored in
    // getGPURecommendations() and estimateLLMMemoryForPrecision() with slight
    // per-call-site differences; keep the copies in sync when editing.
    calculateMemoryRequirements() {
        const model = this.models[this.currentConfig.model];
        const params = this.currentConfig.customParams || (model ? model.params : 7.0e9);
        const quantization = this.quantizationFactors[this.currentConfig.quantization];
        // dtype-aware bytes per value for KV/activation
        const bytesPerValue = (this.currentConfig.quantization === 'fp32') ? 4 :
            (this.currentConfig.quantization === 'fp16' || this.currentConfig.quantization === 'bf16') ? 2 : 1;

        // Resolve architecture specs (layers, hiddenDim, heads)
        let layers, hiddenDim, heads;
        if (model) {
            ({ layers, hiddenDim, heads } = model);
        } else {
            // Heuristic defaults for custom model based on parameter count
            const p = params || 7.0e9;
            if (p <= 7.0e9) { layers = 32; hiddenDim = 4096; heads = 32; }
            else if (p <= 13.0e9) { layers = 40; hiddenDim = 5120; heads = 40; }
            else if (p <= 70.0e9) { layers = 80; hiddenDim = 8192; heads = 64; }
            else if (p <= 175.0e9) { layers = 96; hiddenDim = 12288; heads = 96; }
            else { layers = 120; hiddenDim = 12288; heads = 120; }
        }

        // Model weights memory
        const effectiveParams = params || 7.0e9;
        const weightsMemory = (effectiveParams * quantization) / (1024 ** 3); // Convert to GB

        // KV cache memory
        // Use concurrency to scale KV cache (each concurrent request maintains its own KV)
        const baseCacheMemory = (2 * layers * hiddenDim * this.currentConfig.contextLength * this.currentConfig.concurrency * bytesPerValue) / (1024 ** 3);
        const kvFactor = 1 + ((typeof this.currentConfig.kvOverheadPercent === 'number') ? (this.currentConfig.kvOverheadPercent / 100) : 0);
        const cacheMemory = baseCacheMemory * kvFactor;

        // Activation memory
        const actOverheadFactor = (hiddenDim >= 8192 || layers >= 80) ? 1.2 : 1.0; // apply overhead only for large models
        const activationMemory = (this.currentConfig.batchSize * this.currentConfig.contextLength * hiddenDim * bytesPerValue * actOverheadFactor) / (1024 ** 3);

        // Total with overhead
        const subtotal = weightsMemory + cacheMemory + activationMemory;
        const overheadPct = (typeof this.currentConfig.sysOverheadPercent === 'number') ? (this.currentConfig.sysOverheadPercent / 100) : 0.3;
        const fixedOverheadGB = (typeof this.currentConfig.sysOverheadGB === 'number') ? this.currentConfig.sysOverheadGB : 0;
        const overheadMemory = subtotal * overheadPct + fixedOverheadGB;
        const totalMemory = subtotal + overheadMemory;

        return {
            weights: weightsMemory,
            cache: cacheMemory,
            activation: activationMemory,
            overhead: overheadMemory,
            total: totalMemory
        };
    }

    calculatePerformance(memory) {
        // Use a reference GPU's memory bandwidth (RTX 4090 by default)
        // Use first enterprise GPU as reference for performance (H200 by default)
        const referenceGPU = this.gpus[0];
        const memoryBandwidth = referenceGPU.bandwidth; // GB/s

        // Computation efficiency per docs (quantizationFactors now covers
        // fp4/int4/int2, so 'int4' here is a reachable selection).
        const efficiencyFactor = this.currentConfig.quantization === 'int8' ? EFFICIENCY_INT8_FP8 :
            this.currentConfig.quantization === 'int4' ? EFFICIENCY_INT4 : EFFICIENCY_DEFAULT; // fp16/bf16 default to 0.7

        // Tokens per second per documentation
        const tokensPerSecond = Math.max(1, memoryBandwidth / (memory.total * efficiencyFactor));

        // Bandwidth utilization per documentation
        const bandwidthUtilizationRaw = (memory.weights * tokensPerSecond) / memoryBandwidth * 100;
        const bandwidthUtilization = Math.min(95, Math.max(0, bandwidthUtilizationRaw));

        const effectiveSpeed = tokensPerSecond / this.currentConfig.concurrency;

        return {
            inferenceSpeed: Math.round(tokensPerSecond),
            bandwidthUtilization: Math.round(bandwidthUtilization),
            effectiveSpeed: Math.round(effectiveSpeed)
        };
    }

    getGPURecommendations(memory) {
        // Compute per-request memory footprint (without concurrency multiplier)
        const model = this.models[this.currentConfig.model];
        const params = this.currentConfig.customParams || (model ? model.params : 7.0e9);
        const quantization = this.quantizationFactors[this.currentConfig.quantization];
        const bytesPerValue = (this.currentConfig.quantization === 'fp32') ? 4 :
            (this.currentConfig.quantization === 'fp16' || this.currentConfig.quantization === 'bf16') ? 2 : 1;

        // Resolve architecture specs
        // NOTE: this heuristic variant intentionally lacks `heads` (not needed here).
        let layers, hiddenDim;
        if (model) {
            ({ layers, hiddenDim } = model);
        } else {
            const p = params || 7.0e9;
            if (p <= 7.0e9) { layers = 32; hiddenDim = 4096; }
            else if (p <= 13.0e9) { layers = 40; hiddenDim = 5120; }
            else if (p <= 70.0e9) { layers = 80; hiddenDim = 8192; }
            else if (p <= 175.0e9) { layers = 96; hiddenDim = 12288; }
            else { layers = 120; hiddenDim = 12288; }
        }

        const weightsGB = (params * quantization) / (1024 ** 3);
        const actOverheadFactor = (hiddenDim >= 8192 || layers >= 80) ? 1.2 : 1.0;
        const kvPerReqGB = (2 * layers * hiddenDim * this.currentConfig.contextLength * bytesPerValue) / (1024 ** 3);
        const kvFactorRec = 1 + ((typeof this.currentConfig.kvOverheadPercent === 'number') ? (this.currentConfig.kvOverheadPercent / 100) : 0);
        const actPerReqGB = (this.currentConfig.batchSize * this.currentConfig.contextLength * hiddenDim * bytesPerValue * actOverheadFactor) / (1024 ** 3);
        const perReqGB = (kvPerReqGB * kvFactorRec) + actPerReqGB;

        const memBudgetFactor = this.infra.memUtilizationMax;
        const overhead = this.infra.overheadFactor;

        const recommendations = this.gpus.map(gpu => {
            const vramOverride = (typeof this.currentConfig.vramPerGPU === 'number' && this.currentConfig.vramPerGPU > 0) ? this.currentConfig.vramPerGPU : gpu.vram;
            const memBudget = vramOverride * memBudgetFactor; // usable VRAM target

            // Minimum GPUs required just to host weights (tensor parallel shards)
            const shardsForWeights = Math.max(1, Math.ceil((weightsGB * overhead) / memBudget));

            // Concurrency capacity per shard group
            const usableVRAMAcrossShards = memBudget * shardsForWeights;
            const headroomAfterWeights = Math.max(0, usableVRAMAcrossShards - (weightsGB * overhead));
            const perReqWithOverhead = perReqGB * overhead;
            const requestsPerShardGroup = Math.max(0, Math.floor(headroomAfterWeights / perReqWithOverhead));

            // Total GPUs needed to satisfy requested concurrency
            const desiredConcurrency = this.currentConfig.concurrency;
            const shardGroupsNeeded = requestsPerShardGroup > 0 ? Math.ceil(desiredConcurrency / requestsPerShardGroup) : desiredConcurrency; // if 0, one req per shard group
            const totalGPUsNeeded = shardGroupsNeeded * shardsForWeights;

            // Utilization estimate using full current memory total vs single GPU VRAM
            const memoryUtilization = (memory.total / vramOverride) * 100;
            const isCompatibleSingle = (weightsGB * overhead) <= memBudget; // fits weights on one GPU

            // Server/rack sizing
            const serversNeeded = Math.ceil(totalGPUsNeeded / this.infra.gpusPerServer);
            const racksNeeded = Math.ceil(serversNeeded / this.infra.serversPerRack);

            return {
                ...gpu,
                vram: vramOverride,
                utilization: Math.round(memoryUtilization),
                compatible: isCompatibleSingle && requestsPerShardGroup > 0,
                totalGPUsNeeded,
                shardsPerReplica: shardsForWeights,
                requestsPerReplica: requestsPerShardGroup,
                serversNeeded,
                racksNeeded
            };
        }).sort((a, b) => {
            // Prefer compatible, fewer total GPUs, then lower hourly cost
            if (a.compatible && !b.compatible) return -1;
            if (!a.compatible && b.compatible) return 1;
            return a.totalGPUsNeeded - b.totalGPUsNeeded;
        });

        return recommendations.slice(0, RECOMMENDATION_LIMIT);
    }

    // NOTE: currently dormant — no caller/markup; kept intentionally.
    getOptimizationTips(memory, performance) {
        const tips = [];

        if (memory.total > 40) {
            tips.push({
                type: 'warning',
                title: 'High Memory Usage',
                message: 'Consider using INT8 or INT4 quantization to reduce memory requirements by 50-75%.'
            });
        }

        if (this.currentConfig.contextLength > 32000) {
            tips.push({
                type: 'info',
                title: 'Large Context Window',
                message: 'For very large contexts, consider gradient checkpointing to trade compute for memory.'
            });
        }

        if (this.currentConfig.concurrency > 100) {
            tips.push({
                type: 'info',
                title: 'High Concurrency',
                message: 'Consider model parallelism or multiple GPU setups for better performance.'
            });
        }

        if (performance.bandwidthUtilization < 50) {
            tips.push({
                type: 'success',
                title: 'Good Utilization',
                message: 'You have headroom for larger batch sizes to improve throughput.'
            });
        }

        if (tips.length === 0) {
            tips.push({
                type: 'success',
                title: 'Optimal Configuration',
                message: 'Your current configuration appears well-balanced for the selected model.'
            });
        }

        return tips;
    }

    // ===== Section: Output rendering =====
    updateCalculations() {
        const memory = this.calculateMemoryRequirements();
        const performance = this.calculatePerformance(memory);
        const gpuRecommendations = this.getGPURecommendations(memory);
        const subtotal = (memory.weights + memory.cache + memory.activation);

        // Update memory display (safe)
        this.safeSet('weights-memory', `${memory.weights.toFixed(1)} GB`);
        this.safeSet('cache-memory', `${memory.cache.toFixed(1)} GB`);
        this.safeSet('activation-memory', `${memory.activation.toFixed(1)} GB`);
        this.safeSet('overhead-memory', `${memory.overhead.toFixed(1)} GB`);
        this.safeSet('subtotal-memory', `${subtotal.toFixed(1)} GB`);
        this.safeSet('total-memory', `${memory.total.toFixed(1)} GB`);

        // Update performance metrics
        this.animateValue('inference-speed', performance.inferenceSpeed);
        this.animateValue('memory-bandwidth', performance.bandwidthUtilization);
        this.animateValue('effective-speed', performance.effectiveSpeed);

        // Update parameter count
        const model = this.models[this.currentConfig.model];
        const params = this.currentConfig.customParams || (model ? model.params : 7.0e9);
        this.safeSet('param-count', this.formatNumber((params || 7.0e9) / 1e9, 1) + 'B');

        this.safeSet('summary-total', `${memory.total.toFixed(1)} GB`);
        this.safeSet('summary-subtotal', `${subtotal.toFixed(1)} GB`);
        this.safeSet('summary-weights', `${memory.weights.toFixed(1)} GB`);
        this.safeSet('summary-cache', `${memory.cache.toFixed(1)} GB`);
        this.safeSet('summary-activation', `${memory.activation.toFixed(1)} GB`);
        this.safeSet('summary-overhead', `${memory.overhead.toFixed(1)} GB`);
        this.safeSet('summary-gpucount', String(this.currentConfig.gpuCount));
        this.safeSet('summary-quant', (this.currentConfig.quantization || 'fp16').toUpperCase());
        this.safeSet('summary-context', this.formatNumber(this.currentConfig.contextLength));
        this.safeSet('summary-concurrency', String(this.currentConfig.concurrency));
        this.safeSet('summary-batch', String(this.currentConfig.batchSize));
        this.safeSet('summary-params', this.formatNumber((params || 7.0e9) / 1e9, 1) + 'B');
        this.safeSet('summary-overhead-pct', `${this.currentConfig.sysOverheadPercent}%`);

        // Update GPU recommendations
        this.updateGPURecommendations(gpuRecommendations);

        // Update memory chart
        this.updateMemoryChart(memory);

        // Sync URL and share link
        this.updateURL();
    }

    animateValue(elementId, targetValue) {
        const element = this.el(elementId);
        if (!element) return;
        const currentValue = parseInt(element.textContent) || 0;

        anime({
            targets: { value: currentValue },
            value: targetValue,
            duration: ANIMATE_VALUE_MS,
            easing: 'easeOutCubic',
            update: function (anim) {
                element.textContent = Math.round(anim.animatables[0].target.value);
            }
        });
    }

    updateGPURecommendations(recommendations) {
        const container = this.el('gpu-recommendations');
        if (!container) return;
        container.innerHTML = '';

        if (this.gpuViewMode === 'table') {
            // Render as table
            container.className = '';
            const table = document.createElement('table');
            table.className = 'w-full text-sm bg-navy/50 rounded-lg overflow-hidden';

            const thead = document.createElement('thead');
            thead.className = 'bg-navy/70 text-soft-gray/80';
            thead.innerHTML = `
                <tr>
                    <th class="text-left px-4 py-2">GPU</th>
                    <th class="text-right px-4 py-2">VRAM (GB)</th>
                    <th class="text-right px-4 py-2">Bandwidth (GB/s)</th>
                    <th class="text-right px-4 py-2">Utilization (%)</th>
                    <th class="text-center px-4 py-2">Compatible</th>
                    <th class="text-right px-4 py-2">Total GPUs</th>
                    <th class="text-right px-4 py-2">Shards/Replica</th>
                    <th class="text-right px-4 py-2">Requests/Replica</th>
                    <th class="text-right px-4 py-2">Servers</th>
                    <th class="text-right px-4 py-2">Racks</th>
                    <th class="text-right px-4 py-2">Price ($)</th>
                    <th class="text-right px-4 py-2">Cloud $/hr</th>
                </tr>`;
            table.appendChild(thead);

            const tbody = document.createElement('tbody');
            recommendations.forEach(gpu => {
                const tr = document.createElement('tr');
                tr.className = 'border-t border-soft-gray/10';
                tr.innerHTML = `
                    <td class="px-4 py-2 font-medium">${gpu.name}</td>
                    <td class="px-4 py-2 text-right">${gpu.vram}</td>
                    <td class="px-4 py-2 text-right">${gpu.bandwidth}</td>
                    <td class="px-4 py-2 text-right ${gpu.compatible ? 'text-sage' : 'text-amber'} font-mono">${gpu.utilization}</td>
                    <td class="px-4 py-2 text-center">${gpu.compatible ? 'Yes' : 'No'}</td>
                    <td class="px-4 py-2 text-right font-mono">${gpu.totalGPUsNeeded}</td>
                    <td class="px-4 py-2 text-right font-mono">${gpu.shardsPerReplica}</td>
                    <td class="px-4 py-2 text-right font-mono">${gpu.requestsPerReplica}</td>
                    <td class="px-4 py-2 text-right font-mono">${gpu.serversNeeded}</td>
                    <td class="px-4 py-2 text-right font-mono">${gpu.racksNeeded}</td>
                    <td class="px-4 py-2 text-right">${gpu.price ? `$${gpu.price.toLocaleString()}` : '-'}</td>
                    <td class="px-4 py-2 text-right">${gpu.cloudPrice ? `$${gpu.cloudPrice.toFixed(2)}` : '-'}</td>
                `;
                tbody.appendChild(tr);
            });
            table.appendChild(tbody);
            container.appendChild(table);
            return;
        }

        // Default: render as cards, one per row
        container.className = 'grid grid-cols-1 gap-3';
        recommendations.forEach(gpu => {
            const div = document.createElement('div');
            div.className = `p-4 bg-navy/50 rounded-lg flex items-center justify-between hover:bg-navy/70 transition-colors ${gpu.compatible ? 'border-l-4 border-sage' : 'border-l-4 border-amber'}`;

            div.innerHTML = `
                <div class="flex items-start space-x-4">
                    <div class="w-12 h-12 bg-accent/20 rounded-lg flex items-center justify-center">
                        <span class="text-accent font-bold text-sm">GPU</span>
                    </div>
                    <div class="space-y-1">
                        <div class="font-semibold">${gpu.name}</div>
                        <div class="text-sm text-soft-gray/70">${gpu.vram}GB VRAM • ${gpu.bandwidth} GB/s</div>
                        <div class="text-xs text-soft-gray/60">${gpu.compatible ? 'Single-GPU fits model' : 'Requires sharded model'}</div>
                        <div class="text-xs text-soft-gray/60">Shards/Replica: <span class="font-mono">${gpu.shardsPerReplica}</span> • Requests/Replica: <span class="font-mono">${gpu.requestsPerReplica}</span></div>
                        <div class="text-xs text-soft-gray/60">Price: ${gpu.price ? `$${gpu.price.toLocaleString()}` : '-'} • Cloud: ${gpu.cloudPrice ? `$${gpu.cloudPrice.toFixed(2)}/hr` : '-'}</div>
                    </div>
                </div>
                <div class="text-right space-y-1">
                    <div class="font-mono text-lg ${gpu.compatible ? 'text-sage' : 'text-amber'}">${gpu.utilization}%</div>
                    <div class="text-sm text-soft-gray/70">Needed: <span class="font-mono">${gpu.totalGPUsNeeded}</span> GPUs</div>
                    <div class="text-sm text-soft-gray/70">Servers: <span class="font-mono">${gpu.serversNeeded}</span> • Racks: <span class="font-mono">${gpu.racksNeeded}</span></div>
                </div>
            `;

            container.appendChild(div);
        });
    }

    // NOTE: currently dormant — no caller/markup; kept intentionally.
    updateOptimizationTips(tips) {
        const container = this.el('optimization-tips');
        container.innerHTML = '';

        tips.forEach(tip => {
            const div = document.createElement('div');
            const colorClass = tip.type === 'success' ? 'border-sage' :
                tip.type === 'warning' ? 'border-amber' : 'border-accent';
            const iconClass = tip.type === 'success' ? 'text-sage' :
                tip.type === 'warning' ? 'text-amber' : 'text-accent';

            div.className = `p-4 bg-navy/50 rounded-lg border-l-4 ${colorClass}`;
            div.innerHTML = `
                <div class="flex items-start space-x-3">
                    <div class="w-6 h-6 ${iconClass} flex items-center justify-center mt-0.5">
                        ${tip.type === 'success' ? '✓' : tip.type === 'warning' ? '⚠' : 'ℹ'}
                    </div>
                    <div>
                        <div class="font-semibold mb-1">${tip.title}</div>
                        <div class="text-sm text-soft-gray/70">${tip.message}</div>
                    </div>
                </div>
            `;

            container.appendChild(div);
        });
    }

    // ===== Section: Data loading =====
    async loadJSON(path) {
        try {
            const res = await fetch(path);
            if (!res.ok) throw new Error(`Failed to load ${path}`);
            return await res.json();
        } catch (e) {
            console.warn('JSON load error:', e.message);
            return null;
        }
    }

    async loadGPUData() {
        const data = await this.loadJSON('data/GPUs.json');
        if (!data) return;
        if (Array.isArray(data.gpus)) {
            this.gpuCatalogData = data.gpus;
        } else if (Array.isArray(data)) {
            this.gpuCatalogData = data;
        }
    }

    async loadLLMData() {
        const data = await this.loadJSON('data/LLMs.json');
        if (!data) return;
        if (Array.isArray(data)) {
            this.llms = data;
        } else if (Array.isArray(data.llms)) {
            this.llms = data.llms;
        }
    }

    // Populate GPU model select in calculator from catalog data or built-in list
    populateGPUModelSelect() {
        const select = this.el('gpu-model-select');
        if (!select) return;
        // Preserve current selection if any
        const prev = this.currentConfig.selectedGPUModelName || '';
        // Build a unique, sorted list of GPU names
        const names = new Set();
        (this.gpuCatalogData || []).forEach(g => { if (g?.name) names.add(String(g.name)); });
        if (names.size === 0) {
            // Fallback to minimal built-in list
            (this.gpus || []).forEach(g => { if (g?.name) names.add(String(g.name)); });
        }
        const list = Array.from(names).sort((a, b) => a.localeCompare(b));
        // Render options
        select.innerHTML = '<option value="">Select GPU…</option>' +
            list.map(n => `<option value="${n}">${n}</option>`).join('');
        // Restore previous selection
        if (prev) {
            select.value = prev;
        }
    }

    // ===== Section: Pairing/estimate helpers (LLM ↔ GPU) =====
    bytesPerValueForPrecision(precision) {
        const p = (precision || '').toLowerCase();
        if (p === 'fp32') return 4;
        if (p === 'fp16' || p === 'bf16') return 2;
        if (p === 'int4' || p === 'fp4') return 0.5;
        if (p === 'int2') return 0.25;
        // fp8 / int8 and unknown precisions default to 1 byte per value
        return 1;
    }

    chooseDefaultPrecisionForLLM(m) {
        const listA = Array.isArray(m.precision_supported) ? m.precision_supported.map(x => String(x).toLowerCase()) : [];
        const listB = Array.isArray(m.quantization_types) ? m.quantization_types.map(x => String(x).toLowerCase()) : [];
        const set = new Set([...listA, ...listB]);
        if (set.has('fp16')) return 'fp16';
        if (set.has('bf16')) return 'bf16';
        if (set.has('fp32')) return 'fp32';
        if (set.has('int8')) return 'int8';
        if (set.has('fp8')) return 'fp8';
        return 'fp16';
    }

    getBandwidthGBps(gpu) {
        if (!gpu) return 0;
        // Prefer JSON field: memory_bandwidth_tbps; fallback to legacy bandwidth_tbps
        if (gpu.memory_bandwidth_tbps && !isNaN(Number(gpu.memory_bandwidth_tbps))) {
            return Number(gpu.memory_bandwidth_tbps) * 1024; // TB/s → GB/s
        }
        if (gpu.bandwidth_tbps && !isNaN(Number(gpu.bandwidth_tbps))) {
            return Number(gpu.bandwidth_tbps) * 1024; // TB/s → GB/s
        }
        if (gpu.bandwidth && !isNaN(Number(gpu.bandwidth))) {
            return Number(gpu.bandwidth); // already GB/s (from static list)
        }
        return 0;
    }

    // Parse memory_gb that may be numbers or strings like "40 / 80"
    getMemoryGB(gpu) {
        if (!gpu) return 0;
        const v = gpu.memory_gb;
        if (v == null) return 0;
        if (typeof v === 'number') return v;
        const s = String(v);
        const nums = s.match(/[\d.]+/g);
        if (nums && nums.length) {
            const vals = nums.map(n => Number(n)).filter(n => !isNaN(n));
            if (vals.length) return Math.max(...vals);
        }
        const n = Number(s);
        return isNaN(n) ? 0 : n;
    }

    estimateLLMMemoryForPrecision(m, precision, context, concurrency = 1, batch = 1) {
        const bytesPerValue = this.bytesPerValueForPrecision(precision);
        const paramsB = m.parameter_count_billion ?? m.parameters_billion ?? (m.parameters ? m.parameters / 1e9 : null);
        const params = (paramsB ? paramsB * 1e9 : 7.0e9);
        const layers = m.num_layers ?? (params <= 7.0e9 ? 32 : params <= 13.0e9 ? 40 : params <= 70.0e9 ? 80 : params <= 175.0e9 ? 96 : 120);
        const hiddenDim = m.hidden_size ?? (params <= 7.0e9 ? 4096 : params <= 13.0e9 ? 5120 : params <= 70.0e9 ? 8192 : 12288);
        const ctx = Number(context || m.context_length || 4096);

        const quantBytes = this.bytesPerValueForPrecision(precision);
        const weightsGB = (params * quantBytes) / (1024 ** 3);
        const cacheGBBase = (2 * layers * hiddenDim * ctx * concurrency * bytesPerValue) / (1024 ** 3);
        const kvFactorEst = 1 + ((this.currentConfig && typeof this.currentConfig.kvOverheadPercent === 'number') ? (this.currentConfig.kvOverheadPercent / 100) : 0);
        const cacheGB = cacheGBBase * kvFactorEst;
        const actOverheadFactor = (hiddenDim >= 8192 || layers >= 80) ? 1.2 : 1.0;
        const activationGB = (batch * ctx * hiddenDim * bytesPerValue * actOverheadFactor) / (1024 ** 3);
        const subtotal = weightsGB + cacheGB + activationGB;
        // KNOWN ISSUE (preserved): hardcodes 0.3 overhead instead of currentConfig.sysOverheadPercent.
        const overheadGB = subtotal * 0.3;
        const totalGB = subtotal + overheadGB;
        return { weightsGB, cacheGB, activationGB, overheadGB, totalGB, layers, hiddenDim, ctx };
    }

    resolveGPUByName(name) {
        if (!name) return null;
        const n = String(name).trim().toLowerCase();
        return (this.gpuCatalogData || []).find(g => {
            const gn = String(g.name || '').trim().toLowerCase();
            return gn === n || gn.includes(n) || n.includes(gn);
        }) || null;
    }

    efficiencyFactorForPrecision(precision) {
        const p = (precision || '').toLowerCase();
        if (p === 'int8' || p === 'fp8') return EFFICIENCY_INT8_FP8;
        if (p === 'int4' || p === 'fp4' || p === 'int2') return EFFICIENCY_INT4;
        // fp16/bf16 default
        return EFFICIENCY_DEFAULT;
    }

    estimateTokensPerSecondForPair(totalMemoryGB, gpu, precision, llmRefTps = null, refGpuGBps = null) {
        const bwGBps = this.getBandwidthGBps(gpu);
        const eff = this.efficiencyFactorForPrecision(precision);
        if (llmRefTps && refGpuGBps && refGpuGBps > 0) {
            // Scale reference throughput by bandwidth ratio
            const scale = bwGBps / refGpuGBps;
            return Math.max(1, Math.round(llmRefTps * scale));
        }
        // Heuristic based on bandwidth and memory footprint
        return Math.max(1, Math.round(bwGBps / (Math.max(1, totalMemoryGB) * eff)));
    }

    // ===== Section: Catalog sorting =====
    sortValues(a, b, dir = 'asc') {
        const isAsc = dir === 'asc';
        if (a == null && b == null) return 0;
        if (a == null) return 1;
        if (b == null) return -1;
        const aNum = typeof a === 'number' ? a : (typeof a === 'string' && a.trim() !== '' && !isNaN(Number(a)) ? Number(a) : NaN);
        const bNum = typeof b === 'number' ? b : (typeof b === 'string' && b.trim() !== '' && !isNaN(Number(b)) ? Number(b) : NaN);
        if (!isNaN(aNum) && !isNaN(bNum)) return isAsc ? aNum - bNum : bNum - aNum;
        const aStr = Array.isArray(a) ? a.join(', ').toLowerCase() : String(a).toLowerCase();
        const bStr = Array.isArray(b) ? b.join(', ').toLowerCase() : String(b).toLowerCase();
        if (aStr < bStr) return isAsc ? -1 : 1;
        if (aStr > bStr) return isAsc ? 1 : -1;
        return 0;
    }

    sortGPUList(list, key, dir = 'asc') {
        const getVal = (gpu) => {
            switch (key) {
                case 'name': return gpu.name || '';
                case 'vendor': return gpu.vendor || '';
                case 'architecture': return gpu.architecture || '';
                case 'process_node': return gpu.process_node || '';
                case 'memory_gb': return gpu.memory_gb ?? null;
                case 'memory_type': return gpu.memory_type || '';
                case 'memory_bandwidth_tbps': return gpu.memory_bandwidth_tbps ?? gpu.bandwidth_tbps ?? null;
                case 'bandwidth_tbps': return gpu.bandwidth_tbps ?? gpu.memory_bandwidth_tbps ?? null; // legacy key
                case 'fp32_tflops': return gpu.fp32_tflops ?? null;
                case 'fp16_tflops': return gpu.fp16_tflops ?? null;
                case 'int8_tops': return gpu.int8_tops ?? null;
                case 'tdp_w': return gpu.tdp_w ?? null;
                case 'price_usd': return gpu.price_usd ?? null;
                case 'price_rmb': return gpu.price_rmb ?? null;
                case 'nvlink_bandwidth_gbs': return gpu.nvlink_bandwidth_gbs ?? null;
                case 'pcie_generation': return gpu.pcie_generation || '';
                case 'release_year': return gpu.release_year ?? null;
                case 'mig_support': return gpu.mig_support || '';
                case 'transformer_engine': return gpu.transformer_engine || '';
                case 'cuda_cores': return gpu.cuda_cores ?? null;
                case 'tensor_cores': return gpu.tensor_cores || '';
                case 'rt_cores': return gpu.rt_cores || '';
                default: return '';
            }
        };
        return [...(Array.isArray(list) ? list : [])].sort((a, b) => this.sortValues(getVal(a), getVal(b), dir));
    }

    sortLLMList(list, key, dir = 'asc') {
        const paramsB = (m) => (m.parameter_count_billion ?? m.parameters_billion ?? m.parameters ?? null);
        // KNOWN ISSUE (preserved): casing differs from renderLLMCatalog's moeSummary ('disabled' vs 'Disabled').
        const moeSummary = (m) => {
            const moe = m.moe || {};
            if (!moe.enabled) return 'disabled';
            const ne = moe.num_experts != null ? `E:${moe.num_experts}` : '';
            const ae = moe.active_experts != null ? `A:${moe.active_experts}` : '';
            const ep = moe.expert_parallelism ? moe.expert_parallelism : '';
            return [ne, ae, ep].filter(Boolean).join(' ');
        };
        const getVal = (m) => {
            switch (key) {
                case 'model_name': return m.model_name || '';
                case 'release_date': return m.release_date || '';
                case 'params_b': return paramsB(m);
                case 'context_length': return m.context_length ?? null;
                case 'architecture_type': return (m.architecture_type || m.architecture || '');
                case 'organization': return m.organization || '';
                case 'precision_supported': return m.precision_supported || [];
                case 'quantization_types': return m.quantization_types || [];
                case 'moe_summary': return moeSummary(m);
                case 'serving_frameworks': return m.serving_frameworks || [];
                case 'recommended_gpu': return m.recommended_gpu || [];
                case 'throughput_tokens_per_sec_per_gpu': return m.throughput_tokens_per_sec_per_gpu ?? null;
                case 'memory_footprint_gb': return m.memory_footprint_gb ?? null;
                case 'sequence_length_tested': return m.sequence_length_tested ?? null;
                case 'license': return m.license || '';
                case 'num_layers': return m.num_layers ?? null;
                case 'hidden_size': return m.hidden_size ?? null;
                case 'num_attention_heads': return m.num_attention_heads ?? null;
                case 'vocab_size': return m.vocab_size ?? null;
                case 'source_links': return m.source_links || [];
                default: return '';
            }
        };
        return [...(Array.isArray(list) ? list : [])].sort((a, b) => this.sortValues(getVal(a), getVal(b), dir));
    }

    // ===== Section: Logo helpers =====
    getVendorLogoPath(vendor) {
        const v = String(vendor || '').toLowerCase();
        if (v.includes('nvidia')) return 'resources/nvidia.png';
        if (v.includes('alibaba')) return 'resources/alibaba.png';
        if (v.includes('baidu')) return 'resources/baidu.png';
        if (v.includes('huawei')) return 'resources/huawei.png';
        return null;
    }

    getGpuLogoPath(gpu) {
        return gpu?.logo_path || this.getVendorLogoPath(gpu?.vendor) || 'resources/icon-gpu.png';
    }

    getLLMLogoPath(m) {
        if (m?.logo_path) return m.logo_path;
        const org = String(m?.organization || '').toLowerCase();
        const name = String(m?.model_name || '').toLowerCase();
        // Organization-based mapping
        if (org.includes('openai') || name.includes('gpt-oss')) return 'resources/openai.png';
        if (org.includes('zhipu') || org.includes('thudm') || name.includes('glm')) return 'resources/zhipu.png';
        if (org.includes('alibaba') || name.includes('qwen')) return 'resources/alibaba.png';
        if (org.includes('deepseek') || name.includes('deepseek')) return 'resources/deepseek.png';
        if (org.includes('moonshot') || name.includes('kimi')) return 'resources/kimi.png';
        if (org.includes('meta') || name.includes('llama')) return 'resources/llama.png';
        if (org.includes('mistral')) return 'resources/mistral.png';
        return 'resources/icon-llm.svg';
    }

    // ===== Section: Catalog rendering =====
    // Sort direction arrow for a header cell, based on the current sort state
    sortIndicator(sortState, key) {
        return (sortState && sortState.key === key) ? (sortState.dir === 'asc' ? ' ▲' : ' ▼') : '';
    }

    sortableHeaderCell(col, sortState) {
        const alignClass = col.align === 'right' ? 'text-right' : 'text-left';
        return `                    <th data-sort-key="${col.key}" class="${alignClass} px-4 py-2 cursor-pointer select-none whitespace-nowrap">${col.label}${this.sortIndicator(sortState, col.key)}</th>`;
    }

    // Build a sortable catalog table: shared shell (table + thead + tbody) for
    // the GPU and LLM catalog table views. columns: [{ key, label, align }].
    buildSortableTableShell(columns, rowsHtml, sortState) {
        const table = document.createElement('table');
        table.className = 'min-w-full w-full table-auto text-sm bg-white/5 rounded-lg overflow-hidden';

        const thead = document.createElement('thead');
        thead.className = 'bg-white/10 text-soft-gray/80';
        thead.innerHTML = `
                <tr>
${columns.map(col => this.sortableHeaderCell(col, sortState)).join('\n')}
                </tr>
            `;
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        tbody.innerHTML = rowsHtml;
        table.appendChild(tbody);
        return table;
    }

    // Large stat tile used on catalog cards
    statTile(label, value) {
        return `
                        <div class="p-2 bg-white/5 rounded">
                            <div class="text-soft-gray/70">${label}</div>
                            <div class="font-mono text-accent">${value}</div>
                        </div>`;
    }

    // Small inline "Label: value" chip used on catalog cards
    statChip(label, value, extraClass = '') {
        return `<div class="p-2 bg-white/5 rounded${extraClass ? ' ' + extraClass : ''}">${label}: <span class="font-mono">${value}</span></div>`;
    }

    // Three-column grid row of stat tiles/chips.
    // modifierClasses is the class tail, e.g. 'text-sm' or 'text-xs mt-2 text-soft-gray/70'.
    statGridRow(cells, modifierClasses) {
        return `<div class="grid grid-cols-3 gap-2 ${modifierClasses}">${cells.join('')}</div>`;
    }

    gpuCatalogRowHtml(gpu) {
        return `
                <tr class="border-t border-soft-gray/10 hover:bg-white/10">
                    <td class="px-4 py-2">${gpu.name || '-'}</td>
                    <td class="px-4 py-2">${gpu.vendor || '-'}</td>
                    <td class="px-4 py-2">${gpu.architecture || '-'}</td>
                    <td class="px-4 py-2">${gpu.process_node || '-'}</td>
                    <td class="px-4 py-2 text-right">${gpu.memory_gb ?? '-'}</td>
                    <td class="px-4 py-2">${gpu.memory_type || '-'}</td>
                    <td class="px-4 py-2 text-right">${gpu.memory_bandwidth_tbps ?? gpu.bandwidth_tbps ?? '-'}</td>
                    <td class="px-4 py-2 text-right">${gpu.fp32_tflops ?? '-'}</td>
                    <td class="px-4 py-2 text-right">${gpu.int8_tops ?? '-'}</td>
                    <td class="px-4 py-2 text-right">${gpu.tdp_w ?? '-'}</td>
                    <td class="px-4 py-2 text-right">${gpu.price_usd ?? '-'}</td>
                    <td class="px-4 py-2 text-right">${gpu.fp16_tflops ?? '-'}</td>
                    <td class="px-4 py-2 text-right">${gpu.nvlink_bandwidth_gbs ?? '-'}</td>
                    <td class="px-4 py-2">${gpu.pcie_generation || '-'}</td>
                    <td class="px-4 py-2 text-right">${gpu.release_year ?? '-'}</td>
                    <td class="px-4 py-2">${gpu.mig_support || '-'}</td>
                    <td class="px-4 py-2">${gpu.transformer_engine || '-'}</td>
                    <td class="px-4 py-2 text-right">${gpu.cuda_cores ?? '-'}</td>
                    <td class="px-4 py-2">${gpu.tensor_cores || '-'}</td>
                    <td class="px-4 py-2">${gpu.rt_cores || '-'}</td>
                    <td class="px-4 py-2 text-right">${gpu.price_rmb ?? '-'}</td>
                </tr>`;
    }

    gpuCatalogCardHtml(gpu) {
        const perfPerW = (gpu.fp16_tflops && gpu.tdp_w && !isNaN(parseFloat(gpu.tdp_w))) ? (gpu.fp16_tflops / parseFloat(gpu.tdp_w)).toFixed(2) : null;
        const perfPerDollar = (gpu.fp16_tflops && gpu.price_usd) ? (gpu.fp16_tflops / gpu.price_usd).toFixed(2) : null;
        const logoSrc = this.getGpuLogoPath(gpu);
        return `
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center gap-2">
                            <img src="${logoSrc}" alt="${gpu.vendor || 'GPU'}" class="w-6 h-6 rounded-sm">
                            <div class="font-semibold">${gpu.name || '-'}</div>
                        </div>
                        <div class="text-xs text-soft-gray/70">${gpu.vendor || ''}${gpu.architecture ? ' • ' + gpu.architecture : ''}</div>
                    </div>
                    <div class="text-xs text-soft-gray/60 mb-2">Process: ${gpu.process_node || '-'}</div>
                    ${this.statGridRow([
                        this.statTile('Memory', `${gpu.memory_gb ?? '-'} GB`),
                        this.statTile('Bandwidth', `${gpu.memory_bandwidth_tbps ?? gpu.bandwidth_tbps ?? '-'} TB/s`),
                        this.statTile('Memory Type', gpu.memory_type || '-')
                    ], 'text-sm')}
                    ${this.statGridRow([
                        this.statTile('FP32 TFLOPs', gpu.fp32_tflops ?? '-'),
                        this.statTile('INT8 TOPS', gpu.int8_tops ?? '-'),
                        this.statTile('TDP (W)', gpu.tdp_w ?? '-')
                    ], 'text-sm mt-2')}
                    ${this.statGridRow([
                        this.statChip('FP16 TFLOPs', gpu.fp16_tflops ?? '-'),
                        this.statChip('NVLink', gpu.nvlink_bandwidth_gbs ?? '-'),
                        this.statChip('PCIe', gpu.pcie_generation || '-')
                    ], 'text-xs mt-2 text-soft-gray/70')}
                    ${this.statGridRow([
                        this.statChip('Release', gpu.release_year ?? '-'),
                        this.statChip('MIG', gpu.mig_support || '-'),
                        this.statChip('Transformer Engine', gpu.transformer_engine || '-')
                    ], 'text-xs mt-2 text-soft-gray/70')}
                    ${this.statGridRow([
                        this.statChip('CUDA Cores', gpu.cuda_cores ?? '-'),
                        this.statChip('Tensor Cores', gpu.tensor_cores || '-'),
                        this.statChip('RT Cores', gpu.rt_cores || '-')
                    ], 'text-xs mt-2 text-soft-gray/70')}
                    ${this.statGridRow([
                        this.statChip('Price', gpu.price_usd ?? '-'),
                        this.statChip('Perf/W', perfPerW ?? '-'),
                        this.statChip('Perf/$', perfPerDollar ?? '-')
                    ], 'text-xs mt-2 text-soft-gray/70')}
                    ${this.statGridRow([
                        this.statChip('Price RMB', gpu.price_rmb ?? '-')
                    ], 'text-xs mt-2 text-soft-gray/70')}
                    ${gpu.notes ? `<div class="mt-2 text-xs text-soft-gray/60">${gpu.notes}</div>` : ''}
                `;
    }

    // Shared LLM display formatters used by both the LLM table and card views
    formatList(arr) {
        return Array.isArray(arr) ? arr.join(', ') : (arr || '-');
    }

    llmParamsB(m) {
        return (m.parameter_count_billion ?? m.parameters_billion ?? m.parameters ?? '-');
    }

    // KNOWN ISSUE (preserved): casing differs from sortLLMList's moeSummary ('Disabled' vs 'disabled').
    llmMoeSummary(m) {
        const moe = m.moe || {};
        if (!moe.enabled) return 'Disabled';
        const ne = moe.num_experts != null ? `E:${moe.num_experts}` : '';
        const ae = moe.active_experts != null ? `A:${moe.active_experts}` : '';
        const ep = moe.expert_parallelism ? moe.expert_parallelism : '';
        return [ne, ae, ep].filter(Boolean).join(' ');
    }

    llmCatalogRowHtml(m) {
        return `
                <tr class="border-t border-soft-gray/10 hover:bg-white/10">
                    <td class="px-4 py-2">${m.model_name || '-'}</td>
                    <td class="px-4 py-2">${m.release_date || '-'}</td>
                    <td class="px-4 py-2 text-right">${this.llmParamsB(m)}</td>
                    <td class="px-4 py-2 text-right">${m.context_length ?? '-'}</td>
                    <td class="px-4 py-2">${m.architecture_type || m.architecture || '-'}</td>
                    <td class="px-4 py-2 text-right">${m.num_layers ?? '-'}</td>
                    <td class="px-4 py-2 text-right">${m.hidden_size ?? '-'}</td>
                    <td class="px-4 py-2 text-right">${m.num_attention_heads ?? '-'}</td>
                    <td class="px-4 py-2 text-right">${m.vocab_size ?? '-'}</td>
                    <td class="px-4 py-2">${m.organization || '-'}</td>
                    <td class="px-4 py-2">${this.formatList(m.precision_supported)}</td>
                    <td class="px-4 py-2">${this.formatList(m.quantization_types)}</td>
                    <td class="px-4 py-2">${this.llmMoeSummary(m)}</td>
                    <td class="px-4 py-2">${this.formatList(m.serving_frameworks)}</td>
                    <td class="px-4 py-2">${this.formatList(m.recommended_gpu)}</td>
                    <td class="px-4 py-2 text-right">${m.throughput_tokens_per_sec_per_gpu ?? '-'}</td>
                    <td class="px-4 py-2 text-right">${m.memory_footprint_gb ?? '-'}</td>
                    <td class="px-4 py-2">${m.license || '-'}</td>
                    <td class="px-4 py-2">${this.formatList(m.source_links)}</td>
                </tr>`;
    }

    llmCatalogCardHtml(m) {
        const logoSrc = this.getLLMLogoPath(m);
        return `
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center gap-2">
                            <img src="${logoSrc}" alt="${m.organization || 'LLM'}" class="w-6 h-6 rounded-sm">
                            <div class="font-semibold">${m.model_name || '-'}</div>
                        </div>
                        <div class="text-xs text-soft-gray/70">${m.organization || ''}</div>
                    </div>
                    <div class="text-xs text-soft-gray/60 mb-2">Release: ${m.release_date || '-'}</div>
                    ${this.statGridRow([
                        this.statTile('Params', `${this.llmParamsB(m)} B`),
                        this.statTile('Context', m.context_length ?? '-'),
                        this.statTile('Arch', m.architecture_type || m.architecture || '-')
                    ], 'text-sm')}
                    ${this.statGridRow([
                        this.statTile('Precision', this.formatList(m.precision_supported)),
                        this.statTile('Quantization', this.formatList(m.quantization_types)),
                        this.statTile('MoE', this.llmMoeSummary(m))
                    ], 'text-sm mt-2')}
                    ${this.statGridRow([
                        this.statTile('Layers', m.num_layers ?? '-'),
                        this.statTile('Hidden', m.hidden_size ?? '-'),
                        this.statTile('Heads', m.num_attention_heads ?? '-')
                    ], 'text-sm mt-2')}
                    ${this.statGridRow([
                        this.statChip('Serving', this.formatList(m.serving_frameworks)),
                        this.statChip('Rec. GPU', this.formatList(m.recommended_gpu)),
                        this.statChip('License', m.license || '-')
                    ], 'text-xs mt-2 text-soft-gray/70')}
                    ${this.statGridRow([
                        this.statChip('Throughput', m.throughput_tokens_per_sec_per_gpu ?? '-'),
                        this.statChip('Memory', m.memory_footprint_gb ?? '-'),
                        this.statChip('Seq Tested', m.sequence_length_tested ?? '-')
                    ], 'text-xs mt-2 text-soft-gray/70')}
                    ${this.statGridRow([
                        this.statChip('Vocab', m.vocab_size ?? '-'),
                        this.statChip('Sources', this.formatList(m.source_links), 'col-span-2')
                    ], 'text-xs mt-2 text-soft-gray/70')}
                    ${m.notes ? `<div class="mt-2 text-xs text-soft-gray/60">${m.notes}</div>` : ''}
                `;
    }

    // Generic GPU catalog renderer for both GPU and Models pages
    renderGPUCatalog(containerId, viewMode) {
        const container = this.el(containerId);
        if (!container) return;
        container.innerHTML = '';

        if (viewMode === 'table') {
            // Ensure the table spans the full panel width and supports horizontal scroll if needed
            container.className = 'w-full overflow-x-auto';
            // Render table of available GPUs
            const sorted = this.sortGPUList(this.filteredGpuData, this.gpuCatalogSort.key, this.gpuCatalogSort.dir);
            const rowsHtml = sorted.map(gpu => this.gpuCatalogRowHtml(gpu)).join('');
            container.appendChild(this.buildSortableTableShell(GPU_TABLE_COLUMNS, rowsHtml, this.gpuCatalogSort));

            // Sorting handler (delegate to header cells; re-wired on each render)
            container.onclick = (e) => {
                const th = e.target.closest('th[data-sort-key]');
                if (!th) return;
                const key = th.getAttribute('data-sort-key');
                const dir = (this.gpuCatalogSort && this.gpuCatalogSort.key === key && this.gpuCatalogSort.dir === 'asc') ? 'desc' : 'asc';
                this.gpuCatalogSort = { key, dir };
                this.renderGPUCatalog(containerId, viewMode);
            };
        } else {
            // Render cards grid of GPUs: one card per row
            container.className = 'grid grid-cols-1 gap-3';
            container.onclick = null; // disable table-specific handlers
            this.filteredGpuData.forEach(gpu => {
                const card = document.createElement('div');
                card.className = 'p-4 bg-white/5 rounded-lg hover-lift border border-soft-gray/10';
                card.innerHTML = this.gpuCatalogCardHtml(gpu);
                // GPU card - no click handler needed
                container.appendChild(card);
            });
        }
        // Adjust wrapper span so cards occupy full page width when in cards view
        this.setCatalogWrapperSpan(containerId, viewMode === 'cards');
    }

    // LLM catalog renderer (uses JSON-loaded llms)
    renderLLMCatalog(containerId, viewMode) {
        const container = this.el(containerId);
        if (!container) return;
        container.innerHTML = '';

        const list = this.filteredLlmData || [];

        if (viewMode === 'table') {
            // Ensure the table spans the full panel width and supports horizontal scroll if needed
            container.className = 'w-full overflow-x-auto';
            const sorted = this.sortLLMList(list, this.llmCatalogSort.key, this.llmCatalogSort.dir);
            const rowsHtml = sorted.map(m => this.llmCatalogRowHtml(m)).join('');
            container.appendChild(this.buildSortableTableShell(LLM_TABLE_COLUMNS, rowsHtml, this.llmCatalogSort));

            // Sorting handler (delegate to header cells; re-wired on each render)
            container.onclick = (e) => {
                const th = e.target.closest('th[data-sort-key]');
                if (!th) return;
                const key = th.getAttribute('data-sort-key');
                const dir = (this.llmCatalogSort && this.llmCatalogSort.key === key && this.llmCatalogSort.dir === 'asc') ? 'desc' : 'asc';
                this.llmCatalogSort = { key, dir };
                this.renderLLMCatalog(containerId, viewMode);
            };
        } else {
            // Render cards grid of LLMs: one card per row
            container.className = 'grid grid-cols-1 gap-3';
            container.onclick = null; // disable table-specific handlers
            list.forEach(m => {
                const card = document.createElement('div');
                card.className = 'p-4 bg-white/5 rounded-lg hover-lift border border-soft-gray/10';
                card.innerHTML = this.llmCatalogCardHtml(m);
                // LLM card - no click handler needed
                container.appendChild(card);
            });
        }
        // Adjust wrapper span so cards occupy full page width when in cards view
        this.setCatalogWrapperSpan(containerId, viewMode === 'cards');
    }

    // ----- Sliding panel layout helpers -----
    // Toggle a catalog wrapper between full width (cards) and 8/12 width (table)
    setCatalogWrapperSpan(containerId, isCards) {
        const wrapper = this.el(containerId)?.parentElement;
        if (!wrapper) return;
        wrapper.classList.toggle('xl:col-span-12', isCards);
        wrapper.classList.toggle('xl:col-span-8', !isCards);
    }

    // Force a single catalog wrapper to full width
    setWrapperFullWidth(containerId) {
        const wrapper = this.el(containerId)?.parentElement;
        if (!wrapper) return;
        wrapper.classList.add('xl:col-span-12');
        wrapper.classList.remove('xl:col-span-8');
    }

    setFullWidthLayout() {
        // Set both GPU and LLM catalog wrappers to full width
        this.setWrapperFullWidth('gpu-catalog');
        this.setWrapperFullWidth('llm-catalog');
    }

    // ===== Section: Filters =====
    populateFilterOptions() {
        // Populate GPU filter options
        if (this.gpuCatalogData.length > 0) {
            const vendors = [...new Set(this.gpuCatalogData.map(gpu => gpu.vendor).filter(Boolean))].sort();
            const architectures = [...new Set(this.gpuCatalogData.map(gpu => gpu.architecture).filter(Boolean))].sort();

            const vendorSelect = this.el('gpu-vendor-filter');
            const archSelect = this.el('gpu-architecture-filter');

            if (vendorSelect) {
                const options = ['<option value="">All Vendors</option>'];
                vendors.forEach(vendor => {
                    options.push(`<option value="${vendor}">${vendor}</option>`);
                });
                vendorSelect.innerHTML = options.join('');
            }

            if (archSelect) {
                const options = ['<option value="">All Architectures</option>'];
                architectures.forEach(arch => {
                    options.push(`<option value="${arch}">${arch}</option>`);
                });
                archSelect.innerHTML = options.join('');
            }
        }

        // Populate LLM filter options
        if (this.llms.length > 0) {
            // License options are populated dynamically from data
            const licenses = [...new Set(this.llms.map(llm => llm.license).filter(Boolean))].sort();

            // Type options are fixed to Dense and MoE as requested
            const typeSelect = this.el('llm-type-filter');
            const licenseSelect = this.el('llm-license-filter');

            if (typeSelect) {
                typeSelect.innerHTML = '<option value="">All Types</option>' +
                    '<option value="dense">Dense</option>' +
                    '<option value="moe">MoE</option>';
            }

            if (licenseSelect) {
                const options = ['<option value="">All Licenses</option>'];
                licenses.forEach(license => {
                    options.push(`<option value="${license}">${license}</option>`);
                });
                licenseSelect.innerHTML = options.join('');
            }
        }
    }

    applyGPUFilters() {
        let filtered = [...this.gpuCatalogData];

        // Apply vendor filter
        if (this.gpuFilters.vendor) {
            filtered = filtered.filter(gpu => gpu.vendor === this.gpuFilters.vendor);
        }

        // Apply architecture filter
        if (this.gpuFilters.architecture) {
            filtered = filtered.filter(gpu => gpu.architecture === this.gpuFilters.architecture);
        }

        // Apply memory filter
        if (this.gpuFilters.memory) {
            filtered = filtered.filter(gpu => {
                const memory = gpu.memory_gb || 0;
                switch (this.gpuFilters.memory) {
                    case '8-16': return memory >= 8 && memory < 16;
                    case '16-32': return memory >= 16 && memory < 32;
                    case '32-64': return memory >= 32 && memory < 64;
                    case '64+': return memory >= 64;
                    default: return true;
                }
            });
        }

        this.filteredGpuData = filtered;
        this.renderGPUCatalog('gpu-catalog', this.gpuCatalogViewMode);
    }

    applyLLMFilters() {
        let filtered = [...this.llms];

        // Apply size filter
        if (this.llmFilters.size) {
            filtered = filtered.filter(llm => {
                const params = (llm.parameter_count_billion ?? llm.parameters_billion ?? llm.parameters ?? 0);
                switch (this.llmFilters.size) {
                    case 'small': return params < 10;
                    case 'medium': return params >= 10 && params < 50;
                    case 'large': return params >= 50;
                    default: return true;
                }
            });
        }

        // Apply type filter (Dense vs MoE)
        if (this.llmFilters.type) {
            const isMoE = (llm) => {
                const moe = llm.moe || {};
                return !!(moe.enabled || moe.is_enabled);
            };
            if (this.llmFilters.type === 'moe') {
                filtered = filtered.filter(llm => isMoE(llm));
            } else if (this.llmFilters.type === 'dense') {
                filtered = filtered.filter(llm => !isMoE(llm));
            }
        }

        // Apply license filter
        if (this.llmFilters.license) {
            filtered = filtered.filter(llm => llm.license === this.llmFilters.license);
        }

        this.filteredLlmData = filtered;
        this.renderLLMCatalog('llm-catalog', this.llmCatalogViewMode);
    }

    clearGPUFilters() {
        this.gpuFilters = { vendor: '', architecture: '', memory: '' };

        // Reset filter UI
        const vendorSelect = this.el('gpu-vendor-filter');
        const archSelect = this.el('gpu-architecture-filter');
        const memorySelect = this.el('gpu-memory-filter');

        if (vendorSelect) vendorSelect.value = '';
        if (archSelect) archSelect.value = '';
        if (memorySelect) memorySelect.value = '';

        this.filteredGpuData = [...this.gpuCatalogData];
        this.renderGPUCatalog('gpu-catalog', this.gpuCatalogViewMode);
    }

    clearLLMFilters() {
        this.llmFilters = { size: '', type: '', license: '' };

        // Reset filter UI
        const sizeSelect = this.el('llm-size-filter');
        const typeSelect = this.el('llm-type-filter');
        const licenseSelect = this.el('llm-license-filter');

        if (sizeSelect) sizeSelect.value = '';
        if (typeSelect) typeSelect.value = '';
        if (licenseSelect) licenseSelect.value = '';

        this.filteredLlmData = [...this.llms];
        this.renderLLMCatalog('llm-catalog', this.llmCatalogViewMode);
    }

    // ===== Section: SPA page navigation =====
    initPageNavigation() {
        // Hide all sections except currentPage
        this.pageOrder.forEach(id => {
            const section = this.el(id);
            if (!section) return;
            if (id !== this.currentPage) {
                section.classList.add('hidden');
            } else {
                section.classList.remove('hidden');
            }
        });
        this.setActiveNav(this.currentPage);
    }

    setActiveNav(id) {
        const navLinks = document.querySelectorAll('header nav a[href^="#"]');
        navLinks.forEach(link => {
            const targetId = (link.getAttribute('href') || '').replace('#', '');
            if (targetId === id) {
                link.classList.add('text-accent');
            } else {
                link.classList.remove('text-accent');
            }
        });
    }

    switchPage(targetId) {
        if (!targetId || targetId === this.currentPage) return;
        const fromIdx = this.pageOrder.indexOf(this.currentPage);
        const toIdx = this.pageOrder.indexOf(targetId);
        const direction = toIdx > fromIdx ? 1 : -1; // 1: left→right, -1: right→left

        const fromEl = this.el(this.currentPage);
        const toEl = this.el(targetId);
        if (!toEl) return;

        // Prepare target
        toEl.classList.remove('hidden');
        toEl.style.opacity = '0';
        toEl.style.transform = `translateX(${direction === 1 ? 50 : -50}px)`;

        // Animate out old
        if (fromEl) {
            anime({
                targets: fromEl,
                translateX: [0, direction === 1 ? -50 : 50],
                opacity: [1, 0],
                duration: PAGE_SLIDE_OUT_MS,
                easing: 'easeOutCubic',
                complete: () => {
                    fromEl.classList.add('hidden');
                    fromEl.style.transform = '';
                    fromEl.style.opacity = '';
                }
            });
        }

        // Animate in new
        anime({
            targets: toEl,
            translateX: [direction === 1 ? 50 : -50, 0],
            opacity: [0, 1],
            duration: PAGE_SLIDE_IN_MS,
            easing: 'easeOutCubic',
            complete: () => {
                toEl.style.transform = '';
                toEl.style.opacity = '';
                this.currentPage = targetId;
                this.setActiveNav(targetId);
                if (targetId === 'calculator' && this.memoryChart) {
                    // Ensure chart sizes correctly when page becomes visible
                    this.memoryChart.resize();
                }
                const anchor = this.el(targetId);
                if (anchor) {
                    anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }
        });
    }

    // ===== Section: Charts (ECharts memory donut) =====
    initializeMemoryChart() {
        const chartDom = this.el('memory-chart');
        if (!chartDom) {
            this.memoryChart = null;
            return;
        }
        this.memoryChart = echarts.init(chartDom);

        const option = {
            backgroundColor: 'transparent',
            tooltip: {
                trigger: 'item',
                backgroundColor: '#1a202c',
                borderColor: '#3b82f6',
                textStyle: { color: '#e2e8f0' }
            },
            series: [{
                type: 'pie',
                radius: ['40%', '70%'],
                center: ['50%', '50%'],
                data: [
                    { value: 13.0, name: 'Model Weights', itemStyle: { color: '#3b82f6' } },
                    { value: 2.0, name: 'KV Cache', itemStyle: { color: '#ffb347' } },
                    { value: 0.5, name: 'Activation', itemStyle: { color: '#7fb069' } },
                    { value: 4.9, name: 'Overhead', itemStyle: { color: '#94a3b8' } }
                ],
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowOffsetX: 0,
                        shadowColor: 'rgba(0, 212, 255, 0.5)'
                    }
                },
                label: {
                    show: false
                },
                labelLine: {
                    show: false
                }
            }]
        };

        this.memoryChart.setOption(option);
    }

    updateMemoryChart(memory) {
        if (!this.memoryChart) return;
        const option = {
            series: [{
                data: [
                    { value: memory.weights, name: 'Model Weights', itemStyle: { color: '#3b82f6' } },
                    { value: memory.cache, name: 'KV Cache', itemStyle: { color: '#ffb347' } },
                    { value: memory.activation, name: 'Activation', itemStyle: { color: '#7fb069' } },
                    { value: memory.overhead, name: 'Overhead', itemStyle: { color: '#94a3b8' } }
                ]
            }]
        };

        this.memoryChart.setOption(option);
    }

    // ===== Section: Animations =====
    initializeAnimations() {
        // Scroll animations
        this.setupScrollAnimations();

        // Hover effects
        this.setupHoverEffects();
    }

    setupScrollAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    anime({
                        targets: entry.target,
                        opacity: [0, 1],
                        translateY: [20, 0],
                        duration: SCROLL_REVEAL_MS,
                        easing: 'easeOutCubic',
                        delay: anime.stagger(100)
                    });
                }
            });
        }, observerOptions);

        document.querySelectorAll('.hover-lift').forEach(el => {
            observer.observe(el);
        });
    }

    setupHoverEffects() {
        document.querySelectorAll('.hover-lift').forEach(element => {
            element.addEventListener('mouseenter', () => {
                anime({
                    targets: element,
                    translateY: -4,
                    scale: 1.02,
                    duration: HOVER_LIFT_MS,
                    easing: 'easeOutCubic'
                });
            });

            element.addEventListener('mouseleave', () => {
                anime({
                    targets: element,
                    translateY: 0,
                    scale: 1,
                    duration: HOVER_LIFT_MS,
                    easing: 'easeOutCubic'
                });
            });
        });
    }

    // ===== Section: Theme =====
    // NOTE: currently dormant — no caller/markup ('#theme-toggle' is not in index.html); kept intentionally.
    toggleTheme() {
        // Simple theme toggle implementation
        const body = document.body;
        // Consider both possible dark background classes
        const isDark = body.classList.contains('bg-charcoal') || body.classList.contains('bg-deep-charcoal');
        const toggleBtn = this.el('theme-toggle');
        const iconSpan = toggleBtn ? toggleBtn.querySelector('span') : null;

        if (isDark) {
            body.classList.remove('bg-charcoal', 'bg-deep-charcoal', 'text-soft-gray');
            body.classList.add('bg-gray-100', 'text-gray-900');
            if (iconSpan) iconSpan.textContent = '☀️';
        } else {
            body.classList.remove('bg-gray-100', 'text-gray-900');
            body.classList.add('bg-charcoal', 'text-soft-gray');
            if (iconSpan) iconSpan.textContent = '🌙';
        }
    }

    // ===== Section: URL state =====
    // Keep the URL and share link in sync with current configuration
    updateURL() {
        const p = new URLSearchParams();
        const cfg = this.currentConfig;
        if (cfg.model) p.set('model', cfg.model);
        if (cfg.quantization) p.set('q', String(cfg.quantization).toLowerCase());
        if (cfg.contextLength) p.set('ctx', String(cfg.contextLength));
        if (cfg.concurrency) p.set('conc', String(cfg.concurrency));
        if (cfg.batchSize) p.set('batch', String(cfg.batchSize));
        if (cfg.kvOverheadPercent != null) p.set('kvpct', String(cfg.kvOverheadPercent));
        if (cfg.gpuCount) p.set('gpucount', String(cfg.gpuCount));
        if (cfg.selectedGPUModelName) p.set('gpumodel', cfg.selectedGPUModelName);
        if (cfg.vramPerGPU) p.set('vram', String(cfg.vramPerGPU));
        if (cfg.sysOverheadGB != null) p.set('sysgb', String(cfg.sysOverheadGB));
        if (cfg.sysOverheadPercent != null) p.set('syspct', String(cfg.sysOverheadPercent));

        const url = `${location.origin}${location.pathname}?${p.toString()}`;
        try {
            history.replaceState(null, '', url);
        } catch (_) { }

        const shareEl = this.el('shareUrl');
        if (shareEl) {
            shareEl.textContent = url;
        }
    }

    // Load configuration from the current URL and update the UI
    // KNOWN ISSUE (preserved): reads IDs 'context-length-slider', 'batch-size-slider',
    // and 'vram-input', but the actual bindings use 'context-slider', 'batch-slider',
    // and 'vram-per-gpu' — the URL restore silently no-ops for those fields.
    loadFromURL() {
        const params = new URLSearchParams(window.location.search || '');
        const get = (k, d = null) => params.has(k) ? params.get(k) : d;
        const num = (k, d = null) => params.has(k) ? Number(params.get(k)) : d;

        const cfg = this.currentConfig;
        const model = get('model', cfg.model);
        if (model) {
            cfg.model = model;
            const ms = this.el('model-select');
            if (ms) ms.value = model;
            this.toggleCustomModel();
        }

        const q = get('q', cfg.quantization);
        if (q) {
            cfg.quantization = String(q).toLowerCase();
            const radio = document.querySelector(`input[name="quantization"][value="${cfg.quantization}"]`);
            if (radio) radio.checked = true;
            this.updateQuantizationUI();
        }

        const ctx = num('ctx', cfg.contextLength);
        if (ctx != null && !Number.isNaN(ctx)) {
            cfg.contextLength = ctx;
            const slider = this.el('context-length-slider');
            if (slider) slider.value = String(ctx);
            this.safeSet('context-length-value', String(ctx));
        }

        const kvpct = num('kvpct', cfg.kvOverheadPercent);
        if (kvpct != null && !Number.isNaN(kvpct)) {
            cfg.kvOverheadPercent = kvpct;
            const slider = this.el('kv-overhead-slider');
            if (slider) slider.value = String(kvpct);
            this.safeSet('kv-overhead-value', `${kvpct}%`);
        }

        const conc = num('conc', cfg.concurrency);
        if (conc != null && !Number.isNaN(conc)) {
            cfg.concurrency = conc;
            const input = this.el('concurrency-input');
            if (input) input.value = String(conc);
        }

        const batch = num('batch', cfg.batchSize);
        if (batch != null && !Number.isNaN(batch)) {
            cfg.batchSize = batch;
            const slider = this.el('batch-size-slider');
            if (slider) slider.value = String(batch);
            this.safeSet('batch-value', String(batch));
        }

        const gpucount = num('gpucount', cfg.gpuCount);
        if (gpucount != null && !Number.isNaN(gpucount)) {
            cfg.gpuCount = gpucount;
            const input = this.el('gpu-count-input');
            if (input) input.value = String(gpucount);
        }

        const gpumodel = get('gpumodel', cfg.selectedGPUModelName);
        if (gpumodel) {
            cfg.selectedGPUModelName = gpumodel;
            const select = this.el('gpu-model-select');
            if (select) select.value = gpumodel;
        }

        const vram = num('vram', cfg.vramPerGPU);
        if (vram != null && !Number.isNaN(vram)) {
            cfg.vramPerGPU = vram;
            const input = this.el('vram-input');
            if (input) input.value = String(vram);
        }

        const sysgb = num('sysgb', cfg.sysOverheadGB);
        if (sysgb != null && !Number.isNaN(sysgb)) {
            cfg.sysOverheadGB = sysgb;
            const input = this.el('sys-overhead-gb');
            if (input) input.value = String(sysgb);
        }

        const syspct = num('syspct', cfg.sysOverheadPercent);
        if (syspct != null && !Number.isNaN(syspct)) {
            cfg.sysOverheadPercent = syspct;
            const slider = this.el('sys-overhead-slider');
            if (slider) slider.value = String(syspct);
            this.safeSet('sys-overhead-value', `${syspct}%`);
        }

        this.updateCalculations();
    }

    // ===== Section: Utilities =====
    // KNOWN ISSUE (preserved): the `decimals` parameter is ignored (no maximumFractionDigits).
    formatNumber(num, decimals = 0) {
        return new Intl.NumberFormat('en-US').format(num);
    }
}

// ===== Section: Global dialog functions =====
// NOTE: currently dormant — no caller/markup; kept intentionally.
// (Designed for inline onclick use; index.html currently has none.)
function showShareDialog(e) {
    if (e) e.preventDefault();
    const o = document.getElementById('overlay');
    const d = document.getElementById('shareDialog');
    if (o) o.classList.add('open');
    if (d) d.classList.add('open');
    if (window.gpuCalculator && typeof window.gpuCalculator.updateURL === 'function') {
        window.gpuCalculator.updateURL();
    }
}

function closeShareDialog() {
    const o = document.getElementById('overlay');
    const d = document.getElementById('shareDialog');
    if (o) o.classList.remove('open');
    if (d) d.classList.remove('open');
}

function showExplanationDialog(e) {
    if (e) e.preventDefault();
    const o = document.getElementById('overlay');
    const d = document.getElementById('explanationDialog');
    if (o) o.classList.add('open');
    if (d) d.classList.add('open');
}

function closeExplanationDialog() {
    const o = document.getElementById('overlay');
    const d = document.getElementById('explanationDialog');
    if (o) o.classList.remove('open');
    if (d) d.classList.remove('open');
}

function showPerformanceExplanation(e) {
    if (e) e.preventDefault();
    const o = document.getElementById('overlay');
    const d = document.getElementById('performanceExplanationDialog');
    if (o) o.classList.add('open');
    if (d) d.classList.add('open');
}

function closePerformanceExplanation() {
    const o = document.getElementById('overlay');
    const d = document.getElementById('performanceExplanationDialog');
    if (o) o.classList.remove('open');
    if (d) d.classList.remove('open');
}

function closeAllDialogs() {
    const o = document.getElementById('overlay');
    if (o) o.classList.remove('open');
    ['shareDialog', 'explanationDialog', 'performanceExplanationDialog'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.remove('open');
    });
}

async function copyShareLink() {
    const el = document.getElementById('shareUrl');
    if (!el) return;
    const text = el.textContent || '';
    try {
        await navigator.clipboard.writeText(text);
        const btns = document.querySelectorAll('#shareDialog .primary');
        if (btns && btns[0]) {
            const old = btns[0].textContent;
            btns[0].textContent = 'Copied!';
            setTimeout(() => btns[0].textContent = old, 1200);
        }
    } catch (_) {
        // Fallback
        const tmp = document.createElement('textarea');
        tmp.value = text;
        document.body.appendChild(tmp);
        tmp.select();
        try { document.execCommand('copy'); } catch (_) { }
        document.body.removeChild(tmp);
    }
}

// ===== Section: Global utility functions =====
function scrollToCalculator() {
    if (window.gpuCalculator && typeof window.gpuCalculator.switchPage === 'function') {
        window.gpuCalculator.switchPage('calculator');
    } else {
        document.getElementById('calculator').scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// ===== Section: Bootstrap =====
// Initialize the calculator when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.gpuCalculator = new GPUCalculator();
});

// Handle window resize for responsive charts
window.addEventListener('resize', () => {
    if (window.gpuCalculator && window.gpuCalculator.memoryChart) {
        window.gpuCalculator.memoryChart.resize();
    }
});
