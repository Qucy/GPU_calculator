// GPU Calculator Pro - Main JavaScript Engine
// Advanced GPU requirements calculator with real-time updates and visual effects

class GPUCalculator {
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
            'int8': 1
        };

        this.gpus = [
            { name: 'H200', vram: 141, bandwidth: 4800, price: 35000, cloudPrice: 6.50 },
            { name: 'H100', vram: 80, bandwidth: 3350, price: 30000, cloudPrice: 5.50 },
            { name: 'H20', vram: 96, bandwidth: 2000, price: 25000, cloudPrice: 4.50 },
            { name: 'A100', vram: 80, bandwidth: 2039, price: 15000, cloudPrice: 3.50 },
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
            customParams: null
        };

        this.memoryChart = null;
        this.gpuViewMode = 'cards'; // Calculator recommendations view: 'cards' or 'table'
        this.gpuCatalogViewMode = 'cards'; // GPU Explorer page
        this.llmCatalogViewMode = 'cards'; // Open Source Models page
        // Catalog sort states
        this.gpuCatalogSort = { key: 'name', dir: 'asc' };
        this.llmCatalogSort = { key: 'model_name', dir: 'asc' };
        // SPA-style page navigation
        this.pageOrder = ['gpu', 'models', 'calculator'];
        this.currentPage = 'gpu';
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeAnimations();
        this.initializeParticles();
        this.initializeMemoryChart();
        this.updateCalculations();

        // Load datasets and render catalogs
        Promise.all([this.loadGPUData(), this.loadLLMData()]).then(() => {
            this.renderGPUCatalog('gpu-catalog', this.gpuCatalogViewMode);
            this.renderLLMCatalog('llm-catalog', this.llmCatalogViewMode);
        }).catch(() => {
            this.renderGPUCatalog('gpu-catalog', this.gpuCatalogViewMode);
            this.renderLLMCatalog('llm-catalog', this.llmCatalogViewMode);
        });

        // Initialize SPA page navigation
        this.initPageNavigation();
    }

    setupEventListeners() {
        // Model selection
        document.getElementById('model-select').addEventListener('change', (e) => {
            this.currentConfig.model = e.target.value;
            this.toggleCustomModel();
            // Clear custom params when not using custom model
            if (this.currentConfig.model !== 'custom') {
                this.currentConfig.customParams = null;
                const customInput = document.getElementById('custom-params');
                if (customInput) customInput.value = '';
            }
            this.updateCalculations();
        });

        // Custom model parameters
        document.getElementById('custom-params').addEventListener('input', (e) => {
            this.currentConfig.customParams = parseFloat(e.target.value) * 1e9;
            this.updateCalculations();
        });

        // Quantization selection
        document.querySelectorAll('input[name="quantization"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.currentConfig.quantization = e.target.value;
                this.updateQuantizationUI();
                this.updateCalculations();
            });
        });

        // Context length slider
        const contextSlider = document.getElementById('context-slider');
        contextSlider.addEventListener('input', (e) => {
            this.currentConfig.contextLength = parseInt(e.target.value);
            document.getElementById('context-value').textContent = this.formatNumber(this.currentConfig.contextLength);
            this.updateCalculations();
        });

        // Context presets
        document.querySelectorAll('.context-preset').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const val = parseInt(btn.getAttribute('data-context'));
                this.currentConfig.contextLength = val;
                contextSlider.value = val;
                document.getElementById('context-value').textContent = this.formatNumber(val);
                this.updateCalculations();
            });
        });

        // Concurrency controls
        document.getElementById('concurrency-inc').addEventListener('click', () => {
            const input = document.getElementById('concurrency-input');
            const value = Math.min(1000, parseInt(input.value) + 1);
            input.value = value;
            this.currentConfig.concurrency = value;
            this.updateCalculations();
        });

        document.getElementById('concurrency-dec').addEventListener('click', () => {
            const input = document.getElementById('concurrency-input');
            const value = Math.max(1, parseInt(input.value) - 1);
            input.value = value;
            this.currentConfig.concurrency = value;
            this.updateCalculations();
        });

        document.getElementById('concurrency-input').addEventListener('input', (e) => {
            const value = Math.max(1, Math.min(1000, parseInt(e.target.value) || 1));
            e.target.value = value;
            this.currentConfig.concurrency = value;
            this.updateCalculations();
        });

        // Batch size slider
        const batchSlider = document.getElementById('batch-slider');
        batchSlider.addEventListener('input', (e) => {
            this.currentConfig.batchSize = parseInt(e.target.value);
            document.getElementById('batch-value').textContent = this.currentConfig.batchSize;
            this.updateCalculations();
        });

        // Batch presets
        document.querySelectorAll('.batch-preset').forEach(btn => {
            btn.addEventListener('click', () => {
                const val = parseInt(btn.getAttribute('data-batch'));
                this.currentConfig.batchSize = val;
                batchSlider.value = val;
                document.getElementById('batch-value').textContent = val;
                this.updateCalculations();
            });
        });

        // Theme toggle
        document.getElementById('theme-toggle').addEventListener('click', () => {
            this.toggleTheme();
        });

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
        const gpuToggleBtn = document.getElementById('gpu-view-toggle');
        const gpuToggleKnob = document.getElementById('gpu-view-knob');
        if (gpuToggleBtn && gpuToggleKnob) {
            const setTogglePosition = () => {
                const isCards = this.gpuViewMode === 'cards';
                gpuToggleBtn.classList.toggle('justify-start', isCards);
                gpuToggleBtn.classList.toggle('justify-end', !isCards);
                gpuToggleBtn.setAttribute('aria-pressed', isCards ? 'false' : 'true');
            };

            // Initialize position
            setTogglePosition();

            gpuToggleBtn.addEventListener('click', () => {
                this.gpuViewMode = this.gpuViewMode === 'cards' ? 'table' : 'cards';
                setTogglePosition();
                this.updateCalculations();
            });
        }

        // GPU catalog toggle (GPU page)
        const catalogToggleBtn = document.getElementById('gpu-catalog-view-toggle');
        const catalogToggleKnob = document.getElementById('gpu-catalog-view-knob');
        if (catalogToggleBtn && catalogToggleKnob) {
            const setCatalogToggle = () => {
                const isCards = this.gpuCatalogViewMode === 'cards';
                catalogToggleBtn.classList.toggle('justify-start', isCards);
                catalogToggleBtn.classList.toggle('justify-end', !isCards);
                catalogToggleBtn.setAttribute('aria-pressed', isCards ? 'false' : 'true');
            };
            setCatalogToggle();
            catalogToggleBtn.addEventListener('click', () => {
                this.gpuCatalogViewMode = this.gpuCatalogViewMode === 'cards' ? 'table' : 'cards';
                setCatalogToggle();
                this.renderGPUCatalog('gpu-catalog', this.gpuCatalogViewMode);
            });
        }

        // LLM catalog toggle (Open Source Models page)
        const llmToggleBtn = document.getElementById('llm-catalog-view-toggle');
        const llmToggleKnob = document.getElementById('llm-catalog-view-knob');
        if (llmToggleBtn && llmToggleKnob) {
            const setLLMToggle = () => {
                const isCards = this.llmCatalogViewMode === 'cards';
                llmToggleBtn.classList.toggle('justify-start', isCards);
                llmToggleBtn.classList.toggle('justify-end', !isCards);
                llmToggleBtn.setAttribute('aria-pressed', isCards ? 'false' : 'true');
            };
            setLLMToggle();
            llmToggleBtn.addEventListener('click', () => {
                this.llmCatalogViewMode = this.llmCatalogViewMode === 'cards' ? 'table' : 'cards';
                setLLMToggle();
                this.renderLLMCatalog('llm-catalog', this.llmCatalogViewMode);
            });
        }

        // Recommendation panels hide buttons
        const gpuRecoHideBtn = document.getElementById('gpu-reco-hide');
        if (gpuRecoHideBtn) {
            gpuRecoHideBtn.addEventListener('click', () => {
                const panel = document.getElementById('gpu-reco-panel');
                if (panel) panel.classList.add('hidden');
            });
        }
        const llmRecoHideBtn = document.getElementById('llm-reco-hide');
        if (llmRecoHideBtn) {
            llmRecoHideBtn.addEventListener('click', () => {
                const panel = document.getElementById('llm-reco-panel');
                if (panel) panel.classList.add('hidden');
            });
        }

        // Hide buttons for recommendation panels
        const hideGpuReco = document.getElementById('gpu-reco-hide');
        if (hideGpuReco) {
            hideGpuReco.addEventListener('click', () => {
                const panel = document.getElementById('gpu-reco-panel');
                if (panel) panel.classList.add('hidden');
            });
        }
        const hideLlmReco = document.getElementById('llm-reco-hide');
        if (hideLlmReco) {
            hideLlmReco.addEventListener('click', () => {
                const panel = document.getElementById('llm-reco-panel');
                if (panel) panel.classList.add('hidden');
            });
        }
    }

    toggleCustomModel() {
        const customDiv = document.getElementById('custom-model-params');
        const customInput = document.getElementById('custom-params');
        if (this.currentConfig.model === 'custom') {
            customDiv.classList.remove('hidden');
            // Initialize default custom params to 7B if unset
            if (!this.currentConfig.customParams) {
                const defaultB = 7.0;
                this.currentConfig.customParams = defaultB * 1e9;
                if (customInput) customInput.value = defaultB;
            }
        } else {
            customDiv.classList.add('hidden');
        }
    }

    updateQuantizationUI() {
        document.querySelectorAll('.quantization-option').forEach(option => {
            const radio = option.querySelector('input[type="radio"]');
            const indicator = option.querySelector('.w-2');
            const border = option.querySelector('.w-4');
            
            if (radio.value === this.currentConfig.quantization) {
                border.classList.add('border-electric');
                border.classList.remove('border-soft-gray/50');
                indicator.classList.add('scale-100');
                indicator.classList.remove('scale-0');
            } else {
                border.classList.remove('border-electric');
                border.classList.add('border-soft-gray/50');
                indicator.classList.remove('scale-100');
                indicator.classList.add('scale-0');
            }
        });
    }

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
        const weightsMemory = (effectiveParams * quantization) / (1024**3); // Convert to GB
        
        // KV cache memory
        // Use concurrency to scale KV cache (each concurrent request maintains its own KV)
        const cacheMemory = (2 * layers * hiddenDim * this.currentConfig.contextLength * this.currentConfig.concurrency * bytesPerValue) / (1024**3);
        
        // Activation memory
        const actOverheadFactor = (hiddenDim >= 8192 || layers >= 80) ? 1.2 : 1.0; // apply overhead only for large models
        const activationMemory = (this.currentConfig.batchSize * this.currentConfig.contextLength * hiddenDim * bytesPerValue * actOverheadFactor) / (1024**3);
        
        // Total with overhead
        const subtotal = weightsMemory + cacheMemory + activationMemory;
        const overheadMemory = subtotal * 0.3; // 30% overhead
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

        // Computation efficiency per docs
        const efficiencyFactor = this.currentConfig.quantization === 'int8' ? 0.85 :
                                 this.currentConfig.quantization === 'int4' ? 0.9 : 0.7; // fp16/bf16 default to 0.7

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

        const weightsGB = (params * quantization) / (1024**3);
        const actOverheadFactor = (hiddenDim >= 8192 || layers >= 80) ? 1.2 : 1.0;
        const kvPerReqGB = (2 * layers * hiddenDim * this.currentConfig.contextLength * bytesPerValue) / (1024**3);
        const actPerReqGB = (this.currentConfig.batchSize * this.currentConfig.contextLength * hiddenDim * bytesPerValue * actOverheadFactor) / (1024**3);
        const perReqGB = kvPerReqGB + actPerReqGB;

        const memBudgetFactor = this.infra.memUtilizationMax;
        const overhead = this.infra.overheadFactor;

        const recommendations = this.gpus.map(gpu => {
            const memBudget = gpu.vram * memBudgetFactor; // usable VRAM target

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
            const memoryUtilization = (memory.total / gpu.vram) * 100;
            const isCompatibleSingle = (weightsGB * overhead) <= memBudget; // fits weights on one GPU

            // Server/rack sizing
            const serversNeeded = Math.ceil(totalGPUsNeeded / this.infra.gpusPerServer);
            const racksNeeded = Math.ceil(serversNeeded / this.infra.serversPerRack);

            return {
                ...gpu,
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

        return recommendations.slice(0, 5);
    }

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

    updateCalculations() {
        const memory = this.calculateMemoryRequirements();
        const performance = this.calculatePerformance(memory);
        const gpuRecommendations = this.getGPURecommendations(memory);
        
        // Update memory display
        document.getElementById('weights-memory').textContent = `${memory.weights.toFixed(1)} GB`;
        document.getElementById('cache-memory').textContent = `${memory.cache.toFixed(1)} GB`;
        document.getElementById('activation-memory').textContent = `${memory.activation.toFixed(1)} GB`;
        document.getElementById('overhead-memory').textContent = `${memory.overhead.toFixed(1)} GB`;
        document.getElementById('total-memory').textContent = `${memory.total.toFixed(1)} GB`;
        
        // Update performance metrics
        this.animateValue('inference-speed', performance.inferenceSpeed);
        this.animateValue('memory-bandwidth', performance.bandwidthUtilization);
        this.animateValue('effective-speed', performance.effectiveSpeed);
        
        // Update parameter count
        const model = this.models[this.currentConfig.model];
        const params = this.currentConfig.customParams || (model ? model.params : 7.0e9);
        document.getElementById('param-count').textContent = this.formatNumber((params || 7.0e9) / 1e9, 1) + 'B';
        
        // Update GPU recommendations
        this.updateGPURecommendations(gpuRecommendations);
        
        // Update memory chart
        this.updateMemoryChart(memory);
    }

    animateValue(elementId, targetValue) {
        const element = document.getElementById(elementId);
        const currentValue = parseInt(element.textContent) || 0;
        
        anime({
            targets: { value: currentValue },
            value: targetValue,
            duration: 800,
            easing: 'easeOutCubic',
            update: function(anim) {
                element.textContent = Math.round(anim.animatables[0].target.value);
            }
        });
    }

    updateGPURecommendations(recommendations) {
        const container = document.getElementById('gpu-recommendations');
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
                    <div class="w-12 h-12 bg-electric/20 rounded-lg flex items-center justify-center">
                        <span class="text-electric font-bold text-sm">GPU</span>
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

    // ----- Data loading for catalogs -----
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

    // ==== Recommendation helpers (LLM ↔ GPU pairing) ====
    bytesPerValueForPrecision(precision) {
        const p = (precision || '').toLowerCase();
        if (p === 'fp32') return 4;
        if (p === 'fp16' || p === 'bf16') return 2;
        // fp8 / int8 / int4 treated as 1 byte per value
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
        if (gpu.bandwidth_tbps && !isNaN(Number(gpu.bandwidth_tbps))) {
            return Number(gpu.bandwidth_tbps) * 1024; // TB/s → GB/s
        }
        if (gpu.bandwidth && !isNaN(Number(gpu.bandwidth))) {
            return Number(gpu.bandwidth); // already GB/s (from static list)
        }
        return 0;
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
        const cacheGB = (2 * layers * hiddenDim * ctx * concurrency * bytesPerValue) / (1024 ** 3);
        const actOverheadFactor = (hiddenDim >= 8192 || layers >= 80) ? 1.2 : 1.0;
        const activationGB = (batch * ctx * hiddenDim * bytesPerValue * actOverheadFactor) / (1024 ** 3);
        const subtotal = weightsGB + cacheGB + activationGB;
        const overheadGB = subtotal * 0.3;
        const totalGB = subtotal + overheadGB;
        return { weightsGB, cacheGB, activationGB, overheadGB, totalGB, layers, hiddenDim, ctx };
    }

    resolveGPUByName(name) {
        if (!name) return null;
        const n = String(name).trim().toLowerCase();
        return (this.gpuCatalogData || []).find(g => String(g.name || '').trim().toLowerCase() === n) || null;
    }

    efficiencyFactorForPrecision(precision) {
        const p = (precision || '').toLowerCase();
        if (p === 'int8' || p === 'fp8') return 0.85;
        if (p === 'int4') return 0.9;
        // fp16/bf16 default
        return 0.7;
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

    renderRecommendationsForGPU(selectedGpu) {
        const panel = document.getElementById('gpu-reco-panel');
        const content = document.getElementById('gpu-reco-content');
        if (!panel || !content) return;
        content.innerHTML = '';
        panel.classList.remove('hidden');

        const vramGB = Number(selectedGpu.memory_gb || 0);
        const vramBudgetGB = vramGB * (this.infra.memUtilizationMax || 0.8);

        const items = (this.llms || []).map(m => {
            const precision = this.chooseDefaultPrecisionForLLM(m);
            const mem = this.estimateLLMMemoryForPrecision(m, precision, m.context_length || 4096, 1, 1);
            const recName = Array.isArray(m.recommended_gpu) ? (m.recommended_gpu[0] || null) : m.recommended_gpu;
            const recGpu = this.resolveGPUByName(recName);
            const llmRefTps = m.throughput_tokens_per_sec_per_gpu || null;
            const refGBps = recGpu ? this.getBandwidthGBps(recGpu) : null;
            const tps = this.estimateTokensPerSecondForPair(mem.totalGB, selectedGpu, precision, llmRefTps, refGBps);
            const fit = mem.totalGB <= vramBudgetGB;
            const utilization = vramGB > 0 ? Math.min(100, Math.round((mem.totalGB / vramGB) * 100)) : 0;
            const shardsNeeded = vramBudgetGB > 0 ? Math.max(1, Math.ceil(mem.totalGB / vramBudgetGB)) : 1;
            return { m, precision, mem, tps, fit, utilization, shardsNeeded };
        });

        // Prefer models that fit; otherwise show top by smallest shards needed
        const fitList = items.filter(x => x.fit).sort((a, b) => b.tps - a.tps).slice(0, 6);
        const nonFitList = items.filter(x => !x.fit).sort((a, b) => a.shardsNeeded - b.shardsNeeded || b.tps - a.tps).slice(0, 6);
        const list = fitList.length > 0 ? fitList : nonFitList;

        list.forEach(({ m, precision, mem, tps, fit, utilization, shardsNeeded }) => {
            const card = document.createElement('div');
            card.className = `p-4 bg-navy/50 rounded-lg border ${fit ? 'border-sage/50' : 'border-amber/50'}`;
            const paramsB = m.parameter_count_billion ?? m.parameters_billion ?? m.parameters ?? '-';
            const ctx = m.context_length || mem.ctx;
            card.innerHTML = `
                <div class="flex items-center justify-between mb-2">
                    <div class="font-semibold">${m.model_name || '-'}</div>
                    <div class="text-xs text-soft-gray/70">${m.organization || ''}</div>
                </div>
                <div class="grid grid-cols-3 gap-2 text-sm">
                    <div class="p-2 bg-navy/40 rounded"><div class="text-soft-gray/70">Params</div><div class="font-mono text-electric">${paramsB}B</div></div>
                    <div class="p-2 bg-navy/40 rounded"><div class="text-soft-gray/70">Precision</div><div class="font-mono text-electric">${precision}</div></div>
                    <div class="p-2 bg-navy/40 rounded"><div class="text-soft-gray/70">Context</div><div class="font-mono text-electric">${ctx}</div></div>
                </div>
                <div class="grid grid-cols-3 gap-2 text-sm mt-2">
                    <div class="p-2 bg-navy/40 rounded"><div class="text-soft-gray/70">VRAM Need</div><div class="font-mono text-electric">${mem.totalGB.toFixed(1)} GB</div></div>
                    <div class="p-2 bg-navy/40 rounded"><div class="text-soft-gray/70">Utilization</div><div class="font-mono ${fit ? 'text-sage' : 'text-amber'}">${utilization}%</div></div>
                    <div class="p-2 bg-navy/40 rounded"><div class="text-soft-gray/70">Tokens/s</div><div class="font-mono text-electric">${tps}</div></div>
                </div>
                <div class="text-xs text-soft-gray/70 mt-2">${fit ? 'Single GPU fits' : 'Requires sharding'}${fit ? '' : ` • Shards: ${shardsNeeded}`}</div>
            `;
            content.appendChild(card);
        });
    }

    renderRecommendationsForLLM(selectedLLM) {
        const panel = document.getElementById('llm-reco-panel');
        const content = document.getElementById('llm-reco-content');
        if (!panel || !content) return;
        content.innerHTML = '';
        panel.classList.remove('hidden');

        const precision = this.chooseDefaultPrecisionForLLM(selectedLLM);
        const mem = this.estimateLLMMemoryForPrecision(selectedLLM, precision, selectedLLM.context_length || 4096, 1, 1);

        const items = (this.gpuCatalogData || []).map(gpu => {
            const vramGB = Number(gpu.memory_gb || 0);
            const vramBudgetGB = vramGB * (this.infra.memUtilizationMax || 0.8);
            const fit = mem.totalGB <= vramBudgetGB;
            const utilization = vramGB > 0 ? Math.min(100, Math.round((mem.totalGB / vramGB) * 100)) : 0;
            const tps = this.estimateTokensPerSecondForPair(mem.totalGB, gpu, precision, selectedLLM.throughput_tokens_per_sec_per_gpu || null, null);
            const shardsNeeded = vramBudgetGB > 0 ? Math.max(1, Math.ceil(mem.totalGB / vramBudgetGB)) : 1;
            return { gpu, tps, fit, utilization, shardsNeeded };
        });

        const fitList = items.filter(x => x.fit).sort((a, b) => b.tps - a.tps).slice(0, 6);
        const nonFitList = items.filter(x => !x.fit).sort((a, b) => a.shardsNeeded - b.shardsNeeded || b.tps - a.tps).slice(0, 6);
        const list = fitList.length > 0 ? fitList : nonFitList;

        list.forEach(({ gpu, tps, fit, utilization, shardsNeeded }) => {
            const card = document.createElement('div');
            card.className = `p-4 bg-navy/50 rounded-lg border ${fit ? 'border-sage/50' : 'border-amber/50'}`;
            const bw = (gpu.bandwidth_tbps ? `${gpu.bandwidth_tbps} TB/s` : (gpu.bandwidth ? `${gpu.bandwidth} GB/s` : '-'));
            card.innerHTML = `
                <div class="flex items-center justify-between mb-2">
                    <div class="font-semibold">${gpu.name || '-'}</div>
                    <div class="text-xs text-soft-gray/70">${gpu.vendor || ''}${gpu.architecture ? ' • ' + gpu.architecture : ''}</div>
                </div>
                <div class="grid grid-cols-3 gap-2 text-sm">
                    <div class="p-2 bg-navy/40 rounded"><div class="text-soft-gray/70">VRAM</div><div class="font-mono text-electric">${gpu.memory_gb ?? '-'} GB</div></div>
                    <div class="p-2 bg-navy/40 rounded"><div class="text-soft-gray/70">Bandwidth</div><div class="font-mono text-electric">${bw}</div></div>
                    <div class="p-2 bg-navy/40 rounded"><div class="text-soft-gray/70">Tokens/s</div><div class="font-mono text-electric">${tps}</div></div>
                </div>
                <div class="grid grid-cols-3 gap-2 text-sm mt-2">
                    <div class="p-2 bg-navy/40 rounded"><div class="text-soft-gray/70">Utilization</div><div class="font-mono ${fit ? 'text-sage' : 'text-amber'}">${utilization}%</div></div>
                    <div class="p-2 bg-navy/40 rounded"><div class="text-soft-gray/70">Fit</div><div class="font-mono">${fit ? 'Single-GPU' : 'Sharded'}</div></div>
                    <div class="p-2 bg-navy/40 rounded"><div class="text-soft-gray/70">Shards</div><div class="font-mono">${fit ? 1 : shardsNeeded}</div></div>
                </div>
            `;
            content.appendChild(card);
        });
    }

    // ----- Catalog sorting helpers -----
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
                case 'bandwidth_tbps': return gpu.bandwidth_tbps ?? null;
                case 'fp32_tflops': return gpu.fp32_tflops ?? null;
                case 'fp16_tflops': return gpu.fp16_tflops ?? null;
                case 'int8_tops': return gpu.int8_tops ?? null;
                case 'tdp_w': return gpu.tdp_w ?? null;
                case 'price_usd': return gpu.price_usd ?? null;
                default: return '';
            }
        };
        return [...(Array.isArray(list) ? list : [])].sort((a, b) => this.sortValues(getVal(a), getVal(b), dir));
    }

    sortLLMList(list, key, dir = 'asc') {
        const paramsB = (m) => (m.parameter_count_billion ?? m.parameters_billion ?? m.parameters ?? null);
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
                default: return '';
            }
        };
        return [...(Array.isArray(list) ? list : [])].sort((a, b) => this.sortValues(getVal(a), getVal(b), dir));
    }

    // Generic GPU catalog renderer for both GPU and Models pages
    renderGPUCatalog(containerId, viewMode) {
        const container = document.getElementById(containerId);
        if (!container) return;
        container.innerHTML = '';

        if (viewMode === 'table') {
            // Ensure the table spans the full panel width and supports horizontal scroll if needed
            container.className = 'w-full overflow-x-auto';
            // Render table of available GPUs
            const table = document.createElement('table');
            table.className = 'min-w-full w-full table-auto text-sm bg-navy/50 rounded-lg overflow-hidden';

            const thead = document.createElement('thead');
            thead.className = 'bg-navy/70 text-soft-gray/80';
            const indicator = (key) => (this.gpuCatalogSort && this.gpuCatalogSort.key === key) ? (this.gpuCatalogSort.dir === 'asc' ? ' ▲' : ' ▼') : '';
            thead.innerHTML = `
                <tr>
                    <th data-sort-key="name" class="text-left px-4 py-2 cursor-pointer select-none whitespace-nowrap">GPU${indicator('name')}</th>
                    <th data-sort-key="vendor" class="text-left px-4 py-2 cursor-pointer select-none whitespace-nowrap">Vendor${indicator('vendor')}</th>
                    <th data-sort-key="architecture" class="text-left px-4 py-2 cursor-pointer select-none whitespace-nowrap">Architecture${indicator('architecture')}</th>
                    <th data-sort-key="process_node" class="text-left px-4 py-2 cursor-pointer select-none whitespace-nowrap">Process${indicator('process_node')}</th>
                    <th data-sort-key="memory_gb" class="text-right px-4 py-2 cursor-pointer select-none whitespace-nowrap">Memory (GB)${indicator('memory_gb')}</th>
                    <th data-sort-key="memory_type" class="text-left px-4 py-2 cursor-pointer select-none whitespace-nowrap">Memory Type${indicator('memory_type')}</th>
                    <th data-sort-key="bandwidth_tbps" class="text-right px-4 py-2 cursor-pointer select-none whitespace-nowrap">Bandwidth (TB/s)${indicator('bandwidth_tbps')}</th>
                    <th data-sort-key="fp32_tflops" class="text-right px-4 py-2 cursor-pointer select-none whitespace-nowrap">FP32 TFLOPs${indicator('fp32_tflops')}</th>
                    <th data-sort-key="int8_tops" class="text-right px-4 py-2 cursor-pointer select-none whitespace-nowrap">INT8 TOPS${indicator('int8_tops')}</th>
                    <th data-sort-key="tdp_w" class="text-right px-4 py-2 cursor-pointer select-none whitespace-nowrap">TDP (W)${indicator('tdp_w')}</th>
                    <th data-sort-key="price_usd" class="text-right px-4 py-2 cursor-pointer select-none whitespace-nowrap">Price (USD)${indicator('price_usd')}</th>
                </tr>
            `;
            table.appendChild(thead);

            const tbody = document.createElement('tbody');
            const sorted = this.sortGPUList(this.gpuCatalogData, this.gpuCatalogSort.key, this.gpuCatalogSort.dir);
            sorted.forEach(gpu => {
                const tr = document.createElement('tr');
                tr.className = 'border-t border-soft-gray/10 hover:bg-navy/40';
                tr.innerHTML = `
                    <td class="px-4 py-2">${gpu.name || '-'}</td>
                    <td class="px-4 py-2">${gpu.vendor || '-'}</td>
                    <td class="px-4 py-2">${gpu.architecture || '-'}</td>
                    <td class="px-4 py-2">${gpu.process_node || '-'}</td>
                    <td class="px-4 py-2 text-right">${gpu.memory_gb ?? '-'}</td>
                    <td class="px-4 py-2">${gpu.memory_type || '-'}</td>
                    <td class="px-4 py-2 text-right">${gpu.bandwidth_tbps ?? '-'}</td>
                    <td class="px-4 py-2 text-right">${gpu.fp32_tflops ?? '-'}</td>
                    <td class="px-4 py-2 text-right">${gpu.int8_tops ?? '-'}</td>
                    <td class="px-4 py-2 text-right">${gpu.tdp_w ?? '-'}</td>
                    <td class="px-4 py-2 text-right">${gpu.price_usd ?? '-'}</td>
                `;
                tbody.appendChild(tr);
            });
            table.appendChild(tbody);
            container.appendChild(table);

            // Sorting handler (delegate to header cells)
            container.onclick = (e) => {
                const th = e.target.closest('th[data-sort-key]');
                if (!th) return;
                const key = th.getAttribute('data-sort-key');
                const dir = (this.gpuCatalogSort && this.gpuCatalogSort.key === key && this.gpuCatalogSort.dir === 'asc') ? 'desc' : 'asc';
                this.gpuCatalogSort = { key, dir };
                this.renderGPUCatalog(containerId, viewMode);
            };
        } else {
            // Render cards grid of GPUs
            container.className = 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3';
            container.onclick = null; // disable table-specific handlers
            this.gpuCatalogData.forEach(gpu => {
                const card = document.createElement('div');
                card.className = 'p-4 bg-navy/50 rounded-lg hover-lift border border-soft-gray/10';
                const perfPerW = (gpu.fp16_tflops && gpu.tdp_w && !isNaN(parseFloat(gpu.tdp_w))) ? (gpu.fp16_tflops / parseFloat(gpu.tdp_w)).toFixed(2) : null;
                const perfPerDollar = (gpu.fp16_tflops && gpu.price_usd) ? (gpu.fp16_tflops / gpu.price_usd).toFixed(2) : null;
                card.innerHTML = `
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center gap-2">
                            <img src="resources/icon-gpu.png" alt="GPU" class="w-6 h-6">
                            <div class="font-semibold">${gpu.name || '-'}</div>
                        </div>
                        <div class="text-xs text-soft-gray/70">${gpu.vendor || ''}${gpu.architecture ? ' • ' + gpu.architecture : ''}</div>
                    </div>
                    <div class="text-xs text-soft-gray/60 mb-2">Process: ${gpu.process_node || '-'}</div>
                    <div class="grid grid-cols-3 gap-2 text-sm">
                        <div class="p-2 bg-navy/40 rounded">
                            <div class="text-soft-gray/70">Memory</div>
                            <div class="font-mono text-electric">${gpu.memory_gb ?? '-'} GB</div>
                        </div>
                        <div class="p-2 bg-navy/40 rounded">
                            <div class="text-soft-gray/70">Bandwidth</div>
                            <div class="font-mono text-electric">${gpu.bandwidth_tbps ?? '-'} TB/s</div>
                        </div>
                        <div class="p-2 bg-navy/40 rounded">
                            <div class="text-soft-gray/70">Memory Type</div>
                            <div class="font-mono text-electric">${gpu.memory_type || '-'}</div>
                        </div>
                    </div>
                    <div class="grid grid-cols-3 gap-2 text-sm mt-2">
                        <div class="p-2 bg-navy/40 rounded">
                            <div class="text-soft-gray/70">FP32 TFLOPs</div>
                            <div class="font-mono text-electric">${gpu.fp32_tflops ?? '-'}</div>
                        </div>
                        <div class="p-2 bg-navy/40 rounded">
                            <div class="text-soft-gray/70">INT8 TOPS</div>
                            <div class="font-mono text-electric">${gpu.int8_tops ?? '-'}</div>
                        </div>
                        <div class="p-2 bg-navy/40 rounded">
                            <div class="text-soft-gray/70">TDP (W)</div>
                            <div class="font-mono text-electric">${gpu.tdp_w ?? '-'}</div>
                        </div>
                    </div>
                    <div class="grid grid-cols-3 gap-2 text-xs mt-2 text-soft-gray/70">
                        <div class="p-2 bg-navy/30 rounded">Price: <span class="font-mono">${gpu.price_usd ?? '-'}</span></div>
                        <div class="p-2 bg-navy/30 rounded">Perf/W: <span class="font-mono">${perfPerW ?? '-'}</span></div>
                        <div class="p-2 bg-navy/30 rounded">Perf/$: <span class="font-mono">${perfPerDollar ?? '-'}</span></div>
                    </div>
                    ${gpu.notes ? `<div class="mt-2 text-xs text-soft-gray/60">${gpu.notes}</div>` : ''}
                `;
                // Click to show recommended LLMs for this GPU
                card.addEventListener('click', () => this.renderRecommendationsForGPU(gpu));
                container.appendChild(card);
            });
        }
    }

    // LLM catalog renderer (uses JSON-loaded llms)
    renderLLMCatalog(containerId, viewMode) {
        const container = document.getElementById(containerId);
        if (!container) return;
        container.innerHTML = '';

        const list = this.llms || [];
        const fmtList = (arr) => Array.isArray(arr) ? arr.join(', ') : (arr || '-');
        const paramsB = (m) => (m.parameter_count_billion ?? m.parameters_billion ?? m.parameters ?? '-');
        const moeSummary = (m) => {
            const moe = m.moe || {};
            if (!moe.enabled) return 'Disabled';
            const ne = moe.num_experts != null ? `E:${moe.num_experts}` : '';
            const ae = moe.active_experts != null ? `A:${moe.active_experts}` : '';
            const ep = moe.expert_parallelism ? moe.expert_parallelism : '';
            return [ne, ae, ep].filter(Boolean).join(' ');
        };

        if (viewMode === 'table') {
            // Ensure the table spans the full panel width and supports horizontal scroll if needed
            container.className = 'w-full overflow-x-auto';
            const table = document.createElement('table');
            table.className = 'min-w-full w-full table-auto text-sm bg-navy/50 rounded-lg overflow-hidden';

            const thead = document.createElement('thead');
            thead.className = 'bg-navy/70 text-soft-gray/80';
            const indicator = (key) => (this.llmCatalogSort && this.llmCatalogSort.key === key) ? (this.llmCatalogSort.dir === 'asc' ? ' ▲' : ' ▼') : '';
            thead.innerHTML = `
                <tr>
                    <th data-sort-key="model_name" class="text-left px-4 py-2 cursor-pointer select-none whitespace-nowrap">Model${indicator('model_name')}</th>
                    <th data-sort-key="release_date" class="text-left px-4 py-2 cursor-pointer select-none whitespace-nowrap">Release${indicator('release_date')}</th>
                    <th data-sort-key="params_b" class="text-right px-4 py-2 cursor-pointer select-none whitespace-nowrap">Params (B)${indicator('params_b')}</th>
                    <th data-sort-key="context_length" class="text-right px-4 py-2 cursor-pointer select-none whitespace-nowrap">Context${indicator('context_length')}</th>
                    <th data-sort-key="architecture_type" class="text-left px-4 py-2 cursor-pointer select-none whitespace-nowrap">Architecture${indicator('architecture_type')}</th>
                    <th data-sort-key="organization" class="text-left px-4 py-2 cursor-pointer select-none whitespace-nowrap">Organization${indicator('organization')}</th>
                    <th data-sort-key="precision_supported" class="text-left px-4 py-2 cursor-pointer select-none whitespace-nowrap">Precision${indicator('precision_supported')}</th>
                    <th data-sort-key="quantization_types" class="text-left px-4 py-2 cursor-pointer select-none whitespace-nowrap">Quantization${indicator('quantization_types')}</th>
                    <th data-sort-key="moe_summary" class="text-left px-4 py-2 cursor-pointer select-none whitespace-nowrap">MoE${indicator('moe_summary')}</th>
                    <th data-sort-key="serving_frameworks" class="text-left px-4 py-2 cursor-pointer select-none whitespace-nowrap">Serving${indicator('serving_frameworks')}</th>
                    <th data-sort-key="recommended_gpu" class="text-left px-4 py-2 cursor-pointer select-none whitespace-nowrap">Recommended GPU${indicator('recommended_gpu')}</th>
                    <th data-sort-key="throughput_tokens_per_sec_per_gpu" class="text-right px-4 py-2 cursor-pointer select-none whitespace-nowrap">Throughput${indicator('throughput_tokens_per_sec_per_gpu')}</th>
                    <th data-sort-key="memory_footprint_gb" class="text-right px-4 py-2 cursor-pointer select-none whitespace-nowrap">Memory (GB)${indicator('memory_footprint_gb')}</th>
                    <th data-sort-key="license" class="text-left px-4 py-2 cursor-pointer select-none whitespace-nowrap">License${indicator('license')}</th>
                </tr>
            `;
            table.appendChild(thead);

            const tbody = document.createElement('tbody');
            const sorted = this.sortLLMList(list, this.llmCatalogSort.key, this.llmCatalogSort.dir);
            sorted.forEach(m => {
                const tr = document.createElement('tr');
                tr.className = 'border-t border-soft-gray/10 hover:bg-navy/40';
                tr.innerHTML = `
                    <td class="px-4 py-2">${m.model_name || '-'}</td>
                    <td class="px-4 py-2">${m.release_date || '-'}</td>
                    <td class="px-4 py-2 text-right">${paramsB(m)}</td>
                    <td class="px-4 py-2 text-right">${m.context_length ?? '-'}</td>
                    <td class="px-4 py-2">${m.architecture_type || m.architecture || '-'}</td>
                    <td class="px-4 py-2">${m.organization || '-'}</td>
                    <td class="px-4 py-2">${fmtList(m.precision_supported)}</td>
                    <td class="px-4 py-2">${fmtList(m.quantization_types)}</td>
                    <td class="px-4 py-2">${moeSummary(m)}</td>
                    <td class="px-4 py-2">${fmtList(m.serving_frameworks)}</td>
                    <td class="px-4 py-2">${fmtList(m.recommended_gpu)}</td>
                    <td class="px-4 py-2 text-right">${m.throughput_tokens_per_sec_per_gpu ?? '-'}</td>
                    <td class="px-4 py-2 text-right">${m.memory_footprint_gb ?? '-'}</td>
                    <td class="px-4 py-2">${m.license || '-'}</td>
                `;
                tbody.appendChild(tr);
            });
            table.appendChild(tbody);
            container.appendChild(table);
            // Sorting handler (delegate to header cells)
            container.onclick = (e) => {
                const th = e.target.closest('th[data-sort-key]');
                if (!th) return;
                const key = th.getAttribute('data-sort-key');
                const dir = (this.llmCatalogSort && this.llmCatalogSort.key === key && this.llmCatalogSort.dir === 'asc') ? 'desc' : 'asc';
                this.llmCatalogSort = { key, dir };
                this.renderLLMCatalog(containerId, viewMode);
            };
        } else {
            container.className = 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3';
            container.onclick = null; // disable table-specific handlers
            list.forEach(m => {
                const card = document.createElement('div');
                card.className = 'p-4 bg-navy/50 rounded-lg hover-lift border border-soft-gray/10';
                card.innerHTML = `
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center gap-2">
                            <img src="resources/icon-llm.png" alt="LLM" class="w-6 h-6">
                            <div class="font-semibold">${m.model_name || '-'}</div>
                        </div>
                        <div class="text-xs text-soft-gray/70">${m.organization || ''}</div>
                    </div>
                    <div class="text-xs text-soft-gray/60 mb-2">Release: ${m.release_date || '-'}</div>
                    <div class="grid grid-cols-3 gap-2 text-sm">
                        <div class="p-2 bg-navy/40 rounded">
                            <div class="text-soft-gray/70">Params</div>
                            <div class="font-mono text-electric">${paramsB(m)} B</div>
                        </div>
                        <div class="p-2 bg-navy/40 rounded">
                            <div class="text-soft-gray/70">Context</div>
                            <div class="font-mono text-electric">${m.context_length ?? '-'}</div>
                        </div>
                        <div class="p-2 bg-navy/40 rounded">
                            <div class="text-soft-gray/70">Arch</div>
                            <div class="font-mono text-electric">${m.architecture_type || m.architecture || '-'}</div>
                        </div>
                    </div>
                    <div class="grid grid-cols-3 gap-2 text-sm mt-2">
                        <div class="p-2 bg-navy/40 rounded">
                            <div class="text-soft-gray/70">Precision</div>
                            <div class="font-mono text-electric">${fmtList(m.precision_supported)}</div>
                        </div>
                        <div class="p-2 bg-navy/40 rounded">
                            <div class="text-soft-gray/70">Quantization</div>
                            <div class="font-mono text-electric">${fmtList(m.quantization_types)}</div>
                        </div>
                        <div class="p-2 bg-navy/40 rounded">
                            <div class="text-soft-gray/70">MoE</div>
                            <div class="font-mono text-electric">${moeSummary(m)}</div>
                        </div>
                    </div>
                    <div class="grid grid-cols-3 gap-2 text-xs mt-2 text-soft-gray/70">
                        <div class="p-2 bg-navy/30 rounded">Serving: <span class="font-mono">${fmtList(m.serving_frameworks)}</span></div>
                        <div class="p-2 bg-navy/30 rounded">Rec. GPU: <span class="font-mono">${fmtList(m.recommended_gpu)}</span></div>
                        <div class="p-2 bg-navy/30 rounded">License: <span class="font-mono">${m.license || '-'}</span></div>
                    </div>
                    <div class="grid grid-cols-3 gap-2 text-xs mt-2 text-soft-gray/70">
                        <div class="p-2 bg-navy/30 rounded">Throughput: <span class="font-mono">${m.throughput_tokens_per_sec_per_gpu ?? '-'}</span></div>
                        <div class="p-2 bg-navy/30 rounded">Memory: <span class="font-mono">${m.memory_footprint_gb ?? '-'}</span></div>
                        <div class="p-2 bg-navy/30 rounded">Seq Tested: <span class="font-mono">${m.sequence_length_tested ?? '-'}</span></div>
                    </div>
                    ${m.notes ? `<div class="mt-2 text-xs text-soft-gray/60">${m.notes}</div>` : ''}
                `;
                // Click to show recommended GPUs for this LLM
                card.addEventListener('click', () => this.renderRecommendationsForLLM(m));
                container.appendChild(card);
            });
        }
    }

    // ----- SPA Page Navigation -----
    initPageNavigation() {
        // Hide all sections except currentPage
        this.pageOrder.forEach(id => {
            const section = document.getElementById(id);
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
                link.classList.add('text-electric');
            } else {
                link.classList.remove('text-electric');
            }
        });
    }

    switchPage(targetId) {
        if (!targetId || targetId === this.currentPage) return;
        const fromIdx = this.pageOrder.indexOf(this.currentPage);
        const toIdx = this.pageOrder.indexOf(targetId);
        const direction = toIdx > fromIdx ? 1 : -1; // 1: left→right, -1: right→left

        const fromEl = document.getElementById(this.currentPage);
        const toEl = document.getElementById(targetId);
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
                duration: 250,
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
            duration: 300,
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
                const anchor = document.getElementById(targetId);
                if (anchor) {
                    anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }
        });
    }

    

    updateOptimizationTips(tips) {
        const container = document.getElementById('optimization-tips');
        container.innerHTML = '';
        
        tips.forEach(tip => {
            const div = document.createElement('div');
            const colorClass = tip.type === 'success' ? 'border-sage' : 
                              tip.type === 'warning' ? 'border-amber' : 'border-electric';
            const iconClass = tip.type === 'success' ? 'text-sage' : 
                             tip.type === 'warning' ? 'text-amber' : 'text-electric';
            
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

    initializeMemoryChart() {
        const chartDom = document.getElementById('memory-chart');
        this.memoryChart = echarts.init(chartDom);
        
        const option = {
            backgroundColor: 'transparent',
            tooltip: {
                trigger: 'item',
                backgroundColor: '#1a202c',
                borderColor: '#00d4ff',
                textStyle: { color: '#e2e8f0' }
            },
            series: [{
                type: 'pie',
                radius: ['40%', '70%'],
                center: ['50%', '50%'],
                data: [
                    { value: 13.0, name: 'Model Weights', itemStyle: { color: '#00d4ff' } },
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
        const option = {
            series: [{
                data: [
                    { value: memory.weights, name: 'Model Weights', itemStyle: { color: '#00d4ff' } },
                    { value: memory.cache, name: 'KV Cache', itemStyle: { color: '#ffb347' } },
                    { value: memory.activation, name: 'Activation', itemStyle: { color: '#7fb069' } },
                    { value: memory.overhead, name: 'Overhead', itemStyle: { color: '#94a3b8' } }
                ]
            }]
        };
        
        this.memoryChart.setOption(option);
    }

    initializeAnimations() {
        // Initialize text splitting for animations
        Splitting();
        
        // Typewriter effect
        if (document.getElementById('typed-text')) {
            new Typed('#typed-text', {
                strings: [
                    'Optimize your GPU requirements with precision calculations',
                    'Get instant recommendations for AI and ML workloads',
                    'Professional-grade accuracy for developers and engineers'
                ],
                typeSpeed: 50,
                backSpeed: 30,
                backDelay: 2000,
                loop: true,
                showCursor: true,
                cursorChar: '|'
            });
        }
        
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
                        duration: 800,
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
                    duration: 300,
                    easing: 'easeOutCubic'
                });
            });
            
            element.addEventListener('mouseleave', () => {
                anime({
                    targets: element,
                    translateY: 0,
                    scale: 1,
                    duration: 300,
                    easing: 'easeOutCubic'
                });
            });
        });
    }

    initializeParticles() {
        // P5.js particle system
        new p5((p) => {
            let particles = [];
            
            p.setup = function() {
                const canvas = p.createCanvas(p.windowWidth, p.windowHeight);
                canvas.parent('particles-canvas');
                
                // Create particles
                for (let i = 0; i < 50; i++) {
                    particles.push({
                        x: p.random(p.width),
                        y: p.random(p.height),
                        vx: p.random(-0.5, 0.5),
                        vy: p.random(-0.5, 0.5),
                        size: p.random(2, 6),
                        opacity: p.random(0.1, 0.3)
                    });
                }
            };
            
            p.draw = function() {
                p.clear();
                
                // Update and draw particles
                particles.forEach(particle => {
                    particle.x += particle.vx;
                    particle.y += particle.vy;
                    
                    // Wrap around edges
                    if (particle.x < 0) particle.x = p.width;
                    if (particle.x > p.width) particle.x = 0;
                    if (particle.y < 0) particle.y = p.height;
                    if (particle.y > p.height) particle.y = 0;
                    
                    // Draw particle
                    p.fill(0, 212, 255, particle.opacity * 255);
                    p.noStroke();
                    p.circle(particle.x, particle.y, particle.size);
                });
                
                // Draw connections
                for (let i = 0; i < particles.length; i++) {
                    for (let j = i + 1; j < particles.length; j++) {
                        const dist = p.dist(particles[i].x, particles[i].y, particles[j].x, particles[j].y);
                        if (dist < 100) {
                            p.stroke(0, 212, 255, (1 - dist / 100) * 50);
                            p.strokeWeight(1);
                            p.line(particles[i].x, particles[i].y, particles[j].x, particles[j].y);
                        }
                    }
                }
            };
            
            p.windowResized = function() {
                p.resizeCanvas(p.windowWidth, p.windowHeight);
            };
        });
    }

    toggleTheme() {
        // Simple theme toggle implementation
        const body = document.body;
        const isDark = body.classList.contains('bg-charcoal');
        
        if (isDark) {
            body.classList.remove('bg-charcoal', 'text-soft-gray');
            body.classList.add('bg-gray-100', 'text-gray-900');
        } else {
            body.classList.remove('bg-gray-100', 'text-gray-900');
            body.classList.add('bg-charcoal', 'text-soft-gray');
        }
    }

    

    formatNumber(num, decimals = 0) {
        return new Intl.NumberFormat('en-US').format(num);
    }
}

// Utility functions
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

function showDocumentation() {
    document.getElementById('documentation').scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
    });
}

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