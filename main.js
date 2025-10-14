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
        this.gpuViewMode = 'cards'; // 'cards' or 'table'
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeAnimations();
        this.initializeParticles();
        this.initializeMemoryChart();
        this.updateCalculations();
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
    document.getElementById('calculator').scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
    });
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