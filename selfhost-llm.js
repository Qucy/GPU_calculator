// URL parameter management
function updateURL() {
    const params = new URLSearchParams();
    
    // GPU Configuration
    const gpuType = document.getElementById('gpu-type').value;
    if (gpuType) params.set('gpu', gpuType);
    params.set('gpu_count', document.getElementById('gpu-count').value);
    params.set('sys_overhead', document.getElementById('system-overhead').value);
    
    // Model Configuration
    const modelInputType = document.querySelector('input[name="model-input-type"]:checked').value;
    params.set('model_type', modelInputType);
    
    if (modelInputType === 'preset') {
        params.set('model', document.getElementById('model-preset').value);
    } else if (modelInputType === 'parameters') {
        params.set('model_params', document.getElementById('model-parameters').value);
    } else if (modelInputType === 'memory') {
        params.set('model_memory', document.getElementById('model-memory-input').value);
    }
    
    // Quantization
    params.set('quant', document.getElementById('quantization').value);
    
    // Context Configuration
    const contextInputType = document.querySelector('input[name="context-input-type"]:checked').value;
    params.set('context_type', contextInputType);
    
    if (contextInputType === 'preset') {
        params.set('context', document.getElementById('context-preset').value);
    } else {
        params.set('context_custom', document.getElementById('context-custom').value);
    }
    
    // KV Cache
    params.set('kv_cache', document.getElementById('kv-cache-overhead').value);
    
    // Update URL without reloading page
    const newURL = window.location.pathname + '?' + params.toString();
    window.history.replaceState({}, '', newURL);
}

function loadFromURL() {
    const params = new URLSearchParams(window.location.search);
    
    // GPU Configuration
    if (params.has('gpu')) {
        document.getElementById('gpu-type').value = params.get('gpu');
        updateGPUSpecs();
    }
    if (params.has('gpu_count')) {
        document.getElementById('gpu-count').value = params.get('gpu_count');
    }
    if (params.has('sys_overhead')) {
        document.getElementById('system-overhead').value = params.get('sys_overhead');
    }
    
    // Model Configuration
    if (params.has('model_type')) {
        const modelType = params.get('model_type');
        document.querySelector(`input[name="model-input-type"][value="${modelType}"]`).checked = true;
        updateModelInputMethod();
        
        if (modelType === 'preset' && params.has('model')) {
            document.getElementById('model-preset').value = params.get('model');
        } else if (modelType === 'parameters' && params.has('model_params')) {
            document.getElementById('model-parameters').value = params.get('model_params');
        } else if (modelType === 'memory' && params.has('model_memory')) {
            document.getElementById('model-memory-input').value = params.get('model_memory');
        }
    }
    
    // Quantization
    if (params.has('quant')) {
        document.getElementById('quantization').value = params.get('quant');
    }
    
    // Context Configuration
    if (params.has('context_type')) {
        const contextType = params.get('context_type');
        document.querySelector(`input[name="context-input-type"][value="${contextType}"]`).checked = true;
        updateContextInputMethod();
        
        if (contextType === 'preset' && params.has('context')) {
            document.getElementById('context-preset').value = params.get('context');
        } else if (contextType === 'custom' && params.has('context_custom')) {
            document.getElementById('context-custom').value = params.get('context_custom');
        }
    }
    
    // KV Cache
    if (params.has('kv_cache')) {
        document.getElementById('kv-cache-overhead').value = params.get('kv_cache');
    }
    
    // Calculate after loading
    calculate();
}

// Get GPU memory bandwidth based on model
function getGPUBandwidth(gpuModel) {
    if (!gpuModel) return 0;
    
    // Prefer bandwidth from the selected option's data attribute if available
    const select = document.getElementById('gpu-type');
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

// Calculate performance estimate
function calculatePerformance(modelMemory, quantization, contextLength, gpuModel, gpuCount) {
    const bandwidth = getGPUBandwidth(gpuModel) * gpuCount;
    if (!bandwidth) return null;
    
    // Get model parameters from preset if available
    let modelParams = 7;
    const modelInputType = document.querySelector('input[name="model-input-type"]:checked').value;
    if (modelInputType === 'preset') {
        const modelSelect = document.getElementById('model-preset');
        modelParams = parseFloat(modelSelect.value) || 7;
    } else if (modelInputType === 'parameters') {
        modelParams = parseFloat(document.getElementById('model-parameters').value) || 7;
    }
    
    // Model size efficiency factor (larger models are less efficient)
    let efficiency = 0.8;
    if (modelParams <= 7) {
        efficiency = 0.85;
    } else if (modelParams <= 30) {
        efficiency = 0.7;
    } else if (modelParams <= 70) {
        efficiency = 0.5;
    } else {
        efficiency = 0.3;
    }
    
    // Quantization speed boost
    let quantBoost = 1.0;
    if (quantization <= 0.25) {
        quantBoost = 2.5;
    } else if (quantization <= 0.3) {
        quantBoost = 2.2;
    } else if (quantization <= 0.5) {
        quantBoost = 1.8;
    } else if (quantization <= 0.75) {
        quantBoost = 1.3;
    }
    
    // Context length impact
    let contextImpact = 1.0;
    if (contextLength >= 131072) {
        contextImpact = 0.3;
    } else if (contextLength >= 32768) {
        contextImpact = 0.6;
    } else if (contextLength >= 8192) {
        contextImpact = 0.85;
    }
    
    // Multi-GPU scaling (not perfect linear scaling)
    let multiGpuScaling = 1.0;
    if (gpuCount > 1) {
        multiGpuScaling = 0.85 + (0.15 / gpuCount);
    }
    
    // Calculate tokens per second
    // Formula: (bandwidth / model_memory_gb) * efficiency * quant_boost * context_impact * scaling
    const baseSpeed = (bandwidth / modelMemory) * efficiency * quantBoost * contextImpact * multiGpuScaling;
    
    // Apply realistic scaling factor
    const tokensPerSecond = baseSpeed * 0.6; // Conservative estimate for datacenter GPUs
    
    return {
        tokensPerSecond: tokensPerSecond,
        bandwidth: bandwidth,
        efficiency: efficiency,
        quantBoost: quantBoost,
        contextImpact: contextImpact,
        multiGpuScaling: multiGpuScaling
    };
}

function updateGPUSpecs() {
    const select = document.getElementById('gpu-type');
    const vramInput = document.getElementById('vram-per-gpu');
    
    if (select.value) {
        const selectedOption = select.options[select.selectedIndex];
        const vram = selectedOption.getAttribute('data-vram');
        vramInput.value = vram;
    } else {
        vramInput.value = '';
    }
    
    calculate();
    if (typeof updateURL === 'function') updateURL();
}

// --- Augment calculator GPU dropdown from JSON catalog ---
async function augmentCalculatorGPUOptionsFromCatalog() {
    const select = document.getElementById('gpu-type');
    if (!select) return;

    // Helper to parse memory_gb values like number or "40 / 80"
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
        const fragments = new Map();
        const getGroup = (label) => {
            let g = groupElements.get(label);
            if (!g) {
                g = ensureGroup(label);
                groupElements.set(label, g);
            }
            return g;
        };
        const getFragment = (label) => {
            let f = fragments.get(label);
            if (!f) {
                f = document.createDocumentFragment();
                fragments.set(label, f);
            }
            // Ensure the group element exists ahead of time
            getGroup(label);
            return f;
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
            opt.textContent = `${base} (${memGB}GB VRAM)`;
            const frag = getFragment(groupLabel);
            frag.appendChild(opt);
        });

        // Flush fragments to their optgroups in one pass
        fragments.forEach((frag, label) => {
            const group = getGroup(label);
            group.appendChild(frag);
        });
    } catch (e) {
        // Silently ignore catalog errors to avoid disrupting existing flow
        console.warn('GPU catalog load failed:', e?.message || e);
    }
}

function updateModelInputMethod() {
    const inputType = document.querySelector('input[name="model-input-type"]:checked').value;
    
    // Toggle visibility using hidden class
    document.getElementById('model-preset-group').classList.toggle('hidden', inputType !== 'preset');
    document.getElementById('model-parameters-group').classList.toggle('hidden', inputType !== 'parameters');
    document.getElementById('model-memory-group').classList.toggle('hidden', inputType !== 'memory');
    
    calculate();
}

function updateModelSelection() {
    const modelSelect = document.getElementById('model-preset');
    // Do not auto-change quantization on model selection; keep user's choice
    calculate();
    if (typeof updateURL === 'function') updateURL();
}

function updateContextInputMethod() {
    const inputType = document.querySelector('input[name="context-input-type"]:checked').value;
    
    // Toggle visibility using hidden class
    document.getElementById('context-preset-group').classList.toggle('hidden', inputType !== 'preset');
    document.getElementById('context-custom-group').classList.toggle('hidden', inputType !== 'custom');
    
    calculate();
}

function calculate() {
    const gpuCount = parseInt(document.getElementById('gpu-count').value) || 1;
    const vramPerGpu = parseFloat(document.getElementById('vram-per-gpu').value) || 0;
    const systemOverhead = parseFloat(document.getElementById('system-overhead').value) || 2;
    
    // Get model memory based on input method
    let modelMemory;
    const modelInputType = document.querySelector('input[name="model-input-type"]:checked').value;
    
    if (modelInputType === 'preset') {
        const modelSelect = document.getElementById('model-preset');
        const selectedOption = modelSelect.options[modelSelect.selectedIndex];
        const activeMemoryAttr = selectedOption.getAttribute('data-active-memory');
        const totalMemoryAttr = selectedOption.getAttribute('data-memory');
        const isMoE = !!activeMemoryAttr;
        const offloadingEnabled = !!document.getElementById('moe-offloading') && document.getElementById('moe-offloading').checked;
        // For VRAM fit: choose active vs total based on offloading toggle
        if (isMoE) {
            modelMemory = offloadingEnabled ? parseFloat(activeMemoryAttr) : (parseFloat(totalMemoryAttr) || 14);
        } else {
            modelMemory = parseFloat(totalMemoryAttr) || 14;
        }
        // Cache MoE context for performance and notes
        window.__isMoESelected = isMoE;
        window.__moeOffloadingEnabled = offloadingEnabled;
        window.__moeActiveMemory = activeMemoryAttr ? parseFloat(activeMemoryAttr) : null;
        window.__moeTotalMemory = totalMemoryAttr ? parseFloat(totalMemoryAttr) : null;
    } else if (modelInputType === 'parameters') {
        const paramCount = parseFloat(document.getElementById('model-parameters').value) || 7;
        modelMemory = paramCount * 2; // Rough estimate: 2GB per billion parameters in FP16
    } else if (modelInputType === 'memory') {
        modelMemory = parseFloat(document.getElementById('model-memory-input').value) || 14;
    }
    
    // Get context length based on input method
    let contextLength;
    const contextInputType = document.querySelector('input[name="context-input-type"]:checked').value;
    
    if (contextInputType === 'preset') {
        contextLength = parseInt(document.getElementById('context-preset').value) || 4096;
    } else {
        contextLength = parseInt(document.getElementById('context-custom').value) || 4096;
    }
    
    const quantization = parseFloat(document.getElementById('quantization').value);
    const kvCacheOverhead = parseFloat(document.getElementById('kv-cache-overhead').value) / 100;
    
    // Calculate memory requirements
    const totalVRAM = gpuCount * vramPerGpu;
    const adjustedModelMemory = modelMemory * quantization;
    
    // Compute KV cache per request using architecture-driven formula
    let kvCachePerRequest;
    const modelSelectForKV = document.getElementById('model-preset');
    const selectedOptionForKV = modelSelectForKV ? modelSelectForKV.options[modelSelectForKV.selectedIndex] : null;
    // KV stored typically in fp16/bf16 → 2 bytes/elem regardless of weight quantization
    const kvBytesPerElem = 2;
    if (selectedOptionForKV) {
        kvCachePerRequest = computeKVCacheGB(contextLength, selectedOptionForKV, kvBytesPerElem, kvCacheOverhead);
    } else {
        // Fallback if not using preset
        kvCachePerRequest = computeKVCacheGB(contextLength, { textContent: `${modelMemory/2}B` }, kvBytesPerElem, kvCacheOverhead);
    }

    const availableMemory = totalVRAM - systemOverhead - adjustedModelMemory;
    
    // Calculate concurrent requests
    const maxConcurrentRequests = availableMemory / kvCachePerRequest;
    const effectiveContext = contextLength;
    
    // Update results
    document.getElementById('total-vram').textContent = totalVRAM.toFixed(1) + ' GB';
    document.getElementById('model-memory').textContent = adjustedModelMemory.toFixed(1) + ' GB';
    document.getElementById('kv-cache-memory').textContent = kvCachePerRequest.toFixed(2) + ' GB';
    document.getElementById('available-memory').textContent = Math.max(0, availableMemory).toFixed(1) + ' GB';
    document.getElementById('concurrent-requests').textContent = Math.max(0, maxConcurrentRequests).toFixed(2);
    document.getElementById('effective-context').textContent = effectiveContext.toLocaleString() + ' tokens';
    
    // Show warnings for insufficient capability
    const warningsDiv = document.getElementById('warnings');
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
    
    // Calculate and display performance if GPU is selected
    const gpuType = document.getElementById('gpu-type').value;
    const performanceSection = document.getElementById('performance-section');
    
    if (gpuType && availableMemory >= 0 && performanceSection) {
        // Align MoE offloading behavior: when ON, use active experts; when OFF, use total
        let perfModelMemoryBase = modelMemory;
        if (window.__isMoESelected) {
            const hasActive = typeof window.__moeActiveMemory === 'number' && !isNaN(window.__moeActiveMemory);
            const hasTotal = typeof window.__moeTotalMemory === 'number' && !isNaN(window.__moeTotalMemory);
            if (window.__moeOffloadingEnabled) {
                perfModelMemoryBase = hasActive ? window.__moeActiveMemory : modelMemory;
            } else {
                perfModelMemoryBase = hasTotal ? window.__moeTotalMemory : modelMemory;
            }
        }
        const perfMemoryAdjusted = perfModelMemoryBase * quantization;
        const perf = calculatePerformance(perfMemoryAdjusted, quantization, contextLength, gpuType, gpuCount);
        
        if (perf) {
            performanceSection.style.display = 'block';
            
            // Update performance metrics
            const tokensPerSec = perf.tokensPerSecond.toFixed(2);
            document.getElementById('tokens-per-second').textContent = `${tokensPerSec} tokens/sec`;
            
            // Generation time for 100 tokens
            const tokensPerSecNum = parseFloat(tokensPerSec);
            const genTime = tokensPerSecNum > 0 ? (100 / tokensPerSecNum).toFixed(1) : 'N/A';
            document.getElementById('generation-time').textContent = `${genTime} seconds`;
            
            // Performance rating
            let rating = '';
            let ratingClass = '';
            if (tokensPerSecNum > 100) {
                rating = '🟢 Excellent';
                ratingClass = 'excellent';
            } else if (tokensPerSecNum > 50) {
                rating = '🟢 Good';
                ratingClass = 'good';
            } else if (tokensPerSecNum > 25) {
                rating = '🟡 Moderate';
                ratingClass = 'moderate';
            } else if (tokensPerSecNum > 10) {
                rating = '🟡 Slow';
                ratingClass = 'slow';
            } else {
                rating = '🔴 Very Slow';
                ratingClass = 'very-slow';
            }
            
            const ratingElement = document.getElementById('performance-rating');
            if (ratingElement) {
                ratingElement.textContent = rating;
                ratingElement.className = `metric-value ${ratingClass}`;
            }
            
            // Performance notes
            const notesDiv = document.getElementById('performance-notes');
            if (notesDiv) {
                let notes = [];
                // MoE mode note at the top
                if (window.__isMoESelected) {
                    const hasActive = typeof window.__moeActiveMemory === 'number' && !isNaN(window.__moeActiveMemory);
                    const hasTotal = typeof window.__moeTotalMemory === 'number' && !isNaN(window.__moeTotalMemory);
                    const totalGB = hasTotal ? (window.__moeTotalMemory * quantization).toFixed(1) : null;
                    const activeGB = hasActive ? (window.__moeActiveMemory * quantization).toFixed(1) : null;
                    const moeLine = window.__moeOffloadingEnabled
                        ? (activeGB ? `• MoE offloading ON: VRAM and performance use active experts (~${activeGB} GB)` : `• MoE offloading ON: Using active experts for calculations`)
                        : (totalGB ? `• MoE offloading OFF: VRAM and performance use full model (~${totalGB} GB)` : `• MoE offloading OFF: Using full model size for calculations`);
                    notes.push(moeLine);
                }
                
                if (tokensPerSecNum < 25) {
                    notes.push('• Consider stronger quantization (INT4) for better speed');
                }
                if (contextLength > 32768 && tokensPerSecNum < 50) {
                    notes.push('• Reduce context length for faster generation');
                }
                if (gpuCount === 1 && tokensPerSecNum < 30) {
                    notes.push('• Consider adding more GPUs for better performance');
                }
                if (tokensPerSecNum > 50) {
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
        } else {
            performanceSection.style.display = 'none';
        }
    } else if (performanceSection) {
        performanceSection.style.display = 'none';
    }
    
    if (totalVRAM > 0) {
        document.getElementById('results').classList.remove('hidden');
    }
    
    // Update URL with current configuration
    if (typeof updateURL === 'function') {
        updateURL();
    }

    // Build/update the performance scenarios table
    if (typeof buildPerformanceScenarioTable === 'function') {
        buildPerformanceScenarioTable();
    }
}

// Build a scenario table showing performance across GPU counts and context lengths
function buildPerformanceScenarioTable() {
    const section = document.getElementById('scenario-table-section');
    const table = document.getElementById('scenario-table');
    const tbody = table ? table.querySelector('tbody') : null;
    if (!section || !tbody) return;

    const gpuTypeEl = document.getElementById('gpu-type');
    const gpuType = gpuTypeEl ? gpuTypeEl.value : '';
    const vramPerGpu = parseFloat(document.getElementById('vram-per-gpu').value) || 0;
    const systemOverhead = parseFloat(document.getElementById('system-overhead').value) || 2;
    const quantization = parseFloat(document.getElementById('quantization').value);
    const kvCacheOverhead = parseFloat(document.getElementById('kv-cache-overhead').value) / 100;

    const modelInputType = document.querySelector('input[name="model-input-type"]:checked').value;
    const modelSelect = document.getElementById('model-preset');
    const selectedModelOption = modelSelect && modelSelect.selectedIndex >= 0 ? modelSelect.options[modelSelect.selectedIndex] : null;

    // If essential selections are missing, hide the section
    if (!gpuType || vramPerGpu <= 0 || !selectedModelOption) {
        section.style.display = 'none';
        return;
    }

    // Determine MoE and memory base consistent with offloading toggle
    const activeMemoryAttr = selectedModelOption.getAttribute('data-active-memory');
    const totalMemoryAttr = selectedModelOption.getAttribute('data-memory');
    const isMoE = !!activeMemoryAttr;
    const offloadingEnabled = !!document.getElementById('moe-offloading') && document.getElementById('moe-offloading').checked;

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
    const currentCount = parseInt(document.getElementById('gpu-count').value) || 1;
    // Build counts around current selection: ±5 window, clip at 1
    const gpuCounts = [];
    const startCount = Math.max(1, currentCount - 5);
    const endCount = currentCount + 5;
    for (let c = startCount; c <= endCount; c++) {
        gpuCounts.push(c);
    }
    // De-duplicate and sort (in case currentCount < 11 and we later expand)
    const seenCounts = new Set();
    const baseCounts = gpuCounts.filter(c => {
        if (seenCounts.has(c)) return false;
        seenCounts.add(c);
        return true;
    }).sort((a, b) => a - b);

    // Include selected context length plus standard tiers
    let selectedContextLength;
    const contextInputType = document.querySelector('input[name="context-input-type"]:checked').value;
    if (contextInputType === 'preset') {
        selectedContextLength = parseInt(document.getElementById('context-preset').value) || 4096;
    } else {
        selectedContextLength = parseInt(document.getElementById('context-custom').value) || 4096;
    }
    // Limit contexts to the requested set: 8K, 16K, 32K, 64K, 128K
    const contexts = [8192, 16384, 32768, 65536, 131072];

    // Helper: derive full vs active parameter counts (billions)
    function deriveModelParamsB() {
        let fullB = null;
        let activeB = null;
        const label = (selectedModelOption.textContent || selectedModelOption.innerText || '').trim();
        // Try catalog first
        if (typeof findLLMConfigFromSelectedOption === 'function') {
            const cfg = findLLMConfigFromSelectedOption(selectedModelOption);
            if (cfg && typeof cfg.parameter_count_billion === 'number') {
                fullB = Number(cfg.parameter_count_billion);
                const moe = cfg.moe || {};
                if (moe.enabled && typeof moe.num_experts === 'number' && typeof moe.active_experts === 'number' && moe.num_experts > 0) {
                    activeB = fullB * (moe.active_experts / moe.num_experts);
                }
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
            const mpEl = document.getElementById('model-parameters');
            const v = mpEl ? parseFloat(mpEl.value) : NaN;
            if (!isNaN(v)) {
                fullB = v;
                activeB = v;
            }
        }
        return { fullB, activeB };
    }

    const { fullB: modelParamsFullB, activeB: modelParamsActiveB } = deriveModelParamsB();

    const formatB = (n) => {
        if (n == null || !isFinite(n)) return '';
        const isInt = Math.abs(n - Math.round(n)) < 1e-9;
        return isInt ? String(Math.round(n)) : n.toFixed(1);
    };
    const paramsDisplay = (() => {
        const f = formatB(modelParamsFullB);
        const a = formatB(modelParamsActiveB);
        if (f && a && f !== a) return `${f} / ${a}`;
        return f || a || '';
    })();

    // Build rows
    const rows = [];
    baseCounts.forEach(gc => {
        const totalVRAM = gc * vramPerGpu;
        const availableMemory = totalVRAM - systemOverhead - perfMemoryAdjusted;
        contexts.forEach(ctx => {
            // KV cache per request for this context
            const kvPerReq = computeKVCacheGB(ctx, selectedModelOption, 2, kvCacheOverhead);
            const maxReqRaw = availableMemory / kvPerReq;
            const maxReq = Math.max(0, maxReqRaw);
            const perf = calculatePerformance(perfMemoryAdjusted, quantization, ctx, gpuType, gc);
            const tpsNum = perf ? Number(perf.tokensPerSecond) : 0;
            const genTimeNum = tpsNum > 0 ? (100 / tpsNum) : Infinity;
            // Use derived display for model params (full / active if available)
            const modelParamsB = paramsDisplay;
            // Filter: must fit model and at least 1 request, context >= 8K, and tps > 0
            const runnable = (availableMemory >= kvPerReq) && (maxReqRaw >= 1) && (ctx >= 8192) && (tpsNum > 0);
            if (runnable) {
                rows.push({
                    model: modelLabel,
                    gpu: gpuLabel || gpuType,
                    gpuCount: gc,
                    quant: quantLabel,
                    context: ctx,
                    maxConcurrent: maxReq.toFixed(2),
                    tokensPerSec: tpsNum.toFixed(2),
                    tokensPerSecNum: tpsNum,
                    genTime: Number.isFinite(genTimeNum) ? `${genTimeNum.toFixed(1)} s` : 'N/A',
                    genTimeNum: genTimeNum,
                    modelParamsB: modelParamsB
                });
            }
        });
    });

    // Guarantee at least 10 recommendations by extending counts upward if needed
    if (rows.length < 10) {
        let gc = (baseCounts.length > 0 ? baseCounts[baseCounts.length - 1] + 1 : currentCount + 1);
        let safety = 0;
        while (rows.length < 10 && safety < 200) {
            const totalVRAM = gc * vramPerGpu;
            const availableMemory = totalVRAM - systemOverhead - perfMemoryAdjusted;
            for (const ctx of contexts) {
                const kvPerReq = computeKVCacheGB(ctx, selectedModelOption, 2, kvCacheOverhead);
                const maxReqRaw = availableMemory / kvPerReq;
                const maxReq = Math.max(0, maxReqRaw);
                const perf = calculatePerformance(perfMemoryAdjusted, quantization, ctx, gpuType, gc);
                const tpsNum = perf ? Number(perf.tokensPerSecond) : 0;
                const genTimeNum = tpsNum > 0 ? (100 / tpsNum) : Infinity;
                const runnable = (availableMemory >= kvPerReq) && (maxReqRaw >= 1) && (ctx >= 8192) && (tpsNum > 0);
                if (runnable) {
                    rows.push({
                        model: modelLabel,
                        gpu: gpuLabel || gpuType,
                        gpuCount: gc,
                        quant: quantLabel,
                        context: ctx,
                        maxConcurrent: maxReq.toFixed(2),
                        tokensPerSec: tpsNum.toFixed(2),
                        tokensPerSecNum: tpsNum,
                        genTime: Number.isFinite(genTimeNum) ? `${genTimeNum.toFixed(1)} s` : 'N/A',
                        genTimeNum: genTimeNum
                    });
                    if (rows.length >= 10) break;
                }
            }
            gc++;
            safety++;
        }
    }

    // Render table
    const toK = (n) => {
        if (n >= 1024) return `${Math.round(n / 1024)}K`;
        return String(n);
    };
    function renderScenarioRows(currentRows) {
        // Persist rows for copy/download and keep in sync with sort
        window.__scenarioRows = currentRows;
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

    // Base rows before filtering/sorting
    window.__scenarioBaseRows = rows;

    // Filtering
    function applyScenarioFilters(baseRows) {
        const ctxSelEl = document.getElementById('scenario-filter-context');
        const minTpsEl = document.getElementById('scenario-filter-min-tps');
        const ctxSel = ctxSelEl ? ctxSelEl.value : 'all';
        const minTps = minTpsEl ? Number(minTpsEl.value) : NaN;
        return baseRows.filter(r => {
            if (ctxSel !== 'all' && r.context !== Number(ctxSel)) return false;
            if (!isNaN(minTps) && r.tokensPerSecNum < minTps) return false;
            return true;
        });
    }

    let filteredRows = applyScenarioFilters(rows);

    // Sorting
    function sortRows(rowsToSort) {
        const prev = window.__scenarioSortState || { key: null, dir: 'asc' };
        const key = prev.key;
        const dir = prev.dir;
        if (!key) return rowsToSort;
        const numericKeys = new Set(['gpuCount', 'context', 'maxConcurrent', 'tokensPerSecNum', 'genTimeNum']);
        const getVal = (row, key) => {
            if (key === 'tokensPerSec') return row.tokensPerSecNum || Number(row.tokensPerSec) || 0;
            return numericKeys.has(key) ? Number(row[key]) : String(row[key] || '').toLowerCase();
        };
        const copy = [...rowsToSort];
        copy.sort((a, b) => {
            const va = getVal(a, key);
            const vb = getVal(b, key);
            if (typeof va === 'number' && typeof vb === 'number' && !isNaN(va) && !isNaN(vb)) {
                return dir === 'asc' ? va - vb : vb - va;
            }
            const cmp = String(va).localeCompare(String(vb));
            return dir === 'asc' ? cmp : -cmp;
        });
        return copy;
    }

    const finalRows = sortRows(filteredRows);
    renderScenarioRows(finalRows);

    // Bind sortable headers (idempotent)
    if (!window.__scenarioSortingBound) {
        const headerCells = table.querySelectorAll('thead th[data-key]');
        const numericKeys = new Set(['gpuCount', 'context', 'maxConcurrent', 'tokensPerSecNum', 'genTimeNum', 'ratingRank']);
        const getVal = (row, key) => {
            if (key === 'tokensPerSec') return row.tokensPerSecNum || Number(row.tokensPerSec) || 0;
            const v = row[key];
            return numericKeys.has(key) ? Number(v) : String(v || '').toLowerCase();
        };
        headerCells.forEach(th => {
            th.addEventListener('click', () => {
                const key = th.getAttribute('data-key');
                const prev = window.__scenarioSortState || { key: null, dir: 'asc' };
                const dir = prev.key === key && prev.dir === 'asc' ? 'desc' : 'asc';
                window.__scenarioSortState = { key, dir };
                const base = Array.isArray(window.__scenarioBaseRows) ? window.__scenarioBaseRows : [];
                const filtered = applyScenarioFilters(base);
                const sorted = sortRows(filtered);
                renderScenarioRows(sorted);

                // Update header sort indicators
                headerCells.forEach(h => { h.classList.remove('sorted-asc', 'sorted-desc'); h.removeAttribute('aria-sort'); });
                th.classList.add(dir === 'asc' ? 'sorted-asc' : 'sorted-desc');
                th.setAttribute('aria-sort', dir);
            });
        });
        window.__scenarioSortingBound = true;
    }

    // Bind filter controls (idempotent)
    if (!window.__scenarioFiltersBound) {
        const ctxSelEl = document.getElementById('scenario-filter-context');
        const minTpsEl = document.getElementById('scenario-filter-min-tps');
        const reapply = () => {
            const base = Array.isArray(window.__scenarioBaseRows) ? window.__scenarioBaseRows : [];
            const filtered = applyScenarioFilters(base);
            const sorted = sortRows(filtered);
            renderScenarioRows(sorted);
        };
        if (ctxSelEl) ctxSelEl.addEventListener('change', reapply);
        if (minTpsEl) minTpsEl.addEventListener('input', reapply);
        window.__scenarioFiltersBound = true;
    }

    section.style.display = rows.length > 0 ? 'block' : 'none';
}

// Copy scenario table as CSV to clipboard
function copyScenarioTable() {
    const rows = Array.isArray(window.__scenarioRows) ? window.__scenarioRows : [];
    if (rows.length === 0) return;
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
    const csv = csvRows.join('\n');
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(csv).catch(() => {});
    }
}

// Download scenario table as CSV
function downloadScenarioTable() {
    const rows = Array.isArray(window.__scenarioRows) ? window.__scenarioRows : [];
    if (rows.length === 0) return;
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
    const csv = csvRows.join('\n');
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

// ASCII Art Style - Circuit Board (GPU version)
const asciiArt = `▓█████ ▓█████  ██▓      █████▒██░ ██  ▒█████    ██████ ▄▄▄█████▓ ██▓     ██▓     ███▄ ▄███▓
▒██    ▒██   ▀ ▓██▒    ▓██   ▒▓██░ ██▒▒██▒  ██▒▒██    ▒ ▓  ██▒ ▓▒▓██▒    ▓██▒    ▓██▒▀█▀ ██▒
▒▓██▄   ▒███   ▓██░    ▒████ ░▒██▀▀██░▒██░  ██▒▒▓██▄    ▒ ▓██░ ▒░▓██░    ▓██░    ▓██    ▓██░
▒██  ▀█▄ ▒▓█  ▄ ▒██▄    ░▓█▒  ░░▓█ ░██ ▒██   ██░ ▒   ██▒░ ▓██▓ ░ ▒██▄    ▒██▄    ▒██    ▒██ 
░██▄▄▄▄██░▒████▒░██████▒░▒█░   ░▓█▒░██▓░ ████▓▒░▒██████▒▒  ▒██▒ ░ ░██████▒░██████▒▒██▒   ░██▒
 ▓█   ▓██▒░ ▒░ ░░ ▒░▓  ░ ▒ ░    ▒ ░░▒░▒░ ▒░▒░▒░ ▒ ▒▓▒ ▒ ░  ▒ ░░   ░ ▒░▓  ░░ ▒░▓  ░░ ▒░   ░  ░`;

// Display ASCII art on page load
function displayAsciiArt() {
    const asciiElement = document.getElementById('ascii-art');
    if (asciiElement) {
        asciiElement.textContent = asciiArt;
    }
}

// Share dialog functions
function showShareDialog() {
    const dialog = document.getElementById('shareDialog');
    const overlay = document.getElementById('overlay');
    const urlContainer = document.getElementById('shareUrl');
    
    urlContainer.textContent = window.location.href;
    
    dialog.classList.add('active');
    overlay.classList.add('active');
}

function closeShareDialog() {
    const dialog = document.getElementById('shareDialog');
    const overlay = document.getElementById('overlay');
    
    dialog.classList.remove('active');
    overlay.classList.remove('active');
}

function copyShareLink() {
    const urlText = document.getElementById('shareUrl').textContent;
    
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
    const dialog = document.getElementById('explanationDialog');
    const overlay = document.getElementById('overlay');
    
    dialog.classList.add('active');
    overlay.classList.add('active');
    overlay.onclick = closeExplanationDialog;
}

// Close explanation dialog
function closeExplanationDialog() {
    const dialog = document.getElementById('explanationDialog');
    const overlay = document.getElementById('overlay');
    
    dialog.classList.remove('active');
    overlay.classList.remove('active');
    overlay.onclick = closeShareDialog;
}

// Show performance explanation dialog
function showPerformanceExplanation(event) {
    event.preventDefault();
    const dialog = document.getElementById('performanceExplanationDialog');
    const overlay = document.getElementById('overlay');
    
    dialog.classList.add('active');
    overlay.classList.add('active');
    overlay.onclick = closePerformanceExplanation;
}

// Close performance explanation dialog
function closePerformanceExplanation() {
    const dialog = document.getElementById('performanceExplanationDialog');
    const overlay = document.getElementById('overlay');
    
    dialog.classList.remove('active');
    overlay.classList.remove('active');
    overlay.onclick = closeShareDialog;
}

// Close dialog on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeShareDialog();
        closeExplanationDialog();
        closePerformanceExplanation();
    }
});

// Initialize on page load
window.onload = async function() {
    displayAsciiArt();
    // Augment GPU dropdown with any missing catalog entries
    await augmentCalculatorGPUOptionsFromCatalog();

    // Load LLM catalog for architecture details (layers, hidden size)
    if (typeof loadLLMCatalog === 'function') {
        try {
            await loadLLMCatalog();
        } catch (e) {
            console.warn('Could not load LLM catalog', e);
        }
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

    // Theme toggle for calculator page
    const themeToggle = document.getElementById('theme-toggle');
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
    const bytesPerElement = bytesPerElem || 2; // fp16/bf16 typical for KV
    const overhead = Math.max(0, overheadFraction || 0);

    // KV bytes = context_len × L × 2 (K+V) × H × bytes_per_elem
    const kvBytes = contextLength * L * 2 * H * bytesPerElement;
    const kvGB = kvBytes / (1024 ** 3);
    return kvGB * (1 + overhead);
}