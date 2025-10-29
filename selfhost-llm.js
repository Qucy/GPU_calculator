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