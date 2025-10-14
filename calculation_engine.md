# GPU Calculator Engine - Mathematical Foundation

## Memory Calculation Formulas

### 1. Model Weights Memory
```
Model_Weights_Memory = Parameters × Bytes_per_Parameter × Quantization_Factor
```

**Parameters by Model:**
- LLaMA 2 7B: 7,000,000,000 parameters
- LLaMA 2 13B: 13,000,000,000 parameters  
- LLaMA 2 70B: 70,000,000,000 parameters
- GPT-3.5: 175,000,000,000 parameters
- GPT-4: 1,760,000,000,000 parameters
- BERT Base: 110,000,000 parameters
- BERT Large: 340,000,000 parameters
- T5 Small: 60,000,000 parameters
- T5 Base: 220,000,000 parameters
- T5 Large: 770,000,000 parameters
- T5 XL: 3,000,000,000 parameters
- T5 XXL: 11,000,000,000 parameters

**Quantization Factors:**
- FP32: 4 bytes per parameter
- FP16: 2 bytes per parameter
- BF16: 2 bytes per parameter
- INT8: 1 byte per parameter
- INT4: 0.5 bytes per parameter
- INT2: 0.25 bytes per parameter

### 2. KV Cache Memory
```
KV_Cache_Memory = 2 × Num_Layers × Hidden_Dim × Context_Length × Batch_Size × Bytes_per_Value
```

**Model Architecture:**
- LLaMA 2 7B: 32 layers, 4096 hidden dim
- LLaMA 2 13B: 40 layers, 5120 hidden dim
- LLaMA 2 70B: 80 layers, 8192 hidden dim
- GPT-3.5: 96 layers, 12288 hidden dim
- GPT-4: 120 layers, 12288 hidden dim
- BERT: 12/24 layers, 768/1024 hidden dim
- T5: 6-24 layers, 512-1024 hidden dim

**Bytes per Value:**
- FP16/BF16: 2 bytes
- FP32: 4 bytes

### 3. Activation Memory
```
Activation_Memory = Batch_Size × Sequence_Length × Hidden_Dim × Bytes_per_Value × Num_Attention_Heads_Factor
```

**Attention Heads Factor:**
- Standard: 1.0 (for most models)
- Large models: 1.2 (additional overhead)

### 4. Total Memory with Overhead
```
Total_Memory = (Model_Weights + KV_Cache + Activation) × 1.3
```

The 1.3 multiplier accounts for:
- Framework overhead (PyTorch/TensorFlow)
- CUDA kernel memory
- Gradient storage (if training)
- Temporary buffers

## Performance Calculation Formulas

### 1. Inference Speed (Tokens/Second)
```
Tokens_Per_Second = GPU_Memory_Bandwidth / (Total_Memory × Computation_Efficiency_Factor)
```

**GPU Memory Bandwidth:**
- RTX 4090: 1,008 GB/s
- A100: 2,039 GB/s  
- H100: 3,350 GB/s
- V100: 897 GB/s
- T4: 320 GB/s
- L40: 846 GB/s

**Computation Efficiency Factor:**
- FP16: 0.7 (70% efficiency)
- INT8: 0.85 (85% efficiency)
- INT4: 0.9 (90% efficiency)

### 2. Memory Bandwidth Utilization
```
Bandwidth_Utilization = (Model_Weights_Memory × Inference_Rate) / GPU_Memory_Bandwidth
```

### 3. Concurrency Impact
```
Effective_Speed_Per_Request = Tokens_Per_Second / Concurrency_Level
```

## GPU Specifications Database

### Consumer GPUs
- **RTX 4090**: 24GB VRAM, 1,008 GB/s bandwidth, $1,599
- **RTX 4080**: 16GB VRAM, 716.8 GB/s bandwidth, $1,199
- **RTX 4070**: 12GB VRAM, 504 GB/s bandwidth, $799

### Data Center GPUs  
- **H100**: 80GB VRAM, 3,350 GB/s bandwidth, $30,000
- **A100**: 80GB VRAM, 2,039 GB/s bandwidth, $15,000
- **V100**: 32GB VRAM, 897 GB/s bandwidth, $10,000
- **T4**: 16GB VRAM, 320 GB/s bandwidth, $2,000
- **L40**: 48GB VRAM, 846 GB/s bandwidth, $3,500

## Cloud Pricing (Per Hour)
- **A100 (AWS)**: $3.06-$4.10
- **V100 (AWS)**: $1.20-$3.06  
- **T4 (AWS)**: $0.35-$0.53
- **H100 (AWS)**: Expected $5.00-$8.00

## Validation Rules

### Memory Constraints
- Maximum context length: 128,000 tokens
- Maximum batch size: 256
- Maximum concurrency: 1,000
- Minimum quantization: INT2

### Performance Constraints  
- Minimum inference speed: 1 token/second
- Maximum bandwidth utilization: 95%
- Minimum memory headroom: 10%

### Compatibility Checks
- Model must fit in GPU memory (with 20% headroom)
- Context length must be ≤ GPU memory / (hidden_dim × layers × 2)
- Batch size limited by available memory after model weights

## Optimization Suggestions Logic

### Memory Optimization
- If memory > 80% of GPU: Suggest quantization or reduce context
- If context > 32k: Suggest gradient checkpointing
- If concurrency > 100: Suggest model parallelism

### Performance Optimization  
- If speed < 10 tok/s: Suggest smaller model or better GPU
- If utilization < 50%: Suggest increasing batch size
- If concurrency impact > 50%: Suggest load balancing

### Cost Optimization
- Compare cost per token across different GPU types
- Suggest optimal GPU for given throughput requirements
- Recommend spot vs on-demand pricing based on usage patterns