# VERL Model Memory Requirements Analysis

## Overview
This document provides detailed memory calculations for training different sized language models using the VERL framework. Calculations include base memory requirements and the impact of various optimization techniques.

## Memory Calculation Assumptions

### Base Parameters
- **Parameter Precision**: BF16 (2 bytes per parameter)
- **Optimizer**: AdamW (2x parameter memory for momentum and variance)
- **Activation Precision**: BF16 (2 bytes per activation)
- **KV Cache Precision**: BF16 (2 bytes per KV pair)
- **System Overhead**: 2GB per GPU (for CUDA context, libraries, etc.)

### Model Architecture Assumptions
- **Hidden Size**: Calculated as `sqrt(num_parameters / (12 * num_layers))`
- **Vocabulary Size**: 32000 (standard for most LLMs)
- **Sequence Length**: 2048 tokens
- **Batch Size**: 32 (unless specified otherwise)

## 3B Parameter Model Memory Analysis

### Model Specifications
- **Parameters**: 3,000,000,000
- **Hidden Size**: ~2560 (assuming 32 layers)
- **FFN Dimension**: ~7040 (standard 2.75x ratio)
- **Attention Heads**: 32

### Base Memory Requirements

#### Model Parameters
```
3B × 2 bytes = 6 GB
```

#### Gradients
```
3B × 2 bytes = 6 GB
```

#### AdamW Optimizer States
```
3B × 2 bytes × 2 = 12 GB
```

#### Activations (without checkpointing)
```
batch_size × seq_len × hidden_size × num_layers × bytes_per_activation
= 32 × 2048 × 2560 × 32 × 2 bytes
= 32 × 2048 × 2560 × 64 bytes
= 10.7 GB
```

#### KV Cache (for rollout)
```
batch_size × seq_len × num_layers × hidden_size × 2 × bytes_per_kv
= 32 × 2048 × 32 × 2560 × 2 × 2 bytes
= 16.8 GB
```

### Total Memory Without Optimizations
```
6 (params) + 6 (gradients) + 12 (optimizer) + 10.7 (activations) + 16.8 (KV) + 2 (system) = 53.5 GB
```

### With Optimizations

#### With FSDP (8 GPUs)
```
(6 + 6 + 12) / 8 + 10.7 + 16.8 + 2 = 3 + 10.7 + 16.8 + 2 = 32.5 GB per GPU
```

#### With Gradient Checkpointing (checkpoint every 4 layers)
```
10.7 / 4 = 2.7 GB (reduced activations)
Total: 3 + 2.7 + 16.8 + 2 = 24.5 GB per GPU
```

#### With Activation Offloading (70% reduction)
```
2.7 × 0.3 = 0.8 GB (offloaded activations)
Total: 3 + 0.8 + 16.8 + 2 = 22.6 GB per GPU
```

#### With Dynamic Batching (40% reduction in effective batch size)
```
16.8 × 0.6 = 10.1 GB (reduced KV cache)
Total: 3 + 0.8 + 10.1 + 2 = 15.9 GB per GPU
```

## 8B Parameter Model Memory Analysis

### Model Specifications
- **Parameters**: 8,000,000,000
- **Hidden Size**: ~4096 (assuming 32 layers)
- **FFN Dimension**: ~11264 (standard 2.75x ratio)
- **Attention Heads**: 32

### Base Memory Requirements

#### Model Parameters
```
8B × 2 bytes = 16 GB
```

#### Gradients
```
8B × 2 bytes = 16 GB
```

#### AdamW Optimizer States
```
8B × 2 bytes × 2 = 32 GB
```

#### Activations (without checkpointing)
```
32 × 2048 × 4096 × 32 × 2 bytes = 16.8 GB
```

#### KV Cache (for rollout)
```
32 × 2048 × 32 × 4096 × 2 × 2 bytes = 26.8 GB
```

### Total Memory Without Optimizations
```
16 + 16 + 32 + 16.8 + 26.8 + 2 = 109.6 GB
```

### With Optimizations

#### With FSDP (8 GPUs)
```
(16 + 16 + 32) / 8 + 16.8 + 26.8 + 2 = 8 + 16.8 + 26.8 + 2 = 53.6 GB per GPU
```

#### With Gradient Checkpointing
```
16.8 / 4 = 4.2 GB
Total: 8 + 4.2 + 26.8 + 2 = 41 GB per GPU
```

#### With Activation Offloading
```
4.2 × 0.3 = 1.3 GB
Total: 8 + 1.3 + 26.8 + 2 = 38.1 GB per GPU
```

#### With Dynamic Batching
```
26.8 × 0.6 = 16.1 GB
Total: 8 + 1.3 + 16.1 + 2 = 27.4 GB per GPU
```

## 32B Parameter Model Memory Analysis

### Model Specifications
- **Parameters**: 32,000,000,000
- **Hidden Size**: ~5120 (assuming 64 layers)
- **FFN Dimension**: ~14080 (standard 2.75x ratio)
- **Attention Heads**: 40

### Base Memory Requirements

#### Model Parameters
```
32B × 2 bytes = 64 GB
```

#### Gradients
```
32B × 2 bytes = 64 GB
```

#### AdamW Optimizer States
```
32B × 2 bytes × 2 = 128 GB
```

#### Activations (without checkpointing)
```
32 × 2048 × 5120 × 64 × 2 bytes = 42.2 GB
```

#### KV Cache (for rollout)
```
32 × 2048 × 64 × 5120 × 2 × 2 bytes = 84.4 GB
```

### Total Memory Without Optimizations
```
64 + 64 + 128 + 42.2 + 84.4 + 2 = 384.6 GB
```

### With Optimizations

#### With FSDP (16 GPUs)
```
(64 + 64 + 128) / 16 + 42.2 + 84.4 + 2 = 16 + 42.2 + 84.4 + 2 = 144.6 GB per GPU
```

#### With Gradient Checkpointing
```
42.2 / 4 = 10.6 GB
Total: 16 + 10.6 + 84.4 + 2 = 113 GB per GPU
```

#### With Activation Offloading
```
10.6 × 0.3 = 3.2 GB
Total: 16 + 3.2 + 84.4 + 2 = 105.6 GB per GPU
```

#### With Dynamic Batching
```
84.4 × 0.6 = 50.6 GB
Total: 16 + 3.2 + 50.6 + 2 = 71.8 GB per GPU
```

## Summary Table

| Model Size | Base Memory | FSDP Only | + Gradient Checkpointing | + Activation Offloading | + Dynamic Batching |
|------------|-------------|-----------|--------------------------|-------------------------|---------------------|
| 3B         | 53.5 GB     | 32.5 GB   | 24.5 GB                  | 22.6 GB                 | 15.9 GB             |
| 8B         | 109.6 GB    | 53.6 GB   | 41 GB                    | 38.1 GB                 | 27.4 GB             |
| 32B        | 384.6 GB    | 144.6 GB  | 113 GB                   | 105.6 GB                | 71.8 GB             |

## Hardware Requirements

### 3B Model Training
- **Minimum**: 2× A100/H100 (40GB) or 4× RTX 4090 (24GB)
- **Recommended**: 4× A100/H100 (40GB) or 8× RTX 4090 (24GB)
- **Optimal**: 8× A100/H100 (80GB)

### 8B Model Training
- **Minimum**: 4× A100/H100 (40GB) or 8× RTX 4090 (24GB)
- **Recommended**: 8× A100/H100 (40GB) or 16× RTX 4090 (24GB)
- **Optimal**: 8× A100/H100 (80GB)

### 32B Model Training
- **Minimum**: 8× A100/H100 (80GB)
- **Recommended**: 16× A100/H100 (80GB)
- **Optimal**: 32× A100/H100 (80GB)

## VERL Configuration Recommendations

### For 3B Models
```yaml
actor_rollout_ref:
  model:
    enable_gradient_checkpointing: true
    enable_activation_offload: true
  actor:
    use_dynamic_bsz: true
    ppo_max_token_len_per_gpu: 12000
  rollout:
    gpu_memory_utilization: 0.7
    tensor_model_parallel_size: 1
```

### For 8B Models
```yaml
actor_rollout_ref:
  model:
    enable_gradient_checkpointing: true
    enable_activation_offload: true
  actor:
    use_dynamic_bsz: true
    ppo_max_token_len_per_gpu: 16000
  rollout:
    gpu_memory_utilization: 0.6
    tensor_model_parallel_size: 2
```

### For 32B Models
```yaml
actor_rollout_ref:
  model:
    enable_gradient_checkpointing: true
    enable_activation_offload: true
  actor:
    use_dynamic_bsz: true
    ppo_max_token_len_per_gpu: 24000
  rollout:
    gpu_memory_utilization: 0.5
    tensor_model_parallel_size: 4
```

## Performance Considerations

### Memory vs. Speed Trade-offs
1. **More Aggressive Checkpointing**: Lower memory, slower training
2. **Higher Offloading**: Lower memory, more CPU-GPU transfer overhead
3. **Smaller Batch Sizes**: Lower memory, less parallel efficiency
4. **More GPUs**: Lower memory per GPU, higher communication overhead

### Optimization Priority
1. **Always Enable**: Gradient checkpointing (high impact, low overhead)
2. **Highly Recommended**: Activation offloading (significant memory savings)
3. **Recommended**: Dynamic batching (improves efficiency)
4. **Situation-dependent**: FSDP sharding degree (depends on available GPUs)

## Monitoring and Tuning

### Memory Monitoring Commands
```bash
# Monitor GPU memory during training
nvidia-smi -l 1

# VERL built-in memory logging
export VERL_LOGGING_LEVEL=INFO
```

### Key Metrics to Monitor
- **GPU Memory Usage**: Should be < 90% of available memory
- **Memory Fragmentation**: High fragmentation may indicate configuration issues
- **Training Speed**: Monitor tokens/second to ensure optimizations aren't too costly
- **Memory Spikes**: Watch for unexpected memory increases during training