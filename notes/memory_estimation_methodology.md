# VERL Memory Usage Estimation Methodology

## Overview
VERL (Volcengine Enhanced Reinforcement Learning) is a distributed reinforcement learning framework designed for training large language models using PPO (Proximal Policy Optimization) and other RL algorithms. This document outlines the memory estimation methodology for VERL training jobs.

## Memory Components

### 1. Model Parameters Memory
The fundamental memory requirement for storing model weights:

```
model_params_memory = num_parameters × parameter_size
```

Where:
- `num_parameters`: Total number of model parameters
- `parameter_size`: Bytes per parameter (typically 2 for BF16/FP16, 4 for FP32)

### 2. Gradient Memory
During training, gradients require the same memory as parameters:

```
gradient_memory = num_parameters × parameter_size
```

### 3. Optimizer States Memory
Different optimizers have varying memory requirements:

#### Adam Optimizer
```
adam_memory = num_parameters × parameter_size × 2  # momentum + variance
```

#### AdamW Optimizer
```
adamw_memory = num_parameters × parameter_size × 2  # momentum + variance
```

#### 8-bit Adam (Adam8bit)
```
adam8bit_memory = num_parameters × 1  # 8-bit quantized states
```

### 4. Activations Memory
Memory required for storing intermediate activations during forward/backward passes:

```
activations_memory = batch_size × sequence_length × hidden_size × num_layers × bytes_per_activation
```

### 5. KV Cache Memory (for inference/rollout)
For transformer models during inference:

```
kv_cache_memory = batch_size × sequence_length × num_layers × hidden_size × 2 × bytes_per_kv
```

### 6. Communication Buffers
For distributed training with FSDP (Fully Sharded Data Parallel):

```
communication_memory = num_parameters × parameter_size × sharding_factor
```

## VERL-Specific Memory Optimizations

### 1. Activation Offloading
VERL implements sophisticated activation offloading mechanisms:

- **Synchronized Offloading**: Blocks computation during transfers
- **Asynchronous Double Buffering**: Overlaps computation with memory transfers
- **Group-based Offloading**: Offloads activations in groups to balance memory and performance

Memory reduction: Up to 60-80% reduction in activation memory usage.

### 2. Gradient Checkpointing
Reduces activation memory by recomputing activations during backward pass:

```
checkpointed_memory = activations_memory / checkpoint_ratio
```

Where `checkpoint_ratio` is typically 2-4 depending on the checkpointing strategy.

### 3. FSDP (Fully Sharded Data Parallel)
Shards model parameters, gradients, and optimizer states across GPUs:

```
fsdp_memory = (model_params + gradients + optimizer_states) / num_gpus
```

### 4. Dynamic Batching
Optimizes memory usage by processing variable-length sequences efficiently:

```
dynamic_memory = max_token_limit × hidden_size × bytes_per_activation
```

## Memory Estimation Formula

### Total Memory per GPU
```
total_memory_per_gpu = (
    model_params_memory / fsdp_degree +
    gradient_memory / fsdp_degree +
    optimizer_states_memory / fsdp_degree +
    activations_memory_with_checkpointing +
    kv_cache_memory +
    communication_buffers +
    system_overhead
)
```

### Simplified Estimation
For quick estimation during training configuration:

```
total_memory_per_gpu = num_parameters × parameter_size × memory_multiplier
```

Where `memory_multiplier` varies by configuration:
- Basic training: ~4-6× (params + gradients + Adam + activations)
- With checkpointing: ~2-3×
- With FSDP + checkpointing: ~1.5-2×
- With activation offloading: ~1.2-1.8×

## VERL Configuration Impact

### Training Strategy Impact
1. **FSDP vs Megatron**: FSDP typically uses 20-30% less memory than tensor parallelism
2. **Activation Offloading**: Can reduce memory usage by 60-80% for activations
3. **Gradient Checkpointing**: Reduces activation memory by 50-75%
4. **Dynamic Batching**: Can improve memory efficiency by 20-40%

### Model Architecture Impact
1. **Model Size**: Larger models benefit more from distributed strategies
2. **Context Length**: Longer sequences increase KV cache and activation memory
3. **Batch Size**: Larger batches increase activation and KV cache memory linearly
4. **Vocabulary Size**: Larger vocabularies increase final layer memory usage

## Memory Monitoring in VERL

VERL provides built-in memory monitoring tools:

```python
from verl.utils.profiler.performance import log_gpu_memory_usage, GPUMemoryLogger

# Log memory usage
log_gpu_memory_usage("Training step", logger=logger)

# Decorator for automatic memory logging
@GPUMemoryLogger(role="actor")
def update_actor(self, batch):
    # Training logic
    pass
```

## Practical Guidelines

### For 3B Parameter Models
- **Minimum GPU Memory**: 16GB (with optimizations)
- **Recommended**: 24GB+ for comfortable training
- **Key Optimizations**: FSDP, gradient checkpointing

### For 8B Parameter Models
- **Minimum GPU Memory**: 40GB (with optimizations)
- **Recommended**: 80GB+ for comfortable training
- **Key Optimizations**: FSDP, activation offloading, gradient checkpointing

### For 32B Parameter Models
- **Minimum GPU Memory**: 160GB (with optimizations)
- **Recommended**: 320GB+ for comfortable training
- **Key Optimizations**: Multi-node FSDP, activation offloading, gradient checkpointing

## Memory Estimation Examples

See `memory_calculations.md` for detailed calculations specific to different model sizes and configurations.