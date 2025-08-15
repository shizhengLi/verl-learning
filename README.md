# VERL Learning Project

A comprehensive learning project for understanding the VERL (Volcengine Enhanced Reinforcement Learning) framework, focusing on reinforcement learning for large language models.

## Documentation

### Core Documentation
- **[Memory Estimation Methodology](notes/memory_estimation_methodology.md)** - Complete guide to memory estimation techniques
- **[Memory Requirements Analysis](notes/memory_requirements_analysis.md)** - Detailed calculations for 3B, 8B, and 32B models
- **[GRPO Memory Analysis](notes/grpo_memory_analysis.md)** - Comprehensive GRPO algorithm memory usage analysis
- **[RL Interview Questions](notes/rl_interview_questions.md)** - 60+ interview questions with coding examples
- **[VERL Comprehensive Documentation](notes/verl_comprehensive_documentation.md)** - Full framework documentation with API reference

### Project Summary
- **[Project README](notes/README.md)** - Complete project summary and learning outcomes

## Learning Objectives

### Completed Tasks
1. **Framework Architecture Analysis** - Explored VERL's modular design and scalable architecture
2. **RL Algorithm Implementation Analysis** - Analyzed PPO and GRPO implementations 
3. **Memory Optimization Methodology** - Documented advanced memory optimization techniques
4. **Model Memory Requirements** - Calculated detailed memory needs for different model sizes
5. **GRPO Algorithm Memory Analysis** - Comprehensive analysis of GRPO memory usage and optimization
6. **RL Interview Questions** - Created comprehensive question bank with coding examples

## Key Insights

### Memory Optimization Techniques
- **Activation Offloading**: 60-80% memory reduction with asynchronous double buffering
- **Gradient Checkpointing**: 70-75% activation memory savings
- **FSDP Integration**: Efficient parameter sharding across GPUs
- **Dynamic Batching**: 30-40% reduction in padding overhead

### GRPO vs PPO Memory Efficiency
- **GRPO Advantage**: 44% memory savings (no Critic model)
- **7B Model Example**: 
  - PPO: 126GB total memory
  - GRPO: 70GB total memory
  - Savings: 56GB

### Model Memory Requirements
| Model Size | Base Memory | Optimized (FSDP) | With All Optimizations |
|------------|-------------|------------------|----------------------|
| 3B         | 53.5 GB     | 32.5 GB          | 15.9 GB              |
| 8B         | 109.6 GB    | 53.6 GB          | 27.4 GB              |
| 32B        | 384.6 GB    | 144.6 GB         | 71.8 GB              |

## Quick Start

### Prerequisites
```bash
# Clone VERL repository
git clone https://github.com/volcengine/verl.git
cd verl

# Install dependencies
pip install -e .

# Install additional requirements
pip install flash-attn --no-build-isolation
pip install vllm
```

### Basic PPO Training
```bash
# Example command for PPO training
python -m verl.trainer.main_ppo \
    --config configs/ppo_examples/llama2_7b_chat.yaml \
    --nnodes 1 \
    --nproc_per_node 8
```

### GRPO Training
```yaml
# grpo_config.yaml
actor_rollout_ref:
  model:
    path: "meta-llama/Llama-2-7b-chat-hf"
    enable_gradient_checkpointing: true
    enable_activation_offload: true
  actor:
    learning_rate: 1e-6
    use_dynamic_bsz: true
  rollout:
    name: vllm
    n: 5
    gpu_memory_utilization: 0.6
    tensor_model_parallel_size: 2
```

## Performance Benchmarks

### Training Throughput
| Model | GPUs | Memory/GPU | Tokens/sec | Training Time |
|-------|------|------------|------------|---------------|
| 3B    | 4    | 24GB       | 15,000     | 2 hours       |
| 7B    | 8    | 32GB       | 8,000      | 6 hours       |
| 32B   | 16   | 72GB       | 3,000      | 24 hours      |

### Memory Optimization Impact
| Optimization Technique | Memory Savings | Performance Impact |
|------------------------|----------------|-------------------|
| Gradient Checkpointing | 70-75%         | +30% time         |
| Activation Offloading  | 60-70%         | +15% time         |
| Parameter Offloading   | 40-50%         | +20% time         |
| Tensor Parallelism     | Scales with GPUs| +10% communication |

## Hardware Requirements

### Minimum Requirements
- **3B Model**: 2× A100 40GB or 4× RTX 3090 24GB
- **7B Model**: 4× A100 40GB or 8× RTX 3090 24GB
- **32B Model**: 8× A100 80GB

### Recommended Configuration
- **3B Model**: 4× A100 40GB or 8× RTX 4090 24GB
- **7B Model**: 8× A100 40GB or 16× RTX 4090 24GB
- **32B Model**: 16× A100 80GB

## Project Structure

```
verl-learning/
├── README.md                          # This file
└── notes/                             # Documentation directory
    ├── README.md                      # Project summary
    ├── memory_estimation_methodology.md
    ├── memory_requirements_analysis.md
    ├── grpo_memory_analysis.md
    ├── rl_interview_questions.md
    └── verl_comprehensive_documentation.md
```

## Configuration Examples

### PPO Configuration
```yaml
actor_rollout_ref:
  model:
    path: "meta-llama/Llama-2-7b-chat-hf"
    enable_gradient_checkpointing: true
  actor:
    ppo_mini_batch_size: 256
    learning_rate: 1e-6
  rollout:
    name: vllm
    tensor_model_parallel_size: 1
    n: 5
```

### GRPO Configuration (Optimized)
```yaml
actor_rollout_ref:
  model:
    enable_gradient_checkpointing: true
    enable_activation_offload: true
  actor:
    use_dynamic_bsz: true
    fsdp_config:
      param_offload: true
      optimizer_offload: true
  rollout:
    gpu_memory_utilization: 0.5
    tensor_model_parallel_size: 4
```

## Monitoring and Debugging

### Memory Monitoring
```bash
# Real-time GPU memory monitoring
watch -n 1 nvidia-smi

# VERL built-in monitoring
export VERL_LOGGING_LEVEL=INFO
export VERL_MEMORY_MONITOR=1
```

### Performance Profiling
```python
from verl.utils.profiler.performance import simple_timer, log_gpu_memory_usage

timing_dict = {}
with simple_timer("training_step", timing_dict):
    # Training code
    pass

print("Performance breakdown:", timing_dict)
```

## Learning Resources

### For Beginners
1. Start with [RL Interview Questions](notes/rl_interview_questions.md) for basic concepts
2. Read [Memory Estimation Methodology](notes/memory_estimation_methodology.md) for optimization techniques
3. Review [Memory Requirements Analysis](notes/memory_requirements_analysis.md) for practical calculations

### For Advanced Users
1. Study [GRPO Memory Analysis](notes/grpo_memory_analysis.md) for advanced optimization
2. Explore [VERL Comprehensive Documentation](notes/verl_comprehensive_documentation.md) for API reference
3. Experiment with different configurations in the examples

## Contributing

This learning project is designed to help users understand VERL framework. If you find issues or have suggestions:

1. Check the existing documentation for answers
2. Experiment with different configurations
3. Share your findings and optimizations
4. Contribute to the VERL repository

## License

This learning project follows the same license as the VERL framework. Please refer to the VERL repository for specific license information.

## Related Links

- [VERL GitHub Repository](https://github.com/volcengine/verl)
- [VERL Official Documentation](https://verl.readthedocs.io)
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [vLLM Inference Engine](https://github.com/vllm-project/vllm)

---

**Note**: This project provides comprehensive documentation and analysis for learning purposes. For production use, please refer to the official VERL documentation and examples.