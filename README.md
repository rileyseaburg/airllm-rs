# AirLLM-RS

High-performance layer-wise LLM inference in Rust with Python bindings.

## Features

- **Layer-wise Loading**: Load transformer layers one at a time, enabling inference on models that don't fit in GPU memory
- **Memory-mapped I/O**: Zero-copy loading from safetensors files using mmap
- **Async Prefetching**: Prefetch next layer while computing current layer
- **Multi-architecture**: Support for Llama, Qwen, GLM-4/GLM-4.7, Mixtral
- **BitNet Support**: Ternary weight unpacking for 1.58-bit models
- **Python Bindings**: Easy-to-use Python API via PyO3

## Installation

### From PyPI (coming soon)

```bash
pip install airllm-rs
```

### From Source

```bash
# Install maturin
pip install maturin

# Clone and build
git clone https://github.com/rileyseaburg/airllm-rs
cd airllm-rs
maturin develop --features python
```

### Rust Library

```bash
cargo add airllm-rs
```

## Usage

### Python

```python
from airllm_rs import AirLLM, GenerationConfig

# Load model (layer-by-layer, memory efficient)
model = AirLLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Check model info
print(model.config())
# ModelConfig(arch=Llama, layers=32, hidden=4096, vocab=32000)

# Generate with custom config
config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9
)

# Input token IDs (use your tokenizer)
input_ids = [1, 15043, 29892, 920, 526, 366]  # "Hello, how are you"
output_ids = model.generate(input_ids, config)
print(output_ids)
```

### Rust

```rust
use airllm_rs::{InferenceEngine, GenerationConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load model
    let engine = InferenceEngine::from_pretrained("./model")?;
    
    println!("Loaded: {:?}", engine.config().architecture);
    
    // Generate
    let config = GenerationConfig::default();
    let input_ids = vec![1, 15043, 29892];
    let output = engine.generate(&input_ids, &config)?;
    
    println!("Generated: {:?}", output);
    Ok(())
}
```

### CLI

```bash
# Show model info
airllm-cli info --model ./model

# List tensors
airllm-cli tensors --model ./model --filter mlp

# Run inference (WIP: needs tokenizer)
airllm-cli run --model ./model --prompt "def fibonacci(n):"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AirLLM-RS                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ LayerCache  │  │ TensorStore │  │ InferenceEngine     │  │
│  │ (mmap pool) │  │ (safetensors│  │ (layer-by-layer)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Attention   │  │ MLP/FFN     │  │ Embeddings          │  │
│  │ (GQA/MHA)   │  │ (SwiGLU)    │  │ (+ RoPE)            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Supported Models

| Architecture | Status | Notes |
|--------------|--------|-------|
| Llama/Llama-2/Llama-3 | ✅ | Full support |
| Mistral | ✅ | Uses Llama naming |
| Qwen/Qwen2 | ✅ | Full support |
| GLM-4 | ✅ | Including GLM-4.7 MoE |
| Mixtral | ⚠️ | MoE routing WIP |
| BitNet | ✅ | Ternary weight support |

## Performance

Memory usage comparison for Llama-2-7B:

| Method | VRAM Required |
|--------|---------------|
| Standard PyTorch | 14GB |
| 8-bit quantization | 7GB |
| AirLLM-RS (layer-wise) | **< 1GB** |

*Note: Layer-wise loading trades memory for speed. Best for memory-constrained environments.*

## License

MIT License - see [LICENSE](LICENSE)

## Related Projects

- [AirLLM](https://github.com/lyogavin/airllm) - Original Python implementation
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - C++ inference
- [candle](https://github.com/huggingface/candle) - Rust ML framework
