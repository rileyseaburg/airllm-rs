//! AirLLM-RS: High-performance layer-wise LLM inference in Rust
//!
//! This library provides memory-efficient inference for large language models
//! by loading transformer layers one at a time, enabling inference on models
//! that wouldn't fit in GPU memory.
//!
//! ## Key Features
//!
//! - **Layer-wise loading**: Load one transformer layer at a time
//! - **Memory-mapped I/O**: Zero-copy loading from safetensors files
//! - **Prefetching**: Async prefetch next layer while computing current
//! - **Quantization**: Support for fp16, bf16, int8, int4 (ternary for BitNet)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      AirLLM-RS                              │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
//! │  │ LayerCache  │  │ TensorStore │  │ InferenceEngine     │  │
//! │  │ (mmap pool) │  │ (safetensors│  │ (layer-by-layer)    │  │
//! │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
//! │  │ Attention   │  │ MLP/FFN     │  │ Embeddings          │  │
//! │  │ (GQA/MHA)   │  │ (SwiGLU)    │  │ (+ RoPE)            │  │
//! │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//! ```

pub mod error;
pub mod tensor;
pub mod layer;
pub mod model;
pub mod inference;

#[cfg(feature = "python")]
pub mod python;

// Re-exports
pub use error::{Error, Result};
pub use tensor::{Tensor, DType};
pub use layer::{LayerLoader, LayerCache};
pub use model::{ModelConfig, ModelArchitecture};
pub use inference::{InferenceEngine, GenerationConfig};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Re-export Python module when building as extension
#[cfg(feature = "python")]
pub use python::*;
