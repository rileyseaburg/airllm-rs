//! Error types for AirLLM-RS

use thiserror::Error;

/// Result type alias for AirLLM operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during AirLLM operations
#[derive(Error, Debug)]
pub enum Error {
    /// I/O error during file operations
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Error parsing safetensors file
    #[error("Safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    /// JSON parsing error (config files)
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Model configuration error
    #[error("Config error: {0}")]
    Config(String),

    /// Tensor shape mismatch
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Unsupported data type
    #[error("Unsupported dtype: {0}")]
    UnsupportedDType(String),

    /// Layer not found in model
    #[error("Layer not found: {0}")]
    LayerNotFound(String),

    /// Tensor not found
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    /// Invalid model architecture
    #[error("Unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    /// Memory allocation error
    #[error("Memory error: {0}")]
    Memory(String),

    /// Tokenizer error
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// Generation error
    #[error("Generation error: {0}")]
    Generation(String),

    /// HuggingFace Hub error
    #[error("HF Hub error: {0}")]
    HfHub(String),
}
