//! Inference engine for layer-wise model execution

mod engine;
mod generation;

pub use engine::InferenceEngine;
pub use generation::GenerationConfig;
