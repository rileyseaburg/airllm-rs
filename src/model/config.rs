//! Model configuration parsing and architecture definitions

use crate::layer::LayerNaming;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Supported model architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArchitecture {
    /// Llama and Llama-like models (Mistral, etc.)
    Llama,
    /// Qwen/Qwen2 models
    Qwen,
    /// GLM-4 and GLM-4 MoE (GLM-4.7)
    Glm4,
    /// Mixtral MoE
    Mixtral,
    /// Distillix/BitNet models
    Distillix,
    /// Generic fallback
    Generic,
}

impl ModelArchitecture {
    /// Detect architecture from config.json
    pub fn from_config(config: &serde_json::Value) -> Self {
        // Check architectures field
        if let Some(archs) = config.get("architectures").and_then(|a| a.as_array()) {
            for arch in archs {
                if let Some(s) = arch.as_str() {
                    if s.contains("Llama") {
                        return ModelArchitecture::Llama;
                    }
                    if s.contains("Qwen") {
                        return ModelArchitecture::Qwen;
                    }
                    if s.contains("GLM") || s.contains("Glm") {
                        return ModelArchitecture::Glm4;
                    }
                    if s.contains("Mixtral") {
                        return ModelArchitecture::Mixtral;
                    }
                }
            }
        }

        // Check model_type field
        if let Some(model_type) = config.get("model_type").and_then(|t| t.as_str()) {
            match model_type {
                "llama" | "mistral" => return ModelArchitecture::Llama,
                "qwen" | "qwen2" => return ModelArchitecture::Qwen,
                "glm4" | "glm4_moe" | "chatglm" => return ModelArchitecture::Glm4,
                "mixtral" => return ModelArchitecture::Mixtral,
                "distillix" | "bitnet" => return ModelArchitecture::Distillix,
                _ => {}
            }
        }

        ModelArchitecture::Generic
    }

    /// Get layer naming convention for this architecture
    pub fn layer_naming(&self) -> LayerNaming {
        match self {
            ModelArchitecture::Llama => LayerNaming::default(),
            ModelArchitecture::Qwen => LayerNaming::qwen(),
            ModelArchitecture::Glm4 => LayerNaming::glm4(),
            ModelArchitecture::Mixtral => LayerNaming::default(), // Similar to Llama
            ModelArchitecture::Distillix => LayerNaming::distillix(),
            ModelArchitecture::Generic => LayerNaming::default(),
        }
    }
}

/// Model configuration parsed from config.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model architecture
    #[serde(default)]
    pub architecture: ModelArchitecture,

    /// Hidden size (embedding dimension)
    pub hidden_size: usize,

    /// Number of transformer layers
    pub num_hidden_layers: usize,

    /// Number of attention heads
    pub num_attention_heads: usize,

    /// Number of KV heads (for GQA)
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Intermediate size (MLP hidden dim)
    pub intermediate_size: usize,

    /// Vocabulary size
    pub vocab_size: usize,

    /// Maximum sequence length
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// RMS norm epsilon
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    /// RoPE theta
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    /// Whether to tie word embeddings
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// Attention bias
    #[serde(default)]
    pub attention_bias: bool,

    /// MLP bias
    #[serde(default)]
    pub mlp_bias: bool,
}

fn default_max_position_embeddings() -> usize {
    4096
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10000.0
}

impl Default for ModelArchitecture {
    fn default() -> Self {
        ModelArchitecture::Generic
    }
}

impl ModelConfig {
    /// Load config from a model directory
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let config_path = model_dir.as_ref().join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::Config(format!("Failed to read config.json: {}", e)))?;

        let raw: serde_json::Value = serde_json::from_str(&config_str)?;

        // Detect architecture first
        let architecture = ModelArchitecture::from_config(&raw);

        // Parse the config
        let mut config: ModelConfig = serde_json::from_value(raw.clone())
            .map_err(|e| Error::Config(format!("Failed to parse config.json: {}", e)))?;

        config.architecture = architecture;

        Ok(config)
    }

    /// Get number of KV heads (defaults to num_attention_heads if not set)
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get GQA ratio (query heads per KV head)
    pub fn gqa_ratio(&self) -> usize {
        self.num_attention_heads / self.num_kv_heads()
    }

    /// Estimate model size in bytes (fp16)
    pub fn estimate_size_bytes(&self) -> usize {
        let embed_params = self.vocab_size * self.hidden_size;
        let layer_params = 
            // Attention: Q, K, V, O projections
            4 * self.hidden_size * self.hidden_size +
            // MLP: gate, up, down projections
            3 * self.hidden_size * self.intermediate_size +
            // Layer norms
            2 * self.hidden_size;

        let total_params = embed_params + 
            self.num_hidden_layers * layer_params +
            self.hidden_size + // Final norm
            self.vocab_size * self.hidden_size; // LM head (if not tied)

        // 2 bytes per param (fp16)
        total_params * 2
    }

    /// Format size for display
    pub fn size_string(&self) -> String {
        let bytes = self.estimate_size_bytes();
        if bytes >= 1_000_000_000 {
            format!("{:.1}GB", bytes as f64 / 1_000_000_000.0)
        } else {
            format!("{:.1}MB", bytes as f64 / 1_000_000.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_detection() {
        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama"
        });
        assert_eq!(
            ModelArchitecture::from_config(&config),
            ModelArchitecture::Llama
        );

        let config = serde_json::json!({
            "architectures": ["GLM4ForCausalLM"],
            "model_type": "glm4_moe"
        });
        assert_eq!(
            ModelArchitecture::from_config(&config),
            ModelArchitecture::Glm4
        );
    }
}
