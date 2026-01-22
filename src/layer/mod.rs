//! Layer-wise loading for memory-efficient inference

mod loader;
mod cache;

pub use loader::LayerLoader;
pub use cache::LayerCache;

use crate::tensor::Tensor;

/// Weights for a single transformer layer
#[derive(Debug)]
pub struct LayerWeights {
    /// Layer index
    pub layer_idx: usize,
    
    /// Attention weights
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,
    
    /// MLP weights (SwiGLU)
    pub gate_proj: Tensor,
    pub up_proj: Tensor,
    pub down_proj: Tensor,
    
    /// Normalization weights
    pub input_layernorm: Tensor,
    pub post_attention_layernorm: Tensor,
    
    /// Optional: QK norm weights (Gemma 2 style)
    pub q_norm: Option<Tensor>,
    pub k_norm: Option<Tensor>,
}

/// Shared weights (embeddings, final norm, lm_head)
#[derive(Debug)]
pub struct SharedWeights {
    /// Token embeddings
    pub embed_tokens: Tensor,
    /// Final layer norm
    pub norm: Tensor,
    /// Language model head (may be tied to embed_tokens)
    pub lm_head: Option<Tensor>,
}

/// Weight naming conventions for different model architectures
#[derive(Debug, Clone)]
pub struct LayerNaming {
    /// Prefix for layers (e.g., "model.layers" for Llama)
    pub layer_prefix: String,
    /// Attention projection names
    pub q_proj: String,
    pub k_proj: String,
    pub v_proj: String,
    pub o_proj: String,
    /// MLP names
    pub gate_proj: String,
    pub up_proj: String,
    pub down_proj: String,
    /// Norm names
    pub input_layernorm: String,
    pub post_attention_layernorm: String,
    /// Shared weight names
    pub embed_tokens: String,
    pub norm: String,
    pub lm_head: String,
}

impl Default for LayerNaming {
    fn default() -> Self {
        // Llama-style naming
        Self {
            layer_prefix: "model.layers".to_string(),
            q_proj: "self_attn.q_proj.weight".to_string(),
            k_proj: "self_attn.k_proj.weight".to_string(),
            v_proj: "self_attn.v_proj.weight".to_string(),
            o_proj: "self_attn.o_proj.weight".to_string(),
            gate_proj: "mlp.gate_proj.weight".to_string(),
            up_proj: "mlp.up_proj.weight".to_string(),
            down_proj: "mlp.down_proj.weight".to_string(),
            input_layernorm: "input_layernorm.weight".to_string(),
            post_attention_layernorm: "post_attention_layernorm.weight".to_string(),
            embed_tokens: "model.embed_tokens.weight".to_string(),
            norm: "model.norm.weight".to_string(),
            lm_head: "lm_head.weight".to_string(),
        }
    }
}

impl LayerNaming {
    /// Create naming for GLM-4 models
    pub fn glm4() -> Self {
        Self {
            layer_prefix: "transformer.encoder.layers".to_string(),
            q_proj: "self_attention.query_key_value.weight".to_string(), // Combined QKV
            k_proj: "".to_string(), // Part of combined
            v_proj: "".to_string(), // Part of combined
            o_proj: "self_attention.dense.weight".to_string(),
            gate_proj: "mlp.dense_h_to_4h.weight".to_string(),
            up_proj: "".to_string(), // Part of combined in GLM
            down_proj: "mlp.dense_4h_to_h.weight".to_string(),
            input_layernorm: "input_layernorm.weight".to_string(),
            post_attention_layernorm: "post_attention_layernorm.weight".to_string(),
            embed_tokens: "transformer.embedding.word_embeddings.weight".to_string(),
            norm: "transformer.encoder.final_layernorm.weight".to_string(),
            lm_head: "transformer.output_layer.weight".to_string(),
        }
    }

    /// Create naming for Distillix/BitNet models (no model. prefix)
    pub fn distillix() -> Self {
        Self {
            layer_prefix: "layers".to_string(),
            q_proj: "self_attn.q_proj.weight".to_string(),
            k_proj: "self_attn.k_proj.weight".to_string(),
            v_proj: "self_attn.v_proj.weight".to_string(),
            o_proj: "self_attn.o_proj.weight".to_string(),
            gate_proj: "mlp.gate_proj.weight".to_string(),
            up_proj: "mlp.up_proj.weight".to_string(),
            down_proj: "mlp.down_proj.weight".to_string(),
            input_layernorm: "input_layernorm.weight".to_string(),
            post_attention_layernorm: "post_attention_layernorm.weight".to_string(),
            embed_tokens: "embed_tokens.weight".to_string(),
            norm: "norm.weight".to_string(),
            lm_head: "lm_head.weight".to_string(),
        }
    }

    /// Create naming for Qwen models
    pub fn qwen() -> Self {
        Self {
            layer_prefix: "model.layers".to_string(),
            q_proj: "self_attn.q_proj.weight".to_string(),
            k_proj: "self_attn.k_proj.weight".to_string(),
            v_proj: "self_attn.v_proj.weight".to_string(),
            o_proj: "self_attn.o_proj.weight".to_string(),
            gate_proj: "mlp.gate_proj.weight".to_string(),
            up_proj: "mlp.up_proj.weight".to_string(),
            down_proj: "mlp.down_proj.weight".to_string(),
            input_layernorm: "input_layernorm.weight".to_string(),
            post_attention_layernorm: "post_attention_layernorm.weight".to_string(),
            embed_tokens: "model.embed_tokens.weight".to_string(),
            norm: "model.norm.weight".to_string(),
            lm_head: "lm_head.weight".to_string(),
        }
    }

    /// Get full tensor name for a layer weight
    pub fn layer_tensor_name(&self, layer_idx: usize, weight_name: &str) -> String {
        format!("{}.{}.{}", self.layer_prefix, layer_idx, weight_name)
    }
}
