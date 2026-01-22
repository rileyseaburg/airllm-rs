//! Core inference engine for layer-wise execution

use crate::layer::{LayerCache, LayerLoader, LayerWeights};
use crate::model::ModelConfig;
use crate::tensor::{ops, DType, Tensor};
use crate::Result;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use tracing::{debug, info};

use super::generation::{sample_logits, GenerationConfig};

/// Layer-wise inference engine
pub struct InferenceEngine {
    /// Model configuration
    config: ModelConfig,
    /// Layer cache with prefetching
    cache: LayerCache,
}

impl InferenceEngine {
    /// Create inference engine from a model directory
    pub fn from_pretrained(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        info!("Loading model from {:?}", model_dir);

        // Load config
        let config = ModelConfig::from_dir(model_dir)?;
        info!(
            "Model: {:?}, {} layers, hidden_size={}, estimated size: {}",
            config.architecture,
            config.num_hidden_layers,
            config.hidden_size,
            config.size_string()
        );

        // Create layer loader
        let naming = config.architecture.layer_naming();
        let loader = LayerLoader::new(model_dir, naming, config.num_hidden_layers)?;

        // Create cache with prefetching
        let cache = LayerCache::new(loader, true)?;

        Ok(Self { config, cache })
    }

    /// Get model config
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Run embedding lookup
    fn embed(&self, input_ids: &[u32]) -> Result<Tensor> {
        let shared = self.cache.shared_weights();
        let embed = &shared.embed_tokens;
        let hidden_size = self.config.hidden_size;

        // Gather embeddings
        let embed_data = embed.to_f32_vec();
        let mut output = vec![0.0f32; input_ids.len() * hidden_size];

        for (i, &token_id) in input_ids.iter().enumerate() {
            let start = token_id as usize * hidden_size;
            let end = start + hidden_size;
            output[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(&embed_data[start..end]);
        }

        Tensor::from_f32(&output, DType::F32, vec![input_ids.len(), hidden_size])
    }

    /// Apply RMS normalization
    fn rms_norm(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        ops::rms_norm(x, weight, self.config.rms_norm_eps)
    }

    /// Compute attention (simplified, no KV cache yet)
    fn attention(&self, hidden: &Tensor, layer: &LayerWeights) -> Result<Tensor> {
        let seq_len = hidden.shape()[0];
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads();
        let head_dim = self.config.head_dim();

        // Project Q, K, V
        let q = ops::matmul(hidden, &layer.q_proj.transpose()?)?;
        let k = ops::matmul(hidden, &layer.k_proj.transpose()?)?;
        let v = ops::matmul(hidden, &layer.v_proj.transpose()?)?;

        // Reshape for multi-head attention
        // [seq_len, hidden] -> [seq_len, num_heads, head_dim]
        let q = q.reshape(vec![seq_len, num_heads, head_dim])?;
        let k = k.reshape(vec![seq_len, num_kv_heads, head_dim])?;
        let v = v.reshape(vec![seq_len, num_kv_heads, head_dim])?;

        // Simplified attention: Q @ K^T / sqrt(d) -> softmax -> @ V
        // This is O(n^2) and doesn't use flash attention
        // TODO: Implement proper scaled dot-product attention with RoPE

        let q_data = q.to_f32_vec();
        let k_data = k.to_f32_vec();
        let v_data = v.to_f32_vec();

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; seq_len * hidden_size];

        // For each head
        for h in 0..num_heads {
            let kv_h = h % num_kv_heads; // GQA mapping

            // Compute attention scores
            let mut scores = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..=i {
                    // Causal mask
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        let q_idx = i * num_heads * head_dim + h * head_dim + d;
                        let k_idx = j * num_kv_heads * head_dim + kv_h * head_dim + d;
                        score += q_data[q_idx] * k_data[k_idx];
                    }
                    scores[i * seq_len + j] = score * scale;
                }
                // Mask future positions
                for j in (i + 1)..seq_len {
                    scores[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }

            // Softmax per row
            for i in 0..seq_len {
                let row = &mut scores[i * seq_len..(i + 1) * seq_len];
                let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for x in row.iter_mut() {
                    *x = (*x - max_val).exp();
                    sum += *x;
                }
                for x in row.iter_mut() {
                    *x /= sum;
                }
            }

            // Apply attention to values
            for i in 0..seq_len {
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for j in 0..seq_len {
                        let v_idx = j * num_kv_heads * head_dim + kv_h * head_dim + d;
                        val += scores[i * seq_len + j] * v_data[v_idx];
                    }
                    output[i * hidden_size + h * head_dim + d] = val;
                }
            }
        }

        let attn_out = Tensor::from_f32(&output, DType::F32, vec![seq_len, hidden_size])?;

        // Output projection
        ops::matmul(&attn_out, &layer.o_proj.transpose()?)
    }

    /// Apply MLP (SwiGLU)
    fn mlp(&self, hidden: &Tensor, layer: &LayerWeights) -> Result<Tensor> {
        // gate = silu(x @ gate_proj)
        let gate = ops::matmul(hidden, &layer.gate_proj.transpose()?)?;
        let gate = ops::silu(&gate)?;

        // up = x @ up_proj
        let up = ops::matmul(hidden, &layer.up_proj.transpose()?)?;

        // hidden = gate * up
        let hidden = ops::mul(&gate, &up)?;

        // output = hidden @ down_proj
        ops::matmul(&hidden, &layer.down_proj.transpose()?)
    }

    /// Forward pass through all layers
    pub fn forward(&self, input_ids: &[u32]) -> Result<Tensor> {
        let num_layers = self.cache.num_layers();

        // Embedding
        let mut hidden = self.embed(input_ids)?;
        debug!("Embedded: {:?}", hidden.shape());

        // Progress bar for layers
        let pb = ProgressBar::new(num_layers as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} layers")
                .unwrap(),
        );

        // Process each layer
        for layer_idx in 0..num_layers {
            let layer = self.cache.get_layer(layer_idx)?;

            // Pre-attention norm
            let normed = self.rms_norm(&hidden, &layer.input_layernorm)?;

            // Self-attention
            let attn_out = self.attention(&normed, &layer)?;

            // Residual connection
            hidden = ops::add(&hidden, &attn_out)?;

            // Post-attention norm
            let normed = self.rms_norm(&hidden, &layer.post_attention_layernorm)?;

            // MLP
            let mlp_out = self.mlp(&normed, &layer)?;

            // Residual connection
            hidden = ops::add(&hidden, &mlp_out)?;

            pb.inc(1);
        }

        pb.finish_with_message("Done");

        // Final norm
        let shared = self.cache.shared_weights();
        hidden = self.rms_norm(&hidden, &shared.norm)?;

        // LM head
        let lm_head = shared.lm_head.as_ref().unwrap_or(&shared.embed_tokens);
        let logits = ops::matmul(&hidden, &lm_head.transpose()?)?;

        Ok(logits)
    }

    /// Generate tokens
    pub fn generate(&self, input_ids: &[u32], config: &GenerationConfig) -> Result<Vec<u32>> {
        let mut tokens = input_ids.to_vec();

        info!("Generating {} tokens...", config.max_new_tokens);

        for i in 0..config.max_new_tokens {
            // Forward pass
            let logits = self.forward(&tokens)?;

            // Get logits for last position
            let logits_data = logits.to_f32_vec();
            let vocab_size = self.config.vocab_size;
            let last_logits = &logits_data[logits_data.len() - vocab_size..];

            // Sample next token
            let next_token = sample_logits(last_logits, config) as u32;

            // Check for EOS
            // TODO: Get EOS token from tokenizer
            if next_token == 2 {
                break;
            }

            tokens.push(next_token);

            if (i + 1) % 10 == 0 {
                debug!("Generated {} tokens", i + 1);
            }
        }

        Ok(tokens)
    }
}
