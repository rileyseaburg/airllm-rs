//! Layer loading from safetensors files

use super::{LayerNaming, LayerWeights, SharedWeights};
use crate::tensor::{view::MappedSafetensors, Tensor};
use crate::{Error, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Loads transformer layers from safetensors files
pub struct LayerLoader {
    /// Model directory
    model_dir: PathBuf,
    /// Memory-mapped safetensors files
    safetensors: Vec<MappedSafetensors>,
    /// Tensor name to file index mapping
    tensor_index: HashMap<String, usize>,
    /// Layer naming convention
    naming: LayerNaming,
    /// Number of layers
    num_layers: usize,
}

impl LayerLoader {
    /// Create a new layer loader from a model directory
    pub fn new(
        model_dir: impl AsRef<Path>,
        naming: LayerNaming,
        num_layers: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();

        // Find all safetensors files
        let mut safetensors_files: Vec<PathBuf> = std::fs::read_dir(&model_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
            .collect();

        safetensors_files.sort();
        info!("Found {} safetensors files", safetensors_files.len());

        if safetensors_files.is_empty() {
            return Err(Error::Config("No safetensors files found".to_string()));
        }

        // Memory-map all files and build tensor index
        let mut safetensors = Vec::new();
        let mut tensor_index = HashMap::new();

        for (file_idx, path) in safetensors_files.iter().enumerate() {
            debug!("Loading {:?}", path);
            let st = MappedSafetensors::open(path)?;

            for name in st.tensor_names() {
                tensor_index.insert(name, file_idx);
            }

            safetensors.push(st);
        }

        info!(
            "Indexed {} tensors across {} files",
            tensor_index.len(),
            safetensors.len()
        );

        Ok(Self {
            model_dir,
            safetensors,
            tensor_index,
            naming,
            num_layers,
        })
    }

    /// Get a tensor by name (zero-copy view)
    pub fn get_tensor(&self, name: &str) -> Result<Tensor> {
        let file_idx = self
            .tensor_index
            .get(name)
            .ok_or_else(|| Error::TensorNotFound(name.to_string()))?;

        let view = self.safetensors[*file_idx].get(name)?;
        Ok(view.to_tensor())
    }

    /// Check if tensor exists
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_index.contains_key(name)
    }

    /// Load shared weights (embeddings, final norm, lm_head)
    pub fn load_shared_weights(&self) -> Result<SharedWeights> {
        info!("Loading shared weights...");

        let embed_tokens = self.get_tensor(&self.naming.embed_tokens)?;
        let norm = self.get_tensor(&self.naming.norm)?;

        // lm_head may be tied to embed_tokens
        let lm_head = if self.has_tensor(&self.naming.lm_head) {
            Some(self.get_tensor(&self.naming.lm_head)?)
        } else {
            None
        };

        Ok(SharedWeights {
            embed_tokens,
            norm,
            lm_head,
        })
    }

    /// Load a single transformer layer
    pub fn load_layer(&self, layer_idx: usize) -> Result<LayerWeights> {
        debug!("Loading layer {}", layer_idx);

        let get = |weight_name: &str| -> Result<Tensor> {
            let full_name = self.naming.layer_tensor_name(layer_idx, weight_name);
            self.get_tensor(&full_name)
        };

        // Load attention weights
        let q_proj = get(&self.naming.q_proj)?;
        let k_proj = get(&self.naming.k_proj)?;
        let v_proj = get(&self.naming.v_proj)?;
        let o_proj = get(&self.naming.o_proj)?;

        // Load MLP weights
        let gate_proj = get(&self.naming.gate_proj)?;
        let up_proj = get(&self.naming.up_proj)?;
        let down_proj = get(&self.naming.down_proj)?;

        // Load norm weights
        let input_layernorm = get(&self.naming.input_layernorm)?;
        let post_attention_layernorm = get(&self.naming.post_attention_layernorm)?;

        // Optional QK norm
        let q_norm = get("self_attn.q_norm.weight").ok();
        let k_norm = get("self_attn.k_norm.weight").ok();

        Ok(LayerWeights {
            layer_idx,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            up_proj,
            down_proj,
            input_layernorm,
            post_attention_layernorm,
            q_norm,
            k_norm,
        })
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get model directory
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensor_index.keys()
    }
}
