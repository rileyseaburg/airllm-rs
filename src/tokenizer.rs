//! Tokenizer support using HuggingFace tokenizers

use crate::{Error, Result};
use std::path::Path;
use tokenizers::Tokenizer;

/// Wrapper around HuggingFace tokenizer
pub struct ModelTokenizer {
    tokenizer: Tokenizer,
}

impl ModelTokenizer {
    /// Load tokenizer from a model directory
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Try tokenizer.json first (fast tokenizer)
        let tokenizer_path = model_dir.join("tokenizer.json");

        if tokenizer_path.exists() {
            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| Error::Tokenizer(format!("Failed to load tokenizer.json: {}", e)))?;
            return Ok(Self { tokenizer });
        }

        Err(Error::Tokenizer(
            "No tokenizer.json found in model directory".to_string(),
        ))
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(format!("Encoding failed: {}", e)))?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Encode with special tokens (BOS, etc.)
    pub fn encode_with_special(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::Tokenizer(format!("Encoding failed: {}", e)))?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| Error::Tokenizer(format!("Decoding failed: {}", e)))
    }

    /// Decode without skipping special tokens
    pub fn decode_raw(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|e| Error::Tokenizer(format!("Decoding failed: {}", e)))
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Get BOS token ID if available
    pub fn bos_token_id(&self) -> Option<u32> {
        self.tokenizer
            .token_to_id("<s>")
            .or_else(|| self.tokenizer.token_to_id("<|begin_of_text|>"))
            .or_else(|| self.tokenizer.token_to_id("<|startoftext|>"))
    }

    /// Get EOS token ID if available
    pub fn eos_token_id(&self) -> Option<u32> {
        self.tokenizer
            .token_to_id("</s>")
            .or_else(|| self.tokenizer.token_to_id("<|end_of_text|>"))
            .or_else(|| self.tokenizer.token_to_id("<|endoftext|>"))
    }

    /// Get PAD token ID if available
    pub fn pad_token_id(&self) -> Option<u32> {
        self.tokenizer
            .token_to_id("<pad>")
            .or_else(|| self.tokenizer.token_to_id("<|pad|>"))
    }
}
