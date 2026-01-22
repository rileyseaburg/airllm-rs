//! Text generation configuration and sampling

use serde::{Deserialize, Serialize};

/// Configuration for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    pub max_new_tokens: usize,

    /// Temperature for sampling (1.0 = normal, lower = more deterministic)
    pub temperature: f32,

    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,

    /// Top-k sampling (0 = disabled)
    pub top_k: usize,

    /// Repetition penalty (1.0 = disabled)
    pub repetition_penalty: f32,

    /// Stop sequences
    pub stop_sequences: Vec<String>,

    /// Whether to use greedy decoding (ignores temperature/top_p/top_k)
    pub do_sample: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.0,
            stop_sequences: vec![],
            do_sample: true,
        }
    }
}

impl GenerationConfig {
    /// Greedy decoding (temperature = 0)
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            do_sample: false,
            ..Default::default()
        }
    }

    /// Creative sampling (higher temperature)
    pub fn creative() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 0,
            ..Default::default()
        }
    }

    /// Code completion settings
    pub fn code() -> Self {
        Self {
            temperature: 0.1,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            stop_sequences: vec!["\n\n".to_string(), "```".to_string()],
            ..Default::default()
        }
    }
}

/// Sample from logits
pub fn sample_logits(logits: &[f32], config: &GenerationConfig) -> usize {
    if !config.do_sample || config.temperature < 1e-6 {
        // Greedy: return argmax
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    // Apply temperature
    let scaled: Vec<f32> = logits.iter().map(|&x| x / config.temperature).collect();

    // Apply top-k if set
    let (indices, probs) = if config.top_k > 0 && config.top_k < scaled.len() {
        let mut indexed: Vec<(usize, f32)> = scaled.iter().cloned().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        indexed.truncate(config.top_k);

        let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
        let probs: Vec<f32> = indexed.iter().map(|(_, p)| *p).collect();
        (indices, probs)
    } else {
        let indices: Vec<usize> = (0..scaled.len()).collect();
        (indices, scaled)
    };

    // Softmax
    let max_val = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_probs: Vec<f32> = probs.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_probs.iter().sum();
    let probs: Vec<f32> = exp_probs.iter().map(|&x| x / sum).collect();

    // Apply top-p
    let (indices, probs) = if config.top_p < 1.0 {
        let mut sorted: Vec<(usize, f32)> =
            indices.iter().cloned().zip(probs.iter().cloned()).collect();
        sorted.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        let mut cumsum = 0.0;
        let mut cutoff = sorted.len();
        for (i, (_, p)) in sorted.iter().enumerate() {
            cumsum += p;
            if cumsum > config.top_p {
                cutoff = i + 1;
                break;
            }
        }

        sorted.truncate(cutoff);
        let indices: Vec<usize> = sorted.iter().map(|(i, _)| *i).collect();
        let probs: Vec<f32> = sorted.iter().map(|(_, p)| *p).collect();
        (indices, probs)
    } else {
        (indices, probs)
    };

    // Renormalize
    let sum: f32 = probs.iter().sum();
    let probs: Vec<f32> = probs.iter().map(|&x| x / sum).collect();

    // Sample
    let r: f32 = rand_f32();
    let mut cumsum = 0.0;
    for (&idx, &p) in indices.iter().zip(probs.iter()) {
        cumsum += p;
        if r < cumsum {
            return idx;
        }
    }

    // Fallback to last index
    indices.last().copied().unwrap_or(0)
}

/// Simple random number generator (xorshift)
fn rand_f32() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0x853c49e6748fea9b);

    let mut s = STATE.load(Ordering::Relaxed);
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    STATE.store(s, Ordering::Relaxed);

    (s.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u64 << 24) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let logits = vec![1.0, 3.0, 2.0, 0.5];
        let config = GenerationConfig::greedy();
        assert_eq!(sample_logits(&logits, &config), 1); // Index of 3.0
    }
}
