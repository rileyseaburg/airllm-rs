//! Tensor operations (matmul, softmax, etc.)

use super::{DType, Tensor};
use crate::Result;

/// Matrix multiplication: C = A @ B
/// A: [M, K], B: [K, N] -> C: [M, N]
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    // Get matrix dimensions
    let (m, k1) = (a_shape[a_shape.len() - 2], a_shape[a_shape.len() - 1]);
    let (k2, n) = (b_shape[b_shape.len() - 2], b_shape[b_shape.len() - 1]);

    if k1 != k2 {
        return Err(crate::Error::ShapeMismatch {
            expected: vec![k1],
            got: vec![k2],
        });
    }

    let k = k1;

    // Compute output shape (batch dims + [m, n])
    let mut out_shape = a_shape[..a_shape.len() - 2].to_vec();
    out_shape.push(m);
    out_shape.push(n);

    // Simple CPU matmul (no SIMD optimization yet)
    let a_data = a.to_f32_vec();
    let b_data = b.to_f32_vec();

    let batch_size: usize = out_shape[..out_shape.len() - 2].iter().product();
    let batch_size = batch_size.max(1);

    let mut out_data = vec![0.0f32; batch_size * m * n];

    for batch in 0..batch_size {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    let a_idx = batch * m * k + i * k + kk;
                    let b_idx = batch * k * n + kk * n + j;
                    sum += a_data[a_idx] * b_data[b_idx];
                }
                out_data[batch * m * n + i * n + j] = sum;
            }
        }
    }

    Tensor::from_f32(&out_data, DType::F32, out_shape)
}

/// Batched matrix multiplication with broadcasting
pub fn batched_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // For now, delegate to simple matmul
    // TODO: Proper batch dimension handling
    matmul(a, b)
}

/// Softmax along last dimension
pub fn softmax(x: &Tensor) -> Result<Tensor> {
    let data = x.to_f32_vec();
    let shape = x.shape().to_vec();
    let last_dim = *shape.last().unwrap();
    let batch_size = data.len() / last_dim;

    let mut out = vec![0.0f32; data.len()];

    for b in 0..batch_size {
        let start = b * last_dim;
        let end = start + last_dim;
        let slice = &data[start..end];

        // Find max for numerical stability
        let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(x - max) and sum
        let mut sum = 0.0f32;
        for i in 0..last_dim {
            let exp_val = (slice[i] - max_val).exp();
            out[start + i] = exp_val;
            sum += exp_val;
        }

        // Normalize
        for i in 0..last_dim {
            out[start + i] /= sum;
        }
    }

    Tensor::from_f32(&out, x.dtype(), shape)
}

/// RMS normalization
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let data = x.to_f32_vec();
    let weight_data = weight.to_f32_vec();
    let shape = x.shape().to_vec();
    let hidden_size = *shape.last().unwrap();
    let batch_size = data.len() / hidden_size;

    let mut out = vec![0.0f32; data.len()];

    for b in 0..batch_size {
        let start = b * hidden_size;

        // Compute RMS
        let mut sum_sq = 0.0f32;
        for i in 0..hidden_size {
            sum_sq += data[start + i] * data[start + i];
        }
        let rms = (sum_sq / hidden_size as f32 + eps).sqrt();

        // Normalize and scale
        for i in 0..hidden_size {
            out[start + i] = (data[start + i] / rms) * weight_data[i];
        }
    }

    Tensor::from_f32(&out, x.dtype(), shape)
}

/// SiLU (Swish) activation: x * sigmoid(x)
pub fn silu(x: &Tensor) -> Result<Tensor> {
    let data = x.to_f32_vec();
    let out: Vec<f32> = data.iter().map(|&v| v * sigmoid(v)).collect();
    Tensor::from_f32(&out, x.dtype(), x.shape().to_vec())
}

/// Sigmoid activation
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Element-wise multiply
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_data = a.to_f32_vec();
    let b_data = b.to_f32_vec();

    if a_data.len() != b_data.len() {
        return Err(crate::Error::ShapeMismatch {
            expected: vec![a_data.len()],
            got: vec![b_data.len()],
        });
    }

    let out: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(&x, &y)| x * y)
        .collect();
    Tensor::from_f32(&out, a.dtype(), a.shape().to_vec())
}

/// Element-wise add
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_data = a.to_f32_vec();
    let b_data = b.to_f32_vec();

    if a_data.len() != b_data.len() {
        return Err(crate::Error::ShapeMismatch {
            expected: vec![a_data.len()],
            got: vec![b_data.len()],
        });
    }

    let out: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(&x, &y)| x + y)
        .collect();
    Tensor::from_f32(&out, a.dtype(), a.shape().to_vec())
}

/// Apply Rotary Position Embeddings (RoPE) to query and key tensors
///
/// q: [seq_len, num_heads, head_dim]
/// k: [seq_len, num_kv_heads, head_dim]
///
/// Returns (q_rotated, k_rotated) with the same shapes
pub fn apply_rope(
    q: &[f32],
    k: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    // Precompute frequency bands
    // freq[i] = 1.0 / (theta ^ (2i / head_dim))
    let half_dim = head_dim / 2;
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect();

    let mut q_rotated = q.to_vec();
    let mut k_rotated = k.to_vec();

    // Apply RoPE to each position
    for pos in 0..seq_len {
        // Compute position-dependent angles
        let angles: Vec<f32> = freqs.iter().map(|&f| pos as f32 * f).collect();
        let cos_vals: Vec<f32> = angles.iter().map(|&a| a.cos()).collect();
        let sin_vals: Vec<f32> = angles.iter().map(|&a| a.sin()).collect();

        // Apply to Q
        for h in 0..num_heads {
            let base = pos * num_heads * head_dim + h * head_dim;
            for i in 0..half_dim {
                let x0 = q[base + i];
                let x1 = q[base + i + half_dim];
                // Rotate: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
                q_rotated[base + i] = x0 * cos_vals[i] - x1 * sin_vals[i];
                q_rotated[base + i + half_dim] = x0 * sin_vals[i] + x1 * cos_vals[i];
            }
        }

        // Apply to K
        for h in 0..num_kv_heads {
            let base = pos * num_kv_heads * head_dim + h * head_dim;
            for i in 0..half_dim {
                let x0 = k[base + i];
                let x1 = k[base + i + half_dim];
                k_rotated[base + i] = x0 * cos_vals[i] - x1 * sin_vals[i];
                k_rotated[base + i + half_dim] = x0 * sin_vals[i] + x1 * cos_vals[i];
            }
        }
    }

    (q_rotated, k_rotated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::from_f32(&data, DType::F32, vec![3]).unwrap();
        let result = softmax(&tensor).unwrap();
        let out = result.to_f32_vec();

        // Check sum is 1
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check ordering preserved
        assert!(out[2] > out[1] && out[1] > out[0]);
    }

    #[test]
    fn test_matmul() {
        // 2x3 @ 3x2 = 2x2
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F32, vec![2, 3]).unwrap();
        let b = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F32, vec![3, 2]).unwrap();

        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);

        // [1,2,3] @ [1,3,5; 2,4,6]^T = [1+6+15, 2+8+18] = [22, 28]
        let out = c.to_f32_vec();
        assert_eq!(out[0], 22.0); // 1*1 + 2*3 + 3*5
        assert_eq!(out[1], 28.0); // 1*2 + 2*4 + 3*6
    }
}
