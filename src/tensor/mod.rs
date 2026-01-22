//! Tensor types and operations for AirLLM-RS
//!
//! Provides a minimal tensor abstraction optimized for layer-wise inference.

mod dtype;
pub mod ops;
pub mod view;

pub use dtype::DType;
pub use view::TensorView;

use crate::{Error, Result};
use std::sync::Arc;

/// A tensor with owned data
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Raw data buffer
    data: Arc<Vec<u8>>,
    /// Data type
    dtype: DType,
    /// Shape (dimensions)
    shape: Vec<usize>,
    /// Strides for indexing
    strides: Vec<usize>,
    /// Offset into data buffer
    offset: usize,
}

impl Tensor {
    /// Create a new tensor from raw bytes
    pub fn from_bytes(data: Vec<u8>, dtype: DType, shape: Vec<usize>) -> Result<Self> {
        let expected_bytes = shape.iter().product::<usize>() * dtype.size_of();
        if data.len() != expected_bytes {
            return Err(Error::ShapeMismatch {
                expected: vec![expected_bytes],
                got: vec![data.len()],
            });
        }

        let strides = Self::compute_strides(&shape, dtype.size_of());

        Ok(Self {
            data: Arc::new(data),
            dtype,
            shape,
            strides,
            offset: 0,
        })
    }

    /// Create a tensor filled with zeros
    pub fn zeros(dtype: DType, shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        let data = vec![0u8; numel * dtype.size_of()];
        let strides = Self::compute_strides(&shape, dtype.size_of());

        Self {
            data: Arc::new(data),
            dtype,
            shape,
            strides,
            offset: 0,
        }
    }

    /// Create from f32 slice (converts to specified dtype)
    pub fn from_f32(data: &[f32], dtype: DType, shape: Vec<usize>) -> Result<Self> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(Error::ShapeMismatch {
                expected: vec![numel],
                got: vec![data.len()],
            });
        }

        let bytes = match dtype {
            DType::F32 => bytemuck::cast_slice::<f32, u8>(data).to_vec(),
            DType::F16 => data
                .iter()
                .flat_map(|&f| half::f16::from_f32(f).to_le_bytes())
                .collect(),
            DType::BF16 => data
                .iter()
                .flat_map(|&f| half::bf16::from_f32(f).to_le_bytes())
                .collect(),
            _ => return Err(Error::UnsupportedDType(format!("{:?}", dtype))),
        };

        Self::from_bytes(bytes, dtype, shape)
    }

    /// Convert to f32 vec
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self.dtype {
            DType::F32 => bytemuck::cast_slice::<u8, f32>(self.as_bytes()).to_vec(),
            DType::F16 => self
                .as_bytes()
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            DType::BF16 => self
                .as_bytes()
                .chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            DType::I8 => self.as_bytes().iter().map(|&b| (b as i8) as f32).collect(),
            DType::I2 => {
                // Ternary: packed 4 values per byte -> {-1, 0, +1}
                let mut result = Vec::with_capacity(self.numel());
                for &byte in self.as_bytes() {
                    // Unpack 4 values: bits 6-7, 4-5, 2-3, 0-1
                    for shift in [6, 4, 2, 0] {
                        let val = ((byte >> shift) & 0x03) as i8 - 1; // 0->-1, 1->0, 2->+1
                        result.push(val as f32);
                    }
                }
                result.truncate(self.numel());
                result
            }
            _ => panic!("Unsupported dtype: {:?}", self.dtype),
        }
    }

    /// Get raw bytes
    pub fn as_bytes(&self) -> &[u8] {
        let len = self.numel() * self.dtype.size_of();
        &self.data[self.offset..self.offset + len]
    }

    /// Get shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get dtype
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Compute strides from shape
    fn compute_strides(shape: &[usize], elem_size: usize) -> Vec<usize> {
        let mut strides = vec![elem_size; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Reshape tensor (must have same numel)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        let old_numel: usize = self.shape.iter().product();
        let new_numel: usize = new_shape.iter().product();

        if old_numel != new_numel {
            return Err(Error::ShapeMismatch {
                expected: vec![old_numel],
                got: vec![new_numel],
            });
        }

        Ok(Self {
            data: Arc::clone(&self.data),
            dtype: self.dtype,
            shape: new_shape.clone(),
            strides: Self::compute_strides(&new_shape, self.dtype.size_of()),
            offset: self.offset,
        })
    }

    /// Transpose last two dimensions
    pub fn transpose(&self) -> Result<Self> {
        if self.ndim() < 2 {
            return Err(Error::ShapeMismatch {
                expected: vec![2],
                got: vec![self.ndim()],
            });
        }

        let mut new_shape = self.shape.clone();
        let n = new_shape.len();
        new_shape.swap(n - 1, n - 2);

        // For transpose, we need to actually rearrange data
        // This is a simplified version - real impl would be more efficient
        let old_data = self.to_f32_vec();
        let rows = self.shape[n - 2];
        let cols = self.shape[n - 1];

        let mut new_data = vec![0.0f32; old_data.len()];
        let batch_size: usize = self.shape[..n - 2].iter().product();
        let matrix_size = rows * cols;

        for b in 0..batch_size {
            for i in 0..rows {
                for j in 0..cols {
                    let old_idx = b * matrix_size + i * cols + j;
                    let new_idx = b * matrix_size + j * rows + i;
                    new_data[new_idx] = old_data[old_idx];
                }
            }
        }

        Self::from_f32(&new_data, self.dtype, new_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_from_f32() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_f32(&data, DType::F32, vec![2, 2]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.to_f32_vec(), data);
    }

    #[test]
    fn test_tensor_reshape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_f32(&data, DType::F32, vec![2, 3]).unwrap();
        let reshaped = tensor.reshape(vec![3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.to_f32_vec(), data);
    }
}
