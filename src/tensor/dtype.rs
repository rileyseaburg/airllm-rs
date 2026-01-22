//! Data types for tensors

use serde::{Deserialize, Serialize};

/// Supported data types for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point (IEEE 754)
    F16,
    /// 16-bit brain floating point
    BF16,
    /// 8-bit signed integer (for int8 quantization)
    I8,
    /// 8-bit unsigned integer
    U8,
    /// 32-bit signed integer (for indices)
    I32,
    /// 64-bit signed integer
    I64,
    /// 2-bit integer (ternary, packed 4 per byte) - for BitNet
    I2,
    /// Boolean
    Bool,
}

impl DType {
    /// Size in bytes of a single element
    pub fn size_of(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::I8 => 1,
            DType::U8 => 1,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::I2 => 1, // 4 values packed per byte, but we round up
            DType::Bool => 1,
        }
    }

    /// Parse from safetensors dtype string
    pub fn from_safetensors(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(DType::F32),
            "F16" => Some(DType::F16),
            "BF16" => Some(DType::BF16),
            "I8" => Some(DType::I8),
            "U8" => Some(DType::U8),
            "I32" => Some(DType::I32),
            "I64" => Some(DType::I64),
            "BOOL" => Some(DType::Bool),
            _ => None,
        }
    }

    /// Convert to safetensors dtype string
    pub fn to_safetensors(&self) -> &'static str {
        match self {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::I8 => "I8",
            DType::U8 => "U8",
            DType::I32 => "I32",
            DType::I64 => "I64",
            DType::I2 => "U8", // Packed as U8
            DType::Bool => "BOOL",
        }
    }

    /// Is this a floating point type?
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::BF16)
    }

    /// Is this an integer type?
    pub fn is_int(&self) -> bool {
        matches!(
            self,
            DType::I8 | DType::U8 | DType::I32 | DType::I64 | DType::I2
        )
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
