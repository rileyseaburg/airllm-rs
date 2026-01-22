//! Zero-copy tensor views from memory-mapped files

use super::DType;
use memmap2::Mmap;
use std::sync::Arc;

/// A view into memory-mapped tensor data (zero-copy)
pub struct TensorView<'a> {
    /// Reference to memory-mapped data
    data: &'a [u8],
    /// Data type
    dtype: DType,
    /// Shape
    shape: Vec<usize>,
}

impl<'a> TensorView<'a> {
    /// Create a view from raw bytes
    pub fn new(data: &'a [u8], dtype: DType, shape: Vec<usize>) -> Self {
        Self { data, dtype, shape }
    }

    /// Get raw bytes
    pub fn as_bytes(&self) -> &[u8] {
        self.data
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

    /// Convert to owned Tensor (copies data)
    pub fn to_tensor(&self) -> super::Tensor {
        super::Tensor::from_bytes(self.data.to_vec(), self.dtype, self.shape.clone())
            .expect("TensorView to Tensor conversion failed")
    }

    /// Convert to f32 vec (may involve type conversion)
    pub fn to_f32_vec(&self) -> Vec<f32> {
        self.to_tensor().to_f32_vec()
    }
}

/// A memory-mapped safetensors file
pub struct MappedSafetensors {
    /// Memory map handle (must stay alive)
    _mmap: Arc<Mmap>,
    /// Parsed header with tensor metadata
    header: safetensors::SafeTensors<'static>,
}

impl MappedSafetensors {
    /// Open a safetensors file with memory mapping
    pub fn open(path: &std::path::Path) -> crate::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap = Arc::new(mmap);

        // Parse the safetensors header
        // SAFETY: We keep mmap alive for lifetime of header
        let header = unsafe {
            let slice: &'static [u8] = std::mem::transmute(mmap.as_ref() as &[u8]);
            safetensors::SafeTensors::deserialize(slice)?
        };

        Ok(Self {
            _mmap: mmap,
            header,
        })
    }

    /// Get tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.header
            .names()
            .into_iter()
            .map(|s: &String| s.clone())
            .collect()
    }

    /// Get a tensor view by name (zero-copy)
    pub fn get(&self, name: &str) -> crate::Result<TensorView<'_>> {
        let info = self
            .header
            .tensor(name)
            .map_err(|_| crate::Error::TensorNotFound(name.to_string()))?;

        let dtype = DType::from_safetensors(&format!("{:?}", info.dtype()))
            .ok_or_else(|| crate::Error::UnsupportedDType(format!("{:?}", info.dtype())))?;

        let shape = info.shape().to_vec();
        let data = info.data();

        Ok(TensorView::new(data, dtype, shape))
    }

    /// Check if tensor exists
    pub fn contains(&self, name: &str) -> bool {
        self.header.tensor(name).is_ok()
    }
}
