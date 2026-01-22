//! Python bindings for AirLLM-RS using PyO3
//!
//! Provides a Python interface to the Rust layer-wise inference engine.
//!
//! Usage from Python:
//! ```python
//! from airllm_rs import AirLLM, GenerationConfig
//!
//! model = AirLLM.from_pretrained("meta-llama/Llama-2-7b-hf")
//! print(model.config())
//!
//! config = GenerationConfig(max_new_tokens=100, temperature=0.7)
//! output = model.generate([1, 15043, 29892], config)
//! ```

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use std::sync::Arc;
use tokio::runtime::Runtime;

use crate::inference::{InferenceEngine, GenerationConfig as RustGenConfig};
use crate::model::ModelConfig as RustModelConfig;

/// Python wrapper for GenerationConfig
#[pyclass(name = "GenerationConfig")]
#[derive(Clone)]
pub struct PyGenerationConfig {
    inner: RustGenConfig,
}

#[pymethods]
impl PyGenerationConfig {
    #[new]
    #[pyo3(signature = (
        max_new_tokens = 256,
        temperature = 0.7,
        top_p = 0.9,
        top_k = 50,
        repetition_penalty = 1.0,
        do_sample = true
    ))]
    fn new(
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
        do_sample: bool,
    ) -> Self {
        Self {
            inner: RustGenConfig {
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                do_sample,
                stop_sequences: vec![],
            },
        }
    }

    /// Create greedy decoding config (temperature=0)
    #[staticmethod]
    fn greedy() -> Self {
        Self {
            inner: RustGenConfig::greedy(),
        }
    }

    /// Create creative sampling config (temperature=1.0)
    #[staticmethod]
    fn creative() -> Self {
        Self {
            inner: RustGenConfig::creative(),
        }
    }

    /// Create code completion config
    #[staticmethod]
    fn code() -> Self {
        Self {
            inner: RustGenConfig::code(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GenerationConfig(max_new_tokens={}, temperature={}, top_p={}, top_k={}, do_sample={})",
            self.inner.max_new_tokens,
            self.inner.temperature,
            self.inner.top_p,
            self.inner.top_k,
            self.inner.do_sample
        )
    }
}

/// Python wrapper for ModelConfig
#[pyclass(name = "ModelConfig")]
pub struct PyModelConfig {
    inner: RustModelConfig,
}

#[pymethods]
impl PyModelConfig {
    /// Get architecture name
    #[getter]
    fn architecture(&self) -> String {
        format!("{:?}", self.inner.architecture)
    }

    /// Get hidden size
    #[getter]
    fn hidden_size(&self) -> usize {
        self.inner.hidden_size
    }

    /// Get number of layers
    #[getter]
    fn num_hidden_layers(&self) -> usize {
        self.inner.num_hidden_layers
    }

    /// Get number of attention heads
    #[getter]
    fn num_attention_heads(&self) -> usize {
        self.inner.num_attention_heads
    }

    /// Get number of KV heads
    #[getter]
    fn num_key_value_heads(&self) -> usize {
        self.inner.num_kv_heads()
    }

    /// Get intermediate size
    #[getter]
    fn intermediate_size(&self) -> usize {
        self.inner.intermediate_size
    }

    /// Get vocab size
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size
    }

    /// Get max position embeddings
    #[getter]
    fn max_position_embeddings(&self) -> usize {
        self.inner.max_position_embeddings
    }

    /// Get estimated model size in bytes
    fn estimate_size_bytes(&self) -> usize {
        self.inner.estimate_size_bytes()
    }

    /// Get human-readable size string
    fn size_string(&self) -> String {
        self.inner.size_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelConfig(arch={:?}, layers={}, hidden={}, vocab={})",
            self.inner.architecture,
            self.inner.num_hidden_layers,
            self.inner.hidden_size,
            self.inner.vocab_size
        )
    }
}

/// Main AirLLM model class for Python
#[pyclass(name = "AirLLM")]
pub struct PyAirLLM {
    engine: Arc<InferenceEngine>,
    runtime: Arc<Runtime>,
}

#[pymethods]
impl PyAirLLM {
    /// Load a model from a directory or HuggingFace repo
    #[staticmethod]
    #[pyo3(signature = (model_path, prefetch = true))]
    fn from_pretrained(model_path: &str, prefetch: bool) -> PyResult<Self> {
        // Create tokio runtime for async operations
        let runtime = Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        let engine = runtime.block_on(async {
            InferenceEngine::from_pretrained(model_path)
        }).map_err(|e| PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;

        Ok(Self {
            engine: Arc::new(engine),
            runtime: Arc::new(runtime),
        })
    }

    /// Get model configuration
    fn config(&self) -> PyModelConfig {
        PyModelConfig {
            inner: self.engine.config().clone(),
        }
    }

    /// Run forward pass and get logits
    ///
    /// Args:
    ///     input_ids: List of token IDs
    ///
    /// Returns:
    ///     List of logits for each position (flattened)
    fn forward(&self, input_ids: Vec<u32>) -> PyResult<Vec<f32>> {
        let logits = self.engine.forward(&input_ids)
            .map_err(|e| PyRuntimeError::new_err(format!("Forward pass failed: {}", e)))?;
        
        Ok(logits.to_f32_vec())
    }

    /// Generate tokens from input
    ///
    /// Args:
    ///     input_ids: List of input token IDs
    ///     config: GenerationConfig (optional, uses default if not provided)
    ///
    /// Returns:
    ///     List of all token IDs (input + generated)
    #[pyo3(signature = (input_ids, config = None))]
    fn generate(&self, input_ids: Vec<u32>, config: Option<PyGenerationConfig>) -> PyResult<Vec<u32>> {
        let gen_config = config.map(|c| c.inner).unwrap_or_default();
        
        let output = self.engine.generate(&input_ids, &gen_config)
            .map_err(|e| PyRuntimeError::new_err(format!("Generation failed: {}", e)))?;
        
        Ok(output)
    }

    /// Get model info as a dictionary
    fn info(&self) -> PyResult<std::collections::HashMap<String, PyObject>> {
        Python::with_gil(|py| {
            let mut info = std::collections::HashMap::new();
            let config = self.engine.config();
            
            info.insert("architecture".to_string(), format!("{:?}", config.architecture).into_pyobject(py)?.into_any().unbind());
            info.insert("hidden_size".to_string(), config.hidden_size.into_pyobject(py)?.into_any().unbind());
            info.insert("num_layers".to_string(), config.num_hidden_layers.into_pyobject(py)?.into_any().unbind());
            info.insert("num_attention_heads".to_string(), config.num_attention_heads.into_pyobject(py)?.into_any().unbind());
            info.insert("num_kv_heads".to_string(), config.num_kv_heads().into_pyobject(py)?.into_any().unbind());
            info.insert("vocab_size".to_string(), config.vocab_size.into_pyobject(py)?.into_any().unbind());
            info.insert("size".to_string(), config.size_string().into_pyobject(py)?.into_any().unbind());
            
            Ok(info)
        })
    }

    fn __repr__(&self) -> String {
        let config = self.engine.config();
        format!(
            "AirLLM(arch={:?}, layers={}, size={})",
            config.architecture,
            config.num_hidden_layers,
            config.size_string()
        )
    }
}

/// Load model configuration without loading weights
#[pyfunction]
fn load_config(model_path: &str) -> PyResult<PyModelConfig> {
    let config = RustModelConfig::from_dir(model_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to load config: {}", e)))?;
    
    Ok(PyModelConfig { inner: config })
}

/// Get library version
#[pyfunction]
fn version() -> &'static str {
    crate::VERSION
}

/// Python module definition
#[pymodule]
fn airllm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAirLLM>()?;
    m.add_class::<PyGenerationConfig>()?;
    m.add_class::<PyModelConfig>()?;
    m.add_function(wrap_pyfunction!(load_config, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    
    // Add some convenient aliases
    m.add("AirLLM", m.getattr("AirLLM")?)?;
    m.add("GenerationConfig", m.getattr("GenerationConfig")?)?;
    m.add("ModelConfig", m.getattr("ModelConfig")?)?;
    
    Ok(())
}
