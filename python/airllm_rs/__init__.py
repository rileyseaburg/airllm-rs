"""
AirLLM-RS: High-performance layer-wise LLM inference in Rust

This package provides Python bindings to the Rust-based AirLLM inference engine,
enabling memory-efficient inference for large language models that wouldn't 
normally fit in GPU memory.

Example usage:
    >>> from airllm_rs import AirLLM, GenerationConfig
    >>> 
    >>> # Load model
    >>> model = AirLLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> print(model.config())
    >>>
    >>> # Generate with custom config
    >>> config = GenerationConfig(max_new_tokens=100, temperature=0.7)
    >>> output = model.generate([1, 15043, 29892], config)
    >>> print(output)
"""

from .airllm_rs import (
    AirLLM,
    GenerationConfig,
    ModelConfig,
    load_config,
    version,
)

__all__ = [
    "AirLLM",
    "GenerationConfig", 
    "ModelConfig",
    "load_config",
    "version",
]

__version__ = version()
