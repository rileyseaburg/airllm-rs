.PHONY: build release run run-fast info tensors clean install test

# Default model path
MODEL ?= /root/distillix/exports/hf_model
PROMPT ?= "def fibonacci(n):"
MAX_TOKENS ?= 10
TEMPERATURE ?= 0.7

# Build commands
build:
	cargo build

release:
	cargo build --release

# Run inference
run: release
	./target/release/airllm-cli run \
		--model $(MODEL) \
		--prompt $(PROMPT) \
		--max-tokens $(MAX_TOKENS) \
		--temperature $(TEMPERATURE)

# Quick run with fewer tokens
run-fast: release
	./target/release/airllm-cli run \
		--model $(MODEL) \
		--prompt $(PROMPT) \
		--max-tokens 5 \
		--temperature 0.1

# Greedy decoding (temperature=0)
run-greedy: release
	./target/release/airllm-cli run \
		--model $(MODEL) \
		--prompt $(PROMPT) \
		--max-tokens $(MAX_TOKENS) \
		--temperature 0.0

# Show model info
info: release
	./target/release/airllm-cli info --model $(MODEL)

# List tensors
tensors: release
	./target/release/airllm-cli tensors --model $(MODEL)

# List MLP tensors only
tensors-mlp: release
	./target/release/airllm-cli tensors --model $(MODEL) --filter mlp

# List attention tensors only
tensors-attn: release
	./target/release/airllm-cli tensors --model $(MODEL) --filter attn

# Build with Python bindings
python: 
	cargo build --release --features python

# Install Python package (requires maturin)
install-python:
	maturin develop --features python

# Run tests
test:
	cargo test

# Clean build artifacts
clean:
	cargo clean

# Help
help:
	@echo "AirLLM-RS Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make run              - Run inference with default settings"
	@echo "  make run-fast         - Quick run (5 tokens, temp=0.1)"
	@echo "  make run-greedy       - Greedy decoding (temp=0)"
	@echo "  make info             - Show model information"
	@echo "  make tensors          - List all tensors"
	@echo "  make tensors-mlp      - List MLP tensors"
	@echo "  make tensors-attn     - List attention tensors"
	@echo ""
	@echo "Variables:"
	@echo "  MODEL=/path/to/model  - Model directory (default: distillix)"
	@echo "  PROMPT=\"text\"         - Input prompt"
	@echo "  MAX_TOKENS=10         - Max tokens to generate"
	@echo "  TEMPERATURE=0.7       - Sampling temperature"
	@echo ""
	@echo "Examples:"
	@echo "  make run PROMPT=\"Hello world\" MAX_TOKENS=20"
	@echo "  make info MODEL=/path/to/llama"
