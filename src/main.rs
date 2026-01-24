//! AirLLM-RS CLI: Layer-wise LLM inference

use airllm_rs::{InferenceEngine, GenerationConfig, ModelTokenizer, Result};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "airllm-cli")]
#[command(about = "High-performance layer-wise LLM inference", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference on a model
    Run {
        /// Path to model directory or HuggingFace repo ID
        #[arg(short, long)]
        model: String,

        /// Input prompt
        #[arg(short, long)]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(long, default_value = "256")]
        max_tokens: usize,

        /// Temperature (0.0 = greedy)
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Top-p (nucleus sampling)
        #[arg(long, default_value = "0.9")]
        top_p: f32,

        /// Repetition penalty (1.0 = disabled, >1.0 = penalize repetition)
        #[arg(long, default_value = "1.1")]
        repetition_penalty: f32,
    },

    /// Show model information
    Info {
        /// Path to model directory
        #[arg(short, long)]
        model: String,
    },

    /// List tensors in a model
    Tensors {
        /// Path to model directory
        #[arg(short, long)]
        model: String,

        /// Filter by pattern
        #[arg(short, long)]
        filter: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("airllm_rs=info".parse().unwrap()))
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            top_p,
            repetition_penalty,
        } => {
            run_inference(&model, &prompt, max_tokens, temperature, top_p, repetition_penalty).await?;
        }

        Commands::Info { model } => {
            show_model_info(&model)?;
        }

        Commands::Tensors { model, filter } => {
            list_tensors(&model, filter.as_deref())?;
        }
    }

    Ok(())
}

async fn run_inference(
    model_path: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
) -> Result<()> {
    println!("Loading model from: {}", model_path);

    // Load tokenizer
    let tokenizer = ModelTokenizer::from_dir(model_path)?;
    println!("Tokenizer loaded (vocab size: {})", tokenizer.vocab_size());

    // Load model
    let engine = InferenceEngine::from_pretrained(model_path)?;

    println!("Model loaded. Config:");
    println!("  Architecture: {:?}", engine.config().architecture);
    println!("  Layers: {}", engine.config().num_hidden_layers);
    println!("  Hidden size: {}", engine.config().hidden_size);
    println!("  Vocab size: {}", engine.config().vocab_size);

    // Create generation config
    let gen_config = GenerationConfig {
        max_new_tokens: max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        do_sample: temperature > 0.0,
        ..Default::default()
    };

    // Tokenize prompt
    println!("\nPrompt: {}", prompt);
    let mut input_ids = tokenizer.encode(prompt)?;
    
    // Add BOS token if available
    if let Some(bos) = tokenizer.bos_token_id() {
        input_ids.insert(0, bos);
    }
    
    println!("Input tokens: {:?} ({} tokens)\n", &input_ids, input_ids.len());

    // Generate
    println!("Generating {} tokens (this will be slow - CPU only, no SIMD)...\n", max_tokens);
    let output_ids = engine.generate(&input_ids, &gen_config)?;

    let new_tokens = output_ids.len() - input_ids.len();
    println!("\nGenerated {} new tokens", new_tokens);
    println!("Output token IDs: {:?}", &output_ids[input_ids.len()..]);

    // Decode output
    let output_text = tokenizer.decode(&output_ids)?;
    println!("\n=== Generated Text ===");
    println!("{}", output_text);
    println!("======================");

    Ok(())
}

fn show_model_info(model_path: &str) -> Result<()> {
    use airllm_rs::ModelConfig;

    let config = ModelConfig::from_dir(model_path)?;

    println!("Model Information");
    println!("=================");
    println!("Architecture: {:?}", config.architecture);
    println!("Hidden size: {}", config.hidden_size);
    println!("Layers: {}", config.num_hidden_layers);
    println!("Attention heads: {}", config.num_attention_heads);
    println!("KV heads: {}", config.num_kv_heads());
    println!("GQA ratio: {}x", config.gqa_ratio());
    println!("Intermediate size: {}", config.intermediate_size);
    println!("Vocab size: {}", config.vocab_size);
    println!("Max position: {}", config.max_position_embeddings);
    println!("RoPE theta: {}", config.rope_theta);
    println!("Estimated size: {}", config.size_string());

    Ok(())
}

fn list_tensors(model_path: &str, filter: Option<&str>) -> Result<()> {
    use airllm_rs::{LayerLoader, ModelConfig};
    

    let config = ModelConfig::from_dir(model_path)?;
    let naming = config.architecture.layer_naming();
    let loader = LayerLoader::new(model_path, naming, config.num_hidden_layers)?;

    println!("Tensors in model:");
    println!("=================");

    let mut count = 0;
    for name in loader.tensor_names() {
        if let Some(f) = filter {
            if !name.contains(f) {
                continue;
            }
        }
        println!("  {}", name);
        count += 1;
    }

    println!("\nTotal: {} tensors", count);

    Ok(())
}
