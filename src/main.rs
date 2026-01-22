//! AirLLM-RS CLI: Layer-wise LLM inference

use airllm_rs::{InferenceEngine, GenerationConfig, Result};
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
        } => {
            run_inference(&model, &prompt, max_tokens, temperature, top_p).await?;
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
) -> Result<()> {
    println!("Loading model from: {}", model_path);

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
        do_sample: temperature > 0.0,
        ..Default::default()
    };

    // TODO: Tokenize prompt
    // For now, use placeholder token IDs
    println!("\nPrompt: {}", prompt);
    println!("(Note: Tokenization not implemented yet, using placeholder IDs)\n");

    // Placeholder: Just use some dummy token IDs
    let input_ids: Vec<u32> = vec![1, 15043, 29892, 920, 526, 366]; // "Hello, how are you"

    // Generate
    let output_ids = engine.generate(&input_ids, &gen_config)?;

    println!("Generated {} tokens", output_ids.len() - input_ids.len());
    println!("Token IDs: {:?}", &output_ids[input_ids.len()..]);

    // TODO: Decode tokens
    println!("\n(Token decoding not implemented yet)");

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
