//! `larql export-hf <VINDEX3> --output <OUT_DIR> [--base <HF_DIR>] [--max-shard-gb <N>]`:
//! export a canonical VINDEX3 container directly to Hugging Face `.safetensors` files with
//! independent read-back SHA-256 verification.

use std::path::PathBuf;

use clap::Args;
use larql_vindex::format::vindex3::export_safetensors::{
    export_safetensors, SafetensorsExportOptions,
};

#[derive(Args)]
pub struct ExportHfArgs {
    /// Path to the canonical VINDEX3 container directory.
    pub vindex: PathBuf,

    /// Output directory to create (must not already exist).
    #[arg(short, long)]
    pub output: PathBuf,

    /// Optional base Hugging Face checkpoint directory to copy config.json (and any missing
    /// tokenizer/generation capability files) from.
    #[arg(long, alias = "source-checkpoint")]
    pub base: Option<PathBuf>,

    /// Maximum payload size per safetensors shard in GiB (default: 4.0).
    #[arg(long, default_value_t = 4.0)]
    pub max_shard_gb: f64,
}

pub fn run(args: ExportHfArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.vindex.is_dir() {
        return Err(format!("not a directory: {}", args.vindex.display()).into());
    }
    if !args.max_shard_gb.is_finite() || args.max_shard_gb <= 0.0 {
        return Err("--max-shard-gb must be a finite positive number".into());
    }
    let max_shard_bytes = (args.max_shard_gb * (1024.0 * 1024.0 * 1024.0)).round() as u64;
    let opts = SafetensorsExportOptions {
        source_checkpoint: args.base.as_deref(),
        max_shard_bytes: max_shard_bytes.max(1),
        live_overrides: None,
    };

    let report = export_safetensors(&args.vindex, &args.output, &opts)?;
    println!(
        "Exported {} tensor(s) ({:.2} MiB) across {} shard(s) to {}",
        report.total_tensors,
        report.total_bytes as f64 / (1024.0 * 1024.0),
        report.shards.len(),
        args.output.display()
    );
    println!("  model:        {}", report.model);
    println!("  verified:     {}", report.verified);
    if !report.copied_files.is_empty() {
        println!("  copied_files: {}", report.copied_files.join(", "));
    }
    for shard in &report.shards {
        println!(
            "  shard:        {} ({} tensors, sha256={})",
            shard.filename, shard.tensor_count, shard.sha256
        );
    }
    Ok(())
}
