//! `larql vindex3 export` — export a canonical VINDEX3 container to Hugging Face safetensors
//! with exact read-back SHA-256 verification.

use std::path::PathBuf;

use clap::{Args, ValueEnum};
use larql_vindex::format::vindex3::export_safetensors::{
    export_safetensors, SafetensorsExportOptions, DEFAULT_MAX_SHARD_BYTES, EXPORT_MANIFEST_JSON,
};

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum ExportFormat {
    /// Exact raw-byte Hugging Face `.safetensors` directory (single file or sharded) +
    /// `export_manifest.json` with per-tensor SHA-256 verification.
    #[default]
    Safetensors,
}

#[derive(Args, Debug)]
pub struct ExportArgs {
    /// Source VINDEX3 container directory (or installed model shorthand).
    pub container: PathBuf,
    /// Output directory to create (must not already exist).
    #[arg(short, long)]
    pub output: PathBuf,
    /// Export format.
    #[arg(long, value_enum, default_value_t = ExportFormat::Safetensors)]
    pub format: ExportFormat,
    /// Optional original HF checkpoint directory to copy `config.json` (and any missing
    /// tokenizer/config sidecars) from.
    #[arg(long)]
    pub source_checkpoint: Option<PathBuf>,
    /// Maximum payload bytes per `.safetensors` shard before sharding (default: 4 GiB).
    #[arg(long, default_value_t = DEFAULT_MAX_SHARD_BYTES)]
    pub max_shard_bytes: u64,
}

pub(super) fn run(args: ExportArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.format {
        ExportFormat::Safetensors => {
            let options = SafetensorsExportOptions {
                source_checkpoint: args.source_checkpoint.as_deref(),
                max_shard_bytes: args.max_shard_bytes,
            };
            let report = export_safetensors(&args.container, &args.output, &options)?;
            println!(
                "Exported: {} ({} tensors, {} bytes, {} shard(s), SHA-256 verified)",
                args.output.display(),
                report.total_tensors,
                report.total_bytes,
                report.shards.len()
            );
            for shard in &report.shards {
                println!(
                    "  {}  ({} tensors, {} bytes, sha256 {})",
                    shard.filename,
                    shard.tensor_count,
                    shard.file_bytes,
                    &shard.sha256[..16]
                );
            }
            if !report.copied_files.is_empty() {
                println!("  copied: {}", report.copied_files.join(", "));
            }
            if args.source_checkpoint.is_none() {
                println!("  note: no --source-checkpoint given, so config.json was not copied");
            }
            println!(
                "  manifest: {}",
                args.output.join(EXPORT_MANIFEST_JSON).display()
            );
            Ok(())
        }
    }
}
