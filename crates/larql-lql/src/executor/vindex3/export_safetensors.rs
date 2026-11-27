//! `EXPORT SAFETENSORS [TO] "<output_dir>" [FROM [BASE] "<hf_checkpoint_dir>"];`:
//! export the bound canonical VINDEX3 container — including any live staged or applied
//! stored-weight edits — directly into a Hugging Face safetensors directory with full
//! independent read-back SHA-256 verification.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use larql_vindex::format::vindex3::export_safetensors::{
    export_safetensors as vindex_export_safetensors, LiveTensorOverrides, SafetensorsExportOptions,
    SafetensorsExportReport, DEFAULT_MAX_SHARD_BYTES,
};
use larql_vindex::format::vindex3::weight_patch::{
    declare_patches, materialise_declared_patches, WeightOpRef,
};

use crate::error::LqlError;
use crate::executor::{Backend, Session};

const STATEMENT: &str = "EXPORT SAFETENSORS";

impl Session {
    /// Typed operation behind `EXPORT SAFETENSORS`. Exports the bound canonical VINDEX3 container
    /// (with any pending stored-weight edits applied in memory) directly into a Hugging Face
    /// safetensors directory and returns the verified export manifest.
    pub fn export_safetensors(
        &self,
        output: &str,
        base_checkpoint: Option<&str>,
    ) -> Result<SafetensorsExportReport, LqlError> {
        let (root, overlay) = match &self.backend {
            Backend::Vindex3 { path, overlay, .. } => (path.as_path(), overlay),
            Backend::None => {
                return Err(LqlError::NoBackend);
            }
            Backend::Vindex { .. } | Backend::Weight { .. } | Backend::Remote { .. } => {
                return Err(LqlError::Execution(format!(
                    "{STATEMENT} requires a bound VINDEX3 container"
                )));
            }
        };

        if overlay.has_knowledge_state() {
            return Err(LqlError::Execution(format!(
                "{STATEMENT} refused: knowledge edits (INSERT / UPDATE / DELETE) are pending in \
                 this session; remove or compile them before exporting safetensors"
            )));
        }

        let live_map: Option<LiveTensorOverrides> = if overlay.has_weight_ops()
            || overlay.external_overlay().is_some()
        {
            let mut map: LiveTensorOverrides = BTreeMap::new();
            match overlay.live_weights() {
                Ok(Some(view)) => {
                    for tensor in view.tensors() {
                        map.insert(
                            (tensor.object.clone(), tensor.tensor.clone()),
                            (Arc::clone(&tensor.bytes), tensor.sha256.clone()),
                        );
                    }
                }
                Ok(None) => {}
                Err(
                    larql_vindex::format::vindex3::knowledge::LiveWeightRefusal::KnowledgePending,
                ) => {
                    return Err(LqlError::Execution(format!(
                        "{STATEMENT} refused: knowledge edits and stored-weight ops cannot both be pending"
                    )));
                }
                Err(_) => {
                    // Fallback for canonical weight ops outside the live execution seam
                    // (e.g., 3-D conv kernels or embedding weights): resolve through the exact
                    // same COMPILE byte path in memory.
                    let ops: Vec<WeightOpRef<'_>> =
                        overlay.weight_ops().into_iter().map(|(_, op)| op).collect();
                    let (patches, _origins) = declare_patches(root, &ops)
                        .map_err(|e| LqlError::Execution(format!("{STATEMENT} refused: {e}")))?;
                    let materialised = materialise_declared_patches(root, &patches)
                        .map_err(|e| LqlError::Execution(format!("{STATEMENT} refused: {e}")))?;
                    for tensor in materialised {
                        let sha = tensor.sha256();
                        map.insert(
                            (tensor.object, tensor.tensor),
                            (Arc::from(tensor.bytes), sha),
                        );
                    }
                }
            }
            Some(map)
        } else {
            None
        };

        let opts = SafetensorsExportOptions {
            source_checkpoint: base_checkpoint.map(Path::new),
            max_shard_bytes: DEFAULT_MAX_SHARD_BYTES,
            live_overrides: live_map.as_ref(),
        };

        vindex_export_safetensors(root, Path::new(output), &opts)
            .map_err(|e| LqlError::Execution(format!("{STATEMENT} failed: {e}")))
    }

    pub(crate) fn exec_export_safetensors(
        &self,
        output: &str,
        base_checkpoint: Option<&str>,
    ) -> Result<Vec<String>, LqlError> {
        let report = self.export_safetensors(output, base_checkpoint)?;
        let mut lines = vec![
            format!(
                "Exported {} tensor(s) ({:.2} MiB) across {} shard(s) to {}",
                report.total_tensors,
                report.total_bytes as f64 / (1024.0 * 1024.0),
                report.shards.len(),
                output,
            ),
            format!("  model:            {}", report.model),
            format!("  verified:         {}", report.verified),
        ];
        if report.live_overridden_tensors > 0 {
            lines.push(format!(
                "  live_patched:     {} tensor(s) applied from active session edits",
                report.live_overridden_tensors
            ));
        }
        if !report.copied_files.is_empty() {
            lines.push(format!(
                "  copied_files:     {}",
                report.copied_files.join(", ")
            ));
        }
        for shard in &report.shards {
            lines.push(format!(
                "  shard:            {} ({} tensors, sha256={})",
                shard.filename, shard.tensor_count, shard.sha256
            ));
        }
        Ok(lines)
    }
}
