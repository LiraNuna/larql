//! Execution of `OVERLAY SAFETENSORS`, `OVERLAY LORA`, and `OVERLAY CLEAR` on VINDEX3 bindings.

use std::path::Path;

use larql_vindex::format::vindex3::knowledge::{
    load_lora_overlay, load_safetensors_overlay, ExternalOverlaySummary,
};

use crate::ast::OverlayKind;
use crate::error::LqlError;
use crate::executor::{Backend, Session};

impl Session {
    /// Typed operation behind `OVERLAY SAFETENSORS` / `OVERLAY LORA` / `OVERLAY CLEAR`.
    /// Returns `Some(summary)` when an overlay was mounted, or `None` when cleared.
    pub fn apply_external_overlay(
        &mut self,
        kind: &OverlayKind,
    ) -> Result<Option<ExternalOverlaySummary>, LqlError> {
        let (path, runtime, overlay, prepared_cache) = match &mut self.backend {
            Backend::Vindex3 {
                path,
                runtime,
                overlay,
                prepared_cache,
                ..
            } => (path, runtime, overlay, prepared_cache),
            Backend::None => return Err(LqlError::NoBackend),
            _ => {
                return Err(LqlError::Execution(
                    "OVERLAY requires a bound canonical VINDEX3 container".into(),
                ));
            }
        };

        if overlay.has_knowledge_state() {
            return Err(LqlError::Execution(
                "OVERLAY refused: knowledge edits (INSERT / UPDATE / DELETE) are pending in this session".into(),
            ));
        }

        let (summary, needs_refresh) = match kind {
            OverlayKind::Clear => {
                overlay.clear_external_overlay();
                prepared_cache.clear_patched();
                (None, overlay.has_weight_ops())
            }
            OverlayKind::Safetensors { path: src } => {
                let loaded = load_safetensors_overlay(path, runtime.plan(), Path::new(src))
                    .map_err(|e| LqlError::exec("OVERLAY SAFETENSORS failed", e))?;
                let summary = loaded.summary();
                overlay.set_external_overlay(loaded);
                prepared_cache.clear_patched();
                (Some(summary), overlay.has_weight_ops())
            }
            OverlayKind::Lora { path: src, scale } => {
                let loaded = load_lora_overlay(path, runtime.plan(), Path::new(src), *scale)
                    .map_err(|e| LqlError::exec("OVERLAY LORA failed", e))?;
                let summary = loaded.summary();
                overlay.set_external_overlay(loaded);
                prepared_cache.clear_patched();
                (Some(summary), overlay.has_weight_ops())
            }
        };
        if needs_refresh {
            let _ = self.refresh_live_weights();
        }
        Ok(summary)
    }

    pub(crate) fn exec_overlay(&mut self, kind: &OverlayKind) -> Result<Vec<String>, LqlError> {
        match self.apply_external_overlay(kind)? {
            None => Ok(vec![
                "Cleared external weight overlay (restored bound base container).".into(),
            ]),
            Some(s) => {
                let mut lines = vec![format!(
                    "Mounted {} overlay from {} ({} modified tensor(s), {} unchanged skipped, {} scanned, {:.0}ms):",
                    s.kind,
                    s.source,
                    s.overlaid_tensors,
                    s.unchanged_skipped,
                    s.total_scanned,
                    s.elapsed_ms,
                )];
                if let Some(sc) = s.lora_scale {
                    lines.push(format!("  lora_scale: {sc}"));
                }
                for name in s.tensors.iter().take(12) {
                    lines.push(format!("  overlaid:   {name}"));
                }
                if s.tensors.len() > 12 {
                    lines.push(format!("  ... and {} more tensor(s)", s.tensors.len() - 12));
                }
                Ok(lines)
            }
        }
    }
}
