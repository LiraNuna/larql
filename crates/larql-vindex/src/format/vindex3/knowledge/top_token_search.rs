//! Exact two-stage branch-and-bound search over `FeatureMeta::top_token`
//! for VINDEX3 containers.
//!
//! A full-layer annotation computes `embedding [V, D] × down [D, F]` across
//! the entire vocabulary (e.g. `128k × 2048 × 7680` per layer). When a query
//! filters on `top_token` (such as `SELECT * FROM EDGES WHERE entity = "France"`
//! or `SHOW FEATURES 19 WHERE token = "French"`), only a tiny subset `S` of
//! vocabulary tokens satisfies the predicate.
//!
//! **Exactness proof:** For any feature column `f`, `annotate_logits` takes
//! the top `ANNOTATION_TOP_K` raw logits and picks the first token whose
//! trimmed decode is non-empty. If *any* non-empty token `p ∉ S` satisfies
//! `embedding[p] · down[:, f] > max_{v ∈ S} (embedding[v] · down[:, f])`
//! strictly, then either `p` ranks ahead of every `v ∈ S` (so `top_token` is
//! `p` or an even higher non-empty token) or `p` itself falls outside the
//! top `ANNOTATION_TOP_K` (so every `v ∈ S` also falls outside the top
//! `ANNOTATION_TOP_K`). In either case `top_token` cannot belong to `S`.
//!
//! Checking a small fixed probe set of non-empty tokens (`P1` of size ~128,
//! then `P2` of size ~1024 on the survivors) eliminates >99.5% of features in
//! milliseconds with zero false negatives, and exact full-vocabulary
//! `feature_promotions` verifies every surviving candidate so zero false
//! positives can escape.

use std::collections::HashSet;

use ndarray::Axis;
use rayon::prelude::*;

use crate::index::types::FeatureMeta;

use super::KnowledgeView;

/// Stage-1 probe budget (half highest-norm non-empty, half evenly spaced).
const PROBE_1_HALF: usize = 64;
/// Stage-2 probe budget on stage-1 survivors.
const PROBE_2_NORM: usize = 256;
const PROBE_2_SPACED: usize = 768;
/// If the matching token set `S` exceeds this size (e.g. a 1-letter
/// substring matching tens of thousands of tokens), a full cached layer
/// annotation is cheaper than a wide `emb_S × down` product.
const MAX_MATCH_SET: usize = 2048;

/// Cached vocabulary decode + probe token indices built once per session.
pub(super) struct VocabProbeIndex {
    decoded: Vec<Option<String>>,
    probe_ids_1: Vec<usize>,
    probe_ids_2: Vec<usize>,
}

impl VocabProbeIndex {
    pub(super) fn build(
        embedding: &ndarray::Array2<f32>,
        tokenizer: &crate::tokenizers::Tokenizer,
    ) -> Self {
        let vocab = embedding.nrows();
        let decoded: Vec<Option<String>> = (0..vocab)
            .into_par_iter()
            .map(|id| {
                tokenizer
                    .decode(&[id as u32], true)
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            })
            .collect();

        // Collect non-empty token ids with their embedding L2 norm².
        let mut nonempty: Vec<(usize, f32)> = decoded
            .par_iter()
            .enumerate()
            .filter_map(|(id, tok)| {
                tok.as_ref()?;
                let norm_sq: f32 = embedding.row(id).iter().map(|v| v * v).sum();
                Some((id, norm_sq))
            })
            .collect();

        let probe_ids_1 = pick_probes(&mut nonempty, PROBE_1_HALF, PROBE_1_HALF);
        let probe_ids_2 = pick_probes(&mut nonempty, PROBE_2_NORM, PROBE_2_SPACED);

        Self {
            decoded,
            probe_ids_1,
            probe_ids_2,
        }
    }
}

fn pick_probes(nonempty: &mut [(usize, f32)], top_norm: usize, spaced: usize) -> Vec<usize> {
    if nonempty.is_empty() {
        return Vec::new();
    }
    // Sort by descending norm² to grab the strongest-norm tokens first.
    nonempty.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut seen = HashSet::with_capacity(top_norm + spaced);
    let mut out = Vec::with_capacity(top_norm + spaced);
    for &(id, _) in nonempty.iter().take(top_norm) {
        if seen.insert(id) {
            out.push(id);
        }
    }
    if let Some(raw_step) = nonempty.len().checked_div(spaced) {
        let step = raw_step.max(1);
        for &(id, _) in nonempty.iter().step_by(step).take(spaced) {
            if seen.insert(id) {
                out.push(id);
            }
        }
    }
    out
}

impl KnowledgeView {
    fn vocab_probe_index(&self) -> &VocabProbeIndex {
        self.vocab_probes
            .get_or_init(|| VocabProbeIndex::build(&self.embedding, &self.tokenizer))
    }

    /// Return every `(feature, FeatureMeta)` at `layer` whose exact
    /// `top_token` satisfies `predicate`, in ascending feature order.
    ///
    /// Uses cached `feature_metas` when already populated; otherwise runs
    /// exact two-stage branch-and-bound pruning before verifying survivors
    /// against the full vocabulary.
    pub fn find_features_by_top_token<F>(
        &self,
        layer: usize,
        predicate: F,
    ) -> Vec<(usize, FeatureMeta)>
    where
        F: Fn(&str) -> bool + Sync,
    {
        let Some(Some(knowledge)) = self.layers.get(layer) else {
            return Vec::new();
        };
        if let Some(metas) = knowledge.metas.get() {
            return metas
                .iter()
                .enumerate()
                .filter_map(|(feat, m)| {
                    let meta = m.as_ref()?;
                    predicate(&meta.top_token).then(|| (feat, meta.clone()))
                })
                .collect();
        }

        let idx = self.vocab_probe_index();
        let s_ids: Vec<usize> = idx
            .decoded
            .iter()
            .enumerate()
            .filter_map(|(id, tok)| tok.as_deref().is_some_and(&predicate).then_some(id))
            .collect();
        if s_ids.is_empty() {
            return Vec::new();
        }
        if s_ids.len() > MAX_MATCH_SET {
            return self
                .feature_metas(layer)
                .map(|metas| {
                    metas
                        .iter()
                        .enumerate()
                        .filter_map(|(feat, m)| {
                            let meta = m.as_ref()?;
                            predicate(&meta.top_token).then(|| (feat, meta.clone()))
                        })
                        .collect()
                })
                .unwrap_or_default();
        }

        let s_set: HashSet<usize> = s_ids.iter().copied().collect();
        let emb_s = self.embedding.select(Axis(0), &s_ids);
        let s_scores = emb_s.dot(&knowledge.down); // [|S|, F]
        let num_features = knowledge.down.ncols();
        let s_max: Vec<f32> = (0..num_features)
            .into_par_iter()
            .map(|f| {
                s_scores
                    .column(f)
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();

        // Stage 1 pruning over all features.
        let p1_ids: Vec<usize> = idx
            .probe_ids_1
            .iter()
            .copied()
            .filter(|id| !s_set.contains(id))
            .collect();
        let mut survivors: Vec<usize> = if p1_ids.is_empty() {
            (0..num_features).collect()
        } else {
            let emb_p1 = self.embedding.select(Axis(0), &p1_ids);
            let p1_scores = emb_p1.dot(&knowledge.down);
            (0..num_features)
                .into_par_iter()
                .filter(|&f| {
                    let p_max = p1_scores
                        .column(f)
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    p_max <= s_max[f]
                })
                .collect()
        };
        if survivors.is_empty() {
            return Vec::new();
        }

        // Stage 2 pruning on stage-1 survivors.
        let p2_ids: Vec<usize> = idx
            .probe_ids_2
            .iter()
            .copied()
            .filter(|id| !s_set.contains(id))
            .collect();
        if !p2_ids.is_empty() && survivors.len() > 8 {
            let emb_p2 = self.embedding.select(Axis(0), &p2_ids);
            let down_sub = knowledge.down.select(Axis(1), &survivors);
            let p2_scores = emb_p2.dot(&down_sub);
            survivors = survivors
                .into_par_iter()
                .enumerate()
                .filter_map(|(col, f)| {
                    let p_max = p2_scores
                        .column(col)
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    (p_max <= s_max[f]).then_some(f)
                })
                .collect();
        }
        if survivors.is_empty() {
            return Vec::new();
        }

        // Stage 3: exact full-vocabulary verification of every survivor.
        let promoted = self
            .feature_promotions(layer, &survivors)
            .unwrap_or_default();
        survivors
            .into_iter()
            .zip(promoted)
            .filter_map(|(feat, meta)| {
                let meta = meta?;
                predicate(&meta.top_token).then_some((feat, meta))
            })
            .collect()
    }
}
