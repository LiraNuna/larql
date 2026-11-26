//! Session-scoped prefix KV & recurrent continuation-state cache for VINDEX3.
//!
//! Reuses exact `CanonicalKvState` snapshots (covering softmax KV rows, `ShortConv`
//! / recurrent buffers, and MLA latent caches) across consecutive statements in one
//! session when prompts share a common prefix or extend a prior prompt.
//!
//! Invalidated automatically whenever live weight patches change (`PreparedCacheKey`),
//! and bypassed for compose vector edits or non-single-stream topologies.

use std::sync::Mutex;

use larql_kv::CanonicalKvState;
use larql_vindex::format::vindex3::opplan::exec::continuation::plan_continuation_geometry;
use larql_vindex::format::vindex3::opplan::exec::prefill_prepared;
use larql_vindex::format::vindex3::opplan::exec::prepared::PreparedOperands;

use super::{PreparedCacheKey, V3Runtime};
use crate::error::LqlError;

/// Minimum shared prefix length (in tokens) before splitting a prefill to snapshot the prefix.
const MIN_SHARED_PREFIX_TOKENS: usize = 4;
/// Maximum number of prefix snapshots kept resident in one session (keeps RAM bounded).
const MAX_SNAPSHOTS: usize = 4;

struct PrefixSnapshot {
    tokens: Vec<u32>,
    kv: CanonicalKvState,
    last_logits: Option<Vec<f32>>,
}

#[derive(Default)]
struct PrefixCacheInner {
    key: Option<PreparedCacheKey>,
    entries: Vec<PrefixSnapshot>,
    last_prompt: Vec<u32>,
}

/// Thread-safe per-session cache of prefix `CanonicalKvState` snapshots.
#[derive(Default)]
pub(crate) struct PrefixStateCache {
    inner: Mutex<PrefixCacheInner>,
}

impl PrefixStateCache {
    pub(crate) fn clear(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.key = None;
            guard.entries.clear();
            guard.last_prompt.clear();
        }
    }

    fn ensure_key(&self, key: &PreparedCacheKey) {
        let mut guard = self.inner.lock().unwrap();
        if guard.key.as_ref() != Some(key) {
            guard.key = Some(key.clone());
            guard.entries.clear();
            guard.last_prompt.clear();
        }
    }

    fn find_exact_with_logits(
        &self,
        key: &PreparedCacheKey,
        tokens: &[u32],
    ) -> Option<(CanonicalKvState, Vec<f32>)> {
        let guard = self.inner.lock().unwrap();
        if guard.key.as_ref() != Some(key) {
            return None;
        }
        guard.entries.iter().rev().find_map(|e| {
            if e.tokens == tokens {
                e.last_logits.as_ref().map(|l| (e.kv.clone(), l.clone()))
            } else {
                None
            }
        })
    }

    fn find_longest_prefix(
        &self,
        key: &PreparedCacheKey,
        tokens: &[u32],
        max_prefix_len: usize,
    ) -> Option<(usize, CanonicalKvState)> {
        let guard = self.inner.lock().unwrap();
        if guard.key.as_ref() != Some(key) {
            return None;
        }
        let limit = max_prefix_len.min(tokens.len().saturating_sub(1));
        guard
            .entries
            .iter()
            .filter(|e| {
                !e.tokens.is_empty() && e.tokens.len() <= limit && tokens.starts_with(&e.tokens)
            })
            .max_by_key(|e| e.tokens.len())
            .map(|e| (e.tokens.len(), e.kv.clone()))
    }

    fn shared_prefix_with_last(
        &self,
        key: &PreparedCacheKey,
        tokens: &[u32],
        max_prefix_len: usize,
    ) -> usize {
        let guard = self.inner.lock().unwrap();
        if guard.key.as_ref() != Some(key) {
            return 0;
        }
        let limit = max_prefix_len.min(tokens.len().saturating_sub(1));
        common_prefix_len(&guard.last_prompt, tokens).min(limit)
    }

    pub(crate) fn record_snapshot(
        &self,
        key: &PreparedCacheKey,
        tokens: Vec<u32>,
        kv: CanonicalKvState,
        last_logits: Option<Vec<f32>>,
    ) {
        let mut guard = self.inner.lock().unwrap();
        if guard.key.as_ref() != Some(key) {
            return;
        }
        guard.last_prompt = tokens.clone();
        if tokens.len() < MIN_SHARED_PREFIX_TOKENS {
            return;
        }
        if let Some(pos) = guard.entries.iter().position(|e| e.tokens == tokens) {
            guard.entries.remove(pos);
        } else if guard.entries.len() >= MAX_SNAPSHOTS {
            guard.entries.remove(0);
        }
        guard.entries.push(PrefixSnapshot {
            tokens,
            kv,
            last_logits,
        });
    }

    fn record_prefix_only(
        &self,
        key: &PreparedCacheKey,
        tokens: Vec<u32>,
        kv: CanonicalKvState,
        last_logits: Option<Vec<f32>>,
    ) {
        if tokens.len() < MIN_SHARED_PREFIX_TOKENS {
            return;
        }
        let mut guard = self.inner.lock().unwrap();
        if guard.key.as_ref() != Some(key) {
            return;
        }
        if let Some(pos) = guard.entries.iter().position(|e| e.tokens == tokens) {
            guard.entries.remove(pos);
        } else if guard.entries.len() >= MAX_SNAPSHOTS {
            guard.entries.remove(0);
        }
        guard.entries.push(PrefixSnapshot {
            tokens,
            kv,
            last_logits,
        });
    }

    fn record_last_prompt(&self, key: &PreparedCacheKey, tokens: &[u32]) {
        let mut guard = self.inner.lock().unwrap();
        if guard.key.as_ref() == Some(key) {
            guard.last_prompt = tokens.to_vec();
        }
    }
}

fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Whether `plan` can be snapshotted into a `CanonicalKvState` for prefix caching.
pub(crate) fn plan_supports_prefix_cache(runtime: &V3Runtime) -> bool {
    let plan = runtime.plan();
    plan.residual_topology.is_single_stream() && plan_continuation_geometry(plan).is_ok()
}

/// Acquire a `CanonicalKvState` positioned at `prefix_len <= max_prefix_len` for `prompt_ids`:
/// - Reuses the longest cached strict prefix snapshot if one matches;
/// - Otherwise, if `prompt_ids` shares a prefix of at least `MIN_SHARED_PREFIX_TOKENS` with
///   the previous prompt, prefills that shared prefix once, caches it, and returns it;
/// - Otherwise returns a fresh `CanonicalKvState` at position 0.
pub(crate) fn acquire_prefix_state(
    runtime: &V3Runtime,
    ops: &PreparedOperands,
    cache: &PrefixStateCache,
    key: &PreparedCacheKey,
    prompt_ids: &[u32],
    max_prefix_len: usize,
) -> Result<(CanonicalKvState, usize), LqlError> {
    cache.ensure_key(key);
    let limit = max_prefix_len.min(prompt_ids.len().saturating_sub(1));
    if let Some((p, state)) = cache.find_longest_prefix(key, prompt_ids, limit) {
        return Ok((state, p));
    }
    let lcp = cache.shared_prefix_with_last(key, prompt_ids, limit);
    if lcp >= MIN_SHARED_PREFIX_TOKENS {
        let mut kv = CanonicalKvState::new();
        let prefix_out = prefill_prepared(
            runtime.plan(),
            ops,
            &prompt_ids[..lcp],
            runtime.backend(),
            &mut kv,
        )
        .map_err(|e| LqlError::exec("v3 prefix prefill failed", e))?;
        cache.record_prefix_only(
            key,
            prompt_ids[..lcp].to_vec(),
            kv.clone(),
            prefix_out.logits,
        );
        return Ok((kv, lcp));
    }
    Ok((CanonicalKvState::new(), 0))
}

/// Prefill `prompt_ids` into a `CanonicalKvState` (reusing an exact cached match or shared
/// prefix snapshot) and return `(kv, last_logits)` ready for autoregressive decoding.
pub(crate) fn prefill_for_generate_cached(
    runtime: &V3Runtime,
    ops: &PreparedOperands,
    cache: &PrefixStateCache,
    key: &PreparedCacheKey,
    prompt_ids: &[u32],
) -> Result<(CanonicalKvState, Vec<f32>), LqlError> {
    cache.ensure_key(key);
    if let Some(hit) = cache.find_exact_with_logits(key, prompt_ids) {
        cache.record_last_prompt(key, prompt_ids);
        return Ok(hit);
    }
    let (mut kv, start) = acquire_prefix_state(
        runtime,
        ops,
        cache,
        key,
        prompt_ids,
        prompt_ids.len().saturating_sub(1),
    )?;
    let out = prefill_prepared(
        runtime.plan(),
        ops,
        &prompt_ids[start..],
        runtime.backend(),
        &mut kv,
    )
    .map_err(|e| LqlError::exec("v3 prefill failed", e))?;
    let logits = out
        .logits
        .ok_or_else(|| LqlError::Execution("the component carries no output head".into()))?;
    cache.record_snapshot(key, prompt_ids.to_vec(), kv.clone(), Some(logits.clone()));
    Ok((kv, logits))
}
