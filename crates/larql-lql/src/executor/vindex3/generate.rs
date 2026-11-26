//! Greedy generation over the session's program: the one path `INFER … GENERATE` and `CAPTURE … GENERATE` both
//! run, so the tokens a capture is taken along are the tokens INFER prints.

use larql_inference::layer_graph::generate::detok::Detokenizer;
use larql_inference::vindex3::{continue_session, SessionGeneration, Vindex3Session};
use larql_inference::{EosConfig, SamplingConfig};
use larql_vindex::format::vindex3::knowledge::KnowledgeOverlay;
use larql_vindex::format::vindex3::opplan::exec::kv::KvState;
use larql_vindex::format::vindex3::opplan::exec::prefill_prepared;
use larql_vindex::tokenizers::Tokenizer;

use super::{
    prefix_cache, prepared_operands_for_session, v3_continuation, PreparedCache, V3Runtime,
};
use crate::error::LqlError;

/// Batch-prefill `prompt_ids` into a continuation (reusing any cached shared prefix), then
/// continue greedily for up to `max_new` tokens, stopping before a built-in stop token.
/// Returns the generated ids and their decoded text.
pub(crate) fn greedy_generate(
    runtime: &V3Runtime,
    overlay: &KnowledgeOverlay,
    prepared_cache: &PreparedCache,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_new: usize,
) -> Result<(SessionGeneration, String), LqlError> {
    let (ops, _, cache_key) = prepared_operands_for_session(runtime, overlay, prepared_cache)?;
    let (mut kv, prefill_logits): (Box<dyn KvState>, Vec<f32>) = if let Some(ref key) = cache_key
        .filter(|_| !prompt_ids.is_empty() && prefix_cache::plan_supports_prefix_cache(runtime))
    {
        let (state, logits) = prefix_cache::prefill_for_generate_cached(
            runtime,
            &ops,
            prepared_cache.prefix(),
            key,
            prompt_ids,
        )?;
        (Box::new(state), logits)
    } else {
        let mut kv = v3_continuation(runtime)?.build();
        let logits = prefill_prepared(
            runtime.plan(),
            &ops,
            prompt_ids,
            runtime.backend(),
            &mut *kv,
        )
        .map_err(|e| LqlError::exec("v3 prefill failed", e))?
        .logits
        .ok_or_else(|| LqlError::Execution("the component carries no output head".into()))?;
        (kv, logits)
    };
    let mut session =
        Vindex3Session::over_prepared(runtime.plan(), &ops, runtime.backend(), &mut *kv)
            .map_err(|e| LqlError::exec("v3 session failed", e))?;
    let mut detok = Detokenizer::new(tokenizer);
    detok.seed(prompt_ids);
    let mut text = String::new();
    let result = continue_session(
        &mut session,
        prefill_logits,
        max_new,
        SamplingConfig::greedy(),
        &EosConfig::builtin(),
        |id| text.push_str(&detok.push(id)),
    )
    .map_err(|e| LqlError::exec("v3 decode failed", e))?;
    Ok((result, text))
}
