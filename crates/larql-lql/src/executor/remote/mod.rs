//! Remote executor — forwards LQL queries to a `larql-server` over HTTP.
//!
//! Layout:
//!
//!   - `mod.rs` (this file): connection plumbing (`exec_use_remote`,
//!     `is_remote`, `require_remote`), generic HTTP helpers, and
//!     centralised endpoint / header constants.
//!   - `query.rs`: read-side verbs (DESCRIBE, WALK, INFER, EXPLAIN
//!     INFER, STATS, SHOW RELATIONS, SELECT) plus shared `knn_override`
//!     formatters.
//!   - `mutation.rs`: write-side verbs (INSERT, DELETE, UPDATE) plus
//!     session patch management (APPLY/SHOW/REMOVE PATCH).
//!
//! Every request carries the session header, so every mutation lands in
//! this session's patch overlay on the server (never the shared model)
//! and every read is answered against that same overlay.

use super::Backend;
use super::Session;
use crate::ast::{Condition, ExplainMode, Statement};
use crate::error::LqlError;
use crate::executor::predicate::{EdgePredicate, PredicateScope};

mod mutation;
mod query;

// ── Endpoint constants ───────────────────────────────────────────
//
// Every remote forwarder used to embed its `/v1/foo` path inline. A
// stale or moved endpoint would silently 404 across multiple call
// sites; centralising means a server-side rename is a one-line patch.

pub(super) const ENDPOINT_STATS: &str = "/v1/stats";
pub(super) const ENDPOINT_DESCRIBE: &str = "/v1/describe";
pub(super) const ENDPOINT_WALK: &str = "/v1/walk";
pub(super) const ENDPOINT_INFER: &str = "/v1/infer";
pub(super) const ENDPOINT_EXPLAIN_INFER: &str = "/v1/explain-infer";
pub(super) const ENDPOINT_RELATIONS: &str = "/v1/relations";
pub(super) const ENDPOINT_SELECT: &str = "/v1/select";
pub(super) const ENDPOINT_INSERT: &str = "/v1/insert";
pub(super) const ENDPOINT_PATCHES_APPLY: &str = "/v1/patches/apply";
/// `GET` lists the session's patches; `DELETE {ENDPOINT_PATCHES}/{name}`
/// removes one.
pub(super) const ENDPOINT_PATCHES: &str = "/v1/patches";

/// HTTP header naming this client's session on the server. Sent on every
/// request: mutations are scoped to the session's patch overlay, and
/// reads are answered against it.
pub(super) const SESSION_HEADER: &str = "x-session-id";

/// Field of the `/v1/stats` response naming the served model — the
/// identity a patch must declare as its `base_model`.
const STATS_MODEL_FIELD: &str = "model";

/// Banner text when `/v1/stats` names no model.
const UNNAMED_MODEL_BANNER: &str = "unknown";

/// Default request timeout. Long enough to absorb a deep-walk INFER
/// on a 4B-class model; short enough that a hung server doesn't
/// freeze the REPL indefinitely.
pub(super) const DEFAULT_REMOTE_TIMEOUT_SECS: u64 = 30;

// ── WHERE-clause helpers ─────────────────────────────────────────
//
// Used by remote DELETE / UPDATE: both verbs require an explicit
// `layer = N AND feature = M` predicate; missing fields used to
// silently default to `0` and would clobber `L0 F0` on the server.

/// Require an explicit `layer = N AND feature = M` from a remote
/// DELETE / UPDATE WHERE clause. Errors with a verb-tagged message
/// when either is missing or non-positive-integer.
pub(super) fn require_layer_feature(
    conditions: &[Condition],
    verb: &str,
) -> Result<(usize, usize), LqlError> {
    // The same grammar as a local mutation: a clause with a condition
    // that cannot be evaluated is refused before anything is sent.
    let predicate = EdgePredicate::parse(conditions, PredicateScope::EDGES)?;
    if predicate.entity.is_some() || predicate.relation.is_some() || predicate.score.is_some() {
        return Err(LqlError::Execution(format!(
            "remote {verb} addresses one slot by `layer` and `feature`; entity, relation \
             and score conditions are not evaluated remotely and would be dropped"
        )));
    }
    let layer = predicate.layer.ok_or_else(|| {
        LqlError::Execution(format!(
            "remote {verb} requires `layer = <int>` in WHERE clause"
        ))
    })?;
    let feature = predicate.feature.ok_or_else(|| {
        LqlError::Execution(format!(
            "remote {verb} requires `feature = <int>` in WHERE clause"
        ))
    })?;
    Ok((layer, feature))
}

impl Session {
    /// Execute a statement against a remote backend.
    pub(super) fn execute_remote(&mut self, stmt: &Statement) -> Result<Vec<String>, LqlError> {
        match stmt {
            Statement::Use { target } => self.exec_use(target),
            Statement::Describe {
                entity, band, mode, ..
            } => self.remote_describe(entity, *band, *mode),
            Statement::Walk {
                prompt,
                top,
                layers,
                ..
            } => self.remote_walk(prompt, *top, layers.as_ref()),
            Statement::Infer {
                prompt,
                top,
                compare,
                route: _,
                generate,
            } => {
                if generate.is_some() {
                    return Err(LqlError::Execution(
                        "INFER GENERATE is not supported on a remote backend yet".into(),
                    ));
                }
                self.remote_infer(prompt, *top, *compare)
            }
            Statement::Stats { .. } => self.remote_stats(),
            Statement::ShowRelations {
                mode,
                with_examples,
                ..
            } => self.remote_show_relations(*mode, *with_examples),
            Statement::Insert {
                entity,
                relation,
                target,
                layer,
                confidence,
                alpha: _,
                mode: _,
            } => {
                // Remote backend doesn't forward ALPHA or MODE — the
                // HTTP protocol doesn't have a schema for them yet.
                // Local backend honours both via `exec_insert`.
                self.remote_insert(entity, relation, target, *layer, *confidence)
            }
            Statement::Delete { conditions } => self.remote_delete(conditions),
            Statement::Update { set, conditions } => self.remote_update(set, conditions),
            Statement::Select {
                source: _,
                fields: _,
                conditions,
                nearest: _,
                order: _,
                limit,
            } => self.remote_select(conditions, *limit),
            Statement::Explain {
                prompt,
                mode,
                layers,
                band,
                verbose: _,
                top,
                relations_only,
                with_attention,
            } => match mode {
                ExplainMode::Infer => self.remote_explain_infer(
                    prompt,
                    *top,
                    *band,
                    *relations_only,
                    *with_attention,
                ),
                ExplainMode::Walk => self.remote_walk(prompt, *top, layers.as_ref()),
            },
            Statement::ApplyPatch { path } => self.remote_apply_patch(path),
            Statement::ShowPatches => self.remote_show_patches(),
            Statement::RemovePatch { path } => self.remote_remove_patch(path),
            Statement::Help { topic } => crate::help::help_lines(topic.as_deref()),
            Statement::Pipe { left, right } => {
                let mut out = self.execute(left)?;
                out.extend(self.execute(right)?);
                Ok(out)
            }
            _ => Err(LqlError::Execution(
                "this statement is not supported on a remote backend. \
                 Supported: DESCRIBE, WALK, INFER, EXPLAIN INFER, EXPLAIN WALK, SELECT, STATS, \
                 SHOW RELATIONS, INSERT, DELETE, UPDATE, APPLY PATCH, SHOW PATCHES, REMOVE PATCH, USE. \
                 TRACE requires a local vindex (USE \"path.vindex\")."
                    .into(),
            )),
        }
    }

    /// Connect to a remote larql-server.
    pub(crate) fn exec_use_remote(&mut self, url: &str) -> Result<Vec<String>, LqlError> {
        let url = url.trim_end_matches('/').to_string();

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(DEFAULT_REMOTE_TIMEOUT_SECS))
            .build()
            .map_err(|e| LqlError::exec("failed to create HTTP client", e))?;

        // Verify the server is reachable by hitting /v1/stats.
        let stats_url = format!("{url}{ENDPOINT_STATS}");
        let resp = client
            .get(&stats_url)
            .send()
            .map_err(|e| LqlError::exec(format!("failed to connect to {url}"), e))?;

        if !resp.status().is_success() {
            return Err(LqlError::Execution(format!(
                "server returned {}: {}",
                resp.status(),
                resp.text().unwrap_or_default()
            )));
        }

        let stats: serde_json::Value = resp
            .json()
            .map_err(|e| LqlError::exec("invalid response from server", e))?;

        // The served model's identity, kept so every patch this session
        // sends can declare it as its `base_model`. Absent or empty means
        // the server did not say, and mutations are refused rather than
        // sent with an empty one.
        let model_name = stats[STATS_MODEL_FIELD]
            .as_str()
            .filter(|name| !name.is_empty())
            .map(str::to_string);
        let model = model_name.as_deref().unwrap_or(UNNAMED_MODEL_BANNER);
        let layers = stats["layers"].as_u64().unwrap_or(0);
        let features = stats["features"].as_u64().unwrap_or(0);

        // Per-session id for the server-side patch overlay. PID + millis is
        // weak entropy but acceptable for the localhost dev model;
        // production deployments should authenticate before this is
        // load-bearing.
        let session_id = format!(
            "larql-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );

        let banner =
            format!("Connected: {model} ({layers} layers, {features} features)\n  Remote: {url}");
        self.backend = Backend::Remote {
            url,
            client,
            session_id,
            model_name,
        };
        self.reset_binding_state();
        Ok(vec![banner])
    }

    /// True iff the active backend is the Remote variant.
    pub(crate) fn is_remote(&self) -> bool {
        matches!(&self.backend, Backend::Remote { .. })
    }

    /// Get the remote URL, client, and session ID, or error.
    pub(super) fn require_remote(
        &self,
    ) -> Result<(&str, &reqwest::blocking::Client, &str), LqlError> {
        match &self.backend {
            Backend::Remote {
                url,
                client,
                session_id,
                ..
            } => Ok((url, client, session_id)),
            _ => Err(LqlError::Execution(
                "not connected to a remote server".into(),
            )),
        }
    }

    /// The served model's identity, as `/v1/stats` named it at `USE
    /// REMOTE` — the `base_model` every patch this session sends must
    /// declare. A server that named no model gets no patch: `verb` is
    /// refused by name instead of sending an empty `base_model`.
    pub(super) fn remote_model_name(&self, verb: &str) -> Result<&str, LqlError> {
        match &self.backend {
            Backend::Remote { model_name, .. } => model_name.as_deref().ok_or_else(|| {
                LqlError::Execution(format!(
                    "remote {verb} refused: the server's {ENDPOINT_STATS} named no model, \
                     so the patch cannot declare which model it belongs to. Nothing was sent."
                ))
            }),
            _ => Err(LqlError::Execution(
                "not connected to a remote server".into(),
            )),
        }
    }

    // ── Generic HTTP forwarding helpers ──
    //
    // Each one attaches the session header: a mutation lands in this
    // session's overlay, and a read is answered against it.

    /// GET `{remote_url}{endpoint}` with optional query parameters,
    /// check the response status, and parse the body as JSON.
    pub(super) fn remote_get_json(
        &self,
        endpoint: &str,
        query: &[(&str, &str)],
    ) -> Result<serde_json::Value, LqlError> {
        let (url, client, sid) = self.require_remote()?;
        let resp = client
            .get(format!("{url}{endpoint}"))
            .query(query)
            .header(SESSION_HEADER, sid)
            .send()
            .map_err(|e| LqlError::exec("request failed", e))?;
        Self::check_and_parse(endpoint, resp)
    }

    /// POST `{remote_url}{endpoint}` with a JSON body.
    pub(super) fn remote_post_json(
        &self,
        endpoint: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, LqlError> {
        let (url, client, sid) = self.require_remote()?;
        let resp = client
            .post(format!("{url}{endpoint}"))
            .json(body)
            .header(SESSION_HEADER, sid)
            .send()
            .map_err(|e| LqlError::exec("request failed", e))?;
        Self::check_and_parse(endpoint, resp)
    }

    /// DELETE `{remote_url}{endpoint}/{segment}` and hand back the raw
    /// response, so the caller can name a refusal. `segment` is
    /// percent-encoded as one path segment: a name with spaces or slashes
    /// addresses exactly that name.
    pub(super) fn remote_delete_segment(
        &self,
        endpoint: &str,
        segment: &str,
    ) -> Result<reqwest::blocking::Response, LqlError> {
        let (url, client, sid) = self.require_remote()?;
        let mut target = reqwest::Url::parse(&format!("{url}{endpoint}"))
            .map_err(|e| LqlError::exec("invalid remote URL", e))?;
        target
            .path_segments_mut()
            .map_err(|()| LqlError::Execution(format!("remote URL {url} cannot take a path")))?
            .push(segment);
        client
            .delete(target)
            .header(SESSION_HEADER, sid)
            .send()
            .map_err(|e| LqlError::exec("request failed", e))
    }

    pub(super) fn check_and_parse(
        endpoint: &str,
        resp: reqwest::blocking::Response,
    ) -> Result<serde_json::Value, LqlError> {
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(LqlError::Execution(format!(
                "{endpoint} failed ({status}): {text}"
            )));
        }
        resp.json::<serde_json::Value>()
            .map_err(|e| LqlError::exec("invalid response", e))
    }
}

#[cfg(test)]
mod tests;
