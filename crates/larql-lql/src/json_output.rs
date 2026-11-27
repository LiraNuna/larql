//! Structured JSON batch execution for `larql lql --format json`.
//!
//! Shares the exact statement splitter, `--` comment stripping, write-after-failure guard,
//! capability-profile gating, and exit-status contract of [`crate::repl::run_batch_outcome_in`],
//! emitting typed JSON payloads for `DLA`, `INFER` (VINDEX3), and `EXPORT SAFETENSORS` and
//! `{"ok": true, "statement": ..., "lines": [...]}` for all other statements.

use serde_json::{json, Value};

use crate::ast::Statement;
use crate::executor::Session;
use crate::parser;
use crate::repl::{split_statements, EXIT_CONTRACT_VIOLATED, EXIT_ERROR, EXIT_OK};

/// Outcome of running a semicolon-separated LQL batch in structured JSON mode.
#[derive(Debug, Clone)]
pub struct BatchJsonOutcome {
    /// One JSON object per non-empty statement, in execution order.
    pub results: Vec<Value>,
    /// Number of statements that failed (parse error, refusal, execution error, write guard).
    pub errors: usize,
    /// Number of `VERIFY` statements in the session that reported a contract violation.
    pub contract_violations: usize,
}

impl BatchJsonOutcome {
    /// Exit status for `larql lql --format json`: identical contract to [`crate::BatchOutcome`].
    pub fn exit_code(&self) -> i32 {
        if self.errors > 0 {
            EXIT_ERROR
        } else if self.contract_violations > 0 {
            EXIT_CONTRACT_VIOLATED
        } else {
            EXIT_OK
        }
    }

    /// Top-level JSON document emitted by `larql lql --format json`.
    pub fn to_json_value(&self) -> Value {
        json!({
            "ok": self.errors == 0 && self.contract_violations == 0,
            "errors": self.errors,
            "contract_violations": self.contract_violations,
            "results": self.results,
        })
    }
}

/// Run a semicolon-separated LQL batch in `session` and return structured JSON results.
pub fn run_batch_json_in(session: &mut Session, input: &str) -> BatchJsonOutcome {
    let mut outcome = BatchJsonOutcome {
        results: Vec::new(),
        errors: 0,
        contract_violations: 0,
    };

    for stmt_text in split_statements(input) {
        let trimmed: String = stmt_text
            .lines()
            .filter(|l| !l.trim().starts_with("--"))
            .collect::<Vec<_>>()
            .join("\n");
        let trimmed = trimmed.trim();
        if trimmed.is_empty() {
            continue;
        }
        match parser::parse(trimmed) {
            Ok(stmt) if outcome.errors > 0 && stmt.writes_artifact() => {
                let err = format!(
                    "refused `{}`: {} earlier statement(s) in this batch failed, \
                     and a batch writes nothing after a failure",
                    stmt.verb(),
                    outcome.errors
                );
                outcome.results.push(json!({
                    "ok": false,
                    "statement": stmt.verb(),
                    "error": err,
                }));
                outcome.errors += 1;
            }
            Ok(stmt) => match execute_json_statement(session, &stmt) {
                Ok(entry) => outcome.results.push(entry),
                Err(e) => {
                    outcome.results.push(json!({
                        "ok": false,
                        "statement": stmt.verb(),
                        "error": e.to_string(),
                    }));
                    outcome.errors += 1;
                }
            },
            Err(e) => {
                outcome.results.push(json!({
                    "ok": false,
                    "error": e.to_string(),
                }));
                outcome.errors += 1;
            }
        }
    }

    outcome.contract_violations = session.contract_violations();
    outcome
}

fn execute_json_statement(
    session: &mut Session,
    stmt: &Statement,
) -> Result<Value, crate::LqlError> {
    // Enforce capability profile identically to `Session::execute`.
    session.profile().check(stmt)?;

    if !session.is_remote() && session.is_vindex3() {
        match stmt {
            Statement::Dla {
                prompt,
                target_a,
                target_b,
                layers,
                top_columns,
                top_heads,
                top_channels,
            } => {
                let report = session.dla(
                    prompt,
                    target_a,
                    target_b,
                    layers.as_ref(),
                    *top_columns,
                    *top_heads,
                    *top_channels,
                )?;
                return Ok(json!({
                    "ok": true,
                    "statement": "DLA",
                    "data": report,
                }));
            }
            Statement::Infer {
                prompt,
                top,
                generate,
                route: None,
                ..
            } => match generate {
                None => {
                    let pred = session.predict(prompt, top.unwrap_or(5) as usize)?;
                    let top_rows: Vec<Value> = pred
                        .top
                        .iter()
                        .map(|p| {
                            json!({
                                "token_id": p.token_id,
                                "token": p.token,
                                "probability": p.probability,
                            })
                        })
                        .collect();
                    let knn = pred.knn_override.as_ref().map(|k| {
                        json!({
                            "token": k.token,
                            "cosine": k.cosine,
                            "layer": k.layer,
                        })
                    });
                    return Ok(json!({
                        "ok": true,
                        "statement": "INFER",
                        "data": {
                            "mode": "predict",
                            "prompt_ids": pred.prompt_ids,
                            "top": top_rows,
                            "knn_override": knn,
                            "elapsed_ms": pred.elapsed_ms,
                        },
                    }));
                }
                Some(n) => {
                    let gen = session.generate(prompt, *n as usize)?;
                    return Ok(json!({
                        "ok": true,
                        "statement": "INFER",
                        "data": {
                            "mode": "generate",
                            "prompt_ids": gen.prompt_ids,
                            "prompt_len": gen.prompt_len,
                            "token_ids": gen.token_ids,
                            "text": gen.text,
                            "elapsed_ms": gen.elapsed_ms,
                        },
                    }));
                }
            },
            Statement::ExportSafetensors {
                output,
                base_checkpoint,
            } => {
                let report = session.export_safetensors(output, base_checkpoint.as_deref())?;
                return Ok(json!({
                    "ok": true,
                    "statement": "EXPORT SAFETENSORS",
                    "data": report,
                }));
            }
            Statement::Overlay { kind } => {
                let summary = session.apply_external_overlay(kind)?;
                return Ok(json!({
                    "ok": true,
                    "statement": "OVERLAY",
                    "data": summary,
                }));
            }
            Statement::LoadSoftPrompt { path } => {
                let summary = session.load_soft_prompt(path)?;
                return Ok(json!({
                    "ok": true,
                    "statement": "LOAD SOFT_PROMPT",
                    "data": summary,
                }));
            }
            Statement::ClearSoftPrompt => {
                let cleared = session.clear_soft_prompt()?;
                return Ok(json!({
                    "ok": true,
                    "statement": "CLEAR SOFT_PROMPT",
                    "data": { "cleared": cleared },
                }));
            }
            _ => {}
        }
    }

    let lines = session.execute(stmt)?;
    Ok(json!({
        "ok": true,
        "statement": stmt.verb(),
        "lines": lines,
    }))
}
