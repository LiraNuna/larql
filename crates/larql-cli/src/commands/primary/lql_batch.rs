//! Structured JSONL batch probe runner for `larql lql --prompts-jsonl <file>`.
//!
//! Executes an optional setup LQL script (e.g. `USE "base.vindex3"; OVERLAY SAFETENSORS "cand";`
//! or `LOAD SOFT_PROMPT "sp16.pt";`) and then streams raw prompts directly into typed `Session`
//! methods (`dla`, `diff_dla`, `predict`, `generate`) with zero shell or LQL string escaping.

use std::path::Path;

use larql_lql::ast::OverlayKind;
use larql_lql::operations::{CaptureLayers, DlaTarget};
use larql_lql::{run_batch_json_in, Session, EXIT_ERROR, EXIT_OK};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Debug, Deserialize)]
struct ProbeRecord {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    op: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    scale: Option<f64>,
    #[serde(default)]
    statement: Option<String>,
    #[serde(default, rename = "for", alias = "target_a")]
    for_target: Option<Value>,
    #[serde(default, rename = "vs", alias = "target_b")]
    vs_target: Option<Value>,
    #[serde(default)]
    layers: Option<Value>,
    #[serde(default, alias = "top_neurons")]
    top_columns: Option<u32>,
    #[serde(default)]
    top_heads: Option<u32>,
    #[serde(default)]
    top_channels: Option<u32>,
    #[serde(default)]
    top: Option<u32>,
    #[serde(default)]
    max_tokens: Option<u32>,
}

pub fn run_prompts_jsonl(
    session: &mut Session,
    setup_statement: Option<&str>,
    prompts_path: &Path,
) -> i32 {
    let mut setup_results = Vec::new();
    if let Some(stmt) = setup_statement.map(str::trim).filter(|s| !s.is_empty()) {
        let setup = run_batch_json_in(session, stmt);
        setup_results = setup.results;
        if setup.errors > 0 || setup.contract_violations > 0 {
            let doc = json!({
                "ok": false,
                "errors": setup.errors,
                "contract_violations": setup.contract_violations,
                "setup": setup_results,
                "probes": [],
            });
            println!("{}", serde_json::to_string_pretty(&doc).unwrap());
            return if setup.errors > 0 {
                EXIT_ERROR
            } else {
                larql_lql::EXIT_CONTRACT_VIOLATED
            };
        }
    }

    let text = match std::fs::read_to_string(prompts_path) {
        Ok(t) => t,
        Err(e) => {
            let doc = json!({
                "ok": false,
                "errors": 1,
                "error": format!("read --prompts-jsonl {}: {e}", prompts_path.display()),
                "setup": setup_results,
                "probes": [],
            });
            println!("{}", serde_json::to_string_pretty(&doc).unwrap());
            return EXIT_ERROR;
        }
    };

    let mut probes = Vec::new();
    let mut errors = 0usize;

    for (line_idx, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let rec: ProbeRecord = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                errors += 1;
                probes.push(json!({
                    "id": format!("line_{}", line_idx + 1),
                    "ok": false,
                    "error": format!("invalid JSONL record at line {}: {e}", line_idx + 1),
                }));
                continue;
            }
        };
        let id = rec
            .id
            .clone()
            .unwrap_or_else(|| format!("probe_{}", probes.len()));
        let op_name = rec
            .op
            .as_deref()
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_else(|| {
                if rec.for_target.is_some() || rec.vs_target.is_some() {
                    "dla".to_string()
                } else {
                    "infer".to_string()
                }
            });

        match exec_probe(session, &op_name, &rec) {
            Ok(data) => probes.push(json!({
                "id": id,
                "op": op_name,
                "ok": true,
                "data": data,
            })),
            Err(err) => {
                errors += 1;
                probes.push(json!({
                    "id": id,
                    "op": op_name,
                    "ok": false,
                    "error": err,
                }));
            }
        }
    }

    let doc = json!({
        "ok": errors == 0,
        "errors": errors,
        "setup": setup_results,
        "probes": probes,
    });
    println!("{}", serde_json::to_string_pretty(&doc).unwrap());
    if errors > 0 {
        EXIT_ERROR
    } else {
        EXIT_OK
    }
}

fn exec_probe(session: &mut Session, op: &str, rec: &ProbeRecord) -> Result<Value, String> {
    match op {
        "overlay_safetensors" => {
            let path = rec
                .path
                .as_deref()
                .ok_or_else(|| "overlay_safetensors requires `path`".to_string())?;
            let summary = session
                .apply_external_overlay(&OverlayKind::Safetensors {
                    path: path.to_string(),
                })
                .map_err(|e| e.to_string())?;
            serde_json::to_value(&summary).map_err(|e| e.to_string())
        }
        "overlay_lora" => {
            let path = rec
                .path
                .as_deref()
                .ok_or_else(|| "overlay_lora requires `path`".to_string())?;
            let summary = session
                .apply_external_overlay(&OverlayKind::Lora {
                    path: path.to_string(),
                    scale: rec.scale,
                })
                .map_err(|e| e.to_string())?;
            serde_json::to_value(&summary).map_err(|e| e.to_string())
        }
        "overlay_clear" => {
            session
                .apply_external_overlay(&OverlayKind::Clear)
                .map_err(|e| e.to_string())?;
            Ok(json!({ "cleared": true }))
        }
        "load_soft_prompt" => {
            let path = rec
                .path
                .as_deref()
                .ok_or_else(|| "load_soft_prompt requires `path`".to_string())?;
            let summary = session.load_soft_prompt(path).map_err(|e| e.to_string())?;
            serde_json::to_value(&summary).map_err(|e| e.to_string())
        }
        "clear_soft_prompt" => {
            let cleared = session.clear_soft_prompt().map_err(|e| e.to_string())?;
            Ok(json!({ "cleared": cleared }))
        }
        "discard_weight_edits" => {
            let stmt = larql_lql::parse("DISCARD WEIGHT EDITS;").map_err(|e| e.to_string())?;
            let lines = session.execute(&stmt).map_err(|e| e.to_string())?;
            Ok(json!({ "lines": lines }))
        }
        "lql" => {
            let stmt_str = rec
                .statement
                .as_deref()
                .ok_or_else(|| "`lql` op requires `statement`".to_string())?;
            let out = run_batch_json_in(session, stmt_str);
            if out.errors > 0 {
                return Err(format!("inline LQL failed: {:?}", out.results));
            }
            Ok(json!({ "results": out.results }))
        }
        "logit_diff" | "diff_logit_diff" => {
            let prompt = rec
                .prompt
                .as_deref()
                .ok_or_else(|| format!("`{op}` requires `prompt`"))?;
            let ta = parse_target(rec.for_target.as_ref(), "for")?;
            let tb = parse_target(rec.vs_target.as_ref(), "vs")?;
            if op == "diff_logit_diff" {
                let report = session
                    .diff_logit_diff(prompt, ta, tb)
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(&report).map_err(|e| e.to_string())
            } else {
                let report = session
                    .logit_diff(prompt, ta, tb)
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(&report).map_err(|e| e.to_string())
            }
        }
        "dla" | "diff_dla" => {
            let prompt = rec
                .prompt
                .as_deref()
                .ok_or_else(|| format!("`{op}` requires `prompt`"))?;
            let ta = parse_target(rec.for_target.as_ref(), "for")?;
            let tb = parse_target(rec.vs_target.as_ref(), "vs")?;
            let layers = parse_layers(rec.layers.as_ref())?;
            if op == "diff_dla" {
                let report = session
                    .diff_dla(
                        prompt,
                        ta,
                        tb,
                        layers.as_ref(),
                        rec.top_columns,
                        rec.top_heads,
                        rec.top_channels,
                    )
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(&report).map_err(|e| e.to_string())
            } else {
                let report = session
                    .dla(
                        prompt,
                        ta,
                        tb,
                        layers.as_ref(),
                        rec.top_columns,
                        rec.top_heads,
                        rec.top_channels,
                    )
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(&report).map_err(|e| e.to_string())
            }
        }
        "infer" | "predict" => {
            let prompt = rec
                .prompt
                .as_deref()
                .ok_or_else(|| format!("`{op}` requires `prompt`"))?;
            let pred = session
                .predict(prompt, rec.top.unwrap_or(5) as usize)
                .map_err(|e| e.to_string())?;
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
            Ok(json!({
                "mode": "predict",
                "prompt_ids": pred.prompt_ids,
                "top": top_rows,
                "elapsed_ms": pred.elapsed_ms,
            }))
        }
        "generate" => {
            let prompt = rec
                .prompt
                .as_deref()
                .ok_or_else(|| format!("`{op}` requires `prompt`"))?;
            let gen = session
                .generate(prompt, rec.max_tokens.unwrap_or(32) as usize)
                .map_err(|e| e.to_string())?;
            serde_json::to_value(&gen).map_err(|e| e.to_string())
        }
        other => Err(format!(
            "unknown probe op `{other}` (expected logit_diff, diff_logit_diff, dla, diff_dla, infer, generate, overlay_safetensors, overlay_lora, overlay_clear, load_soft_prompt, clear_soft_prompt, discard_weight_edits, or lql)"
        )),
    }
}

fn parse_target(val: Option<&Value>, field: &str) -> Result<DlaTarget, String> {
    let v = val.ok_or_else(|| format!("missing required `{field}` target field"))?;
    parse_target_value(v, field, true)
}

fn parse_target_value(v: &Value, field: &str, allow_max_other: bool) -> Result<DlaTarget, String> {
    if let Some(id) = v.as_u64() {
        let id = u32::try_from(id).map_err(|_| format!("`{field}` token id {id} overflows u32"))?;
        return Ok(DlaTarget::Id(id));
    }
    if let Some(s) = v.as_str() {
        if allow_max_other && s.eq_ignore_ascii_case("MAX_OTHER") {
            return Ok(DlaTarget::MaxOther { except: Vec::new() });
        }
        return Ok(DlaTarget::Text(s.to_string()));
    }
    if allow_max_other {
        if let Some(obj) = v.as_object() {
            if let Some(mo) = obj.get("max_other") {
                let items: &[Value] = if let Some(arr) = mo.as_array() {
                    arr.as_slice()
                } else if let Some(inner) = mo.as_object().and_then(|o| o.get("except")) {
                    inner
                        .as_array()
                        .ok_or_else(|| format!("`{field}.max_other.except` must be an array"))?
                        .as_slice()
                } else if mo.is_null() || mo.as_bool() == Some(true) {
                    &[]
                } else {
                    return Err(format!(
                        "`{field}.max_other` must be true, an array, or {{\"except\": [...]}}"
                    ));
                };
                let mut except = Vec::with_capacity(items.len());
                for item in items {
                    except.push(parse_target_value(item, field, false)?);
                }
                return Ok(DlaTarget::MaxOther { except });
            }
        }
    }
    Err(format!(
        "`{field}` must be a token string, unsigned integer token ID, or MAX_OTHER spec, got {v}"
    ))
}

fn parse_layers(val: Option<&Value>) -> Result<Option<CaptureLayers>, String> {
    let Some(v) = val else {
        return Ok(None);
    };
    if v.is_null() {
        return Ok(None);
    }
    if let Some(n) = v.as_u64() {
        return Ok(Some(CaptureLayers::One(n as usize)));
    }
    if let Some(arr) = v.as_array() {
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            let n = item
                .as_u64()
                .ok_or_else(|| format!("layers array element must be integer, got {item}"))?;
            out.push(n as usize);
        }
        return Ok(Some(CaptureLayers::Many(out)));
    }
    if let Some(s) = v.as_str() {
        let t = s.trim();
        if t.eq_ignore_ascii_case("all") || t.is_empty() {
            return Ok(None);
        }
        if let Some((a, b)) = t.split_once("..=") {
            let lo: usize = a
                .trim()
                .parse()
                .map_err(|_| format!("invalid layers `{s}`"))?;
            let hi: usize = b
                .trim()
                .parse()
                .map_err(|_| format!("invalid layers `{s}`"))?;
            if hi < lo {
                return Err(format!("layers `{s}`: end before start"));
            }
            return Ok(Some(CaptureLayers::Many((lo..=hi).collect())));
        }
        if let Some((a, b)) = t.split_once("..") {
            let lo: usize = a
                .trim()
                .parse()
                .map_err(|_| format!("invalid layers `{s}`"))?;
            let hi: usize = b
                .trim()
                .parse()
                .map_err(|_| format!("invalid layers `{s}`"))?;
            if hi <= lo {
                return Err(format!("layers `{s}`: empty range"));
            }
            return Ok(Some(CaptureLayers::Many((lo..hi).collect())));
        }
        if let Some((a, b)) = t.split_once('-') {
            let lo: usize = a
                .trim()
                .parse()
                .map_err(|_| format!("invalid layers `{s}`"))?;
            let hi: usize = b
                .trim()
                .parse()
                .map_err(|_| format!("invalid layers `{s}`"))?;
            if hi < lo {
                return Err(format!("layers `{s}`: end before start"));
            }
            return Ok(Some(CaptureLayers::Many((lo..=hi).collect())));
        }
        let one: usize = t.parse().map_err(|_| format!("invalid layers `{s}`"))?;
        return Ok(Some(CaptureLayers::One(one)));
    }
    Err(format!("invalid `layers` specification: {v}"))
}
