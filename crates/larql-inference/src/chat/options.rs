//! The prompt-shape knobs every chat entry point reads from the
//! environment, read in one place so the single-prompt path
//! ([`super::render_user_prompt`]) and the conversation path
//! ([`super::render_conversation`]) cannot disagree on what they mean.
//!
//! Read through the override-aware helpers (NOT raw `std::env::var`) so
//! tests can toggle these per-thread without `std::env::set_var`, which
//! races concurrent `getenv` on the decode path and SIGSEGVs libc.
//! `env_flag` matches `var(_).is_ok()` (set, even empty) and `env_value`
//! matches `var(_).ok()`.

use larql_compute::options::{env_flag, env_value};

use super::default_system_prompt_for_family;

/// Send the prompt verbatim — no template at all.
pub(crate) const ENV_RAW_PROMPT: &str = "LARQL_RAW_PROMPT";
/// Render with `enable_thinking=true` (or `false` when set to `0`/`false`/`no`/`off`).
pub(crate) const ENV_THINKING: &str = "LARQL_THINKING";
/// A system message to put before the conversation.
pub(crate) const ENV_SYSTEM: &str = "LARQL_SYSTEM";
/// Suppress the family-default system message.
pub(crate) const ENV_NO_DEFAULT_SYSTEM: &str = "LARQL_NO_DEFAULT_SYSTEM";

/// Parse an optional boolean environment variable. Unset → `None`;
/// `"0"` / `"false"` / `"no"` / `"off"` (ASCII case-insensitive, trimmed) →
/// `Some(false)`; any other set value (including `"1"`, `"true"`, or empty) →
/// `Some(true)`.
fn env_optional_bool(name: &str) -> Option<bool> {
    let raw = env_value(name)?;
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("0")
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("no")
        || trimmed.eq_ignore_ascii_case("off")
    {
        Some(false)
    } else {
        Some(true)
    }
}

/// What the environment asked of prompt rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PromptOptions {
    pub(crate) raw: bool,
    pub(crate) thinking: Option<bool>,
    system: Option<String>,
    suppress_default_system: bool,
}

impl PromptOptions {
    pub(crate) fn from_env() -> Self {
        Self {
            raw: env_flag(ENV_RAW_PROMPT),
            thinking: env_optional_bool(ENV_THINKING),
            system: env_value(ENV_SYSTEM),
            suppress_default_system: env_flag(ENV_NO_DEFAULT_SYSTEM),
        }
    }

    /// The system message to render: the caller's own, else the
    /// family's registry default unless suppressed.
    pub(crate) fn system_prompt(&self, family: &str) -> Option<String> {
        self.system.clone().or_else(|| {
            (!self.suppress_default_system)
                .then(|| default_system_prompt_for_family(family))
                .flatten()
                .map(str::to_string)
        })
    }
}
