use super::super::options::PromptOptions;
use super::*;

#[test]
fn nothing_set_reads_as_no_options() {
    let _g = env(&[]);
    let o = PromptOptions::from_env();
    assert!(!o.raw && o.thinking.is_none());
    assert_eq!(o.system_prompt("tinymodel"), None);
}

#[test]
fn explicit_system_wins_over_the_family_default() {
    let _g = env(&[(ENV_SYSTEM, "be brief")]);
    let o = PromptOptions::from_env();
    assert_eq!(o.system_prompt("gemma4").as_deref(), Some("be brief"));
}

#[test]
fn family_default_system_applies_unless_suppressed() {
    let _g = env(&[]);
    assert!(PromptOptions::from_env().system_prompt("gemma4").is_some());
    let _g = env(&[(ENV_NO_DEFAULT_SYSTEM, "1")]);
    assert_eq!(PromptOptions::from_env().system_prompt("gemma4"), None);
}

#[test]
fn raw_and_thinking_flags_read_through() {
    let _g = env(&[(ENV_RAW_PROMPT, "1"), (ENV_THINKING, "1")]);
    let o = PromptOptions::from_env();
    assert!(o.raw && o.thinking == Some(true));
}

#[test]
fn thinking_env_explicit_false_values_disable_thinking() {
    for val in ["0", "false", "FALSE", "no", "off", " 0 "] {
        let _g = env(&[(ENV_THINKING, val)]);
        assert_eq!(
            PromptOptions::from_env().thinking,
            Some(false),
            "value {val:?} should parse as Some(false)"
        );
    }
    for val in ["1", "true", "yes", ""] {
        let _g = env(&[(ENV_THINKING, val)]);
        assert_eq!(
            PromptOptions::from_env().thinking,
            Some(true),
            "value {val:?} should parse as Some(true)"
        );
    }
}
