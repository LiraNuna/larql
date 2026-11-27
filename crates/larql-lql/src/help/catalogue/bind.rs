//! Bind: point the session at a vindex or container.

use crate::help::entry::{GroupHelp, Need, StatementHelp};
use crate::help::topic::HelpGroup;

pub(super) static GROUP: GroupHelp = GroupHelp {
    name: "BIND",
    aliases: &["BINDING", "LIFECYCLE"],
    title: "Bind a model",
    intro: &[
        "Every statement that reads or edits a model acts on what the session is bound to.",
        "A VINDEX3 container is a directory holding index.json and segments/. A VINDEX2 vindex is",
        "the older directory format. USE decides which by the directory's own marker.",
    ],
};

pub(super) static USE: StatementHelp = StatementHelp {
    name: "USE",
    aliases: &["USE MODEL", "USE REMOTE"],
    group: HelpGroup::Bind,
    syntax: &[
        "USE \"<path>\";",
        "USE MODEL \"<id>\" [AUTO_EXTRACT];",
        "USE REMOTE \"<url>\";",
    ],
    does: &[
        "Binds the session to a vindex or container; later statements in the same process act on it.",
        "<path> is a VINDEX2 vindex directory, a VINDEX3 container directory, an hf://owner/repo",
        "published vindex (fetched into the local cache), or a model name the embedding program",
        "resolves (larql lql resolves names already on this machine and never downloads one).",
        "USE MODEL loads a checkpoint's live weights. USE REMOTE forwards statements to a larql server.",
    ],
    examples: &[
        "USE \"model.vindex3\";",
        "USE MODEL \"google/gemma-3-4b-it\";",
        "USE REMOTE \"http://localhost:8080\";",
    ],
    needs: &[Need::Nothing],
    effect: &[
        "At once, until the process ends or another USE runs. The reply names the generation, the",
        "statements the binding serves, and whether a tokenizer is present.",
        "A second USE replaces the binding and drops everything staged: stored-weight ops, applied",
        "patches and an open patch recording. Nothing warns about what was dropped.",
    ],
    operands: &[],
    refusals: &[
        "<path> is not a directory, a published vindex or a known model name.",
        "The PUBLIC_EXPLORER profile refuses USE: the embedding server binds, then tightens.",
    ],
    related: &["STATS", "SHOW MODELS", "EXTRACT", "BATCH"],
};

pub(super) static OVERLAY: StatementHelp = StatementHelp {
    name: "OVERLAY",
    aliases: &["OVERLAY SAFETENSORS", "OVERLAY LORA", "OVERLAY CLEAR"],
    group: HelpGroup::Bind,
    syntax: &[
        "OVERLAY SAFETENSORS \"<ckpt_dir_or_file>\";",
        "OVERLAY LORA \"<adapter_dir>\" [SCALE <number>];",
        "OVERLAY CLEAR;",
    ],
    does: &[
        "Mounts modified BF16 weights from a Hugging Face safetensors checkpoint/delta file or a",
        "PEFT LoRA adapter directly in RAM over the bound read-only VINDEX3 container, avoiding a",
        "full multi-GB container rebuild.",
        "OVERLAY SAFETENSORS scans the candidate safetensors and installs replacements ONLY for",
        "tensors whose BF16 bytes differ from the bound container; untouched layers stay hot in",
        "the session's PreparedCache.",
        "OVERLAY LORA reads adapter_config.json and adapter_model.safetensors, merges",
        "scale * (alpha / r) * (B @ A) onto the bound container's BF16 matrices in parallel, and",
        "mounts only the modified matrices.",
        "OVERLAY CLEAR drops the mounted overlay and restores the bound base container in 0ms.",
    ],
    examples: &[
        "OVERLAY SAFETENSORS \"candidate_ckpt\";",
        "OVERLAY LORA \"lora_adapter\" SCALE 0.5;",
        "OVERLAY CLEAR;",
    ],
    needs: &[Need::Vindex3],
    effect: &[
        "In-memory only: nothing on disk is modified. INFER, GENERATE, CAPTURE, DLA, TRACE, and",
        "EXPORT SAFETENSORS immediately execute the overlaid weights.",
    ],
    operands: &[],
    refusals: &[
        "The session is not bound to a canonical VINDEX3 container.",
        "The safetensors path or adapter directory does not exist or has mismatched tensor shapes.",
        "The PUBLIC_EXPLORER profile refuses OVERLAY.",
    ],
    related: &["USE", "DLA", "EXPORT SAFETENSORS", "DISCARD WEIGHT EDITS"],
};

pub(super) static LOAD_SOFT_PROMPT: StatementHelp = StatementHelp {
    name: "LOAD SOFT_PROMPT",
    aliases: &["SOFT_PROMPT"],
    group: HelpGroup::Bind,
    syntax: &["LOAD SOFT_PROMPT \"<path.pt | path.npy | path.safetensors>\";"],
    does: &[
        "Loads a continuous soft-prompt embedding prefix ([n_soft, hidden] or [1, n_soft, hidden])",
        "from a PyTorch .pt checkpoint, NumPy .npy array, or safetensors file into the bound",
        "VINDEX3 session. Subsequent INFER, GENERATE, DLA, and CAPTURE calls prepend these",
        "continuous embeddings directly before token 0 in the model's traversal.",
        "The soft-prompt KV / ShortConv continuation prefix is prefilled once and cached across",
        "all prompts in the session.",
    ],
    examples: &[
        "LOAD SOFT_PROMPT \"sp16_step160.pt\";",
        "LOAD SOFT_PROMPT \"prefix.npy\";",
    ],
    needs: &[Need::Vindex3],
    effect: &[
        "In-memory only: nothing on disk is modified. Weights stay hot in PreparedCache.",
        "Use CLEAR SOFT_PROMPT; to restore unprefixed prompt execution in 0ms.",
    ],
    operands: &[],
    refusals: &[
        "The session is not bound to a canonical VINDEX3 container.",
        "The file is missing, uses an unsupported format, has non-finite values, or does not match hidden_size.",
        "The PUBLIC_EXPLORER profile refuses LOAD SOFT_PROMPT.",
    ],
    related: &["CLEAR SOFT_PROMPT", "USE", "DLA", "INFER", "CAPTURE"],
};

pub(super) static CLEAR_SOFT_PROMPT: StatementHelp = StatementHelp {
    name: "CLEAR SOFT_PROMPT",
    aliases: &[],
    group: HelpGroup::Bind,
    syntax: &["CLEAR SOFT_PROMPT;"],
    does: &[
        "Clears any active continuous soft-prompt prefix from the bound VINDEX3 session and",
        "restores normal unprefixed prompt execution while keeping prepared weights hot in RAM.",
    ],
    examples: &["CLEAR SOFT_PROMPT;"],
    needs: &[Need::Vindex3],
    effect: &["In-memory only. Clears the session's prefix KV cache and restores normal prompts."],
    operands: &[],
    refusals: &[
        "The session is not bound to a canonical VINDEX3 container.",
        "The PUBLIC_EXPLORER profile refuses CLEAR SOFT_PROMPT.",
    ],
    related: &["LOAD SOFT_PROMPT", "USE", "DLA"],
};

pub(super) static EXTRACT: StatementHelp = StatementHelp {
    name: "EXTRACT",
    aliases: &["EXTRACT MODEL"],
    group: HelpGroup::Bind,
    syntax: &[
        "EXTRACT MODEL \"<id>\" INTO \"<out>\" [COMPONENTS <c>, ...] [LAYERS <a>-<b>] [WITH {INFERENCE | ALL}] [FORMAT {VINDEX2 | VINDEX3}];",
    ],
    does: &[
        "Builds a new vindex at <out> from a model checkpoint (a Hugging Face id or a local",
        "directory), then binds the session to it. FORMAT picks the generation; without it the",
        "vindex crate's extraction-generation policy decides.",
        "A VINDEX3 extraction encodes the checkpoint's own bytes into a container.",
        "WITH shapes a VINDEX2 extraction only: none serves browsing, WITH INFERENCE adds what INFER",
        "needs, WITH ALL adds what COMPILE needs.",
        "COMPONENTS and LAYERS are parsed and ignored.",
    ],
    examples: &[
        "EXTRACT MODEL \"Qwen/Qwen3-0.6B\" INTO \"qwen3-0.6b.vindex3\";",
        "EXTRACT MODEL \"google/gemma-3-4b-it\" INTO \"gemma3-4b.vindex\" WITH ALL FORMAT VINDEX2;",
    ],
    needs: &[
        Need::Nothing,
        Need::Other("The model's checkpoint: a local directory, or a Hugging Face id."),
    ],
    effect: &[
        "Writes <out> and binds it. A failed extraction can leave a partial directory at <out>,",
        "which a re-run refuses as not empty.",
    ],
    operands: &[],
    refusals: &[
        "<out> exists and is not an empty directory. Nothing is ever replaced.",
        "A GGUF file as the source of a VINDEX3 extraction: use FORMAT VINDEX2.",
        "In a batch, after any earlier statement failed (HELP BATCH;).",
    ],
    related: &["USE", "COMPILE", "BATCH"],
};
