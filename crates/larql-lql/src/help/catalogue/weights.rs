//! Edit weights: stage ops on the stored weights of a VINDEX3 container.

use crate::help::entry::{GroupHelp, Need, StatementHelp};
use crate::help::topic::HelpGroup;
use crate::live_scope::{executed_by, reads_stored};

pub(super) static GROUP: GroupHelp = GroupHelp {
    name: "WEIGHTS",
    aliases: &["WEIGHT", "EDIT WEIGHTS", "WEIGHT EDITS", "STORED WEIGHTS"],
    title: "Edit stored weights (VINDEX3)",
    intro: &[
        "Each statement stages one op on one stored BF16 matrix of the bound container. Staging",
        "checks everything it can and writes nothing. The bound container is never modified.",
        concat!("A staged op takes effect at once in the session: ", executed_by!(), " compute with the"),
        "patched weights, held in memory, and the reply lists each patched tensor with its SHA-256.",
        "COMPILE CURRENT INTO VINDEX writes a container that stores exactly those bytes.",
        concat!("If the session cannot execute an op exactly, ", executed_by!(), " are refused while it is"),
        "pending: they never run the stored weights in its place. HELP UPDATE WEIGHTS; lists when.",
        "One compile takes several ops on several tensors of one canonical representation.",
        "A container compiled with stored-weight ops carries a ledger and cannot be edited again:",
        "stage every op on the unedited base and compile once.",
        "The loop: USE, stage, [BEGIN PATCH ... SAVE PATCH], COMPILE, VERIFY. HELP BATCH; for how",
        "state carries between statements and processes.",
    ],
};

/// How a tensor is named, for all three statements.
const TENSOR_NAME: Need = Need::Other(
    "The tensor's name as the container stores it, such as \"0.self_attn.q_proj.weight\", or <representation>::<tensor> where that bare name is ambiguous: SHOW TENSORS; lists every name.",
);

/// What every staged op has in common about visibility.
const STAGED_EFFECT: &[&str] = &[
    concat!("Staging writes nothing to disk. From the next statement, ", executed_by!(), " in this session"),
    "compute with the patched weights: the reply says so and lists each patched tensor with the",
    "SHA-256 of its bytes, which is the digest that tensor has in a container compiled from the op.",
    concat!(reads_stored!(), " ... CURRENT still read the stored weights."),
    concat!(executed_by!(), " are refused, not run on the stored weights, while an op is pending and:"),
    "  the session's representation source is `transient`;",
    "  a knowledge edit (INSERT, UPDATE EDGES, DELETE) is pending too;",
    "  the tensor is one execution reads past the patched bytes (an expert bank, a shared expert, a",
    "  routed/hybrid MoE FFN layer, or a mixer other than softmax attention / ShortConv) or does not",
    "  project through (dense FFN and ShortConv/softmax projections on hybrid ShortConv models run",
    "  live under `stored` or hybrid `auto`, where patched matrices stay exact BF16);",
    "  or COMPILE would refuse the op before writing.",
    "To keep the edit, COMPILE CURRENT INTO VINDEX \"<out>\". To withdraw it, DISCARD WEIGHT EDITS.",
    "While a BEGIN PATCH recording is open the op is also recorded, so SAVE PATCH can write it.",
];

pub(super) static UPDATE_WEIGHTS: StatementHelp = StatementHelp {
    name: "UPDATE WEIGHTS",
    aliases: &["WEIGHT DELTA"],
    group: HelpGroup::Weights,
    syntax: &[
        "UPDATE WEIGHTS SET {DELTA | VALUE} = <number> WHERE TENSOR = \"<tensor>\" AND ROW = <n> AND {COL = <n> | COL IN (<n>, ...)};",
    ],
    does: &[
        "Stages edits on named elements of one stored BF16 matrix or 3D depthwise-conv1d kernel",
        "([channels, 1, kernel_size], with ROW = channel and COL = tap). With SET DELTA, each",
        "element becomes the bfloat16 nearest to (stored value + delta), ties to even, rounded once.",
        "With SET VALUE, per-element deltas are computed against the stored bits so each element",
        "lands on the exact BF16 target value (e.g. SET VALUE = 0.0 writes exact 0x0000 bits).",
        "Edits apply to the values stored in the bound container, so staging never accumulates.",
    ],
    examples: &[
        "UPDATE WEIGHTS SET DELTA = 0.0625 WHERE TENSOR = \"0.mlp.down_proj.weight\" AND ROW = 5 AND COL IN (12, 13);",
        "UPDATE WEIGHTS SET VALUE = 0.0 WHERE TENSOR = \"23.conv.conv.weight\" AND ROW = 1023 AND COL IN (0, 1, 2, 3, 4, 5, 6);",
    ],
    needs: &[Need::Vindex3, TENSOR_NAME],
    effect: STAGED_EFFECT,
    operands: &[],
    refusals: &[
        "The container already carries a weight-patch ledger (it is a compiled edit): refused at staging.",
        "The tensor is not held by a canonical representation, is not BF16, or is not a matrix.",
        "A row or column outside the tensor.",
        "An element already edited in this session or by an applied patch: edits to one element are not composed.",
        "The tensor is already edited by REMOVE DIRECTION or ADD LOW RANK: one tensor takes one form.",
        "A delta that is not finite, or a result that overflows bfloat16.",
        "More than 262144 listed elements in one compile.",
        "A delta of 0 is NOT refused: it stages, compiles and changes no bits.",
    ],
    related: &[
        "REMOVE DIRECTION",
        "ADD LOW RANK",
        "DISCARD WEIGHT EDITS",
        "BEGIN PATCH",
        "COMPILE",
        "WEIGHTS",
    ],
};

pub(super) static REMOVE_DIRECTION: StatementHelp = StatementHelp {
    name: "REMOVE DIRECTION",
    aliases: &["REMOVE INPUT DIRECTION"],
    group: HelpGroup::Weights,
    syntax: &[
        "REMOVE DIRECTION \"<operand.json | vec.npy>\" FROM \"<tensor>\" {COLUMNS (<n>, <lo>..<hi>, ...) | HEAD <h> | HEADS (...)} SCALE <number> NUMERICS \"<profile>\";",
        "REMOVE INPUT DIRECTION \"<operand.json | vec.npy>\" FROM \"<tensor>\" {ROWS (<n>, <lo>..<hi>, ...) | HEAD <h> | HEADS (...)} SCALE <number> NUMERICS \"<profile>\";",
    ],
    does: &[
        "Stages the removal of a direction from one stored BF16 matrix:",
        "  REMOVE DIRECTION: output-space projection W[:, S] <- W[:, S] - scale * u * (u^T W[:, S])",
        "  REMOVE INPUT DIRECTION: row/input-space projection W[R, :] <- W[R, :] - scale * (W[R, :] v) * v^T",
        "Index lists support half-open ranges (512..640) and inclusive ranges (512..=639).",
        "HEAD <h> and HEADS (...) auto-expand attention head slices from the model's head_dim and",
        "num_q_heads / num_kv_heads (o_proj columns for REMOVE DIRECTION; q/k/v_proj rows for REMOVE INPUT DIRECTION).",
        "COMPILE writes the exact BF16 results the profile defines; no f32 delta is computed anywhere.",
    ],
    examples: &[
        "REMOVE DIRECTION \"u.json\" FROM \"1.mlp.down_proj.weight\" COLUMNS (33, 34) SCALE 0.5 NUMERICS \"f64-seq-rne1/v1\";",
        "REMOVE DIRECTION \"u.npy\" FROM \"1.self_attn.o_proj.weight\" HEAD 4 SCALE 1.0 NUMERICS \"f64-seq-rne2/v1\";",
        "REMOVE INPUT DIRECTION \"v.npy\" FROM \"22.self_attn.k_proj.weight\" HEADS (0, 2) SCALE 1.0 NUMERICS \"f64-seq-rne1/v1\";",
    ],
    needs: &[
        Need::Vindex3,
        TENSOR_NAME,
        Need::Other("The operand file, read when the statement runs. A relative path is relative to the working directory."),
    ],
    effect: STAGED_EFFECT,
    operands: &[
        "Either a C-order `<f4` or `<f8` `.npy` vector (`[rows]` or singleton-prefixed `[1, rows]`",
        "as written by CAPTURE; `<f4` widens losslessly to f64 bits and stamps the file's SHA-256),",
        "or a JSON object: {\"bits\": [\"<hex>\", ...], \"origin\": \"<note>\"}",
        "  bits    One string per row of the tensor: that component of u as the bit pattern of an",
        "          f64, exactly 16 lowercase hex digits. \"3ff0000000000000\" is 1.0. Never decimals.",
        "  origin  Optional. Where the direction came from. The op's digest covers it; nothing",
        "          computes with it.",
        "Any other JSON field is refused.",
    ],
    refusals: &[
        "NUMERICS other than \"f64-seq-rne1/v1\" or \"f64-seq-rne2/v1\". \"f32-seq-rne1/v1\" is refused here.",
        "The operand file is missing, is not this JSON, or holds a string that is not 16 lowercase hex digits.",
        "COLUMNS that are empty, not strictly increasing, or outside the tensor.",
        "A column another structured op already writes: two structured ops do not share a column.",
        "The tensor is already edited by UPDATE WEIGHTS: one tensor takes one form.",
        "An op that changes no element, such as SCALE 0.",
        "SCALE as an integer beyond 2^53 (not exactly an f64): a parse error.",
        "On a container that already carries a ledger the op stages, and COMPILE then refuses.",
    ],
    related: &[
        "ADD LOW RANK",
        "UPDATE WEIGHTS",
        "DISCARD WEIGHT EDITS",
        "COMPILE",
        "VERIFY PATCH",
        "WEIGHTS",
    ],
};

pub(super) static ADD_LOW_RANK: StatementHelp = StatementHelp {
    name: "ADD LOW RANK",
    aliases: &["ADD", "LOW RANK"],
    group: HelpGroup::Weights,
    syntax: &[
        "ADD LOW RANK \"<operands.json | delta.npy>\" TO \"<tensor>\" COLUMNS (<n>, ...) NUMERICS \"<profile>\";",
    ],
    does: &[
        "Stages a low-rank update of the chosen columns J of one stored BF16 matrix:",
        "W[r, J[p]] <- W[r, J[p]] + sum over k of a[r, k] * b[k, p]. Rows are the output axis and",
        "COLUMNS index the input axis. COMPILE writes the exact BF16 results the profile defines.",
        "There is no scale: fold it into a or b.",
    ],
    examples: &[
        "ADD LOW RANK \"a_b.json\" TO \"1.mlp.gate_proj.weight\" COLUMNS (8, 9) NUMERICS \"f64-seq-rne1/v1\";",
        "ADD LOW RANK \"delta.npy\" TO \"1.self_attn.o_proj.weight\" COLUMNS (0, 1, 2, 63) NUMERICS \"f32-seq-rne1/v1\";",
    ],
    needs: &[
        Need::Vindex3,
        TENSOR_NAME,
        Need::Other("The operand file, read when the statement runs. A relative path is relative to the working directory."),
    ],
    effect: STAGED_EFFECT,
    operands: &[
        "Either a direct C-order `<f4`/`<f8` `.npy` file (`[rows]` or `[1, rows]` for a rank-1",
        "update added to every listed column, or `[rows, |COLUMNS|]` for a per-column delta matrix),",
        "or a JSON object: {\"rank\": K, \"a\": [\"<hex>\", ...], \"b\": [\"<hex>\", ...], \"origin\": \"<note>\"}",
        "  (where \"a\" and \"b\" may also be `.npy` file path strings instead of hex arrays).",
        "  rank    K, at most 256.",
        "  a       rows x K values, row-major: a[r, k] is at index r * K + k.",
        "  b       K x (number of COLUMNS) values, row-major: b[k, p] is at index k * |COLUMNS| + p.",
        "          b is compact: it covers only the listed columns, in the order listed.",
        "  origin  Optional. The op's digest covers it; nothing computes with it.",
        "Hex values use lowercase hex of the profile's working width: 16 digits (f64) for",
        "\"f64-seq-rne1/v1\" and \"f64-seq-rne2/v1\", 8 digits (f32) for \"f32-seq-rne1/v1\".",
        "`.npy` `<f4` widens losslessly under f64 profiles; `<f8` is refused under f32 profiles.",
        "Any other JSON field is refused.",
    ],
    refusals: &[
        "NUMERICS that is not one of the three profiles above.",
        "a or b of the wrong length for the tensor's rows, the rank and the number of columns.",
        "A value that is not hex of the profile's width.",
        "COLUMNS that are empty, not strictly increasing, or outside the tensor.",
        "A column another structured op already writes: two structured ops do not share a column.",
        "The tensor is already edited by UPDATE WEIGHTS: one tensor takes one form.",
        "An op whose results all round back to the stored bits.",
        "An op with a zero factor in every term is NOT refused: it is a declared sham and compiles to the base's bytes.",
        "On a container that already carries a ledger the op stages, and COMPILE then refuses.",
    ],
    related: &[
        "REMOVE DIRECTION",
        "UPDATE WEIGHTS",
        "PATCH",
        "DISCARD WEIGHT EDITS",
        "COMPILE",
        "VERIFY LEDGER",
        "WEIGHTS",
    ],
};

pub(super) static PATCH_UNIT: StatementHelp = StatementHelp {
    name: "PATCH",
    aliases: &["PATCH HEAD", "PATCH COLUMN", "PATCH NEURON", "PATCH CHANNEL"],
    group: HelpGroup::Weights,
    syntax: &[
        "PATCH {HEAD <h> | HEADS (<h>, ...)} AT LAYER <l> SCALE <number>;",
        "PATCH {COLUMN <c> | COLUMNS (<c>, ...) | NEURON <c> | NEURONS (<c>, ...)} AT LAYER <l> SCALE <number>;",
        "PATCH {CHANNEL <c> | CHANNELS (<c>, ...)} AT LAYER <l> SCALE <number>;",
    ],
    does: &[
        "Stages an exact BF16 counterfactual scaling/ablation op (W[:, c] <- RNE_bf16(scale * W[:, c]),",
        "with SCALE 0.0 writing exact 0x0000 bits) on one layer's attention head(s) (o_proj), dense",
        "FFN neuron(s)/column(s) (down projection), or ShortConv channel(s) (conv.out_proj).",
        "Auto-resolves the target tensor and head_dim slice directly from the bound container's",
        "executable plan, incrementally re-prepares only that single layer in PreparedCache, and",
        "rolls back in 0ms via DISCARD WEIGHT EDITS;.",
    ],
    examples: &[
        "PATCH HEAD 4 AT LAYER 22 SCALE 0.0;",
        "PATCH NEURON 2886 AT LAYER 23 SCALE 0.5;",
        "PATCH CHANNEL 1023 AT LAYER 23 SCALE 0.0;",
    ],
    needs: &[Need::Vindex3],
    effect: STAGED_EFFECT,
    operands: &[],
    refusals: &[
        "The session is not bound to a canonical VINDEX3 container.",
        "The layer index is out of range or does not carry the requested operator (softmax / dense FFN / ShortConv).",
        "A head, column, or channel index is out of range, duplicated, or already edited by another staged op.",
        "SCALE is non-finite or changes no elements (e.g. SCALE 1.0).",
    ],
    related: &[
        "DLA",
        "DIFF",
        "PROJECT OUT",
        "DISCARD WEIGHT EDITS",
        "REMOVE DIRECTION",
        "WEIGHTS",
    ],
};

pub(super) static PROJECT_OUT: StatementHelp = StatementHelp {
    name: "PROJECT OUT",
    aliases: &["NULLSPACE PROJECTION"],
    group: HelpGroup::Weights,
    syntax: &[
        "PROJECT OUT {\"<tok_a>\" | ID <id_a>} - {\"<tok_b>\" | ID <id_b>} [ON | AT] LAYERS (<n>, <lo>..<hi>, ...) [TARGET {W2 | O_PROJ | CONV_OUT | ALL}] [SCALE <number>];",
    ],
    does: &[
        "Stages an exact sandwich-norm-aware rank-1 nullspace projection removing the logit",
        "difference direction (tok_a - tok_b) from the output projection matrices on the selected",
        "layers: W <- W - scale * u_hat * (u_hat^T W) where u_hat folds final_norm and each",
        "sublayer's post_ffn_norm / post_attention_norm weight (automatically preserving zero-gamma",
        "RMS ballast channels with zero blowup).",
        "Defaults to TARGET W2 and SCALE 1.0. Composes directly over any mounted OVERLAY",
        "SAFETENSORS / OVERLAY LORA and exports cleanly via EXPORT SAFETENSORS.",
    ],
    examples: &[
        "PROJECT OUT \" Paris\" - \" London\" ON LAYERS (20..24);",
        "PROJECT OUT ID 4741 - ID 4595 ON LAYERS (20, 22, 23) TARGET W2 SCALE 1.0;",
    ],
    needs: &[Need::Vindex3, Need::Tokenizer],
    effect: STAGED_EFFECT,
    operands: &[],
    refusals: &[
        "The session is not bound to a canonical VINDEX3 container or carries no tokenizer.json.",
        "Either target encodes to zero or more than one token, or both resolve to the same ID.",
        "No matching target projection exists on the requested layers, or SCALE is non-finite/zero.",
    ],
    related: &[
        "OVERLAY",
        "EXPORT SAFETENSORS",
        "REMOVE DIRECTION",
        "LOGIT DIFF",
        "DISCARD WEIGHT EDITS",
    ],
};
