//! The `.vlp` form of a **structured** weight edit: `remove_direction`, declared by its operands, not by
//! the elements it changes.
//!
//! A [`super::WeightDeltaOp`] lists elements and f32 deltas. That cannot always carry what the
//! structured numerics compute: the BF16 result of `w - (alpha * u) * c` can be an element a single f32
//! delta added to the stored value never rounds to (a near-cancellation leaves a tiny nonzero where the
//! delta would have to name a value below the f32 spacing of the old one). This op declares the
//! operation instead:
//!
//! - the **operand** `u`, as stored bit patterns, and `alpha` as bits;
//! - the **columns** it is applied to and the **numerical profile** that fixes every rounding;
//! - the same **source binding** as a delta op: the exact container and weights it was written for.
//!
//! [`StructuredWeightOp::resolve`] runs the frozen interpreter ([`super::structured::run`]) over the
//! bound tensor and returns the exact BF16 results of the elements that change, ready for
//! [`super::apply_declared_patches`]. The compiled container then records those results, not deltas.
//!
//! Like the delta op, the digest detects accidental damage and casual edits and is not a signature.
//! Whether a container implements the declared op is for an independent verifier, which recomputes the
//! op by a second route and compares every element.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::apply::{read_bf16_tensor, tensor_object_kind, ResultEdit, ResultPatch};
use super::error::WeightPatchError;
use super::op::{bind_source, SourceBinding};
use super::structured::{
    run, BaseTensor, OpKind, OperandBits, Profile, ScalarBits, StructuredOp, TensorRole,
};
use crate::format::vindex3::graph::ObjectKind;

/// Version tag mixed into the digest, so a change to what it covers is a new digest.
const DIGEST_DOMAIN: &str = "structured_op/remove_direction/v1";
/// Hex digits of one stored operand element: eight for f32 bits, sixteen for f64.
const F32_HEX_DIGITS: usize = 8;
const F64_HEX_DIGITS: usize = 16;

fn is_false(b: &bool) -> bool {
    !*b
}

/// A `remove_direction` (or row-space `remove_input_direction`) op as it appears in a `.vlp` patch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuredWeightOp {
    pub source: SourceBinding,
    pub tensor: String,
    /// The numerical profile id, such as `f64-seq-rne1/v1`.
    pub numerics: String,
    /// `u` (output-axis, `len == rows`) or `v` (input-axis, `len == cols`), one fixed-width
    /// lowercase hex string per element: the stored bits, never a decimal.
    pub operand: Vec<String>,
    /// `alpha`, as stored bits in the profile's working width.
    pub alpha: String,
    /// Strictly increasing column indices (when `input_axis == false`) or row indices
    /// (when `input_axis == true`).
    pub columns: Vec<usize>,
    /// When `true`, removes input-space direction `v` from rows listed in `columns`:
    /// `W[R, :] <- W[R, :] - alpha * (W[R, :] v) v^T`. Defaults to `false` so existing v1
    /// `.vlp` files and digests are bit-for-bit unchanged.
    #[serde(default, skip_serializing_if = "is_false")]
    pub input_axis: bool,
    /// When `true`, scales columns listed in `columns` directly by `alpha` (`W[:, c] <- alpha * W[:, c]`,
    /// with `operand` empty). Used by counterfactual `PATCH HEAD / COLUMN / CHANNEL ... SCALE <s>`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub scale_columns_only: bool,
    /// Where the operand came from, if its writer says: a feature artifact id, a file hash, a note. The
    /// operand's bits are what the op computes with; this only makes its provenance separately identifiable
    /// rather than lost inside the op. It is covered by the digest and read by nothing else.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    /// SHA-256 over every other field; see the module note.
    pub digest: String,
}

impl StructuredWeightOp {
    /// Build an output-axis `remove_direction` op bound to `container` and seal it.
    pub fn bind(
        container: &Path,
        tensor: &str,
        numerics: &str,
        operand: &OperandBits,
        alpha: ScalarBits,
        columns: Vec<usize>,
    ) -> Result<Self, WeightPatchError> {
        Self::bind_axis(container, tensor, numerics, operand, alpha, columns, false)
    }

    /// Build a row-space (`input_axis == true`) `remove_input_direction` op bound to `container`
    /// and seal it: `W[R, :] <- W[R, :] - alpha * (W[R, :] v) v^T` over `rows`.
    pub fn bind_input(
        container: &Path,
        tensor: &str,
        numerics: &str,
        operand: &OperandBits,
        alpha: ScalarBits,
        rows: Vec<usize>,
    ) -> Result<Self, WeightPatchError> {
        Self::bind_axis(container, tensor, numerics, operand, alpha, rows, true)
    }

    /// Build a column-scaling op (`W[:, c] <- alpha * W[:, c]`) bound to `container` and seal it.
    pub fn bind_scale_columns(
        container: &Path,
        tensor: &str,
        numerics: &str,
        alpha: ScalarBits,
        columns: Vec<usize>,
    ) -> Result<Self, WeightPatchError> {
        let source = bind_source(container, tensor)?;
        let alpha = match alpha {
            ScalarBits::F64(b) => format!("{b:016x}"),
            ScalarBits::F32(b) => format!("{b:08x}"),
        };
        let mut op = Self {
            source,
            tensor: super::TensorName::parse(tensor).tensor.to_string(),
            numerics: numerics.to_string(),
            operand: Vec::new(),
            alpha,
            columns,
            input_axis: false,
            scale_columns_only: true,
            origin: None,
            digest: String::new(),
        };
        op.digest = op.compute_digest();
        Ok(op)
    }

    fn bind_axis(
        container: &Path,
        tensor: &str,
        numerics: &str,
        operand: &OperandBits,
        alpha: ScalarBits,
        indices: Vec<usize>,
        input_axis: bool,
    ) -> Result<Self, WeightPatchError> {
        let source = bind_source(container, tensor)?;
        let hex = |bits: &OperandBits| -> Vec<String> {
            match bits {
                OperandBits::F64(v) => v.iter().map(|b| format!("{b:016x}")).collect(),
                OperandBits::F32(v) => v.iter().map(|b| format!("{b:08x}")).collect(),
            }
        };
        let alpha = match alpha {
            ScalarBits::F64(b) => format!("{b:016x}"),
            ScalarBits::F32(b) => format!("{b:08x}"),
        };
        let mut op = Self {
            source,
            tensor: super::TensorName::parse(tensor).tensor.to_string(),
            numerics: numerics.to_string(),
            operand: hex(operand),
            alpha,
            columns: indices,
            input_axis,
            scale_columns_only: false,
            origin: None,
            digest: String::new(),
        };
        op.digest = op.compute_digest();
        Ok(op)
    }

    /// This op with its operand's origin recorded, re-sealed.
    pub fn with_origin(mut self, origin: impl Into<String>) -> Self {
        self.origin = Some(origin.into());
        self.digest = self.compute_digest();
        self
    }

    /// The digest of this op's fields, independent of the stored `digest`.
    pub fn compute_digest(&self) -> String {
        let columns: Vec<String> = self.columns.iter().map(|c| c.to_string()).collect();
        let axis_line = if self.input_axis { "axis=input\n" } else { "" };
        let mode_line = if self.scale_columns_only {
            "mode=scale_columns\n"
        } else {
            ""
        };
        let text = format!(
            "{DIGEST_DOMAIN}\n{axis_line}{mode_line}model={}\nindex_sha256={}\nrepresentation={}\npayload_sha256={}\n\
             tensor={}\nnumerics={}\nalpha={}\ncolumns={}\noperand={}\norigin={}\n",
            self.source.model,
            self.source.index_sha256,
            self.source.representation,
            self.source.payload_sha256,
            self.tensor,
            self.numerics,
            self.alpha,
            columns.join(","),
            self.operand.join(","),
            self.origin
                .as_deref()
                .map_or("-".to_string(), |o| format!("+{o}")),
        );
        super::apply::hex(&Sha256::digest(text.as_bytes()))
    }

    /// Whether the op is intact, names a profile this build implements, and carries operands of that
    /// profile's width. Does not look at any container.
    pub fn check_intact(&self) -> Result<(), WeightPatchError> {
        if self.compute_digest() != self.digest {
            return Err(WeightPatchError::PatchOp(
                "its digest does not match its contents: the patch was corrupted or edited"
                    .to_string(),
            ));
        }
        self.operands().map(|_| ())
    }

    /// Refuse unless `container` is the exact artifact this op was written against.
    pub fn check_binding(&self, container: &Path) -> Result<(), WeightPatchError> {
        super::op::check_same_source(&self.source, container, &self.tensor)
    }

    /// The op as the interpreter takes it, with its operands decoded.
    pub(crate) fn op_kind(&self) -> Result<OpKind, WeightPatchError> {
        let (operand, alpha) = self.operands()?;
        if self.scale_columns_only {
            if !self.operand.is_empty() {
                return Err(WeightPatchError::PatchOp(
                    "scale_columns op must not carry a vector operand".to_string(),
                ));
            }
            Ok(OpKind::ScaleColumns {
                alpha,
                columns: self.columns.clone(),
            })
        } else if self.input_axis {
            Ok(OpKind::RemoveInputDirection {
                v: operand,
                alpha,
                rows: self.columns.clone(),
            })
        } else {
            Ok(OpKind::RemoveDirection {
                u: operand,
                alpha,
                columns: self.columns.clone(),
            })
        }
    }

    /// The operand and alpha as the interpreter takes them.
    fn operands(&self) -> Result<(OperandBits, ScalarBits), WeightPatchError> {
        let operand = parse_operand_hex(&self.numerics, &self.operand)?;
        let alpha = match parse_operand_hex(&self.numerics, std::slice::from_ref(&self.alpha))? {
            OperandBits::F64(v) => ScalarBits::F64(v[0]),
            OperandBits::F32(v) => ScalarBits::F32(v[0]),
        };
        Ok((operand, alpha))
    }

    /// Run the op over the bound tensor of `container` and return the exact BF16 result of every element
    /// that changes. An op that changes nothing is refused: a patch edits at least one element.
    pub fn resolve(&self, container: &Path) -> Result<ResultPatch, WeightPatchError> {
        self.check_intact()?;
        self.check_binding(container)?;
        let kind = self.op_kind()?;
        resolve_on_container(
            container,
            &super::qualified(&self.source.representation, &self.tensor),
            &self.numerics,
            |_| kind,
        )
    }
}

/// Stored operand bits, one fixed-width lowercase hex string per element, at the width `numerics` works in:
/// sixteen digits for the f64 profiles, eight for the f32 one. Shared by every structured `.vlp` op.
pub(super) fn parse_operand_hex(
    numerics: &str,
    texts: &[String],
) -> Result<OperandBits, WeightPatchError> {
    let profile =
        Profile::parse(Some(numerics)).map_err(|r| WeightPatchError::PatchOp(r.to_string()))?;
    let width = if profile.works_in_f64() {
        F64_HEX_DIGITS
    } else {
        F32_HEX_DIGITS
    };
    let parse = |text: &str| -> Result<u64, WeightPatchError> {
        if text.len() != width
            || !text
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            return Err(WeightPatchError::PatchOp(format!(
                "`{text}` is not {width} lowercase hex digits, the width `{numerics}` works in"
            )));
        }
        u64::from_str_radix(text, 16)
            .map_err(|e| WeightPatchError::PatchOp(format!("`{text}`: {e}")))
    };
    let bits: Vec<u64> = texts.iter().map(|t| parse(t)).collect::<Result<_, _>>()?;
    Ok(if profile.works_in_f64() {
        OperandBits::F64(bits)
    } else {
        OperandBits::F32(bits.into_iter().map(|b| b as u32).collect())
    })
}

/// What `tensor` of `container` is, where that changes whether a structured op may touch it. What the tensor IS
/// comes from the container's graph, not from what it is called.
pub(super) fn tensor_role(container: &Path, tensor: &str) -> Result<TensorRole, WeightPatchError> {
    Ok(
        if tensor_object_kind(container, tensor)? == ObjectKind::Embedding {
            TensorRole::TokenEmbedding
        } else {
            TensorRole::Ordinary
        },
    )
}

/// Run a structured op over `tensor` of `container` and return the exact BF16 result of every element that
/// changes. `kind` is given the tensor's `[rows, cols]`. An op that changes nothing is refused: a patch edits
/// at least one element. Shared by every structured `.vlp` op.
pub(super) fn resolve_on_container(
    container: &Path,
    tensor: &str,
    numerics: &str,
    kind: impl FnOnce([usize; 2]) -> OpKind,
) -> Result<ResultPatch, WeightPatchError> {
    let ([rows, cols], base) = read_bf16_tensor(container, tensor)?;
    let op = StructuredOp {
        numerics: Some(numerics.to_string()),
        shape: [rows, cols],
        kind: kind([rows, cols]),
        fixups: Vec::new(),
    };
    let role = tensor_role(container, tensor)?;
    let result = run(
        &op,
        &BaseTensor {
            shape: [rows, cols],
            bits: &base,
            role,
        },
    )
    .map_err(|e| WeightPatchError::Structured(e.to_string()))?;
    let edits: Vec<ResultEdit> = base
        .iter()
        .zip(&result)
        .enumerate()
        .filter(|(_, (old, new))| old != new)
        .map(|(i, (_, &new_bits))| ResultEdit {
            row: i / cols,
            col: i % cols,
            new_bits,
        })
        .collect();
    if edits.is_empty() {
        return Err(WeightPatchError::Empty);
    }
    Ok(ResultPatch {
        tensor: tensor.to_string(),
        edits,
    })
}
