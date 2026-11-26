//! Patch statement parsers: BEGIN PATCH, SAVE PATCH, APPLY PATCH, SHOW PATCHES, REMOVE PATCH.

use super::{ParseError, Parser};
use crate::ast::*;
use crate::lexer::{Keyword, Token};

impl Parser {
    /// Parse a statement starting with BEGIN (BEGIN PATCH "file.vlp").
    pub(crate) fn parse_begin(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Begin)?;
        self.expect_keyword_after("BEGIN", Keyword::Patch)?;
        let path = self.expect_string()?;
        self.eat_semicolon();
        Ok(Statement::BeginPatch { path })
    }

    /// Parse SAVE PATCH.
    pub(crate) fn parse_save(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Save)?;
        self.expect_keyword_after("SAVE", Keyword::Patch)?;
        self.eat_semicolon();
        Ok(Statement::SavePatch)
    }

    /// Parse APPLY PATCH "file.vlp".
    pub(crate) fn parse_apply(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Apply)?;
        self.expect_keyword_after("APPLY", Keyword::Patch)?;
        let path = self.expect_string()?;
        self.eat_semicolon();
        Ok(Statement::ApplyPatch { path })
    }

    /// Parse REMOVE PATCH "file.vlp", REMOVE DIRECTION, or REMOVE INPUT DIRECTION.
    pub(crate) fn parse_remove(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Remove)?;
        if matches!(self.peek(), Token::Ident(ref s) if s.eq_ignore_ascii_case("DIRECTION")) {
            return self.parse_remove_direction(false);
        }
        if matches!(self.peek(), Token::Ident(ref s) if s.eq_ignore_ascii_case("INPUT")) {
            self.advance();
            return self.parse_remove_direction(true);
        }
        self.expect_keyword_after("REMOVE", Keyword::Patch)?;
        let path = self.expect_string()?;
        self.eat_semicolon();
        Ok(Statement::RemovePatch { path })
    }
}

/// The largest integer magnitude an f64 holds exactly.
const MAX_EXACT_INTEGER_SCALE: u64 = 1 << 53;

impl Parser {
    /// `REMOVE [INPUT] DIRECTION "<operand file>" FROM "<tensor>" {COLUMNS (...) | ROWS (...) | HEAD <n> | HEADS (...)} SCALE <f> NUMERICS "<profile>"`;
    /// `REMOVE` (and optional `INPUT`) is already consumed.
    fn parse_remove_direction(&mut self, input_axis: bool) -> Result<Statement, ParseError> {
        self.expect_ident_eq("DIRECTION")?;
        let operand = self.expect_string()?;
        self.expect_keyword(Keyword::From)?;
        let tensor = self.expect_string()?;
        let (columns, heads) = self.parse_axis_target(input_axis)?;
        self.expect_ident_eq("SCALE")?;
        let negative = matches!(self.peek(), Token::Dash);
        if negative {
            self.advance();
        }
        let magnitude = match self.peek() {
            Token::NumberLit(v) => {
                self.advance();
                v
            }
            // `SCALE 1` lexes as an integer; it is the f64 it names.
            Token::IntegerLit(v) => {
                // Beyond 2^53 an integer is not exactly an f64; the scale is never rounded silently.
                if v.unsigned_abs() > MAX_EXACT_INTEGER_SCALE {
                    return Err(ParseError(format!(
                        "SCALE {v} is not exactly representable as an f64"
                    )));
                }
                self.advance();
                v as f64
            }
            other => {
                return Err(ParseError(format!(
                    "expected a number for SCALE, got {other:?}"
                )))
            }
        };
        let scale = if negative { -magnitude } else { magnitude };
        self.expect_ident_eq("NUMERICS")?;
        let numerics = self.expect_string()?;
        self.eat_semicolon();
        Ok(Statement::RemoveDirection {
            operand,
            tensor,
            columns,
            heads,
            input_axis,
            scale,
            numerics,
        })
    }

    /// Parse `{COLUMNS (...) | HEAD <n> | HEADS (...)}` (or `ROWS (...)` when `input_axis` is true).
    fn parse_axis_target(
        &mut self,
        input_axis: bool,
    ) -> Result<(Vec<usize>, Option<Vec<usize>>), ParseError> {
        let index_word = if input_axis { "ROWS" } else { "COLUMNS" };
        match self.peek() {
            Token::Ident(ref s) if s.eq_ignore_ascii_case(index_word) => {
                self.advance();
                Ok((self.parse_column_list()?, None))
            }
            Token::Ident(ref s) if s.eq_ignore_ascii_case("HEAD") => {
                self.advance();
                let h = self.expect_u32()? as usize;
                Ok((Vec::new(), Some(vec![h])))
            }
            Token::Ident(ref s) if s.eq_ignore_ascii_case("HEADS") => {
                self.advance();
                let hs = self.parse_column_list()?;
                Ok((Vec::new(), Some(hs)))
            }
            other => Err(ParseError(format!(
                "expected {index_word}, HEAD, or HEADS, got {other:?}"
            ))),
        }
    }
}

impl Parser {
    /// `(<item>, <item>, …)`: a parenthesised, non-empty list of non-negative indices or ranges.
    /// Each item may be a single index `<n>`, a half-open range `<lo>..<hi>` (`[lo, hi)`), or
    /// an inclusive range `<lo>..=<hi>` (`[lo, hi]`).
    pub(super) fn parse_column_list(&mut self) -> Result<Vec<usize>, ParseError> {
        self.expect_token(&Token::LParen)?;
        let mut indices = Vec::new();
        self.parse_index_or_range(&mut indices)?;
        while matches!(self.peek(), Token::Comma) {
            self.advance();
            self.parse_index_or_range(&mut indices)?;
        }
        self.expect_token(&Token::RParen)?;
        Ok(indices)
    }

    fn parse_index_or_range(&mut self, out: &mut Vec<usize>) -> Result<(), ParseError> {
        let lo = self.expect_u32()? as usize;
        match self.peek() {
            Token::DotDot => {
                self.advance();
                let hi = self.expect_u32()? as usize;
                if hi <= lo {
                    return Err(ParseError(format!(
                        "empty or reversed half-open range {lo}..{hi}: end must exceed start"
                    )));
                }
                out.extend(lo..hi);
            }
            Token::DotDotEq => {
                self.advance();
                let hi = self.expect_u32()? as usize;
                if hi < lo {
                    return Err(ParseError(format!(
                        "reversed inclusive range {lo}..={hi}: end must be >= start"
                    )));
                }
                out.extend(lo..=hi);
            }
            _ => out.push(lo),
        }
        Ok(())
    }

    /// `DISCARD WEIGHT EDITS`. All three are read as words in this position, not reserved.
    pub(crate) fn parse_discard_weight_edits(&mut self) -> Result<Statement, ParseError> {
        self.expect_ident_eq("DISCARD")?;
        self.expect_words_after("DISCARD", &["WEIGHT", "EDITS"])?;
        self.eat_semicolon();
        Ok(Statement::DiscardWeightEdits)
    }

    /// `ADD LOW RANK "<operand file>" TO "<tensor>" {COLUMNS (...) | HEAD <n> | HEADS (...)} NUMERICS "<profile>"`.
    pub(crate) fn parse_add_low_rank(&mut self) -> Result<Statement, ParseError> {
        self.expect_ident_eq("ADD")?;
        self.expect_words_after("ADD", &["LOW", "RANK"])?;
        let operands = self.expect_string()?;
        self.expect_keyword(Keyword::To)?;
        let tensor = self.expect_string()?;
        let (columns, heads) = self.parse_axis_target(false)?;
        self.expect_ident_eq("NUMERICS")?;
        let numerics = self.expect_string()?;
        self.eat_semicolon();
        Ok(Statement::AddLowRank {
            operands,
            tensor,
            columns,
            heads,
            numerics,
        })
    }
}
