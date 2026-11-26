//! `EXPORT WEIGHT "<tensor>" [ROWS (<i>, …)] [COLUMNS (<j>, …)] TO "<file>.npy" [AS BITS]`
//! and `EXPORT SAFETENSORS [TO] "<output_dir>" [FROM [BASE] "<hf_checkpoint_dir>"]`.
//!
//! `EXPORT`, `WEIGHT`, `SAFETENSORS`, `BASE`, `ROWS`, `COLUMNS`, `AS` and `BITS` are read as
//! words in this position, not reserved, so no existing identifier changes meaning.

use super::{ParseError, Parser};
use crate::ast::Statement;
use crate::lexer::{Keyword, Token};

/// Whether the next token is the word `word`.
fn is_word(token: &Token, word: &str) -> bool {
    matches!(token, Token::Ident(s) if s.eq_ignore_ascii_case(word))
}

impl Parser {
    pub(crate) fn parse_export(&mut self) -> Result<Statement, ParseError> {
        self.expect_ident_eq("EXPORT")?;
        if is_word(&self.peek(), "WEIGHT") {
            self.advance();
            self.parse_export_weight_body()
        } else if self.check_keyword(Keyword::Safetensors) || is_word(&self.peek(), "SAFETENSORS") {
            self.advance();
            self.parse_export_safetensors_body()
        } else {
            Err(self.expected_after("EXPORT"))
        }
    }

    fn parse_export_weight_body(&mut self) -> Result<Statement, ParseError> {
        let tensor = self.expect_string()?;
        let rows = if is_word(&self.peek(), "ROWS") {
            self.advance();
            Some(self.parse_column_list()?)
        } else {
            None
        };
        let columns = if is_word(&self.peek(), "COLUMNS") {
            self.advance();
            Some(self.parse_column_list()?)
        } else {
            None
        };
        if !self.check_keyword(Keyword::To) {
            return Err(self.expected_one_of(
                "EXPORT WEIGHT \"<tensor>\"",
                &["ROWS", "COLUMNS", "TO"],
                "EXPORT WEIGHT",
            ));
        }
        self.advance();
        let output = self.expect_string()?;
        let bits = if is_word(&self.peek(), "AS") {
            self.advance();
            self.expect_ident_eq("BITS")?;
            true
        } else {
            false
        };
        self.eat_semicolon();
        Ok(Statement::ExportWeight {
            tensor,
            rows,
            columns,
            output,
            bits,
        })
    }

    fn parse_export_safetensors_body(&mut self) -> Result<Statement, ParseError> {
        if self.check_keyword(Keyword::To) {
            self.advance();
        }
        let output = self.expect_string()?;
        let base_checkpoint = if self.check_keyword(Keyword::From) {
            self.advance();
            if is_word(&self.peek(), "BASE") {
                self.advance();
            }
            Some(self.expect_string()?)
        } else {
            None
        };
        self.eat_semicolon();
        Ok(Statement::ExportSafetensors {
            output,
            base_checkpoint,
        })
    }
}
