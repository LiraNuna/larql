//! `OVERLAY SAFETENSORS "<path>";`, `OVERLAY LORA "<adapter_dir>" [SCALE <f>];`, `OVERLAY CLEAR;`.

use super::{ParseError, Parser};
use crate::ast::{OverlayKind, Statement};
use crate::lexer::{Keyword, Token};

const TOPIC: &str = "OVERLAY";

fn is_word(token: &Token, word: &str) -> bool {
    matches!(token, Token::Ident(s) if s.eq_ignore_ascii_case(word))
}

impl Parser {
    pub(crate) fn parse_overlay(&mut self) -> Result<Statement, ParseError> {
        self.expect_ident_eq(TOPIC)?;
        let kind =
            if self.check_keyword(Keyword::Safetensors) || is_word(&self.peek(), "SAFETENSORS") {
                self.advance();
                let path = self.expect_string()?;
                OverlayKind::Safetensors { path }
            } else if is_word(&self.peek(), "LORA") {
                self.advance();
                let path = self.expect_string()?;
                let scale = if is_word(&self.peek(), "SCALE") {
                    self.advance();
                    Some(self.parse_signed_f64()?)
                } else {
                    None
                };
                OverlayKind::Lora { path, scale }
            } else if is_word(&self.peek(), "CLEAR") {
                self.advance();
                OverlayKind::Clear
            } else {
                return Err(self.expected_one_of(TOPIC, &["SAFETENSORS", "LORA", "CLEAR"], TOPIC));
            };
        self.eat_semicolon();
        Ok(Statement::Overlay { kind })
    }

    fn parse_signed_f64(&mut self) -> Result<f64, ParseError> {
        let neg = if matches!(self.peek(), Token::Dash) {
            self.advance();
            true
        } else {
            false
        };
        let raw = match self.peek() {
            Token::NumberLit(n) => {
                self.advance();
                n
            }
            Token::IntegerLit(n) => {
                self.advance();
                n as f64
            }
            other => {
                return Err(ParseError(format!(
                    "expected numeric SCALE value, got {other:?}"
                )));
            }
        };
        Ok(if neg { -raw } else { raw })
    }
}
