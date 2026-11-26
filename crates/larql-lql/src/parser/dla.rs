//! `DLA "<prompt>" FOR "<token_a>" VS "<token_b>" [LAYERS <lo>-<hi> | AT LAYER <n> | AT LAYERS (...)] [TOP <k> [COLUMNS]];`
//!
//! Direct Logit Attribution: decomposes `logit(token_a) - logit(token_b)` at the prompt's final
//! position into embedding + per-layer operator + per-layer FFN contributions, and optionally
//! attributes each dense FFN's contribution down to individual `W_down` columns.

use super::{ParseError, Parser};
use crate::ast::{CaptureLayers, Statement};
use crate::lexer::{Keyword, Token};

const TOPIC: &str = "DLA";

fn is_word(token: &Token, word: &str) -> bool {
    matches!(token, Token::Ident(s) if s.eq_ignore_ascii_case(word))
}

impl Parser {
    pub(crate) fn parse_dla(&mut self) -> Result<Statement, ParseError> {
        self.expect_ident_eq(TOPIC)?;
        let prompt = self.expect_string()?;
        if !self.check_keyword(Keyword::For) {
            return Err(self.expected_one_of("DLA \"<prompt>\"", &["FOR"], TOPIC));
        }
        self.advance();
        let target_a = self.expect_string()?;
        if !is_word(&self.peek(), "VS") {
            return Err(self.expected_one_of("FOR \"<token_a>\"", &["VS"], TOPIC));
        }
        self.advance();
        let target_b = self.expect_string()?;

        let mut layers: Option<CaptureLayers> = None;
        let mut top_columns: Option<u32> = None;

        loop {
            if self.check_keyword(Keyword::At) {
                if layers.is_some() {
                    return Err(ParseError(
                        "DLA: layer selection specified more than once".into(),
                    ));
                }
                layers = self.parse_dla_at_layers()?;
            } else if self.check_keyword(Keyword::Layers) {
                if layers.is_some() {
                    return Err(ParseError(
                        "DLA: layer selection specified more than once".into(),
                    ));
                }
                self.advance();
                layers = self.parse_dla_layer_spec()?;
            } else if self.check_keyword(Keyword::Top) {
                if top_columns.is_some() {
                    return Err(ParseError("DLA: TOP specified more than once".into()));
                }
                self.advance();
                let k = self.expect_u32()?;
                if is_word(&self.peek(), "COLUMNS") || is_word(&self.peek(), "COLUMN") {
                    self.advance();
                }
                top_columns = Some(k);
            } else {
                break;
            }
        }

        self.eat_semicolon();
        Ok(Statement::Dla {
            prompt,
            target_a,
            target_b,
            layers,
            top_columns,
        })
    }

    fn parse_dla_at_layers(&mut self) -> Result<Option<CaptureLayers>, ParseError> {
        self.advance(); // consume AT
        if self.check_keyword(Keyword::Layer) {
            self.advance();
            return Ok(Some(CaptureLayers::One(self.expect_u32()? as usize)));
        }
        if self.check_keyword(Keyword::Layers) {
            self.advance();
            return self.parse_dla_layer_spec();
        }
        Err(self.expected_one_of("DLA ... AT", &["LAYER", "LAYERS"], TOPIC))
    }

    fn parse_dla_layer_spec(&mut self) -> Result<Option<CaptureLayers>, ParseError> {
        if self.check_keyword(Keyword::All) {
            self.advance();
            return Ok(None);
        }
        if matches!(self.peek(), Token::LParen) {
            return Ok(Some(CaptureLayers::Many(self.parse_column_list()?)));
        }
        let lo = self.expect_u32()? as usize;
        if matches!(self.peek(), Token::Dash) {
            self.advance();
            let hi = self.expect_u32()? as usize;
            if hi < lo {
                return Err(ParseError(format!(
                    "LAYERS {lo}-{hi}: end layer {hi} is before start layer {lo}"
                )));
            }
            return Ok(Some(CaptureLayers::Many((lo..=hi).collect())));
        }
        Ok(Some(CaptureLayers::Many(vec![lo])))
    }
}
