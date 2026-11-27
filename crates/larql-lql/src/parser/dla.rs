//! `DLA "<prompt>" FOR "<token_a>" VS "<token_b>" [LAYERS <lo>-<hi> | AT LAYER <n> | AT LAYERS (...)] [TOP <k> [COLUMNS]];`
//!
//! Direct Logit Attribution: decomposes `logit(token_a) - logit(token_b)` at the prompt's final
//! position into embedding + per-layer operator + per-layer FFN contributions, and optionally
//! attributes each dense FFN's contribution down to individual `W_down` columns.

use super::{ParseError, Parser};
use crate::ast::{CaptureLayers, DlaTarget, Statement};
use crate::lexer::{Keyword, Token};

const TOPIC: &str = "DLA";

fn is_word(token: &Token, word: &str) -> bool {
    matches!(token, Token::Ident(s) if s.eq_ignore_ascii_case(word))
}

pub(crate) struct ParsedDlaBody {
    pub prompt: String,
    pub target_a: DlaTarget,
    pub target_b: DlaTarget,
    pub layers: Option<CaptureLayers>,
    pub top_columns: Option<u32>,
    pub top_heads: Option<u32>,
    pub top_channels: Option<u32>,
}

impl Parser {
    pub(crate) fn parse_dla(&mut self) -> Result<Statement, ParseError> {
        self.expect_ident_eq(TOPIC)?;
        let b = self.parse_dla_body()?;
        Ok(Statement::Dla {
            prompt: b.prompt,
            target_a: b.target_a,
            target_b: b.target_b,
            layers: b.layers,
            top_columns: b.top_columns,
            top_heads: b.top_heads,
            top_channels: b.top_channels,
        })
    }

    pub(crate) fn parse_dla_body(&mut self) -> Result<ParsedDlaBody, ParseError> {
        let prompt = self.expect_string()?;
        if !self.check_keyword(Keyword::For) {
            return Err(self.expected_one_of("DLA \"<prompt>\"", &["FOR"], TOPIC));
        }
        self.advance();
        let target_a = self.parse_dla_target("FOR")?;
        if !is_word(&self.peek(), "VS") {
            return Err(self.expected_one_of("FOR <target>", &["VS"], TOPIC));
        }
        self.advance();
        let target_b = self.parse_dla_target("VS")?;

        let mut layers: Option<CaptureLayers> = None;
        let mut top_columns: Option<u32> = None;
        let mut top_heads: Option<u32> = None;
        let mut top_channels: Option<u32> = None;

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
                self.advance();
                let k = self.expect_u32()?;
                if is_word(&self.peek(), "HEADS") || is_word(&self.peek(), "HEAD") {
                    self.advance();
                    if top_heads.is_some() {
                        return Err(ParseError("DLA: TOP HEADS specified more than once".into()));
                    }
                    top_heads = Some(k);
                } else if is_word(&self.peek(), "CHANNELS") || is_word(&self.peek(), "CHANNEL") {
                    self.advance();
                    if top_channels.is_some() {
                        return Err(ParseError(
                            "DLA: TOP CHANNELS specified more than once".into(),
                        ));
                    }
                    top_channels = Some(k);
                } else {
                    if is_word(&self.peek(), "COLUMNS")
                        || is_word(&self.peek(), "COLUMN")
                        || is_word(&self.peek(), "NEURONS")
                        || is_word(&self.peek(), "NEURON")
                    {
                        self.advance();
                    }
                    if top_columns.is_some() {
                        return Err(ParseError(
                            "DLA: TOP COLUMNS / NEURONS specified more than once".into(),
                        ));
                    }
                    top_columns = Some(k);
                }
            } else {
                break;
            }
        }

        self.eat_semicolon();
        Ok(ParsedDlaBody {
            prompt,
            target_a,
            target_b,
            layers,
            top_columns,
            top_heads,
            top_channels,
        })
    }

    pub(crate) fn parse_logit(&mut self) -> Result<Statement, ParseError> {
        self.advance(); // consume LOGIT
        if !self.check_keyword(Keyword::Diff) {
            return Err(self.expected_after("LOGIT"));
        }
        self.advance(); // consume DIFF
        let (prompt, target_a, target_b) = self.parse_logit_diff_body("LOGIT DIFF")?;
        Ok(Statement::LogitDiff {
            prompt,
            target_a,
            target_b,
        })
    }

    pub(crate) fn parse_logit_diff_body(
        &mut self,
        topic: &str,
    ) -> Result<(String, DlaTarget, DlaTarget), ParseError> {
        let prompt = self.expect_string()?;
        if !self.check_keyword(Keyword::For) {
            return Err(self.expected_one_of(&format!("{topic} \"<prompt>\""), &["FOR"], topic));
        }
        self.advance();
        let target_a = self.parse_dla_target("FOR")?;
        if !is_word(&self.peek(), "VS") {
            return Err(self.expected_one_of("FOR <target>", &["VS"], topic));
        }
        self.advance();
        let target_b = self.parse_dla_target("VS")?;
        self.eat_semicolon();
        Ok((prompt, target_a, target_b))
    }

    pub(crate) fn parse_project(&mut self) -> Result<Statement, ParseError> {
        self.advance(); // consume PROJECT
        if !is_word(&self.peek(), "OUT") {
            return Err(self.expected_after("PROJECT"));
        }
        self.advance(); // consume OUT
        let target_a = self.parse_dla_target("PROJECT OUT")?;
        if matches!(self.peek(), Token::Dash) || is_word(&self.peek(), "VS") {
            self.advance();
        } else {
            return Err(self.expected_one_of(
                "PROJECT OUT <target_a>",
                &["-", "VS"],
                "PROJECT OUT",
            ));
        }
        let target_b = self.parse_dla_target("-")?;

        if self.check_keyword(Keyword::On) || self.check_keyword(Keyword::At) {
            self.advance();
        }
        let layers = if self.check_keyword(Keyword::Layer) {
            self.advance();
            CaptureLayers::One(self.expect_u32()? as usize)
        } else if self.check_keyword(Keyword::Layers) {
            self.advance();
            self.parse_dla_layer_spec()?.ok_or_else(|| {
                ParseError("PROJECT OUT requires an explicit layer list or range".into())
            })?
        } else {
            return Err(self.expected_one_of(
                "PROJECT OUT <A> - <B>",
                &["ON LAYERS", "AT LAYERS", "LAYERS"],
                "PROJECT OUT",
            ));
        };

        let mut target = crate::ast::ProjectOutTarget::W2;
        let mut scale = 1.0f64;
        loop {
            if is_word(&self.peek(), "TARGET") {
                self.advance();
                target = match self.peek() {
                    Token::Ident(ref s)
                        if s.eq_ignore_ascii_case("W2") || s.eq_ignore_ascii_case("FFN") =>
                    {
                        self.advance();
                        crate::ast::ProjectOutTarget::W2
                    }
                    Token::Ident(ref s)
                        if s.eq_ignore_ascii_case("O_PROJ") || s.eq_ignore_ascii_case("ATTN") =>
                    {
                        self.advance();
                        crate::ast::ProjectOutTarget::OProj
                    }
                    Token::Ident(ref s)
                        if s.eq_ignore_ascii_case("CONV_OUT") || s.eq_ignore_ascii_case("CONV") =>
                    {
                        self.advance();
                        crate::ast::ProjectOutTarget::ConvOut
                    }
                    Token::Keyword(Keyword::All) => {
                        self.advance();
                        crate::ast::ProjectOutTarget::All
                    }
                    Token::Ident(ref s) if s.eq_ignore_ascii_case("ALL") => {
                        self.advance();
                        crate::ast::ProjectOutTarget::All
                    }
                    _ => {
                        return Err(self.expected_one_of(
                            "PROJECT OUT ... TARGET",
                            &["W2", "O_PROJ", "CONV_OUT", "ALL"],
                            "PROJECT OUT",
                        ));
                    }
                };
            } else if is_word(&self.peek(), "SCALE") {
                self.advance();
                let negative = matches!(self.peek(), Token::Dash);
                if negative {
                    self.advance();
                }
                let mag = match self.peek() {
                    Token::NumberLit(v) => {
                        self.advance();
                        v
                    }
                    Token::IntegerLit(v) => {
                        self.advance();
                        v as f64
                    }
                    other => {
                        return Err(ParseError(format!(
                            "expected a number for SCALE, got {other:?}"
                        )));
                    }
                };
                scale = if negative { -mag } else { mag };
            } else {
                break;
            }
        }
        self.eat_semicolon();
        Ok(Statement::ProjectOut {
            target_a,
            target_b,
            layers,
            target,
            scale,
        })
    }

    fn parse_dla_target(&mut self, clause: &str) -> Result<DlaTarget, ParseError> {
        if is_word(&self.peek(), "ID") {
            self.advance();
            let id = self.expect_u32()?;
            return Ok(DlaTarget::Id(id));
        }
        if matches!(self.peek(), Token::StringLit(_)) {
            return Ok(DlaTarget::Text(self.expect_string()?));
        }
        Err(self.expected_one_of(
            &format!("DLA ... {clause}"),
            &["\"<token>\"", "ID <n>"],
            TOPIC,
        ))
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
