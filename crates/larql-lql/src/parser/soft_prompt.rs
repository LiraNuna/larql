//! Parsers for `LOAD SOFT_PROMPT "<path>";` and `CLEAR SOFT_PROMPT;`.

use super::{ParseError, Parser};
use crate::ast::Statement;
use crate::lexer::Token;

impl Parser {
    pub(crate) fn parse_load(&mut self) -> Result<Statement, ParseError> {
        self.advance(); // consume LOAD
        match self.peek() {
            Token::Ident(ref s) if s.eq_ignore_ascii_case("SOFT_PROMPT") => {
                self.advance();
                let path = self.expect_string()?;
                self.eat_semicolon();
                Ok(Statement::LoadSoftPrompt { path })
            }
            _ => Err(self.expected_after("LOAD")),
        }
    }

    pub(crate) fn parse_clear(&mut self) -> Result<Statement, ParseError> {
        self.advance(); // consume CLEAR
        match self.peek() {
            Token::Ident(ref s) if s.eq_ignore_ascii_case("SOFT_PROMPT") => {
                self.advance();
                self.eat_semicolon();
                Ok(Statement::ClearSoftPrompt)
            }
            _ => Err(self.expected_after("CLEAR")),
        }
    }
}
