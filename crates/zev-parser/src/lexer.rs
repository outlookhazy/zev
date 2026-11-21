use crate::error::ParseError;
use crate::token::{Token, TokenKind};
use zev_ast::{DurationUnit, Span};

/// Tokenize `.zev` source code into a `Vec<Token>` ending with [`TokenKind::Eof`].
pub fn lex(source: &str, filename: Option<&str>) -> Result<Vec<Token>, ParseError> {
    let mut lexer = Lexer::new(source, filename);
    lexer.tokenize()
}

struct Lexer<'a> {
    source: &'a str,
    filename: Option<&'a str>,
    pos: usize,
    line: u32,
    col: u32,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str, filename: Option<&'a str>) -> Self {
        Self {
            source,
            filename,
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    fn peek(&self) -> Option<char> {
        self.source[self.pos..].chars().next()
    }

    fn peek_second(&self) -> Option<char> {
        let mut iter = self.source[self.pos..].chars();
        iter.next()?;
        iter.next()
    }

    fn bump(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += ch.len_utf8();
        if ch == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(ch)
    }

    fn span_from(&self, start: usize, start_line: u32, start_col: u32) -> Span {
        Span::new(start as u32, self.pos as u32, start_line, start_col)
    }

    fn err_at(&self, msg: impl Into<String>, span: Span) -> ParseError {
        ParseError::new(msg, span, self.source, self.filename)
    }

    fn tokenize(&mut self) -> Result<Vec<Token>, ParseError> {
        let mut tokens = Vec::new();

        while let Some(ch) = self.peek() {
            // Skip whitespace
            if ch.is_whitespace() {
                self.bump();
                continue;
            }

            // Skip line comments `// ...`
            if ch == '/' && self.peek_second() == Some('/') {
                while let Some(c) = self.peek() {
                    self.bump();
                    if c == '\n' {
                        break;
                    }
                }
                continue;
            }

            // Skip block comments `/* ... */` (supports nesting)
            if ch == '/' && self.peek_second() == Some('*') {
                let start = self.pos;
                let start_line = self.line;
                let start_col = self.col;
                self.bump();
                self.bump();
                let mut depth = 1usize;
                while depth > 0 {
                    match self.bump() {
                        Some('/') if self.peek() == Some('*') => {
                            self.bump();
                            depth += 1;
                        }
                        Some('*') if self.peek() == Some('/') => {
                            self.bump();
                            depth -= 1;
                        }
                        Some(_) => {}
                        None => {
                            return Err(self
                                .err_at(
                                    "unterminated `/* ... */` block comment",
                                    Span::new(
                                        start as u32,
                                        self.pos as u32,
                                        start_line,
                                        start_col,
                                    ),
                                )
                                .with_doc_topic("syntax"));
                        }
                    }
                }
                continue;
            }

            let start = self.pos;
            let start_line = self.line;
            let start_col = self.col;

            // Identifiers & Keywords
            if is_ident_start(ch) {
                self.bump();
                while let Some(c) = self.peek() {
                    if is_ident_continue(c) {
                        self.bump();
                    } else {
                        break;
                    }
                }
                let text = &self.source[start..self.pos];
                let kind = TokenKind::keyword_from_str(text)
                    .unwrap_or_else(|| TokenKind::Ident(text.to_owned()));
                tokens.push(Token::new(kind, self.span_from(start, start_line, start_col)));
                continue;
            }

            // Numeric & Duration Literals
            if ch.is_ascii_digit() {
                let tok = self.lex_number(start, start_line, start_col)?;
                tokens.push(tok);
                continue;
            }

            // String Literals
            if ch == '"' {
                let tok = self.lex_string(start, start_line, start_col)?;
                tokens.push(tok);
                continue;
            }

            // Character Literals `'A'` -> Int, or foreign Rust-style Lifetimes `'a` / `'static` -> TokenKind::Lifetime
            if ch == '\'' {
                let tok = self.lex_char_or_lifetime(start, start_line, start_col)?;
                tokens.push(tok);
                continue;
            }

            // Punctuation & Operators
            self.bump();
            let kind = match ch {
                '(' => TokenKind::LParen,
                ')' => TokenKind::RParen,
                '{' => TokenKind::LBrace,
                '}' => TokenKind::RBrace,
                '[' => TokenKind::LBracket,
                ']' => TokenKind::RBracket,
                ';' => TokenKind::Semi,
                ',' => TokenKind::Comma,
                '@' => TokenKind::At,
                '?' => TokenKind::Question,
                '~' => TokenKind::Tilde,
                ':' => {
                    if self.peek() == Some(':') {
                        self.bump();
                        TokenKind::ColonColon
                    } else {
                        TokenKind::Colon
                    }
                }
                '.' => {
                    if self.peek() == Some('.') {
                        self.bump();
                        if self.peek() == Some('.') {
                            self.bump();
                            TokenKind::DotDotDot
                        } else {
                            TokenKind::DotDot
                        }
                    } else {
                        TokenKind::Dot
                    }
                }
                '+' => match self.peek() {
                    Some('+') => {
                        self.bump();
                        TokenKind::PlusPlus
                    }
                    Some('=') => {
                        self.bump();
                        TokenKind::PlusEq
                    }
                    _ => TokenKind::Plus,
                },
                '-' => match self.peek() {
                    Some('-') => {
                        self.bump();
                        TokenKind::MinusMinus
                    }
                    Some('=') => {
                        self.bump();
                        TokenKind::MinusEq
                    }
                    Some('>') => {
                        self.bump();
                        TokenKind::Arrow
                    }
                    _ => TokenKind::Minus,
                },
                '*' => {
                    if self.peek() == Some('=') {
                        self.bump();
                        TokenKind::StarEq
                    } else {
                        TokenKind::Star
                    }
                }
                '/' => {
                    if self.peek() == Some('=') {
                        self.bump();
                        TokenKind::SlashEq
                    } else {
                        TokenKind::Slash
                    }
                }
                '%' => {
                    if self.peek() == Some('=') {
                        self.bump();
                        TokenKind::PercentEq
                    } else {
                        TokenKind::Percent
                    }
                }
                '&' => match self.peek() {
                    Some('&') => {
                        self.bump();
                        TokenKind::AmpAmp
                    }
                    Some('=') => {
                        self.bump();
                        TokenKind::AmpEq
                    }
                    _ => TokenKind::Amp,
                },
                '|' => match self.peek() {
                    Some('|') => {
                        self.bump();
                        TokenKind::PipePipe
                    }
                    Some('=') => {
                        self.bump();
                        TokenKind::PipeEq
                    }
                    _ => TokenKind::Pipe,
                },
                '^' => {
                    if self.peek() == Some('=') {
                        self.bump();
                        TokenKind::CaretEq
                    } else {
                        TokenKind::Caret
                    }
                }
                '!' => {
                    if self.peek() == Some('=') {
                        self.bump();
                        TokenKind::BangEq
                    } else {
                        TokenKind::Bang
                    }
                }
                '=' => match self.peek() {
                    Some('=') => {
                        self.bump();
                        TokenKind::EqEq
                    }
                    Some('>') => {
                        self.bump();
                        TokenKind::FatArrow
                    }
                    _ => TokenKind::Eq,
                },
                '<' => match self.peek() {
                    Some('=') => {
                        self.bump();
                        TokenKind::LtEq
                    }
                    Some('<') => {
                        self.bump();
                        if self.peek() == Some('=') {
                            self.bump();
                            TokenKind::ShlEq
                        } else {
                            TokenKind::Shl
                        }
                    }
                    _ => TokenKind::Lt,
                },
                '>' => match self.peek() {
                    Some('=') => {
                        self.bump();
                        TokenKind::GtEq
                    }
                    Some('>') => {
                        self.bump();
                        if self.peek() == Some('=') {
                            self.bump();
                            TokenKind::ShrEq
                        } else {
                            TokenKind::Shr
                        }
                    }
                    _ => TokenKind::Gt,
                },
                other => {
                    let span = self.span_from(start, start_line, start_col);
                    return Err(self
                        .err_at(format!("unexpected character `{other}`"), span)
                        .with_doc_topic("syntax"));
                }
            };

            tokens.push(Token::new(kind, self.span_from(start, start_line, start_col)));
        }

        let eof_span = Span::new(self.pos as u32, self.pos as u32, self.line, self.col);
        tokens.push(Token::new(TokenKind::Eof, eof_span));
        Ok(tokens)
    }

    fn lex_number(
        &mut self,
        start: usize,
        start_line: u32,
        start_col: u32,
    ) -> Result<Token, ParseError> {
        let mut radix = 10u32;
        if self.peek() == Some('0') {
            match self.peek_second() {
                Some('x') | Some('X') => {
                    self.bump();
                    self.bump();
                    radix = 16;
                }
                Some('b') | Some('B') => {
                    self.bump();
                    self.bump();
                    radix = 2;
                }
                Some('o') | Some('O') => {
                    self.bump();
                    self.bump();
                    radix = 8;
                }
                _ => {}
            }
        }

        let digits_start = self.pos;
        let mut clean_digits = String::new();
        while let Some(c) = self.peek() {
            if c == '_' {
                self.bump();
            } else if c.is_digit(radix) {
                clean_digits.push(c);
                self.bump();
            } else if c == '.' && self.peek_second() == Some('.') {
                // Range operator `0..5`: stop right before `..`!
                break;
            } else {
                break;
            }
        }

        // Check for optional trailing alphabetic suffix (`us`, `ms`, `s`, or integer suffix `u32`, etc.)
        let suffix_start = self.pos;
        if let Some(c) = self.peek() {
            if is_ident_start(c) {
                while let Some(sc) = self.peek() {
                    if is_ident_continue(sc) {
                        self.bump();
                    } else {
                        break;
                    }
                }
            }
        }
        let suffix = &self.source[suffix_start..self.pos];
        let span = self.span_from(start, start_line, start_col);

        if clean_digits.is_empty() {
            return Err(self
                .err_at(
                    format!(
                        "expected base-{radix} digits in numeric literal `{}`",
                        &self.source[start..digits_start.max(self.pos)]
                    ),
                    span,
                )
                .with_doc_topic("syntax"));
        }

        let value = u64::from_str_radix(&clean_digits, radix).map_err(|e| {
            self.err_at(
                format!("integer literal out of 64-bit unsigned range: {e}"),
                span,
            )
            .with_doc_topic("syntax")
        })?;

        let kind = match suffix {
            "" => TokenKind::Int(value),
            "us" => TokenKind::Duration(value, DurationUnit::Us),
            "ms" => TokenKind::Duration(value, DurationUnit::Ms),
            "s" => TokenKind::Duration(value, DurationUnit::S),
            "u" | "U" | "l" | "L" | "ul" | "UL" | "ull" | "ULL" | "u8" | "u16" | "u32"
            | "u64" | "usize" | "i8" | "i16" | "i32" | "i64" | "isize" => TokenKind::Int(value),
            bad => {
                return Err(self
                    .err_at(format!("unknown numeric/duration suffix `{bad}`"), span)
                    .with_doc_topic("syntax")
                    .with_hint("valid duration suffixes are `us`, `ms`, and `s`"));
            }
        };

        Ok(Token::new(kind, span))
    }

    fn lex_string(
        &mut self,
        start: usize,
        start_line: u32,
        start_col: u32,
    ) -> Result<Token, ParseError> {
        self.bump(); // opening `"`
        let mut out = String::new();
        loop {
            match self.bump() {
                Some('"') => break,
                Some('\\') => {
                    let esc = self.lex_escape(start, start_line, start_col)?;
                    out.push(esc);
                }
                Some('\n') | None => {
                    let span = self.span_from(start, start_line, start_col);
                    return Err(self
                        .err_at("unterminated string literal", span)
                        .with_doc_topic("syntax"));
                }
                Some(c) => out.push(c),
            }
        }
        let span = self.span_from(start, start_line, start_col);
        Ok(Token::new(TokenKind::String(out), span))
    }

    fn lex_char_or_lifetime(
        &mut self,
        start: usize,
        start_line: u32,
        start_col: u32,
    ) -> Result<Token, ParseError> {
        self.bump(); // opening `'`

        // Detect foreign Rust lifetime `'a`, `'static`, `'_` when NOT closed as `'c'`.
        if let Some(first) = self.peek() {
            if is_ident_start(first) && self.peek_second() != Some('\'') {
                let ident_start = self.pos;
                self.bump();
                while let Some(c) = self.peek() {
                    if is_ident_continue(c) {
                        self.bump();
                    } else {
                        break;
                    }
                }
                let lt_name = self.source[ident_start..self.pos].to_owned();
                let span = self.span_from(start, start_line, start_col);
                return Ok(Token::new(TokenKind::Lifetime(lt_name), span));
            }
        }

        let ch = match self.bump() {
            Some('\\') => self.lex_escape(start, start_line, start_col)?,
            Some('\'') => {
                let span = self.span_from(start, start_line, start_col);
                return Err(self
                    .err_at("empty character literal", span)
                    .with_doc_topic("syntax"));
            }
            Some(c) => c,
            None => {
                let span = self.span_from(start, start_line, start_col);
                return Err(self
                    .err_at("unterminated character literal", span)
                    .with_doc_topic("syntax"));
            }
        };
        if self.bump() != Some('\'') {
            let span = self.span_from(start, start_line, start_col);
            return Err(self
                .err_at("unterminated character literal (expected closing `'`)", span)
                .with_doc_topic("syntax"));
        }
        let span = self.span_from(start, start_line, start_col);
        Ok(Token::new(TokenKind::Int(ch as u64), span))
    }

    fn lex_escape(
        &mut self,
        start: usize,
        start_line: u32,
        start_col: u32,
    ) -> Result<char, ParseError> {
        match self.bump() {
            Some('n') => Ok('\n'),
            Some('r') => Ok('\r'),
            Some('t') => Ok('\t'),
            Some('0') => Ok('\0'),
            Some('\\') => Ok('\\'),
            Some('"') => Ok('"'),
            Some('\'') => Ok('\''),
            Some('x') => {
                let h1 = self.bump().and_then(|c| c.to_digit(16));
                let h2 = self.bump().and_then(|c| c.to_digit(16));
                match (h1, h2) {
                    (Some(a), Some(b)) => Ok(((a * 16 + b) as u8) as char),
                    _ => {
                        let span = self.span_from(start, start_line, start_col);
                        Err(self
                            .err_at("invalid `\\xHH` hex escape sequence", span)
                            .with_doc_topic("syntax"))
                    }
                }
            }
            Some(other) => {
                let span = self.span_from(start, start_line, start_col);
                Err(self
                    .err_at(format!("unknown escape sequence `\\{other}`"), span)
                    .with_doc_topic("syntax"))
            }
            None => {
                let span = self.span_from(start, start_line, start_col);
                Err(self
                    .err_at("unterminated escape sequence", span)
                    .with_doc_topic("syntax"))
            }
        }
    }
}

fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn is_ident_continue(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphanumeric()
}
