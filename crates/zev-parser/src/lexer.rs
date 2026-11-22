use crate::error::ParseError;
use crate::token::{LexedFormatPart, Token, TokenKind};
use zev_ast::{DurationUnit, FormatSpec, PrimType, Span};

/// Tokenize `.zev` source code into a `Vec<Token>` ending with [`TokenKind::Eof`].
pub fn lex(source: &str, filename: Option<&str>) -> Result<Vec<Token>, ParseError> {
    let mut lexer = Lexer::new(source, filename);
    lexer.tokenize()
}

/// Best-effort lexing used by [`crate::collect_declared_generic_names`] so top-level generic
/// `struct` / `fn` names are still discovered even if a later token in the file is malformed.
pub(crate) fn lex_prefix_best_effort(source: &str) -> Vec<Token> {
    let mut lexer = Lexer::new(source, None);
    let mut tokens = Vec::new();
    let _ = lexer.tokenize_into(&mut tokens);
    let eof_span = Span::new(
        lexer.pos as u32,
        lexer.pos as u32,
        lexer.line,
        lexer.col,
    );
    tokens.push(Token::new(TokenKind::Eof, eof_span));
    tokens
}

struct Lexer<'a> {
    source: &'a str,
    filename: Option<&'a str>,
    pos: usize,
    end_pos: usize,
    line: u32,
    col: u32,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str, filename: Option<&'a str>) -> Self {
        Self {
            source,
            filename,
            pos: 0,
            end_pos: source.len(),
            line: 1,
            col: 1,
        }
    }

    fn sub_slice(
        source: &'a str,
        filename: Option<&'a str>,
        start_pos: usize,
        end_pos: usize,
        start_line: u32,
        start_col: u32,
    ) -> Self {
        Self {
            source,
            filename,
            pos: start_pos,
            end_pos,
            line: start_line,
            col: start_col,
        }
    }

    fn peek(&self) -> Option<char> {
        if self.pos >= self.end_pos {
            None
        } else {
            self.source[self.pos..self.end_pos].chars().next()
        }
    }

    fn peek_second(&self) -> Option<char> {
        if self.pos >= self.end_pos {
            return None;
        }
        let mut iter = self.source[self.pos..self.end_pos].chars();
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
        self.tokenize_into(&mut tokens)?;
        let eof_span = Span::new(self.pos as u32, self.pos as u32, self.line, self.col);
        tokens.push(Token::new(TokenKind::Eof, eof_span));
        Ok(tokens)
    }

    fn tokenize_into(&mut self, tokens: &mut Vec<Token>) -> Result<(), ParseError> {
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

            // Zero-heap format string `f"..."` — MUST be checked BEFORE `is_ident_start(ch)`!
            if ch == 'f' && self.peek_second() == Some('"') {
                let tok = self.lex_format_string(start, start_line, start_col)?;
                tokens.push(tok);
                continue;
            }

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

            // Numeric, Float & Duration Literals
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
                        if self.peek() == Some('=') {
                            self.bump();
                            TokenKind::DotDotEq
                        } else if self.peek() == Some('.') {
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

        Ok(())
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
                // Range operator `0..5` / `0..=5`: stop right before `..`!
                break;
            } else {
                break;
            }
        }

        if clean_digits.is_empty() {
            // Consume any trailing suffix for span clarity before reporting error
            while let Some(sc) = self.peek() {
                if is_ident_continue(sc) {
                    self.bump();
                } else {
                    break;
                }
            }
            let span = self.span_from(start, start_line, start_col);
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

        let mut is_float = false;
        let mut float_str = String::new();

        // Reject non-decimal float literals (`0x1.5`, `0b1.0`, `0o7.5`) while leaving ranges (`0x00..0x10`) intact.
        if radix != 10 && self.peek() == Some('.') && self.peek_second() != Some('.') {
            if self
                .peek_second()
                .is_some_and(|nc| nc.is_digit(radix) || nc.is_ascii_digit() || nc == '_')
            {
                self.bump(); // `.`
                while self
                    .peek()
                    .is_some_and(|c| is_ident_continue(c) || matches!(c, '+' | '-'))
                {
                    self.bump();
                }
                let span = self.span_from(start, start_line, start_col);
                return Err(self
                    .err_at(
                        "non-decimal float literals (e.g. `0x1.5`) are not supported; write decimal floating-point literals (e.g. `1.3125`)",
                        span,
                    )
                    .with_doc_topic("syntax"));
            }
        }

        // Decimal fractional part `.digits` — ONLY when `radix == 10 && c == '.' && next.is_ascii_digit()`!
        // Never touches `0..4`, `0..=4`, or method/field access, while catching trailing-dot `1.;` / `1.f32`.
        if radix == 10 && self.peek() == Some('.') {
            match self.peek_second() {
                Some('.') => {
                    // Range `0..4` or `0..=4`: leave `.` untouched!
                }
                Some('_') => {
                    self.bump(); // `.`
                    while self.peek().is_some_and(is_ident_continue) {
                        self.bump();
                    }
                    let span = self.span_from(start, start_line, start_col);
                    return Err(self
                        .err_at(
                            "malformed float literal: underscore `_` cannot immediately follow decimal point (write `1.5` or `1.500_000`)",
                            span,
                        )
                        .with_doc_topic("syntax"));
                }
                Some(nc) if nc.is_ascii_digit() => {
                    is_float = true;
                    float_str.push_str(&clean_digits);
                    self.bump(); // `.`
                    float_str.push('.');
                    while let Some(fc) = self.peek() {
                        if fc == '_' {
                            self.bump();
                        } else if fc.is_ascii_digit() {
                            float_str.push(fc);
                            self.bump();
                        } else {
                            break;
                        }
                    }
                    // Reject `1.2.3` double decimal points cleanly
                    if self.peek() == Some('.')
                        && self.peek_second().is_some_and(|nc2| nc2.is_ascii_digit())
                    {
                        self.bump();
                        while self.peek().is_some_and(|d| d.is_ascii_digit() || d == '_') {
                            self.bump();
                        }
                        let span = self.span_from(start, start_line, start_col);
                        return Err(self
                            .err_at(
                                "malformed float literal: unexpected second decimal point",
                                span,
                            )
                            .with_doc_topic("syntax"));
                    }
                }
                None
                | Some(
                    ';' | ',' | ')' | ']' | '}' | ' ' | '\t' | '\r' | '\n' | '+' | '-' | '*'
                    | '/' | '%' | '<' | '>' | '=' | '!' | '&' | '|' | '^' | ':',
                ) => {
                    self.bump(); // `.`
                    let span = self.span_from(start, start_line, start_col);
                    return Err(self
                        .err_at(
                            "malformed float literal: expected digits after decimal point (e.g. `1.0`)",
                            span,
                        )
                        .with_doc_topic("syntax"));
                }
                Some(nc) if is_ident_start(nc) => {
                    // Check if it looks like `1.f32`, `1.f64`, or `1.e5` (missing digit after `.`)
                    let rest = &self.source[self.pos + 1..self.end_pos];
                    let ident_len = rest
                        .chars()
                        .take_while(|&c| is_ident_continue(c))
                        .map(char::len_utf8)
                        .sum::<usize>();
                    let word = &rest[..ident_len];
                    if matches!(word, "f32" | "f64")
                        || word.starts_with('e')
                        || word.starts_with('E')
                    {
                        self.bump(); // `.`
                        for _ in word.chars() {
                            self.bump();
                        }
                        let span = self.span_from(start, start_line, start_col);
                        return Err(self
                            .err_at(
                                "malformed float literal: expected digits after decimal point (write `1.0f32` or `1.0e5`)",
                                span,
                            )
                            .with_doc_topic("syntax"));
                    }
                }
                _ => {}
            }
        }

        // Decimal scientific exponent `[eE][+-]?[0-9_]+`
        if radix == 10 && matches!(self.peek(), Some('e' | 'E')) {
            if !is_float {
                is_float = true;
                float_str.push_str(&clean_digits);
            }
            self.bump(); // `e` or `E`
            float_str.push('e');
            if let Some(sign @ ('+' | '-')) = self.peek() {
                self.bump();
                float_str.push(sign);
            }
            let mut exp_digits = String::new();
            while let Some(ec) = self.peek() {
                if ec == '_' {
                    self.bump();
                } else if ec.is_ascii_digit() {
                    exp_digits.push(ec);
                    self.bump();
                } else {
                    break;
                }
            }
            if exp_digits.is_empty() {
                let span = self.span_from(start, start_line, start_col);
                return Err(self
                    .err_at(
                        "malformed float literal: expected exponent digits after `e`",
                        span,
                    )
                    .with_doc_topic("syntax"));
            }
            float_str.push_str(&exp_digits);
        }

        // Optional trailing alphabetic suffix (`us`, `ms`, `s`, `f32`, `f64`, or integer suffix `u32`, etc.)
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

        // `<int>f32` / `<int>f64` (e.g. `1f32`, `100_f64`) produces a `FloatLit`
        if !is_float && radix == 10 && matches!(suffix, "f32" | "f64") {
            is_float = true;
            float_str.push_str(&clean_digits);
        }

        if is_float {
            let float_suffix = match suffix {
                "" => None,
                "f32" => Some(PrimType::F32),
                "f64" => Some(PrimType::F64),
                bad => {
                    return Err(self
                        .err_at(
                            format!("invalid float literal suffix `{bad}` (expected `f32` or `f64`)"),
                            span,
                        )
                        .with_doc_topic("syntax"));
                }
            };

            let bits = if float_suffix == Some(PrimType::F32) {
                let value32 = float_str.parse::<f32>().map_err(|e| {
                    self.err_at(format!("malformed float literal: {e}"), span)
                        .with_doc_topic("syntax")
                })?;
                if !value32.is_finite() {
                    return Err(self
                        .err_at("float literal out of finite IEEE-754 range", span)
                        .with_doc_topic("syntax"));
                }
                (value32 as f64).to_bits()
            } else {
                let value64 = float_str.parse::<f64>().map_err(|e| {
                    self.err_at(format!("malformed float literal: {e}"), span)
                        .with_doc_topic("syntax")
                })?;
                if !value64.is_finite() {
                    return Err(self
                        .err_at("float literal out of finite IEEE-754 range", span)
                        .with_doc_topic("syntax"));
                }
                value64.to_bits()
            };

            return Ok(Token::new(
                TokenKind::FloatLit {
                    bits,
                    suffix: float_suffix,
                },
                span,
            ));
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
                    .with_hint("valid suffixes are `us`, `ms`, `s`, `f32`, `f64`, or integer widths (`u32`, etc.)"));
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
                    push_escape(&mut out, esc);
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

    fn lex_format_string(
        &mut self,
        start: usize,
        start_line: u32,
        start_col: u32,
    ) -> Result<Token, ParseError> {
        self.bump(); // `f`
        self.bump(); // opening `"`

        let mut parts: Vec<LexedFormatPart> = Vec::new();
        let mut cur_lit = String::new();

        loop {
            match self.peek() {
                Some('"') => {
                    self.bump();
                    if !cur_lit.is_empty() {
                        parts.push(LexedFormatPart::Literal(cur_lit));
                    }
                    break;
                }
                Some('\n') | None => {
                    let span = self.span_from(start, start_line, start_col);
                    return Err(self
                        .err_at("unterminated format string literal", span)
                        .with_doc_topic("syntax"));
                }
                Some('\\') => {
                    self.bump();
                    let esc = self.lex_escape(start, start_line, start_col)?;
                    push_escape(&mut cur_lit, esc);
                }
                Some('}') => {
                    let r_start = self.pos;
                    let r_line = self.line;
                    let r_col = self.col;
                    self.bump();
                    if self.peek() == Some('}') {
                        self.bump();
                        cur_lit.push('}');
                    } else {
                        let span = self.span_from(r_start, r_line, r_col);
                        return Err(self
                            .err_at("unmatched `}` in format string", span)
                            .with_doc_topic("syntax")
                            .with_hint("escape literal closing braces as `}}`"));
                    }
                }
                Some('{') => {
                    if self.peek_second() == Some('{') {
                        self.bump();
                        self.bump();
                        cur_lit.push('{');
                        continue;
                    }

                    if !cur_lit.is_empty() {
                        parts.push(LexedFormatPart::Literal(std::mem::take(&mut cur_lit)));
                    }

                    let hole_start = self.pos;
                    let hole_line = self.line;
                    let hole_col = self.col;
                    self.bump(); // consume `{`

                    let expr_start_pos = self.pos;
                    let expr_start_line = self.line;
                    let expr_start_col = self.col;

                    let mut brace_depth = 0usize;
                    let mut paren_depth = 0usize;
                    let mut bracket_depth = 0usize;
                    let mut top_colon: Option<(usize, u32, u32)> = None;
                    let hole_close_pos: usize;

                    loop {
                        match self.peek() {
                            None | Some('\n') => {
                                let span = Span::new(
                                    hole_start as u32,
                                    self.pos as u32,
                                    hole_line,
                                    hole_col,
                                );
                                return Err(self
                                    .err_at(
                                        "unterminated `{...}` interpolation hole in format string",
                                        span,
                                    )
                                    .with_doc_topic("syntax"));
                            }
                            Some('"') => {
                                self.bump();
                                loop {
                                    match self.bump() {
                                        Some('"') => break,
                                        Some('\\') => {
                                            let _ = self.bump();
                                        }
                                        Some('\n') | None => {
                                            let span = Span::new(
                                                hole_start as u32,
                                                self.pos as u32,
                                                hole_line,
                                                hole_col,
                                            );
                                            return Err(self
                                                .err_at(
                                                    "unterminated string literal inside format string hole",
                                                    span,
                                                )
                                                .with_doc_topic("syntax"));
                                        }
                                        Some(_) => {}
                                    }
                                }
                            }
                            Some('\'') => {
                                self.bump();
                                if self.peek() == Some('\\') {
                                    self.bump();
                                    let _ = self.bump();
                                    if self.peek() == Some('\'') {
                                        self.bump();
                                    }
                                } else if self.peek_second() == Some('\'') {
                                    self.bump();
                                    self.bump();
                                }
                            }
                            Some('\\') if self.peek_second() == Some('"') => {
                                let esc_start = self.pos;
                                let esc_line = self.line;
                                let esc_col = self.col;
                                self.bump();
                                self.bump();
                                let span = self.span_from(esc_start, esc_line, esc_col);
                                return Err(self
                                    .err_at(
                                        "escaped `\\\"` is not valid inside an `f\"...\"` `{...}` expression hole; write unescaped `\"...\"` quotes",
                                        span,
                                    )
                                    .with_doc_topic("syntax")
                                    .with_hint(
                                        "inside `f\"...{...}\"`, write unescaped string quotes: e.g. `f\"hdr={\"ok\"}\"`",
                                    ));
                            }
                            Some('/') if self.peek_second() == Some('*') => {
                                let c_start = self.pos;
                                let c_line = self.line;
                                let c_col = self.col;
                                self.bump();
                                self.bump();
                                let mut c_depth = 1usize;
                                while c_depth > 0 {
                                    match self.bump() {
                                        Some('/') if self.peek() == Some('*') => {
                                            self.bump();
                                            c_depth += 1;
                                        }
                                        Some('*') if self.peek() == Some('/') => {
                                            self.bump();
                                            c_depth -= 1;
                                        }
                                        Some('\n') | None => {
                                            let span = self.span_from(c_start, c_line, c_col);
                                            return Err(self
                                                .err_at(
                                                    "unterminated `/* ... */` block comment inside format string hole",
                                                    span,
                                                )
                                                .with_doc_topic("syntax"));
                                        }
                                        Some(_) => {}
                                    }
                                }
                            }
                            Some('/') if self.peek_second() == Some('/') => {
                                let c_start = self.pos;
                                let c_line = self.line;
                                let c_col = self.col;
                                while let Some(ch) = self.peek() {
                                    if ch == '\n' {
                                        break;
                                    }
                                    self.bump();
                                }
                                let span = self.span_from(c_start, c_line, c_col);
                                return Err(self
                                    .err_at(
                                        "line comment `//` inside single-line format string hole comments out closing `}`",
                                        span,
                                    )
                                    .with_doc_topic("syntax")
                                    .with_hint("use a `/* ... */` block comment inside `{...}`"));
                            }
                            Some('{') => {
                                brace_depth += 1;
                                self.bump();
                            }
                            Some('}') => {
                                if brace_depth == 0 {
                                    hole_close_pos = self.pos;
                                    self.bump(); // consume closing `}`
                                    break;
                                } else {
                                    brace_depth -= 1;
                                    self.bump();
                                }
                            }
                            Some('(') => {
                                paren_depth += 1;
                                self.bump();
                            }
                            Some(')') => {
                                paren_depth = paren_depth.saturating_sub(1);
                                self.bump();
                            }
                            Some('[') => {
                                bracket_depth += 1;
                                self.bump();
                            }
                            Some(']') => {
                                bracket_depth = bracket_depth.saturating_sub(1);
                                self.bump();
                            }
                            Some(':')
                                if top_colon.is_none()
                                    && brace_depth == 0
                                    && paren_depth == 0
                                    && bracket_depth == 0 =>
                            {
                                if self.peek_second() == Some(':') {
                                    // `::` path separator (`a::b` or `clamp::<f32>`)
                                    self.bump();
                                    self.bump();
                                } else {
                                    let c_pos = self.pos;
                                    self.bump();
                                    top_colon = Some((c_pos, self.line, self.col));
                                }
                            }
                            Some(_) => {
                                self.bump();
                            }
                        }
                    }

                    let hole_span = self.span_from(hole_start, hole_line, hole_col);
                    let expr_end_pos = top_colon.map_or(hole_close_pos, |(c_pos, _, _)| c_pos);

                    if self.source[expr_start_pos..expr_end_pos].trim().is_empty() {
                        return Err(self
                            .err_at("empty `{}` interpolation hole in format string", hole_span)
                            .with_doc_topic("syntax"));
                    }

                    let spec = if let Some((c_pos, spec_line, spec_col)) = top_colon {
                        let spec_raw = &self.source[c_pos + 1..hole_close_pos];
                        let spec_span = Span::new(
                            (c_pos + 1) as u32,
                            hole_close_pos as u32,
                            spec_line,
                            spec_col,
                        );
                        self.parse_format_spec(spec_raw, spec_span)?
                    } else {
                        FormatSpec::Default
                    };

                    let mut sub_lexer = Lexer::sub_slice(
                        self.source,
                        self.filename,
                        expr_start_pos,
                        expr_end_pos,
                        expr_start_line,
                        expr_start_col,
                    );
                    let hole_tokens = sub_lexer.tokenize()?;
                    parts.push(LexedFormatPart::Hole {
                        tokens: hole_tokens,
                        spec,
                        span: hole_span,
                    });
                }
                Some(c) => {
                    self.bump();
                    cur_lit.push(c);
                }
            }
        }

        let span = self.span_from(start, start_line, start_col);
        Ok(Token::new(TokenKind::FormatString(parts), span))
    }

    fn parse_format_spec(&self, raw: &str, span: Span) -> Result<FormatSpec, ParseError> {
        let s = raw.trim();
        if s.is_empty() {
            return Err(self
                .err_at("empty format specifier after `:`", span)
                .with_doc_topic("syntax")
                .with_hint(
                    "supported specifiers: `:x`, `:X`, `:04x`, `:b`, `:08b`, `:.2`, `:e`, `:.2e`, `:?`",
                ));
        }

        if s == "?" {
            return Ok(FormatSpec::Debug);
        }

        if s == "e" || s == "E" {
            return Ok(FormatSpec::Sci { precision: None });
        }

        if let Some(rest) = s.strip_prefix('.') {
            if let Some(digits) = rest.strip_suffix('e').or_else(|| rest.strip_suffix('E')) {
                if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
                    let p = digits.parse::<u8>().map_err(|_| {
                        self.err_at(format!("format precision `{digits}` out of `u8` range"), span)
                            .with_doc_topic("syntax")
                    })?;
                    return Ok(FormatSpec::Sci { precision: Some(p) });
                }
            } else {
                let digits = rest.strip_suffix('f').unwrap_or(rest);
                if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
                    let p = digits.parse::<u8>().map_err(|_| {
                        self.err_at(format!("format precision `{digits}` out of `u8` range"), span)
                            .with_doc_topic("syntax")
                    })?;
                    return Ok(FormatSpec::Fixed { precision: p });
                }
            }
        }

        if let Some(prefix) = s.strip_suffix('x').or_else(|| s.strip_suffix('X')) {
            let upper = s.ends_with('X');
            let suffix_ch = if upper { 'X' } else { 'x' };
            if prefix.is_empty() {
                return Ok(FormatSpec::Hex { upper, width: None });
            }
            if prefix.chars().all(|c| c.is_ascii_digit()) {
                if !prefix.starts_with('0') {
                    return Err(self
                        .err_at(
                            format!("space-padded format specifier `:{s}` is not supported; use zero-padded `:0{s}`"),
                            span,
                        )
                        .with_doc_topic("syntax")
                        .with_hint(format!("write `:0{s}` for zero-padded hex formatting")));
                }
                let w = prefix.parse::<u8>().map_err(|_| {
                    self.err_at(format!("hex width `{prefix}` out of `u8` range"), span)
                        .with_doc_topic("syntax")
                })?;
                if prefix.len() < 2 || w == 0 {
                    return Err(self
                        .err_at(
                            format!("invalid format specifier `:{s}` (zero-pad width must be >= 1)"),
                            span,
                        )
                        .with_doc_topic("syntax")
                        .with_hint(format!(
                            "write `0x{{val:{suffix_ch}}}` for a literal `0x` prefix, or `:02{suffix_ch}` / `:04{suffix_ch}` to zero-pad"
                        )));
                }
                return Ok(FormatSpec::Hex {
                    upper,
                    width: Some(w),
                });
            }
        }

        if let Some(prefix) = s.strip_suffix('b') {
            if prefix.is_empty() {
                return Ok(FormatSpec::Bin { width: None });
            }
            if prefix.chars().all(|c| c.is_ascii_digit()) {
                if !prefix.starts_with('0') {
                    return Err(self
                        .err_at(
                            format!("space-padded format specifier `:{s}` is not supported; use zero-padded `:0{s}`"),
                            span,
                        )
                        .with_doc_topic("syntax")
                        .with_hint(format!("write `:0{s}` for zero-padded binary formatting")));
                }
                let w = prefix.parse::<u8>().map_err(|_| {
                    self.err_at(format!("binary width `{prefix}` out of `u8` range"), span)
                        .with_doc_topic("syntax")
                })?;
                if prefix.len() < 2 || w == 0 {
                    return Err(self
                        .err_at(
                            format!("invalid format specifier `:{s}` (zero-pad width must be >= 1)"),
                            span,
                        )
                        .with_doc_topic("syntax")
                        .with_hint(
                            "write `0b{val:b}` for a literal `0b` prefix, or `:08b` to zero-pad",
                        ));
                }
                return Ok(FormatSpec::Bin { width: Some(w) });
            }
        }

        Err(self
            .err_at(format!("unsupported format specifier `:{s}`"), span)
            .with_doc_topic("syntax")
            .with_hint(
                "supported specifiers: `:x`, `:X`, `:04x`, `:b`, `:08b`, `:.2`, `:e`, `:.2e`, `:?`",
            ))
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

        let byte_or_char: u64 = match self.bump() {
            Some('\\') => match self.lex_escape(start, start_line, start_col)? {
                EscapeVal::Char(c) => c as u64,
                EscapeVal::RawByte(b) => b as u64,
            },
            Some('\'') => {
                let span = self.span_from(start, start_line, start_col);
                return Err(self
                    .err_at("empty character literal", span)
                    .with_doc_topic("syntax"));
            }
            Some(c) => c as u64,
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
        Ok(Token::new(TokenKind::Int(byte_or_char), span))
    }

    fn lex_escape(
        &mut self,
        start: usize,
        start_line: u32,
        start_col: u32,
    ) -> Result<EscapeVal, ParseError> {
        match self.bump() {
            Some('n') => Ok(EscapeVal::Char('\n')),
            Some('r') => Ok(EscapeVal::Char('\r')),
            Some('t') => Ok(EscapeVal::Char('\t')),
            Some('0') => Ok(EscapeVal::Char('\0')),
            Some('\\') => Ok(EscapeVal::Char('\\')),
            Some('"') => Ok(EscapeVal::Char('"')),
            Some('\'') => Ok(EscapeVal::Char('\'')),
            Some('x') => {
                let h1 = self.bump().and_then(|c| c.to_digit(16));
                let h2 = self.bump().and_then(|c| c.to_digit(16));
                match (h1, h2) {
                    (Some(a), Some(b)) => Ok(EscapeVal::RawByte(((a * 16) + b) as u8)),
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum EscapeVal {
    Char(char),
    RawByte(u8),
}

fn push_escape(dst: &mut String, esc: EscapeVal) {
    match esc {
        EscapeVal::Char(c) => dst.push(c),
        EscapeVal::RawByte(b) => zev_ast::push_raw_byte_escape(dst, b),
    }
}

fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn is_ident_continue(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphanumeric()
}
