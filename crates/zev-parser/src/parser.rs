use std::collections::BTreeSet;

use crate::error::ParseError;
use crate::token::{LexedFormatPart, Token, TokenKind};
use zev_ast::*;

#[derive(Copy, Clone)]
struct Checkpoint {
    pos: usize,
    pending_gt: Option<Span>,
    allow_struct_lit: bool,
}

pub struct Parser<'a> {
    source: &'a str,
    filename: Option<&'a str>,
    tokens: Vec<Token>,
    pos: usize,
    /// When a `>>` (`TokenKind::Shr`) token is split to close an outer generic `<...>`,
    /// this stores the second `>` span to be consumed next.
    pending_gt: Option<Span>,
    /// Disambiguates `Foo { a: 1 }` struct literals vs `{ ... }` body blocks after unparenthesized conditions.
    allow_struct_lit: bool,
    /// Optional whole-program set of declared generic `struct` and `fn` names across all input files.
    known_generics: Option<&'a BTreeSet<String>>,
    /// Top-level generic `struct` and `fn` names declared in this file.
    local_generics: BTreeSet<String>,
    /// Top-level value bindings (`const`, OS primitives, and global variables) declared in this file.
    top_level_values: BTreeSet<String>,
    /// Module alias names bound by `import a.b as alias;` (or `import m;`).
    import_aliases: BTreeSet<String>,
    /// Active lexical value scopes inside `fn` / `task` / `isr` bodies.
    value_scopes: Vec<BTreeSet<String>>,
}

impl<'a> Parser<'a> {
    pub fn new(source: &'a str, filename: Option<&'a str>, tokens: Vec<Token>) -> Self {
        Self::new_with_generics(source, filename, tokens, None)
    }

    pub fn new_with_generics(
        source: &'a str,
        filename: Option<&'a str>,
        tokens: Vec<Token>,
        known_generics: Option<&'a BTreeSet<String>>,
    ) -> Self {
        let mut p = Self {
            source,
            filename,
            tokens,
            pos: 0,
            pending_gt: None,
            allow_struct_lit: true,
            known_generics,
            local_generics: BTreeSet::new(),
            top_level_values: BTreeSet::new(),
            import_aliases: BTreeSet::new(),
            value_scopes: Vec::new(),
        };
        p.prescan_top_level();
        p
    }

    fn new_hole_subparser(&self, tokens: Vec<Token>) -> Parser<'a> {
        Parser {
            source: self.source,
            filename: self.filename,
            tokens,
            pos: 0,
            pending_gt: None,
            allow_struct_lit: true,
            known_generics: self.known_generics,
            local_generics: self.local_generics.clone(),
            top_level_values: self.top_level_values.clone(),
            import_aliases: self.import_aliases.clone(),
            value_scopes: self.value_scopes.clone(),
        }
    }

    pub(crate) fn collect_declared_generics(source: &str, tokens: Vec<Token>) -> BTreeSet<String> {
        let p = Parser::new(source, None, tokens);
        p.local_generics
    }

    fn prescan_top_level(&mut self) {
        let saved = self.save();
        self.pos = 0;
        self.pending_gt = None;
        self.allow_struct_lit = true;

        while !self.at(&TokenKind::Eof) {
            let iter_start = self.pos;
            let _ = self.parse_attrs();
            while self.eat(&TokenKind::KwPub) || self.eat(&TokenKind::KwStatic) {}
            let _ = self.parse_attrs();

            match self.peek_kind() {
                TokenKind::Eof => break,
                TokenKind::KwImport => {
                    if let Ok(imp) = self.parse_import() {
                        if let Some(alias) = imp.alias {
                            self.import_aliases.insert(alias.name);
                        } else if let Some(last) = imp.path.last() {
                            self.import_aliases.insert(last.name.clone());
                        }
                    } else {
                        self.prescan_skip_until_semi_depth0();
                    }
                }
                TokenKind::Ident(ref kw) if kw == "mmio" && self.is_mmio_decl_start() => {
                    self.bump();
                    if let TokenKind::Ident(name) = self.peek_kind() {
                        self.bump();
                        if !self.at(&TokenKind::LBrace) {
                            self.top_level_values.insert(name);
                        }
                    }
                    self.prescan_skip_item();
                }
                TokenKind::Ident(ref kw) if kw == "irq" && self.is_irq_decl_start() => {
                    self.bump();
                    if let TokenKind::Ident(name) = self.peek_kind() {
                        self.bump();
                        self.top_level_values.insert(name);
                    }
                    self.prescan_skip_until_semi_depth0();
                }
                TokenKind::KwPacked | TokenKind::KwStruct => {
                    let _ = self.eat(&TokenKind::KwPacked);
                    let _ = self.eat(&TokenKind::KwStruct);
                    if let TokenKind::Ident(name) = self.peek_kind() {
                        self.bump();
                        if self.at(&TokenKind::Lt) {
                            self.local_generics.insert(name);
                        }
                    }
                    self.prescan_skip_item();
                }
                TokenKind::KwEnum | TokenKind::KwTask | TokenKind::KwIsr => {
                    self.prescan_skip_item();
                }
                TokenKind::KwConst => {
                    self.bump();
                    if let Ok((_, name)) = self.parse_field_or_param("prescan") {
                        self.top_level_values.insert(name.name);
                    }
                    self.prescan_skip_until_semi_depth0();
                }
                TokenKind::KwExtern => {
                    self.bump();
                    if matches!(self.peek_kind(), TokenKind::String(_)) {
                        self.bump();
                    }
                    self.prescan_skip_item();
                }
                TokenKind::KwFn => {
                    self.bump();
                    let is_std = matches!(self.peek_kind(), TokenKind::Ident(_))
                        && matches!(self.peek_n_kind(1), TokenKind::LParen | TokenKind::Lt);
                    if is_std {
                        if let TokenKind::Ident(name) = self.bump().kind {
                            if self.at(&TokenKind::Lt) {
                                self.local_generics.insert(name);
                            }
                        }
                    } else if let Ok((_, name)) = self.parse_typed_declarator("prescan") {
                        if self.at(&TokenKind::Lt) {
                            self.local_generics.insert(name.name);
                        }
                    }
                    self.prescan_skip_item();
                }
                _ => {
                    if let Ok((_, name)) = self.parse_typed_declarator("prescan") {
                        if self.at(&TokenKind::Lt) {
                            self.local_generics.insert(name.name);
                            self.prescan_skip_item();
                        } else if self.at(&TokenKind::LParen) {
                            self.prescan_skip_item();
                        } else {
                            self.top_level_values.insert(name.name);
                            self.prescan_skip_until_semi_depth0();
                        }
                    } else {
                        self.prescan_skip_item();
                    }
                }
            }

            if self.pos <= iter_start && !self.at(&TokenKind::Eof) {
                self.bump();
            }
        }

        self.restore(saved);
    }

    fn prescan_skip_until_semi_depth0(&mut self) {
        self.pending_gt = None;
        let mut brace_depth = 0usize;
        while !self.at(&TokenKind::Eof) {
            match self.bump().kind {
                TokenKind::LBrace => {
                    brace_depth += 1;
                }
                TokenKind::RBrace => {
                    if brace_depth == 0 {
                        break;
                    }
                    brace_depth -= 1;
                }
                TokenKind::Semi if brace_depth == 0 => {
                    break;
                }
                _ => {}
            }
        }
    }

    fn prescan_skip_item(&mut self) {
        self.pending_gt = None;
        let mut brace_depth = 0usize;
        while !self.at(&TokenKind::Eof) {
            match self.bump().kind {
                TokenKind::LBrace => {
                    brace_depth += 1;
                }
                TokenKind::RBrace => {
                    if brace_depth <= 1 {
                        let _ = self.eat(&TokenKind::Semi);
                        break;
                    }
                    brace_depth -= 1;
                }
                TokenKind::Semi if brace_depth == 0 => {
                    break;
                }
                _ => {}
            }
        }
    }

    fn push_value_scope(&mut self) {
        self.value_scopes.push(BTreeSet::new());
    }

    fn pop_value_scope(&mut self) {
        let _ = self.value_scopes.pop();
    }

    fn record_local_value(&mut self, name: &str) {
        if let Some(scope) = self.value_scopes.last_mut() {
            scope.insert(name.to_owned());
        }
    }

    fn is_value_name(&self, name: &str) -> bool {
        self.top_level_values.contains(name)
            || self.value_scopes.iter().rev().any(|s| s.contains(name))
    }

    fn can_speculate_bare_generic_call(&self, name: &str) -> bool {
        if self.is_value_name(name) {
            return false;
        }
        if let Some(known) = self.known_generics {
            return known.contains(name);
        }
        true
    }

    fn parse_speculative_generic_args(&mut self) -> Option<Vec<GenericArg>> {
        let Ok((args, _)) = self.parse_generic_args() else {
            return None;
        };
        if self.pending_gt.is_some() {
            return None;
        }
        // A bare identifier in local `value_scopes` (e.g. `b` in `a < b, c > (d + 1)`)
        // cannot be a valid compile-time generic type or constant parameter (`E0029`).
        for arg in &args {
            if let GenericArg::Type(te) = arg {
                if let TypeKind::Named(id) = &te.kind {
                    if self.value_scopes.iter().rev().any(|s| s.contains(&id.name)) {
                        return None;
                    }
                }
            }
        }
        Some(args)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Token Cursor Helpers
    // ─────────────────────────────────────────────────────────────────────────

    fn save(&self) -> Checkpoint {
        Checkpoint {
            pos: self.pos,
            pending_gt: self.pending_gt,
            allow_struct_lit: self.allow_struct_lit,
        }
    }

    fn restore(&mut self, cp: Checkpoint) {
        self.pos = cp.pos;
        self.pending_gt = cp.pending_gt;
        self.allow_struct_lit = cp.allow_struct_lit;
    }

    fn peek_token(&self) -> Token {
        if let Some(span) = self.pending_gt {
            return Token::new(TokenKind::Gt, span);
        }
        self.tokens
            .get(self.pos)
            .cloned()
            .unwrap_or_else(|| Token::new(TokenKind::Eof, Span::dummy()))
    }

    fn peek_kind(&self) -> TokenKind {
        if self.pending_gt.is_some() {
            return TokenKind::Gt;
        }
        self.tokens
            .get(self.pos)
            .map(|t| t.kind.clone())
            .unwrap_or(TokenKind::Eof)
    }

    fn peek_span(&self) -> Span {
        if let Some(span) = self.pending_gt {
            return span;
        }
        self.tokens
            .get(self.pos)
            .map(|t| t.span)
            .unwrap_or_else(Span::dummy)
    }

    fn peek_n_kind(&self, offset: usize) -> TokenKind {
        if offset == 0 {
            return self.peek_kind();
        }
        let adjusted = if self.pending_gt.is_some() {
            self.pos + offset - 1
        } else {
            self.pos + offset
        };
        self.tokens
            .get(adjusted)
            .map(|t| t.kind.clone())
            .unwrap_or(TokenKind::Eof)
    }

    fn peek_n_span(&self, offset: usize) -> Span {
        if offset == 0 {
            return self.peek_span();
        }
        let adjusted = if self.pending_gt.is_some() {
            self.pos + offset - 1
        } else {
            self.pos + offset
        };
        self.tokens
            .get(adjusted)
            .map(|t| t.span)
            .unwrap_or_else(Span::dummy)
    }

    fn prev_span(&self) -> Span {
        if self.pending_gt.is_some() {
            return self
                .tokens
                .get(self.pos.saturating_sub(1))
                .map(|t| Span::new(t.span.start, t.span.start + 1, t.span.line, t.span.col))
                .unwrap_or_else(Span::dummy);
        }
        self.tokens
            .get(self.pos.saturating_sub(1))
            .map(|t| t.span)
            .unwrap_or_else(Span::dummy)
    }

    fn bump(&mut self) -> Token {
        if let Some(span) = self.pending_gt.take() {
            return Token::new(TokenKind::Gt, span);
        }
        let tok = self.peek_token();
        if self.pos < self.tokens.len() && tok.kind != TokenKind::Eof {
            self.pos += 1;
        }
        tok
    }

    fn at(&self, kind: &TokenKind) -> bool {
        &self.peek_kind() == kind
    }

    fn eat(&mut self, kind: &TokenKind) -> bool {
        if self.at(kind) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, expected: &TokenKind) -> Result<Token, ParseError> {
        let tok = self.peek_token();
        if &tok.kind == expected {
            Ok(self.bump())
        } else {
            // Check if the unexpected token is actually a foreign-idiom trap!
            if let Some(trap_err) = self.check_unexpected_token_trap(&tok) {
                return Err(trap_err);
            }
            Err(self
                .err_at(format!("expected {expected}, found {}", tok.kind), tok.span)
                .with_doc_topic("syntax"))
        }
    }

    /// Consume a closing `>` inside generic type arguments (`<...>`), splitting `>>` if needed.
    fn expect_type_gt(&mut self) -> Result<Span, ParseError> {
        if let Some(span) = self.pending_gt.take() {
            return Ok(span);
        }
        let tok = self.peek_token();
        match tok.kind {
            TokenKind::Gt => {
                self.bump();
                Ok(tok.span)
            }
            TokenKind::Shr => {
                self.bump();
                let first_gt = Span::new(
                    tok.span.start,
                    tok.span.start + 1,
                    tok.span.line,
                    tok.span.col,
                );
                let second_gt = Span::new(
                    tok.span.start + 1,
                    tok.span.end,
                    tok.span.line,
                    tok.span.col + 1,
                );
                self.pending_gt = Some(second_gt);
                Ok(first_gt)
            }
            TokenKind::Lifetime(ref lt) => Err(self.trap_p0009_lifetime(lt, tok.span)),
            _ => Err(self
                .err_at(format!("expected `>`, found {}", tok.kind), tok.span)
                .with_doc_topic("syntax")),
        }
    }

    fn err_at(&self, message: impl Into<String>, span: Span) -> ParseError {
        ParseError::new(message, span, self.source, self.filename)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Foreign-Idiom Trap Constructors (`P0001`..`P0010`)
    // ─────────────────────────────────────────────────────────────────────────

    fn trap_p0001(&self, kw: &str, span: Span) -> ParseError {
        self.err_at(
            format!(
                "`{kw}` variable bindings are not used in Zev; declare variables with explicit C-style types"
            ),
            span,
        )
        .with_code("P0001")
        .with_doc_topic("syntax")
        .with_hint(
            "write `u16 x = 42;` instead of `let x = 42;` or `var x: u16 = 42;` (locals are mutable in their task/fn; top-level constants use `const u32 LIMIT = 100;`)",
        )
    }

    fn trap_p0002_colon_binding(&self, name: &str, ty_hint: Option<&str>, span: Span) -> ParseError {
        let rewrite = match ty_hint {
            Some(t) => format!(
                "write C-style `{t} {name}` instead of `{name}: {t}`; for value-returning functions put the return type before the name: `{t} foo({t} {name}) {{ ... }}` (`fn name(...)` is only for `void` functions)"
            ),
            None => format!(
                "write C-style `Type {name}` instead of `{name}: Type`; for value-returning functions put the return type before the name: `u16 foo(u16 {name}) {{ ... }}` (`fn name(...)` is only for `void` functions)"
            ),
        };
        self.err_at(
            format!("Rust/Zig/Go `{name}: Type` syntax is not valid in Zev"),
            span,
        )
        .with_code("P0002")
        .with_doc_topic("syntax")
        .with_hint(rewrite)
    }

    fn trap_p0002_fn_arrow(&self, fn_name: &str, ret_hint: Option<&str>, span: Span) -> ParseError {
        let hint = match ret_hint {
            Some("void") => format!(
                "remove `-> void`; `fn {fn_name}(...) {{ ... }}` already returns `void`"
            ),
            Some(rt) => format!(
                "put the return type before the function name without `fn`: `{rt} {fn_name}(...) {{ ... }}` (`fn name(...)` is only for `void`-returning functions with C-style `Type name` parameters)"
            ),
            None => format!(
                "put the return type before the function name without `fn`: `u16 {fn_name}(u16 x) {{ ... }}` (`fn name(...)` is only for `void`-returning functions with C-style `Type name` parameters)"
            ),
        };
        self.err_at(
            format!("`-> RetType` after `fn {fn_name}(...)` is not valid in Zev"),
            span,
        )
        .with_code("P0002")
        .with_doc_topic("syntax")
        .with_hint(hint)
    }

    fn trap_p0003(&self, which: &str, span: Span) -> ParseError {
        self.err_at(
            format!(
                "`{which}` is not used in Zev; cooperative yielding is inferred automatically across the call graph"
            ),
            span,
        )
        .with_code("P0003")
        .with_doc_topic("yielding")
        .with_hint(
            "remove `async` and `.await` (e.g. write `u16 rpm = adc_read_rpm(ch);`); use `noyield { ... }` when you need to forbid yielding",
        )
    }

    fn trap_p0004(&self, span: Span) -> ParseError {
        self.err_at(
            "`select` multi-wait is omitted in Zev to keep task wait state O(1) and sequential",
            span,
        )
        .with_code("P0004")
        .with_doc_topic("primitives")
        .with_hint(
            "pull typed events from an `inbox<Ev, 8>` followed by `switch (ev) { ... }`, or use sequential timeouts: `Ev ev = inbox.recv(timeout: 50ms) or Fallback;`",
        )
    }

    fn trap_p0005(&self, span: Span) -> ParseError {
        self.err_at("`match` is spelled `switch` in Zev", span)
            .with_code("P0005")
            .with_doc_topic("syntax")
            .with_hint("rewrite as `switch (expr) { Variant => { ... } }`")
    }

    fn trap_p0006(&self, span: Span) -> ParseError {
        self.err_at(
            "Zig-style `catch |err|` pipe capture is not valid in Zev",
            span,
        )
        .with_code("P0006")
        .with_doc_topic("errors")
        .with_hint(
            "use C-style typed parentheses: `catch (ParseErr err) { ... }`, `catch (IoErr err, usize partial) { ... }`, or bare `catch { ... }`",
        )
    }

    fn trap_p0007(&self, construct: &str, span: Span) -> ParseError {
        self.err_at(
            format!(
                "{construct} is not valid in Zev; Zev uses explicit dual-exit `return` vs `error` paths"
            ),
            span,
        )
        .with_code("P0007")
        .with_doc_topic("errors")
        .with_hint(
            "declare fallible functions with `u16 parse([]u8 buf) error ParseErr { ... }`, fail via `error BadCrc;` (or `error;` / `error Timeout, count;`), and handle with `catch (ParseErr err) { ... }` or `or fallback`",
        )
    }

    fn trap_p0008(&self, span: Span) -> ParseError {
        self.err_at(
            "C-style `for (;;)` and untyped `for` loops are not supported; use bounded `for (usize i in 0..n) { ... }`, `for (u8 b in slice) { ... }`, `while (cond)`, or `loop`",
            span,
        )
        .with_code("P0008")
        .with_doc_topic("syntax")
        .with_hint(
            "write an explicit loop variable type with `in`: `for (usize i in 0..n) { ... }`, `for (u32 i in 0..=last) { ... }`, or `for (u8 b in slice) { ... }`",
        )
    }

    fn trap_p0011(&self, span: Span, period_hint: Option<&str>) -> ParseError {
        let example = match period_hint.filter(|s| !s.is_empty()) {
            Some(p) => format!(
                "replace `every {p} {{ ... }}` with `loop every {p} {{ ... }}` (periodic loops also accept runtime expressions like `loop every period_ms {{ ... }}`)"
            ),
            None => "replace bare `every <period> { ... }` with `loop every 20ms { ... }` or `loop every period_expr { ... }`".to_owned(),
        };
        self.err_at(
            "bare `every` loops are deprecated/ambiguous; write `loop every 20ms { ... }` so sequential loop control flow is explicit",
            span,
        )
        .with_code("P0011")
        .with_doc_topic("tasks")
        .with_hint(example)
    }

    fn trap_p0012(&self, span: Span) -> ParseError {
        self.err_at(
            "`chan<T, N>` has been unified into `inbox<T, N>` (which supports both `.post()` and backpressured `.send()`); replace `chan<...>` with `inbox<...>`",
            span,
        )
        .with_code("P0012")
        .with_doc_topic("primitives")
        .with_hint(
            "declare `inbox<T, N> name;` instead of `chan<T, N> name;` (use `.post(v)` from ISRs/tasks, yielding `.send(v)` from tasks, and `.recv()` / `.try_recv()` to consume)",
        )
    }

    fn trap_p0009_lifetime(&self, lt: &str, span: Span) -> ParseError {
        self.err_at(
            format!(
                "Rust-style lifetime `'{lt}` is not used in Zev; scope depth and DMA leases are tracked automatically"
            ),
            span,
        )
        .with_code("P0009")
        .with_doc_topic("dma")
        .with_hint(
            "remove `'a` lifetime annotations and pass `*T` / `&x` pointers and `[]T` slices directly (e.g. `struct ByteReader { []u8 buf; }`)",
        )
    }

    fn trap_p0009_mut_or_ref(&self, what: &str, span: Span) -> ParseError {
        self.err_at(
            format!(
                "`{what}` is not used in Zev; Zev has no `'a` lifetimes or `&mut` borrow syntax"
            ),
            span,
        )
        .with_code("P0009")
        .with_doc_topic("dma")
        .with_hint(
            "declare pointer parameters as `T *p` or `*T p`, slices as `[]T`, and pass `&x` at call sites (scope depth and DMA leases are verified automatically)",
        )
    }

    fn trap_p0010(&self, span: Span) -> ParseError {
        self.err_at("`static` variable modifier is not used in Zev", span)
            .with_code("P0010")
            .with_doc_topic("primitives")
            .with_hint(
                "task locals declared before `loop`/`loop every` (e.g. `i32 integral = 0;`) persist in static task frames automatically, persistent `isr` state uses `state Type x = init;` inside `isr`, top-level constants use `const`, and shared task/ISR state uses `watch<T>`, `inbox<T, N>`, or `signal`",
            )
    }

    fn check_unexpected_token_trap(&self, tok: &Token) -> Option<ParseError> {
        match &tok.kind {
            TokenKind::Lifetime(lt) => Some(self.trap_p0009_lifetime(lt, tok.span)),
            TokenKind::KwLet => Some(self.trap_p0001("let", tok.span)),
            TokenKind::KwVar => Some(self.trap_p0001("var", tok.span)),
            TokenKind::KwMut => Some(self.trap_p0001("mut", tok.span)),
            TokenKind::KwAsync => Some(self.trap_p0003("async", tok.span)),
            TokenKind::KwAwait => Some(self.trap_p0003("await", tok.span)),
            TokenKind::KwSelect => Some(self.trap_p0004(tok.span)),
            TokenKind::KwMatch => Some(self.trap_p0005(tok.span)),
            TokenKind::KwTry => Some(self.trap_p0007("`try`", tok.span)),
            TokenKind::KwFor => Some(self.trap_p0008(tok.span)),
            TokenKind::KwEvery => Some(self.trap_p0011(tok.span, None)),
            TokenKind::KwStatic => Some(self.trap_p0010(tok.span)),
            TokenKind::Question => Some(self.trap_p0007("`?` operator", tok.span)),
            _ => None,
        }
    }

    fn check_generic_lifetime_after_decl_name(&self) -> Result<(), ParseError> {
        if self.at(&TokenKind::Lt) {
            if let TokenKind::Lifetime(lt) = self.peek_n_kind(1) {
                return Err(self.trap_p0009_lifetime(&lt, self.peek_n_span(1)));
            }
        }
        Ok(())
    }

    fn parse_ident(&mut self) -> Result<Ident, ParseError> {
        let tok = self.peek_token();
        match tok.kind {
            TokenKind::Ident(name) => {
                self.bump();
                Ok(Ident::new(name, tok.span))
            }
            _ => {
                if let Some(trap_err) = self.check_unexpected_token_trap(&tok) {
                    return Err(trap_err);
                }
                Err(self
                    .err_at(
                        format!("expected identifier, found {}", tok.kind),
                        tok.span,
                    )
                    .with_doc_topic("syntax"))
            }
        }
    }

    /// Parse a word after `.` or `@` where keywords like `critical` (`@critical`) may appear.
    fn parse_word_ident(&mut self, context: &str) -> Result<Ident, ParseError> {
        let tok = self.peek_token();
        if let Some(w) = tok.kind.as_word() {
            let name = w.to_owned();
            self.bump();
            Ok(Ident::new(name, tok.span))
        } else {
            if let Some(trap_err) = self.check_unexpected_token_trap(&tok) {
                return Err(trap_err);
            }
            Err(self
                .err_at(format!("expected {context}, found {}", tok.kind), tok.span)
                .with_doc_topic("syntax"))
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Program & Top-Level Items
    // ─────────────────────────────────────────────────────────────────────────

    pub fn parse_program(&mut self) -> Result<Program, ParseError> {
        let start_span = self.peek_span();
        let mut items = Vec::new();

        while !self.at(&TokenKind::Eof) {
            self.parse_top_level_items(&mut items)?;
        }

        let end_span = if items.is_empty() {
            start_span
        } else {
            start_span.merge(self.prev_span())
        };
        Ok(Program::new(items, end_span))
    }

    fn is_mmio_decl_start(&self) -> bool {
        matches!(self.peek_kind(), TokenKind::Ident(ref s) if s == "mmio")
            && (self.peek_n_kind(1) == TokenKind::LBrace
                || (matches!(self.peek_n_kind(1), TokenKind::Ident(_))
                    && !matches!(
                        self.peek_n_kind(2),
                        TokenKind::LParen
                            | TokenKind::Lt
                            | TokenKind::Eq
                            | TokenKind::Semi
                            | TokenKind::LBracket
                    )))
    }

    fn is_irq_decl_start(&self) -> bool {
        matches!(self.peek_kind(), TokenKind::Ident(ref s) if s == "irq")
            && matches!(self.peek_n_kind(1), TokenKind::Ident(_))
            && !matches!(
                self.peek_n_kind(2),
                TokenKind::LParen | TokenKind::Lt | TokenKind::LBracket
            )
    }

    fn parse_top_level_items(&mut self, out: &mut Vec<Item>) -> Result<(), ParseError> {
        let first_span = self.peek_span();
        let mut attrs = self.parse_attrs()?;
        let pub_span = if self.at(&TokenKind::KwPub) {
            Some(self.bump().span)
        } else {
            None
        };
        attrs.extend(self.parse_attrs()?);
        if self.at(&TokenKind::KwPub) {
            return Err(self
                .err_at("duplicate `pub` visibility modifier", self.peek_span())
                .with_doc_topic("syntax"));
        }
        let is_pub = pub_span.is_some();
        let cur_span = self.peek_span();

        match self.peek_kind() {
            // ── Foreign-idiom top-level traps ──
            TokenKind::KwStatic => return Err(self.trap_p0010(cur_span)),
            TokenKind::KwLet => return Err(self.trap_p0001("let", cur_span)),
            TokenKind::KwVar => return Err(self.trap_p0001("var", cur_span)),
            TokenKind::KwMut => return Err(self.trap_p0001("mut", cur_span)),
            TokenKind::KwAsync => return Err(self.trap_p0003("async", cur_span)),
            TokenKind::KwSelect => return Err(self.trap_p0004(cur_span)),
            TokenKind::KwMatch => return Err(self.trap_p0005(cur_span)),
            TokenKind::KwFor => return Err(self.trap_p0008(cur_span)),
            TokenKind::KwEvery => return Err(self.trap_p0011(cur_span, None)),
            TokenKind::Bang => {
                return Err(self.trap_p0007("Zig-style `!Type` error-union return type", cur_span));
            }
            TokenKind::Question => {
                return Err(self.trap_p0007("Zig-style `?Type` optional/error return type", cur_span));
            }
            TokenKind::Lifetime(ref lt) => {
                return Err(self.trap_p0009_lifetime(lt, cur_span));
            }

            // ── Valid Zev top-level constructs ──
            TokenKind::KwImport => {
                if let Some(ps) = pub_span {
                    return Err(self
                        .err_at(
                            "`pub` visibility modifier is not allowed on `import` declarations",
                            ps,
                        )
                        .with_doc_topic("syntax"));
                }
                if !attrs.is_empty() {
                    return Err(self
                        .err_at(
                            "attributes are not allowed on `import` declarations",
                            attrs[0].span,
                        )
                        .with_doc_topic("syntax"));
                }
                let imp = self.parse_import()?;
                let span = imp.span;
                out.push(Item::new(attrs, ItemKind::Import(imp), span));
            }
            TokenKind::KwPacked | TokenKind::KwStruct => {
                let mut st = self.parse_struct()?;
                if attrs
                    .iter()
                    .any(|a| matches!(&a.kind, AttrKind::Custom { name, .. } if name == "packed"))
                {
                    st.is_packed = true;
                }
                let span = first_span.merge(st.span);
                st.span = span;
                out.push(Item::with_pub(attrs, is_pub, ItemKind::Struct(st), span));
            }
            TokenKind::KwEnum => {
                let en = self.parse_enum()?;
                let span = first_span.merge(en.span);
                out.push(Item::with_pub(attrs, is_pub, ItemKind::Enum(en), span));
            }
            TokenKind::KwIsr => {
                let isr = self.parse_isr(attrs.clone(), first_span)?;
                let span = isr.span;
                out.push(Item::with_pub(attrs, is_pub, ItemKind::Isr(isr), span));
            }
            TokenKind::KwTask => {
                let task = self.parse_task(attrs.clone(), first_span)?;
                let span = task.span;
                out.push(Item::with_pub(attrs, is_pub, ItemKind::Task(task), span));
            }
            TokenKind::KwConst => {
                let c = self.parse_const(first_span)?;
                let span = c.span;
                out.push(Item::with_pub(attrs, is_pub, ItemKind::Const(c), span));
            }
            TokenKind::KwExtern => {
                self.parse_extern_items(attrs, is_pub, first_span, out)?;
            }
            TokenKind::KwFn => {
                let f = self.parse_kw_fn(attrs.clone(), false, first_span)?;
                let span = f.span;
                out.push(Item::with_pub(attrs, is_pub, ItemKind::Fn(f), span));
            }
            TokenKind::Ident(ref kw) if kw == "mmio" && self.is_mmio_decl_start() => {
                let mmio = self.parse_mmio(first_span)?;
                let span = mmio.span;
                out.push(Item::with_pub(attrs, is_pub, ItemKind::Mmio(mmio), span));
            }
            TokenKind::Ident(ref kw) if kw == "irq" && self.is_irq_decl_start() => {
                let irq = self.parse_irq(first_span)?;
                let span = irq.span;
                out.push(Item::with_pub(attrs, is_pub, ItemKind::Irq(irq), span));
            }
            _ => {
                // Catch `ErrSet!RetType func(...)` at top-level (`P0007`)
                if matches!(self.peek_kind(), TokenKind::Ident(_))
                    && self.peek_n_kind(1) == TokenKind::Bang
                {
                    return Err(self.trap_p0007(
                        "Zig-style `ErrType!RetType` error-union return type",
                        cur_span.merge(self.peek_n_span(1)),
                    ));
                }

                let (ty, name) = self.parse_typed_declarator("top-level declaration")?;
                self.check_generic_lifetime_after_decl_name()?;

                if self.at(&TokenKind::LParen) || self.at(&TokenKind::Lt) {
                    let f = self.finish_fn_after_name(attrs.clone(), false, ty, name, first_span)?;
                    let span = f.span;
                    out.push(Item::with_pub(attrs, is_pub, ItemKind::Fn(f), span));
                } else {
                    let init = if self.eat(&TokenKind::Eq) {
                        Some(self.parse_expr()?)
                    } else {
                        None
                    };
                    let semi = self.expect(&TokenKind::Semi)?;
                    let span = first_span.merge(semi.span);
                    match ty.kind {
                        TypeKind::OsPrimitive(os_ty) => {
                            out.push(Item::with_pub(
                                attrs.clone(),
                                is_pub,
                                ItemKind::OsPrimitive(OsPrimitiveDecl {
                                    attrs,
                                    ty: os_ty,
                                    name,
                                    init,
                                    span,
                                }),
                                span,
                            ));
                        }
                        _ => {
                            out.push(Item::with_pub(
                                attrs,
                                is_pub,
                                ItemKind::GlobalVar(GlobalVarDecl {
                                    ty,
                                    name,
                                    init,
                                    span,
                                }),
                                span,
                            ));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Attributes & Priorities
    // ─────────────────────────────────────────────────────────────────────────

    fn parse_attrs(&mut self) -> Result<Vec<Attr>, ParseError> {
        let mut attrs = Vec::new();
        while self.at(&TokenKind::At) {
            attrs.push(self.parse_attr()?);
        }
        Ok(attrs)
    }

    fn parse_attr(&mut self) -> Result<Attr, ParseError> {
        let at_tok = self.expect(&TokenKind::At)?;
        let word = self.parse_word_ident("attribute name after `@`")?;
        let mut end_span = word.span;

        let kind = match word.name.as_str() {
            "dma_safe" => AttrKind::DmaSafe,
            "manual_yield" => AttrKind::ManualYield,
            "critical" => AttrKind::Priority(Priority::Critical),
            "high" => AttrKind::Priority(Priority::High),
            "normal" => AttrKind::Priority(Priority::Normal),
            "low" => AttrKind::Priority(Priority::Low),
            "core" => {
                self.expect(&TokenKind::LParen)?;
                let tok = self.peek_token();
                let core_id = match tok.kind {
                    TokenKind::Int(n) => {
                        self.bump();
                        u32::try_from(n).map_err(|_| {
                            self.err_at("core index out of `u32` range", tok.span)
                                .with_doc_topic("tasks")
                        })?
                    }
                    _ => {
                        return Err(self
                            .err_at(
                                format!(
                                    "expected integer core index in `@core(N)`, found {}",
                                    tok.kind
                                ),
                                tok.span,
                            )
                            .with_doc_topic("tasks"));
                    }
                };
                let rp = self.expect(&TokenKind::RParen)?;
                end_span = rp.span;
                AttrKind::Core(core_id)
            }
            custom => {
                let arg = if self.eat(&TokenKind::LParen) {
                    let arg_tok = self.bump();
                    let s = match arg_tok.kind {
                        TokenKind::Ident(s) | TokenKind::String(s) => s,
                        TokenKind::Int(n) => n.to_string(),
                        other => {
                            return Err(self
                                .err_at(format!("unexpected attribute argument {other}"), arg_tok.span)
                                .with_doc_topic("tasks"));
                        }
                    };
                    let rp = self.expect(&TokenKind::RParen)?;
                    end_span = rp.span;
                    Some(s)
                } else {
                    None
                };
                AttrKind::Custom {
                    name: custom.to_owned(),
                    arg,
                }
            }
        };

        Ok(Attr {
            kind,
            span: at_tok.span.merge(end_span),
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Imports, Structs, Enums, Constants, Externs, Functions, Tasks, ISRs
    // ─────────────────────────────────────────────────────────────────────────

    fn parse_import(&mut self) -> Result<ImportDecl, ParseError> {
        let start = self.expect(&TokenKind::KwImport)?.span;
        let mut path = vec![self.parse_ident()?];
        while self.eat(&TokenKind::Dot) || self.eat(&TokenKind::ColonColon) {
            path.push(self.parse_ident()?);
        }
        let alias = if self.eat(&TokenKind::KwAs) {
            Some(self.parse_ident()?)
        } else {
            None
        };
        let semi = self.expect(&TokenKind::Semi)?;
        Ok(ImportDecl {
            path,
            alias,
            span: start.merge(semi.span),
        })
    }

    fn parse_struct(&mut self) -> Result<StructDecl, ParseError> {
        let start = self.peek_span();
        let is_packed = self.eat(&TokenKind::KwPacked);
        self.expect(&TokenKind::KwStruct)?;
        let name = self.parse_ident()?;
        let generic_params = self.parse_optional_generic_params()?;
        self.expect(&TokenKind::LBrace)?;

        let mut fields = Vec::new();
        while !self.at(&TokenKind::RBrace) && !self.at(&TokenKind::Eof) {
            let f_start = self.peek_span();
            if self.at(&TokenKind::KwPub) {
                return Err(self
                    .err_at(
                        "`pub` visibility modifier is not allowed on struct fields",
                        f_start,
                    )
                    .with_doc_topic("syntax"));
            }
            if self.at(&TokenKind::KwStatic) {
                return Err(self.trap_p0010(f_start));
            }
            let (ty, field_name) = self.parse_field_or_param("struct field")?;
            let f_end = if self.eat(&TokenKind::Semi) || self.eat(&TokenKind::Comma) {
                self.prev_span()
            } else if self.at(&TokenKind::RBrace) {
                field_name.span
            } else {
                return Err(self
                    .err_at(
                        format!("expected `;` after struct field `{}`", field_name.name),
                        self.peek_span(),
                    )
                    .with_doc_topic("syntax"));
            };
            fields.push(StructField {
                ty,
                name: field_name,
                span: f_start.merge(f_end),
            });
        }

        let rb = self.expect(&TokenKind::RBrace)?;
        let end = if self.eat(&TokenKind::Semi) {
            self.prev_span()
        } else {
            rb.span
        };

        Ok(StructDecl {
            is_packed,
            name,
            generic_params,
            fields,
            span: start.merge(end),
        })
    }

    fn parse_enum(&mut self) -> Result<EnumDecl, ParseError> {
        let start = self.expect(&TokenKind::KwEnum)?.span;
        let name = self.parse_ident()?;
        self.check_generic_lifetime_after_decl_name()?;
        self.expect(&TokenKind::LBrace)?;

        let mut variants = Vec::new();
        while !self.at(&TokenKind::RBrace) && !self.at(&TokenKind::Eof) {
            let v_name = self.parse_ident()?;
            let mut v_end = v_name.span;
            let discriminant = if self.eat(&TokenKind::Eq) {
                let neg = self.eat(&TokenKind::Minus);
                let tok = self.peek_token();
                match tok.kind {
                    TokenKind::Int(n) => {
                        self.bump();
                        v_end = tok.span;
                        let val = i64::try_from(n).map_err(|_| {
                            self.err_at("enum discriminant out of `i64` range", tok.span)
                                .with_doc_topic("syntax")
                        })?;
                        Some(if neg { -val } else { val })
                    }
                    _ => {
                        return Err(self
                            .err_at(
                                format!("expected integer discriminant, found {}", tok.kind),
                                tok.span,
                            )
                            .with_doc_topic("syntax"));
                    }
                }
            } else {
                None
            };

            variants.push(EnumVariant {
                span: v_name.span.merge(v_end),
                name: v_name,
                discriminant,
            });

            if !self.eat(&TokenKind::Comma) && !self.eat(&TokenKind::Semi) {
                break;
            }
        }

        let rb = self.expect(&TokenKind::RBrace)?;
        let end = if self.eat(&TokenKind::Semi) {
            self.prev_span()
        } else {
            rb.span
        };

        Ok(EnumDecl {
            name,
            variants,
            span: start.merge(end),
        })
    }

    fn parse_const(&mut self, first_span: Span) -> Result<ConstDecl, ParseError> {
        self.expect(&TokenKind::KwConst)?;
        let (ty, name) = self.parse_field_or_param("`const` declaration")?;
        self.expect(&TokenKind::Eq)?;
        let value = self.parse_expr()?;
        let semi = self.expect(&TokenKind::Semi)?;
        Ok(ConstDecl {
            ty,
            name,
            value,
            span: first_span.merge(semi.span),
        })
    }

    fn parse_extern_items(
        &mut self,
        attrs: Vec<Attr>,
        is_pub: bool,
        first_span: Span,
        out: &mut Vec<Item>,
    ) -> Result<(), ParseError> {
        self.expect(&TokenKind::KwExtern)?;
        if let TokenKind::String(abi) = self.peek_kind() {
            let abi_span = self.bump().span;
            if abi != "C" {
                return Err(self
                    .err_at(
                        format!("unsupported extern ABI `{abi}`; only `\"C\"` is supported"),
                        abi_span,
                    )
                    .with_doc_topic("yielding"));
            }
        }

        if self.eat(&TokenKind::LBrace) {
            while !self.at(&TokenKind::RBrace) && !self.at(&TokenKind::Eof) {
                let item_start = self.peek_span();
                if self.at(&TokenKind::KwStatic) {
                    return Err(self.trap_p0010(item_start));
                }
                let mut item_attrs = attrs.clone();
                item_attrs.extend(self.parse_attrs()?);
                let inner_pub_span = if self.at(&TokenKind::KwPub) {
                    Some(self.bump().span)
                } else {
                    None
                };
                item_attrs.extend(self.parse_attrs()?);
                if self.at(&TokenKind::KwPub) {
                    return Err(self
                        .err_at("duplicate `pub` visibility modifier", self.peek_span())
                        .with_doc_topic("syntax"));
                }
                if is_pub {
                    if let Some(ps) = inner_pub_span {
                        return Err(self
                            .err_at(
                                "duplicate `pub` visibility modifier on `extern \"C\"` item",
                                ps,
                            )
                            .with_doc_topic("syntax"));
                    }
                }
                let item_is_pub = is_pub || inner_pub_span.is_some();
                let decl = if self.at(&TokenKind::KwFn) {
                    self.parse_kw_fn(item_attrs.clone(), true, item_start)?
                } else {
                    let (ty, name) = self.parse_typed_declarator("extern \"C\" function")?;
                    self.finish_fn_after_name(item_attrs.clone(), true, ty, name, item_start)?
                };
                let span = item_start.merge(decl.span);
                out.push(Item::with_pub(item_attrs, item_is_pub, ItemKind::Fn(decl), span));
            }
            self.expect(&TokenKind::RBrace)?;
        } else if self.at(&TokenKind::KwFn) {
            let decl = self.parse_kw_fn(attrs.clone(), true, first_span)?;
            let span = decl.span;
            out.push(Item::with_pub(attrs, is_pub, ItemKind::Fn(decl), span));
        } else {
            let (ty, name) = self.parse_typed_declarator("extern \"C\" function")?;
            let decl = self.finish_fn_after_name(attrs.clone(), true, ty, name, first_span)?;
            let span = decl.span;
            out.push(Item::with_pub(attrs, is_pub, ItemKind::Fn(decl), span));
        }

        Ok(())
    }

    fn parse_mmio(&mut self, first_span: Span) -> Result<MmioDecl, ParseError> {
        // Consume contextual `mmio` keyword.
        let kw = self.bump();
        debug_assert!(matches!(kw.kind, TokenKind::Ident(ref s) if s == "mmio"));
        let name = self.parse_ident()?;
        self.check_generic_lifetime_after_decl_name()?;

        let (kind, end_span) = if self.at(&TokenKind::LBrace) {
            let (regs, rb_span) = self.parse_mmio_reg_block()?;
            let end = if self.eat(&TokenKind::Semi) {
                self.prev_span()
            } else {
                rb_span
            };
            (MmioDeclKind::Layout { regs }, end)
        } else if self.eat(&TokenKind::Colon) {
            let layout = self.parse_type()?;
            self.expect(&TokenKind::At)?;
            let base_addr = self.parse_expr()?;
            let semi = self.expect(&TokenKind::Semi)?;
            (MmioDeclKind::Instance { layout, base_addr }, semi.span)
        } else if self.eat(&TokenKind::At) {
            // `allow_struct_lit = false` so `RCC_BASE { ... }` is never misparsed as a struct lit.
            let base_addr = self.parse_condition_expr()?;
            let (regs, rb_span) = self.parse_mmio_reg_block()?;
            let end = if self.eat(&TokenKind::Semi) {
                self.prev_span()
            } else {
                rb_span
            };
            (MmioDeclKind::Inline { base_addr, regs }, end)
        } else {
            return Err(self
                .err_at(
                    format!(
                        "expected `{{`, `:`, or `@` after `mmio` name `{}`, found {}",
                        name.name,
                        self.peek_kind()
                    ),
                    self.peek_span(),
                )
                .with_doc_topic("syntax")
                .with_hint(
                    "use `mmio Regs { ... }`, `mmio PERIPH : Regs @ 0x...;`, or `mmio PERIPH @ 0x... { ... }`",
                ));
        };

        Ok(MmioDecl {
            name,
            kind,
            span: first_span.merge(end_span),
        })
    }

    fn parse_mmio_reg_block(&mut self) -> Result<(Vec<MmioRegDecl>, Span), ParseError> {
        self.expect(&TokenKind::LBrace)?;
        let mut regs = Vec::new();
        while !self.at(&TokenKind::RBrace) && !self.at(&TokenKind::Eof) {
            let r_start = self.peek_span();
            if self.at(&TokenKind::KwStatic) {
                return Err(self.trap_p0010(r_start));
            }
            if self.at(&TokenKind::KwPub) {
                return Err(self
                    .err_at(
                        "`pub` visibility modifier is not allowed on `mmio` registers",
                        r_start,
                    )
                    .with_doc_topic("syntax"));
            }
            if !self.eat(&TokenKind::At) {
                return Err(self
                    .err_at(
                        format!(
                            "expected MMIO register access qualifier (`@rw`, `@ro`, `@wo`, or `@w1c`), found {}",
                            self.peek_kind()
                        ),
                        r_start,
                    )
                    .with_doc_topic("syntax")
                    .with_hint("prefix every `mmio` register with `@rw`, `@ro`, `@wo`, or `@w1c`"));
            }
            let access_ident =
                self.parse_word_ident("MMIO access qualifier (`rw`, `ro`, `wo`, or `w1c`)")?;
            let access = match access_ident.name.as_str() {
                "rw" => MmioAccess::Rw,
                "ro" => MmioAccess::Ro,
                "wo" => MmioAccess::Wo,
                "w1c" => MmioAccess::W1c,
                other => {
                    return Err(self
                        .err_at(
                            format!(
                                "invalid MMIO access qualifier `@{other}`; expected `@rw`, `@ro`, `@wo`, or `@w1c`"
                            ),
                            r_start.merge(access_ident.span),
                        )
                        .with_doc_topic("syntax"));
                }
            };
            let (ty, name) = self.parse_typed_declarator_inner("MMIO register", true)?;
            self.expect(&TokenKind::At).map_err(|_| {
                self.err_at(
                    format!("expected `@ <offset>` on MMIO register `{}`", name.name),
                    self.peek_span(),
                )
                .with_doc_topic("syntax")
                .with_hint("specify register byte offset, e.g. `@rw u32 cr1 @ 0x00;`")
            })?;
            let offset = self.parse_expr()?;
            let semi = self.expect(&TokenKind::Semi)?;
            regs.push(MmioRegDecl {
                access,
                ty,
                name,
                offset,
                span: r_start.merge(semi.span),
            });
        }
        let rb = self.expect(&TokenKind::RBrace)?;
        Ok((regs, rb.span))
    }

    fn parse_irq(&mut self, first_span: Span) -> Result<IrqDecl, ParseError> {
        let kw = self.bump();
        debug_assert!(matches!(kw.kind, TokenKind::Ident(ref s) if s == "irq"));
        let name = self.parse_ident()?;
        self.expect(&TokenKind::Eq)?;
        let number = self.parse_expr()?;
        let semi = self.expect(&TokenKind::Semi)?;
        Ok(IrqDecl {
            name,
            number,
            span: first_span.merge(semi.span),
        })
    }

    /// Parse `fn name(params...) [error ...] ({ ... } | ;)`.
    /// Traps Rust-style `-> RetType` after `fn name(...)` with `P0002` (or `P0007` if `-> !Type`).
    fn parse_kw_fn(
        &mut self,
        attrs: Vec<Attr>,
        is_extern_c: bool,
        first_span: Span,
    ) -> Result<FnDecl, ParseError> {
        let fn_tok = self.expect(&TokenKind::KwFn)?;
        let is_standard_fn = matches!(self.peek_kind(), TokenKind::Ident(_))
            && matches!(self.peek_n_kind(1), TokenKind::LParen | TokenKind::Lt);

        if !is_standard_fn {
            // `fn !u16 foo(...)` -> P0007
            if self.at(&TokenKind::Bang) {
                return Err(self.trap_p0007(
                    "Zig-style `!Type` error-union return type",
                    self.peek_span(),
                ));
            }
            let (ret_type, name) = self.parse_typed_declarator("function declaration")?;
            return self.finish_fn_after_name(attrs, is_extern_c, ret_type, name, first_span);
        }

        let name = self.parse_ident()?;
        let generic_params = self.parse_optional_generic_params()?;
        let (params, is_variadic) = self.parse_param_list()?;

        if self.at(&TokenKind::Arrow) {
            let arrow = self.bump();
            if self.at(&TokenKind::Bang) {
                let bang = self.bump();
                return Err(self.trap_p0007(
                    "Zig/Rust `-> !Type` error-union return type",
                    arrow.span.merge(bang.span),
                ));
            }
            let (ret_hint, end_span) = self.sniff_type_display_and_span(arrow.span);
            return Err(self.trap_p0002_fn_arrow(
                &name.name,
                ret_hint.as_deref(),
                arrow.span.merge(end_span),
            ));
        }

        let ret_type = TypeExpr::void(fn_tok.span);
        let error_spec = self.parse_optional_error_spec()?;
        self.push_value_scope();
        for p in &params {
            self.record_local_value(&p.name.name);
        }
        let body_res = self.parse_fn_body(is_extern_c, &name);
        self.pop_value_scope();
        let (body, end_span) = body_res?;

        Ok(FnDecl {
            attrs,
            is_extern_c,
            name,
            generic_params,
            params,
            is_variadic,
            ret_type,
            error_spec,
            body,
            span: first_span.merge(end_span),
        })
    }

    fn finish_fn_after_name(
        &mut self,
        attrs: Vec<Attr>,
        is_extern_c: bool,
        ret_type: TypeExpr,
        name: Ident,
        first_span: Span,
    ) -> Result<FnDecl, ParseError> {
        let generic_params = self.parse_optional_generic_params()?;
        let (params, is_variadic) = self.parse_param_list()?;

        if self.at(&TokenKind::Arrow) {
            let arrow = self.bump();
            let (ret_hint, end_span) = self.sniff_type_display_and_span(arrow.span);
            return Err(self.trap_p0002_fn_arrow(
                &name.name,
                ret_hint.as_deref(),
                arrow.span.merge(end_span),
            ));
        }

        let error_spec = self.parse_optional_error_spec()?;
        self.push_value_scope();
        for p in &params {
            self.record_local_value(&p.name.name);
        }
        let body_res = self.parse_fn_body(is_extern_c, &name);
        self.pop_value_scope();
        let (body, end_span) = body_res?;

        Ok(FnDecl {
            attrs,
            is_extern_c,
            name,
            generic_params,
            params,
            is_variadic,
            ret_type,
            error_spec,
            body,
            span: first_span.merge(end_span),
        })
    }

    /// Parse optional `<T>` / `<T, usize N>` generic declaration parameters on `struct` or `fn`.
    fn parse_optional_generic_params(&mut self) -> Result<Vec<GenericParam>, ParseError> {
        if !self.at(&TokenKind::Lt) {
            return Ok(Vec::new());
        }
        self.check_generic_lifetime_after_decl_name()?;
        let lt_tok = self.expect(&TokenKind::Lt)?;

        if self.at(&TokenKind::Gt) || self.at(&TokenKind::Shr) {
            let gt_span = self.expect_type_gt()?;
            return Err(self
                .err_at("empty generic parameter list `<>`", lt_tok.span.merge(gt_span))
                .with_doc_topic("syntax"));
        }

        let mut params = Vec::new();
        while !self.at(&TokenKind::Gt) && !self.at(&TokenKind::Shr) && !self.at(&TokenKind::Eof) {
            let p_start = self.peek_span();
            if let TokenKind::Lifetime(ref lt) = self.peek_kind() {
                return Err(self.trap_p0009_lifetime(lt, p_start));
            }
            if self.eat(&TokenKind::KwConst) {
                // Guide Rust `<const N: usize>` habits cleanly to Zev `<usize N>`
                if matches!(self.peek_kind(), TokenKind::Ident(ref s) if s == "usize")
                    && matches!(self.peek_n_kind(1), TokenKind::Ident(_))
                {
                    self.bump(); // `usize`
                    let name = self.parse_ident()?;
                    params.push(GenericParam {
                        span: p_start.merge(name.span),
                        name,
                        kind: GenericParamKind::ConstUsize,
                    });
                } else {
                    let name = self.parse_ident()?;
                    return Err(self
                        .err_at(
                            format!("Rust-style `const {}` generic parameter is written `usize {}` in Zev", name.name, name.name),
                            p_start.merge(self.peek_span()),
                        )
                        .with_code("P0002")
                        .with_doc_topic("syntax")
                        .with_hint(format!("write `<T, usize {}>` instead of `<T, const {}: usize>`", name.name, name.name)));
                }
            } else if matches!(self.peek_kind(), TokenKind::Ident(ref s) if s == "usize")
                && matches!(self.peek_n_kind(1), TokenKind::Ident(_))
            {
                self.bump(); // `usize`
                let name = self.parse_ident()?;
                params.push(GenericParam {
                    span: p_start.merge(name.span),
                    name,
                    kind: GenericParamKind::ConstUsize,
                });
            } else {
                let name = self.parse_ident()?;
                if self.at(&TokenKind::Colon) {
                    let colon = self.bump();
                    return Err(self
                        .err_at(
                            format!("`: ...` bounds on generic parameter `{}` are not supported in Zev", name.name),
                            name.span.merge(colon.span),
                        )
                        .with_doc_topic("syntax")
                        .with_hint("Zev monomorphized generics omit trait hierarchies; write `<T>` or `<T, usize N>`"));
                }
                params.push(GenericParam {
                    span: name.span,
                    name,
                    kind: GenericParamKind::Type,
                });
            }

            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }

        self.expect_type_gt()?;
        Ok(params)
    }

    /// Parse `<Arg1, Arg2, ...>` generic arguments (splitting `>>` cleanly via `expect_type_gt`).
    fn parse_generic_args(&mut self) -> Result<(Vec<GenericArg>, Span), ParseError> {
        self.check_generic_lifetime_after_decl_name()?;
        let lt_tok = self.expect(&TokenKind::Lt)?;

        if self.at(&TokenKind::Gt) || self.at(&TokenKind::Shr) {
            let gt_span = self.expect_type_gt()?;
            return Err(self
                .err_at("empty generic argument list `<>`", lt_tok.span.merge(gt_span))
                .with_doc_topic("syntax"));
        }

        let mut args = Vec::new();
        while !self.at(&TokenKind::Gt) && !self.at(&TokenKind::Shr) && !self.at(&TokenKind::Eof) {
            args.push(self.parse_generic_arg()?);
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }

        let gt_span = self.expect_type_gt()?;
        Ok((args, gt_span))
    }

    fn parse_generic_arg(&mut self) -> Result<GenericArg, ParseError> {
        if let TokenKind::Lifetime(ref lt) = self.peek_kind() {
            return Err(self.trap_p0009_lifetime(lt, self.peek_span()));
        }

        let cp = self.save();
        match self.parse_type() {
            Ok(ty)
                if self.at(&TokenKind::Comma)
                    || self.at(&TokenKind::Gt)
                    || self.at(&TokenKind::Shr) =>
            {
                return Ok(GenericArg::Type(ty));
            }
            Err(e) if e.code.is_some() => {
                return Err(e);
            }
            _ => {
                self.restore(cp);
            }
        }

        let expr = self.parse_generic_const_expr()?;
        Ok(GenericArg::Const(expr))
    }

    /// Parse a constant expression inside `<...>` generic arguments, supporting `|`, `^`, `&`,
    /// `<<` (`TokenKind::Shl`), `+`, `-`, `*`, `/`, `%`, casts, and unary ops while stopping
    /// cleanly before closing `>` and `>>` (`TokenKind::Shr`).
    fn parse_generic_const_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_generic_bitxor_expr()?;
        while self.eat(&TokenKind::Pipe) {
            let rhs = self.parse_generic_bitxor_expr()?;
            let span = lhs.span.merge(rhs.span);
            lhs = Expr::new(
                ExprKind::Binary {
                    op: BinaryOp::BitOr,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            );
        }
        Ok(lhs)
    }

    fn parse_generic_bitxor_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_generic_bitand_expr()?;
        while self.eat(&TokenKind::Caret) {
            let rhs = self.parse_generic_bitand_expr()?;
            let span = lhs.span.merge(rhs.span);
            lhs = Expr::new(
                ExprKind::Binary {
                    op: BinaryOp::BitXor,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            );
        }
        Ok(lhs)
    }

    fn parse_generic_bitand_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_generic_shl_expr()?;
        while self.eat(&TokenKind::Amp) {
            let rhs = self.parse_generic_shl_expr()?;
            let span = lhs.span.merge(rhs.span);
            lhs = Expr::new(
                ExprKind::Binary {
                    op: BinaryOp::BitAnd,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            );
        }
        Ok(lhs)
    }

    fn parse_generic_shl_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_additive_expr()?;
        while self.eat(&TokenKind::Shl) {
            let rhs = self.parse_additive_expr()?;
            let span = lhs.span.merge(rhs.span);
            lhs = Expr::new(
                ExprKind::Binary {
                    op: BinaryOp::Shl,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            );
        }
        Ok(lhs)
    }

    /// Best-effort read of a foreign type token sequence solely for formatting a 1-shot rewrite hint.
    fn sniff_type_display_and_span(&mut self, fallback: Span) -> (Option<String>, Span) {
        let cp = self.save();
        if let Ok(ty) = self.parse_type() {
            let slice = self
                .source
                .get(ty.span.start as usize..ty.span.end as usize)
                .map(|s| s.trim().to_owned());
            return (slice, ty.span);
        }
        self.restore(cp);
        (None, fallback)
    }

    fn parse_optional_error_spec(&mut self) -> Result<Option<ErrorSpec>, ParseError> {
        if !self.at(&TokenKind::KwError) {
            return Ok(None);
        }
        let err_tok = self.bump();
        if self.at(&TokenKind::LBrace) || self.at(&TokenKind::Semi) {
            Ok(Some(ErrorSpec::Bare(err_tok.span)))
        } else {
            let err_ty = self.parse_type()?;
            Ok(Some(ErrorSpec::Typed(err_ty)))
        }
    }

    fn parse_fn_body(
        &mut self,
        is_extern_c: bool,
        name: &Ident,
    ) -> Result<(Option<Block>, Span), ParseError> {
        if self.at(&TokenKind::Semi) {
            let semi = self.bump();
            if !is_extern_c {
                return Err(self
                    .err_at(
                        format!(
                            "non-extern function `{}` must have a `{{ ... }}` body",
                            name.name
                        ),
                        semi.span,
                    )
                    .with_doc_topic("syntax")
                    .with_hint("mark external prototypes with `extern \"C\"`"));
            }
            Ok((None, semi.span))
        } else if self.at(&TokenKind::LBrace) {
            let block = self.parse_block()?;
            let end = block.span;
            Ok((Some(block), end))
        } else {
            if let Some(trap_err) = self.check_unexpected_token_trap(&self.peek_token()) {
                return Err(trap_err);
            }
            Err(self
                .err_at(
                    format!(
                        "expected `{}` after function signature `{}`",
                        if is_extern_c { ";" } else { "{" },
                        name.name
                    ),
                    self.peek_span(),
                )
                .with_doc_topic("syntax"))
        }
    }

    fn parse_task(&mut self, attrs: Vec<Attr>, first_span: Span) -> Result<TaskDecl, ParseError> {
        self.expect(&TokenKind::KwTask)?;
        let name = self.parse_ident()?;
        self.check_generic_lifetime_after_decl_name()?;
        let (params, is_variadic) = self.parse_param_list()?;
        if is_variadic {
            return Err(self
                .err_at("`task` blueprints cannot be variadic (`...`)", name.span)
                .with_doc_topic("tasks"));
        }
        if self.at(&TokenKind::Arrow) {
            let arrow = self.bump();
            return Err(self
                .err_at("`task` blueprints do not return values (`-> ...`)", arrow.span)
                .with_code("P0002")
                .with_doc_topic("tasks")
                .with_hint("remove `-> ...`; tasks communicate outputs via `watch<T>` or `inbox<T, N>` parameters"));
        }
        self.push_value_scope();
        for p in &params {
            self.record_local_value(&p.name.name);
        }
        let body_res = self.parse_block();
        self.pop_value_scope();
        let body = body_res?;
        let span = first_span.merge(body.span);
        Ok(TaskDecl {
            attrs,
            name,
            params,
            body,
            span,
        })
    }

    fn parse_isr(&mut self, attrs: Vec<Attr>, first_span: Span) -> Result<IsrDecl, ParseError> {
        self.expect(&TokenKind::KwIsr)?;
        let vector = self.parse_ident()?;
        self.check_generic_lifetime_after_decl_name()?;
        let params = if self.at(&TokenKind::LParen) {
            let (params, is_variadic) = self.parse_param_list()?;
            if is_variadic {
                return Err(self
                    .err_at("`isr` handlers cannot be variadic (`...`)", vector.span)
                    .with_doc_topic("tasks"));
            }
            params
        } else {
            Vec::new()
        };
        if self.at(&TokenKind::Arrow) {
            let arrow = self.bump();
            return Err(self
                .err_at("`isr` handlers do not return values (`-> ...`)", arrow.span)
                .with_code("P0002")
                .with_doc_topic("tasks")
                .with_hint("remove `-> ...`; `isr` handlers return `void`"));
        }
        self.push_value_scope();
        for p in &params {
            self.record_local_value(&p.name.name);
        }
        let body_res = self.parse_block();
        self.pop_value_scope();
        let body = body_res?;
        let span = first_span.merge(body.span);
        Ok(IsrDecl {
            attrs,
            vector,
            params,
            body,
            span,
        })
    }

    fn parse_param_list(&mut self) -> Result<(Vec<Param>, bool), ParseError> {
        self.expect(&TokenKind::LParen)?;
        let mut params = Vec::new();
        let mut is_variadic = false;

        if matches!(self.peek_kind(), TokenKind::Ident(ref s) if s == "void")
            && self.peek_n_kind(1) == TokenKind::RParen
        {
            self.bump();
            self.expect(&TokenKind::RParen)?;
            return Ok((params, false));
        }

        while !self.at(&TokenKind::RParen) && !self.at(&TokenKind::Eof) {
            if self.eat(&TokenKind::DotDotDot) {
                is_variadic = true;
                let _ = self.eat(&TokenKind::Comma);
                break;
            }

            let p_start = self.peek_span();
            let (ty, name) = self.parse_field_or_param("parameter")?;
            let span = p_start.merge(name.span);
            params.push(Param { ty, name, span });

            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }

        self.expect(&TokenKind::RParen)?;
        Ok((params, is_variadic))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Types & Declarators
    // ─────────────────────────────────────────────────────────────────────────

    /// Parse a standalone [`TypeExpr`] (handles prefix `*T`, `[]T`, `[N]T`, `fn(...) -> Ret`,
    /// OS primitives, built-in primitives, qualified paths, and postfix `T*`, `T[]`, `T[N]`).
    pub fn parse_type(&mut self) -> Result<TypeExpr, ParseError> {
        let start = self.peek_span();

        // Trap foreign modifiers/references in type position:
        match self.peek_kind() {
            TokenKind::KwMut => {
                return Err(self.trap_p0001("mut", start));
            }
            TokenKind::Lifetime(ref lt) => {
                return Err(self.trap_p0009_lifetime(lt, start));
            }
            TokenKind::Amp => {
                self.bump();
                if let TokenKind::Lifetime(ref lt) = self.peek_kind() {
                    let lt_span = self.peek_span();
                    return Err(self.trap_p0009_lifetime(lt, start.merge(lt_span)));
                }
                if self.eat(&TokenKind::KwMut) {
                    return Err(self.trap_p0009_mut_or_ref("&mut", start.merge(self.prev_span())));
                }
                return Err(self.trap_p0009_mut_or_ref("&Type", start));
            }
            TokenKind::Bang => {
                return Err(self.trap_p0007("Zig-style `!Type` error-union type", start));
            }
            TokenKind::Question => {
                return Err(self.trap_p0007("Zig-style `?Type` optional/error type", start));
            }
            _ => {}
        }

        // Optional `const` qualifier on types (e.g. `*const u8` or `const u8*`)
        let _ = self.eat(&TokenKind::KwConst);

        let mut ty = match self.peek_kind() {
            TokenKind::Star => {
                self.bump();
                if self.at(&TokenKind::KwMut) {
                    let mut_span = self.peek_span();
                    return Err(self.trap_p0009_mut_or_ref("*mut", start.merge(mut_span)));
                }
                let _ = self.eat(&TokenKind::KwConst);
                let inner = self.parse_type()?;
                let span = start.merge(inner.span);
                TypeExpr::new(TypeKind::Ptr(Box::new(inner)), span)
            }
            TokenKind::LBracket => {
                self.bump();
                if self.eat(&TokenKind::RBracket) {
                    if self.at(&TokenKind::KwMut) {
                        return Err(self.trap_p0001("mut", self.peek_span()));
                    }
                    let _ = self.eat(&TokenKind::KwConst);
                    let elem = self.parse_type()?;
                    let span = start.merge(elem.span);
                    TypeExpr::new(TypeKind::Slice(Box::new(elem)), span)
                } else {
                    let len = self.parse_array_len()?;
                    self.expect(&TokenKind::RBracket)?;
                    let elem = self.parse_type()?;
                    let span = start.merge(elem.span);
                    TypeExpr::new(
                        TypeKind::Array {
                            elem: Box::new(elem),
                            len,
                        },
                        span,
                    )
                }
            }
            TokenKind::KwFn => {
                // Note: function-pointer types `fn(T1, T2) -> Ret` KEEP `-> Ret`!
                self.bump();
                let fn_params = self.parse_fn_ptr_param_types()?;
                let (ret, end_span) = if self.eat(&TokenKind::Arrow) {
                    let r = self.parse_type()?;
                    let s = r.span;
                    (r, s)
                } else {
                    let s = self.prev_span();
                    (TypeExpr::void(s), s)
                };
                TypeExpr::new(
                    TypeKind::FnPtr {
                        params: fn_params,
                        ret: Box::new(ret),
                    },
                    start.merge(end_span),
                )
            }
            TokenKind::Ident(ref s) => {
                let s_clone = s.clone();
                // Trap Zig `ErrType!PayloadType`
                if self.peek_n_kind(1) == TokenKind::Bang {
                    return Err(self.trap_p0007(
                        "Zig-style `ErrType!Type` error-union type",
                        start.merge(self.peek_n_span(1)),
                    ));
                }
                // Trap Rust `Result<T, E>` in type position
                if s_clone == "Result" && self.peek_n_kind(1) == TokenKind::Lt {
                    return Err(self.trap_p0007("Rust-style `Result<T, E>` type", start));
                }

                match s_clone.as_str() {
                    "signal" if self.peek_n_kind(1) != TokenKind::Dot => {
                        let tok = self.bump();
                        TypeExpr::new(TypeKind::OsPrimitive(OsPrimitiveType::Signal), tok.span)
                    }
                    "watch" if self.peek_n_kind(1) == TokenKind::Lt => {
                        self.bump();
                        self.expect(&TokenKind::Lt)?;
                        let elem = self.parse_type()?;
                        let gt_span = self.expect_type_gt()?;
                        TypeExpr::new(
                            TypeKind::OsPrimitive(OsPrimitiveType::Watch {
                                elem: Box::new(elem),
                            }),
                            start.merge(gt_span),
                        )
                    }
                    "chan" if self.peek_n_kind(1) == TokenKind::Lt => {
                        // Trap retired `chan<T, N>` (or `chan<T>`) with `P0012` -> unified `inbox<T, N>`!
                        self.bump(); // `chan`
                        let lt_tok = self.expect(&TokenKind::Lt)?;
                        let mut end_span = lt_tok.span;
                        let mut depth = 1usize;
                        while depth > 0 && !self.at(&TokenKind::Eof) && !self.at(&TokenKind::Semi) {
                            let tok = self.bump();
                            end_span = tok.span;
                            match tok.kind {
                                TokenKind::Lt => depth += 1,
                                TokenKind::Gt => depth = depth.saturating_sub(1),
                                TokenKind::Shr => depth = depth.saturating_sub(2),
                                _ => {}
                            }
                        }
                        return Err(self.trap_p0012(start.merge(end_span)));
                    }
                    "inbox" | "ping_pong" | "lease"
                        if self.peek_n_kind(1) == TokenKind::Lt =>
                    {
                        self.bump();
                        self.expect(&TokenKind::Lt)?;
                        let elem = self.parse_type()?;
                        self.expect(&TokenKind::Comma)?;
                        let cap = self.parse_usize_literal("OS primitive capacity")?;
                        let gt_span = self.expect_type_gt()?;
                        let os_ty = match s_clone.as_str() {
                            "inbox" => OsPrimitiveType::Inbox {
                                elem: Box::new(elem),
                                cap,
                            },
                            "ping_pong" => OsPrimitiveType::PingPong {
                                elem: Box::new(elem),
                                cap,
                            },
                            "lease" => OsPrimitiveType::Lease {
                                elem: Box::new(elem),
                                cap,
                            },
                            _ => unreachable!(),
                        };
                        TypeExpr::new(TypeKind::OsPrimitive(os_ty), start.merge(gt_span))
                    }
                    other => {
                        if let Some(prim) = PrimType::from_name(other) {
                            let tok = self.bump();
                            TypeExpr::new(TypeKind::Prim(prim), tok.span)
                        } else {
                            let first_id = self.parse_ident()?;
                            // Trap `Reader<'a>` lifetime argument on user type
                            self.check_generic_lifetime_after_decl_name()?;

                            let mut segs = vec![first_id];
                            while self.at(&TokenKind::Dot)
                                || (self.at(&TokenKind::ColonColon)
                                    && self.peek_n_kind(1) != TokenKind::Lt)
                            {
                                self.bump();
                                segs.push(self.parse_ident()?);
                            }

                            let has_generic_args = self.at(&TokenKind::Lt)
                                || (self.at(&TokenKind::ColonColon)
                                    && self.peek_n_kind(1) == TokenKind::Lt);
                            if has_generic_args {
                                let _ = self.eat(&TokenKind::ColonColon);
                                let (args, gt_span) = self.parse_generic_args()?;
                                if segs.len() == 1 {
                                    let base = segs.pop().unwrap();
                                    TypeExpr::new(
                                        TypeKind::Generic { base, args },
                                        start.merge(gt_span),
                                    )
                                } else {
                                    TypeExpr::new(
                                        TypeKind::GenericPath { path: segs, args },
                                        start.merge(gt_span),
                                    )
                                }
                            } else if segs.len() > 1 {
                                let end = segs.last().unwrap().span;
                                TypeExpr::new(TypeKind::Path(segs), start.merge(end))
                            } else {
                                let only = segs.pop().unwrap();
                                let span = only.span;
                                TypeExpr::new(TypeKind::Named(only), span)
                            }
                        }
                    }
                }
            }
            other => {
                if let Some(trap_err) = self.check_unexpected_token_trap(&self.peek_token()) {
                    return Err(trap_err);
                }
                return Err(self
                    .err_at(format!("expected type, found {other}"), start)
                    .with_doc_topic("syntax"));
            }
        };

        // Postfix type modifiers: `T*`, `T[]`, `T[N]`
        loop {
            if self.eat(&TokenKind::Star) {
                let _ = self.eat(&TokenKind::KwConst);
                let span = ty.span.merge(self.prev_span());
                ty = TypeExpr::new(TypeKind::Ptr(Box::new(ty)), span);
            } else if self.at(&TokenKind::LBracket) {
                self.bump();
                if self.eat(&TokenKind::RBracket) {
                    let span = ty.span.merge(self.prev_span());
                    ty = TypeExpr::new(TypeKind::Slice(Box::new(ty)), span);
                } else {
                    let len = self.parse_array_len()?;
                    let rb = self.expect(&TokenKind::RBracket)?;
                    let span = ty.span.merge(rb.span);
                    ty = TypeExpr::new(
                        TypeKind::Array {
                            elem: Box::new(ty),
                            len,
                        },
                        span,
                    );
                }
            } else {
                break;
            }
        }

        Ok(ty)
    }

    fn parse_fn_ptr_param_types(&mut self) -> Result<Vec<TypeExpr>, ParseError> {
        self.expect(&TokenKind::LParen)?;
        let mut params = Vec::new();

        if matches!(self.peek_kind(), TokenKind::Ident(ref s) if s == "void")
            && self.peek_n_kind(1) == TokenKind::RParen
        {
            self.bump();
            self.expect(&TokenKind::RParen)?;
            return Ok(params);
        }

        while !self.at(&TokenKind::RParen) && !self.at(&TokenKind::Eof) {
            let ty = self.parse_type()?;
            if matches!(self.peek_kind(), TokenKind::Ident(_)) {
                self.bump();
            }
            params.push(ty);
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }

        self.expect(&TokenKind::RParen)?;
        Ok(params)
    }

    /// Parse `Type name` (including C-style `Type *name`, `Type name[N]`, and `Ret (*cb)(Args...)`).
    fn parse_typed_declarator(&mut self, context: &str) -> Result<(TypeExpr, Ident), ParseError> {
        self.parse_typed_declarator_inner(context, false)
    }

    fn parse_typed_declarator_inner(
        &mut self,
        context: &str,
        allow_keyword_name: bool,
    ) -> Result<(TypeExpr, Ident), ParseError> {
        let start = self.peek_span();
        if self.at(&TokenKind::KwMut) {
            return Err(self.trap_p0001("mut", start));
        }

        let mut ty = self.parse_type().map_err(|e| {
            if e.code.is_some() {
                e
            } else {
                self.err_at(
                    format!("expected type in {context}, found {}", self.peek_kind()),
                    start,
                )
                .with_doc_topic("syntax")
            }
        })?;

        // Trap `u16 mut x = ...;`
        if self.at(&TokenKind::KwMut) {
            return Err(self.trap_p0001("mut", self.peek_span()));
        }

        // Check C-style function pointer declarator: `RetType (*name)(ParamTypes...)`
        if self.at(&TokenKind::LParen) && self.peek_n_kind(1) == TokenKind::Star {
            self.bump(); // `(`
            self.bump(); // `*`
            let name = self.parse_ident()?;
            self.expect(&TokenKind::RParen)?;
            let fn_params = self.parse_fn_ptr_param_types()?;
            let span = start.merge(self.prev_span());
            let fn_ty = TypeExpr::new(
                TypeKind::FnPtr {
                    params: fn_params,
                    ret: Box::new(ty),
                },
                span,
            );
            return Ok((fn_ty, name));
        }

        let name = if allow_keyword_name {
            self.parse_word_ident(&format!("{context} name"))?
        } else {
            self.parse_ident()?
        };

        // Trailing C-style array declarator on variable/field name: `u8 buf[16]`
        while self.at(&TokenKind::LBracket) {
            self.bump();
            if self.eat(&TokenKind::RBracket) {
                let span = ty.span.merge(self.prev_span());
                ty = TypeExpr::new(TypeKind::Slice(Box::new(ty)), span);
            } else {
                let len = self.parse_array_len()?;
                let rb = self.expect(&TokenKind::RBracket)?;
                let span = ty.span.merge(rb.span);
                ty = TypeExpr::new(
                    TypeKind::Array {
                        elem: Box::new(ty),
                        len,
                    },
                    span,
                );
            }
        }

        Ok((ty, name))
    }

    /// Parse a parameter, struct field, `const` binding, or `catch` binding,
    /// actively trapping foreign `name: Type` (`P0002`), `&mut` / `'a` (`P0009`), and `mut` (`P0001`).
    fn parse_field_or_param(&mut self, context: &str) -> Result<(TypeExpr, Ident), ParseError> {
        if self.at(&TokenKind::KwMut) {
            return Err(self.trap_p0001("mut", self.peek_span()));
        }

        if matches!(self.peek_kind(), TokenKind::Ident(_))
            && self.peek_n_kind(1) == TokenKind::Colon
        {
            let name = self.parse_ident()?;
            let colon = self.expect(&TokenKind::Colon)?;

            // Check if the type after `:` is specifically testing `&mut` / `'a` (`P0009`) or `!Type` (`P0007`)!
            match self.peek_kind() {
                TokenKind::Amp => {
                    let amp_span = self.bump().span;
                    if let TokenKind::Lifetime(ref lt) = self.peek_kind() {
                        return Err(self.trap_p0009_lifetime(lt, amp_span.merge(self.peek_span())));
                    }
                    if self.eat(&TokenKind::KwMut) {
                        return Err(
                            self.trap_p0009_mut_or_ref("&mut", amp_span.merge(self.prev_span()))
                        );
                    }
                    return Err(self.trap_p0009_mut_or_ref("&Type", amp_span));
                }
                TokenKind::Lifetime(ref lt) => {
                    return Err(self.trap_p0009_lifetime(lt, self.peek_span()));
                }
                TokenKind::Bang => {
                    return Err(self.trap_p0007(
                        "Zig-style `!Type` error-union type",
                        self.peek_span(),
                    ));
                }
                _ => {}
            }

            let (ty_hint, end_span) = self.sniff_type_display_and_span(colon.span);
            return Err(self.trap_p0002_colon_binding(
                &name.name,
                ty_hint.as_deref(),
                name.span.merge(end_span),
            ));
        }

        self.parse_typed_declarator(context)
    }

    /// Parse a concrete `[16]` (`ArrayLen::Fixed(16)`) or symbolic/const `[N]` (`ArrayLen::Expr(...)`) array bound.
    fn parse_array_len(&mut self) -> Result<ArrayLen, ParseError> {
        if matches!(self.peek_kind(), TokenKind::Int(_))
            && self.peek_n_kind(1) == TokenKind::RBracket
        {
            let n = self.parse_usize_literal("array length")?;
            Ok(ArrayLen::Fixed(n))
        } else {
            let prev_allow = self.allow_struct_lit;
            self.allow_struct_lit = true;
            let expr = self.parse_expr();
            self.allow_struct_lit = prev_allow;
            Ok(ArrayLen::Expr(Box::new(expr?)))
        }
    }

    fn parse_usize_literal(&mut self, context: &str) -> Result<usize, ParseError> {
        let tok = self.peek_token();
        match tok.kind {
            TokenKind::Int(n) => {
                self.bump();
                usize::try_from(n).map_err(|_| {
                    self.err_at(format!("{context} out of `usize` range"), tok.span)
                        .with_doc_topic("syntax")
                })
            }
            _ => Err(self
                .err_at(
                    format!("expected integer literal for {context}, found {}", tok.kind),
                    tok.span,
                )
                .with_doc_topic("syntax")),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Blocks & Statements
    // ─────────────────────────────────────────────────────────────────────────

    pub fn parse_block(&mut self) -> Result<Block, ParseError> {
        self.parse_block_inner(false)
    }

    /// Parse a `catch { ... }` body block, allowing an optional semicolon-omitted
    /// trailing fallback expression (`{ stmt1; fallback_expr }`) in addition to `{ stmt1; fallback_expr; }`.
    fn parse_catch_block(&mut self) -> Result<Block, ParseError> {
        self.parse_block_inner(true)
    }

    fn parse_block_inner(&mut self, allow_tail_expr: bool) -> Result<Block, ParseError> {
        let lb = self.expect(&TokenKind::LBrace)?;
        let prev_allow = self.allow_struct_lit;
        self.allow_struct_lit = true;
        self.push_value_scope();

        let mut stmts = Vec::new();
        while !self.at(&TokenKind::RBrace) && !self.at(&TokenKind::Eof) {
            match self.parse_stmt_in_block(allow_tail_expr) {
                Ok(stmt) => stmts.push(stmt),
                Err(e) => {
                    self.pop_value_scope();
                    self.allow_struct_lit = prev_allow;
                    return Err(e);
                }
            }
        }

        self.pop_value_scope();
        self.allow_struct_lit = prev_allow;
        let rb = self.expect(&TokenKind::RBrace)?;
        Ok(Block::new(stmts, lb.span.merge(rb.span)))
    }

    fn parse_stmt_in_block(&mut self, allow_tail_expr: bool) -> Result<Stmt, ParseError> {
        let start = self.peek_span();
        let attrs = self.parse_attrs()?;
        let cur_span = self.peek_span();

        match self.peek_kind() {
            // ── Foreign-idiom statement traps (`P0001`..`P0010`) ──
            TokenKind::KwLet => return Err(self.trap_p0001("let", cur_span)),
            TokenKind::KwVar => return Err(self.trap_p0001("var", cur_span)),
            TokenKind::KwMut => return Err(self.trap_p0001("mut", cur_span)),
            TokenKind::KwAsync => return Err(self.trap_p0003("async", cur_span)),
            TokenKind::KwAwait => return Err(self.trap_p0003("await", cur_span)),
            TokenKind::KwSelect => return Err(self.trap_p0004(cur_span)),
            TokenKind::KwMatch => return Err(self.trap_p0005(cur_span)),
            TokenKind::KwTry => return Err(self.trap_p0007("`try`", cur_span)),
            TokenKind::KwStatic => return Err(self.trap_p0010(cur_span)),
            TokenKind::KwPub => {
                return Err(self
                    .err_at(
                        "`pub` visibility modifier is only allowed on top-level declarations, not inside blocks",
                        cur_span,
                    )
                    .with_doc_topic("syntax"));
            }
            TokenKind::KwConst => {
                return Err(self
                    .err_at(
                        "block-local `const` declarations are not supported; declare `const` items at module top level",
                        cur_span,
                    )
                    .with_doc_topic("syntax")
                    .with_hint(
                        "move `const Type NAME = ...;` to module top level, or omit `const` for a local variable (`Type name = ...;`)",
                    ));
            }
            TokenKind::Lifetime(ref lt) => return Err(self.trap_p0009_lifetime(lt, cur_span)),

            // ── Standard Zev statements ──
            TokenKind::LBrace => {
                let b = self.parse_block()?;
                let span = start.merge(b.span);
                Ok(Stmt::with_attrs(attrs, StmtKind::Block(b), span))
            }
            TokenKind::KwReturn => {
                let ret_tok = self.bump();
                // Trap `return error.BadCrc;` (`P0007`)
                if self.at(&TokenKind::KwError) {
                    let err_kw = self.bump();
                    return Err(self.trap_p0007(
                        "`return error...`",
                        ret_tok.span.merge(err_kw.span),
                    ));
                }
                // Trap `return Err(...);` (`P0007`)
                if matches!(self.peek_kind(), TokenKind::Ident(ref s) if s == "Err")
                    && self.peek_n_kind(1) == TokenKind::LParen
                {
                    let err_id = self.bump();
                    return Err(self.trap_p0007(
                        "`return Err(...)`",
                        ret_tok.span.merge(err_id.span),
                    ));
                }

                let val = if self.at(&TokenKind::Semi) {
                    None
                } else {
                    Some(self.parse_expr()?)
                };
                let semi = self.expect(&TokenKind::Semi)?;
                Ok(Stmt::with_attrs(
                    attrs,
                    StmtKind::Return(val),
                    start.merge(semi.span),
                ))
            }
            TokenKind::KwError => {
                self.bump();
                if self.at(&TokenKind::Semi) {
                    let semi = self.bump();
                    return Ok(Stmt::with_attrs(
                        attrs,
                        StmtKind::Error {
                            reason: None,
                            partial: None,
                        },
                        start.merge(semi.span),
                    ));
                }
                let reason = self.parse_expr()?;
                let partial = if self.eat(&TokenKind::Comma) {
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                let semi = self.expect(&TokenKind::Semi)?;
                Ok(Stmt::with_attrs(
                    attrs,
                    StmtKind::Error {
                        reason: Some(reason),
                        partial,
                    },
                    start.merge(semi.span),
                ))
            }
            TokenKind::KwIf => self.parse_if_stmt(attrs, start),
            TokenKind::KwWhile => self.parse_while_stmt(attrs, start),
            TokenKind::KwFor => self.parse_for_stmt(attrs, start),
            TokenKind::KwLoop => self.parse_loop_stmt(attrs, start),
            TokenKind::KwEvery => self.parse_bare_every_trap(),
            TokenKind::KwDefer => self.parse_defer_stmt(attrs, start),
            TokenKind::KwNoYield => {
                self.bump();
                let body = self.parse_block()?;
                let span = start.merge(body.span);
                Ok(Stmt::with_attrs(attrs, StmtKind::NoYield { body }, span))
            }
            TokenKind::KwCritical => {
                self.bump();
                let body = self.parse_block()?;
                let span = start.merge(body.span);
                Ok(Stmt::with_attrs(attrs, StmtKind::Critical { body }, span))
            }
            TokenKind::KwSwitch => self.parse_switch_stmt(attrs, start),
            TokenKind::KwSpawn => self.parse_spawn_stmt(attrs, start),
            TokenKind::KwContinue => {
                self.bump();
                let semi = self.expect(&TokenKind::Semi)?;
                Ok(Stmt::with_attrs(
                    attrs,
                    StmtKind::Continue,
                    start.merge(semi.span),
                ))
            }
            TokenKind::KwBreak => {
                self.bump();
                let semi = self.expect(&TokenKind::Semi)?;
                Ok(Stmt::with_attrs(
                    attrs,
                    StmtKind::Break,
                    start.merge(semi.span),
                ))
            }
            TokenKind::KwRestart => {
                self.bump();
                let semi = self.expect(&TokenKind::Semi)?;
                Ok(Stmt::with_attrs(
                    attrs,
                    StmtKind::Restart,
                    start.merge(semi.span),
                ))
            }
            TokenKind::Ident(ref s)
                if s == "checkpoint"
                    && self.peek_n_kind(1) == TokenKind::LParen
                    && self.peek_n_kind(2) == TokenKind::RParen
                    && self.peek_n_kind(3) == TokenKind::Semi =>
            {
                self.bump(); // `checkpoint`
                self.bump(); // `(`
                self.bump(); // `)`
                let semi = self.bump(); // `;`
                Ok(Stmt::with_attrs(
                    attrs,
                    StmtKind::Checkpoint,
                    start.merge(semi.span),
                ))
            }
            _ => {
                // Trap `x: u16 = 42;` in statement position (`P0002` or `P0009`)
                if matches!(self.peek_kind(), TokenKind::Ident(_))
                    && self.peek_n_kind(1) == TokenKind::Colon
                {
                    let (_, _) = self.parse_field_or_param("variable declaration")?;
                }

                // Trap `!u16 x = ...;` or `ParseErr!u16 x = ...;` in statement position (`P0007`)
                if self.at(&TokenKind::Bang) {
                    if let TokenKind::Ident(ref tname) = self.peek_n_kind(1) {
                        if PrimType::from_name(tname).is_some()
                            || matches!(self.peek_n_kind(2), TokenKind::Ident(_))
                        {
                            return Err(self.trap_p0007(
                                "Zig-style `!Type` error-union variable type",
                                cur_span.merge(self.peek_n_span(1)),
                            ));
                        }
                    }
                }
                if matches!(self.peek_kind(), TokenKind::Ident(_))
                    && self.peek_n_kind(1) == TokenKind::Bang
                    && matches!(
                        self.peek_n_kind(2),
                        TokenKind::Ident(_) | TokenKind::LBracket | TokenKind::Star
                    )
                {
                    return Err(self.trap_p0007(
                        "Zig-style `ErrType!Type` error-union variable type",
                        cur_span.merge(self.peek_n_span(1)),
                    ));
                }

                if self.looks_like_state_var_decl() {
                    self.bump(); // `state`
                    let (ty, name) = self.parse_typed_declarator("`state` variable declaration")?;
                    let init = if self.eat(&TokenKind::Eq) {
                        Some(self.parse_expr()?)
                    } else {
                        None
                    };
                    let semi = self.expect(&TokenKind::Semi)?;
                    self.record_local_value(&name.name);
                    return Ok(Stmt::with_attrs(
                        attrs,
                        StmtKind::StateVar { ty, name, init },
                        start.merge(semi.span),
                    ));
                }

                if self.looks_like_var_decl() {
                    let (ty, name) = self.parse_typed_declarator("variable declaration")?;
                    let init = if self.eat(&TokenKind::Eq) {
                        Some(self.parse_expr()?)
                    } else {
                        None
                    };
                    let semi = self.expect(&TokenKind::Semi)?;
                    self.record_local_value(&name.name);
                    return Ok(Stmt::with_attrs(
                        attrs,
                        StmtKind::VarDecl { ty, name, init },
                        start.merge(semi.span),
                    ));
                }

                if self.eat(&TokenKind::PlusPlus) {
                    let plus_span = self.prev_span();
                    let target = self.parse_unary_expr()?;
                    let semi = self.expect(&TokenKind::Semi)?;
                    return Ok(Stmt::with_attrs(
                        attrs,
                        StmtKind::Assign {
                            target,
                            op: AssignOp::AddAssign,
                            value: Expr::new(ExprKind::Literal(Literal::Int(1)), plus_span),
                        },
                        start.merge(semi.span),
                    ));
                }
                if self.eat(&TokenKind::MinusMinus) {
                    let minus_span = self.prev_span();
                    let target = self.parse_unary_expr()?;
                    let semi = self.expect(&TokenKind::Semi)?;
                    return Ok(Stmt::with_attrs(
                        attrs,
                        StmtKind::Assign {
                            target,
                            op: AssignOp::SubAssign,
                            value: Expr::new(ExprKind::Literal(Literal::Int(1)), minus_span),
                        },
                        start.merge(semi.span),
                    ));
                }

                let expr = self.parse_expr()?;
                if self.eat(&TokenKind::PlusPlus) {
                    let plus_span = self.prev_span();
                    let semi = self.expect(&TokenKind::Semi)?;
                    Ok(Stmt::with_attrs(
                        attrs,
                        StmtKind::Assign {
                            target: expr,
                            op: AssignOp::AddAssign,
                            value: Expr::new(ExprKind::Literal(Literal::Int(1)), plus_span),
                        },
                        start.merge(semi.span),
                    ))
                } else if self.eat(&TokenKind::MinusMinus) {
                    let minus_span = self.prev_span();
                    let semi = self.expect(&TokenKind::Semi)?;
                    Ok(Stmt::with_attrs(
                        attrs,
                        StmtKind::Assign {
                            target: expr,
                            op: AssignOp::SubAssign,
                            value: Expr::new(ExprKind::Literal(Literal::Int(1)), minus_span),
                        },
                        start.merge(semi.span),
                    ))
                } else if let Some(op) = self.eat_assign_op() {
                    let value = self.parse_expr()?;
                    let semi = self.expect(&TokenKind::Semi)?;
                    Ok(Stmt::with_attrs(
                        attrs,
                        StmtKind::Assign {
                            target: expr,
                            op,
                            value,
                        },
                        start.merge(semi.span),
                    ))
                } else if self.eat(&TokenKind::Semi) {
                    let semi_span = self.prev_span();
                    Ok(Stmt::with_attrs(
                        attrs,
                        StmtKind::Expr(expr),
                        start.merge(semi_span),
                    ))
                } else if allow_tail_expr && self.at(&TokenKind::RBrace) {
                    let end_span = expr.span;
                    Ok(Stmt::with_attrs(
                        attrs,
                        StmtKind::Expr(expr),
                        start.merge(end_span),
                    ))
                } else {
                    let semi = self.expect(&TokenKind::Semi)?;
                    Ok(Stmt::with_attrs(
                        attrs,
                        StmtKind::Expr(expr),
                        start.merge(semi.span),
                    ))
                }
            }
        }
    }

    fn looks_like_state_var_decl(&mut self) -> bool {
        if !matches!(self.peek_kind(), TokenKind::Ident(ref s) if s == "state") {
            return false;
        }
        let starts_like_type = match self.peek_n_kind(1) {
            TokenKind::KwConst | TokenKind::KwFn | TokenKind::Star | TokenKind::LBracket => true,
            TokenKind::Ident(ref t) => {
                (PrimType::from_name(t).is_some()
                    && self.peek_n_kind(2) != TokenKind::Dot
                    && self.peek_n_kind(2) != TokenKind::KwMut)
                    || matches!(
                        self.peek_n_kind(2),
                        TokenKind::Star
                            | TokenKind::LBracket
                            | TokenKind::Lt
                            | TokenKind::Ident(_)
                            | TokenKind::Dot
                            | TokenKind::ColonColon
                    )
            }
            _ => false,
        };
        if !starts_like_type {
            return false;
        }
        let cp = self.save();
        self.bump(); // `state`
        let ok = self.looks_like_var_decl();
        self.restore(cp);
        ok
    }

    /// Speculatively check whether the upcoming tokens form a C-style local variable declaration
    /// (`Type name = ...;` or `Type name;`).
    fn looks_like_var_decl(&mut self) -> bool {
        match self.peek_kind() {
            TokenKind::KwConst | TokenKind::KwFn | TokenKind::Star | TokenKind::LBracket => {}
            TokenKind::Ident(ref s) => {
                if PrimType::from_name(s).is_some()
                    && self.peek_n_kind(1) != TokenKind::Dot
                    && self.peek_n_kind(1) != TokenKind::KwMut
                {
                    return true;
                }
            }
            _ => return false,
        }

        let cp = self.save();
        let ok = match self.parse_typed_declarator("lookahead") {
            Ok((_, _)) => matches!(self.peek_kind(), TokenKind::Eq | TokenKind::Semi),
            Err(ref e) if e.code.is_some() => {
                // If `parse_typed_declarator` hit an explicit foreign-idiom trap (e.g. `u16 mut x = 1;`
                // or `*mut u8 p = ...;` or `Reader<'a> r = ...;`), let the caller re-run it to report that trap!
                true
            }
            Err(_) => false,
        };
        self.restore(cp);
        ok
    }

    fn eat_assign_op(&mut self) -> Option<AssignOp> {
        let op = match self.peek_kind() {
            TokenKind::Eq => AssignOp::Assign,
            TokenKind::PlusEq => AssignOp::AddAssign,
            TokenKind::MinusEq => AssignOp::SubAssign,
            TokenKind::StarEq => AssignOp::MulAssign,
            TokenKind::SlashEq => AssignOp::DivAssign,
            TokenKind::PercentEq => AssignOp::RemAssign,
            TokenKind::AmpEq => AssignOp::BitAndAssign,
            TokenKind::PipeEq => AssignOp::BitOrAssign,
            TokenKind::CaretEq => AssignOp::BitXorAssign,
            TokenKind::ShlEq => AssignOp::ShlAssign,
            TokenKind::ShrEq => AssignOp::ShrAssign,
            _ => return None,
        };
        self.bump();
        Some(op)
    }

    fn parse_condition_expr(&mut self) -> Result<Expr, ParseError> {
        let prev = self.allow_struct_lit;
        self.allow_struct_lit = false;
        let expr = self.parse_expr()?;
        self.allow_struct_lit = prev;
        Ok(expr)
    }

    fn parse_if_stmt(&mut self, attrs: Vec<Attr>, start: Span) -> Result<Stmt, ParseError> {
        self.expect(&TokenKind::KwIf)?;
        // Also trap Rust `if let Some(x) = ...` -> P0001!
        if self.at(&TokenKind::KwLet) {
            return Err(self.trap_p0001("let", self.peek_span()));
        }
        let cond = self.parse_condition_expr()?;
        let then_block = self.parse_block()?;
        let mut end_span = then_block.span;

        let else_block = if self.eat(&TokenKind::KwElse) {
            if self.at(&TokenKind::KwIf) {
                let elif_start = self.peek_span();
                let elif_stmt = self.parse_if_stmt(Vec::new(), elif_start)?;
                end_span = elif_stmt.span;
                Some(Block::new(vec![elif_stmt], elif_start.merge(end_span)))
            } else {
                let b = self.parse_block()?;
                end_span = b.span;
                Some(b)
            }
        } else {
            None
        };

        Ok(Stmt::with_attrs(
            attrs,
            StmtKind::If {
                cond,
                then_block,
                else_block,
            },
            start.merge(end_span),
        ))
    }

    fn parse_while_stmt(&mut self, attrs: Vec<Attr>, start: Span) -> Result<Stmt, ParseError> {
        self.expect(&TokenKind::KwWhile)?;
        if self.at(&TokenKind::KwLet) {
            return Err(self.trap_p0001("let", self.peek_span()));
        }
        let cond = self.parse_condition_expr()?;
        let body = self.parse_block()?;
        let span = start.merge(body.span);
        Ok(Stmt::with_attrs(attrs, StmtKind::While { cond, body }, span))
    }

    /// Parse `loop { ... }` or `loop every <period_expr> { ... }`.
    fn parse_loop_stmt(&mut self, attrs: Vec<Attr>, start: Span) -> Result<Stmt, ParseError> {
        self.expect(&TokenKind::KwLoop)?;
        if self.eat(&TokenKind::KwEvery) {
            if self.at(&TokenKind::LBrace) || self.at(&TokenKind::Eof) {
                return Err(self
                    .err_at(
                        format!(
                            "expected period expression (e.g. `20ms` or `period_ms`) after `loop every`, found {}",
                            self.peek_kind()
                        ),
                        self.peek_span(),
                    )
                    .with_doc_topic("tasks")
                    .with_hint("write `loop every 20ms { ... }` or `loop every period_var { ... }`"));
            }
            // Parse period expression with `allow_struct_lit = false` so `loop every period_ms { ... }`
            // never treats `period_ms { ... }` as a struct literal (`loop every (period_ms) { ... }` works too).
            let period = self.parse_condition_expr()?;
            let body = self.parse_block()?;
            let span = start.merge(body.span);
            Ok(Stmt::with_attrs(
                attrs,
                StmtKind::LoopEvery {
                    period: Box::new(period),
                    body,
                },
                span,
            ))
        } else {
            let body = self.parse_block()?;
            let span = start.merge(body.span);
            Ok(Stmt::with_attrs(attrs, StmtKind::Loop { body }, span))
        }
    }

    /// Consume bare `every ...` up to `{` and emit `P0011` guiding users to `loop every <period> { ... }`.
    fn parse_bare_every_trap(&mut self) -> Result<Stmt, ParseError> {
        let every_tok = self.expect(&TokenKind::KwEvery)?;
        let period_start = self.peek_span();
        let mut last_span = every_tok.span;
        let mut has_period_toks = false;
        while !self.at(&TokenKind::LBrace)
            && !self.at(&TokenKind::Semi)
            && !self.at(&TokenKind::RBrace)
            && !self.at(&TokenKind::Eof)
        {
            last_span = self.bump().span;
            has_period_toks = true;
        }
        let period_str = if has_period_toks {
            self.source
                .get(period_start.start as usize..last_span.end as usize)
                .map(str::trim)
        } else {
            None
        };
        Err(self.trap_p0011(every_tok.span.merge(last_span), period_str))
    }

    /// Parse bounded `for (Type x in start..end) { ... }`, `for (Type x in start..=end) { ... }`,
    /// or `for (Type x in slice_or_array) { ... }`; trap C-style `for (;;)` and untyped `for` as `P0008`.
    fn parse_for_stmt(&mut self, attrs: Vec<Attr>, start: Span) -> Result<Stmt, ParseError> {
        self.expect(&TokenKind::KwFor)?;

        if !self.eat(&TokenKind::LParen) {
            return Err(self.trap_p0008(start.merge(self.peek_span())));
        }

        if self.at(&TokenKind::Semi) || self.at(&TokenKind::RParen) {
            return Err(self.trap_p0008(start.merge(self.peek_span())));
        }

        let (ty, var_name) = match self.parse_typed_declarator("`for` loop binding") {
            Ok(pair) => pair,
            Err(e) => {
                // Preserve specific type traps like `chan<...>` (`P0012`) or `&mut`/`'a` (`P0009`);
                // map untyped / C-style `for (i = 0; ...)` or `for (let i ...)` to `P0008`.
                if matches!(e.code, Some("P0009" | "P0012")) {
                    return Err(e);
                }
                return Err(self.trap_p0008(start.merge(self.peek_span())));
            }
        };

        // Contextual `"in"` check right after `(TypeExpr, Ident)`
        let has_in = matches!(self.peek_kind(), TokenKind::Ident(ref s) if s == "in");
        if !has_in {
            return Err(self.trap_p0008(start.merge(self.peek_span())));
        }
        self.bump(); // contextual `"in"`

        let prev_allow = self.allow_struct_lit;
        self.allow_struct_lit = true;

        let first = match self.parse_expr() {
            Ok(e) => e,
            Err(err) => {
                self.allow_struct_lit = prev_allow;
                return Err(err);
            }
        };

        if self.at(&TokenKind::DotDot) || self.at(&TokenKind::DotDotEq) {
            let inclusive = self.bump().kind == TokenKind::DotDotEq;
            let end_expr = match self.parse_expr() {
                Ok(e) => e,
                Err(err) => {
                    self.allow_struct_lit = prev_allow;
                    return Err(err);
                }
            };
            self.allow_struct_lit = prev_allow;
            self.expect(&TokenKind::RParen)?;
            self.push_value_scope();
            self.record_local_value(&var_name.name);
            let body_res = self.parse_block();
            self.pop_value_scope();
            let body = body_res?;
            let span = start.merge(body.span);
            Ok(Stmt::with_attrs(
                attrs,
                StmtKind::ForRange {
                    var_ty: ty,
                    var_name,
                    start: Box::new(first),
                    end: Box::new(end_expr),
                    inclusive,
                    body,
                },
                span,
            ))
        } else {
            self.allow_struct_lit = prev_allow;
            self.expect(&TokenKind::RParen)?;
            self.push_value_scope();
            self.record_local_value(&var_name.name);
            let body_res = self.parse_block();
            self.pop_value_scope();
            let body = body_res?;
            let span = start.merge(body.span);
            Ok(Stmt::with_attrs(
                attrs,
                StmtKind::ForEach {
                    elem_ty: ty,
                    elem_name: var_name,
                    iter: Box::new(first),
                    body,
                },
                span,
            ))
        }
    }

    /// Parse `defer expr_or_stmt;` or `defer { stmt* }` (no trailing `;` required on `{ ... }`).
    fn parse_defer_stmt(&mut self, attrs: Vec<Attr>, start: Span) -> Result<Stmt, ParseError> {
        self.expect(&TokenKind::KwDefer)?;
        if self.at(&TokenKind::LBrace) {
            let b = self.parse_block()?;
            let b_span = b.span;
            let inner = Stmt::new(StmtKind::Block(b), b_span);
            Ok(Stmt::with_attrs(
                attrs,
                StmtKind::Defer(Box::new(inner)),
                start.merge(b_span),
            ))
        } else {
            let inner = self.parse_stmt_in_block(false)?;
            let end = inner.span;
            Ok(Stmt::with_attrs(
                attrs,
                StmtKind::Defer(Box::new(inner)),
                start.merge(end),
            ))
        }
    }

    fn parse_switch_stmt(&mut self, attrs: Vec<Attr>, start: Span) -> Result<Stmt, ParseError> {
        self.expect(&TokenKind::KwSwitch)?;
        let scrutinee = self.parse_condition_expr()?;
        self.expect(&TokenKind::LBrace)?;

        let mut arms = Vec::new();
        while !self.at(&TokenKind::RBrace) && !self.at(&TokenKind::Eof) {
            let arm_start = self.peek_span();
            let pat = self.parse_switch_pat()?;
            self.expect(&TokenKind::FatArrow)?;

            let body = if self.at(&TokenKind::LBrace) {
                let b = self.parse_block()?;
                let _ = self.eat(&TokenKind::Comma);
                b
            } else {
                let stmt = self.parse_switch_single_arm_stmt()?;
                let s_span = stmt.span;
                Block::new(vec![stmt], s_span)
            };

            let arm_span = arm_start.merge(body.span);
            arms.push(SwitchArm {
                pat,
                body,
                span: arm_span,
            });
        }

        let rb = self.expect(&TokenKind::RBrace)?;
        Ok(Stmt::with_attrs(
            attrs,
            StmtKind::Switch { scrutinee, arms },
            start.merge(rb.span),
        ))
    }

    fn parse_switch_pat(&mut self) -> Result<SwitchPat, ParseError> {
        let tok = self.peek_token();
        match tok.kind {
            TokenKind::KwDefault => {
                self.bump();
                Ok(SwitchPat::Wildcard(tok.span))
            }
            TokenKind::Ident(ref s) if s == "_" => {
                self.bump();
                Ok(SwitchPat::Wildcard(tok.span))
            }
            TokenKind::Ident(_) => {
                let first = self.parse_ident()?;
                let mut rest = Vec::new();
                while self.eat(&TokenKind::Dot) || self.eat(&TokenKind::ColonColon) {
                    rest.push(self.parse_ident()?);
                }
                if rest.is_empty() {
                    let span = first.span;
                    Ok(SwitchPat::Ident {
                        prefix: None,
                        name: first,
                        span,
                    })
                } else if rest.len() == 1 {
                    let name = rest.pop().expect("len 1");
                    let span = first.span.merge(name.span);
                    Ok(SwitchPat::Ident {
                        prefix: Some(first),
                        name,
                        span,
                    })
                } else {
                    let mut path = Vec::with_capacity(1 + rest.len());
                    path.push(first);
                    path.extend(rest);
                    let span = path[0].span.merge(path.last().unwrap().span);
                    Ok(SwitchPat::Qualified { path, span })
                }
            }
            TokenKind::Int(n) => {
                self.bump();
                Ok(SwitchPat::Literal(Literal::Int(n), tok.span))
            }
            TokenKind::KwTrue => {
                self.bump();
                Ok(SwitchPat::Literal(Literal::Bool(true), tok.span))
            }
            TokenKind::KwFalse => {
                self.bump();
                Ok(SwitchPat::Literal(Literal::Bool(false), tok.span))
            }
            _ => Err(self
                .err_at(
                    format!("expected switch pattern, found {}", tok.kind),
                    tok.span,
                )
                .with_doc_topic("syntax")),
        }
    }

    /// Parse a single-statement switch arm body (`Variant => stmt,`), allowing `,` or `;` as terminator.
    fn parse_switch_single_arm_stmt(&mut self) -> Result<Stmt, ParseError> {
        let start = self.peek_span();
        match self.peek_kind() {
            TokenKind::KwReturn => {
                let ret_tok = self.bump();
                if self.at(&TokenKind::KwError) {
                    let err_kw = self.bump();
                    return Err(self.trap_p0007(
                        "`return error...`",
                        ret_tok.span.merge(err_kw.span),
                    ));
                }
                if matches!(self.peek_kind(), TokenKind::Ident(ref s) if s == "Err")
                    && self.peek_n_kind(1) == TokenKind::LParen
                {
                    let err_id = self.bump();
                    return Err(self.trap_p0007(
                        "`return Err(...)`",
                        ret_tok.span.merge(err_id.span),
                    ));
                }
                let val = if self.at(&TokenKind::Comma)
                    || self.at(&TokenKind::Semi)
                    || self.at(&TokenKind::RBrace)
                {
                    None
                } else {
                    Some(self.parse_expr()?)
                };
                let end = self.eat_arm_terminator(start)?;
                Ok(Stmt::new(StmtKind::Return(val), start.merge(end)))
            }
            TokenKind::KwError => {
                self.bump();
                if self.at(&TokenKind::Comma)
                    || self.at(&TokenKind::Semi)
                    || self.at(&TokenKind::RBrace)
                {
                    let end = self.eat_arm_terminator(start)?;
                    return Ok(Stmt::new(
                        StmtKind::Error {
                            reason: None,
                            partial: None,
                        },
                        start.merge(end),
                    ));
                }
                let reason = self.parse_expr()?;
                let partial = if self.at(&TokenKind::Comma) && !self.comma_starts_next_switch_arm()
                {
                    self.bump();
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                let end = self.eat_arm_terminator(start)?;
                Ok(Stmt::new(
                    StmtKind::Error {
                        reason: Some(reason),
                        partial,
                    },
                    start.merge(end),
                ))
            }
            TokenKind::KwContinue => {
                let t = self.bump();
                let end = self.eat_arm_terminator(t.span)?;
                Ok(Stmt::new(StmtKind::Continue, start.merge(end)))
            }
            TokenKind::KwBreak => {
                let t = self.bump();
                let end = self.eat_arm_terminator(t.span)?;
                Ok(Stmt::new(StmtKind::Break, start.merge(end)))
            }
            TokenKind::KwRestart => {
                let t = self.bump();
                let end = self.eat_arm_terminator(t.span)?;
                Ok(Stmt::new(StmtKind::Restart, start.merge(end)))
            }
            TokenKind::Ident(ref s)
                if s == "checkpoint"
                    && self.peek_n_kind(1) == TokenKind::LParen
                    && self.peek_n_kind(2) == TokenKind::RParen =>
            {
                self.bump();
                self.bump();
                let rp = self.bump();
                let end = self.eat_arm_terminator(rp.span)?;
                Ok(Stmt::new(StmtKind::Checkpoint, start.merge(end)))
            }
            _ => {
                let expr = self.parse_expr()?;
                if let Some(op) = self.eat_assign_op() {
                    let value = self.parse_expr()?;
                    let end = self.eat_arm_terminator(value.span)?;
                    Ok(Stmt::new(
                        StmtKind::Assign {
                            target: expr,
                            op,
                            value,
                        },
                        start.merge(end),
                    ))
                } else {
                    let e_span = expr.span;
                    let end = self.eat_arm_terminator(e_span)?;
                    Ok(Stmt::new(StmtKind::Expr(expr), start.merge(end)))
                }
            }
        }
    }

    fn comma_starts_next_switch_arm(&mut self) -> bool {
        let cp = self.save();
        let _ = self.eat(&TokenKind::Comma);
        let is_next_arm = self.at(&TokenKind::RBrace)
            || (self.parse_switch_pat().is_ok() && self.at(&TokenKind::FatArrow));
        self.restore(cp);
        is_next_arm
    }

    fn eat_arm_terminator(&mut self, fallback: Span) -> Result<Span, ParseError> {
        if self.eat(&TokenKind::Comma) || self.eat(&TokenKind::Semi) {
            Ok(self.prev_span())
        } else if self.at(&TokenKind::RBrace) {
            Ok(fallback)
        } else {
            Err(self
                .err_at(
                    format!(
                        "expected `,` or `;` after switch arm, found {}",
                        self.peek_kind()
                    ),
                    self.peek_span(),
                )
                .with_doc_topic("syntax"))
        }
    }

    fn parse_spawn_stmt(&mut self, mut attrs: Vec<Attr>, start: Span) -> Result<Stmt, ParseError> {
        self.expect(&TokenKind::KwSpawn)?;
        let first = self.parse_ident()?;
        let (module, task) = if self.eat(&TokenKind::Dot) || self.eat(&TokenKind::ColonColon) {
            (Some(first), self.parse_ident()?)
        } else {
            (None, first)
        };
        let args = self.parse_call_args()?;

        let mut priority: Option<Priority> = None;
        let mut core: Option<u32> = None;

        let mut trailing_attrs = self.parse_attrs()?;
        let mut retained_stmt_attrs = Vec::new();
        for attr in attrs.drain(..).chain(trailing_attrs.drain(..)) {
            match attr.kind {
                AttrKind::Priority(p) => {
                    if priority.is_some() {
                        return Err(self
                            .err_at("duplicate priority attribute on `spawn`", attr.span)
                            .with_doc_topic("tasks"));
                    }
                    priority = Some(p);
                }
                AttrKind::Core(c) => {
                    if core.is_some() {
                        return Err(self
                            .err_at("duplicate `@core(N)` attribute on `spawn`", attr.span)
                            .with_doc_topic("tasks"));
                    }
                    core = Some(c);
                }
                _ => retained_stmt_attrs.push(attr),
            }
        }

        let semi = self.expect(&TokenKind::Semi)?;
        let span = start.merge(semi.span);
        Ok(Stmt::with_attrs(
            retained_stmt_attrs,
            StmtKind::Spawn(SpawnStmt {
                module,
                task,
                args,
                priority,
                core,
                span,
            }),
            span,
        ))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Expressions (Precedence Climbing)
    // ─────────────────────────────────────────────────────────────────────────

    pub fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_or_fallback_expr()
    }

    /// Lowest precedence: `expr or fallback_expr` (right-associative).
    fn parse_or_fallback_expr(&mut self) -> Result<Expr, ParseError> {
        let lhs = self.parse_logical_or_expr()?;
        if self.eat(&TokenKind::KwOr) {
            let rhs = self.parse_or_fallback_expr()?;
            let span = lhs.span.merge(rhs.span);
            Ok(Expr::new(
                ExprKind::Or {
                    expr: Box::new(lhs),
                    fallback: Box::new(rhs),
                },
                span,
            ))
        } else {
            Ok(lhs)
        }
    }

    fn parse_logical_or_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_logical_and_expr()?;
        while self.eat(&TokenKind::PipePipe) {
            let rhs = self.parse_logical_and_expr()?;
            let span = lhs.span.merge(rhs.span);
            lhs = Expr::new(
                ExprKind::Binary {
                    op: BinaryOp::Or,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            );
        }
        Ok(lhs)
    }

    fn parse_logical_and_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_equality_expr()?;
        while self.eat(&TokenKind::AmpAmp) {
            let rhs = self.parse_equality_expr()?;
            let span = lhs.span.merge(rhs.span);
            lhs = Expr::new(
                ExprKind::Binary {
                    op: BinaryOp::And,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            );
        }
        Ok(lhs)
    }

    fn parse_equality_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_relational_expr()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::EqEq => BinaryOp::Eq,
                TokenKind::BangEq => BinaryOp::Ne,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_relational_expr()?;
            let span = lhs.span.merge(rhs.span);
            lhs = Expr::new(
                ExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            );
        }
        Ok(lhs)
    }

    fn parse_relational_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_bitor_expr()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Lt => BinaryOp::Lt,
                TokenKind::LtEq => BinaryOp::Le,
                TokenKind::Gt => BinaryOp::Gt,
                TokenKind::GtEq => BinaryOp::Ge,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_bitor_expr()?;
            let span = lhs.span.merge(rhs.span);
            lhs = Expr::new(
                ExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            );
        }
        Ok(lhs)
    }

    fn parse_bitor_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_bitxor_expr()?;
        while self.eat(&TokenKind::Pipe) {
            let rhs = self.parse_bitxor_expr()?;
            let span = lhs.span.merge(rhs.span);
            lhs = Expr::new(
                ExprKind::Binary {
                    op: BinaryOp::BitOr,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            );
        }
        Ok(lhs)
    }

    fn parse_bitxor_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_bitand_expr()?;
        while self.eat(&TokenKind::Caret) {
            let rhs = self.parse_bitand_expr()?;
            let span = lhs.span.merge(rhs.span);
            lhs = Expr::new(
                ExprKind::Binary {
                    op: BinaryOp::BitXor,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            );
        }
        Ok(lhs)
    }

    fn parse_bitand_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_shift_expr()?;
        while self.eat(&TokenKind::Amp) {
            let rhs = self.parse_shift_expr()?;
            let span = lhs.span.merge(rhs.span);
            lhs = Expr::new(
                ExprKind::Binary {
                    op: BinaryOp::BitAnd,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            );
        }
        Ok(lhs)
    }

    fn parse_shift_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_additive_expr()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Shl => BinaryOp::Shl,
                TokenKind::Shr => BinaryOp::Shr,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_additive_expr()?;
            let span = lhs.span.merge(rhs.span);
            lhs = Expr::new(
                ExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            );
        }
        Ok(lhs)
    }

    fn parse_additive_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_multiplicative_expr()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_multiplicative_expr()?;
            let span = lhs.span.merge(rhs.span);
            lhs = Expr::new(
                ExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            );
        }
        Ok(lhs)
    }

    fn parse_multiplicative_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_cast_expr()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                TokenKind::Percent => BinaryOp::Rem,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_cast_expr()?;
            let span = lhs.span.merge(rhs.span);
            lhs = Expr::new(
                ExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            );
        }
        Ok(lhs)
    }

    /// Postfix `expr as Type`.
    fn parse_cast_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_unary_expr()?;
        while self.eat(&TokenKind::KwAs) {
            let ty = self.parse_type()?;
            let span = expr.span.merge(ty.span);
            expr = Expr::new(
                ExprKind::Cast {
                    expr: Box::new(expr),
                    ty,
                },
                span,
            );
        }
        Ok(expr)
    }

    fn parse_unary_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.peek_span();

        // Foreign-idiom traps in prefix/unary expression position:
        match self.peek_kind() {
            TokenKind::KwTry => {
                return Err(self.trap_p0007("`try expr`", start));
            }
            TokenKind::KwAwait => {
                return Err(self.trap_p0003("await", start));
            }
            TokenKind::KwAsync => {
                return Err(self.trap_p0003("async", start));
            }
            TokenKind::KwMatch => {
                return Err(self.trap_p0005(start));
            }
            TokenKind::KwSelect => {
                return Err(self.trap_p0004(start));
            }
            TokenKind::Amp => {
                // Trap `&mut expr` and `&'a expr` (`P0009`)
                if self.peek_n_kind(1) == TokenKind::KwMut {
                    let amp = self.bump();
                    let m = self.bump();
                    return Err(self.trap_p0009_mut_or_ref("&mut", amp.span.merge(m.span)));
                }
                if let TokenKind::Lifetime(ref lt) = self.peek_n_kind(1) {
                    let amp = self.bump();
                    let lt_span = self.peek_span();
                    return Err(self.trap_p0009_lifetime(lt, amp.span.merge(lt_span)));
                }
            }
            _ => {}
        }

        // Check C-style cast `(PrimType)expr` or `(*PrimType)expr`
        if self.at(&TokenKind::LParen) {
            if let Some(ty) = self.try_parse_c_cast_type() {
                let inner = self.parse_unary_expr()?;
                let span = start.merge(inner.span);
                return Ok(Expr::new(
                    ExprKind::Cast {
                        expr: Box::new(inner),
                        ty,
                    },
                    span,
                ));
            }
        }

        let op = match self.peek_kind() {
            TokenKind::Amp => Some(UnaryOp::AddrOf),
            TokenKind::Star => Some(UnaryOp::Deref),
            TokenKind::Bang => Some(UnaryOp::Not),
            TokenKind::Tilde => Some(UnaryOp::BitNot),
            TokenKind::Minus => Some(UnaryOp::Neg),
            _ => None,
        };

        if let Some(op) = op {
            self.bump();
            let inner = self.parse_unary_expr()?;
            let span = start.merge(inner.span);
            Ok(Expr::new(
                ExprKind::Unary {
                    op,
                    expr: Box::new(inner),
                },
                span,
            ))
        } else {
            self.parse_postfix_expr()
        }
    }

    /// Speculatively check if `(` starts a C-style cast involving a built-in `PrimType`
    /// (e.g. `(u16)x`, `(i32)(a - b)`, `(*u8)addr`, `(u32*)addr`).
    fn try_parse_c_cast_type(&mut self) -> Option<TypeExpr> {
        let cp = self.save();
        self.bump(); // `(`
        let ty = match self.parse_type() {
            Ok(t) if type_contains_builtin_prim(&t) => t,
            _ => {
                self.restore(cp);
                return None;
            }
        };
        if !self.eat(&TokenKind::RParen) || !self.can_start_unary_expr() {
            self.restore(cp);
            return None;
        }
        Some(ty)
    }

    fn can_start_unary_expr(&self) -> bool {
        matches!(
            self.peek_kind(),
            TokenKind::Ident(_)
                | TokenKind::Int(_)
                | TokenKind::FloatLit { .. }
                | TokenKind::Duration(_, _)
                | TokenKind::String(_)
                | TokenKind::FormatString(_)
                | TokenKind::KwTrue
                | TokenKind::KwFalse
                | TokenKind::LParen
                | TokenKind::Amp
                | TokenKind::Star
                | TokenKind::Bang
                | TokenKind::Tilde
                | TokenKind::Minus
        )
    }

    fn parse_postfix_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary_expr()?;

        loop {
            match self.peek_kind() {
                TokenKind::Dot
                | TokenKind::Arrow
                | TokenKind::ColonColon
                    if !(self.at(&TokenKind::ColonColon)
                        && self.peek_n_kind(1) == TokenKind::Lt) =>
                {
                    let dot_tok = self.bump();
                    let field = self.parse_word_ident("field or method name")?;
                    // Trap `.await` (`P0003`) and `.match` (`P0005`)
                    if field.name == "await" {
                        return Err(self.trap_p0003(".await", dot_tok.span.merge(field.span)));
                    }
                    if field.name == "match" && self.at(&TokenKind::LBrace) {
                        return Err(self.trap_p0005(dot_tok.span.merge(field.span)));
                    }

                    // Support module-qualified generic function calls (`mod.clamp<f32>(...)` ONLY when `mod`
                    // is an `import` alias so struct field comparisons `s.a < b, c > (d + 1)` never speculate;
                    // `::` turbofish and module-qualified struct literals (`mod.Box<u32> { ... }`) also handled).
                    let is_mod_ident = dot_tok.kind == TokenKind::Dot
                        && matches!(&expr.kind, ExprKind::Ident(_));
                    let allow_bare_lt_call = match dot_tok.kind {
                        TokenKind::Dot => matches!(
                            &expr.kind,
                            ExprKind::Ident(id)
                                if !self.is_value_name(&id.name)
                                    && self.import_aliases.contains(&id.name)
                                    && self.can_speculate_bare_generic_call(&field.name)
                        ),
                        TokenKind::ColonColon => self
                            .known_generics
                            .map_or(true, |set| set.contains(&field.name)),
                        _ => false,
                    };
                    let allow_bare_lt_struct_lit = is_mod_ident
                        && self.allow_struct_lit
                        && matches!(&expr.kind, ExprKind::Ident(id) if !self.is_value_name(&id.name));

                    if self.at(&TokenKind::ColonColon) && self.peek_n_kind(1) == TokenKind::Lt {
                        self.bump(); // `::`
                        let (generic_args, _) = self.parse_generic_args()?;
                        if self.at(&TokenKind::LParen) {
                            let args = self.parse_call_args()?;
                            let callee_span = expr.span.merge(field.span);
                            let callee = Expr::new(
                                ExprKind::Field {
                                    target: Box::new(expr),
                                    field,
                                },
                                callee_span,
                            );
                            let span = callee_span.merge(self.prev_span());
                            expr = Expr::new(
                                ExprKind::Call {
                                    callee: Box::new(callee),
                                    generic_args,
                                    args,
                                },
                                span,
                            );
                        } else if is_mod_ident
                            && self.allow_struct_lit
                            && self.looks_like_struct_lit_body()
                        {
                            let expr_span = expr.span;
                            let ExprKind::Ident(module_ident) = expr.kind else {
                                unreachable!()
                            };
                            expr = self.parse_struct_lit_with_module(
                                Some(module_ident),
                                field,
                                generic_args,
                                expr_span,
                            )?;
                        } else {
                            return Err(self
                                .err_at(
                                    format!(
                                        "expected `(` or `{{ ... }}` after `::<...>` generic arguments, found {}",
                                        self.peek_kind()
                                    ),
                                    self.peek_span(),
                                )
                                .with_doc_topic("syntax"));
                        }
                        continue;
                    }

                    if (allow_bare_lt_call || allow_bare_lt_struct_lit) && self.at(&TokenKind::Lt)
                    {
                        let cp = self.save();
                        if let Some(generic_args) = self.parse_speculative_generic_args() {
                            if allow_bare_lt_call && self.at(&TokenKind::LParen) {
                                let args = self.parse_call_args()?;
                                let callee_span = expr.span.merge(field.span);
                                let callee = Expr::new(
                                    ExprKind::Field {
                                        target: Box::new(expr),
                                        field,
                                    },
                                    callee_span,
                                );
                                let span = callee_span.merge(self.prev_span());
                                expr = Expr::new(
                                    ExprKind::Call {
                                        callee: Box::new(callee),
                                        generic_args,
                                        args,
                                    },
                                    span,
                                );
                                continue;
                            } else if allow_bare_lt_struct_lit
                                && self.looks_like_struct_lit_body()
                            {
                                let expr_span = expr.span;
                                let ExprKind::Ident(module_ident) = expr.kind else {
                                    unreachable!()
                                };
                                expr = self.parse_struct_lit_with_module(
                                    Some(module_ident),
                                    field,
                                    generic_args,
                                    expr_span,
                                )?;
                                continue;
                            }
                        }
                        self.restore(cp);
                    }

                    if is_mod_ident && self.allow_struct_lit && self.looks_like_struct_lit_body() {
                        let expr_span = expr.span;
                        let ExprKind::Ident(module_ident) = expr.kind else {
                            unreachable!()
                        };
                        expr = self.parse_struct_lit_with_module(
                            Some(module_ident),
                            field,
                            Vec::new(),
                            expr_span,
                        )?;
                    } else if self.at(&TokenKind::LParen) {
                        let args = self.parse_call_args()?;
                        let span = expr.span.merge(self.prev_span());
                        expr = Expr::new(
                            ExprKind::MethodCall {
                                receiver: Box::new(expr),
                                method: field,
                                args,
                            },
                            span,
                        );
                    } else {
                        let span = expr.span.merge(field.span);
                        expr = Expr::new(
                            ExprKind::Field {
                                target: Box::new(expr),
                                field,
                            },
                            span,
                        );
                    }
                }
                TokenKind::LParen => {
                    let args = self.parse_call_args()?;
                    let span = expr.span.merge(self.prev_span());
                    expr = Expr::new(
                        ExprKind::Call {
                            callee: Box::new(expr),
                            generic_args: Vec::new(),
                            args,
                        },
                        span,
                    );
                }
                TokenKind::LBracket => {
                    expr = self.parse_index_or_slice(expr)?;
                }
                TokenKind::KwCatch => {
                    let clause = self.parse_catch_clause()?;
                    let span = expr.span.merge(clause.span);
                    expr = Expr::new(
                        ExprKind::Catch {
                            expr: Box::new(expr),
                            clause,
                        },
                        span,
                    );
                }
                TokenKind::Question => {
                    let q = self.bump();
                    return Err(self.trap_p0007("Rust-style `expr?` try operator", q.span));
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    /// Parse turbofish `::<Args>` unconditionally, or speculatively parse `<Args>` ONLY when
    /// `allow_bare_lt` holds and `<Args>` is immediately followed by `(`.
    fn parse_call_args(&mut self) -> Result<Vec<CallArg>, ParseError> {
        self.expect(&TokenKind::LParen)?;
        let prev_allow = self.allow_struct_lit;
        self.allow_struct_lit = true;

        let mut args = Vec::new();
        while !self.at(&TokenKind::RParen) && !self.at(&TokenKind::Eof) {
            let arg_start = self.peek_span();
            // Named argument: `timeout: 5s`, `boost: @critical`
            if matches!(self.peek_kind(), TokenKind::Ident(_))
                && self.peek_n_kind(1) == TokenKind::Colon
            {
                let name = self.parse_ident()?;
                self.expect(&TokenKind::Colon)?;
                let value = self.parse_expr()?;
                let span = arg_start.merge(value.span);
                args.push(CallArg::named(name, value, span));
            } else {
                let value = self.parse_expr()?;
                args.push(CallArg::positional(value));
            }

            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }

        self.allow_struct_lit = prev_allow;
        self.expect(&TokenKind::RParen)?;
        Ok(args)
    }

    fn parse_index_or_slice(&mut self, target: Expr) -> Result<Expr, ParseError> {
        self.expect(&TokenKind::LBracket)?;
        let prev_allow = self.allow_struct_lit;
        self.allow_struct_lit = true;

        if self.eat(&TokenKind::DotDot) {
            let end = if self.at(&TokenKind::RBracket) {
                None
            } else {
                Some(Box::new(self.parse_expr()?))
            };
            self.allow_struct_lit = prev_allow;
            let rb = self.expect(&TokenKind::RBracket)?;
            let span = target.span.merge(rb.span);
            return Ok(Expr::new(
                ExprKind::Slice {
                    target: Box::new(target),
                    start: None,
                    end,
                },
                span,
            ));
        }

        let first = self.parse_expr()?;
        if self.eat(&TokenKind::DotDot) {
            let end = if self.at(&TokenKind::RBracket) {
                None
            } else {
                Some(Box::new(self.parse_expr()?))
            };
            self.allow_struct_lit = prev_allow;
            let rb = self.expect(&TokenKind::RBracket)?;
            let span = target.span.merge(rb.span);
            Ok(Expr::new(
                ExprKind::Slice {
                    target: Box::new(target),
                    start: Some(Box::new(first)),
                    end,
                },
                span,
            ))
        } else {
            self.allow_struct_lit = prev_allow;
            let rb = self.expect(&TokenKind::RBracket)?;
            let span = target.span.merge(rb.span);
            Ok(Expr::new(
                ExprKind::Index {
                    target: Box::new(target),
                    index: Box::new(first),
                },
                span,
            ))
        }
    }

    fn parse_catch_clause(&mut self) -> Result<CatchClause, ParseError> {
        let catch_tok = self.expect(&TokenKind::KwCatch)?;

        // Trap Zig-style `catch |err|` or `catch ||` (`P0006`)
        if self.at(&TokenKind::Pipe) || self.at(&TokenKind::PipePipe) {
            let pipe_span = self.peek_span();
            return Err(self.trap_p0006(catch_tok.span.merge(pipe_span)));
        }

        let mut err_param = None;
        let mut partial_param = None;

        if self.eat(&TokenKind::LParen) {
            if !self.at(&TokenKind::RParen) {
                let p1_start = self.peek_span();
                let (ty1, name1) = self.parse_field_or_param("`catch` error binding")?;
                err_param = Some(CatchParam {
                    span: p1_start.merge(name1.span),
                    ty: ty1,
                    name: name1,
                });

                if self.eat(&TokenKind::Comma) && !self.at(&TokenKind::RParen) {
                    let p2_start = self.peek_span();
                    let (ty2, name2) =
                        self.parse_field_or_param("`catch` partial-progress binding")?;
                    partial_param = Some(CatchParam {
                        span: p2_start.merge(name2.span),
                        ty: ty2,
                        name: name2,
                    });
                    let _ = self.eat(&TokenKind::Comma);
                }
            }
            self.expect(&TokenKind::RParen)?;
        }

        self.push_value_scope();
        if let Some(ref ep) = err_param {
            self.record_local_value(&ep.name.name);
        }
        if let Some(ref pp) = partial_param {
            self.record_local_value(&pp.name.name);
        }
        let body_res = self.parse_catch_block();
        self.pop_value_scope();
        let body = body_res?;
        let span = catch_tok.span.merge(body.span);
        Ok(CatchClause {
            err_param,
            partial_param,
            body,
            span,
        })
    }

    fn parse_primary_expr(&mut self) -> Result<Expr, ParseError> {
        let tok = self.peek_token();
        match tok.kind {
            TokenKind::Int(n) => {
                self.bump();
                Ok(Expr::new(ExprKind::Literal(Literal::Int(n)), tok.span))
            }
            TokenKind::FloatLit { bits, suffix } => {
                self.bump();
                Ok(Expr::new(
                    ExprKind::Literal(Literal::Float { bits, suffix }),
                    tok.span,
                ))
            }
            TokenKind::Duration(value, unit) => {
                self.bump();
                Ok(Expr::new(
                    ExprKind::Literal(Literal::Duration(DurationLit {
                        value,
                        unit,
                        span: tok.span,
                    })),
                    tok.span,
                ))
            }
            TokenKind::KwTrue => {
                self.bump();
                Ok(Expr::new(ExprKind::Literal(Literal::Bool(true)), tok.span))
            }
            TokenKind::KwFalse => {
                self.bump();
                Ok(Expr::new(ExprKind::Literal(Literal::Bool(false)), tok.span))
            }
            TokenKind::String(s) => {
                self.bump();
                Ok(Expr::new(ExprKind::Literal(Literal::String(s)), tok.span))
            }
            TokenKind::FormatString(parts) => {
                self.bump();
                self.parse_format_string_expr(parts, tok.span)
            }
            TokenKind::At => {
                let attr = self.parse_attr()?;
                match attr.kind {
                    AttrKind::Priority(p) => Ok(Expr::new(ExprKind::Priority(p), attr.span)),
                    _ => Err(self
                        .err_at(
                            "only priority literals (`@critical`, `@high`, `@normal`, `@low`) may be used as expressions",
                            attr.span,
                        )
                        .with_doc_topic("tasks")),
                }
            }
            TokenKind::LParen => {
                self.bump();
                let prev_allow = self.allow_struct_lit;
                self.allow_struct_lit = true;
                let inner = self.parse_expr()?;
                self.allow_struct_lit = prev_allow;
                let rp = self.expect(&TokenKind::RParen)?;
                Ok(Expr::new(inner.kind, tok.span.merge(rp.span)))
            }
            TokenKind::LBracket => {
                self.bump();
                let prev_allow = self.allow_struct_lit;
                self.allow_struct_lit = true;
                let mut elems = Vec::new();
                while !self.at(&TokenKind::RBracket) && !self.at(&TokenKind::Eof) {
                    elems.push(self.parse_expr()?);
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
                self.allow_struct_lit = prev_allow;
                let rb = self.expect(&TokenKind::RBracket)?;
                Ok(Expr::new(
                    ExprKind::ArrayLit(elems),
                    tok.span.merge(rb.span),
                ))
            }
            TokenKind::Ident(_) => {
                let ident = self.parse_ident()?;
                let start = ident.span;

                // 1. Explicit turbofish `clamp::<f32>(...)` or `FixedVec::<u32, 4> { ... }`
                if self.at(&TokenKind::ColonColon) && self.peek_n_kind(1) == TokenKind::Lt {
                    self.bump(); // `::`
                    let (generic_args, _) = self.parse_generic_args()?;
                    if self.at(&TokenKind::LParen) {
                        let args = self.parse_call_args()?;
                        let span = start.merge(self.prev_span());
                        return Ok(Expr::new(
                            ExprKind::Call {
                                callee: Box::new(Expr::new(ExprKind::Ident(ident), start)),
                                generic_args,
                                args,
                            },
                            span,
                        ));
                    } else if self.allow_struct_lit && self.looks_like_struct_lit_body() {
                        return self.parse_struct_lit(ident, generic_args);
                    } else {
                        return Err(self
                            .err_at(
                                format!(
                                    "expected `(` or `{{ ... }}` after `::<...>` generic arguments, found {}",
                                    self.peek_kind()
                                ),
                                self.peek_span(),
                            )
                            .with_doc_topic("syntax"));
                    }
                }

                // 2. Speculative `clamp<f32>(...)` or `FixedVec<u32, 4> { ... }`
                //    Skipped immediately for value bindings (`value_scopes` / `top_level_values`) and,
                //    for bare `<...>(...)` calls, when `known_generics` is provided and excludes `ident`.
                if self.at(&TokenKind::Lt) && !self.is_value_name(&ident.name) {
                    let cp = self.save();
                    if let Some(generic_args) = self.parse_speculative_generic_args() {
                        if self.at(&TokenKind::LParen)
                            && self.can_speculate_bare_generic_call(&ident.name)
                        {
                            let args = self.parse_call_args()?;
                            let span = start.merge(self.prev_span());
                            return Ok(Expr::new(
                                ExprKind::Call {
                                    callee: Box::new(Expr::new(ExprKind::Ident(ident), start)),
                                    generic_args,
                                    args,
                                },
                                span,
                            ));
                        } else if self.allow_struct_lit && self.looks_like_struct_lit_body() {
                            return self.parse_struct_lit(ident, generic_args);
                        }
                    }
                    self.restore(cp);
                }

                if self.allow_struct_lit && self.looks_like_struct_lit_body() {
                    self.parse_struct_lit(ident, Vec::new())
                } else {
                    Ok(Expr::new(ExprKind::Ident(ident), start))
                }
            }
            TokenKind::Dot
                if matches!(
                    self.peek_n_kind(1),
                    TokenKind::Int(_) | TokenKind::FloatLit { .. }
                ) =>
            {
                let dot_tok = self.bump();
                let num_tok = self.bump();
                Err(self
                    .err_at(
                        "malformed float literal: leading digit is required before decimal point (e.g. `0.5`)",
                        dot_tok.span.merge(num_tok.span),
                    )
                    .with_doc_topic("syntax"))
            }
            TokenKind::KwError if self.peek_n_kind(1) == TokenKind::Dot => {
                // Zig `error.Variant` used in an expression (`P0007`)
                let e_tok = self.bump();
                let dot_tok = self.bump();
                Err(self.trap_p0007("Zig-style `error.Variant` value", e_tok.span.merge(dot_tok.span)))
            }
            _ => {
                if let Some(trap_err) = self.check_unexpected_token_trap(&tok) {
                    return Err(trap_err);
                }
                Err(self
                    .err_at(format!("expected expression, found {}", tok.kind), tok.span)
                    .with_doc_topic("syntax"))
            }
        }
    }

    fn parse_format_string_expr(
        &mut self,
        lexed_parts: Vec<LexedFormatPart>,
        span: Span,
    ) -> Result<Expr, ParseError> {
        let mut parts = Vec::with_capacity(lexed_parts.len());
        for lp in lexed_parts {
            match lp {
                LexedFormatPart::Literal(s) => {
                    parts.push(FormatPart::Literal(s));
                }
                LexedFormatPart::Hole {
                    tokens,
                    spec,
                    span: hole_span,
                } => {
                    let mut sub_parser = self.new_hole_subparser(tokens);
                    let expr = sub_parser.parse_expr()?;
                    if !sub_parser.at(&TokenKind::Eof) {
                        let extra = sub_parser.peek_token();
                        return Err(self
                            .err_at(
                                format!(
                                    "unexpected {} inside format string hole",
                                    extra.kind
                                ),
                                extra.span,
                            )
                            .with_doc_topic("syntax"));
                    }
                    parts.push(FormatPart::Hole {
                        expr: Box::new(expr),
                        spec,
                        span: hole_span,
                    });
                }
            }
        }
        Ok(Expr::new(ExprKind::FormatString(parts), span))
    }

    fn looks_like_struct_lit_body(&self) -> bool {
        if !self.at(&TokenKind::LBrace) {
            return false;
        }
        match self.peek_n_kind(1) {
            TokenKind::RBrace => true,
            TokenKind::Ident(_) => self.peek_n_kind(2) == TokenKind::Colon,
            TokenKind::Dot => {
                matches!(self.peek_n_kind(2), TokenKind::Ident(_))
                    && self.peek_n_kind(3) == TokenKind::Eq
            }
            _ => false,
        }
    }

    fn parse_struct_lit(
        &mut self,
        name: Ident,
        generic_args: Vec<GenericArg>,
    ) -> Result<Expr, ParseError> {
        let start = name.span;
        self.parse_struct_lit_with_module(None, name, generic_args, start)
    }

    fn parse_struct_lit_with_module(
        &mut self,
        module: Option<Ident>,
        name: Ident,
        generic_args: Vec<GenericArg>,
        start: Span,
    ) -> Result<Expr, ParseError> {
        self.expect(&TokenKind::LBrace)?;
        let prev_allow = self.allow_struct_lit;
        self.allow_struct_lit = true;

        let mut fields = Vec::new();
        while !self.at(&TokenKind::RBrace) && !self.at(&TokenKind::Eof) {
            let f_start = self.peek_span();
            let field_name = if self.eat(&TokenKind::Dot) {
                let id = self.parse_ident()?;
                self.expect(&TokenKind::Eq)?;
                id
            } else {
                let id = self.parse_ident()?;
                self.expect(&TokenKind::Colon)?;
                id
            };
            let value = self.parse_expr()?;
            let f_span = f_start.merge(value.span);
            fields.push(FieldInit {
                name: field_name,
                value,
                span: f_span,
            });
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }

        self.allow_struct_lit = prev_allow;
        let rb = self.expect(&TokenKind::RBrace)?;
        Ok(Expr::new(
            ExprKind::StructLit {
                module,
                name,
                generic_args,
                fields,
            },
            start.merge(rb.span),
        ))
    }
}

fn type_contains_builtin_prim(ty: &TypeExpr) -> bool {
    match &ty.kind {
        TypeKind::Prim(_) => true,
        TypeKind::Ptr(inner) | TypeKind::Slice(inner) => type_contains_builtin_prim(inner),
        TypeKind::Array { elem, .. } => type_contains_builtin_prim(elem),
        _ => false,
    }
}
