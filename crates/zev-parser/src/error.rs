use std::fmt;
use zev_ast::Span;

/// A syntax or lexical error with source location, snippet context, optional error code,
/// rewrite hint, and progressive-disclosure `zevc doc` topic pointer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    /// Human-readable error description.
    pub message: String,
    /// Offending token/character span (`line` and `col` are 1-indexed).
    pub span: Span,
    /// Optional filename or logical source label.
    pub file: Option<Box<str>>,
    /// Optional offending single line of source text for rich diagnostic formatting.
    pub source_line: Option<Box<str>>,
    /// Optional 1-shot Zev rewrite suggestion/hint.
    pub hint: Option<Box<str>>,
    /// Optional stable parser error code (e.g. `"P0001"`..`"P0010"`).
    pub code: Option<&'static str>,
    /// Optional `zevc doc <topic>` guide topic (e.g. `"syntax"`, `"yielding"`, `"errors"`, `"primitives"`, `"dma"`).
    pub doc_topic: Option<&'static str>,
}

impl ParseError {
    /// Create a new `ParseError` and extract the offending source line from `source`.
    pub fn new(message: impl Into<String>, span: Span, source: &str, file: Option<&str>) -> Self {
        let line_idx = span.line.saturating_sub(1) as usize;
        let source_line = source.lines().nth(line_idx).map(Box::from);
        Self {
            message: message.into(),
            span,
            file: file.map(Box::from),
            source_line,
            hint: None,
            code: None,
            doc_topic: None,
        }
    }

    /// Attach a helpful Zev rewrite hint to this error.
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into().into_boxed_str());
        self
    }

    /// Attach a stable parser diagnostic code (e.g. `"P0001"`).
    pub fn with_code(mut self, code: &'static str) -> Self {
        self.code = Some(code);
        self
    }

    /// Attach a relevant `zevc doc <topic>` documentation topic.
    pub fn with_doc_topic(mut self, topic: &'static str) -> Self {
        self.doc_topic = Some(topic);
        self
    }

    /// Return the primary `zevc doc ...` command an agent or user should run for this error.
    pub fn doc_command(&self) -> String {
        if let Some(code) = self.code {
            format!("zevc doc {code}")
        } else if let Some(topic) = self.doc_topic {
            format!("zevc doc {topic}")
        } else {
            "zevc doc syntax".to_owned()
        }
    }

    /// Format this error against an explicit `source` string (overriding any stored line).
    pub fn format_with_source(&self, source: &str) -> String {
        let mut clone = self.clone();
        if clone.source_line.is_none() {
            let line_idx = clone.span.line.saturating_sub(1) as usize;
            clone.source_line = source.lines().nth(line_idx).map(Box::from);
        }
        clone.to_string()
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = self.file.as_deref().unwrap_or("<input>");
        if let Some(code) = self.code {
            writeln!(
                f,
                "{}:{}:{}: error[{}]: {}",
                file, self.span.line, self.span.col, code, self.message
            )?;
        } else {
            writeln!(
                f,
                "{}:{}:{}: error: {}",
                file, self.span.line, self.span.col, self.message
            )?;
        }

        if let Some(line_str) = &self.source_line {
            let line_num = self.span.line.to_string();
            let pad = " ".repeat(line_num.len());
            writeln!(f, " {} |", pad)?;
            writeln!(f, " {} | {}", line_num, line_str)?;

            let char_col = self.span.col.saturating_sub(1) as usize;
            let total_chars = line_str.chars().count();
            let clamped_col = char_col.min(total_chars);
            let raw_width = self.span.end.saturating_sub(self.span.start) as usize;
            let caret_len = raw_width.clamp(1, total_chars.saturating_sub(clamped_col).max(1));
            let leading: String = line_str
                .chars()
                .take(clamped_col)
                .map(|c| if c == '\t' { '\t' } else { ' ' })
                .collect();
            writeln!(f, " {} | {}{}", pad, leading, "^".repeat(caret_len))?;
        }

        if let Some(hint) = &self.hint {
            writeln!(f, " = help: {}", hint)?;
        }

        match (self.code, self.doc_topic) {
            (Some(code), Some(topic)) => {
                writeln!(f, " = docs: run 'zevc doc {code}' or 'zevc doc {topic}'")?;
            }
            (Some(code), None) => {
                writeln!(f, " = docs: run 'zevc doc {code}'")?;
            }
            (None, Some(topic)) => {
                writeln!(f, " = docs: run 'zevc doc {topic}'")?;
            }
            (None, None) => {
                writeln!(f, " = docs: run 'zevc doc syntax'")?;
            }
        }

        Ok(())
    }
}

impl std::error::Error for ParseError {}
