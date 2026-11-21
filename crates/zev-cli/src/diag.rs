use std::fmt::Write;
use std::path::{Path, PathBuf};
use zev_ast::Span;

#[derive(Debug, Clone)]
pub struct SourceFile {
    pub path: PathBuf,
    pub src: String,
    line_starts: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct JsonDiag {
    pub code: Option<String>,
    pub file: String,
    pub line: usize,
    pub col: usize,
    pub message: String,
    pub help: Option<String>,
    pub doc_cmd: String,
}

impl SourceFile {
    pub fn load(path: &Path) -> Result<Self, String> {
        let src = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read `{}`: {e}", path.display()))?;
        Ok(Self::new(path.to_path_buf(), src))
    }

    pub fn new(path: PathBuf, src: String) -> Self {
        let mut line_starts = vec![0];
        for (idx, b) in src.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(idx + 1);
            }
        }
        Self {
            path,
            src,
            line_starts,
        }
    }

    /// Converts a byte offset into 1-based `(line, col)` safely even across multi-byte UTF-8.
    pub fn line_col(&self, byte_offset: usize) -> (usize, usize) {
        let clamped = self.clamp_char_boundary(byte_offset.min(self.src.len()));
        let line_idx = match self.line_starts.binary_search(&clamped) {
            Ok(idx) => idx,
            Err(idx) => idx.saturating_sub(1),
        };
        let line_start = self.line_starts[line_idx];
        let slice = &self.src[line_start..clamped];
        let col = slice.chars().count() + 1;
        (line_idx + 1, col)
    }

    fn clamp_char_boundary(&self, mut idx: usize) -> usize {
        while idx > 0 && !self.src.is_char_boundary(idx) {
            idx -= 1;
        }
        idx
    }

    pub fn line_text(&self, line_1based: usize) -> &str {
        if line_1based == 0 || line_1based > self.line_starts.len() {
            return "";
        }
        let start = self.line_starts[line_1based - 1];
        let end = if line_1based < self.line_starts.len() {
            self.line_starts[line_1based].saturating_sub(1)
        } else {
            self.src.len()
        };
        let raw = &self.src[start..end];
        raw.strip_suffix('\r').unwrap_or(raw)
    }

    /// Formats a rich compiler diagnostic with `error[CODE]`, source snippet, `= help:`, and `= docs:`.
    pub fn format_rich_span_error(
        &self,
        span: Span,
        code: Option<&str>,
        message: &str,
        help: Option<&str>,
        doc_topic: Option<&str>,
    ) -> String {
        let start_byte = self.clamp_char_boundary((span.start as usize).min(self.src.len()));
        let end_byte = self.clamp_char_boundary((span.end as usize).min(self.src.len()));
        let (line, col) = self.line_col(start_byte);
        let line_str = self.line_text(line);
        let width_digits = line.to_string().len();

        let span_chars = if end_byte > start_byte && end_byte <= self.src.len() {
            let line_end_col = line_str.chars().count().saturating_sub(col.saturating_sub(1));
            let raw_len = self
                .src
                .get(start_byte..end_byte)
                .map(|s| s.chars().count())
                .unwrap_or(1);
            raw_len.min(line_end_col.max(1)).max(1)
        } else {
            1
        };

        let mut out = String::new();
        match code {
            Some(c) => {
                let _ = writeln!(out, "error[{c}]: {message}");
            }
            None => {
                let _ = writeln!(out, "error: {message}");
            }
        }
        let _ = writeln!(
            out,
            "{:>width$}--> {}:{}:{}",
            "",
            self.path.display(),
            line,
            col,
            width = width_digits
        );
        let _ = writeln!(out, "{:>width$} |", "", width = width_digits);
        let _ = writeln!(out, "{line:>width$} | {line_str}", width = width_digits);
        let pad = " ".repeat(col.saturating_sub(1));
        let carets = "^".repeat(span_chars);
        let _ = writeln!(out, "{:>width$} | {pad}{carets}", "", width = width_digits);

        if let Some(h) = help {
            for (idx, hline) in h.lines().enumerate() {
                if idx == 0 {
                    let _ = writeln!(out, "{:>width$} = help: {hline}", "", width = width_digits);
                } else {
                    let _ = writeln!(out, "{:>width$}         {hline}", "", width = width_digits);
                }
            }
        }

        let docs_line = format_docs_directive(code, doc_topic);
        let _ = writeln!(out, "{:>width$} = docs: {docs_line}", "", width = width_digits);
        out
    }
}

pub fn format_docs_directive(code: Option<&str>, doc_topic: Option<&str>) -> String {
    match (code, doc_topic) {
        (Some(c), Some(t)) => format!("run 'zevc doc {c}' or 'zevc doc {t}'"),
        (Some(c), None) => format!("run 'zevc doc {c}'"),
        (None, Some(t)) => format!("run 'zevc doc {t}'"),
        (None, None) => "run 'zevc doc syntax'".to_string(),
    }
}

pub fn primary_doc_cmd(code: Option<&str>, doc_topic: Option<&str>) -> String {
    if let Some(c) = code {
        format!("zevc doc {c}")
    } else if let Some(t) = doc_topic {
        format!("zevc doc {t}")
    } else {
        "zevc doc syntax".to_string()
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

pub fn render_check_json_ok(
    files: usize,
    spawned_tasks: usize,
    primitives: usize,
    tick_period_us: u64,
) -> String {
    format!(
        "{{\"ok\": true, \"files\": {files}, \"spawned_tasks\": {spawned_tasks}, \"primitives\": {primitives}, \"tick_period_us\": {tick_period_us}, \"diagnostics\": []}}"
    )
}

pub fn render_check_json_err(diags: &[JsonDiag]) -> String {
    let mut out = String::from("{\"ok\": false, \"diagnostics\": [");
    for (idx, d) in diags.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        let code_json = match &d.code {
            Some(c) => format!("\"{}\"", json_escape(c)),
            None => "null".to_string(),
        };
        let help_json = match &d.help {
            Some(h) => format!("\"{}\"", json_escape(h)),
            None => "null".to_string(),
        };
        let _ = write!(
            out,
            "{{\"code\": {}, \"file\": \"{}\", \"line\": {}, \"col\": {}, \"message\": \"{}\", \"help\": {}, \"doc_cmd\": \"{}\"}}",
            code_json,
            json_escape(&d.file),
            d.line,
            d.col,
            json_escape(&d.message),
            help_json,
            json_escape(&d.doc_cmd),
        );
    }
    out.push_str("]}");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_rich_line_caret_help_and_docs_cleanly() {
        let sf = SourceFile::new(
            PathBuf::from("test.zev"),
            "fn main() {\n    noyield {\n        checkpoint();\n    }\n}\n".to_string(),
        );
        let start = sf.src.find("checkpoint()").unwrap() as u32;
        let end = start + "checkpoint()".len() as u32;
        let span = Span::new(start, end, 3, 9);
        let rendered = sf.format_rich_span_error(
            span,
            Some("E0003"),
            "cannot call yielding function inside `noyield`",
            Some("move `checkpoint()` outside `noyield { ... }`"),
            Some("yielding"),
        );
        assert!(rendered.contains("error[E0003]: cannot call yielding function inside `noyield`"));
        assert!(rendered.contains("test.zev:3:9"));
        assert!(rendered.contains("^^^^^^^^^^^^"));
        assert!(rendered.contains("= help: move `checkpoint()`"));
        assert!(rendered.contains("= docs: run 'zevc doc E0003' or 'zevc doc yielding'"));
    }

    #[test]
    fn renders_valid_json_diagnostics() {
        let diags = vec![JsonDiag {
            code: Some("P0001".to_string()),
            file: "bad.zev".to_string(),
            line: 2,
            col: 5,
            message: "unexpected `let` keyword".to_string(),
            help: Some("write `u16 x = 1;`".to_string()),
            doc_cmd: "zevc doc P0001".to_string(),
        }];
        let json = render_check_json_err(&diags);
        assert!(json.starts_with("{\"ok\": false, \"diagnostics\": [{\"code\": \"P0001\""));
        assert!(json.contains("\"doc_cmd\": \"zevc doc P0001\""));
    }
}
