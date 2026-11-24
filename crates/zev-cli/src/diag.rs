use std::fmt::Write;
use std::path::{Path, PathBuf};
use zev_ast::Span;
use zev_sema::MonomorphizedInstance;

#[derive(Debug, Clone)]
pub struct SourceFile {
    pub path: PathBuf,
    pub src: String,
    line_starts: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskRunSummary {
    pub slot_id: usize,
    pub instance: String,
    pub blueprint: String,
    pub priority: String,
    pub max_run_kind: String,
    pub max_run_blocks: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndirectCallSiteSummary {
    pub caller: String,
    pub file: String,
    pub line: usize,
    pub col: usize,
    pub signature: String,
    pub is_noyield: bool,
    pub candidates: Vec<String>,
    pub frame_bytes_32: usize,
    pub frame_bytes_64: usize,
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
            let line_end_col = line_str
                .chars()
                .count()
                .saturating_sub(col.saturating_sub(1));
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
        let _ = writeln!(
            out,
            "{:>width$} = docs: {docs_line}",
            "",
            width = width_digits
        );
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

pub fn format_check_human_ok(
    files: usize,
    spawned_tasks: usize,
    primitives: usize,
    tick_period_us: u64,
    dual_specialized: &[String],
    monomorphized: &[MonomorphizedInstance],
) -> String {
    let mut sorted_dual: Vec<&str> = dual_specialized.iter().map(String::as_str).collect();
    sorted_dual.sort_unstable();
    sorted_dual.dedup();

    let mut sorted_mono: Vec<&str> = monomorphized.iter().map(|m| m.signature.as_str()).collect();
    sorted_mono.sort_unstable();
    sorted_mono.dedup();

    let mut out = format!(
        "OK ({files} file(s) verified, {spawned_tasks} task instance(s), {primitives} primitive(s), tick={tick_period_us}us"
    );
    if !sorted_dual.is_empty() {
        let _ = write!(out, ", dual_specialized=[{}]", sorted_dual.join(", "));
    }
    if !sorted_mono.is_empty() {
        let _ = write!(out, ", monomorphized=[{}]", sorted_mono.join(", "));
    }
    out.push(')');
    out
}

#[cfg(test)]
pub fn render_check_json_ok(
    files: usize,
    spawned_tasks: usize,
    primitives: usize,
    tick_period_us: u64,
    dual_specialized: &[String],
    monomorphized: &[MonomorphizedInstance],
) -> String {
    render_check_json_ok_with_tasks(
        files,
        spawned_tasks,
        primitives,
        tick_period_us,
        dual_specialized,
        monomorphized,
        &[],
    )
}

#[cfg(test)]
pub fn render_check_json_ok_with_tasks(
    files: usize,
    spawned_tasks: usize,
    primitives: usize,
    tick_period_us: u64,
    dual_specialized: &[String],
    monomorphized: &[MonomorphizedInstance],
    task_runs: &[TaskRunSummary],
) -> String {
    render_check_json_ok_full(
        files,
        spawned_tasks,
        primitives,
        tick_period_us,
        dual_specialized,
        monomorphized,
        task_runs,
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn render_check_json_ok_full(
    files: usize,
    spawned_tasks: usize,
    primitives: usize,
    tick_period_us: u64,
    dual_specialized: &[String],
    monomorphized: &[MonomorphizedInstance],
    task_runs: &[TaskRunSummary],
    indirect_call_sites: &[IndirectCallSiteSummary],
) -> String {
    let mut sorted: Vec<&str> = dual_specialized.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    sorted.dedup();
    let mut dual_json = String::from("[");
    for (idx, name) in sorted.iter().enumerate() {
        if idx > 0 {
            dual_json.push_str(", ");
        }
        let _ = write!(dual_json, "\"{}\"", json_escape(name));
    }
    dual_json.push(']');

    let mut sorted_mono: Vec<(&str, &str, &str)> = monomorphized
        .iter()
        .map(|inst| {
            (
                inst.kind.as_str(),
                inst.signature.as_str(),
                inst.mangled_name.as_str(),
            )
        })
        .collect();
    sorted_mono.sort_unstable_by(|a, b| {
        a.1.cmp(b.1)
            .then_with(|| a.0.cmp(b.0))
            .then_with(|| a.2.cmp(b.2))
    });
    sorted_mono.dedup();

    let mut mono_json = String::from("[");
    for (idx, (kind, sig, mangled)) in sorted_mono.iter().enumerate() {
        if idx > 0 {
            mono_json.push_str(", ");
        }
        let _ = write!(
            mono_json,
            "{{\"kind\": \"{}\", \"signature\": \"{}\", \"mangled_name\": \"{}\"}}",
            json_escape(kind),
            json_escape(sig),
            json_escape(mangled),
        );
    }
    mono_json.push(']');

    let mut runs_json = String::from("[");
    for (idx, t) in task_runs.iter().enumerate() {
        if idx > 0 {
            runs_json.push_str(", ");
        }
        let _ = write!(
            runs_json,
            "{{\"slot_id\": {}, \"instance\": \"{}\", \"blueprint\": \"{}\", \"priority\": \"{}\", \"max_run_kind\": \"{}\", \"max_run_blocks\": {}}}",
            t.slot_id,
            json_escape(&t.instance),
            json_escape(&t.blueprint),
            json_escape(&t.priority),
            json_escape(&t.max_run_kind),
            t.max_run_blocks,
        );
    }
    runs_json.push(']');

    let mut sites_json = String::from("[");
    for (idx, s) in indirect_call_sites.iter().enumerate() {
        if idx > 0 {
            sites_json.push_str(", ");
        }
        let mut cands_json = String::from("[");
        for (ci, c) in s.candidates.iter().enumerate() {
            if ci > 0 {
                cands_json.push_str(", ");
            }
            let _ = write!(cands_json, "\"{}\"", json_escape(c));
        }
        cands_json.push(']');
        let _ = write!(
            sites_json,
            "{{\"caller\": \"{}\", \"file\": \"{}\", \"line\": {}, \"col\": {}, \"signature\": \"{}\", \"is_noyield\": {}, \"candidates\": {}, \"frame_bytes_32\": {}, \"frame_bytes_64\": {}}}",
            json_escape(&s.caller),
            json_escape(&s.file),
            s.line,
            s.col,
            json_escape(&s.signature),
            s.is_noyield,
            cands_json,
            s.frame_bytes_32,
            s.frame_bytes_64,
        );
    }
    sites_json.push(']');

    format!(
        "{{\"ok\": true, \"files\": {files}, \"spawned_tasks\": {spawned_tasks}, \"primitives\": {primitives}, \"tick_period_us\": {tick_period_us}, \"dual_specialized_functions\": {dual_json}, \"monomorphized_instances\": {mono_json}, \"task_runs\": {runs_json}, \"indirect_call_sites\": {sites_json}, \"diagnostics\": []}}"
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

    #[test]
    fn renders_dual_specialized_and_monomorphized_instances_in_json_and_human_ok_outputs() {
        let empty_json = render_check_json_ok(1, 2, 3, 1000, &[], &[]);
        assert_eq!(
            empty_json,
            "{\"ok\": true, \"files\": 1, \"spawned_tasks\": 2, \"primitives\": 3, \"tick_period_us\": 1000, \"dual_specialized_functions\": [], \"monomorphized_instances\": [], \"task_runs\": [], \"indirect_call_sites\": [], \"diagnostics\": []}"
        );
        assert_eq!(
            format_check_human_ok(1, 2, 3, 1000, &[], &[]),
            "OK (1 file(s) verified, 2 task instance(s), 3 primitive(s), tick=1000us)"
        );

        let fns = vec!["sha256_block".to_string(), "crc_chunk".to_string()];
        // Unsorted + duplicate entry to verify internal sorting and deduplication
        let mono = vec![
            MonomorphizedInstance {
                kind: zev_sema::MonomorphizedKind::Fn,
                signature: "clamp<f32>".to_string(),
                mangled_name: "clamp__f32".to_string(),
            },
            MonomorphizedInstance {
                kind: zev_sema::MonomorphizedKind::Struct,
                signature: "FixedVec<u32, 8>".to_string(),
                mangled_name: "FixedVec__u32__8".to_string(),
            },
            MonomorphizedInstance {
                kind: zev_sema::MonomorphizedKind::Fn,
                signature: "clamp<f32>".to_string(),
                mangled_name: "clamp__f32".to_string(),
            },
        ];
        let dual_json = render_check_json_ok(2, 4, 5, 1000, &fns, &mono);
        assert!(
            dual_json.contains("\"dual_specialized_functions\": [\"crc_chunk\", \"sha256_block\"]"),
            "json={dual_json}"
        );
        assert!(
            dual_json.contains("\"monomorphized_instances\": [{\"kind\": \"struct\", \"signature\": \"FixedVec<u32, 8>\", \"mangled_name\": \"FixedVec__u32__8\"}, {\"kind\": \"fn\", \"signature\": \"clamp<f32>\", \"mangled_name\": \"clamp__f32\"}]"),
            "json={dual_json}"
        );
        assert_eq!(
            format_check_human_ok(2, 4, 5, 1000, &fns, &mono),
            "OK (2 file(s) verified, 4 task instance(s), 5 primitive(s), tick=1000us, dual_specialized=[crc_chunk, sha256_block], monomorphized=[FixedVec<u32, 8>, clamp<f32>])"
        );
    }
}
