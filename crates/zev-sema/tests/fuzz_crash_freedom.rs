use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use zev_parser::{parse_str, parse_str_named};
use zev_sema::analyze_and_monomorphize_programs;
use zev_sema::checker::AnalyzeOptions;

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn gen_range(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() as usize) % bound
    }

    fn pick<'a, T>(&mut self, slice: &'a [T]) -> &'a T {
        &slice[self.gen_range(slice.len())]
    }
}

fn load_example_sources() -> Vec<(String, String)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let Ok(entries) = fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut paths: Vec<_> = entries
        .filter_map(|e| e.ok().map(|ent| ent.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("zev"))
        .collect();
    paths.sort();

    let mut out = Vec::new();
    for path in paths {
        if let Ok(src) = fs::read_to_string(&path) {
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("example.zev")
                .to_string();
            out.push((name, src));
        }
    }
    out
}

fn exercise_frontend(label: &str, src: &str) {
    let opts = AnalyzeOptions::default();
    let res = catch_unwind(AssertUnwindSafe(|| {
        let _ = parse_str(src);
        if let Ok(prog) = parse_str_named(src, "<fuzz>") {
            let _ = analyze_and_monomorphize_programs(&[prog], &opts);
        }
    }));
    if let Err(payload) = res {
        let msg = if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".to_string()
        };
        let preview: String = src.chars().take(240).collect();
        panic!("panic in frontend ({label}): {msg}\ninput preview: {preview:?}");
    }
}

const TOKENS: &[&str] = &[
    "fn", "task", "shared", "inbox", "signal", "watch", "byte_ring", "struct", "enum", "switch",
    "if", "else", "while", "loop", "for", "in", "spawn", "irq", "attach", "lease", "critical",
    "noyield", "main", "import", "pub", "const", "static", "extern", "mmio", "register", "as",
    "return", "break", "continue", "defer", "true", "false", "u8", "u16", "u32", "u64", "i32",
    "f32", "f64", "bool", "str", "void", "->", "=>", "::", ".", ",", ";", ":", "=", "==", "!=",
    "<", "<=", ">", ">=", "+", "-", "*", "/", "%", "&", "|", "^", "!", "&&", "||", "<<", ">>",
    "(", ")", "{", "}", "[", "]", "x", "y", "T", "0", "1", "255", "\"abc\"", "f\"{x}\"",
    "sys", "delay", "fmt_buf", "fmt_ring", "print", "println",
];

#[test]
fn fuzz_parser_and_sema_crash_freedom() {
    let mut rng = XorShift64::new(0x5EED_F00D_CAFE_BEEF);
    let examples = load_example_sources();
    assert!(!examples.is_empty(), "expected examples/*.zev to exist");

    let mut total_cases = 0usize;

    // 1. Random UTF-8 / ASCII byte soups and token soups (200 cases)
    let unicode_fragments = ["a", "0", " ", "\n", "\t", "{", "}", "(", ")", "<", ">", "\"", "\\",
        "{x}", "é", "λ", "中", "🦀", "\u{0000}", "\u{007f}", "//", "/*", "*/", "->"];
    for i in 0..200 {
        let mut s = String::new();
        if i % 2 == 0 {
            let n = 1 + rng.gen_range(90);
            for _ in 0..n {
                s.push_str(rng.pick(TOKENS));
                if rng.gen_range(3) == 0 {
                    s.push('\n');
                } else {
                    s.push(' ');
                }
            }
        } else {
            let n = 1 + rng.gen_range(160);
            for _ in 0..n {
                if rng.gen_range(5) == 0 {
                    s.push_str(rng.pick(&unicode_fragments));
                } else {
                    let b = (rng.next_u64() & 0x7f) as u8;
                    if b >= 0x20 || b == b'\n' || b == b'\t' {
                        s.push(b as char);
                    }
                }
            }
        }
        exercise_frontend(&format!("soup_{i}"), &s);
        total_cases += 1;
    }

    // 2. Mutations, truncations, slices, and splices of real examples/*.zev (240 cases)
    for i in 0..240 {
        let (name_a, src_a) = rng.pick(&examples);
        let (_, src_b) = rng.pick(&examples);
        let bytes_a = src_a.as_bytes();
        let bytes_b = src_b.as_bytes();
        let mutated = match i % 4 {
            0 => {
                // Truncation at random UTF-8 boundary
                let cut = rng.gen_range(src_a.len().max(1));
                let mut idx = cut.min(src_a.len());
                while idx > 0 && !src_a.is_char_boundary(idx) {
                    idx -= 1;
                }
                src_a[..idx].to_string()
            }
            1 => {
                // Random byte flip edits converted cleanly via lossy UTF-8
                let mut buf = bytes_a.to_vec();
                let edits = 1 + rng.gen_range(12);
                for _ in 0..edits {
                    if !buf.is_empty() {
                        let pos = rng.gen_range(buf.len());
                        buf[pos] = (rng.next_u64() & 0xff) as u8;
                    }
                }
                String::from_utf8_lossy(&buf).into_owned()
            }
            2 => {
                // Line-level slice/shuffle/deletion
                let lines: Vec<&str> = src_a.lines().collect();
                let mut out = String::new();
                for line in &lines {
                    let action = rng.gen_range(10);
                    if action == 0 {
                        continue;
                    }
                    out.push_str(line);
                    out.push('\n');
                    if action == 1 {
                        out.push_str(line);
                        out.push('\n');
                    }
                }
                out
            }
            _ => {
                // Random splice between two examples
                let cut_a = rng.gen_range(bytes_a.len().max(1)).min(bytes_a.len());
                let cut_b = rng.gen_range(bytes_b.len().max(1)).min(bytes_b.len());
                let mut combined = bytes_a[..cut_a].to_vec();
                combined.extend_from_slice(&bytes_b[cut_b..]);
                String::from_utf8_lossy(&combined).into_owned()
            }
        };
        exercise_frontend(&format!("example_mut_{name_a}_{i}"), &mutated);
        total_cases += 1;
    }

    // 3. Malformed and extreme f"..." format strings (120 cases)
    let fmt_fragments = [
        "{", "}", "{{", "}}", "{x", "x}", "{x:}", "{x:d}", "{x:099999999999d}",
        "{x:.99999999s}", "{x:99999999999999999999x}", "{x:zzzz}", "{a + {b}}",
        "{f\"{x}\"}", "{0}", "{:}", "\\", "\\n", "\\x", "\\xZZ", "\"", "\0", "{x:<1000}",
        "{x:>00010000000000}",
    ];
    for i in 0..120 {
        let mut fmt_body = String::new();
        let pieces = 1 + rng.gen_range(8);
        for _ in 0..pieces {
            fmt_body.push_str(rng.pick(&fmt_fragments));
        }
        let src = format!(
            "fn main() {{\n    u32 x = 42;\n    [16]u8 b = [0u8; 16];\n    println(f\"{fmt_body}\");\n    fmt_buf(b, f\"{fmt_body}\");\n    loop {{ delay(1ms); }}\n}}\n"
        );
        exercise_frontend(&format!("fstring_{i}"), &src);
        total_cases += 1;
    }

    // 4. Extreme numeric literals and edge constant expressions (80 cases)
    let num_tokens = [
        "0", "1", "0x0", "0xFFFFFFFFFFFFFFFF", "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "999999999999999999999999999999999999", "0b", "0x", "0o", "0b1010101010102",
        "1.7976931348623157e308", "1e99999", "1e-99999", "0.0.0", "1e+", "1_000_000_u32",
        "0xffffffffffffffffu64", "-9223372036854775808", "1 / 0", "1 % 0", "1 << 999",
    ];
    for i in 0..80 {
        let a = rng.pick(&num_tokens);
        let b = rng.pick(&num_tokens);
        let op = rng.pick(&["+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^"]);
        let src = format!(
            "const u32 C = {a} {op} {b};\nfn main() {{\n    u32 x = {a};\n    u32 y = {b};\n    u32 _z = x {op} y;\n    loop {{ delay(1ms); }}\n}}\n"
        );
        exercise_frontend(&format!("num_{i}"), &src);
        total_cases += 1;
    }

    // 5. Deep nesting of parens, braces, brackets, generics up to ~150 depth (80 cases)
    for i in 0..80 {
        let depth = 1 + (i * 2).min(150);
        let src = match i % 4 {
            0 => {
                // Deeply nested parentheses expressions
                let open = "(".repeat(depth);
                let close = ")".repeat(depth);
                format!("fn main() {{\n    u32 _x = {open}1{close};\n    loop {{ delay(1ms); }}\n}}\n")
            }
            1 => {
                // Deeply nested blocks
                let open = "{ ".repeat(depth);
                let close = "} ".repeat(depth);
                format!("fn main() {{\n    {open}u32 _x = 1;{close}\n    loop {{ delay(1ms); }}\n}}\n")
            }
            2 => {
                // Deeply nested generic type expressions + arrays
                let open = "Box<".repeat(depth);
                let close = ">".repeat(depth);
                format!("struct Box<T> {{ T v; }}\nfn f({open}u32{close} x) {{ }}\nfn main() {{ loop {{ delay(1ms); }} }}\n")
            }
            _ => {
                // Deeply nested array brackets / mismatched delimiters
                let open = "[".repeat(depth);
                let close = "]".repeat(depth / 2);
                format!("fn main() {{\n    u32 _a = {open}1{close};\n    loop {{ delay(1ms); }}\n}}\n")
            }
        };
        exercise_frontend(&format!("nest_{i}"), &src);
        total_cases += 1;
    }

    assert!(total_cases >= 700, "ran {total_cases} fuzz cases");
}
