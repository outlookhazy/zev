use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use zev_codegen_llvm::{CodegenOptions, emit_llvm_ir_with_options};
use zev_mir::lower_programs;
use zev_parser::parse_str_named;
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

fn exercise_full_llvm_pipeline(label: &str, src: &str) -> bool {
    let opts = AnalyzeOptions::default();
    let res = catch_unwind(AssertUnwindSafe(|| {
        let Ok(prog) = parse_str_named(src, "<fuzz_mir>") else {
            return false;
        };
        let mono = analyze_and_monomorphize_programs(&[prog], &opts);
        let Ok((monos, _)) = mono.into_result() else {
            return false;
        };
        if let Ok(mir) = lower_programs(&monos) {
            let host_opts = CodegenOptions::host_linux();
            if let Ok(ll_src) = emit_llvm_ir_with_options(&mir, &host_opts) {
                assert!(!ll_src.is_empty(), "emit_llvm_ir_with_options produced empty output");
                return true;
            }
        }
        false
    }));
    match res {
        Ok(lowered) => lowered,
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<non-string panic payload>".to_string()
            };
            let preview: String = src.chars().take(280).collect();
            panic!("panic in MIR/LLVM pipeline ({label}): {msg}\ninput preview: {preview:?}");
        }
    }
}

fn synthesize_program(rng: &mut XorShift64, idx: usize) -> String {
    let int_tys = ["u8", "u16", "u32", "i32", "usize"];
    let ty = *rng.pick(&int_tys);
    let arr_len = match idx % 4 {
        0 => 1,
        1 => 2,
        2 => 4,
        _ => 8,
    };
    let elem = *rng.pick(&["0u8", "1u8", "7u8", "255u8"]);
    let arr_init = format!("[{}]", vec![elem; arr_len].join(", "));
    let op = *rng.pick(&["+", "-", "*", "&", "|", "^"]);
    let cmp_op = *rng.pick(&["==", "!=", "<", "<=", ">", ">="]);

    let use_shared = idx % 2 == 0;
    let use_generics = idx % 3 == 0;
    let use_switch = idx % 4 == 0;
    let use_while = idx % 5 == 0;
    let use_fmt = idx % 3 == 1;
    let use_yield_task = idx % 2 == 1;

    let mut s = String::new();
    s.push_str(
        "struct Pair { u32 a; u32 b; }\n\
         enum Mode { Idle, Active, Fault }\n",
    );

    if use_generics {
        s.push_str(
            "T id_val<T>(T x) {\n\
                 return x;\n\
             }\n",
        );
    }

    if use_shared {
        s.push_str(
            "watch<u32> shared_w = 0;\n\
             inbox<u32, 4> q;\n",
        );
    }

    s.push_str(&format!(
        "u32 helper_calc(Pair p, {ty} n) {{\n\
             u32 acc = p.a {op} p.b;\n"
    ));

    if use_generics {
        s.push_str("    acc = id_val<u32>(acc);\n");
    }

    if use_switch {
        s.push_str(
            "    Mode m = Active;\n\
                 switch (m) {\n\
                     Idle => { acc = acc + 1u32; }\n\
                     Active => { acc = acc ^ 3u32; }\n\
                     Fault => { acc = acc + 7u32; }\n\
                 }\n",
        );
    }

    if use_while {
        s.push_str(&format!(
            "    {ty} i = 0 as {ty};\n\
                 noyield {{\n\
                     while (i {cmp_op} ((n as {ty}) & (3 as {ty}))) {{\n\
                         acc = acc + (i as u32);\n\
                         i = i + (1 as {ty});\n\
                         if (i >= (4 as {ty})) {{ break; }}\n\
                     }}\n\
                 }}\n"
        ));
    }

    if use_fmt {
        let zeros16 = vec!["0u8"; 16].join(", ");
        s.push_str(&format!(
            "    [16]u8 buf = [{zeros16}];\n\
                 []u8 _fmt = fmt_buf(buf, f\"acc={{acc}}\");\n"
        ));
    } else {
        s.push_str(&format!(
            "    [{arr_len}]u8 arr = {arr_init};\n\
                 acc = acc + (arr[0] as u32);\n"
        ));
    }

    s.push_str("    return acc;\n}\n");

    if use_yield_task {
        if use_shared {
            s.push_str(&format!(
                "task worker(*inbox<u32, 4> qq) {{\n\
                     loop {{\n\
                         Pair p = Pair {{ a: 3u32, b: 5u32 }};\n\
                         u32 v = helper_calc(p, 2 as {ty});\n\
                         qq.post(v);\n\
                         shared_w.publish(v);\n\
                         delay(10ms);\n\
                     }}\n\
                 }}\n\
                 task consumer(*inbox<u32, 4> qq) {{\n\
                     loop {{\n\
                         u32 v = qq.recv(timeout: 50ms) or 0u32;\n\
                         assert(v < 1000000u32);\n\
                     }}\n\
                 }}\n"
            ));
        } else {
            s.push_str(&format!(
                "task stepper(u32 seed) {{\n\
                     u32 s = seed;\n\
                     loop {{\n\
                         Pair p = Pair {{ a: s, b: 1u32 }};\n\
                         s = helper_calc(p, 2 as {ty});\n\
                         delay(5ms);\n\
                     }}\n\
                 }}\n"
            ));
        }
    }

    s.push_str("fn main() {\n");
    if use_yield_task {
        if use_shared {
            s.push_str("    spawn worker(&q) @normal;\n    spawn consumer(&q) @normal;\n    loop { delay(100ms); }\n");
        } else {
            s.push_str("    spawn stepper(7u32) @normal;\n    loop { delay(100ms); }\n");
        }
    } else {
        s.push_str(&format!(
            "    u32 _res = helper_calc(Pair {{ a: 2u32, b: 3u32 }}, 1 as {ty});\n    sys.exit(0);\n"
        ));
    }
    s.push_str("}\n");

    // Introduce small structured mutations on a fraction of generated programs
    if idx % 9 == 0 {
        s.replacen("sys.exit(0);", "loop { }", 1)
    } else if idx % 13 == 0 {
        s.replacen("return acc;", "", 1)
    } else {
        s
    }
}

#[test]
fn fuzz_mir_and_llvm_codegen_crash_freedom() {
    let mut rng = XorShift64::new(0xC0DE_99C9_1234_5678);
    let examples = load_example_sources();
    assert!(!examples.is_empty(), "expected examples/*.zev to exist");

    let mut total_cases = 0usize;
    let mut reached_llvm = 0usize;

    // 1. Seed directly with clean examples/*.zev and structural line-mutated variants (220 cases)
    for (i, (name, src)) in examples.iter().enumerate() {
        if exercise_full_llvm_pipeline(&format!("clean_example_{name}_{i}"), src) {
            reached_llvm += 1;
        }
        total_cases += 1;
    }
    for i in 0..220 {
        let (name, src) = rng.pick(&examples);
        let lines: Vec<&str> = src.lines().collect();
        let mutated = match i % 3 {
            0 => {
                // Swap numeric tokens to stress const folding, bit widths, capacities
                let nums = ["0", "1", "2", "4", "8", "16", "255"];
                let replacement = *rng.pick(&nums);
                src.replacen("1", replacement, 2)
            }
            1 => {
                // Drop 1-3 random lines
                let drop_line = rng.gen_range(lines.len().max(1));
                let mut out = String::new();
                for (li, line) in lines.iter().enumerate() {
                    if li != drop_line {
                        out.push_str(line);
                        out.push('\n');
                    }
                }
                out
            }
            _ => {
                // Append an extra unreachable helper fn
                format!("{src}\nfn fuzz_extra_{i}(x: u32) -> u32 {{ return x ^ 1; }}\n")
            }
        };
        if exercise_full_llvm_pipeline(&format!("mut_example_{name}_{i}"), &mutated) {
            reached_llvm += 1;
        }
        total_cases += 1;
    }

    // 2. Synthesized structured programs exercising structs/enums/generics/tasks/fmt_buf (340 cases)
    for i in 0..340 {
        let synth = synthesize_program(&mut rng, i);
        if exercise_full_llvm_pipeline(&format!("synth_{i}"), &synth) {
            reached_llvm += 1;
        }
        total_cases += 1;
    }

    assert!(total_cases >= 550, "ran {total_cases} MIR/LLVM fuzz cases");
    assert!(
        reached_llvm >= 150,
        "expected many programs to lower all the way to LLVM IR, got {reached_llvm}"
    );
}
