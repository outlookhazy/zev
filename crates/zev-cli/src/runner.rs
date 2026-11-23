use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static RUN_COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct TempWorkDir {
    pub dir: PathBuf,
}

impl TempWorkDir {
    pub fn create(prefix: &str) -> Result<Self, String> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let pid = std::process::id();
        let seq = RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("{prefix}-{pid}-{nanos}-{seq}"));
        std::fs::create_dir_all(&dir).map_err(|e| {
            format!(
                "failed to create temporary build directory `{}`: {e}",
                dir.display()
            )
        })?;
        Ok(Self { dir })
    }
}

impl Drop for TempWorkDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

pub(crate) fn find_clang() -> String {
    zev_codegen_llvm::find_tool("/bin/clang", "clang")
}

pub fn compile_and_run_c99(
    c_source: &str,
    sanitize: bool,
    max_ticks: Option<u64>,
    run_args: &[String],
) -> Result<i32, String> {
    let temp = TempWorkDir::create("zevc-run")?;
    let c_path = temp.dir.join("program.c");
    let bin_path = temp.dir.join("program");

    std::fs::write(&c_path, c_source)
        .map_err(|e| format!("failed to write generated C99 to `{}`: {e}", c_path.display()))?;

    let clang = find_clang();
    let mut cc = Command::new(&clang);
    cc.arg("-std=c99").arg("-O2");
    if sanitize {
        cc.arg("-g").arg("-fsanitize=address,undefined");
    }
    if let Some(ticks) = max_ticks {
        cc.arg(format!("-DSYN_SIM_MAX_TICKS={ticks}"));
    }
    cc.arg(&c_path).arg("-o").arg(&bin_path);

    let cc_out = cc
        .output()
        .map_err(|e| format!("failed to invoke `{clang}`: {e}"))?;

    if !cc_out.status.success() {
        let stderr = String::from_utf8_lossy(&cc_out.stderr);
        return Err(format!(
            "`{clang}` compilation failed (status {}):\n{stderr}",
            cc_out.status
        ));
    }

    let status = Command::new(&bin_path)
        .args(run_args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| format!("failed to execute compiled simulation binary: {e}"))?;

    Ok(status.code().unwrap_or(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_and_runs_c99_program() {
        let c_src = r#"
#include <stdio.h>
int main(void) {
    int sum = 0;
    for (int i = 1; i <= 5; i++) {
        sum += i;
    }
    return (sum == 15) ? 0 : 1;
}
"#;
        let code = compile_and_run_c99(c_src, true, Some(100), &[]).unwrap();
        assert_eq!(code, 0);
    }
}
