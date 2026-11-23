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

fn compile_host_llvm_binary(
    temp: &TempWorkDir,
    ll_source: &str,
    sanitize: bool,
) -> Result<PathBuf, String> {
    let ll_path = temp.dir.join("program.ll");
    let bin_path = temp.dir.join("program");

    std::fs::write(&ll_path, ll_source)
        .map_err(|e| format!("failed to write generated LLVM IR to `{}`: {e}", ll_path.display()))?;

    let clang = find_clang();
    let mut cc = Command::new(&clang);
    cc.arg("-Wno-override-module").arg("-no-pie");
    if sanitize {
        cc.arg("-O1")
            .arg("-g")
            .arg("-fsanitize=address,undefined");
    } else {
        cc.arg("-O2");
    }
    cc.arg(&ll_path).arg("-o").arg(&bin_path);

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

    Ok(bin_path)
}

pub fn compile_and_run_host_llvm(
    ll_source: &str,
    sanitize: bool,
    run_args: &[String],
) -> Result<i32, String> {
    let temp = TempWorkDir::create("zevc-run")?;
    let bin_path = compile_host_llvm_binary(&temp, ll_source, sanitize)?;

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
    fn compiles_and_runs_host_llvm_program() {
        let ll_src = r#"
target triple = "x86_64-unknown-linux-gnu"
define i32 @main() nounwind {
entry:
  %sum = add i32 7, 8
  %ok = icmp eq i32 %sum, 15
  %ret = select i1 %ok, i32 0, i32 1
  ret i32 %ret
}
"#;
        let code = compile_and_run_host_llvm(ll_src, true, &[]).unwrap();
        assert_eq!(code, 0);
    }
}
