use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn collect_zev_files(dir: &Path, dirs_out: &mut Vec<PathBuf>, files_out: &mut Vec<PathBuf>) {
    if !dir.is_dir() {
        return;
    }
    dirs_out.push(dir.to_path_buf());
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|ent| ent.path())).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            collect_zev_files(&path, dirs_out, files_out);
        } else if path.extension().and_then(|s| s.to_str()) == Some("zev") {
            files_out.push(path);
        }
    }
}

fn module_keys_for_rel(rel_parts: &[String]) -> Vec<String> {
    if rel_parts.is_empty() {
        return Vec::new();
    }
    if rel_parts[0] == "hal" && rel_parts.len() >= 3 {
        let target = &rel_parts[1];
        let mut mod_parts = rel_parts[2..].to_vec();
        let last = mod_parts.len() - 1;
        let stem = mod_parts[last].trim_end_matches(".zev").to_string();
        if stem == "mod" && mod_parts.len() >= 2 {
            mod_parts.pop();
            let prefix = mod_parts.join(".");
            return vec![
                format!("hal/{target}.{prefix}"),
                format!("hal/{target}.{prefix}.mod"),
            ];
        }
        mod_parts[last] = stem;
        return vec![format!("hal/{target}.{}", mod_parts.join("."))];
    }

    let mut parts = rel_parts.to_vec();
    let last = parts.len() - 1;
    let stem = parts[last].trim_end_matches(".zev").to_string();
    if stem == "mod" && parts.len() >= 2 {
        parts.pop();
        let pkg = parts.join(".");
        return vec![pkg.clone(), format!("{pkg}.mod")];
    }
    parts[last] = stem;
    vec![parts.join(".")]
}

fn main() {
    let Ok(manifest_raw) = env::var("CARGO_MANIFEST_DIR") else {
        eprintln!("CARGO_MANIFEST_DIR must be set");
        std::process::exit(1);
    };
    let manifest_dir = PathBuf::from(manifest_raw);
    let repo_root = manifest_dir.join("../..");
    let Ok(out_raw) = env::var("OUT_DIR") else {
        eprintln!("OUT_DIR must be set");
        std::process::exit(1);
    };
    let out_dir = PathBuf::from(out_raw);

    let mut walked_dirs = Vec::new();
    let mut zev_files = Vec::new();
    for root_name in ["arch", "chip", "bsp", "std", "hal"] {
        let root_dir = repo_root.join(root_name);
        println!("cargo:rerun-if-changed={}", root_dir.display());
        collect_zev_files(&root_dir, &mut walked_dirs, &mut zev_files);
    }

    walked_dirs.sort();
    walked_dirs.dedup();
    for dir in &walked_dirs {
        println!("cargo:rerun-if-changed={}", dir.display());
    }

    zev_files.sort();
    zev_files.dedup();

    // Map `(pattern_string, normalized_rel_path)` keyed deterministically by relative path.
    let mut arms: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for abs_path in &zev_files {
        println!("cargo:rerun-if-changed={}", abs_path.display());
        let Ok(rel) = abs_path.strip_prefix(&repo_root) else {
            continue;
        };
        let parts: Vec<String> = rel
            .components()
            .filter_map(|c| c.as_os_str().to_str().map(str::to_owned))
            .collect();
        let norm_rel = parts.join("/");
        let keys = module_keys_for_rel(&parts);
        if !keys.is_empty() {
            arms.insert(norm_rel, keys);
        }
    }

    let mut generated = String::new();
    generated.push_str(
        "/// Embedded fallback source lookup for built-in library modules when no disk override exists.\n\
         pub fn embedded_module_source(dot_path: &str) -> Option<&'static str> {\n\
         \x20   match dot_path {\n",
    );
    for (norm_rel, keys) in &arms {
        let pattern = keys
            .iter()
            .map(|k| format!("{k:?}"))
            .collect::<Vec<_>>()
            .join(" | ");
        generated.push_str(&format!(
            "        {pattern} => Some(include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/../../{norm_rel}\"))),\n"
        ));
    }
    generated.push_str(
        "        _ => None,\n\
         \x20   }\n\
         }\n",
    );

    let dest_path = out_dir.join("embedded_modules.rs");
    if let Err(err) = fs::write(&dest_path, generated) {
        eprintln!(
            "failed to write generated embedded_modules.rs to {}: {err}",
            dest_path.display()
        );
        std::process::exit(1);
    }
}
