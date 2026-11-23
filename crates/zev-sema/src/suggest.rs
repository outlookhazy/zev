//! Fuzzy Levenshtein edit-distance matching and foreign-idiom method/type hints
//! for agent-guided `E0020` (`UndefinedSymbol`) diagnostics.

use crate::types::ResolvedOsPrimitive;

/// Compute classic Levenshtein edit distance between `a` and `b`.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = usize::from(a_chars[i - 1] != b_chars[j - 1]);
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Find the closest candidate within `max_dist` (default `2`), preferring case-insensitive exact
/// matches first and shorter distances/lexicographic tie-breaking next.
pub fn closest_match<'a>(
    needle: &str,
    candidates: impl IntoIterator<Item = &'a str>,
    max_dist: usize,
) -> Option<&'a str> {
    if needle.is_empty() {
        return None;
    }
    let needle_lower = needle.to_ascii_lowercase();
    let mut best: Option<(&'a str, usize)> = None;

    for cand in candidates {
        if cand == needle {
            continue;
        }
        if cand.eq_ignore_ascii_case(&needle_lower) {
            return Some(cand);
        }
        // Avoid noisy suggestions on 1-2 character identifiers unless distance == 1.
        let allowed = if needle.len() <= 2 {
            1.min(max_dist)
        } else {
            max_dist
        };
        let dist = edit_distance(&needle_lower, &cand.to_ascii_lowercase());
        if dist <= allowed {
            match best {
                None => best = Some((cand, dist)),
                Some((prev_cand, prev_dist)) => {
                    if dist < prev_dist || (dist == prev_dist && cand < prev_cand) {
                        best = Some((cand, dist));
                    }
                }
            }
        }
    }

    best.map(|(c, _)| c)
}

/// Detect common C / Rust / Zig / Go scalar type names and map them to idiomatic Zev types.
pub fn suggest_foreign_type_name(name: &str) -> Option<&'static str> {
    match name {
        "uint8_t" | "uchar" | "byte" | "u_int8_t" => Some("u8"),
        "uint16_t" | "ushort" | "u_int16_t" => Some("u16"),
        "uint32_t" | "uint" | "ulong" | "u_int32_t" => Some("u32"),
        "uint64_t" | "ullong" | "u_int64_t" => Some("u64"),
        "int8_t" | "schar" => Some("i8"),
        "int16_t" | "short" => Some("i16"),
        "int32_t" | "int" | "long" => Some("i32"),
        "int64_t" | "llong" => Some("i64"),
        "size_t" | "uintptr_t" => Some("usize"),
        "ssize_t" | "ptrdiff_t" | "intptr_t" => Some("isize"),
        "float" | "float32" => Some("f32"),
        "double" | "float64" => Some("f64"),
        "boolean" | "Bool" | "_Bool" => Some("bool"),
        "String" | "string" | "str" => Some("[]u8"),
        "u16_be" | "be16" => Some("u16be"),
        "u16_le" | "le16" => Some("u16le"),
        "u32_be" | "be32" => Some("u32be"),
        "u32_le" | "le32" => Some("u32le"),
        _ => None,
    }
}

/// Produce targeted `"did you mean ...?"` + complete valid signature table when an unknown method
/// is called on a Zev OS primitive (`signal`, `chan`, `inbox`, `watch`, `ping_pong`, `lease`).
pub fn advise_primitive_method(prim: &ResolvedOsPrimitive, method: &str) -> (Option<String>, String) {
    match prim {
        ResolvedOsPrimitive::Signal => {
            let valid = ["notify", "wait", "try_wait", "is_set", "clear"];
            let direct = match method {
                "set" | "send" | "post" | "trigger" | "pulse" | "wake" | "emit" | "fire"
                | "signal" => Some("`.notify()` (or `.notify(boost: @critical)` from an ISR)".to_string()),
                "recv" | "acquire" | "pend" | "take" | "block" => {
                    Some("`.wait()` (or `.wait(timeout: 10ms) catch { ... }`)".to_string())
                }
                "get" | "read" | "poll" | "check" => {
                    Some("`.is_set()` or `.try_wait()`".to_string())
                }
                "reset" => Some("`.clear()`".to_string()),
                _ => closest_match(method, valid, 2).map(|m| format!("`.{}()`", m)),
            };
            let table = "valid methods on `signal`: `.notify([boost: @prio])`, `.wait([timeout: <dur>])`, `.is_set() -> bool`, `.try_wait() -> bool`, `.clear()`".to_string();
            (direct, table)
        }

        ResolvedOsPrimitive::WaitQueue => {
            let valid = ["prepare_wait", "cancel_wait", "commit_wait", "wake_one", "wake_all"];
            let direct = match method {
                "wait" | "block" | "park" => {
                    Some("`.prepare_wait()` inside `critical { ... }` followed by `.commit_wait([timeout: <dur>])`".to_string())
                }
                "wake" | "notify" | "signal" => {
                    Some("`.wake_one([boost: @prio])` or `.wake_all([boost: @prio])`".to_string())
                }
                _ => closest_match(method, valid, 2).map(|m| format!("`.{}()`", m)),
            };
            let table = "valid methods on `wait_queue`: `.prepare_wait()`, `.cancel_wait()`, `.commit_wait([timeout: <dur>])`, `.wake_one([boost: @prio]) -> bool`, `.wake_all([boost: @prio])`".to_string();
            (direct, table)
        }

        ResolvedOsPrimitive::Watch { elem } => {
            let valid = ["read", "publish", "wait_change"];
            let direct = match method {
                "get" | "load" | "peek" | "value" | "val" | "recv" | "borrow" => {
                    Some(format!("`.read() -> {}`", elem))
                }
                "set" | "write" | "store" | "send" | "post" | "put" | "update" => {
                    Some(format!("`.publish({} val)`", elem))
                }
                "wait" | "changed" => {
                    Some(format!("`.wait_change() -> {}` (or `.read()` for immediate non-blocking read)", elem))
                }
                _ => closest_match(method, valid, 2).map(|m| format!("`.{}(...)`", m)),
            };
            let table = format!(
                "valid methods on `{}`: `.read() -> {}`, `.publish({} val)`, `.wait_change([timeout: <dur>]) -> {}`",
                prim, elem, elem, elem
            );
            (direct, table)
        }

        ResolvedOsPrimitive::Inbox { elem, .. } | ResolvedOsPrimitive::InboxUnsized { elem } => {
            let valid = [
                "post", "try_post", "send", "try_send", "recv", "try_recv", "len", "cap",
                "is_empty", "is_full",
            ];
            let direct = match method {
                "push" | "put" | "enqueue" | "write" | "publish" | "emit" | "notify" => {
                    Some(format!(
                        "`.post({} val)` (non-blocking, ISR-safe) or `.send({} val)` (task-only backpressured)",
                        elem, elem
                    ))
                }
                "pop" | "get" | "read" | "wait" | "dequeue" | "next" | "pull" => Some(format!(
                    "`.recv() -> {}` (or `.recv(timeout: 5s) or Fallback`)",
                    elem
                )),
                _ => closest_match(method, valid, 2).map(|m| format!("`.{}(...)`", m)),
            };
            let table = format!(
                "valid methods on `{}`: `.post({} val [, boost: @prio])`, `.send({} val [, timeout: <dur>])`, `.try_send({} val) -> bool`, `.recv([timeout: <dur>]) -> {}`, `.try_recv() -> {} error`, `.len()`, `.cap()`, `.is_empty()`, `.is_full()`",
                prim, elem, elem, elem, elem, elem
            );
            (direct, table)
        }

        ResolvedOsPrimitive::PingPong { elem, .. } => {
            let valid = [
                "wait_chunk",
                "flip",
                "complete_half",
                "base_addr",
                "start",
                "activate",
                "stop",
                "deactivate",
                "release",
                "active_buf",
                "dma_buf",
                "half_len",
                "cap",
            ];
            let direct = match method {
                "recv" | "read" | "get" | "wait" | "next" | "next_chunk" | "acquire" | "pop"
                | "read_chunk" | "try_recv" => {
                    Some(format!("`.wait_chunk() -> []{}`", elem))
                }
                "swap" | "post" | "send" | "notify" | "complete" | "commit" | "publish" => {
                    Some("`.flip()` or `.complete_half(u8 half)` (zero-copy buffer swap in DMA ISR)".to_string())
                }
                _ => closest_match(method, valid, 2).map(|m| format!("`.{}()`", m)),
            };
            let table = format!(
                "valid methods on `{}`: `.wait_chunk([timeout: <dur>]) -> []{}` (task), `.flip()` / `.complete_half(u8 half)` (ISR swap), `.base_addr() -> usize`, `.start()`, `.stop()`",
                prim, elem
            );
            (direct, table)
        }

        ResolvedOsPrimitive::Pool { elem, .. } => {
            let valid = [
                "try_alloc",
                "alloc",
                "avail",
                "in_use",
                "cap",
                "is_exhausted",
            ];
            let direct = match method {
                "acquire" | "try_acquire" | "get" | "pop" | "take" | "wait_chunk" => {
                    Some(format!("`.alloc()` or `.try_alloc() -> Lease<{}> error`", elem))
                }
                "release" | "free" | "drop" | "put" => {
                    Some("`lease.release()` on the linear `Lease<T>` handle".to_string())
                }
                "free_space" | "len" | "remaining" | "count" => Some("`.avail()`".to_string()),
                "is_empty" | "is_full" => Some("`.is_exhausted()`".to_string()),
                _ => closest_match(method, valid, 2).map(|m| format!("`.{}(...)`", m)),
            };
            let table = format!(
                "valid methods on `{}`: `.try_alloc() -> Lease<{}> error`, `.alloc([timeout: <dur>])`, `.avail() -> usize`, `.in_use() -> usize`, `.cap() -> usize`, `.is_exhausted() -> bool`",
                prim, elem
            );
            (direct, table)
        }

        ResolvedOsPrimitive::DmaSlot { elem } => {
            let valid = ["park", "take", "release", "is_busy", "is_empty", "addr"];
            let direct = match method {
                "put" | "store" | "push" | "set" | "send" | "post" => {
                    Some("`.park(lease) -> void error`".to_string())
                }
                "get" | "pop" | "recv" | "extract" | "acquire" => {
                    Some(format!("`.take() -> Lease<{}> error`", elem))
                }
                "free" | "drop" | "clear" | "complete" => {
                    Some("`.release([boost: @prio]) -> bool`".to_string())
                }
                "is_full" | "busy" => Some("`.is_busy() -> bool`".to_string()),
                "ptr" | "data" => Some("`.addr() -> usize`".to_string()),
                _ => closest_match(method, valid, 2).map(|m| format!("`.{}(...)`", m)),
            };
            let table = format!(
                "valid methods on `{}`: `.park(lease) -> void error`, `.take() -> Lease<{}> error`, `.release([boost: @prio]) -> bool`, `.is_busy() -> bool`, `.is_empty() -> bool`, `.addr() -> usize`",
                prim, elem
            );
            (direct, table)
        }

        ResolvedOsPrimitive::ByteRing { .. } | ResolvedOsPrimitive::ByteRingUnsized => {
            let valid = [
                "push",
                "push_byte",
                "grant_write",
                "commit_write",
                "read_slice",
                "wait_read",
                "consume",
                "pop",
                "try_pop_byte",
                "clear",
                "len",
                "cap",
                "free_space",
                "is_empty",
                "is_full",
            ];
            let direct = match method {
                "write" | "send" | "post" | "enqueue" | "put" => {
                    Some("`.push(slice)` or `.push_byte(b)` (or `.grant_write(n)` / `.commit_write(n)`)".to_string())
                }
                "read" | "recv" | "take" | "wait" | "peek" | "slice" => {
                    Some("`.read_slice() -> []u8`, `.wait_read([timeout: <dur>]) -> []u8`, or `.pop(dst) -> usize`".to_string())
                }
                "advance" | "drop" | "ack" | "discard" => Some("`.consume(n)`".to_string()),
                "pop_byte" | "getc" | "try_recv" => Some("`.try_pop_byte() -> u8 error`".to_string()),
                "reset" => Some("`.clear()`".to_string()),
                "space" | "avail" | "remaining" => Some("`.free_space() -> usize`".to_string()),
                _ => closest_match(method, valid, 2).map(|m| format!("`.{}(...)`", m)),
            };
            let table = format!(
                "valid methods on `{}`: `.push(slice [, boost: @prio])`, `.push_byte(b [, boost: @prio])`, `.grant_write(len) -> []u8 error`, `.commit_write(n [, boost: @prio])`, `.read_slice() -> []u8`, `.wait_read([timeout: <dur>]) -> []u8`, `.consume(n)`, `.pop(dst)`, `.try_pop_byte() -> u8 error`, `.len()`, `.free_space()`, `.cap()`, `.is_empty()`, `.is_full()`, `.clear()`",
                prim
            );
            (direct, table)
        }
    }
}
