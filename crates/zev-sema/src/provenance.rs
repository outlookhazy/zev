//! Lexical scope-depth, local-reference, DMA lease generation, and C-callback provenance tracking.
//!
//! Enforces memory and DMA buffer safety with zero `'a` lifetime annotations:
//! - Scope depths: `0 = static`, `1 = fn/task body`, `2..N = nested blocks`.
//! - DMA leases (`ping_pong.wait_chunk()` / `lease.wait_chunk()`): tied to a canonical endpoint key
//!   and monotonic generation counter; invalidated on any subsequent `wait_chunk()` on the same source.

use std::collections::{BTreeMap, BTreeSet};
use zev_ast::Span;

/// A single active DMA or `byte_ring` buffer lease token.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LeaseToken {
    /// Canonical endpoint identifier (e.g. `"uart1_dma"` or `"usb_rx#write"`).
    pub source: String,
    /// Generation number of `source` at the moment this lease was acquired.
    pub generation: u64,
    /// Method that produced this lease (`"wait_chunk"`, `"read_slice"`, `"wait_read"`, `"grant_write"`).
    pub acq_method: &'static str,
    /// Source span of the acquisition call that produced this lease.
    pub acquired_at: Span,
}

impl PartialOrd for LeaseToken {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LeaseToken {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (
            &self.source,
            self.generation,
            self.acq_method,
            self.acquired_at.start,
            self.acquired_at.end,
        )
            .cmp(&(
                &other.source,
                other.generation,
                other.acq_method,
                other.acquired_at.start,
                other.acquired_at.end,
            ))
    }
}

/// Abstract provenance carried by expressions and variables.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Provenance {
    /// Maximum lexical scope depth of any referenced data (`0` = static/pure value,
    /// `1` = fn/task body or parameter, `2..N` = nested blocks).
    pub origin_depth: u32,
    /// Depth of the memory location pointed to when dereferencing `*p = val`
    /// (`Some(0)` for static globals, `Some(d)` for `&local` at depth `d`, `None` for non-pointers).
    pub pointee_depth: Option<u32>,
    /// True if `pointee` is a caller-owned location (pointer parameter) that local/leased refs cannot escape into.
    pub pointee_is_caller_or_static: bool,
    /// Names of current-function local variables (or local DMA leases) referenced by this value.
    pub local_origins: BTreeSet<String>,
    /// Indices of current-function parameters referenced by this value (for interprocedural return summaries).
    pub param_origins: BTreeSet<usize>,
    /// Active DMA buffer leases referenced by this value (directly or via wrapper struct fields).
    pub leases: BTreeSet<LeaseToken>,
    /// Declared function names referenced as function pointers (directly or inside structs).
    pub fn_callbacks: BTreeSet<String>,
    /// ISR-private `state` variable names whose address (`&state_var` or slice) was taken.
    pub isr_state_origins: BTreeSet<String>,
}

impl Provenance {
    /// Pure value / static data provenance (`origin_depth = 0`, no locals or leases).
    pub fn pure_static() -> Self {
        Self::default()
    }

    /// Provenance for taking `&global_item` (`origin_depth = 0`, points to static depth `0`).
    pub fn addr_of_static() -> Self {
        Self {
            origin_depth: 0,
            pointee_depth: Some(0),
            pointee_is_caller_or_static: true,
            ..Self::default()
        }
    }

    /// Provenance for a function/task parameter of reference-bearing type at parameter index `idx`.
    pub fn for_param(idx: usize, is_ptr: bool, target_prim: Option<String>) -> Self {
        let mut p = Self {
            origin_depth: 1,
            pointee_depth: if is_ptr { Some(0) } else { None },
            pointee_is_caller_or_static: is_ptr,
            ..Self::default()
        };
        p.param_origins.insert(idx);
        let _ = target_prim;
        p
    }

    /// Provenance for taking `&local_var` or slicing a local array declared at `decl_depth`.
    pub fn addr_of_local(name: impl Into<String>, decl_depth: u32) -> Self {
        let mut local_origins = BTreeSet::new();
        local_origins.insert(name.into());
        Self {
            origin_depth: decl_depth,
            pointee_depth: Some(decl_depth),
            pointee_is_caller_or_static: false,
            local_origins,
            ..Self::default()
        }
    }

    /// Provenance for taking `&state_var` of an ISR-private persistent `state` variable.
    pub fn addr_of_isr_state(name: impl Into<String>, decl_depth: u32) -> Self {
        let name = name.into();
        let mut local_origins = BTreeSet::new();
        local_origins.insert(name.clone());
        let mut isr_state_origins = BTreeSet::new();
        isr_state_origins.insert(name);
        Self {
            origin_depth: decl_depth,
            pointee_depth: Some(decl_depth),
            pointee_is_caller_or_static: false,
            local_origins,
            isr_state_origins,
            ..Self::default()
        }
    }

    /// Provenance for a DMA slice newly acquired via `source.wait_chunk()` at `scope_depth`.
    pub fn dma_lease(source: String, generation: u64, scope_depth: u32, span: Span) -> Self {
        let mut leases = BTreeSet::new();
        leases.insert(LeaseToken {
            source: source.clone(),
            generation,
            acq_method: "wait_chunk",
            acquired_at: span,
        });
        let mut local_origins = BTreeSet::new();
        local_origins.insert(format!("<dma lease {}>", source));
        Self {
            origin_depth: scope_depth,
            pointee_depth: Some(scope_depth),
            pointee_is_caller_or_static: false,
            local_origins,
            param_origins: BTreeSet::new(),
            leases,
            fn_callbacks: BTreeSet::new(),
            isr_state_origins: BTreeSet::new(),
        }
    }

    /// Provenance for a `*T` pointer borrowed via linear `Lease<T>.get()` at `scope_depth`.
    pub fn dma_lease_get(source: String, generation: u64, scope_depth: u32, span: Span) -> Self {
        let mut leases = BTreeSet::new();
        leases.insert(LeaseToken {
            source: source.clone(),
            generation,
            acq_method: "get",
            acquired_at: span,
        });
        let mut local_origins = BTreeSet::new();
        local_origins.insert(format!("<dma lease {}>", source));
        Self {
            origin_depth: scope_depth,
            pointee_depth: Some(scope_depth),
            pointee_is_caller_or_static: false,
            local_origins,
            param_origins: BTreeSet::new(),
            leases,
            fn_callbacks: BTreeSet::new(),
            isr_state_origins: BTreeSet::new(),
        }
    }

    /// Provenance for a slice lease acquired via `byte_ring` (`read_slice`, `wait_read`, `grant_write`).
    pub fn ring_lease(
        source: String,
        generation: u64,
        scope_depth: u32,
        span: Span,
        acq_method: &'static str,
    ) -> Self {
        let mut leases = BTreeSet::new();
        leases.insert(LeaseToken {
            source: source.clone(),
            generation,
            acq_method,
            acquired_at: span,
        });
        let mut local_origins = BTreeSet::new();
        local_origins.insert(format!("<ring lease {}>", source));
        Self {
            origin_depth: scope_depth,
            pointee_depth: Some(scope_depth),
            pointee_is_caller_or_static: false,
            local_origins,
            param_origins: BTreeSet::new(),
            leases,
            fn_callbacks: BTreeSet::new(),
            isr_state_origins: BTreeSet::new(),
        }
    }

    /// Provenance for a function identifier used as a callback value.
    pub fn fn_callback(fn_name: impl Into<String>) -> Self {
        let mut fn_callbacks = BTreeSet::new();
        fn_callbacks.insert(fn_name.into());
        Self {
            fn_callbacks,
            ..Self::default()
        }
    }

    /// Combine provenances (e.g. across struct literal fields, `or` fallback, or control-flow join).
    pub fn merge(&self, other: &Self) -> Self {
        let pointee_depth = match (self.pointee_depth, other.pointee_depth) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        Self {
            origin_depth: self.origin_depth.max(other.origin_depth),
            pointee_depth,
            pointee_is_caller_or_static: self.pointee_is_caller_or_static
                || other.pointee_is_caller_or_static,
            local_origins: self
                .local_origins
                .union(&other.local_origins)
                .cloned()
                .collect(),
            param_origins: self
                .param_origins
                .union(&other.param_origins)
                .copied()
                .collect(),
            leases: self.leases.union(&other.leases).cloned().collect(),
            fn_callbacks: self
                .fn_callbacks
                .union(&other.fn_callbacks)
                .cloned()
                .collect(),
            isr_state_origins: self
                .isr_state_origins
                .union(&other.isr_state_origins)
                .cloned()
                .collect(),
        }
    }

    /// Strip reference/lease provenance when producing a pure non-reference value (e.g. `u16`, `usize`, `bool`),
    /// while preserving `fn_callbacks` only if applicable.
    pub fn into_scalar_value(self) -> Self {
        Self::pure_static()
    }

    /// Preserve backing-buffer provenance (leases, stack depth, local/param/ISR origins) when
    /// projecting an address inside a pointer, slice, or lease (`&sl[i]`, `sl.ptr`, `&(*p)`).
    pub fn as_backing_ptr(mut self) -> Self {
        if self.pointee_depth.is_none() {
            if (self.origin_depth == 0 || !self.param_origins.is_empty())
                && self.local_origins.is_empty()
                && self.leases.is_empty()
                && self.isr_state_origins.is_empty()
                && self.origin_depth <= 1
            {
                self.pointee_depth = Some(0);
                self.pointee_is_caller_or_static = true;
            } else {
                self.pointee_depth = Some(self.origin_depth);
            }
        }
        self
    }

    /// Merge a memory location's `place` provenance (`self`, keeping its `pointee_depth` and
    /// `pointee_is_caller_or_static`) with the provenance origins of the value stored inside that
    /// place.
    pub fn merge_place_with_val_origins(mut self, val: &Self) -> Self {
        self.origin_depth = self.origin_depth.max(val.origin_depth);
        self.local_origins.extend(val.local_origins.iter().cloned());
        self.param_origins.extend(val.param_origins.iter().copied());
        self.leases.extend(val.leases.iter().cloned());
        self.fn_callbacks.extend(val.fn_callbacks.iter().cloned());
        self.isr_state_origins
            .extend(val.isr_state_origins.iter().cloned());
        self
    }

    /// Returns `true` if this value carries any local stack origin or active DMA lease that
    /// cannot outlive the current function frame.
    pub fn has_non_returning_origin(&self) -> bool {
        !self.local_origins.is_empty() || !self.leases.is_empty() || self.origin_depth > 1
    }
}

/// Tracks current valid DMA/`byte_ring` lease generations and invalidation spans across control flow.
#[derive(Clone, Debug, Default)]
pub struct LeaseEnv {
    /// Current generation number per lease source (`0` = never acquired yet).
    current_gen: BTreeMap<String, u64>,
    /// Span of the most recent call that invalidated prior generations for `source`.
    last_wait_span: BTreeMap<String, Span>,
    /// Name of the method that most recently invalidated `source`.
    last_inv_method: BTreeMap<String, &'static str>,
    /// Monotonic global counter so merged control-flow branches never accidentally collide.
    next_id: u64,
}

impl LeaseEnv {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the active generation for `source` without invalidating prior leases
    /// (creating generation `1` if `source` has not been acquired yet).
    pub fn acquire_current(&mut self, source: &str) -> u64 {
        if let Some(&gen) = self.current_gen.get(source) {
            if gen > 0 {
                return gen;
            }
        }
        self.next_id += 1;
        let gen = self.next_id;
        self.current_gen.insert(source.to_string(), gen);
        gen
    }

    /// Advance the lease generation for `source` using `inv_method`,
    /// invalidating all previously acquired leases from `source`.
    pub fn invalidate(&mut self, source: &str, span: Span, inv_method: &'static str) -> u64 {
        self.next_id += 1;
        let gen = self.next_id;
        self.current_gen.insert(source.to_string(), gen);
        self.last_wait_span.insert(source.to_string(), span);
        self.last_inv_method.insert(source.to_string(), inv_method);
        gen
    }

    /// Advance the DMA lease generation for `source` upon `source.wait_chunk()`,
    /// invalidating all previously acquired leases from `source`.
    pub fn acquire_next(&mut self, source: &str, span: Span) -> u64 {
        self.invalidate(source, span, "wait_chunk")
    }

    /// Return the name of the method that last invalidated `source` (defaults to `"wait_chunk"`).
    pub fn last_invalidator(&self, source: &str) -> &'static str {
        self.last_inv_method.get(source).copied().unwrap_or("wait_chunk")
    }

    /// Check if `token` is still valid in the current state.
    /// Returns `Err(invalidated_by_span)` if a subsequent invalidating call invalidated it.
    pub fn check_valid(&self, token: &LeaseToken) -> Result<(), Option<Span>> {
        match self.current_gen.get(&token.source) {
            Some(&cur) if cur == token.generation => Ok(()),
            _ => Err(self.last_wait_span.get(&token.source).copied()),
        }
    }

    /// Check all leases in `prov`, returning the first stale lease and the span that invalidated it.
    pub fn find_stale_lease<'a>(
        &self,
        prov: &'a Provenance,
    ) -> Option<(&'a LeaseToken, Option<Span>)> {
        for token in &prov.leases {
            if let Err(inv_span) = self.check_valid(token) {
                return Some((token, inv_span));
            }
        }
        None
    }

    /// Join two branch lease environments after `if / else` or `switch` arms.
    /// If a lease source's generation differs across branches, any lease held from before or
    /// only one branch is conservatively invalidated.
    pub fn join_branches(&mut self, branches: &[LeaseEnv]) {
        for b in branches {
            self.next_id = self.next_id.max(b.next_id);
        }
        let mut all_sources: BTreeSet<String> = self.current_gen.keys().cloned().collect();
        for b in branches {
            all_sources.extend(b.current_gen.keys().cloned());
        }
        for src in all_sources {
            let mut gens = BTreeSet::new();
            let mut latest_span = self.last_wait_span.get(&src).copied();
            let mut latest_method = self.last_inv_method.get(&src).copied();
            if branches.is_empty() {
                continue;
            }
            for b in branches {
                gens.insert(b.current_gen.get(&src).copied().unwrap_or(0));
                if let Some(sp) = b.last_wait_span.get(&src).copied() {
                    latest_span = Some(sp);
                    if let Some(m) = b.last_inv_method.get(&src).copied() {
                        latest_method = Some(m);
                    }
                }
            }
            if gens.len() == 1 {
                if let Some(&g) = gens.iter().next() {
                    if g > 0 {
                        self.current_gen.insert(src.clone(), g);
                    }
                }
            } else {
                // Branches diverged on DMA/`byte_ring` invalidations -> bump generation.
                self.next_id += 1;
                self.current_gen.insert(src.clone(), self.next_id);
            }
            if let Some(sp) = latest_span {
                self.last_wait_span.insert(src.clone(), sp);
            }
            if let Some(m) = latest_method {
                self.last_inv_method.insert(src, m);
            }
        }
    }

    /// Sources whose lease generation changed between `before` and `after`.
    pub fn modified_sources_since(&self, before: &LeaseEnv) -> BTreeMap<String, Option<Span>> {
        let mut out = BTreeMap::new();
        for (src, &gen) in &self.current_gen {
            if before.current_gen.get(src).copied().unwrap_or(0) != gen {
                out.insert(src.clone(), self.last_wait_span.get(src).copied());
            }
        }
        out
    }
}
