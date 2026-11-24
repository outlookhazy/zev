//! Whole-program call graph, 3-tier `FnEffect` inference, dual-specialization reachability,
//! shortest-path yield/spawn call-chain traces, C-callback checks, and Tarjan SCC cycle detection.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use zev_ast::Span;

use crate::model::FnEffect;

/// Kind of executable root/callable node in the call graph.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CallableKind {
    Fn,
    ExternCFn,
    Task,
    Isr,
}

/// Restricted atomic or control-flow context active at a call site.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CallContext {
    pub in_noyield: bool,
    pub in_critical: bool,
    pub in_isr: bool,
    pub in_defer: bool,
    pub in_loop: bool,
    pub after_diverge: bool,
    pub after_main_supervisor: bool,
}

/// Recorded call edge from `caller` to `callee` (`fn`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallEdge {
    pub caller: String,
    pub caller_kind: CallableKind,
    pub callee: String,
    pub span: Span,
    pub ctx: CallContext,
}

/// Recorded site where a user `fn` is passed as a callback pointer into an `extern "C"` function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternCallbackSite {
    pub extern_fn: String,
    pub callback_fn: String,
    pub span: Span,
}

/// Node in the whole-program call graph.
#[derive(Clone, Debug)]
pub struct CallNode {
    pub name: String,
    pub kind: CallableKind,
    pub direct_yield: bool,
    pub direct_effect: FnEffect,
    pub direct_yield_reason: Option<String>,
    pub direct_hard_yield_reason: Option<String>,
    pub can_yield: bool,
    pub effect: FnEffect,
    /// Transitive maximum effect across all outgoing edges (even across `noyield` blocks),
    /// used to ensure `critical`, `isr`, `defer`, and `extern "C"` never invoke soft-yielding compute.
    pub any_yield_effect: FnEffect,
    pub direct_spawn_task: Option<String>,
    pub spawns_in_boot: bool,
    /// Shortest witness chain explaining why `spawns_in_boot` is true.
    pub spawn_chain: Vec<String>,
    pub direct_attach_irq: Option<String>,
    pub attaches_in_boot: bool,
    /// Shortest witness chain explaining why `attaches_in_boot` is true.
    pub attach_chain: Vec<String>,
    /// Shortest witness chain explaining why `can_yield` is true
    /// (e.g. `["log_fault()", "write_page()", "checkpoint() [yields here]"]`).
    pub yield_chain: Vec<String>,
    pub span: Span,
}

/// Whole-program call graph and yield/boot-spawn inference engine.
#[derive(Clone, Debug, Default)]
pub struct CallGraph {
    pub nodes: BTreeMap<String, CallNode>,
    pub edges: Vec<CallEdge>,
    pub extern_callbacks: Vec<ExternCallbackSite>,
}

impl CallGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, name: String, kind: CallableKind, span: Span) {
        self.nodes.insert(
            name.clone(),
            CallNode {
                name,
                kind,
                direct_yield: false,
                direct_effect: FnEffect::NonYielding,
                direct_yield_reason: None,
                direct_hard_yield_reason: None,
                can_yield: false,
                effect: FnEffect::NonYielding,
                any_yield_effect: FnEffect::NonYielding,
                direct_spawn_task: None,
                spawns_in_boot: false,
                spawn_chain: Vec::new(),
                direct_attach_irq: None,
                attaches_in_boot: false,
                attach_chain: Vec::new(),
                yield_chain: Vec::new(),
                span,
            },
        );
    }

    pub fn mark_direct_yield(&mut self, name: &str, tier: FnEffect, reason: impl Into<String>) {
        if !tier.can_yield() {
            return;
        }
        if let Some(node) = self.nodes.get_mut(name) {
            if node.kind == CallableKind::ExternCFn {
                return;
            }
            let r = clean_yield_site_label(&reason.into());
            node.direct_yield = true;
            node.direct_effect = node.direct_effect.max(tier);
            node.effect = node.effect.max(tier);
            node.any_yield_effect = node.any_yield_effect.max(tier);
            node.can_yield = true;
            if node.direct_yield_reason.is_none() {
                node.direct_yield_reason = Some(r.clone());
                node.yield_chain = vec![
                    format_callable_step(&node.name),
                    format!("{} [yields here]", r),
                ];
            }
            if tier == FnEffect::HardBlocking && node.direct_hard_yield_reason.is_none() {
                node.direct_hard_yield_reason = Some(r);
            }
        }
    }

    pub fn mark_direct_spawn(&mut self, name: &str, task_name: &str) {
        if let Some(node) = self.nodes.get_mut(name) {
            if node.kind == CallableKind::ExternCFn {
                return;
            }
            node.spawns_in_boot = true;
            if node.direct_spawn_task.is_none() {
                node.direct_spawn_task = Some(task_name.to_string());
                node.spawn_chain = vec![
                    format_callable_step(&node.name),
                    format!("spawn {}", task_name),
                ];
            }
        }
    }

    pub fn mark_direct_attach(&mut self, name: &str, irq_name: &str) {
        if let Some(node) = self.nodes.get_mut(name) {
            if node.kind == CallableKind::ExternCFn {
                return;
            }
            node.attaches_in_boot = true;
            if node.direct_attach_irq.is_none() {
                node.direct_attach_irq = Some(irq_name.to_string());
                node.attach_chain = vec![
                    format_callable_step(&node.name),
                    format!("{}.attach(...)", irq_name),
                ];
            }
        }
    }

    pub fn add_edge(&mut self, edge: CallEdge) {
        self.edges.push(edge);
    }

    pub fn add_extern_callback(&mut self, site: ExternCallbackSite) {
        self.extern_callbacks.push(site);
    }

    /// Run fixed-point 3-tier `FnEffect` and `spawns_in_boot` inference across all functions and tasks,
    /// then populate shortest-hop `yield_chain` and `spawn_chain` traces.
    /// `extern "C"` functions remain `NonYielding`.
    pub fn infer_yields(&mut self) {
        for node in self.nodes.values_mut() {
            if node.kind == CallableKind::ExternCFn {
                node.effect = FnEffect::NonYielding;
                node.any_yield_effect = FnEffect::NonYielding;
                node.can_yield = false;
            } else {
                node.effect = node.direct_effect;
                node.any_yield_effect = node.direct_effect;
                node.can_yield = node.effect.can_yield();
            }
        }

        // 1. Propagate coroutine `effect` (`SoftYielding` suppressed across `noyield` edges; `HardBlocking` always propagates)
        //    and `any_yield_effect` (propagates across all edges for `critical`/`isr`/`defer`/`extern "C"` checks).
        loop {
            let mut changed = false;
            for edge in &self.edges {
                let Some((callee_eff, callee_any_eff, callee_spawns)) = self
                    .nodes
                    .get(&edge.callee)
                    .map(|c| (c.effect, c.any_yield_effect, c.spawns_in_boot))
                else {
                    continue;
                };

                if let Some(caller) = self.nodes.get_mut(&edge.caller) {
                    if caller.kind == CallableKind::ExternCFn {
                        continue;
                    }

                    let edge_eff = match callee_eff {
                        FnEffect::HardBlocking => FnEffect::HardBlocking,
                        FnEffect::SoftYielding if !edge.ctx.in_noyield => FnEffect::SoftYielding,
                        _ => FnEffect::NonYielding,
                    };
                    let next_eff = caller.effect.max(edge_eff);
                    if next_eff != caller.effect {
                        caller.effect = next_eff;
                        caller.can_yield = next_eff.can_yield();
                        changed = true;
                    }

                    let next_any = caller.any_yield_effect.max(callee_any_eff);
                    if next_any != caller.any_yield_effect {
                        caller.any_yield_effect = next_any;
                        changed = true;
                    }

                    if callee_spawns && caller.kind == CallableKind::Fn && !caller.spawns_in_boot {
                        caller.spawns_in_boot = true;
                        changed = true;
                    }
                }
            }
            for edge in &self.edges {
                let callee_attaches = self
                    .nodes
                    .get(&edge.callee)
                    .is_some_and(|c| c.attaches_in_boot);
                if callee_attaches {
                    if let Some(caller) = self.nodes.get_mut(&edge.caller) {
                        if caller.kind == CallableKind::Fn && !caller.attaches_in_boot {
                            caller.attaches_in_boot = true;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // 2. Populate shortest witness chains for `can_yield`, `spawns_in_boot`, and `attaches_in_boot`.
        let names: Vec<String> = self.nodes.keys().cloned().collect();
        for name in names {
            let (needs_yield_chain, needs_spawn_chain, needs_attach_chain) = self
                .nodes
                .get(&name)
                .map(|n| (n.can_yield, n.spawns_in_boot, n.attaches_in_boot))
                .unwrap_or((false, false, false));
            if needs_yield_chain {
                let chain = self.shortest_yield_chain(&name);
                if let Some(node) = self.nodes.get_mut(&name) {
                    node.yield_chain = chain;
                }
            }
            if needs_spawn_chain {
                let sc = self.shortest_spawn_chain(&name);
                if let Some(node) = self.nodes.get_mut(&name) {
                    node.spawn_chain = sc;
                }
            }
            if needs_attach_chain {
                let ac = self.shortest_attach_chain(&name);
                if let Some(node) = self.nodes.get_mut(&name) {
                    node.attach_chain = ac;
                }
            }
        }
    }

    /// Compute `(soft_yielding_in_coro, soft_yielding_in_noyield, dual_specialized_fns)` via 2-state
    /// `(node, Coro | NoYield)` reachability from `main`, all `task`s, uncalled root `fn`s, and
    /// function-pointer targets (which invoke the non-yielding specialization).
    pub fn compute_soft_specializations(
        &self,
        fn_ptr_targets: &BTreeSet<String>,
    ) -> (BTreeSet<String>, BTreeSet<String>, BTreeSet<String>) {
        #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
        enum ExecMode {
            Coro,
            NoYield,
        }

        let mut has_incoming: BTreeSet<&str> = BTreeSet::new();
        let mut adj: BTreeMap<&str, Vec<&CallEdge>> = BTreeMap::new();
        for e in &self.edges {
            if self.nodes.contains_key(&e.callee) {
                has_incoming.insert(e.callee.as_str());
                adj.entry(e.caller.as_str()).or_default().push(e);
            }
        }

        let mut queue: VecDeque<(&str, ExecMode)> = VecDeque::new();
        let mut visited: BTreeSet<(&str, ExecMode)> = BTreeSet::new();

        for (name, node) in &self.nodes {
            let is_root = match node.kind {
                CallableKind::Task => true,
                CallableKind::Fn => name == "main" || !has_incoming.contains(name.as_str()),
                CallableKind::ExternCFn | CallableKind::Isr => false,
            };
            if is_root && visited.insert((name.as_str(), ExecMode::Coro)) {
                queue.push_back((name.as_str(), ExecMode::Coro));
            }
        }

        let mut soft_in_coro = BTreeSet::new();
        let mut soft_in_noyield = BTreeSet::new();

        for fname in fn_ptr_targets {
            if let Some(node) = self.nodes.get(fname.as_str()) {
                if node.kind == CallableKind::Fn && node.effect == FnEffect::SoftYielding {
                    soft_in_noyield.insert(node.name.clone());
                    if visited.insert((node.name.as_str(), ExecMode::NoYield)) {
                        queue.push_back((node.name.as_str(), ExecMode::NoYield));
                    }
                }
            }
        }

        while let Some((curr, mode)) = queue.pop_front() {
            let Some(out_edges) = adj.get(curr) else {
                continue;
            };
            for edge in out_edges {
                // Calls inside `critical`, `isr`, or `defer` are illegal if yielding, never stack-specialized.
                if edge.ctx.in_critical || edge.ctx.in_isr || edge.ctx.in_defer {
                    continue;
                }
                let Some(callee_node) = self.nodes.get(&edge.callee) else {
                    continue;
                };
                if callee_node.kind != CallableKind::Fn {
                    continue;
                }
                let next_mode = if mode == ExecMode::NoYield || edge.ctx.in_noyield {
                    ExecMode::NoYield
                } else {
                    ExecMode::Coro
                };

                if callee_node.effect == FnEffect::SoftYielding {
                    match next_mode {
                        ExecMode::Coro => {
                            soft_in_coro.insert(callee_node.name.clone());
                        }
                        ExecMode::NoYield => {
                            soft_in_noyield.insert(callee_node.name.clone());
                        }
                    }
                }

                if visited.insert((callee_node.name.as_str(), next_mode)) {
                    queue.push_back((callee_node.name.as_str(), next_mode));
                }
            }
        }

        let dual_specialized: BTreeSet<String> = soft_in_coro
            .intersection(&soft_in_noyield)
            .cloned()
            .collect();

        (soft_in_coro, soft_in_noyield, dual_specialized)
    }

    /// Shortest call-chain path from `start` down to any active direct yield operation.
    pub fn shortest_yield_chain(&self, start: &str) -> Vec<String> {
        let Some(start_node) = self.nodes.get(start) else {
            return Vec::new();
        };
        if !start_node.can_yield {
            return Vec::new();
        }
        if start_node.direct_yield {
            let site = start_node.direct_yield_reason.as_deref().unwrap_or("yield");
            return vec![
                format_callable_step(start),
                format!("{} [yields here]", site),
            ];
        }

        self.bfs_yield_chain(
            start,
            |e, callee| {
                callee.effect == FnEffect::HardBlocking
                    || (callee.effect == FnEffect::SoftYielding && !e.ctx.in_noyield)
            },
            |n| {
                n.direct_yield
                    .then(|| n.direct_yield_reason.as_deref().unwrap_or("yield"))
            },
        )
    }

    /// Shortest call-chain path from `start` down to any direct yield (even through `noyield` wrappers),
    /// used for `critical`, `isr`, `defer`, and `extern "C"` diagnostics.
    pub fn shortest_any_yield_chain(&self, start: &str) -> Vec<String> {
        let Some(start_node) = self.nodes.get(start) else {
            return Vec::new();
        };
        if !start_node.any_yield_effect.can_yield() {
            return Vec::new();
        }
        if start_node.direct_yield {
            let site = start_node.direct_yield_reason.as_deref().unwrap_or("yield");
            return vec![
                format_callable_step(start),
                format!("{} [yields here]", site),
            ];
        }

        self.bfs_yield_chain(
            start,
            |_e, callee| callee.any_yield_effect.can_yield(),
            |n| {
                n.direct_yield
                    .then(|| n.direct_yield_reason.as_deref().unwrap_or("yield"))
            },
        )
    }

    /// Shortest call-chain path from `start` down to a `HardBlocking` direct yield operation,
    /// used when reporting `E0003` inside `noyield { ... }` (where `SoftYielding` calls are permitted).
    pub fn shortest_hard_yield_chain(&self, start: &str) -> Vec<String> {
        let Some(start_node) = self.nodes.get(start) else {
            return Vec::new();
        };
        if start_node.effect != FnEffect::HardBlocking {
            return Vec::new();
        }
        if start_node.direct_effect == FnEffect::HardBlocking {
            let site = start_node
                .direct_hard_yield_reason
                .as_deref()
                .or(start_node.direct_yield_reason.as_deref())
                .unwrap_or("yield");
            return vec![
                format_callable_step(start),
                format!("{} [yields here]", site),
            ];
        }

        self.bfs_yield_chain(
            start,
            |_e, callee| callee.effect == FnEffect::HardBlocking,
            |n| {
                (n.direct_effect == FnEffect::HardBlocking).then(|| {
                    n.direct_hard_yield_reason
                        .as_deref()
                        .or(n.direct_yield_reason.as_deref())
                        .unwrap_or("yield")
                })
            },
        )
    }

    /// Shortest call-chain path from `start` down to a `spawn` statement.
    pub fn shortest_spawn_chain(&self, start: &str) -> Vec<String> {
        let Some(start_node) = self.nodes.get(start) else {
            return Vec::new();
        };
        if !start_node.spawns_in_boot {
            return Vec::new();
        }
        if let Some(t) = &start_node.direct_spawn_task {
            return vec![format_callable_step(start), format!("spawn {}", t)];
        }
        self.bfs_boot_chain(
            start,
            |n| n.spawns_in_boot,
            |n| n.direct_spawn_task.as_ref().map(|t| format!("spawn {t}")),
        )
    }

    /// Shortest call-chain path from `start` down to an `irq.attach` / `irq.attach_shared` call.
    pub fn shortest_attach_chain(&self, start: &str) -> Vec<String> {
        let Some(start_node) = self.nodes.get(start) else {
            return Vec::new();
        };
        if !start_node.attaches_in_boot {
            return Vec::new();
        }
        if let Some(q) = &start_node.direct_attach_irq {
            return vec![format_callable_step(start), format!("{q}.attach(...)")];
        }
        self.bfs_boot_chain(
            start,
            |n| n.attaches_in_boot,
            |n| {
                n.direct_attach_irq
                    .as_ref()
                    .map(|q| format!("{q}.attach(...)"))
            },
        )
    }

    fn bfs_boot_chain(
        &self,
        start: &str,
        node_ok: impl Fn(&CallNode) -> bool,
        leaf_label: impl Fn(&CallNode) -> Option<String>,
    ) -> Vec<String> {
        let mut adj: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for e in &self.edges {
            if self.nodes.get(&e.callee).is_some_and(&node_ok) {
                adj.entry(e.caller.as_str())
                    .or_default()
                    .push(e.callee.as_str());
            }
        }
        for list in adj.values_mut() {
            list.sort_unstable();
            list.dedup();
        }

        let mut queue: VecDeque<&str> = VecDeque::new();
        let mut parent: BTreeMap<&str, &str> = BTreeMap::new();
        let mut visited: BTreeSet<&str> = BTreeSet::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(curr) = queue.pop_front() {
            if let Some(neighbors) = adj.get(curr) {
                for &nxt in neighbors {
                    if !visited.insert(nxt) {
                        continue;
                    }
                    parent.insert(nxt, curr);
                    if let Some(nxt_node) = self.nodes.get(nxt) {
                        if let Some(label) = leaf_label(nxt_node) {
                            let mut hops = vec![format_callable_step(nxt)];
                            let mut p = nxt;
                            while let Some(&prev) = parent.get(p) {
                                hops.push(format_callable_step(prev));
                                p = prev;
                            }
                            hops.reverse();
                            hops.push(label);
                            return hops;
                        }
                    }
                    queue.push_back(nxt);
                }
            }
        }

        vec![format_callable_step(start)]
    }

    fn bfs_yield_chain<'a>(
        &'a self,
        start: &'a str,
        edge_ok: impl Fn(&CallEdge, &CallNode) -> bool,
        leaf_site: impl Fn(&'a CallNode) -> Option<&'a str>,
    ) -> Vec<String> {
        let mut adj: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for e in &self.edges {
            if let Some(c) = self.nodes.get(&e.callee) {
                if edge_ok(e, c) {
                    adj.entry(e.caller.as_str())
                        .or_default()
                        .push(e.callee.as_str());
                }
            }
        }
        for list in adj.values_mut() {
            list.sort_unstable();
            list.dedup();
        }

        let mut queue: VecDeque<&str> = VecDeque::new();
        let mut parent: BTreeMap<&str, &str> = BTreeMap::new();
        let mut visited: BTreeSet<&str> = BTreeSet::new();

        queue.push_back(start);
        visited.insert(start);

        while let Some(curr) = queue.pop_front() {
            if let Some(neighbors) = adj.get(curr) {
                for &nxt in neighbors {
                    if !visited.insert(nxt) {
                        continue;
                    }
                    parent.insert(nxt, curr);
                    if let Some(nxt_node) = self.nodes.get(nxt) {
                        if let Some(site) = leaf_site(nxt_node) {
                            let mut fn_hops = vec![format_callable_step(nxt)];
                            let mut p = nxt;
                            while let Some(&prev) = parent.get(p) {
                                fn_hops.push(format_callable_step(prev));
                                p = prev;
                            }
                            fn_hops.reverse();
                            fn_hops.push(format!("{} [yields here]", site));
                            return fn_hops;
                        }
                    }
                    queue.push_back(nxt);
                }
            }
        }

        vec![format_callable_step(start)]
    }

    /// Distinct sorted callees of `caller`.
    pub fn callees_of(&self, caller: &str) -> Vec<String> {
        let mut set = BTreeSet::new();
        for e in &self.edges {
            if e.caller == caller {
                set.insert(e.callee.clone());
            }
        }
        set.into_iter().collect()
    }

    /// Distinct sorted yielding coroutine callees of `caller` (after `infer_yields`).
    /// Excludes `SoftYielding` callees invoked exclusively inside `noyield { ... }`.
    pub fn yielding_callees_of(&self, caller: &str) -> Vec<String> {
        let mut set = BTreeSet::new();
        for e in &self.edges {
            if e.caller != caller {
                continue;
            }
            if let Some(c) = self.nodes.get(&e.callee) {
                let yields_at_edge = c.effect == FnEffect::HardBlocking
                    || (c.effect == FnEffect::SoftYielding && !e.ctx.in_noyield);
                if yields_at_edge {
                    set.insert(e.callee.clone());
                }
            }
        }
        set.into_iter().collect()
    }

    /// Detect all call-graph cycles containing at least one yielding function (`can_yield == true`).
    pub fn find_yielding_cycles(&self) -> Vec<(Vec<String>, Span)> {
        self.find_cycles_matching(|n| n.can_yield)
    }

    /// Detect all call-graph cycles containing at least one boot-spawning function (`spawns_in_boot == true`).
    pub fn find_spawning_cycles(&self) -> Vec<(Vec<String>, Span)> {
        self.find_cycles_matching(|n| n.spawns_in_boot)
    }

    /// Detect all call-graph cycles containing at least one boot-attaching function (`attaches_in_boot == true`).
    pub fn find_attaching_cycles(&self) -> Vec<(Vec<String>, Span)> {
        self.find_cycles_matching(|n| n.attaches_in_boot)
    }

    fn find_cycles_matching(&self, pred: impl Fn(&CallNode) -> bool) -> Vec<(Vec<String>, Span)> {
        let mut adj: BTreeMap<String, Vec<(String, Span)>> = BTreeMap::new();
        for name in self.nodes.keys() {
            adj.insert(name.clone(), Vec::new());
        }
        for edge in &self.edges {
            if self.nodes.contains_key(&edge.caller) && self.nodes.contains_key(&edge.callee) {
                adj.entry(edge.caller.clone())
                    .or_default()
                    .push((edge.callee.clone(), edge.span));
            }
        }

        struct Tarjan<'a> {
            adj: &'a BTreeMap<String, Vec<(String, Span)>>,
            index: usize,
            indices: BTreeMap<String, usize>,
            lowlink: BTreeMap<String, usize>,
            stack: Vec<String>,
            on_stack: BTreeSet<String>,
            sccs: Vec<Vec<String>>,
        }

        impl<'a> Tarjan<'a> {
            fn strongconnect(&mut self, v: &str) {
                let idx = self.index;
                self.index += 1;
                self.indices.insert(v.to_string(), idx);
                self.lowlink.insert(v.to_string(), idx);
                self.stack.push(v.to_string());
                self.on_stack.insert(v.to_string());

                if let Some(neighbors) = self.adj.get(v) {
                    for (w, _) in neighbors {
                        if !self.indices.contains_key(w) {
                            self.strongconnect(w);
                            let lw = self.lowlink[w];
                            if let Some(lv) = self.lowlink.get_mut(v) {
                                if lw < *lv {
                                    *lv = lw;
                                }
                            }
                        } else if self.on_stack.contains(w) {
                            let iw = self.indices[w];
                            if let Some(lv) = self.lowlink.get_mut(v) {
                                if iw < *lv {
                                    *lv = iw;
                                }
                            }
                        }
                    }
                }

                if self.lowlink[v] == self.indices[v] {
                    let mut scc = Vec::new();
                    while let Some(w) = self.stack.pop() {
                        self.on_stack.remove(&w);
                        let done = w == v;
                        scc.push(w);
                        if done {
                            break;
                        }
                    }
                    scc.sort();
                    self.sccs.push(scc);
                }
            }
        }

        let mut tarjan = Tarjan {
            adj: &adj,
            index: 0,
            indices: BTreeMap::new(),
            lowlink: BTreeMap::new(),
            stack: Vec::new(),
            on_stack: BTreeSet::new(),
            sccs: Vec::new(),
        };

        for name in self.nodes.keys() {
            if !tarjan.indices.contains_key(name) {
                tarjan.strongconnect(name);
            }
        }

        let mut cycles = Vec::new();
        for scc in tarjan.sccs {
            let is_cycle = if scc.len() > 1 {
                true
            } else if let Some(only) = scc.first() {
                adj.get(only)
                    .is_some_and(|ns| ns.iter().any(|(dst, _)| dst == only))
            } else {
                false
            };

            if !is_cycle {
                continue;
            }

            let matches_pred = scc.iter().any(|f| self.nodes.get(f).is_some_and(&pred));
            if !matches_pred {
                continue;
            }

            let scc_set: BTreeSet<&str> = scc.iter().map(String::as_str).collect();
            let start = &scc[0];
            if let Some((cycle_path, span)) = find_cycle_from(start, &scc_set, &adj) {
                cycles.push((cycle_path, span));
            }
        }

        cycles
    }
}

fn clean_yield_site_label(raw: &str) -> String {
    raw.trim_matches('\'').to_string()
}

fn format_callable_step(node_key: &str) -> String {
    if let Some(t) = node_key.strip_prefix("task::") {
        format!("task {}", t)
    } else if let Some(i) = node_key.strip_prefix("isr::") {
        format!("isr {}", i)
    } else {
        format!("{}()", node_key)
    }
}

fn find_cycle_from(
    start: &str,
    scc_set: &BTreeSet<&str>,
    adj: &BTreeMap<String, Vec<(String, Span)>>,
) -> Option<(Vec<String>, Span)> {
    let neighbors = adj.get(start)?;
    for (first, first_span) in neighbors {
        if !scc_set.contains(first.as_str()) {
            continue;
        }
        if first == start {
            return Some((vec![start.to_string(), start.to_string()], *first_span));
        }
        let mut queue = VecDeque::new();
        let mut parent: BTreeMap<String, String> = BTreeMap::new();
        let mut visited: BTreeSet<String> = BTreeSet::new();
        queue.push_back(first.clone());
        visited.insert(first.clone());

        while let Some(curr) = queue.pop_front() {
            if let Some(nexts) = adj.get(&curr) {
                for (nxt, back_span) in nexts {
                    if !scc_set.contains(nxt.as_str()) {
                        continue;
                    }
                    if nxt == start {
                        let mut path = vec![start.to_string(), curr.clone()];
                        let mut p = curr;
                        while let Some(prev) = parent.get(&p) {
                            path.push(prev.clone());
                            p = prev.clone();
                        }
                        path.push(start.to_string());
                        path.reverse();
                        return Some((path, *back_span));
                    }
                    if visited.insert(nxt.clone()) {
                        parent.insert(nxt.clone(), curr.clone());
                        queue.push_back(nxt.clone());
                    }
                }
            }
        }
    }
    None
}
