//! Whole-program call graph, shortest-path yield call-chain inference, C-callback checks,
//! and Tarjan SCC yielding-recursion cycle detection.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use zev_ast::Span;

/// Kind of executable root/callable node in the call graph.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CallableKind {
    Fn,
    ExternCFn,
    Task,
    Isr,
}

/// Restricted atomic context active at a call site.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CallContext {
    pub in_noyield: bool,
    pub in_critical: bool,
    pub in_isr: bool,
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
    pub direct_yield_reason: Option<String>,
    pub can_yield: bool,
    /// Shortest witness chain explaining why `can_yield` is true
    /// (e.g. `["log_fault()", "write_page()", "checkpoint() [yields here]"]`).
    pub yield_chain: Vec<String>,
    pub span: Span,
}

/// Whole-program call graph and yield-inference engine.
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
                direct_yield_reason: None,
                can_yield: false,
                yield_chain: Vec::new(),
                span,
            },
        );
    }

    pub fn mark_direct_yield(&mut self, name: &str, reason: impl Into<String>) {
        if let Some(node) = self.nodes.get_mut(name) {
            if node.kind == CallableKind::ExternCFn {
                return;
            }
            node.direct_yield = true;
            node.can_yield = true;
            if node.direct_yield_reason.is_none() {
                let r = clean_yield_site_label(&reason.into());
                node.direct_yield_reason = Some(r.clone());
                node.yield_chain = vec![
                    format_callable_step(&node.name),
                    format!("{} [yields here]", r),
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

    /// Run fixed-point transitive `can_yield` inference across all functions and tasks,
    /// then populate exact BFS shortest-hop `yield_chain` traces to a direct yield site.
    /// `extern "C"` functions remain `can_yield = false`.
    pub fn infer_yields(&mut self) {
        loop {
            let mut changed = false;
            for edge in &self.edges {
                let callee_yields = self
                    .nodes
                    .get(&edge.callee)
                    .is_some_and(|c| c.can_yield);
                if callee_yields {
                    if let Some(caller) = self.nodes.get_mut(&edge.caller) {
                        if caller.kind != CallableKind::ExternCFn && !caller.can_yield {
                            caller.can_yield = true;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Compute shortest BFS witness chain from every yielding node down to a `direct_yield` leaf.
        let names: Vec<String> = self.nodes.keys().cloned().collect();
        for name in names {
            if self.nodes.get(&name).is_some_and(|n| n.can_yield) {
                let chain = self.shortest_yield_chain(&name);
                if let Some(node) = self.nodes.get_mut(&name) {
                    node.yield_chain = chain;
                }
            }
        }
    }

    /// Compute the shortest call-chain path from `start` down to a direct yield operation,
    /// formatted as `["f1()", "f2()", "checkpoint() [yields here]"]`.
    pub fn shortest_yield_chain(&self, start: &str) -> Vec<String> {
        let Some(start_node) = self.nodes.get(start) else {
            return Vec::new();
        };
        if !start_node.can_yield {
            return Vec::new();
        }
        if start_node.direct_yield {
            let site = start_node
                .direct_yield_reason
                .as_deref()
                .unwrap_or("yield");
            return vec![
                format_callable_step(start),
                format!("{} [yields here]", site),
            ];
        }

        // Build deterministic outgoing adjacency map.
        let mut adj: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for e in &self.edges {
            if self.nodes.get(&e.callee).is_some_and(|n| n.can_yield) {
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
                        if nxt_node.direct_yield {
                            let mut fn_hops = vec![format_callable_step(nxt)];
                            let mut p = nxt;
                            while let Some(&prev) = parent.get(p) {
                                fn_hops.push(format_callable_step(prev));
                                p = prev;
                            }
                            fn_hops.reverse();
                            let site = nxt_node
                                .direct_yield_reason
                                .as_deref()
                                .unwrap_or("yield");
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

    /// Distinct sorted yielding callees of `caller` (after `infer_yields`).
    pub fn yielding_callees_of(&self, caller: &str) -> Vec<String> {
        self.callees_of(caller)
            .into_iter()
            .filter(|callee| self.nodes.get(callee).is_some_and(|n| n.can_yield))
            .collect()
    }

    /// Detect all call-graph cycles containing at least one yielding function (`can_yield == true`).
    /// Returns `(cycle_path, call_span)` pairs such as `(["a", "b", "a"], span)`.
    pub fn find_yielding_cycles(&self) -> Vec<(Vec<String>, Span)> {
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
                            let lv = self.lowlink.get_mut(v).unwrap();
                            if lw < *lv {
                                *lv = lw;
                            }
                        } else if self.on_stack.contains(w) {
                            let iw = self.indices[w];
                            let lv = self.lowlink.get_mut(v).unwrap();
                            if iw < *lv {
                                *lv = iw;
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

            let has_yield = scc
                .iter()
                .any(|f| self.nodes.get(f).is_some_and(|n| n.can_yield));
            if !has_yield {
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
