"""search.py – Parallel beam search with knowledge-guided search.

The search uses *beam search* (bounded-width BFS) and supports:

* **Parallel expansion**: spread beam nodes across CPU cores via
  ``ThreadPoolExecutor`` for near-linear speed-up.
* **Knowledge shortcuts**: before expanding, the search checks the
  ``KnowledgeStore`` proven cache; a cache hit skips the search
  entirely and returns the stored proof.
* **Per-depth deduplication**: conclusions already derived within the
  same depth level are skipped to avoid redundant states.
* **Knowledge-guided scoring**: accumulated experience boosts states
  reached via historically successful rules, creating a mutual
  promotion loop where past proofs guide future search.
* **Rule ordering**: rules are re-ordered by predicted usefulness for
  the current goal predicate, so the most promising rules are tried
  first within each expansion.

Author:  Jiangsheng Yu
License: MIT
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, FrozenSet, List, Optional, Sequence

from .dsl import Fact, GeoState, Goal
from .lean_bridge import LeanChecker
from .rules import Rule, RuleApplication

logger = logging.getLogger(__name__)

Scorer = Callable[[GeoState, Goal], float]

_CPU_COUNT = os.cpu_count() or 4


@dataclass
class SearchConfig:
    beam_width: int = 8
    max_depth: int = 8
    parallel_workers: int = 0  # 0 = min(beam_width, cpu_count)


@dataclass
class SearchResult:
    success: bool
    final_state: GeoState
    explored_nodes: int
    cache_hits: int = 0


def default_scorer(state: GeoState, goal: Goal) -> float:
    """Goal-directed scorer: rewards proximity to goal over raw fact count."""
    if state.has_fact(goal.fact):
        return 1e6

    goal_pred = goal.fact.predicate
    goal_args = set(goal.fact.args)

    # Scan for facts matching goal predicate and point overlap
    best_overlap = 0
    pred_matches = 0
    for f in state.facts:
        if f.predicate == goal_pred:
            pred_matches += 1
            overlap = len(goal_args & set(f.args))
            if overlap > best_overlap:
                best_overlap = overlap

    # Proximity dominates; breadth is logarithmic to avoid
    # rewarding explosive symmetry/transitivity derivations.
    import math
    proximity = best_overlap * 30.0 + pred_matches * 5.0
    breadth = math.log1p(len(state.facts))
    return proximity + breadth


def make_knowledge_scorer(
    knowledge_store: "KnowledgeStore",  # type: ignore[name-defined]
    goal: Goal,
) -> Scorer:
    """Build a scorer that incorporates accumulated experience.

    The knowledge-guided scorer combines goal proximity (from
    ``default_scorer``) with a **rule-experience bonus**: states
    reached via rules that historically succeeded for this goal
    predicate receive a higher score.

    This is the primary mechanism by which knowledge guides search,
    closing the mutual promotion loop:

        past success → rule ordering → faster future proofs
        → new knowledge → updated ordering → ...
    """
    from .knowledge import KnowledgeStore

    # Pre-compute rule scores for this goal predicate
    rule_scores: dict[str, float] = {}
    try:
        rule_order = knowledge_store.suggest_rule_order(goal.fact.predicate)
        if rule_order:
            # Normalise to [0, 1] range
            max_score = rule_order[0][1] if rule_order else 1.0
            for rule_name, score in rule_order:
                rule_scores[rule_name] = score / max(max_score, 1e-9)
    except Exception:
        pass  # graceful degradation if no experience yet

    # Pre-compute known-good rule chains for sub-goal matching
    known_chains: list[tuple[str, ...]] = []
    try:
        known_chains = knowledge_store.proven_rule_chains(
            goal.fact.predicate, max_chains=5,
        )
    except Exception:
        pass

    # Build set of rules that appear in known-good chains
    chain_rules: set[str] = set()
    for chain in known_chains:
        chain_rules.update(chain)

    def scorer(state: GeoState, goal_: Goal) -> float:
        base = default_scorer(state, goal_)
        if base >= 1e6:
            return base  # already solved

        # Rule-experience bonus: reward states reached via proven-good rules
        rule_bonus = 0.0
        if state.history and rule_scores:
            last_step = state.history[-1]
            rs = rule_scores.get(last_step.rule_name, 0.0)
            rule_bonus = rs * 15.0  # significant but not dominant

            # Extra bonus if the rule is in a known-good chain
            if last_step.rule_name in chain_rules:
                rule_bonus += 5.0

        return base + rule_bonus

    return scorer


def clone_state(state: GeoState) -> GeoState:
    """Deep-copy a GeoState (facts set + history list + predicate index)."""
    return GeoState(facts=set(state.facts), history=list(state.history))


# ── Single-node expansion (unit of parallel work) ────────────────────

def _expand_node(
    state: GeoState,
    rules: Sequence[Rule],
    checker: LeanChecker,
    seen_conclusions: set[Fact],
) -> List[GeoState]:
    """Expand one beam node: apply all rules, check each step, return children."""
    children: List[GeoState] = []
    for rule in rules:
        for app in rule.apply(state):
            # Skip conclusions already generated in this depth level
            if app.step.conclusion_fact in seen_conclusions:
                continue
            check = checker.check_step(state, app.step)
            if not check.ok:
                continue
            child = clone_state(state)
            child.add_fact(app.step.conclusion_fact, via=app.step)
            children.append(child)
            seen_conclusions.add(app.step.conclusion_fact)
    return children


# ── Main entry point ─────────────────────────────────────────────────

def beam_search(
    init_state: GeoState,
    goal: Goal,
    rules: Sequence[Rule],
    checker: LeanChecker,
    config: SearchConfig,
    scorer: Scorer = default_scorer,
    knowledge_store: Optional["KnowledgeStore"] = None,  # type: ignore[name-defined]
) -> SearchResult:
    """Beam search with optional parallelism and knowledge-store lookup.

    When a ``knowledge_store`` is provided AND the caller did not supply
    a custom scorer, the search automatically uses a **knowledge-guided
    scorer** that incorporates accumulated experience.  This closes the
    mutual promotion loop: proofs found by search enrich the knowledge
    store, which in turn guides future searches to be more efficient.
    """
    from .knowledge import KnowledgeStore  # deferred to avoid circular

    # ── Knowledge cache shortcut ─────────────────────────────
    cache_hits = 0
    if knowledge_store is not None:
        frozen_assumptions = frozenset(init_state.facts)
        cached = knowledge_store.lookup_proven(frozen_assumptions, goal.fact)
        if cached is not None:
            cache_hits += 1
            solved = clone_state(init_state)
            for step in cached.steps:
                solved.add_fact(step.conclusion_fact, via=step)
            return SearchResult(True, solved, 0, cache_hits)

    # ── Auto-upgrade to knowledge-guided scorer ──────────────
    active_scorer = scorer
    if knowledge_store is not None and scorer is default_scorer:
        active_scorer = make_knowledge_scorer(knowledge_store, goal)

    # ── Rule ordering from experience ────────────────────────
    ordered_rules = list(rules)
    if knowledge_store is not None:
        try:
            rule_order = knowledge_store.suggest_rule_order(goal.fact.predicate)
            if rule_order:
                rule_priority = {
                    name: idx for idx, (name, _) in enumerate(rule_order)
                }
                # Sort rules: those with experience data first (by score),
                # unknown rules afterward in original order.
                ordered_rules.sort(
                    key=lambda r: rule_priority.get(r.name, len(rule_priority))
                )
        except Exception:
            pass  # graceful degradation

    # ── Determine parallelism ────────────────────────────────
    n_workers = config.parallel_workers or min(config.beam_width, _CPU_COUNT)
    use_parallel = n_workers > 1 and config.beam_width > 1

    frontier: List[GeoState] = [clone_state(init_state)]
    explored_nodes = 0

    for depth in range(config.max_depth):
        # Early termination check
        for state in frontier:
            if state.has_fact(goal.fact):
                _record_success(knowledge_store, init_state, goal, state)
                return SearchResult(True, state, explored_nodes, cache_hits)

        # ── Expand all frontier nodes ────────────────────────
        next_frontier: List[GeoState] = []
        # Shared set to deduplicate conclusions within a depth level
        depth_seen: set[Fact] = set()

        if use_parallel and len(frontier) > 1:
            next_frontier = _parallel_expand(
                frontier, ordered_rules, checker, depth_seen, n_workers,
            )
        else:
            for state in frontier:
                next_frontier.extend(
                    _expand_node(state, ordered_rules, checker, depth_seen)
                )

        explored_nodes += len(frontier)

        if not next_frontier:
            break

        # ── Score, prune, keep top-k ─────────────────────────
        next_frontier.sort(key=lambda s: active_scorer(s, goal), reverse=True)
        frontier = next_frontier[: config.beam_width]

    # ── Final check ──────────────────────────────────────────
    best = max(frontier, key=lambda s: active_scorer(s, goal), default=clone_state(init_state))
    success = best.has_fact(goal.fact)
    if success:
        _record_success(knowledge_store, init_state, goal, best)
    return SearchResult(success, best, explored_nodes, cache_hits)


# ── Parallel expansion helper ────────────────────────────────────────

def _parallel_expand(
    frontier: List[GeoState],
    rules: Sequence[Rule],
    checker: LeanChecker,
    depth_seen: set[Fact],
    n_workers: int,
) -> List[GeoState]:
    all_children: List[GeoState] = []
    # Share depth_seen directly across workers — under CPython's GIL,
    # set.__contains__ and set.add are effectively atomic, so the worst
    # case is an occasional benign duplicate (pruned at scoring time).
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_expand_node, state, rules, checker, depth_seen): i
            for i, state in enumerate(frontier)
        }
        for future in as_completed(futures):
            all_children.extend(future.result())
    return all_children


# ── Knowledge recording helper ───────────────────────────────────────

def _record_success(
    store: Optional["KnowledgeStore"],  # type: ignore[name-defined]
    init_state: GeoState,
    goal: Goal,
    solved_state: GeoState,
) -> None:
    if store is None:
        return
    from .knowledge import KnowledgeStore
    store.record_proven(
        assumptions=frozenset(init_state.facts),
        goal=goal.fact,
        steps=solved_state.history,
        source="beam-search",
    )
