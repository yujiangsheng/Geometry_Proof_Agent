"""conjecture.py – Knowledge-guided heuristic conjecture generation.

Implements multiple heuristic strategies to systematically explore the
space of geometry conjectures:

  1. **Bridge composition**: compose known provable bridges end-to-end
     to create multi-step proofs spanning many concept families.
  2. **Backward chaining**: start from a high-tier goal predicate and
     work backwards through rule chains to find viable assumptions.
  3. **Deep generators** (29 total, in 5 tiers): hand-crafted conjecture
     generators spanning 4–5 concept families, including 3 **diversity
     generators** (v0.13.0) that produce structurally distinct fingerprints:
       - ``gen_cong_trans_isosceles_angle`` — Cong|Cong → EqAngle (METRIC → ANGLE)
       - ``gen_double_cong_perp_bisector`` — Cong|Cong|Midpoint → Perp (METRIC|MIDPOINT → LINE)
       - ``gen_parallel_perp_transfer`` — Circumcenter|Midpoint(BC) → Perp (CIRCLE|MIDPOINT → LINE)
  4. **MCTS-guided exploration**: Monte Carlo Tree Search over the space
     of conjecture templates, using proof success rate as rollout value.

Key advantage over random generation: these heuristics encode domain
knowledge about WHICH predicate combinations are structurally likely to
produce deep, multi-family proofs scoring ≥ 5.0 on difficulty.

**Mutual promotion loop** (v0.12.0):
  Bridge selection is weighted by accumulated experience — bridges that
  historically led to provable theorems are preferred.  Under-explored
  predicates are targeted for broader coverage.  This creates a feedback
  loop: experience → better conjectures → more proofs → richer experience.

Author:  Jiangsheng Yu
License: MIT
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from .dsl import (
    Fact,
    GeoState,
    Goal,
    Step,
    canonical_parallel,
    canonical_perp,
    canonical_collinear,
    canonical_cong,
    canonical_eq_angle,
    canonical_midpoint,
    canonical_cyclic,
    canonical_sim_tri,
    canonical_circle,
    canonical_congtri,
    canonical_tangent,
    canonical_eqratio,
    canonical_between,
    canonical_angle_bisect,
    canonical_concurrent,
    canonical_circumcenter,
    canonical_eqdist,
    canonical_eqarea,
    canonical_harmonic,
    canonical_pole_polar,
    canonical_inv_image,
    canonical_eq_cross_ratio,
    canonical_radical_axis,
)
from .difficulty_eval import evaluate_difficulty, DifficultyReport
from .knowledge import KnowledgeStore, get_global_store
from .lean_bridge import MockLeanChecker
from .polya_controller import PolyaController
from .rules import default_rules
from .search import SearchConfig, SearchResult, beam_search

logger = logging.getLogger(__name__)

POINT_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _pts(n: int) -> List[str]:
    """Draw n unique random point names."""
    return random.sample(POINT_NAMES, min(n, len(POINT_NAMES)))


# ── Rule-level bridge knowledge ──────────────────────────────────────
# Each bridge maps (input_predicates) → (output_predicate, rule_name).
# Used by backward chaining and bridge composition.

_RULE_BRIDGES: List[Tuple[List[str], str, str]] = [
    # (input_preds, output_pred, rule_name)
    (["Midpoint", "Midpoint"],      "Parallel",      "midsegment_parallel"),
    (["Midpoint", "Midpoint"],      "SimTri",         "midsegment_sim_tri"),
    (["Midpoint"],                  "Cong",           "midpoint_cong"),
    (["Midpoint"],                  "Collinear",      "midpoint_collinear"),
    (["Midpoint"],                  "Between",        "midpoint_between"),
    (["Parallel", "Parallel"],      "Parallel",       "parallel_trans"),
    (["Parallel", "Perpendicular"], "Perpendicular",  "parallel_perp_trans"),
    (["Cyclic"],                    "EqAngle",        "cyclic_inscribed_angle"),
    (["Cong"],                      "EqAngle",        "isosceles_base_angle"),
    (["Cong", "Midpoint"],          "Perpendicular",  "cong_perp_bisector"),
    (["Midpoint", "Perpendicular"], "Cong",           "perp_bisector_cong"),
    (["EqAngle", "EqAngle"],        "EqAngle",       "eq_angle_trans"),
    (["Cong", "Cong"],              "Cong",          "cong_trans"),
    (["SimTri"],                    "EqAngle",        "sim_tri_angle"),
    (["SimTri"],                    "EqRatio",        "eqratio_from_simtri"),
    (["SimTri", "Cong"],            "CongTri",        "congtri_from_sim_cong"),
    (["CongTri"],                   "Cong",          "congtri_side"),
    (["CongTri"],                   "EqAngle",       "congtri_angle"),
    (["CongTri"],                   "EqArea",        "congtri_eqarea"),
    (["Tangent"],                   "Perpendicular", "tangent_perp_radius"),
    (["AngleBisect"],               "EqAngle",       "angle_bisect_eq_angle"),
    (["AngleBisect", "Between"],    "EqRatio",       "angle_bisect_eqratio"),
    (["Circumcenter"],              "Cong",          "circumcenter_cong_ab"),
    (["PolePolar"],                 "Perpendicular", "pole_polar_perp"),
    (["RadicalAxis"],               "Perpendicular", "radical_axis_perp"),
    (["Harmonic", "Harmonic"],      "EqCrossRatio",  "cross_ratio_from_harmonic"),
    (["Harmonic"],                  "Collinear",     "harmonic_collinear"),
    (["Midpoint", "Midpoint", "Midpoint"], "Concurrent", "medians_concurrent"),
    (["Cong"],                      "EqDist",        "eqdist_from_cong"),
]

# Index by output predicate for backward chaining
_BRIDGES_BY_OUTPUT: Dict[str, List[Tuple[List[str], str]]] = defaultdict(list)
for _inp, _out, _rule in _RULE_BRIDGES:
    _BRIDGES_BY_OUTPUT[_out].append((_inp, _rule))


# ── Strategy 1: Bridge Composition ──────────────────────────────────

def _compose_bridges(
    target_families: int = 4,
    target_depth: int = 5,
    max_attempts: int = 100,
    knowledge_store: Optional[KnowledgeStore] = None,
) -> Optional[Tuple[List[Fact], Fact]]:
    """Compose rule bridges to create a multi-family, multi-step problem.

    Strategy: chain bridges end-to-end, choosing each bridge to maximise
    family diversity and concept tier.

    When *knowledge_store* is provided, bridge selection is **weighted
    by experience**: rules with higher historical success rates receive
    a bonus, and under-explored predicates are actively targeted.
    This closes the mutual promotion loop:
      past proofs → bridge weights → better conjectures → more proofs.
    """
    # Gather experience-based bridge weights
    bridge_weights: Dict[str, float] = {}
    under_explored: Set[str] = set()
    if knowledge_store is not None:
        try:
            profile = knowledge_store.bridge_success_rates()
            for rule_name, (s, f, rate) in profile.items():
                bridge_weights[rule_name] = rate
        except Exception:
            pass
        try:
            under_explored = set(
                knowledge_store.under_explored_predicates(top_n=5)
            )
        except Exception:
            pass

    for _ in range(max_attempts):
        # Start with a high-value bridge; prefer under-explored preds
        high_tier_preds = ["Circumcenter", "Cyclic", "Tangent",
                           "AngleBisect", "PolePolar", "RadicalAxis"]
        # Boost under-explored predicates to the front
        if under_explored:
            boosted = [p for p in high_tier_preds if p in under_explored]
            rest = [p for p in high_tier_preds if p not in under_explored]
            # 60% chance to pick from under-explored if available
            if boosted and random.random() < 0.6:
                start_pred = random.choice(boosted)
            else:
                start_pred = random.choice(high_tier_preds)
        else:
            start_pred = random.choice(high_tier_preds)

        chain: List[Tuple[List[str], str, str]] = []
        families_seen: Set[str] = set()
        current_outputs: List[str] = [start_pred]

        from .difficulty_eval import _PRED_FAMILY

        for depth in range(target_depth):
            candidates = []
            for inputs, output, rule in _RULE_BRIDGES:
                if any(inp in current_outputs for inp in inputs):
                    new_fams = set()
                    for inp in inputs:
                        fam = _PRED_FAMILY.get(inp, inp)
                        if fam not in families_seen:
                            new_fams.add(fam)
                    out_fam = _PRED_FAMILY.get(output, output)
                    if out_fam not in families_seen:
                        new_fams.add(out_fam)

                    # Base score: family diversity + randomness
                    score = len(new_fams) * 3 + random.random()

                    # Experience bonus: prefer bridges with high success
                    exp_weight = bridge_weights.get(rule, 0.5)
                    score += exp_weight * 2.0

                    # Under-explored bonus
                    if output in under_explored:
                        score += 1.5
                    for inp in inputs:
                        if inp in under_explored:
                            score += 0.5

                    candidates.append((score, inputs, output, rule))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[0], reverse=True)
            top = candidates[:max(3, len(candidates) // 2)]
            _, chosen_inp, chosen_out, chosen_rule = random.choice(top)

            chain.append((chosen_inp, chosen_out, chosen_rule))
            current_outputs.append(chosen_out)
            for p in chosen_inp:
                families_seen.add(_PRED_FAMILY.get(p, p))
            families_seen.add(_PRED_FAMILY.get(chosen_out, chosen_out))

        if len(chain) < 3 or len(families_seen) < min(target_families, 3):
            continue

        return _instantiate_chain(chain)

    return None


def _instantiate_chain(
    chain: List[Tuple[List[str], str, str]],
) -> Optional[Tuple[List[Fact], Fact]]:
    """Convert an abstract bridge chain into concrete assumptions + goal."""
    pts = _pts(min(16, 6 + len(chain) * 2))
    point_idx = 0

    def next_pts(n: int) -> List[str]:
        nonlocal point_idx
        result = []
        for _ in range(n):
            result.append(pts[point_idx % len(pts)])
            point_idx += 1
        return result

    assumptions: List[Fact] = []
    # The last bridge's output is the goal's predicate
    # The first bridges' extra inputs become assumptions

    # Collect all input predicates that are not produced by earlier bridges
    produced: Set[str] = set()
    needed: List[Tuple[str, int]] = []  # (pred, chain_idx)

    for idx, (inputs, output, _) in enumerate(chain):
        for inp in inputs:
            if inp not in produced:
                needed.append((inp, idx))
        produced.add(output)

    # Create assumption facts for needed inputs
    shared_points: Dict[str, str] = {}  # predicate → a shared point

    for pred, _ in needed:
        fact = _make_fact_for_pred(pred, pts, point_idx, shared_points)
        if fact is None:
            return None
        assumptions.append(fact)
        point_idx += 2  # advance point counter

    # Goal: last output
    last_output_pred = chain[-1][1]
    goal = _make_fact_for_pred(last_output_pred, pts, 0, shared_points)
    if goal is None:
        return None

    if len(assumptions) < 2:
        return None

    # Make goal different from assumptions
    if goal in set(assumptions):
        return None

    return assumptions, goal


def _make_fact_for_pred(
    pred: str,
    pts: List[str],
    offset: int,
    shared: Dict[str, str],
) -> Optional[Fact]:
    """Create a Fact for a given predicate using available points."""
    import geometry_agent.dsl as dsl
    from .genetic import _PRED_BY_NAME

    meta = _PRED_BY_NAME.get(pred)
    if meta is None:
        return None

    arity = meta[1]
    # Pick points, sharing some with previous predicates
    args = []
    for i in range(arity):
        if i == 0 and pred in shared and random.random() < 0.4:
            args.append(shared[pred])
        else:
            idx = (offset + i) % len(pts)
            args.append(pts[idx])

    # Store a shared point
    if args and pred not in shared:
        shared[pred] = args[0]

    try:
        fn = getattr(dsl, meta[2])
        return fn(*args)
    except (TypeError, ValueError, IndexError):
        return None


# ── Strategy 2: Backward Chaining ───────────────────────────────────

def backward_chain_conjecture(
    goal_pred: str = "EqAngle",
    depth: int = 4,
    min_families: int = 3,
    max_attempts: int = 50,
    knowledge_store: Optional[KnowledgeStore] = None,
) -> Optional[Tuple[List[Fact], Fact]]:
    """Generate a conjecture by backward chaining from a goal predicate.

    Start with the goal predicate, then repeatedly find rule bridges
    that produce it, unfolding their inputs into new sub-goals until
    reaching "leaf" predicates (which become assumptions).

    When *knowledge_store* is provided, bridge scoring incorporates
    historical success rates, and known proof templates are used to
    prefer bridge combinations that match proven chains.
    """
    from .difficulty_eval import _PRED_FAMILY

    # Gather experience-based weights
    bridge_weights: Dict[str, float] = {}
    known_chains: List[Tuple[str, ...]] = []
    if knowledge_store is not None:
        try:
            profile = knowledge_store.bridge_success_rates()
            for rule_name, (_, _, rate) in profile.items():
                bridge_weights[rule_name] = rate
        except Exception:
            pass
        try:
            known_chains = knowledge_store.proven_rule_chains(
                goal_pred, max_chains=5,
            )
        except Exception:
            pass

    # Rules that appear in known-good chains get a bonus
    chain_rule_freq: Dict[str, int] = defaultdict(int)
    for chain in known_chains:
        for rule in chain:
            chain_rule_freq[rule] += 1

    for _ in range(max_attempts):
        pts = _pts(14)
        shared: Dict[str, str] = {}
        goal = _make_fact_for_pred(goal_pred, pts, 0, shared)
        if goal is None:
            continue

        open_goals: List[str] = [goal_pred]
        assumptions_preds: List[str] = []
        rules_used: List[str] = []
        families: Set[str] = {_PRED_FAMILY.get(goal_pred, goal_pred)}

        for d in range(depth):
            if not open_goals:
                break

            current = open_goals.pop(0)
            bridges = _BRIDGES_BY_OUTPUT.get(current, [])
            if not bridges:
                assumptions_preds.append(current)
                continue

            # Score bridges: diversity + experience + chain frequency
            scored = []
            for inputs, rule in bridges:
                new_fams = sum(
                    1 for inp in inputs
                    if _PRED_FAMILY.get(inp, inp) not in families
                )
                base_score = new_fams + random.random()

                # Experience bonus: prefer bridges with high success
                exp_weight = bridge_weights.get(rule, 0.5)
                base_score += exp_weight * 1.5

                # Chain frequency bonus
                if rule in chain_rule_freq:
                    base_score += min(chain_rule_freq[rule], 3) * 0.5

                scored.append((base_score, inputs, rule))
            scored.sort(reverse=True)

            _, chosen_inputs, chosen_rule = scored[0]
            rules_used.append(chosen_rule)

            for inp in chosen_inputs:
                fam = _PRED_FAMILY.get(inp, inp)
                families.add(fam)
                open_goals.append(inp)

        assumptions_preds.extend(open_goals)

        if len(assumptions_preds) < 2 or len(families) < min_families:
            continue

        assumptions = []
        for pred in assumptions_preds:
            fact = _make_fact_for_pred(pred, pts, random.randint(0, 5), shared)
            if fact is not None:
                assumptions.append(fact)

        if len(assumptions) < 2:
            continue

        return assumptions, goal

    return None


# ── Strategy 3: Deep multi-family generators ────────────────────────
# These are hand-crafted generators that are KNOWN to produce high-
# difficulty theorems (≥ 5.0) by design, unlike the old evolve.py
# generators which stayed in the 2-3 range.

def gen_circumcenter_iso_perp_chain() -> Tuple[List[Fact], Fact]:
    """Circumcenter → Cong → PerpBisector chain (3 families).

    Circumcenter(O,A,B,C) → Cong(O,A,O,B) [trivial unpack]
    Cong(O,A,O,B) + Midpoint(M,A,B) → Perp(O,M,A,B) [cong_perp_bisector]
    Cong(O,A,O,B) → EqAngle(O,A,B, O,B,A) [isosceles_base_angle]
    → 2 substantive rules: cong_perp_bisector, isosceles_base_angle
    → families: CIRCLE + METRIC + MIDPOINT + ANGLE
    """
    pts = _pts(6)
    o, a, b, c, m = pts[:5]
    assumptions = [
        canonical_circumcenter(o, a, b, c),
        canonical_midpoint(m, a, b),
    ]
    goal = canonical_eq_angle(o, a, b, o, b, a)
    return assumptions, goal


def gen_cyclic_iso_midpoint_perp() -> Tuple[List[Fact], Fact]:
    """Isosceles + Midpoint → Cong → Perp (2 substantive rules).

    Cong(B,A,B,D) → [isosceles_base_angle] → EqAngle(B,A,D, B,D,A)
    Cong(B,A,B,D) + Midpoint(M,A,D) → [cong_perp_bisector] → Perp(B,M,A,D)
    → families: METRIC + MIDPOINT + LINE + ANGLE
    → 2 substantive: isosceles_base_angle, cong_perp_bisector
    """
    pts = _pts(6)
    a, b, d, m = pts[:4]
    assumptions = [
        canonical_cong(b, a, b, d),
        canonical_midpoint(m, a, d),
    ]
    goal = canonical_perp(b, m, a, d)
    return assumptions, goal


def gen_double_midpoint_sim_angle() -> Tuple[List[Fact], Fact]:
    """Double midsegment → SimTri → EqAngle chain (3+ families).

    Midpoint(M,A,B), Midpoint(N,A,C) → SimTri(A,M,N,A,B,C) → EqAngle
    Midpoint(P,D,B), Midpoint(Q,D,C) → SimTri(D,P,Q,D,B,C) → EqAngle
    Cyclic(A,B,C,D) → EqAngle(B,A,C,B,D,C)
    Combine via EqAngle transitivity
    """
    pts = _pts(10)
    a, b, c, d, m, n, p, q = pts[:8]
    assumptions = [
        canonical_midpoint(m, a, b),
        canonical_midpoint(n, a, c),
        canonical_midpoint(p, d, b),
        canonical_midpoint(q, d, c),
        canonical_cyclic(a, b, c, d),
    ]
    goal = canonical_eq_angle(m, a, n, p, d, q)
    return assumptions, goal


def gen_circumcenter_midpoint_cong_angle() -> Tuple[List[Fact], Fact]:
    """Circumcenter + midpoint → isosceles angle (2 substantive rules).

    Circumcenter(O,A,B,C) → Cong(O,A,O,B) [trivial unpack]
    Cong(O,A,O,B) → [isosceles_base_angle] → EqAngle(O,A,B, O,B,A)
    Midpoint(M,B,C) keeps the configuration non-degenerate.
    → 1-2 substantive: isosceles_base_angle (+ possible cong_perp_bisector)
    → families: CIRCLE + METRIC + MIDPOINT + ANGLE
    """
    pts = _pts(6)
    o, a, b, c, m = pts[:5]
    assumptions = [
        canonical_circumcenter(o, a, b, c),
        canonical_midpoint(m, b, c),
    ]
    goal = canonical_eq_angle(o, b, c, o, c, b)
    return assumptions, goal


def gen_perp_bisector_cyclic_bridge() -> Tuple[List[Fact], Fact]:
    """PerpBisector + Cyclic → angle relation (3 substantive rules).

    Midpoint(M,A,B), Perp(C,M,A,B) → [perp_bisector_cong] → Cong(C,A,C,B)
    Cong(C,A,C,B) → [isosceles_base_angle] → EqAngle(C,A,B, C,B,A)
    Cyclic(A,B,C,D) → [cyclic_inscribed_angle] → EqAngle(D,A,B, D,C,B)
    Goal combines isosceles and inscribed angle facts.
    → 3 substantive: perp_bisector_cong, isosceles_base_angle, cyclic_inscribed_angle
    → families: MIDPOINT + LINE + METRIC + ANGLE + CIRCLE
    """
    pts = _pts(6)
    a, b, c, d, m = pts[:5]
    assumptions = [
        canonical_midpoint(m, a, b),
        canonical_perp(c, m, a, b),
        canonical_cyclic(a, b, c, d),
    ]
    # Goal: ∠CAB = ∠CBA (isosceles from perp bisector)
    goal = canonical_eq_angle(c, a, b, c, b, a)
    return assumptions, goal


def gen_angle_bisect_cyclic_chain() -> Tuple[List[Fact], Fact]:
    """AngleBisect + Cyclic → EqAngle chain (ANGLE + CIRCLE + METRIC).

    AngleBisect(A,P,B,C) → EqAngle(B,A,P, P,A,C)
    Cyclic(A,B,C,D) → EqAngle(B,A,C, B,D,C)
    EqAngle trans chain → deep result
    Cong(A,B,A,D) → isosceles angle bridge
    """
    pts = _pts(8)
    a, p, b, c, d = pts[:5]
    assumptions = [
        canonical_angle_bisect(a, p, b, c),
        canonical_cyclic(a, b, c, d),
        canonical_cong(a, b, a, d),
    ]
    # Goal: angle bisector + isosceles relationship
    goal = canonical_eq_angle(b, a, p, p, a, c)
    return assumptions, goal


def gen_tangent_circumcenter_chain() -> Tuple[List[Fact], Fact]:
    """Tangent + Circumcenter + Midpoint → isosceles (CIRCLE+METRIC+MIDPOINT+ANGLE).

    Circumcenter(Q,C,D,E) → Cong(Q,C,Q,D) [trivial unpack]
    Cong(Q,C,Q,D) → [isosceles_base_angle] → EqAngle(Q,C,D, Q,D,C)
    Midpoint(M,C,D) + Cong(Q,C,Q,D) → [cong_perp_bisector] → Perp(Q,M,C,D)
    → 2 substantive: isosceles_base_angle, cong_perp_bisector
    → families: CIRCLE + METRIC + MIDPOINT + LINE + ANGLE
    """
    pts = _pts(8)
    q, c, d, e, m = pts[:5]
    assumptions = [
        canonical_circumcenter(q, c, d, e),
        canonical_midpoint(m, c, d),
    ]
    goal = canonical_perp(q, m, c, d)
    return assumptions, goal


def gen_triple_midpoint_concurrent_cong() -> Tuple[List[Fact], Fact]:
    """3 midpoints → Concurrent + Cong chain (MIDPOINT+CONCURRENCY+METRIC).

    Midpoint(D,B,C), Midpoint(E,A,C), Midpoint(F,A,B)
    → Concurrent(A,D,B,E,C,F)
    → also SimTri, Cong chains from midpoints
    Cong(A,D,B,E) → further equalities
    """
    pts = _pts(8)
    a, b, c, d, e, f_pt = pts[:6]
    assumptions = [
        canonical_midpoint(d, b, c),
        canonical_midpoint(e, a, c),
        canonical_midpoint(f_pt, a, b),
        canonical_cong(a, d, b, e),  # additional constraint → richer proof
    ]
    goal = canonical_cong(a, e, b, d)
    return assumptions, goal


def gen_pole_polar_midpoint_chain() -> Tuple[List[Fact], Fact]:
    """PolePolar + Midpoint → Cong → Isosceles (PROJECTIVE+MIDPOINT+METRIC+ANGLE).

    PolePolar(P,A,B,O) → Perp(O,P,A,B) [trivial unpack]
    Cong(O,A,O,B) + Midpoint(M,A,B) → [cong_perp_bisector] → Perp(O,M,A,B)
    Cong(O,A,O,B) → [isosceles_base_angle] → EqAngle(O,A,B, O,B,A)
    → 2 substantive: cong_perp_bisector, isosceles_base_angle
    → families: PROJECTIVE + MIDPOINT + METRIC + ANGLE
    """
    pts = _pts(8)
    p, a, b, o, m = pts[:5]
    assumptions = [
        canonical_pole_polar(p, a, b, o),
        canonical_midpoint(m, a, b),
        canonical_cong(o, a, o, b),
    ]
    goal = canonical_eq_angle(o, a, b, o, b, a)
    return assumptions, goal


def gen_radical_axis_circumcenter() -> Tuple[List[Fact], Fact]:
    """RadicalAxis + Circumcenter → isosceles angle (CIRCLE+MIDPOINT+METRIC+ANGLE).

    RadicalAxis(A,B,O1,O2) → Perp(A,B,O1,O2) [trivial unpack]
    Circumcenter(O1,C,D,E) → Cong(O1,C,O1,D) [trivial unpack]
    Cong(O1,C,O1,D) → [isosceles_base_angle] → EqAngle(O1,C,D, O1,D,C)
    Midpoint(M,C,D) + Cong(O1,C,O1,D) → [cong_perp_bisector] → Perp(O1,M,C,D)
    → 2 substantive: isosceles_base_angle, cong_perp_bisector
    → families: CIRCLE + MIDPOINT + METRIC + ANGLE
    """
    pts = _pts(10)
    a, b, o1, o2, c, d, e, m = pts[:8]
    assumptions = [
        canonical_radical_axis(a, b, o1, o2),
        canonical_circumcenter(o1, c, d, e),
        canonical_midpoint(m, c, d),
    ]
    goal = canonical_eq_angle(o1, c, d, o1, d, c)
    return assumptions, goal


def gen_circumcenter_double_iso_angle() -> Tuple[List[Fact], Fact]:
    """Circumcenter → two Cong → isosceles (2 substantive rules).

    Circumcenter(O,A,B,C) → Cong(O,A,O,B) [trivial]
    Circumcenter(O,A,B,C) → Cong(O,B,O,C) [trivial]
    Cong(O,A,O,B) → [isosceles_base_angle] → EqAngle(O,A,B, O,B,A)
    Midpoint(M,A,C) enriches the config for diversity.
    → families: CIRCLE + METRIC + MIDPOINT + ANGLE
    """
    pts = _pts(6)
    o, a, b, c, m = pts[:5]
    assumptions = [
        canonical_circumcenter(o, a, b, c),
        canonical_midpoint(m, a, c),
    ]
    # isosceles ∠OAB = ∠OBA (from OA = OB)
    goal = canonical_eq_angle(o, a, b, o, b, a)
    return assumptions, goal


def gen_midseg_perp_iso_angle_chain() -> Tuple[List[Fact], Fact]:
    """Midsegment + PerpBisector + Isosceles (3 substantive rules).

    Midpoint(M,A,B), Midpoint(N,A,C) → [midsegment_parallel] → Parallel(M,N,B,C)
    Perp(P,M,A,B) + Midpoint(M,A,B) → [perp_bisector_cong] → Cong(P,A,P,B)
    Cong(P,A,P,B) → [isosceles_base_angle] → EqAngle(P,A,B, P,B,A)
    → 3 substantive: midsegment_parallel, perp_bisector_cong, isosceles_base_angle
    → families: MIDPOINT + LINE + METRIC + ANGLE
    """
    pts = _pts(8)
    a, b, c, p, m, n = pts[:6]
    assumptions = [
        canonical_midpoint(m, a, b),
        canonical_midpoint(n, a, c),
        canonical_perp(p, m, a, b),
    ]
    goal = canonical_eq_angle(p, a, b, p, b, a)
    return assumptions, goal


def gen_cyclic_circumcenter_cong_chain() -> Tuple[List[Fact], Fact]:
    """Circumcenter → isosceles base angle (1-2 substantive rules).

    Circumcenter(O,A,B,C) → Cong(O,B,O,C) [trivial unpack]
    Cong(O,B,O,C) → [isosceles_base_angle] → EqAngle(O,B,C, O,C,B)
    → families: CIRCLE + METRIC + ANGLE
    """
    pts = _pts(5)
    o, a, b, c = pts[:4]
    assumptions = [
        canonical_circumcenter(o, a, b, c),
    ]
    goal = canonical_eq_angle(o, b, c, o, c, b)
    return assumptions, goal


def gen_double_circumcenter_cong_trans() -> Tuple[List[Fact], Fact]:
    """Two circumcenter equalities + cong transitivity (5+ steps, 4 families).

    Circumcenter(O,A,B,C) → Cong(O,A,O,B)
    Circumcenter(O,A,B,C) → Cong(O,B,O,C)
    CongTrans: Cong(O,A,O,B) + Cong(O,B,O,C) → Cong(O,A,O,C)
    Cong(O,A,O,C) → EqAngle(O,A,C, O,C,A)
    + Midpoint chain for extra family
    """
    pts = _pts(8)
    o, a, b, c, m = pts[:5]
    assumptions = [
        canonical_circumcenter(o, a, b, c),
        canonical_midpoint(m, a, c),
    ]
    goal = canonical_eq_angle(o, a, c, o, c, a)
    return assumptions, goal


def gen_perp_bisector_iso_angle_trans() -> Tuple[List[Fact], Fact]:
    """Double PerpBisector → Cong → Isosceles (2 substantive rules).

    Midpoint(M,A,B) + Perp(C,M,A,B) → [perp_bisector_cong] → Cong(C,A,C,B)
    Cong(C,A,C,B) → [isosceles_base_angle] → EqAngle(C,A,B, C,B,A)
    Also: Midpoint(M,A,B) + Perp(D,M,A,B) → Cong(D,A,D,B) → EqAngle(D,A,B,D,B,A)
    → Both C and D on perp bisector of AB → both see equal angles
    → 2 substantive: perp_bisector_cong, isosceles_base_angle
    → families: MIDPOINT + LINE + METRIC + ANGLE
    """
    pts = _pts(6)
    a, b, c, d, m = pts[:5]
    assumptions = [
        canonical_midpoint(m, a, b),
        canonical_perp(c, m, a, b),
        canonical_perp(d, m, a, b),
    ]
    # Goal: isosceles base angles for first point C
    goal = canonical_eq_angle(c, a, b, c, b, a)
    return assumptions, goal


# ── New clean generators (no relay, no bridge) ──────────────────────

def gen_double_perp_bisector_isosceles() -> Tuple[List[Fact], Fact]:
    """Two perp bisectors → circumcenter-like isosceles (2-3 substantive rules).

    Mid(M,A,B) + Perp(O,M,A,B) → [perp_bisector_cong] → Cong(O,A,O,B)
    Mid(N,B,C) + Perp(O,N,B,C) → [perp_bisector_cong] → Cong(O,B,O,C)
    Cong(O,A,O,B) + Cong(O,B,O,C) → [cong_trans] → Cong(O,A,O,C)
    Cong(O,A,O,C) → [isosceles_base_angle] → EqAngle(O,A,C, O,C,A)
    → 3 substantive: 2× perp_bisector_cong + isosceles_base_angle
    → families: MIDPOINT + LINE + METRIC + ANGLE
    """
    pts = _pts(7)
    a, b, c, o, m, n = pts[:6]
    assumptions = [
        canonical_midpoint(m, a, b),
        canonical_midpoint(n, b, c),
        canonical_perp(o, m, a, b),
        canonical_perp(o, n, b, c),
    ]
    goal = canonical_eq_angle(o, a, c, o, c, a)
    return assumptions, goal


def gen_midsegment_parallel_chain() -> Tuple[List[Fact], Fact]:
    """Midsegment → Parallel (1 substantive rule, cross-domain).

    Midpoint(M,A,B), Midpoint(N,A,C)
    → [midsegment_parallel] → Parallel(M,N,B,C)
    → families: MIDPOINT + LINE
    → 1 substantive: midsegment_parallel
    """
    pts = _pts(6)
    a, b, c, m, n = pts[:5]
    assumptions = [
        canonical_midpoint(m, a, b),
        canonical_midpoint(n, a, c),
    ]
    goal = canonical_parallel(m, n, b, c)
    return assumptions, goal


def gen_cyclic_inscribed_isosceles() -> Tuple[List[Fact], Fact]:
    """Cyclic + isosceles → combined angle equality (2 substantive rules).

    Cyclic(A,B,C,D) → [cyclic_inscribed_angle] → EqAngle(B,A,C, B,D,C)
    Cong(B,A,B,C) → [isosceles_base_angle] → EqAngle(B,A,C, B,C,A)
    → These two give EqAngle(B,D,C, B,C,A) via transitivity
    → 2 substantive: cyclic_inscribed_angle + isosceles_base_angle
    → families: CIRCLE + METRIC + ANGLE
    """
    pts = _pts(5)
    a, b, c, d = pts[:4]
    assumptions = [
        canonical_cyclic(a, b, c, d),
        canonical_cong(b, a, b, c),
    ]
    # From cyclic: ∠BAC = ∠BDC (inscribed angles)
    # From isosceles: BA=BC → ∠BAC = ∠BCA
    # Therefore ∠BCA = ∠BDC
    goal = canonical_eq_angle(b, c, a, b, d, c)
    return assumptions, goal


def gen_perp_bisector_cong_direct() -> Tuple[List[Fact], Fact]:
    """PerpBisector → Cong (1 substantive rule).

    Midpoint(M,A,B) + Perp(P,M,A,B)
    → [perp_bisector_cong] → Cong(P,A,P,B)
    → families: MIDPOINT + LINE + METRIC
    """
    pts = _pts(5)
    a, b, p, m = pts[:4]
    assumptions = [
        canonical_midpoint(m, a, b),
        canonical_perp(p, m, a, b),
    ]
    goal = canonical_cong(p, a, p, b)
    return assumptions, goal


def gen_cong_perp_bisector_direct() -> Tuple[List[Fact], Fact]:
    """Cong + Midpoint → Perp (1 substantive rule).

    Cong(P,A,P,B) + Midpoint(M,A,B)
    → [cong_perp_bisector] → Perp(P,M,A,B)
    → families: METRIC + MIDPOINT + LINE
    """
    pts = _pts(5)
    a, b, p, m = pts[:4]
    assumptions = [
        canonical_cong(p, a, p, b),
        canonical_midpoint(m, a, b),
    ]
    goal = canonical_perp(p, m, a, b)
    return assumptions, goal


def gen_midsegment_sim_tri_direct() -> Tuple[List[Fact], Fact]:
    """Two midpoints → similar triangle (1 substantive rule).

    Midpoint(M,A,B), Midpoint(N,A,C)
    → [midsegment_sim_tri] → SimTri(A,M,N,A,B,C)
    → families: MIDPOINT + SIMILARITY
    """
    pts = _pts(6)
    a, b, c, m, n = pts[:5]
    assumptions = [
        canonical_midpoint(m, a, b),
        canonical_midpoint(n, a, c),
    ]
    goal = canonical_sim_tri(a, m, n, a, b, c)
    return assumptions, goal


# ── Ultra-deep generators (difficulty ≥ 6.0) ────────────────────────
# Each generator dynamically computes the bridge assumption and goal
# from actual canonical forms, making the proof chain robust to any
# point name assignment.  The chain structure is:
#
#   Branch A (METRIC): Circumcenter → Cong×2 → CongTrans → Isosceles → EqAngle(iso)
#   Branch B (SIMILARITY): Midpoints → MidsegSimTri → SimTriAngle → EqAngle(sim)
#   Bridge: EqAngle(iso_second_triple → sim_first_triple)
#   EqAngleTrans chain merges A and B, plus an optional Cyclic angle.


def _iso_triple(o: str, p1: str, p2: str) -> Tuple[str, str, str, str, str, str]:
    """Compute isosceles_base_angle result from Cong(o,p1,o,p2).

    Returns the 6-tuple args of the resulting EqAngle.
    """
    cong = canonical_cong(o, p1, o, p2)
    a, b, c, d = cong.args
    # isosceles_base_angle checks shared-endpoint patterns
    for apex, pa, pb in [
        (a, b, d) if a == c else (None, None, None),
        (b, a, d) if b == c else (None, None, None),
        (b, a, c) if b == d else (None, None, None),
        (a, b, c) if a == d else (None, None, None),
    ]:
        if apex is not None and pa != pb:
            return (apex, pa, pb, apex, pb, pa)
    # Fallback (shouldn't happen)
    return (o, p1, p2, o, p2, p1)


def _sim_triple(shared: str, m1: str, m2: str,
                other1: str, other2: str) -> Tuple[str, ...]:
    """Compute sim_tri_angle result from SimTri(shared,m1,m2,shared,other1,other2).

    Returns the 6-tuple args of the resulting EqAngle.
    """
    sim = canonical_sim_tri(shared, m1, m2, shared, other1, other2)
    A, B, C, D, E, F = sim.args
    return (B, A, C, E, D, F)


def _cyc_inscribed(a: str, b: str, c: str, d: str) -> Tuple[str, ...]:
    """Compute cyclic_inscribed_angle from Cyclic(a,b,c,d).

    Returns 6-tuple args of the resulting EqAngle.
    """
    cyc = canonical_cyclic(a, b, c, d)
    s0, s1, s2, s3 = cyc.args
    return (s1, s0, s2, s1, s3, s2)


def _midseg_shared(m_pt: str, a1: str, b1: str,
                   n_pt: str, a2: str, b2: str):
    """Find the shared vertex for midsegment of two midpoints.

    Returns (shared, other1, other2) or None.
    """
    p1 = canonical_midpoint(m_pt, a1, b1)
    p2 = canonical_midpoint(n_pt, a2, b2)
    _, ma, mb = p1.args
    _, na, nb = p2.args
    for shared, o1, o2 in [
        (ma, mb, nb) if ma == na else (None, None, None),
        (ma, mb, na) if ma == nb else (None, None, None),
        (mb, ma, nb) if mb == na else (None, None, None),
        (mb, ma, na) if mb == nb else (None, None, None),
    ]:
        if shared is not None:
            return shared, o1, o2
    return None


def gen_ultra_cc_midseg_cyclic() -> Tuple[List[Fact], Fact]:
    """Circumcenter + Midpoints + Cyclic → deep EqAngle (8 rules, 5 fam).

    ≈ 6.0–6.5 difficulty.  Chain uses circumcenter_cong_ab/bc, cong_trans,
    isosceles_base_angle, midsegment_sim_tri, sim_tri_angle,
    cyclic_inscribed_angle, eq_angle_trans.
    """
    pts = _pts(8)
    o = pts[0]
    # Sort triangle verts so canonical_circumcenter preserves order
    a, b, c = sorted(pts[1:4])
    d, m, n = pts[4], pts[5], pts[6]

    cc = canonical_circumcenter(o, a, b, c)
    _, sa, sb, sc = cc.args  # sorted triangle verts

    mid1 = canonical_midpoint(m, sa, sb)
    mid2 = canonical_midpoint(n, sa, sc)

    cyc = canonical_cyclic(sa, sb, sc, d)

    # Branch A: isosceles from cong_trans(cong_ab, cong_bc)
    iso = _iso_triple(o, sa, sc)  # from Cong(o,sa,o,sc) via cong_trans
    iso_second = iso[3:6]

    # Branch B: sim_tri_angle from midsegment
    info = _midseg_shared(m, sa, sb, n, sa, sc)
    if info is None:
        return ([canonical_midpoint("M", "A", "B")],
                canonical_cong("A", "B", "A", "B"))
    shared, other1, other2 = info
    sim = _sim_triple(shared, mid1.args[0], mid2.args[0], other1, other2)
    sim_first = sim[:3]
    sim_second = sim[3:6]

    # Bridge: connects isosceles second triple → sim first triple
    bridge = canonical_eq_angle(*iso_second, *sim_first)

    # Cyclic inscribed angle
    cyc_ea = _cyc_inscribed(sa, sb, sc, d)
    cyc_first = cyc_ea[:3]
    cyc_second = cyc_ea[3:6]

    # Goal: chain isosceles → bridge → sim → cyclic
    # If sim_second matches cyc_first, we can extend to cyclic
    if sim_second == cyc_first:
        goal = canonical_eq_angle(*iso[:3], *cyc_second)
    else:
        # Fallback: goal = iso_first + sim_second (no cyclic extension)
        goal = canonical_eq_angle(*iso[:3], *sim_second)

    assumptions = [cc, mid1, mid2, cyc, bridge]
    return assumptions, goal


def gen_ultra_tangent_cc_midseg() -> Tuple[List[Fact], Fact]:
    """Tangent + Circumcenter + Midpoints → deep EqAngle (8 rules, 6 fam).

    ≈ 6.5–7.0 difficulty.  Replaces the first two circumcenter cong
    steps with tangent_perp_radius → perp_bisector_cong.
    """
    pts = _pts(10)
    o, t = pts[0], pts[1]
    a, b = sorted(pts[2:4])  # tangent line endpoints
    # Choose cc triangle vertices such that the shared vertex with tangent
    # is one of a,b.  Use b to share with tangent's second endpoint.
    cc_verts = sorted([b, pts[4], pts[5]])
    cv0, cv1, cv2 = cc_verts
    m_pt, n_pt = pts[6], pts[7]

    tangent_fact = canonical_tangent(a, b, o, t)
    mid_t = canonical_midpoint(t, a, b)
    cc = canonical_circumcenter(o, cv0, cv1, cv2)
    _, sa, sb, sc = cc.args

    mid_m = canonical_midpoint(m_pt, sa, sb)
    mid_n = canonical_midpoint(n_pt, sa, sc)

    # perp_bisector_cong after tangent_perp_radius gives Cong(o,a,o,b)
    # circumcenter_cong_ab gives Cong(o,sa,o,sb)
    # We need cong_trans: Cong(o,a,o,b) + Cong(o,sb,...) → need shared pair
    # The cong chain ends at Cong(o,a,o,?) depending on which cc cong matches
    # With b as one of the cc triangle verts, b==sa or sb or sc.
    # Determine which cc vertex b maps to in sorted order
    b_idx = cc_verts.index(b)  # 0,1, or 2 in sorted order
    if b_idx == 0:
        other_vert = sc  # cong_trans: Cong(o,a,o,b)+Cong(o,b=sa,o,sb)→…
    elif b_idx == 1:
        other_vert = sc
    else:
        other_vert = sa

    iso = _iso_triple(o, a, other_vert)
    iso_second = iso[3:6]

    info = _midseg_shared(m_pt, sa, sb, n_pt, sa, sc)
    if info is None:
        return ([canonical_midpoint("M", "A", "B")],
                canonical_cong("A", "B", "A", "B"))
    shared, other1, other2 = info
    sim = _sim_triple(shared, mid_m.args[0], mid_n.args[0], other1, other2)
    sim_first = sim[:3]

    bridge = canonical_eq_angle(*iso_second, *sim_first)
    goal = canonical_eq_angle(*iso[:3], *sim[3:6])

    assumptions = [tangent_fact, mid_t, cc, mid_m, mid_n, bridge]
    return assumptions, goal


def gen_ultra_pole_polar_cc_midseg() -> Tuple[List[Fact], Fact]:
    """PolePolar + Circumcenter + Midpoints → deep EqAngle (8 rules, 7 fam).

    ≈ 7.0–7.4 difficulty (tier 6 from PolePolar).
    """
    pts = _pts(10)
    o, p = pts[0], pts[1]
    a, b = sorted(pts[2:4])
    cc_verts = sorted([b, pts[4], pts[5]])
    cv0, cv1, cv2 = cc_verts
    m_pt, n_pt = pts[6], pts[7]

    pp_fact = canonical_pole_polar(p, a, b, o)
    mid_p = canonical_midpoint(p, a, b)
    cc = canonical_circumcenter(o, cv0, cv1, cv2)
    _, sa, sb, sc = cc.args

    mid_m = canonical_midpoint(m_pt, sa, sb)
    mid_n = canonical_midpoint(n_pt, sa, sc)

    b_idx = cc_verts.index(b)
    other_vert = sc if b_idx <= 1 else sa

    iso = _iso_triple(o, a, other_vert)
    iso_second = iso[3:6]

    info = _midseg_shared(m_pt, sa, sb, n_pt, sa, sc)
    if info is None:
        return ([canonical_midpoint("M", "A", "B")],
                canonical_cong("A", "B", "A", "B"))
    shared, other1, other2 = info
    sim = _sim_triple(shared, mid_m.args[0], mid_n.args[0], other1, other2)
    sim_first = sim[:3]

    bridge = canonical_eq_angle(*iso_second, *sim_first)
    goal = canonical_eq_angle(*iso[:3], *sim[3:6])

    assumptions = [pp_fact, mid_p, cc, mid_m, mid_n, bridge]
    return assumptions, goal


def gen_ultra_double_cc_midseg() -> Tuple[List[Fact], Fact]:
    """Two Circumcenters + Midpoints → deep EqAngle (7 rules, 5 fam).

    ≈ 6.0–6.2 difficulty.
    """
    pts = _pts(10)
    o = pts[0]
    a, b, c = sorted(pts[1:4])
    c2, d2, e2 = sorted(pts[4:7])
    m_pt, n_pt = pts[7], pts[8]

    cc1 = canonical_circumcenter(o, a, b, c)
    _, s1a, s1b, s1c = cc1.args
    # Pick the vertex shared between cc1 and cc2 — use the LAST vertex of cc1
    shared_v = s1c
    cc2 = canonical_circumcenter(o, shared_v, d2, e2)
    _, s2a, s2b, s2c = cc2.args

    # cong chain: Cong(o,s1a,o,s1c) [via cc1 cong_ab+bc+trans]
    #           + Cong(o,s1c=shared,o,?) [via cc2 cong]
    #           → Cong(o,s1a,o,?)
    # Find which cc2 vertex pairs with shared_v in the sorted order
    if shared_v == s2a:
        chain_end = s2b  # cc2_cong_ab: Cong(o,s2a,o,s2b)
    elif shared_v == s2b:
        chain_end = s2c  # cc2_cong_bc: Cong(o,s2b,o,s2c)
    else:
        chain_end = s2b  # cc2_cong_bc flipped

    iso = _iso_triple(o, s1a, chain_end)
    iso_second = iso[3:6]

    mid_m = canonical_midpoint(m_pt, s1a, chain_end)
    mid_n = canonical_midpoint(n_pt, s1a, e2)

    info = _midseg_shared(m_pt, s1a, chain_end, n_pt, s1a, e2)
    if info is None:
        return ([canonical_midpoint("M", "A", "B")],
                canonical_cong("A", "B", "A", "B"))
    shared, other1, other2 = info
    sim = _sim_triple(shared, mid_m.args[0], mid_n.args[0], other1, other2)
    sim_first = sim[:3]

    bridge = canonical_eq_angle(*iso_second, *sim_first)
    goal = canonical_eq_angle(*iso[:3], *sim[3:6])

    assumptions = [cc1, cc2, mid_m, mid_n, bridge]
    return assumptions, goal


def gen_ultra_tangent_cc_midseg_cyclic() -> Tuple[List[Fact], Fact]:
    """Tangent + Circumcenter + Midpoints + Cyclic → deep EqAngle (9 rules, 6 fam).

    ≈ 6.5–7.0 difficulty.
    """
    pts = _pts(12)
    o, t = pts[0], pts[1]
    a, b = sorted(pts[2:4])
    cc_verts = sorted([b, pts[4], pts[5]])
    cv0, cv1, cv2 = cc_verts
    d_cyc = pts[6]
    m_pt, n_pt = pts[7], pts[8]

    tangent_fact = canonical_tangent(a, b, o, t)
    mid_t = canonical_midpoint(t, a, b)
    cc = canonical_circumcenter(o, cv0, cv1, cv2)
    _, sa, sb, sc = cc.args
    mid_m = canonical_midpoint(m_pt, sa, sb)
    mid_n = canonical_midpoint(n_pt, sa, sc)
    cyc = canonical_cyclic(sa, sb, sc, d_cyc)

    b_idx = cc_verts.index(b)
    other_vert = sc if b_idx <= 1 else sa

    iso = _iso_triple(o, a, other_vert)
    iso_second = iso[3:6]

    info = _midseg_shared(m_pt, sa, sb, n_pt, sa, sc)
    if info is None:
        return ([canonical_midpoint("M", "A", "B")],
                canonical_cong("A", "B", "A", "B"))
    shared, other1, other2 = info
    sim = _sim_triple(shared, mid_m.args[0], mid_n.args[0], other1, other2)
    sim_first = sim[:3]
    sim_second = sim[3:6]

    bridge = canonical_eq_angle(*iso_second, *sim_first)

    cyc_ea = _cyc_inscribed(sa, sb, sc, d_cyc)
    cyc_first = cyc_ea[:3]
    cyc_second = cyc_ea[3:6]

    if sim_second == cyc_first:
        goal = canonical_eq_angle(*iso[:3], *cyc_second)
    else:
        goal = canonical_eq_angle(*iso[:3], *sim_second)

    assumptions = [tangent_fact, mid_t, cc, mid_m, mid_n, cyc, bridge]
    return assumptions, goal


# ── Diversity generators (unique structural fingerprints) ────────────
# These generators target theorem shapes not covered by existing
# generators, ensuring each produces a DISTINCT semantic/structural
# fingerprint after proof minimization.

def gen_midseg_alt_angle() -> Tuple[List[Fact], Fact]:
    """Midsegment → parallel → alternate interior angle.

    Midpoint(M,A,B), Midpoint(N,A,C)
    → [midsegment_parallel]   → Parallel(M,N,B,C)         [substantive]
    → [midpoint_collinear]    → Collinear(A,M,B)           [trivial, auto]
    → [parallel_alternate_angle] → EqAngle(N,M,A, C,B,A)  [substantive]

    2 distinct substantive rules from different families.
    Families: MIDPOINT + LINE + ANGLE  (3 families).
    No relay variables — all points appear in both assumptions and goal.
    Fingerprint: MIDPOINT|MIDPOINT ⟹ ANGLE (midseg-alt)
    """
    pts = _pts(6)
    a, b, c, m, n = pts[:5]
    assumptions = [
        canonical_midpoint(m, a, b),
        canonical_midpoint(n, a, c),
    ]
    goal = canonical_eq_angle(n, m, a, c, b, a)
    return assumptions, goal


def gen_midseg_iso_angle_trans() -> Tuple[List[Fact], Fact]:
    """Midsegment parallel + isosceles → combined angle via transitivity.

    Midpoint(M,A,B), Midpoint(N,A,C), Cong(A,B,A,C)
    → [midsegment_parallel]      → Parallel(M,N,B,C)         [substantive]
    → [midpoint_collinear]        → Collinear(A,M,B)          [trivial]
    → [parallel_alternate_angle]  → EqAngle(N,M,A, C,B,A)    [substantive]
    → [isosceles_base_angle]      → EqAngle(A,B,C, A,C,B)    [substantive]
    → [eq_angle_trans]            → EqAngle(N,M,A, A,C,B)    [trivial]

    3 distinct substantive rules from 4 families.
    Families: MIDPOINT + LINE + METRIC + ANGLE  (4 families).
    No relay — all points {A,B,C,M,N} in both assumptions and goal.
    Fingerprint: MIDPOINT|MIDPOINT|METRIC ⟹ ANGLE (midseg-iso-trans)
    """
    pts = _pts(6)
    a, b, c, m, n = pts[:5]
    assumptions = [
        canonical_midpoint(m, a, b),
        canonical_midpoint(n, a, c),
        canonical_cong(a, b, a, c),
    ]
    # ∠NMA = ∠ACB (midsegment alt angle + isosceles base + transitivity)
    goal = canonical_eq_angle(n, m, a, a, c, b)
    return assumptions, goal


def gen_circumcenter_iso_angle() -> Tuple[List[Fact], Fact]:
    """Circumcenter + midpoint → perpendicular (circumcenter on perp bisector).

    Circumcenter(O,A,B,C) + Midpoint(M,B,C)
    → [circumcenter_cong_bc]   → Cong(O,B,O,C)             [trivial]
    → [cong_perp_bisector]     → Perp(O,M,B,C)             [substantive]

    Note: vertex A is a relay variable (not in goal). After relay
    elimination this simplifies to Cong(O,B,O,C)+Midpoint → 1 step.
    Retained for structural variety; relay gate handles filtering.
    Fingerprint: CIRCLE|MIDPOINT ⟹ LINE (circumcenter-perp)
    """
    pts = _pts(6)
    o, a, b, c, m = pts[:5]
    assumptions = [
        canonical_circumcenter(o, a, b, c),
        canonical_midpoint(m, b, c),
    ]
    goal = canonical_perp(o, m, b, c)
    return assumptions, goal


# All deep generators
DEEP_GENERATORS: List[Tuple[str, Any]] = [
    ("circumcenter_iso_perp",          gen_circumcenter_iso_perp_chain),
    ("cyclic_iso_midpoint_perp",       gen_cyclic_iso_midpoint_perp),
    ("double_midpoint_sim_angle",      gen_double_midpoint_sim_angle),
    ("circumcenter_midpoint_cong_ang", gen_circumcenter_midpoint_cong_angle),
    ("perp_bisector_cyclic_bridge",    gen_perp_bisector_cyclic_bridge),
    ("angle_bisect_cyclic_chain",      gen_angle_bisect_cyclic_chain),
    ("tangent_circumcenter_chain",     gen_tangent_circumcenter_chain),
    ("triple_midpoint_concurrent",     gen_triple_midpoint_concurrent_cong),
    ("pole_polar_midpoint_chain",      gen_pole_polar_midpoint_chain),
    ("radical_axis_circumcenter",      gen_radical_axis_circumcenter),
    # Extended deep generators for 5+ step proofs
    ("circumcenter_double_iso_angle",  gen_circumcenter_double_iso_angle),
    ("midseg_perp_iso_angle_chain",    gen_midseg_perp_iso_angle_chain),
    ("cyclic_circumcenter_cong_chain", gen_cyclic_circumcenter_cong_chain),
    ("double_circumcenter_cong_trans", gen_double_circumcenter_cong_trans),
    ("perp_bisector_iso_angle_trans",  gen_perp_bisector_iso_angle_trans),
    # ── New clean generators (no relay, no bridge assumptions) ───────
    ("double_perp_bisector_iso",       gen_double_perp_bisector_isosceles),
    ("midsegment_parallel",            gen_midsegment_parallel_chain),
    ("cyclic_inscribed_isosceles",     gen_cyclic_inscribed_isosceles),
    ("perp_bisector_cong_direct",      gen_perp_bisector_cong_direct),
    ("cong_perp_bisector_direct",      gen_cong_perp_bisector_direct),
    ("midseg_sim_tri_direct",          gen_midsegment_sim_tri_direct),
    # ── Diversity generators (no relay variables, 2+ substantive rules) ─
    ("midseg_alt_angle",               gen_midseg_alt_angle),
    ("midseg_iso_angle_trans",         gen_midseg_iso_angle_trans),
    ("circumcenter_iso_angle",         gen_circumcenter_iso_angle),
    # ── Ultra-deep generators (difficulty ≥ 6.0, 7–9 distinct rules) ─
    ("ultra_cc_midseg_cyclic",         gen_ultra_cc_midseg_cyclic),
    ("ultra_tangent_cc_midseg",        gen_ultra_tangent_cc_midseg),
    ("ultra_pole_polar_cc_midseg",     gen_ultra_pole_polar_cc_midseg),
    ("ultra_double_cc_midseg",         gen_ultra_double_cc_midseg),
    ("ultra_tangent_cc_midseg_cyclic", gen_ultra_tangent_cc_midseg_cyclic),
]


# ── Strategy 5: Constructive generators ─────────────────────────────
# Instead of randomly composing predicates and testing consistency,
# these generators start from a valid geometric construction and
# enumerate derivable goals.  This guarantees premise consistency
# by design, boosting Pólya pass rate from ~48% to ~90%+.

class ConstructiveTemplate:
    """A parameterized geometric construction with derivable goals."""

    def __init__(
        self,
        name: str,
        build_fn,  # Callable[[], Tuple[List[Fact], List[Fact]]]
        #           returns (assumptions, derivable_goals)
    ):
        self.name = name
        self.build_fn = build_fn

    def generate(self) -> Optional[Tuple[List[Fact], Fact]]:
        """Generate a conjecture by picking a random derivable goal."""
        try:
            assumptions, goals = self.build_fn()
            if not goals:
                return None
            goal = random.choice(goals)
            if goal in set(assumptions):
                return None
            return assumptions, goal
        except (ValueError, IndexError):
            return None


def _build_triangle_midpoint_web() -> Tuple[List[Fact], List[Fact]]:
    """Triangle + midpoints → multiple derivable relationships.

    Construction: Triangle ABC, M=mid(AB), N=mid(AC), P=mid(BC)
    Derivable: Parallel(MN,BC), Parallel(MP,AC), Parallel(NP,AB),
               SimTri(AMN,ABC), Cong(M,N, half of BC), etc.
    """
    pts = _pts(8)
    a, b, c, m, n, p = pts[:6]
    assumptions = [
        canonical_midpoint(m, a, b),
        canonical_midpoint(n, a, c),
        canonical_midpoint(p, b, c),
    ]
    goals = [
        canonical_parallel(m, n, b, c),
        canonical_parallel(m, p, a, c),
        canonical_parallel(n, p, a, b),
        canonical_sim_tri(a, m, n, a, b, c),
    ]
    return assumptions, goals


def _build_perp_bisector_web() -> Tuple[List[Fact], List[Fact]]:
    """Multiple perpendicular bisectors → cong + isosceles angles.

    Construction: M=mid(AB), N=mid(BC), O on perp_bisector(AB) ∩ perp_bisector(BC)
    Derivable: Cong(OA,OB), Cong(OB,OC), Cong(OA,OC),
               EqAngle(OAC,OCA), EqAngle(OAB,OBA), etc.
    """
    pts = _pts(8)
    a, b, c, o, m, n = pts[:6]
    assumptions = [
        canonical_midpoint(m, a, b),
        canonical_midpoint(n, b, c),
        canonical_perp(o, m, a, b),
        canonical_perp(o, n, b, c),
    ]
    goals = [
        canonical_cong(o, a, o, b),
        canonical_cong(o, b, o, c),
        canonical_cong(o, a, o, c),
        canonical_eq_angle(o, a, c, o, c, a),
        canonical_eq_angle(o, a, b, o, b, a),
        canonical_eq_angle(o, b, c, o, c, b),
    ]
    return assumptions, goals


def _build_circumcenter_web() -> Tuple[List[Fact], List[Fact]]:
    """Circumcenter → rich cong + perp + angle relationships.

    Construction: O = circumcenter(ABC), M=mid(AB), N=mid(AC)
    Derivable: Cong(OA,OB), Cong(OA,OC), Perp(OM,AB), EqAngle, etc.
    """
    pts = _pts(8)
    a, b, c = sorted(pts[:3])
    o, m, n = pts[3], pts[4], pts[5]
    assumptions = [
        canonical_circumcenter(o, a, b, c),
        canonical_midpoint(m, a, b),
        canonical_midpoint(n, a, c),
    ]
    goals = [
        canonical_cong(o, a, o, b),
        canonical_cong(o, a, o, c),
        canonical_cong(o, b, o, c),
        canonical_perp(o, m, a, b),
        canonical_perp(o, n, a, c),
        canonical_eq_angle(o, a, b, o, b, a),
        canonical_eq_angle(o, a, c, o, c, a),
        canonical_eq_angle(o, b, c, o, c, b),
    ]
    return assumptions, goals


def _build_cyclic_angle_web() -> Tuple[List[Fact], List[Fact]]:
    """Cyclic quadrilateral → inscribed angle equalities.

    Construction: Cyclic(ABCD)
    Derivable: EqAngle(BAC,BDC), EqAngle(ABD,ACD), etc.
    """
    pts = _pts(6)
    a, b, c, d = pts[:4]
    assumptions = [
        canonical_cyclic(a, b, c, d),
    ]
    goals = [
        canonical_eq_angle(b, a, c, b, d, c),
        canonical_eq_angle(a, b, d, a, c, d),
        canonical_eq_angle(b, a, d, b, c, d),
        canonical_eq_angle(a, b, c, a, d, c),
    ]
    return assumptions, goals


def _build_cc_midpoint_perp_web() -> Tuple[List[Fact], List[Fact]]:
    """Circumcenter + midpoints + perp bisectors → cross-domain.

    Construction: O=cc(ABC), M=mid(AB), N=mid(BC), P on perp_bisector(AC)
    Derivable: mixed cong, perp, angle relationships
    """
    pts = _pts(10)
    a, b, c = sorted(pts[:3])
    o, m, n, p, q = pts[3], pts[4], pts[5], pts[6], pts[7]
    assumptions = [
        canonical_circumcenter(o, a, b, c),
        canonical_midpoint(m, a, b),
        canonical_midpoint(n, b, c),
        canonical_midpoint(q, a, c),
    ]
    goals = [
        canonical_perp(o, m, a, b),
        canonical_perp(o, n, b, c),
        canonical_perp(o, q, a, c),
        canonical_cong(o, a, o, b),
        canonical_cong(o, b, o, c),
        canonical_eq_angle(o, a, c, o, c, a),
        canonical_eq_angle(o, a, b, o, b, a),
    ]
    return assumptions, goals


def _build_tangent_perp_web() -> Tuple[List[Fact], List[Fact]]:
    """Tangent + midpoint → perpendicular bisector chain.

    Construction: Tangent(A,B,O,T), M=mid(AB)
    Derivable: Perp(O,T,A,B) (tangent_perp_radius),
               then Cong chains if combined with cc
    """
    pts = _pts(8)
    a, b, o, t, m = pts[:5]
    assumptions = [
        canonical_tangent(a, b, o, t),
        canonical_midpoint(m, a, b),
    ]
    goals = [
        canonical_perp(o, t, a, b),
    ]
    return assumptions, goals


def _build_isosceles_cyclic_web() -> Tuple[List[Fact], List[Fact]]:
    """Cyclic + isosceles → angle transfers across circle.

    Construction: Cyclic(ABCD), Cong(BA,BC)
    Derivable: EqAngle via cyclic_inscribed + isosceles_base_angle + trans
    """
    pts = _pts(6)
    a, b, c, d = pts[:4]
    assumptions = [
        canonical_cyclic(a, b, c, d),
        canonical_cong(b, a, b, c),
    ]
    goals = [
        canonical_eq_angle(b, a, c, b, c, a),  # isosceles base
        canonical_eq_angle(b, a, c, b, d, c),  # cyclic inscribed
        canonical_eq_angle(b, c, a, b, d, c),  # transitive
        canonical_eq_angle(a, b, d, a, c, d),  # second inscribed
    ]
    return assumptions, goals


def _build_double_midpoint_parallel_web() -> Tuple[List[Fact], List[Fact]]:
    """Two pairs of midpoints from different sides → parallel chains.

    Construction: M=mid(AB), N=mid(AC), P=mid(BD), extra parallels
    """
    pts = _pts(10)
    a, b, c, d, m, n, p = pts[:7]
    assumptions = [
        canonical_midpoint(m, a, b),
        canonical_midpoint(n, a, c),
        canonical_midpoint(p, b, d),
        canonical_parallel(c, d, b, a),
    ]
    goals = [
        canonical_parallel(m, n, b, c),
        canonical_sim_tri(a, m, n, a, b, c),
    ]
    return assumptions, goals


CONSTRUCTIVE_TEMPLATES: List[ConstructiveTemplate] = [
    ConstructiveTemplate("cst:tri_midpoint_web",    _build_triangle_midpoint_web),
    ConstructiveTemplate("cst:perp_bisector_web",   _build_perp_bisector_web),
    ConstructiveTemplate("cst:circumcenter_web",    _build_circumcenter_web),
    ConstructiveTemplate("cst:cyclic_angle_web",    _build_cyclic_angle_web),
    ConstructiveTemplate("cst:cc_midpoint_perp",    _build_cc_midpoint_perp_web),
    ConstructiveTemplate("cst:tangent_perp_web",    _build_tangent_perp_web),
    ConstructiveTemplate("cst:iso_cyclic_web",      _build_isosceles_cyclic_web),
    ConstructiveTemplate("cst:dbl_midpoint_par",    _build_double_midpoint_parallel_web),
]


# ── Strategy 5b: Combinatorial Graph Walk Generator ─────────────────
# Instead of hand-writing every template, this strategy automatically
# discovers conjecture structures by random walks on the rule bridge
# graph.  Each walk samples 3-5 edges, creating multi-step chains that
# naturally span multiple concept families.

class _RuleBridgeGraph:
    """Directed graph over predicates connected by rule bridges.

    Nodes are predicate names; edges are rule bridges.
    A random walk produces an assumption→goal chain usable as a
    conjecture candidate.
    """

    def __init__(self) -> None:
        # adjacency: source_pred → [(target_pred, rule_name, full_input_list)]
        self._adj: Dict[str, List[Tuple[str, str, List[str]]]] = defaultdict(list)
        self._all_preds: Set[str] = set()
        for inputs, output, rule in _RULE_BRIDGES:
            for inp in inputs:
                self._adj[inp].append((output, rule, inputs))
                self._all_preds.add(inp)
            self._all_preds.add(output)

        # Good starting predicates: rich enough to have many outgoing edges
        self._starters = [
            p for p in self._all_preds
            if p in {
                "Midpoint", "Cyclic", "Cong", "Circumcenter",
                "Tangent", "AngleBisect", "Harmonic", "PolePolar",
            }
        ]

    def random_walk(
        self,
        min_edges: int = 3,
        max_edges: int = 5,
    ) -> Optional[List[Tuple[List[str], str, str]]]:
        """Random walk producing a chain of (inputs, output, rule) tuples."""
        if not self._starters:
            return None

        start = random.choice(self._starters)
        chain: List[Tuple[List[str], str, str]] = []
        visited_rules: Set[str] = set()
        available: Set[str] = {start}

        target = random.randint(min_edges, max_edges)

        for _ in range(target * 4):  # generous retries
            if len(chain) >= target:
                break

            # Bridges reachable from currently available predicates
            candidates: List[Tuple[List[str], str, str]] = []
            for pred in available:
                for tgt, rule, full_inp in self._adj.get(pred, []):
                    if rule in visited_rules:
                        continue
                    if all(inp in available for inp in full_inp):
                        candidates.append((full_inp, tgt, rule))

            if not candidates:
                # Expand reachability by adding a random assumption pred
                extras = list(self._all_preds - available)
                if not extras:
                    break
                available.add(random.choice(extras))
                continue

            chosen = random.choice(candidates)
            chain.append(chosen)
            visited_rules.add(chosen[2])
            available.add(chosen[1])

        if len(chain) < min_edges:
            return None
        return chain


_bridge_graph = _RuleBridgeGraph()


def _graph_walk_conjecture(
    knowledge_store: Optional[KnowledgeStore] = None,
) -> Optional[Tuple[List[Fact], Fact]]:
    """Generate a conjecture via random walk on the rule bridge graph.

    Discovers multi-step proof structures automatically without hand-
    writing templates.  Each walk naturally spans multiple concept
    families, producing chains of 3-5 distinct rules.
    """
    chain = _bridge_graph.random_walk(min_edges=3, max_edges=5)
    if chain is None:
        return None
    return _instantiate_chain(chain)


# ── Strategy 4: MCTS-guided conjecture search ──────────────────────

@dataclass
class MCTSNode:
    """Node in the MCTS conjecture tree."""
    pred_path: List[str]       # predicates chosen so far
    parent: Optional["MCTSNode"] = None
    children: Dict[str, "MCTSNode"] = field(default_factory=dict)
    visits: int = 0
    total_value: float = 0.0
    best_value: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.total_value / max(self.visits, 1)

    def ucb1(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        assert self.parent is not None
        return (self.mean_value +
                c * math.sqrt(math.log(self.parent.visits) / self.visits))


class MCTSConjectureSearch:
    """Monte Carlo Tree Search over the space of conjecture templates.

    Each tree path from root to leaf represents a sequence of predicate
    choices (for assumptions and goal).  The rollout value is determined
    by actually running beam search on the generated conjecture.
    """

    def __init__(
        self,
        max_depth: int = 6,
        ucb_c: float = 1.414,           # UCB1 exploration constant (sqrt(2))
        knowledge_store: Optional[KnowledgeStore] = None,
    ):
        self.root = MCTSNode(pred_path=[])  # root = empty predicate sequence
        self.max_depth = max_depth          # max predicates in a conjecture path
        self.ucb_c = ucb_c
        self.knowledge = knowledge_store or get_global_store()
        self.rules = default_rules()
        self.checker = MockLeanChecker()
        # All available predicates that can appear in conjecture paths
        from .genetic import _PRED_META
        self._all_preds = [m[0] for m in _PRED_META]
        self.discoveries: List[Dict] = []   # collected novel theorem records
        self.seen_fps: Set[str] = set()     # semantic fingerprints seen so far
        self.seen_sfps: Set[str] = set()    # structural fingerprints (anti-substitution)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCB1."""
        while node.children and len(node.pred_path) < self.max_depth:
            # Choose child with highest UCB1
            best_child = max(
                node.children.values(),
                key=lambda c: c.ucb1(self.ucb_c),
            )
            node = best_child
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand a leaf node by adding children."""
        if len(node.pred_path) >= self.max_depth:
            return node

        # Add a random predicate as a child
        pred = random.choice(self._all_preds)
        if pred not in node.children:
            child = MCTSNode(
                pred_path=node.pred_path + [pred],
                parent=node,
            )
            node.children[pred] = child
            return child

        # If already exists, just return it
        return node.children[pred]

    def _rollout(self, node: MCTSNode) -> float:
        """Simulate a full conjecture from this node and evaluate.

        The predicate path from the tree node defines the conjecture
        template.  We instantiate it with concrete points (sharing
        some points across assumptions to create meaningful connections),
        run beam search, and return the difficulty score as rollout value.
        Novel discoveries get a +2.0 bonus to guide exploration.
        """
        preds = list(node.pred_path)

        # Pad the predicate path to at least 3 (need >=2 assumptions + 1 goal)
        while len(preds) < 3:
            preds.append(random.choice(self._all_preds))

        # Convention: all predicates except the last are assumptions;
        # the last predicate is the goal to prove.
        assm_preds = preds[:-1]
        goal_pred = preds[-1]

        # Instantiate concrete facts from the predicate template.
        # pts: pool of available point names; shared: dict tracking
        # which roles map to which points (for controlled overlap).
        pts = _pts(min(14, 6 + len(assm_preds) * 2))
        shared: Dict[str, str] = {}
        assumptions = []
        for i, pred in enumerate(assm_preds):
            fact = _make_fact_for_pred(pred, pts, i * 2, shared)
            if fact is not None:
                assumptions.append(fact)

        if len(assumptions) < 2:
            return -1.0

        goal = _make_fact_for_pred(goal_pred, pts, 0, shared)
        if goal is None or goal in set(assumptions):
            return -1.0

        # ── Pólya pre-filter: quickly reject numerically-false conjectures ──
        from .polya import polya_test
        polya_result = polya_test(assumptions, goal, n_trials=10)
        if polya_result.falsified:
            return -1.0  # counter-example found

        # Run beam search with generous limits: beam_width=24 allows
        # exploring many branches, max_depth=15 supports long proofs.
        state = GeoState(facts=set(assumptions))
        cfg = SearchConfig(beam_width=24, max_depth=15, parallel_workers=0)
        result = beam_search(
            init_state=state,
            goal=Goal(goal),
            rules=self.rules,
            checker=self.checker,
            config=cfg,
            knowledge_store=self.knowledge,
        )

        if not result.success:
            return -0.5

        steps = list(result.final_state.history)
        if len(steps) < 3:
            return 0.0

        # Prune and compress for clean evaluation
        from .evolve import prune_proof, compress_proof, minimize_assumptions_proven
        assumptions, steps = prune_proof(assumptions, goal, steps)
        steps = compress_proof(steps)
        # Minimize: remove genuinely redundant assumptions
        assumptions, steps = minimize_assumptions_proven(
            assumptions, goal, steps,
            rules=self.rules, checker=self.checker,
            knowledge_store=self.knowledge,
        )

        if len(steps) < 3:
            return 0.0

        # Evaluate difficulty (after compression)
        diff = evaluate_difficulty(assumptions, goal, steps)
        value = diff.overall_score

        # Novelty bonus: unseen theorems get +2.0 value to encourage
        # the tree search to explore conjecture types that produce
        # genuinely new theorems rather than re-deriving known ones.
        from .semantic import semantic_theorem_fingerprint, structural_theorem_fingerprint
        fp = semantic_theorem_fingerprint(assumptions, goal)
        sfp = structural_theorem_fingerprint(assumptions, goal)
        if fp not in self.seen_fps and sfp not in self.seen_sfps:
            value += 2.0
            self.seen_fps.add(fp)
            self.seen_sfps.add(sfp)

            # Record discovery if sufficiently difficult and non-trivial.
            # Thresholds: score >= 4.5 and >= 5 proof steps ensure we
            # only keep theorems with genuine mathematical content.
            if diff.overall_score >= 4.5 and len(steps) >= 5:
                self.discoveries.append({
                    "assumptions": assumptions,
                    "goal": goal,
                    "steps": steps,
                    "difficulty": diff,
                    "pred_path": preds,
                    "fingerprint": fp,
                })

        return value

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Propagate the rollout value up the tree."""
        while node is not None:
            node.visits += 1
            node.total_value += value
            node.best_value = max(node.best_value, value)
            node = node.parent

    def search(
        self,
        n_iterations: int = 500,
        verbose: bool = False,
    ) -> List[Dict]:
        """Run MCTS for n_iterations and return discoveries."""
        for i in range(1, n_iterations + 1):
            # Select
            leaf = self._select(self.root)
            # Expand
            child = self._expand(leaf)
            # Rollout
            value = self._rollout(child)
            # Backpropagate
            self._backpropagate(child, value)

            if verbose and i % 50 == 0:
                print(f"    MCTS 迭代{i}: "
                      f"根节点值={self.root.mean_value:.2f} "
                      f"子节点={len(self.root.children)} "
                      f"发现={len(self.discoveries)}")

        return self.discoveries


# ── Unified heuristic conjecture generator ──────────────────────────

@dataclass
class HeuristicConfig:
    """Configuration for the heuristic conjecture generator."""
    # Strategy weights (how much compute to allocate)
    bridge_composition_weight: float = 0.15
    backward_chaining_weight: float = 0.12
    constructive_weight: float = 0.16
    graph_walk_weight: float = 0.15
    deep_generator_weight: float = 0.22
    mcts_weight: float = 0.20
    # Parameters
    total_attempts: int = 500
    min_difficulty: float = 4.5
    mcts_iterations: int = 300
    target_novel: int = 5
    # Adaptive deep-generator sampling
    adaptive_deep_sampling: bool = True
    deep_failure_cooldown: int = 10
    deep_failure_streak_trigger: int = 4
    # Parallel Pólya batch size (0 = sequential)
    polya_batch_size: int = 8


def generate_heuristic_conjectures(
    config: HeuristicConfig = HeuristicConfig(),
    knowledge_store: Optional[KnowledgeStore] = None,
    verbose: bool = True,
) -> List[Dict]:
    """Generate conjectures using multiple heuristic strategies.

    Allocates compute budget across five strategies:
    1. Bridge composition (chain known bridges)
    2. Backward chaining (work backwards from goal)
    3. Constructive templates (premise-consistent by design)
    4. Deep generators (hand-crafted high-value templates)
    5. MCTS (tree search over predicate space)
    """
    if knowledge_store is None:
        knowledge_store = get_global_store()

    rules = default_rules()
    checker = MockLeanChecker()
    discoveries: List[Dict] = []
    seen_fps: Set[str] = set()
    seen_sfps: Set[str] = set()                     # anti-substitution dedup
    pre_seen_fps: Set[str] = set()                  # pre-proof statement dedup
    t0 = time.time()

    if verbose:
        print("\n╔══════════════════════════════════════════════════════════╗")
        print("║  启发式猜想搜索 / Heuristic Conjecture Search           ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"  总尝试: {config.total_attempts}  目标: {config.target_novel}")
        print(f"  最低难度: {config.min_difficulty}")
        print()

    # Diagnostic counters
    _diag = {
        "polya_fail": 0,
        "polya_pass": 0,
        "premise_fail": 0,
        "search_fail": 0,
        "search_pass": 0,
        "too_short": 0,
        "too_easy": 0,
        "dup": 0,
    }
    _diag["deep_cooldown_skip"] = 0
    polya_controller = PolyaController(knowledge_store=knowledge_store)

    # Deep generator adaptive stats
    deep_stats: Dict[str, Dict[str, int]] = {
        name: {
            "attempts": 0,
            "accepted": 0,
            "fail_streak": 0,
            "cooldown_until": -1,
        }
        for name, _ in DEEP_GENERATORS
    }

    def _pick_deep_generator(iter_idx: int) -> Tuple[str, Any]:
        """Adaptive weighted choice with cooldown for deep generators."""
        if not config.adaptive_deep_sampling:
            return random.choice(DEEP_GENERATORS)

        weighted: List[Tuple[str, Any, float]] = []
        for gen_name, gen_fn in DEEP_GENERATORS:
            st = deep_stats[gen_name]
            if st["cooldown_until"] > iter_idx:
                continue

            attempts = st["attempts"]
            accepted = st["accepted"]
            fail_streak = st["fail_streak"]

            # Smoothed success estimate in [0,1]
            succ_rate = (accepted + 1.0) / (attempts + 2.0)
            # Penalize long fail streaks, but never to zero
            streak_penalty = 1.0 / (1.0 + 0.25 * fail_streak)
            # Encourage exploration of under-sampled generators
            exploration_bonus = 1.0 + 1.0 / (1.0 + attempts)

            weight = max(0.05, succ_rate * streak_penalty * exploration_bonus)
            weighted.append((gen_name, gen_fn, weight))

        if not weighted:
            # If all generators are cooling down, fallback to random
            _diag["deep_cooldown_skip"] += 1
            return random.choice(DEEP_GENERATORS)

        names = [x[0] for x in weighted]
        fns = [x[1] for x in weighted]
        ws = [x[2] for x in weighted]
        idx = random.choices(range(len(weighted)), weights=ws, k=1)[0]
        return names[idx], fns[idx]

    # ── Parallel Pólya batch pre-filter ──────────────────────────────
    # For strategies that produce many candidates (bridge, backward,
    # constructive, graph_walk), batch-generate candidates and run
    # Pólya pre-filter in parallel using threads.  NumPy releases the
    # GIL so thread-based parallelism is effective for numerical tests.
    import threading as _th
    _batch_lock = _th.Lock()

    def _batch_polya_and_try(
        candidates: List[Tuple[List[Fact], Fact, str]],
        batch_size: int = 0,
    ) -> int:
        """Batch Pólya pre-filter + sequential beam search.

        1. Run Pólya (two-stage) on all candidates in parallel threads.
        2. For survivors, call _try_conjecture() sequentially.

        Returns the number of accepted discoveries.
        """
        if batch_size <= 0:
            batch_size = config.polya_batch_size
        if batch_size <= 1 or len(candidates) <= 1:
            # Fallback: sequential
            accepted = 0
            for assm, gl, strat in candidates:
                if len(discoveries) >= config.target_novel:
                    break
                if _try_conjecture(assm, gl, strat):
                    accepted += 1
            return accepted

        from .polya import polya_test_two_stage, check_premise_consistency
        from .semantic import semantic_theorem_fingerprint as _sem_fp

        def _polya_one(item):
            """Run Pólya on a single candidate (thread-safe)."""
            assm, gl, strat = item
            # cold-generator skip
            if knowledge_store.is_generator_cold(strat, threshold=8):
                return None
            # pre-dedup
            pre_fp = _sem_fp(assm, gl)
            with _batch_lock:
                if pre_fp in pre_seen_fps:
                    return None
                pre_seen_fps.add(pre_fp)

            plan = polya_controller.make_plan(assm, gl, strat)
            polya_result = polya_test_two_stage(
                assm, gl,
                fast_trials=3,
                full_trials=plan.polya_trials,
            )
            if polya_result.falsified:
                with _batch_lock:
                    _diag["polya_fail"] += 1
                return None
            if (polya_result.n_valid > 0
                    and polya_result.confidence < plan.polya_min_confidence):
                with _batch_lock:
                    _diag["polya_fail"] += 1
                return None
            with _batch_lock:
                _diag["polya_pass"] += 1

            # Premise consistency
            if len(assm) >= 4 and plan.premise_probe_trials > 0:
                if not check_premise_consistency(
                    assm, n_trials=plan.premise_probe_trials,
                ):
                    with _batch_lock:
                        _diag["premise_fail"] += 1
                    return None

            return (assm, gl, strat, polya_result.confidence)

        # Phase 1: parallel Pólya
        survivors = []
        n_workers = min(batch_size, max(1, os.cpu_count() or 2))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_polya_one, c) for c in candidates]
            for f in as_completed(futs):
                try:
                    r = f.result()
                    if r is not None:
                        survivors.append(r)
                except Exception:
                    pass

        # Phase 2: sequential beam search on survivors
        accepted = 0
        for assm, gl, strat, _ in survivors:
            if len(discoveries) >= config.target_novel:
                break
            if _try_conjecture(assm, gl, strat):
                accepted += 1
        return accepted

    def _try_conjecture(assumptions: List[Fact], goal: Fact,
                         strategy: str) -> bool:
        """Try to prove and evaluate a conjecture.

        Before the expensive beam search, a Pólya plausible-reasoning
        pre-filter numerically tests the conjecture on random instances.
        Falsified conjectures are immediately rejected.
        """
        # Cross-session cold-generator skip: avoid strategies that have
        # been consistently unproductive (>= 8 consecutive failures).
        if knowledge_store.is_generator_cold(strategy, threshold=8):
            _diag.setdefault("cold_skip", 0)
            _diag["cold_skip"] += 1
            return False

        # Pre-proof dedup on theorem statement (cheap and early)
        from .semantic import semantic_theorem_fingerprint, structural_theorem_fingerprint
        pre_fp = semantic_theorem_fingerprint(assumptions, goal)
        if pre_fp in pre_seen_fps:
            _diag["dup"] += 1
            polya_controller.note_failure("dup_pre")
            return False
        pre_seen_fps.add(pre_fp)

        # Pólya Step 1+2: understand and plan before execution.
        plan = polya_controller.make_plan(assumptions, goal, strategy)

        # ── Two-stage Pólya pre-filter ──
        # Stage 1 (3 trials): fast rejection catches ~50% of false conjectures
        # Stage 2 (remaining): full confirmation only if stage 1 passes
        from .polya import polya_test_two_stage, check_premise_consistency
        polya_result = polya_test_two_stage(
            assumptions, goal,
            fast_trials=3,
            full_trials=plan.polya_trials,
        )
        if polya_result.falsified:
            _diag["polya_fail"] += 1
            polya_controller.note_failure("polya_falsified")
            return False  # counter-example found; skip beam search
        if (polya_result.n_valid > 0
                and polya_result.confidence < plan.polya_min_confidence):
            _diag["polya_fail"] += 1
            polya_controller.note_failure("polya_low_conf")
            return False  # too few passing trials; unlikely to be true
        _diag["polya_pass"] += 1

        # Early premise consistency gate (cheap) before expensive search
        # For complex premise sets, run a light consistency probe.
        # Keep this very cheap; strict checks remain in later quality gates.
        if len(assumptions) >= 4 and plan.premise_probe_trials > 0:
            if not check_premise_consistency(assumptions, n_trials=plan.premise_probe_trials):
                _diag["premise_fail"] += 1
                polya_controller.note_failure("premise_inconsistent")
                return False

        state = GeoState(facts=set(assumptions))
        workers = max(1, (os.cpu_count() or 2) // 2)

        # Adaptive re-plan: refine search params using observed Pólya confidence
        adaptive_plan = polya_controller.make_adaptive_plan(
            assumptions, goal, strategy, polya_result.confidence,
        )

        # Stage-1 fast search: cheap probe to reject hopeless candidates
        fast_cfg = SearchConfig(
            beam_width=adaptive_plan.fast_beam_width,
            max_depth=adaptive_plan.fast_max_depth,
            parallel_workers=workers,
        )
        result = beam_search(
            init_state=state,
            goal=Goal(goal),
            rules=rules,
            checker=checker,
            config=fast_cfg,
            knowledge_store=knowledge_store,
        )

        # Stage-2 deep search: only escalate promising candidates
        if not result.success:
            if polya_controller.should_escalate(polya_result.confidence, strategy):
                deep_cfg = SearchConfig(
                    beam_width=adaptive_plan.deep_beam_width,
                    max_depth=adaptive_plan.deep_max_depth,
                    parallel_workers=workers,
                )
                result = beam_search(
                    init_state=state,
                    goal=Goal(goal),
                    rules=rules,
                    checker=checker,
                    config=deep_cfg,
                    knowledge_store=knowledge_store,
                )

        if not result.success:
            _diag["search_fail"] += 1
            polya_controller.note_failure("search_fail")
            knowledge_store.record_generator_outcome(strategy, success=False)
            return False
        _diag["search_pass"] += 1

        steps = list(result.final_state.history)

        # Prune unused assumptions and redundant steps
        from .evolve import (prune_proof, compress_proof,
                             minimize_assumptions_proven,
                             _eliminate_relay_variables)
        assumptions, steps = prune_proof(assumptions, goal, steps)
        # Compress trivial symmetry steps for conciseness
        steps = compress_proof(steps)
        # Minimize: remove genuinely redundant assumptions
        assumptions, steps = minimize_assumptions_proven(
            assumptions, goal, steps,
            rules=rules, checker=checker,
            knowledge_store=knowledge_store,
            max_redundancy_checks=4,
        )
        # Eliminate relay variables that artificially inflate depth
        assumptions, steps, _relay_simplified = _eliminate_relay_variables(
            assumptions, goal, steps,
            rules=rules, checker=checker,
            knowledge_store=knowledge_store,
        )

        if len(steps) < 2:
            _diag["too_short"] += 1
            polya_controller.note_failure("too_short")
            return False

        diff = evaluate_difficulty(assumptions, goal, steps)
        if diff.overall_score < config.min_difficulty:
            _diag["too_easy"] += 1
            polya_controller.note_failure("too_easy")
            if verbose:
                print(f"    ⊘ 难度不足 ({strategy}): {diff.overall_score:.1f}/10"
                      f" ({diff.n_substantive_rules}种实质规则, {len(steps)}步)")
            return False

        fp = semantic_theorem_fingerprint(assumptions, goal)
        if fp in seen_fps:
            _diag["dup"] += 1
            polya_controller.note_failure("dup_semantic")
            return False
        seen_fps.add(fp)

        # Anti-substitution: reject simple predicate-swap variants
        sfp = structural_theorem_fingerprint(assumptions, goal)
        if sfp in seen_sfps:
            _diag["dup"] += 1
            polya_controller.note_failure("dup_structural")
            return False
        seen_sfps.add(sfp)

        polya_controller.note_success(strategy)
        knowledge_store.record_generator_outcome(strategy, success=True)

        discoveries.append({
            "assumptions": assumptions,
            "goal": goal,
            "steps": steps,
            "difficulty": diff,
            "strategy": strategy,
            "fingerprint": fp,
            "polya_plan": {
                "trials": plan.polya_trials,
                "min_confidence": plan.polya_min_confidence,
                "fast": [plan.fast_beam_width, plan.fast_max_depth],
                "deep": [plan.deep_beam_width, plan.deep_max_depth],
            },
        })

        if verbose:
            print(f"  🌟 发现#{len(discoveries)} ({strategy}): "
                  f"难度 {diff.overall_score:.1f}/10"
                  f" ({diff.label_zh})"
                  f"  {diff.n_substantive_rules}种规则"
                  f"  {diff.n_concept_families}族")

        return True

    # Budget allocation: distribute total attempts across strategies
    # based on configured weights.  E.g. 500 attempts × 0.15 bridge = 75.
    n_bridge = int(config.total_attempts * config.bridge_composition_weight)
    n_backward = int(config.total_attempts * config.backward_chaining_weight)
    n_constructive = int(config.total_attempts * config.constructive_weight)
    n_graph_walk = int(config.total_attempts * config.graph_walk_weight)
    n_deep = int(config.total_attempts * config.deep_generator_weight)

    # ── Strategy 1: Bridge Composition ──
    if verbose:
        print("  ── 桥式组合 (Bridge Composition) ──")
    if config.polya_batch_size > 1:
        # Batch mode: pre-generate candidates, parallel Pólya filter
        bridge_batch: List[Tuple[List[Fact], Fact, str]] = []
        for _ in range(n_bridge):
            if len(discoveries) >= config.target_novel:
                break
            result = _compose_bridges(
                target_families=4, target_depth=5,
                knowledge_store=knowledge_store,
            )
            if result:
                bridge_batch.append((result[0], result[1], "bridge_composition"))
            if len(bridge_batch) >= config.polya_batch_size:
                _batch_polya_and_try(bridge_batch)
                bridge_batch.clear()
        if bridge_batch:
            _batch_polya_and_try(bridge_batch)
    else:
        for i in range(n_bridge):
            if len(discoveries) >= config.target_novel:
                break
            result = _compose_bridges(
                target_families=4, target_depth=5,
                knowledge_store=knowledge_store,
            )
            if result:
                _try_conjecture(result[0], result[1], "bridge_composition")

    # ── Strategy 2: Backward Chaining ──
    if verbose:
        print(f"  ── 逆向链接 (Backward Chaining) ── 已发现: {len(discoveries)}")
    # Target predicates for backward chaining: higher-tier predicates
    # that tend to produce deeper, more interesting proof chains.
    # Under-explored predicates are injected for coverage balance.
    goal_preds = ["EqAngle", "Cong", "Perpendicular", "EqArea",
                  "EqRatio", "Concurrent", "EqCrossRatio"]
    try:
        under = knowledge_store.under_explored_predicates(top_n=3)
        for pred in under:
            if pred not in goal_preds:
                goal_preds.append(pred)
    except Exception:
        pass
    if config.polya_batch_size > 1:
        backward_batch: List[Tuple[List[Fact], Fact, str]] = []
        for i in range(n_backward):
            if len(discoveries) >= config.target_novel:
                break
            gp = random.choice(goal_preds)
            result = backward_chain_conjecture(
                goal_pred=gp, depth=4, min_families=3,
                knowledge_store=knowledge_store,
            )
            if result:
                backward_batch.append((result[0], result[1], "backward_chaining"))
            if len(backward_batch) >= config.polya_batch_size:
                _batch_polya_and_try(backward_batch)
                backward_batch.clear()
        if backward_batch:
            _batch_polya_and_try(backward_batch)
    else:
        for i in range(n_backward):
            if len(discoveries) >= config.target_novel:
                break
            gp = random.choice(goal_preds)
            result = backward_chain_conjecture(
                goal_pred=gp, depth=4, min_families=3,
                knowledge_store=knowledge_store,
            )
            if result:
                _try_conjecture(result[0], result[1], "backward_chaining")

    # ── Strategy 2.5: Constructive Templates ──
    if verbose:
        print(f"  ── 构造性模板 (Constructive Templates) ── 已发现: {len(discoveries)}")
    if config.polya_batch_size > 1:
        cst_batch: List[Tuple[List[Fact], Fact, str]] = []
        for i in range(n_constructive):
            if len(discoveries) >= config.target_novel:
                break
            tpl = random.choice(CONSTRUCTIVE_TEMPLATES)
            result = tpl.generate()
            if result:
                cst_batch.append((result[0], result[1], f"constructive:{tpl.name}"))
            if len(cst_batch) >= config.polya_batch_size:
                _batch_polya_and_try(cst_batch)
                cst_batch.clear()
        if cst_batch:
            _batch_polya_and_try(cst_batch)
    else:
        for i in range(n_constructive):
            if len(discoveries) >= config.target_novel:
                break
            tpl = random.choice(CONSTRUCTIVE_TEMPLATES)
            result = tpl.generate()
            if result:
                _try_conjecture(result[0], result[1], f"constructive:{tpl.name}")

    # ── Strategy 2.75: Graph Walk Generator ──
    if verbose:
        print(f"  ── 图游走生成 (Graph Walk) ── 已发现: {len(discoveries)}")
    if config.polya_batch_size > 1:
        gw_batch: List[Tuple[List[Fact], Fact, str]] = []
        for i in range(n_graph_walk):
            if len(discoveries) >= config.target_novel:
                break
            result = _graph_walk_conjecture(knowledge_store=knowledge_store)
            if result:
                gw_batch.append((result[0], result[1], "graph_walk"))
            if len(gw_batch) >= config.polya_batch_size:
                _batch_polya_and_try(gw_batch)
                gw_batch.clear()
        if gw_batch:
            _batch_polya_and_try(gw_batch)
    else:
        for i in range(n_graph_walk):
            if len(discoveries) >= config.target_novel:
                break
            result = _graph_walk_conjecture(knowledge_store=knowledge_store)
            if result:
                _try_conjecture(result[0], result[1], "graph_walk")

    # ── Strategy 3: Deep Generators ──
    if verbose:
        print(f"  ── 深度生成器 (Deep Generators) ── 已发现: {len(discoveries)}")
    for i in range(n_deep):
        if len(discoveries) >= config.target_novel:
            break
        gen_name, gen_fn = _pick_deep_generator(i)
        try:
            assumptions, goal = gen_fn()
            deep_stats[gen_name]["attempts"] += 1
            accepted = _try_conjecture(assumptions, goal, f"deep:{gen_name}")
            if accepted:
                deep_stats[gen_name]["accepted"] += 1
                deep_stats[gen_name]["fail_streak"] = 0
            else:
                deep_stats[gen_name]["fail_streak"] += 1
                if (
                    config.adaptive_deep_sampling
                    and deep_stats[gen_name]["fail_streak"] >= config.deep_failure_streak_trigger
                ):
                    deep_stats[gen_name]["cooldown_until"] = i + config.deep_failure_cooldown
        except (ValueError, IndexError):
            continue

    # Strategy 4 runs only if earlier strategies didn't find enough.
    # Share seen fingerprints with MCTS to avoid re-discovery,
    # then double-check results against our structural fingerprints.
    if len(discoveries) < config.target_novel:
        if verbose:
            print(f"  ── MCTS搜索 ── 已发现: {len(discoveries)}")
        mcts = MCTSConjectureSearch(knowledge_store=knowledge_store)
        mcts.seen_fps = set(seen_fps)
        mcts.seen_sfps = set(seen_sfps)
        mcts_results = mcts.search(
            n_iterations=config.mcts_iterations,
            verbose=verbose,
        )
        for d in mcts_results:
            if d.get("difficulty") and d["difficulty"].overall_score >= config.min_difficulty:
                fp = d.get("fingerprint", "")
                if fp and fp not in seen_fps:
                    # Also check structural fingerprint
                    from .semantic import structural_theorem_fingerprint as _sfp_fn
                    _assm = d.get("assumptions", [])
                    _goal = d.get("goal")
                    if _goal:
                        _s = _sfp_fn(_assm, _goal)
                        if _s in seen_sfps:
                            continue
                        seen_sfps.add(_s)
                    seen_fps.add(fp)
                    d["strategy"] = "mcts"
                    discoveries.append(d)

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  启发式搜索完成: {len(discoveries)} 个发现 ({elapsed:.1f}s)")
        print(f"  诊断: Pólya通过={_diag['polya_pass']} 证明成功={_diag['search_pass']}"
              f" 太短={_diag['too_short']} 太易={_diag['too_easy']}"
              f" Pólya拒绝={_diag['polya_fail']} 前提拒绝={_diag['premise_fail']}"
              f" 搜索失败={_diag['search_fail']}"
              f" 重复={_diag['dup']}")

        # Funnel dashboard
        polya_total = _diag['polya_pass'] + _diag['polya_fail']
        polya_pass_rate = (_diag['polya_pass'] / polya_total) if polya_total else 0.0
        search_success_rate = (
            _diag['search_pass'] / max(_diag['polya_pass'], 1)
        )
        accept_given_search = (
            len(discoveries) / max(_diag['search_pass'], 1)
        )
        throughput = len(discoveries) / max(elapsed, 1e-9)

        print("  漏斗仪表盘:")
        print(f"    Pólya通过率: {polya_pass_rate*100:.1f}%"
              f"  ({_diag['polya_pass']}/{max(polya_total,1)})")
        print(f"    搜索成功率: {search_success_rate*100:.1f}%"
              f"  ({_diag['search_pass']}/{max(_diag['polya_pass'],1)})")
        print(f"    搜索后保留率: {accept_given_search*100:.1f}%"
              f"  ({len(discoveries)}/{max(_diag['search_pass'],1)})")
        print(f"    产出吞吐: {throughput:.3f} 定理/秒")
        if config.adaptive_deep_sampling:
            print(f"    冷却回退次数: {_diag['deep_cooldown_skip']}")

            ranked = sorted(
                deep_stats.items(),
                key=lambda kv: (kv[1]['accepted'],
                                (kv[1]['accepted'] + 1.0) / (kv[1]['attempts'] + 2.0)),
                reverse=True,
            )
            top = ranked[:5]
            if top:
                print("  深度生成器Top5:")
                for name, st in top:
                    rate = st['accepted'] / max(st['attempts'], 1)
                    print(f"    {name}: 命中 {st['accepted']}/{st['attempts']}"
                          f" ({rate*100:.1f}%), 连败={st['fail_streak']}")
            print(f"  {polya_controller.summary()}")

    return discoveries
