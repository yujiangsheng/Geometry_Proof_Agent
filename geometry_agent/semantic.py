"""semantic.py – Semantic-level knowledge: isomorphism, NL, Lean4, visualisation.

Four core problems solved here:

1. **Semantic fingerprinting (isomorphism detection)**
   A proof about ``Parallel(A,B,C,D), Parallel(C,D,E,F) ⊢ Parallel(A,B,E,F)``
   is *the same theorem* as ``Parallel(X,Y,U,V), Parallel(U,V,P,Q) ⊢ Parallel(X,Y,P,Q)``
   — only the point names differ.  We detect this by computing a
   *canonical relabeling* that maps every concrete name to an abstract
   token (``P0, P1, …``) in order of first appearance, then fingerprint
   the relabeled form.  Simple substitution therefore never creates a
   "new" theorem in the knowledge store.

   **Symmetry-variant canonicalization** (v0.14.0): Fingerprinting now
   enumerates all logically-equivalent argument orderings for each
   predicate (e.g. ``Midpoint(M,A,B) ↔ Midpoint(M,B,A)``) combined
   with all permutations of the assumption list.  The lexicographically
   minimal canonical-relabeled string is chosen, guaranteeing true
   isomorphism invariance regardless of original point names.

2. **Natural-language description**
   Every ``Fact`` and proof trace can be rendered into readable Chinese /
   English text so that problems and conclusions are expressed as
   natural-language statements.

3. **Lean4 theorem statement generation**
   Given assumptions + goal, produce a complete Lean4 ``theorem``
   declaration (with ``sorry`` proof) that precisely formalises the
   problem.

4. **Geometry visualisation**
   Given a set of ``Fact``s, assign 2-D coordinates to the points
   and draw lines / markers with matplotlib, producing a PNG or SVG
   that makes the geometry intuitive.

Author:  Jiangsheng Yu
License: MIT
"""

from __future__ import annotations

import hashlib
import math
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .dsl import Fact, Step

# ═══════════════════════════════════════════════════════════════════════
# 1.  Semantic fingerprinting  (isomorphism-invariant)
# ═══════════════════════════════════════════════════════════════════════


def _canonical_relabel(
    facts: Iterable[Fact],
) -> Tuple[List[Fact], Dict[str, str]]:
    """Re-label all point names to ``P0, P1, …`` based on *structural*
    position, not alphabetical order of original names.

    We build equivalence classes of point occurrences (fact_idx, arg_idx)
    that share the same original name, sort these classes by their first
    occurrence position, then assign canonical names in that order.

    This guarantees that two isomorphic fact-sequences (differing only
    by point names) produce *identical* relabeled output.

    Returns (relabeled_facts, mapping  original→canonical).
    """
    facts_list = list(facts)
    # Step 1: collect all (fact_idx, arg_idx) positions per original name
    positions: Dict[str, List[Tuple[int, int]]] = {}
    for fi, f in enumerate(facts_list):
        for ai, arg in enumerate(f.args):
            positions.setdefault(arg, []).append((fi, ai))

    # Step 2: sort equivalence classes by their minimum position
    # This ensures the *structural role* (not the original name) determines
    # the canonical label.
    classes = sorted(positions.values(), key=lambda ps: ps[0])

    # Step 3: assign canonical names
    mapping: Dict[str, str] = {}
    for idx, pos_list in enumerate(classes):
        # All positions in this class share the same original name —
        # pick any to recover it
        fi, ai = pos_list[0]
        orig_name = facts_list[fi].args[ai]
        mapping[orig_name] = f"P{idx}"

    # Step 4: relabel
    relabeled: List[Fact] = []
    for f in facts_list:
        new_args = tuple(mapping[a] for a in f.args)
        relabeled.append(Fact(predicate=f.predicate, args=new_args))
    return relabeled, mapping


def compute_isomorphism_map(
    source_facts: Sequence[Fact],
    target_facts: Sequence[Fact],
) -> Optional[Dict[str, str]]:
    """Compute point-name bijection σ such that σ(source) = target.

    Given two isomorphic fact-sequences (same structure, different point
    names), returns a dict mapping source point names → target point
    names.  Returns ``None`` if no isomorphism exists.

    Uses constraint-propagation with backtracking, efficient for small
    geometry problems (< 30 points).

    Example::

        >>> src = [Fact("Parallel", ("A","B","C","D"))]
        >>> tgt = [Fact("Parallel", ("X","Y","U","V"))]
        >>> compute_isomorphism_map(src, tgt)
        {'A': 'X', 'B': 'Y', 'C': 'U', 'D': 'V'}
    """
    from collections import defaultdict as _ddict

    # Group by (predicate, arity)
    src_by_sig: Dict[tuple, List[Fact]] = _ddict(list)
    tgt_by_sig: Dict[tuple, List[Fact]] = _ddict(list)
    for f in source_facts:
        src_by_sig[(f.predicate, len(f.args))].append(f)
    for f in target_facts:
        tgt_by_sig[(f.predicate, len(f.args))].append(f)

    if set(src_by_sig.keys()) != set(tgt_by_sig.keys()):
        return None
    for sig in src_by_sig:
        if len(src_by_sig[sig]) != len(tgt_by_sig[sig]):
            return None

    # Flatten source facts deterministically
    src_list: List[Fact] = []
    for sig in sorted(src_by_sig.keys()):
        src_list.extend(src_by_sig[sig])

    def _consistent(sf: Fact, tf: Fact, m: Dict[str, str]) -> bool:
        """Check if mapping args of sf→tf is consistent with m."""
        rev: Dict[str, str] = {v: k for k, v in m.items()}
        for sa, ta in zip(sf.args, tf.args):
            if sa in m:
                if m[sa] != ta:
                    return False
            else:
                if ta in rev:      # ta already used by different source
                    return False
        return True

    def _extend(sf: Fact, tf: Fact, m: Dict[str, str]) -> Dict[str, str]:
        new_m = dict(m)
        for sa, ta in zip(sf.args, tf.args):
            new_m[sa] = ta
        return new_m

    def _solve(idx: int, m: Dict[str, str], used: Set[int]) -> Optional[Dict[str, str]]:
        if idx == len(src_list):
            return m
        sf = src_list[idx]
        sig = (sf.predicate, len(sf.args))
        for i, tf in enumerate(tgt_by_sig[sig]):
            tid = id(tf)
            if tid in used:
                continue
            if _consistent(sf, tf, m):
                result = _solve(idx + 1, _extend(sf, tf, m), used | {tid})
                if result is not None:
                    return result
        return None

    return _solve(0, {}, set())


def remap_fact(fact: Fact, mapping: Dict[str, str]) -> Fact:
    """Apply a point-name mapping to a fact."""
    return Fact(
        predicate=fact.predicate,
        args=tuple(mapping.get(a, a) for a in fact.args),
    )


def remap_step(step: "Step", mapping: Dict[str, str]) -> "Step":
    """Apply a point-name mapping to a derivation step."""
    return Step(
        rule_name=step.rule_name,
        premise_facts=tuple(remap_fact(f, mapping) for f in step.premise_facts),
        conclusion_fact=remap_fact(step.conclusion_fact, mapping),
    )


def semantic_fact_fingerprint(facts: Iterable[Fact]) -> str:
    """Isomorphism-invariant fingerprint for a *set* of facts.

    Two fact-sets that differ only by point renaming produce the same
    fingerprint.  Facts are relabeled structurally, then sorted.
    """
    facts_list = list(facts)
    relabeled, _ = _canonical_relabel(facts_list)
    # Sort AFTER relabeling so the order is canonical
    relabeled.sort(key=lambda f: (f.predicate, f.args))
    canon = "|".join(f"{f.predicate}({','.join(f.args)})" for f in relabeled)
    return hashlib.sha256(canon.encode()).hexdigest()[:20]


def _normalize_symmetry(f: Fact) -> Fact:
    """Normalize a fact's arguments using predicate-specific symmetries.

    This ensures that two logically equivalent representations of the
    same predicate (e.g. ``Perpendicular(A,B,C,D)`` vs
    ``Perpendicular(B,A,D,C)``) produce the same canonical form.

    Symmetry rules:
      - Perpendicular / Parallel: sort within each pair, then sort pairs
      - Cong: sort within each pair, then sort pairs
      - EqAngle: sort the two angle-triples (keeping vertex as 2nd)
      - Midpoint / IsMidpoint: sort the two endpoints
      - Circumcenter: sort the three triangle vertices
      - Collinear: sort all args
      - Cyclic: sort all args
      - Between: sort the two outer points (middle stays)
      - SimTri / CongTri: sort the two triples
      - EqArea: sort the two triples
      - Concurrent: sort the three line-pairs
    """
    p = f.predicate
    args = list(f.args)

    if p in ("Perpendicular", "Parallel"):
        if len(args) == 4:
            pair1 = tuple(sorted(args[0:2]))
            pair2 = tuple(sorted(args[2:4]))
            if pair1 > pair2:
                pair1, pair2 = pair2, pair1
            return Fact(p, pair1 + pair2)

    elif p == "Cong":
        if len(args) == 4:
            pair1 = tuple(sorted(args[0:2]))
            pair2 = tuple(sorted(args[2:4]))
            if pair1 > pair2:
                pair1, pair2 = pair2, pair1
            return Fact(p, pair1 + pair2)

    elif p == "EqAngle":
        if len(args) == 6:
            t1 = tuple(args[0:3])
            t2 = tuple(args[3:6])
            if t1 > t2:
                t1, t2 = t2, t1
            return Fact(p, t1 + t2)

    elif p in ("Midpoint", "IsMidpoint"):
        if len(args) == 3:
            return Fact(p, (args[0],) + tuple(sorted(args[1:3])))

    elif p == "Circumcenter":
        if len(args) == 4:
            return Fact(p, (args[0],) + tuple(sorted(args[1:4])))

    elif p in ("Collinear", "Cyclic"):
        return Fact(p, tuple(sorted(args)))

    elif p == "Between":
        if len(args) == 3:
            outer = sorted([args[0], args[2]])
            return Fact(p, (outer[0], args[1], outer[1]))

    elif p in ("SimTri", "CongTri", "EqArea"):
        if len(args) == 6:
            t1 = tuple(args[0:3])
            t2 = tuple(args[3:6])
            if t1 > t2:
                t1, t2 = t2, t1
            return Fact(p, t1 + t2)

    elif p == "Concurrent":
        if len(args) == 6:
            pairs = [
                tuple(sorted(args[0:2])),
                tuple(sorted(args[2:4])),
                tuple(sorted(args[4:6])),
            ]
            pairs.sort()
            return Fact(p, pairs[0] + pairs[1] + pairs[2])

    return f


def _symmetry_variants(f: Fact) -> List[Fact]:
    """Return all logically equivalent forms of *f* under its symmetry group.

    For example ``Midpoint(M, A, B) ≡ Midpoint(M, B, A)`` — both forms
    are returned so that canonical-relabeling can explore both orderings
    and choose the lexicographic minimum.
    """
    import itertools as _it

    p = f.predicate
    a = f.args

    if p in ("Perpendicular", "Parallel", "Cong"):
        # swap within each pair, swap the two pairs → up to 8 variants
        if len(a) == 4:
            out: set = set()
            for p1, p2 in [
                ((a[0], a[1]), (a[2], a[3])),
                ((a[2], a[3]), (a[0], a[1])),
            ]:
                for x in [p1, (p1[1], p1[0])]:
                    for y in [p2, (p2[1], p2[0])]:
                        out.add(x + y)
            return [Fact(p, v) for v in sorted(out)]

    elif p == "EqAngle":
        # swap the two angle-triples
        if len(a) == 6:
            return sorted(
                {Fact(p, a[:3] + a[3:]), Fact(p, a[3:] + a[:3])},
                key=lambda ff: ff.args,
            )

    elif p in ("Midpoint", "IsMidpoint"):
        # swap the two endpoints   (M,A,B) ↔ (M,B,A)
        if len(a) == 3:
            return sorted(
                {Fact(p, (a[0], a[1], a[2])), Fact(p, (a[0], a[2], a[1]))},
                key=lambda ff: ff.args,
            )

    elif p == "Circumcenter":
        # center fixed, permute the triangle vertices → 6 variants
        if len(a) == 4:
            return [
                Fact(p, (a[0],) + q)
                for q in sorted(set(_it.permutations(a[1:4])))
            ]

    elif p in ("Collinear", "Cyclic"):
        return [Fact(p, q) for q in sorted(set(_it.permutations(a)))]

    elif p == "Between":
        # swap the two outer points  (A,M,B) ↔ (B,M,A)
        if len(a) == 3:
            return sorted(
                {Fact(p, (a[0], a[1], a[2])), Fact(p, (a[2], a[1], a[0]))},
                key=lambda ff: ff.args,
            )

    elif p in ("SimTri", "CongTri", "EqArea"):
        if len(a) == 6:
            return sorted(
                {Fact(p, a[:3] + a[3:]), Fact(p, a[3:] + a[:3])},
                key=lambda ff: ff.args,
            )

    elif p == "Concurrent":
        if len(a) == 6:
            pairs = [(a[0], a[1]), (a[2], a[3]), (a[4], a[5])]
            out2: set = set()
            for pp in _it.permutations(pairs):
                for s0 in [pp[0], (pp[0][1], pp[0][0])]:
                    for s1 in [pp[1], (pp[1][1], pp[1][0])]:
                        for s2 in [pp[2], (pp[2][1], pp[2][0])]:
                            out2.add(s0 + s1 + s2)
            return [Fact(p, v) for v in sorted(out2)]

    return [f]


def semantic_theorem_fingerprint(
    assumptions: Iterable[Fact],
    goal: Fact,
) -> str:
    """Fingerprint a theorem (assumptions ⊢ goal) up to isomorphism.

    ``Parallel(A,B,C,D) ⊢ Parallel(C,D,A,B)`` and
    ``Parallel(X,Y,U,V) ⊢ Parallel(U,V,X,Y)`` produce the same hash.

    The approach enumerates all **symmetry-equivalent forms** of each
    fact (e.g. Midpoint(M,A,B) ↔ Midpoint(M,B,A)) combined with all
    permutations of the assumption list, and picks the lexicographically
    minimal canonical-relabeled string.  This guarantees isomorphism
    invariance regardless of the original point names.

    For assumption lists of size ≤ 8 and total search space ≤ 200 000,
    full enumeration is used.  Otherwise, a heuristic fallback applies.
    """
    import itertools

    assums_list = list(assumptions)
    n_assums = len(assums_list)

    # -- Pre-compute symmetry variants for each assumption and for goal -
    assum_variants = [_symmetry_variants(a) for a in assums_list]
    goal_variants = _symmetry_variants(goal)

    from functools import reduce
    total_combos = reduce(lambda x, y: x * y,
                          (len(v) for v in assum_variants), 1) * len(goal_variants)
    total_perms = math.factorial(min(n_assums, 8)) if n_assums <= 8 else 1
    total_search = total_combos * total_perms

    MAX_SEARCH = 200_000

    def _cs_raw(ordered_assums: List[Fact], g: Fact) -> str:
        """Canonical-relabel (no pre-normalisation) and serialise."""
        all_facts = list(ordered_assums) + [g]
        relabeled, _ = _canonical_relabel(all_facts)
        ra = sorted(relabeled[:n_assums], key=lambda f: (f.predicate, f.args))
        rg = relabeled[-1]
        a_str = "|".join(
            f"{f.predicate}({','.join(f.args)})" for f in ra
        )
        return f"{a_str}=>{rg.predicate}({','.join(rg.args)})"

    best: Optional[str] = None

    if total_search <= MAX_SEARCH:
        # Full enumeration: variants × permutations
        for combo in itertools.product(*assum_variants):
            for gv in goal_variants:
                if n_assums <= 8:
                    for perm in itertools.permutations(combo):
                        cs = _cs_raw(list(perm), gv)
                        if best is None or cs < best:
                            best = cs
                else:
                    sc = sorted(combo, key=lambda f: (f.predicate, f.args))
                    cs = _cs_raw(list(sc), gv)
                    if best is None or cs < best:
                        best = cs
    else:
        # Fallback: normalise symmetry, then try permutations only
        norm_a = [_normalize_symmetry(a) for a in assums_list]
        norm_g = _normalize_symmetry(goal)
        if n_assums <= 8:
            for perm in itertools.permutations(norm_a):
                all_f = list(perm) + [norm_g]
                all_f, _ = _canonical_relabel(all_f)
                all_f = [_normalize_symmetry(f) for f in all_f]
                ra = sorted(all_f[:n_assums], key=lambda f: (f.predicate, f.args))
                rg = all_f[-1]
                a_str = "|".join(
                    f"{f.predicate}({','.join(f.args)})" for f in ra
                )
                cs = f"{a_str}=>{rg.predicate}({','.join(rg.args)})"
                if best is None or cs < best:
                    best = cs
        else:
            sa = sorted(norm_a, key=lambda f: (f.predicate, f.args))
            all_f = sa + [norm_g]
            all_f, _ = _canonical_relabel(all_f)
            all_f = [_normalize_symmetry(f) for f in all_f]
            ra = sorted(all_f[:n_assums], key=lambda f: (f.predicate, f.args))
            rg = all_f[-1]
            a_str = "|".join(
                f"{f.predicate}({','.join(f.args)})" for f in ra
            )
            best = f"{a_str}=>{rg.predicate}({','.join(rg.args)})"

    return hashlib.sha256(best.encode()).hexdigest()[:20]


# ── Predicate → family mapping for structural fingerprinting ─────────

_PRED_FAMILY_STRUCTURAL: Dict[str, str] = {
    "Parallel":      "LINE",
    "Perpendicular": "LINE",
    "Collinear":     "LINE",
    "Between":       "LINE",
    "Midpoint":      "MIDPOINT",
    "AngleBisect":   "ANGLE",
    "Cong":          "METRIC",
    "EqAngle":       "ANGLE",
    "EqDist":        "METRIC",
    "EqRatio":       "METRIC",
    "EqArea":        "METRIC",
    "Cyclic":        "CIRCLE",
    "OnCircle":      "CIRCLE",
    "Circumcenter":  "CIRCLE",
    "Tangent":       "CIRCLE",
    "RadicalAxis":   "CIRCLE",
    "SimTri":        "SIMILARITY",
    "CongTri":       "SIMILARITY",
    "Concurrent":    "CONCURRENCY",
    "Harmonic":      "PROJECTIVE",
    "PolePolar":     "PROJECTIVE",
    "InvImage":      "PROJECTIVE",
    "EqCrossRatio":  "PROJECTIVE",
}


def structural_theorem_fingerprint(
    assumptions: Iterable[Fact],
    goal: Fact,
) -> str:
    """Fingerprint a theorem up to predicate-family substitution.

    This produces the **same** hash for theorems that differ only by
    swapping predicates within the same concept family.  For example,
    ``Parallel(A,B,C,D) ⊢ Perpendicular(E,F,G,H)`` and
    ``Perpendicular(A,B,C,D) ⊢ Parallel(E,F,G,H)`` produce the same
    structural fingerprint, because Parallel and Perpendicular both
    belong to the LINE family.

    Uses symmetry-variant + permutation enumeration (like
    ``semantic_theorem_fingerprint``) to correctly identify isomorphic
    theorems regardless of point names or assumption order.
    """
    import itertools

    assums_list = list(assumptions)
    n_assums = len(assums_list)

    # Replace each predicate with its family name
    def _familify(f: Fact) -> Fact:
        fam = _PRED_FAMILY_STRUCTURAL.get(f.predicate, f.predicate)
        return Fact(predicate=fam, args=f.args)

    family_assums = [_familify(f) for f in assums_list]
    family_goal = _familify(goal)

    # -- symmetry variants: compute on ORIGINAL facts, then familify ----
    #    (_symmetry_variants uses predicate names like "Midpoint" that
    #     would not match after familification to "MIDPOINT".)
    assum_variants = [
        [_familify(v) for v in _symmetry_variants(a)]
        for a in assums_list
    ]
    goal_variants = [_familify(v) for v in _symmetry_variants(goal)]

    from functools import reduce
    total_combos = reduce(lambda x, y: x * y,
                          (len(v) for v in assum_variants), 1) * len(goal_variants)
    total_perms = math.factorial(min(n_assums, 8)) if n_assums <= 8 else 1
    total_search = total_combos * total_perms

    MAX_SEARCH = 200_000

    def _cs_raw(ordered_assums: List[Fact], g: Fact) -> str:
        all_facts = list(ordered_assums) + [g]
        relabeled, _ = _canonical_relabel(all_facts)
        ra = sorted(relabeled[:n_assums], key=lambda f: (f.predicate, f.args))
        rg = relabeled[-1]
        a_str = "|".join(
            f"{f.predicate}({','.join(f.args)})" for f in ra
        )
        return f"STRUCT:{a_str}=>{rg.predicate}({','.join(rg.args)})"

    best: Optional[str] = None

    if total_search <= MAX_SEARCH:
        for combo in itertools.product(*assum_variants):
            for gv in goal_variants:
                if n_assums <= 8:
                    for perm in itertools.permutations(combo):
                        cs = _cs_raw(list(perm), gv)
                        if best is None or cs < best:
                            best = cs
                else:
                    sc = sorted(combo, key=lambda f: (f.predicate, f.args))
                    cs = _cs_raw(list(sc), gv)
                    if best is None or cs < best:
                        best = cs
    else:
        norm_a = [_normalize_symmetry(a) for a in family_assums]
        norm_g = _normalize_symmetry(family_goal)
        if n_assums <= 8:
            for perm in itertools.permutations(norm_a):
                all_f = list(perm) + [norm_g]
                all_f, _ = _canonical_relabel(all_f)
                all_f = [_normalize_symmetry(f) for f in all_f]
                ra = sorted(all_f[:n_assums], key=lambda f: (f.predicate, f.args))
                rg = all_f[-1]
                a_str = "|".join(
                    f"{f.predicate}({','.join(f.args)})" for f in ra
                )
                cs = f"STRUCT:{a_str}=>{rg.predicate}({','.join(rg.args)})"
                if best is None or cs < best:
                    best = cs
        else:
            sa = sorted(norm_a, key=lambda f: (f.predicate, f.args))
            all_f = sa + [norm_g]
            all_f, _ = _canonical_relabel(all_f)
            all_f = [_normalize_symmetry(f) for f in all_f]
            ra = sorted(all_f[:n_assums], key=lambda f: (f.predicate, f.args))
            rg = all_f[-1]
            a_str = "|".join(
                f"{f.predicate}({','.join(f.args)})" for f in ra
            )
            best = f"STRUCT:{a_str}=>{rg.predicate}({','.join(rg.args)})"

    return hashlib.sha256(best.encode()).hexdigest()[:20]


def semantic_proof_fingerprint(
    assumptions: Iterable[Fact],
    goal: Fact,
    steps: Sequence[Step],
) -> str:
    """Fingerprint a full proof trace up to isomorphism.

    Includes the theorem fingerprint *plus* the proof structure
    (rule sequence and connectivity), all canonically relabeled.
    """
    thm_fp = semantic_theorem_fingerprint(assumptions, goal)
    # Relabel everything under one canonical naming
    assums_list = list(assumptions)
    all_facts: List[Fact] = list(assums_list) + [goal]
    for s in steps:
        all_facts.extend(s.premise_facts)
        all_facts.append(s.conclusion_fact)
    _, mapping = _canonical_relabel(all_facts)

    def _rename(f: Fact) -> str:
        args = ",".join(mapping.get(a, a) for a in f.args)
        return f"{f.predicate}({args})"

    step_sigs = []
    for s in steps:
        prems = "|".join(sorted(_rename(f) for f in s.premise_facts))
        step_sigs.append(f"{s.rule_name}:{prems}->{_rename(s.conclusion_fact)}")
    proof_str = ";".join(sorted(step_sigs))
    raw = f"{thm_fp}::{proof_str}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


# ═══════════════════════════════════════════════════════════════════════
# 2.  Natural-language descriptions
# ═══════════════════════════════════════════════════════════════════════

_NL_TEMPLATES_ZH: Dict[str, str] = {
    "Parallel":      "直线 {0}{1} 平行于直线 {2}{3}",
    "Perpendicular": "直线 {0}{1} 垂直于直线 {2}{3}",
    "Collinear":     "点 {0}、{1}、{2} 共线",
    "Cyclic":        "点 {0}、{1}、{2}、{3} 共圆",
    "Midpoint":      "{0} 是线段 {1}{2} 的中点",
    "EqAngle":       "∠{0}{1}{2} = ∠{3}{4}{5}",
    "Cong":          "线段 {0}{1} ≅ 线段 {2}{3}",
    "SimTri":        "△{0}{1}{2} ∼ △{3}{4}{5} (相似)",
    "OnCircle":      "{1} 在以 {0} 为圆心的圆上",
    # New predicates
    "CongTri":       "△{0}{1}{2} ≅ △{3}{4}{5} (全等)",
    "Tangent":       "直线 {0}{1} 与圆 {2} 相切于点 {3}",
    "EqRatio":       "|{0}{1}|/|{2}{3}| = |{4}{5}|/|{6}{7}|",
    "Between":       "{1} 在 {0} 与 {2} 之间",
    "AngleBisect":   "射线 {0}{1} 平分∠{2}{0}{3}",
    "Concurrent":    "直线 {0}{1}、{2}{3}、{4}{5} 共点",
    "Circumcenter":  "{0} 是 △{1}{2}{3} 的外心",
    "EqDist":        "|{0}{1}| = |{0}{2}| (等距)",
    "EqArea":        "S(△{0}{1}{2}) = S(△{3}{4}{5})",
    "Harmonic":      "({0},{1};{2},{3}) 为调和点列",
    "PolePolar":     "{0} 是直线 {1}{2} 关于圆 {3} 的极点",
    "InvImage":      "{0} 是 {1} 关于圆({2},{3})的反演像",
    "EqCrossRatio":  "({0},{1};{2},{3}) = ({4},{5};{6},{7})",
    "RadicalAxis":   "直线 {0}{1} 是圆 {2} 与圆 {3} 的根轴",
}

_NL_TEMPLATES_EN: Dict[str, str] = {
    "Parallel":      "line {0}{1} is parallel to line {2}{3}",
    "Perpendicular": "line {0}{1} is perpendicular to line {2}{3}",
    "Collinear":     "points {0}, {1}, {2} are collinear",
    "Cyclic":        "points {0}, {1}, {2}, {3} are concyclic",
    "Midpoint":      "{0} is the midpoint of segment {1}{2}",
    "EqAngle":       "∠{0}{1}{2} = ∠{3}{4}{5}",
    "Cong":          "segment {0}{1} ≅ segment {2}{3}",
    "SimTri":        "△{0}{1}{2} ~ △{3}{4}{5} (similar)",
    "OnCircle":      "{1} lies on circle centred at {0}",
    # New predicates
    "CongTri":       "△{0}{1}{2} ≅ △{3}{4}{5} (congruent)",
    "Tangent":       "line {0}{1} is tangent to circle {2} at {3}",
    "EqRatio":       "|{0}{1}|/|{2}{3}| = |{4}{5}|/|{6}{7}|",
    "Between":       "{1} lies between {0} and {2}",
    "AngleBisect":   "ray {0}{1} bisects ∠{2}{0}{3}",
    "Concurrent":    "lines {0}{1}, {2}{3}, {4}{5} are concurrent",
    "Circumcenter":  "{0} is the circumcentre of △{1}{2}{3}",
    "EqDist":        "|{0}{1}| = |{0}{2}| (equidistant)",
    "EqArea":        "area(△{0}{1}{2}) = area(△{3}{4}{5})",
    "Harmonic":      "({0},{1};{2},{3}) is a harmonic range",
    "PolePolar":     "{0} is the pole of line {1}{2} w.r.t. circle {3}",
    "InvImage":      "{0} is the inversion of {1} w.r.t. circle({2},{3})",
    "EqCrossRatio":  "({0},{1};{2},{3}) = ({4},{5};{6},{7})",
    "RadicalAxis":   "line {0}{1} is the radical axis of circles {2} and {3}",
}


def fact_to_nl(fact: Fact, lang: str = "zh") -> str:
    """Render a Fact as a natural-language sentence.

    Parameters
    ----------
    fact : Fact
    lang : ``"zh"`` (Chinese) or ``"en"`` (English)
    """
    templates = _NL_TEMPLATES_ZH if lang == "zh" else _NL_TEMPLATES_EN
    tpl = templates.get(fact.predicate)
    if tpl is None:
        return str(fact)
    try:
        return tpl.format(*fact.args)
    except (IndexError, KeyError):
        return str(fact)


def theorem_to_nl(
    assumptions: Sequence[Fact],
    goal: Fact,
    lang: str = "zh",
) -> str:
    """Render a theorem as a natural-language statement.

    Example (zh)::

        已知：直线 AB 平行于直线 CD，直线 CD 平行于直线 EF。
        求证：直线 AB 平行于直线 EF。
    """
    if lang == "zh":
        given = "，".join(fact_to_nl(f, lang) for f in assumptions)
        concl = fact_to_nl(goal, lang)
        return f"已知：{given}。\n求证：{concl}。"
    else:
        given = "; ".join(fact_to_nl(f, lang) for f in assumptions)
        concl = fact_to_nl(goal, lang)
        return f"Given: {given}.\nProve: {concl}."


def proof_to_nl(
    assumptions: Sequence[Fact],
    steps: Sequence[Step],
    goal: Fact,
    lang: str = "zh",
) -> str:
    """Render a complete proof trace as natural-language text."""
    lines: List[str] = [theorem_to_nl(assumptions, goal, lang), ""]

    if lang == "zh":
        lines.append("证明：")
        for i, step in enumerate(steps, 1):
            prems = "、".join(fact_to_nl(f, lang) for f in step.premise_facts)
            concl = fact_to_nl(step.conclusion_fact, lang)
            rule_zh = _RULE_NL_ZH.get(step.rule_name, step.rule_name)
            lines.append(f"  {i}. 由{prems}，根据{rule_zh}，得{concl}。")
        lines.append("证毕。 ∎")
    else:
        lines.append("Proof:")
        for i, step in enumerate(steps, 1):
            prems = ", ".join(fact_to_nl(f, lang) for f in step.premise_facts)
            concl = fact_to_nl(step.conclusion_fact, lang)
            rule_en = _RULE_NL_EN.get(step.rule_name, step.rule_name)
            lines.append(f"  {i}. Since {prems}, by {rule_en}, {concl}.")
        lines.append("QED ∎")
    return "\n".join(lines)


_RULE_NL_ZH: Dict[str, str] = {
    "parallel_symmetry":     "平行线对称性",
    "parallel_transitivity": "平行线传递性",
    "perp_symmetry":         "垂直线对称性",
    "parallel_perp_trans":   "平行传递垂直",
    "midpoint_collinear":    "中点共线",
    "midpoint_cong":         "中点等分",
    "midsegment_parallel":   "三角形中位线定理",
    "cong_symm":             "线段全等对称性",
    "cong_trans":            "线段全等传递性",
    "eq_angle_symm":         "等角对称性",
    "eq_angle_trans":        "等角传递性",
    "cyclic_inscribed_angle": "圆周角定理",
    "perp_bisector_cong":    "垂直平分线定理",
    "isosceles_base_angle":  "等腰三角形底角定理",
    "cong_perp_bisector":    "垂直平分线逆定理",
    "parallel_alternate_angle": "平行线内错角定理",
    "cyclic_chord_angle":    "圆内弦切角定理",
    "midsegment_sim_tri":    "中位线相似三角形定理",
    "sim_tri_angle":         "相似三角形等角",
    "sim_tri_cong":          "相似+全等边⇒全等三角形",
    # ── New rules ──
    "congtri_side":          "全等三角形对应边",
    "congtri_angle":         "全等三角形对应角",
    "congtri_from_sim_cong": "相似+对应边全等⇒全等",
    "congtri_eqarea":        "全等三角形等面积",
    "tangent_perp_radius":   "切线垂直于半径",
    "tangent_oncircle":      "切点在圆上",
    "eqratio_from_simtri":   "相似三角形对应边比",
    "eqratio_sym":           "等比对称性",
    "eqratio_trans":         "等比传递性",
    "between_collinear":     "介于性⇒共线",
    "midpoint_between":      "中点⇒介于性",
    "angle_bisect_eqangle":  "角平分线等角",
    "angle_bisect_eqratio":  "角平分线等比",
    "medians_concurrent":    "三角形中线共点定理",
    "circumcenter_cong_ab":  "外心等距 AB",
    "circumcenter_cong_bc":  "外心等距 BC",
    "circumcenter_oncircle": "外心的外接圆",
    "eqdist_from_cong":      "全等⇒等距",
    "eqdist_to_cong":        "等距⇒全等",
    "eqarea_sym":            "等面积对称性",
    "harmonic_swap":         "调和点列交换",
    "harmonic_collinear":    "调和点列共线",
    "pole_polar_perp":       "极点极线垂直",
    "pole_polar_tangent":    "极点极线切线",
    "inversion_collinear":   "反演共线",
    "inversion_circle_fixed": "反演不动圆",
    "cross_ratio_sym":       "交比对称性",
    "cross_ratio_from_harmonic": "调和点列⇒等交比",
    "radical_axis_perp":     "根轴垂直于连心线",
}

_RULE_NL_EN: Dict[str, str] = {
    "parallel_symmetry":     "parallel symmetry",
    "parallel_transitivity": "parallel transitivity",
    "perp_symmetry":         "perpendicular symmetry",
    "parallel_perp_trans":   "parallel-perpendicular transfer",
    "midpoint_collinear":    "midpoint collinearity",
    "midpoint_cong":         "midpoint congruence",
    "midsegment_parallel":   "midsegment theorem",
    "cong_symm":             "congruence symmetry",
    "cong_trans":            "congruence transitivity",
    "eq_angle_symm":         "angle equality symmetry",
    "eq_angle_trans":        "angle equality transitivity",
    "cyclic_inscribed_angle": "inscribed angle theorem",
    "perp_bisector_cong":    "perpendicular bisector theorem",
    "isosceles_base_angle":  "isosceles base angle theorem",
    "cong_perp_bisector":    "converse perpendicular bisector",
    "parallel_alternate_angle": "alternate interior angle theorem",
    "cyclic_chord_angle":    "cyclic chord angle theorem",
    "midsegment_sim_tri":    "midsegment similar triangle",
    "sim_tri_angle":         "similar triangle equal angles",
    "sim_tri_cong":          "similar + congruent side ⇒ congruent triangle",
    # ── New rules ──
    "congtri_side":          "congruent triangle corresponding side",
    "congtri_angle":         "congruent triangle corresponding angle",
    "congtri_from_sim_cong": "similar + equal side ⇒ congruent",
    "congtri_eqarea":        "congruent triangle equal area",
    "tangent_perp_radius":   "tangent perpendicular to radius",
    "tangent_oncircle":      "tangent point on circle",
    "eqratio_from_simtri":   "similar triangle side ratio",
    "eqratio_sym":           "ratio equality symmetry",
    "eqratio_trans":         "ratio equality transitivity",
    "between_collinear":     "betweenness ⇒ collinearity",
    "midpoint_between":      "midpoint ⇒ betweenness",
    "angle_bisect_eqangle":  "angle bisector equal angles",
    "angle_bisect_eqratio":  "angle bisector ratio",
    "medians_concurrent":    "triangle medians concurrence",
    "circumcenter_cong_ab":  "circumcenter equidistant AB",
    "circumcenter_cong_bc":  "circumcenter equidistant BC",
    "circumcenter_oncircle": "circumcenter circumscribed circle",
    "eqdist_from_cong":      "congruence ⇒ equidistance",
    "eqdist_to_cong":        "equidistance ⇒ congruence",
    "eqarea_sym":            "equal area symmetry",
    "harmonic_swap":         "harmonic range pair swap",
    "harmonic_collinear":    "harmonic range collinearity",
    "pole_polar_perp":       "pole-polar perpendicularity",
    "pole_polar_tangent":    "pole-polar tangent",
    "inversion_collinear":   "inversion collinearity",
    "inversion_circle_fixed": "inversion fixed circle",
    "cross_ratio_sym":       "cross-ratio symmetry",
    "cross_ratio_from_harmonic": "harmonic range ⇒ equal cross-ratio",
    "radical_axis_perp":     "radical axis perpendicular to center line",
}


# ═══════════════════════════════════════════════════════════════════════
# 3.  Lean4 theorem statement generation
# ═══════════════════════════════════════════════════════════════════════

_PRED_LEAN_NAME: Dict[str, str] = {
    "Parallel":      "Parallel",
    "Perpendicular": "Perpendicular",
    "Collinear":     "Collinear",
    "Cyclic":        "Cyclic",
    "Midpoint":      "IsMidpoint",
    "EqAngle":       "EqAngle",
    "Cong":          "Cong",
    "SimTri":        "SimTri",
    "OnCircle":      "OnCircle",
    # New predicates
    "CongTri":       "CongTri",
    "Tangent":       "Tangent",
    "EqRatio":       "EqRatio",
    "Between":       "Between",
    "AngleBisect":   "AngleBisect",
    "Concurrent":    "Concurrent",
    "Circumcenter":  "Circumcenter",
    "EqDist":        "EqDist",
    "EqArea":        "EqArea",
    "Harmonic":      "Harmonic",
    "PolePolar":     "PolePolar",
    "InvImage":      "InvImage",
    "EqCrossRatio":  "EqCrossRatio",
    "RadicalAxis":   "RadicalAxis",
}


def _fact_to_lean_prop(fact: Fact) -> str:
    pred = _PRED_LEAN_NAME.get(fact.predicate, fact.predicate)
    return f"{pred} {' '.join(fact.args)}"


def theorem_to_lean(
    assumptions: Sequence[Fact],
    goal: Fact,
    name: str = "geo_theorem",
    with_proof: bool = False,
    proof_steps: Sequence[Step] | None = None,
) -> str:
    """Generate a self-contained Lean4 theorem statement.

    Parameters
    ----------
    assumptions : premises
    goal : conclusion
    name : theorem name in the generated Lean source
    with_proof : if True and *proof_steps* given, produce a real
                 proof term; otherwise produce ``sorry``.
    proof_steps : derivation steps (used when ``with_proof=True``)

    Returns
    -------
    A complete ``.lean`` source string with ``import LeanGeo``.
    """
    from .lean_bridge import RULE_LEAN_MAP   # reuse existing mapping

    # Gather all point names
    points: Set[str] = set()
    for f in assumptions:
        points.update(f.args)
    points.update(goal.args)
    if proof_steps:
        for s in proof_steps:
            for f in s.premise_facts:
                points.update(f.args)
            points.update(s.conclusion_fact.args)

    lines: List[str] = ["import LeanGeo", ""]

    # Declare points
    for p in sorted(points):
        lines.append(f"variable ({p} : GPoint)")
    lines.append("")

    # Build theorem signature
    hyp_parts: List[str] = []
    hyp_names: List[str] = []
    for i, f in enumerate(assumptions):
        h = f"h{i}"
        hyp_names.append(h)
        hyp_parts.append(f"({h} : {_fact_to_lean_prop(f)})")
    hyps_str = " ".join(hyp_parts)
    goal_str = _fact_to_lean_prop(goal)
    lines.append(f"theorem {name} {hyps_str} : {goal_str} :=")

    if with_proof and proof_steps:
        # Build proof term step by step via let-bindings
        fact_var: Dict[Fact, str] = {}
        for i, f in enumerate(assumptions):
            fact_var[f] = hyp_names[i]
        let_lines: List[str] = []
        final_var = ""
        for idx, step in enumerate(proof_steps):
            spec = RULE_LEAN_MAP.get(step.rule_name)
            if spec is None:
                # Fallback to sorry
                lines.append("  sorry")
                return "\n".join(lines)
            vname = f"s{idx}"
            hyps = [fact_var.get(f, "sorry") for f in step.premise_facts]
            pt_args = spec.point_extractor(step.premise_facts)
            proof = f"{spec.lean_lemma} {' '.join(pt_args)} {' '.join(hyps)}"
            let_lines.append(f"  let {vname} := {proof}")
            fact_var[step.conclusion_fact] = vname
            final_var = vname
        for ll in let_lines:
            lines.append(ll)
        lines.append(f"  {final_var}")
    else:
        lines.append("  sorry")

    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 4.  Geometry visualisation
# ═══════════════════════════════════════════════════════════════════════

# Predefined point positions on a unit circle / grid for common cases.
# For production use, a constraint solver would be better, but this
# gives stable, deterministic results for small diagrams.


def _assign_coordinates(
    points: Sequence[str],
    facts: Sequence[Fact],
) -> Dict[str, Tuple[float, float]]:
    """Heuristic coordinate assignment for geometry drawing.

    Strategy:
    - For ≤ 8 points, place on a circle of radius 3.
    - Adjust positions to honour parallel / perpendicular constraints
      where possible.
    """
    n = len(points)
    coords: Dict[str, Tuple[float, float]] = {}

    # Default: circle layout
    radius = 3.0
    for i, p in enumerate(sorted(points)):
        angle = 2 * math.pi * i / max(n, 1) - math.pi / 2
        coords[p] = (radius * math.cos(angle), radius * math.sin(angle))

    # Try to adjust for Parallel facts: make lines actually parallel
    for f in facts:
        if f.predicate == "Parallel" and len(f.args) == 4:
            a, b, c, d = f.args
            if all(p in coords for p in (a, b, c, d)):
                # Make CD vector equal to AB vector (translation)
                ax, ay = coords[a]
                bx, by = coords[b]
                dx_vec, dy_vec = bx - ax, by - ay
                cx, cy = coords[c]
                coords[d] = (cx + dx_vec, cy + dy_vec)

    # Perpendicular: rotate vector 90°
    for f in facts:
        if f.predicate == "Perpendicular" and len(f.args) == 4:
            a, b, c, d = f.args
            if all(p in coords for p in (a, b, c)):
                ax, ay = coords[a]
                bx, by = coords[b]
                dx_vec, dy_vec = bx - ax, by - ay
                cx, cy = coords[c]
                # Rotate 90°
                coords[d] = (cx - dy_vec, cy + dx_vec)

    # Midpoint
    for f in facts:
        if f.predicate == "Midpoint" and len(f.args) == 3:
            m, a, b = f.args
            if a in coords and b in coords:
                ax, ay = coords[a]
                bx, by = coords[b]
                coords[m] = ((ax + bx) / 2, (ay + by) / 2)

    return coords


def _collect_points(facts: Iterable[Fact]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for f in facts:
        for a in f.args:
            if a not in seen:
                seen.add(a)
                result.append(a)
    return result


def draw_geometry(
    facts: Sequence[Fact],
    goal: Optional[Fact] = None,
    title: str = "",
    output_path: Optional[str | Path] = None,
    show: bool = False,
    figsize: Tuple[float, float] = (8, 6),
) -> Optional[str]:
    """Draw a geometry diagram from facts.

    Parameters
    ----------
    facts : the geometric facts to visualise
    goal : highlight the goal relation in a different colour
    title : figure title
    output_path : save to this file (PNG / SVG / PDF); if None, auto-name
    show : whether to call ``plt.show()`` (interactive mode)
    figsize : figure size in inches

    Returns
    -------
    The path of the saved image file, or None if ``show=True`` only.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Use a CJK-capable font if available (macOS: Heiti, PingFang, etc.)
    import matplotlib.font_manager as fm
    _cjk_candidates = ["Heiti TC", "Heiti SC", "PingFang SC", "PingFang TC",
                        "Hiragino Sans GB", "Hiragino Sans",
                        "Noto Sans CJK SC", "SimHei", "Microsoft YaHei"]
    _available = {f.name for f in fm.fontManager.ttflist}
    _cjk_font = None
    for cand in _cjk_candidates:
        if cand in _available:
            _cjk_font = cand
            break
    if _cjk_font:
        plt.rcParams["font.sans-serif"] = [_cjk_font] + plt.rcParams["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False

    points = _collect_points(facts)
    if goal:
        points = _collect_points(list(facts) + [goal])
    coords = _assign_coordinates(points, facts)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_aspect("equal")
    ax.set_title(title or "Geometry Diagram", fontsize=14, fontfamily="serif")
    ax.grid(True, alpha=0.2)

    # Colour scheme
    colour_parallel = "#2196F3"   # blue
    colour_perp     = "#F44336"   # red
    colour_collinear = "#9E9E9E"  # grey
    colour_cyclic   = "#4CAF50"   # green
    colour_midpoint = "#FF9800"   # orange
    colour_goal     = "#E91E63"   # pink (for goal highlight)
    colour_default  = "#607D8B"   # blue-grey

    drawn_lines: Set[Tuple[str, str]] = set()

    def _draw_line(p1: str, p2: str, colour: str, lw: float = 1.5,
                   ls: str = "-", extend: float = 0.3) -> None:
        """Draw line through two points, extending slightly beyond."""
        if (p1, p2) in drawn_lines or (p2, p1) in drawn_lines:
            return
        drawn_lines.add((p1, p2))
        x1, y1 = coords[p1]
        x2, y2 = coords[p2]
        dx, dy = x2 - x1, y2 - y1
        ax.plot(
            [x1 - extend * dx, x2 + extend * dx],
            [y1 - extend * dy, y2 + extend * dy],
            color=colour, linewidth=lw, linestyle=ls, zorder=1,
        )

    def _draw_right_angle(vertex: str, p1: str, p2: str,
                          size: float = 0.3) -> None:
        """Draw a small right-angle square at vertex."""
        vx, vy = coords[vertex]
        x1, y1 = coords[p1]
        x2, y2 = coords[p2]
        # Unit vectors from vertex toward p1 and p2
        d1 = math.hypot(x1 - vx, y1 - vy)
        d2 = math.hypot(x2 - vx, y2 - vy)
        if d1 < 1e-9 or d2 < 1e-9:
            return
        ux1, uy1 = (x1 - vx) / d1, (y1 - vy) / d1
        ux2, uy2 = (x2 - vx) / d2, (y2 - vy) / d2
        sq_x = [vx, vx + size * ux1, vx + size * (ux1 + ux2),
                vx + size * ux2, vx]
        sq_y = [vy, vy + size * uy1, vy + size * (uy1 + uy2),
                vy + size * uy2, vy]
        ax.plot(sq_x, sq_y, color=colour_perp, linewidth=1, zorder=2)

    # Draw facts
    for f in facts:
        is_goal = (goal is not None and f == goal)
        if f.predicate == "Parallel" and len(f.args) == 4:
            a, b, c, d = f.args
            col = colour_goal if is_goal else colour_parallel
            _draw_line(a, b, col, lw=2)
            _draw_line(c, d, col, lw=2)
            # Draw parallel arrows (small tick marks)
            for p1, p2 in [(a, b), (c, d)]:
                mx = (coords[p1][0] + coords[p2][0]) / 2
                my = (coords[p1][1] + coords[p2][1]) / 2
                ax.annotate("∥", (mx, my), fontsize=10, color=col,
                           ha="center", va="center",
                           bbox=dict(boxstyle="round,pad=0.1",
                                     fc="white", ec="none", alpha=0.8))

        elif f.predicate == "Perpendicular" and len(f.args) == 4:
            a, b, c, d = f.args
            col = colour_goal if is_goal else colour_perp
            _draw_line(a, b, col, lw=2)
            _draw_line(c, d, col, lw=2)
            # Mark right angle symbol at intersection if possible
            ax.annotate("⊥", (
                (coords[a][0] + coords[b][0] + coords[c][0] + coords[d][0]) / 4,
                (coords[a][1] + coords[b][1] + coords[c][1] + coords[d][1]) / 4,
            ), fontsize=12, color=col, ha="center", va="center")

        elif f.predicate == "Collinear" and len(f.args) == 3:
            a, b, c = f.args
            col = colour_goal if is_goal else colour_collinear
            _draw_line(a, c, col, lw=1.5, ls="--")

        elif f.predicate == "Cyclic" and len(f.args) == 4:
            a, b, c, d = f.args
            col = colour_goal if is_goal else colour_cyclic
            cx = sum(coords[p][0] for p in (a, b, c, d)) / 4
            cy = sum(coords[p][1] for p in (a, b, c, d)) / 4
            r = max(math.hypot(coords[p][0] - cx, coords[p][1] - cy)
                    for p in (a, b, c, d))
            circle = plt.Circle((cx, cy), r, fill=False,
                               color=col, linewidth=1.5, linestyle="--")
            ax.add_patch(circle)

        elif f.predicate == "Midpoint" and len(f.args) == 3:
            m, a, b = f.args
            col = colour_goal if is_goal else colour_midpoint
            _draw_line(a, b, col, lw=1.5)
            mx, my = coords[m]
            ax.plot(mx, my, "D", color=col, markersize=6, zorder=4)

    # Draw goal relation in distinct style if not already drawn
    if goal is not None and goal not in facts:
        if goal.predicate in ("Parallel", "Perpendicular") and len(goal.args) == 4:
            a, b, c, d = goal.args
            _draw_line(a, b, colour_goal, lw=2.5, ls="--")
            _draw_line(c, d, colour_goal, lw=2.5, ls="--")

    # Draw point markers and labels
    for p, (x, y) in coords.items():
        ax.plot(x, y, "o", color="#212121", markersize=7, zorder=5)
        ax.annotate(
            f" {p}", (x, y), fontsize=13, fontweight="bold",
            color="#212121", zorder=6,
            xytext=(6, 6), textcoords="offset points",
        )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = []
    preds_used = {f.predicate for f in facts}
    if goal:
        preds_used.add(goal.predicate)
    if "Parallel" in preds_used:
        legend_elements.append(
            Line2D([0], [0], color=colour_parallel, lw=2, label="平行 Parallel"))
    if "Perpendicular" in preds_used:
        legend_elements.append(
            Line2D([0], [0], color=colour_perp, lw=2, label="垂直 Perp"))
    if "Collinear" in preds_used:
        legend_elements.append(
            Line2D([0], [0], color=colour_collinear, lw=1.5, ls="--",
                   label="共线 Collinear"))
    if "Cyclic" in preds_used:
        legend_elements.append(
            Line2D([0], [0], color=colour_cyclic, lw=1.5, ls="--",
                   label="共圆 Cyclic"))
    if goal is not None:
        legend_elements.append(
            Line2D([0], [0], color=colour_goal, lw=2.5, ls="--",
                   label="目标 Goal"))
    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    plt.tight_layout()

    saved_path: Optional[str] = None
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        saved_path = str(out)
    elif not show:
        # Default: save next to knowledge data
        default_dir = Path(__file__).resolve().parent.parent / "data" / "figures"
        default_dir.mkdir(parents=True, exist_ok=True)
        fname = f"diagram_{semantic_fact_fingerprint(facts)[:10]}.png"
        out = default_dir / fname
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        saved_path = str(out)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path
