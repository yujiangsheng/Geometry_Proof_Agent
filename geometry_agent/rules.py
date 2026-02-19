"""rules.py – Deduction rules for the geometry reasoning engine.

Each Rule subclass encodes one logical inference schema.  Rules use
``GeoState.by_predicate()`` for O(1) indexed lookup instead of
scanning all facts, making rule application scale to large states.

Adding a new rule:
  1. Subclass ``Rule`` and implement ``apply()``.
  2. Register it in ``default_rules()``.
  3. Add the matching axiom to ``lean_geo/LeanGeo/Rules.lean``.
  4. Add a ``RuleLeanSpec`` entry in ``lean_bridge.RULE_LEAN_MAP``.

Author:  Jiangsheng Yu
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .dsl import (
    Fact, GeoState, Step,
    canonical_parallel, canonical_perp,
    canonical_collinear, canonical_cong, canonical_eq_angle,
    canonical_midpoint, canonical_cyclic,
    canonical_sim_tri, canonical_circle,
    # New predicates
    canonical_congtri, canonical_tangent, canonical_eqratio,
    canonical_between, canonical_angle_bisect, canonical_concurrent,
    canonical_circumcenter, canonical_eqdist, canonical_eqarea,
    canonical_harmonic, canonical_pole_polar, canonical_inv_image,
    canonical_eq_cross_ratio, canonical_radical_axis,
)


@dataclass(frozen=True)
class RuleApplication:
    """A concrete rule firing: one step ready to be checked and applied."""
    step: Step


class Rule:
    """Abstract base for all deduction rules."""
    name: str = "UnnamedRule"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        """Yield all possible applications of this rule to *state*."""
        raise NotImplementedError


# ── Parallel ─────────────────────────────────────────────────────────


class ParallelSymmetryRule(Rule):
    """If AB ∥ CD then CD ∥ AB  (Lean: ``parallel_symm``)."""
    name = "parallel_symmetry"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("Parallel"):
            if len(f.args) != 4:
                continue
            a, b, c, d = f.args
            inferred = canonical_parallel(c, d, a, b)
            if state.has_fact(inferred):
                continue
            yield RuleApplication(
                step=Step(self.name, (f,), inferred),
            )


class ParallelTransitivityRule(Rule):
    """If AB ∥ CD and CD ∥ EF then AB ∥ EF  (Lean: ``parallel_trans``)."""
    name = "parallel_transitivity"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        parallels = [f for f in state.by_predicate("Parallel") if len(f.args) == 4]
        for f1 in parallels:
            _, _, c, d = f1.args
            for f2 in parallels:
                if f2 is f1:
                    continue
                e, f, g, h = f2.args
                if (c, d) != (e, f):
                    continue
                inferred = canonical_parallel(f1.args[0], f1.args[1], g, h)
                if state.has_fact(inferred):
                    continue
                yield RuleApplication(
                    step=Step(self.name, (f1, f2), inferred),
                )


# ── Perpendicular ────────────────────────────────────────────────────


class PerpSymmetryRule(Rule):
    """If AB ⊥ CD then CD ⊥ AB  (Lean: ``perp_symm``)."""
    name = "perp_symmetry"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("Perpendicular"):
            if len(f.args) != 4:
                continue
            a, b, c, d = f.args
            inferred = canonical_perp(c, d, a, b)
            if state.has_fact(inferred):
                continue
            yield RuleApplication(
                step=Step(self.name, (f,), inferred),
            )


class ParallelPerpTransRule(Rule):
    """If AB ∥ CD and CD ⊥ EF then AB ⊥ EF  (Lean: ``parallel_perp_trans``)."""
    name = "parallel_perp_trans"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        parallels = [f for f in state.by_predicate("Parallel") if len(f.args) == 4]
        perps = [f for f in state.by_predicate("Perpendicular") if len(f.args) == 4]
        for fp in parallels:
            _, _, c, d = fp.args
            for fq in perps:
                e, f, g, h = fq.args
                if (c, d) != (e, f):
                    continue
                inferred = canonical_perp(fp.args[0], fp.args[1], g, h)
                if state.has_fact(inferred):
                    continue
                yield RuleApplication(
                    step=Step(self.name, (fp, fq), inferred),
                )


# ── Collinear ────────────────────────────────────────────────────────
# NOTE: CollinearPerm12Rule and CollinearCycleRule were removed because
# canonical_collinear() sorts all 3 points — permuting then re-canonicalising
# always yields the same Fact, so the rules never fire.


# ── Midpoint ─────────────────────────────────────────────────────────


class MidpointCollinearRule(Rule):
    """IsMidpoint(M,A,B) → Collinear(A,M,B)  (Lean: ``midpoint_collinear``)."""
    name = "midpoint_collinear"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("Midpoint"):
            if len(f.args) != 3:
                continue
            m, a, b = f.args
            inferred = canonical_collinear(a, m, b)
            if state.has_fact(inferred):
                continue
            yield RuleApplication(step=Step(self.name, (f,), inferred))


class MidpointCongRule(Rule):
    """IsMidpoint(M,A,B) → Cong(A,M,M,B)  (Lean: ``midpoint_cong``)."""
    name = "midpoint_cong"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("Midpoint"):
            if len(f.args) != 3:
                continue
            m, a, b = f.args
            inferred = canonical_cong(a, m, m, b)
            if state.has_fact(inferred):
                continue
            yield RuleApplication(step=Step(self.name, (f,), inferred))


class MidsegmentParallelRule(Rule):
    """Midsegment theorem:
    IsMidpoint(M,A,B) ∧ IsMidpoint(N,A,C) → Parallel(M,N,B,C).

    If M,N are midpoints of two sides of triangle ABC sharing vertex A,
    then MN ∥ BC.  (Lean: ``midsegment_parallel``).
    """
    name = "midsegment_parallel"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        midpoints = [f for f in state.by_predicate("Midpoint") if len(f.args) == 3]
        for i, f1 in enumerate(midpoints):
            m1, a1, b1 = f1.args
            for f2 in midpoints[i + 1:]:
                m2, a2, b2 = f2.args
                # Find shared endpoint (the triangle vertex)
                shared = None
                other1 = other2 = None
                if a1 == a2:
                    shared, other1, other2 = a1, b1, b2
                elif a1 == b2:
                    shared, other1, other2 = a1, b1, a2
                elif b1 == a2:
                    shared, other1, other2 = b1, a1, b2
                elif b1 == b2:
                    shared, other1, other2 = b1, a1, a2
                if shared is None:
                    continue
                # MN ∥ (the two non-shared endpoints)
                inferred = canonical_parallel(m1, m2, other1, other2)
                if state.has_fact(inferred):
                    continue
                yield RuleApplication(
                    step=Step(self.name, (f1, f2), inferred),
                )


# ── Congruence ───────────────────────────────────────────────────────


class CongSymmRule(Rule):
    """Cong(A,B,C,D) → Cong(C,D,A,B)  (Lean: ``cong_symm``)."""
    name = "cong_symm"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("Cong"):
            if len(f.args) != 4:
                continue
            a, b, c, d = f.args
            inferred = canonical_cong(c, d, a, b)
            if state.has_fact(inferred):
                continue
            yield RuleApplication(step=Step(self.name, (f,), inferred))


class CongTransRule(Rule):
    """Cong(A,B,C,D) ∧ Cong(C,D,E,F) → Cong(A,B,E,F)  (Lean: ``cong_trans``)."""
    name = "cong_trans"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        congs = [f for f in state.by_predicate("Cong") if len(f.args) == 4]
        for f1 in congs:
            _, _, c1, d1 = f1.args
            for f2 in congs:
                if f2 is f1:
                    continue
                e, f, g, h = f2.args
                if (c1, d1) != (e, f):
                    continue
                inferred = canonical_cong(f1.args[0], f1.args[1], g, h)
                if state.has_fact(inferred):
                    continue
                yield RuleApplication(
                    step=Step(self.name, (f1, f2), inferred),
                )


# ── Angle equality ───────────────────────────────────────────────────


class EqAngleSymmRule(Rule):
    """EqAngle(A,B,C,D,E,F) → EqAngle(D,E,F,A,B,C)  (Lean: ``eq_angle_symm``)."""
    name = "eq_angle_symm"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("EqAngle"):
            if len(f.args) != 6:
                continue
            a, b, c, d, e, ff = f.args
            inferred = canonical_eq_angle(d, e, ff, a, b, c)
            if state.has_fact(inferred):
                continue
            yield RuleApplication(step=Step(self.name, (f,), inferred))


class EqAngleTransRule(Rule):
    """EqAngle(A,B,C,D,E,F) ∧ EqAngle(D,E,F,G,H,I) → EqAngle(A,B,C,G,H,I).

    Lean: ``eq_angle_trans``.
    """
    name = "eq_angle_trans"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        angles = [f for f in state.by_predicate("EqAngle") if len(f.args) == 6]
        for f1 in angles:
            _, _, _, d1, e1, ff1 = f1.args
            for f2 in angles:
                if f2 is f1:
                    continue
                g, h, i, j, k, l = f2.args
                if (d1, e1, ff1) != (g, h, i):
                    continue
                inferred = canonical_eq_angle(
                    f1.args[0], f1.args[1], f1.args[2], j, k, l,
                )
                if state.has_fact(inferred):
                    continue
                yield RuleApplication(
                    step=Step(self.name, (f1, f2), inferred),
                )


# ── Cyclic ───────────────────────────────────────────────────────────


# NOTE: CyclicPermRule removed — canonical_cyclic() sorts all 4 points,
# so cyclic permutation always yields the same Fact.


class CyclicEqAngleRule(Rule):
    """Inscribed angle theorem: Cyclic(A,B,C,D) → EqAngle(B,A,C, B,D,C).

    Angles subtended by the same chord BC from two points on the circle.
    Lean: ``cyclic_inscribed_angle``.
    """
    name = "cyclic_inscribed_angle"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("Cyclic"):
            if len(f.args) != 4:
                continue
            a, b, c, d = f.args
            # ∠BAC = ∠BDC (inscribed angles on same arc)
            inferred = canonical_eq_angle(b, a, c, b, d, c)
            if state.has_fact(inferred):
                continue
            yield RuleApplication(step=Step(self.name, (f,), inferred))


class PerpBisectorCongRule(Rule):
    """Perpendicular bisector theorem:
    IsMidpoint(M,A,B) ∧ Perpendicular(C,M,A,B) → Cong(C,A,C,B).

    If M is the midpoint of AB and CM ⊥ AB, then |CA| = |CB|.
    Lean: ``perp_bisector_cong``.
    """
    name = "perp_bisector_cong"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        midpoints = [f for f in state.by_predicate("Midpoint") if len(f.args) == 3]
        perps = [f for f in state.by_predicate("Perpendicular") if len(f.args) == 4]
        for fm in midpoints:
            mid, a, b = fm.args
            ab_pair = tuple(sorted((a, b)))
            for fp in perps:
                p, q, r, s = fp.args
                rs_pair = (r, s)
                # Check if perp's second line pair is {A,B}
                if rs_pair != ab_pair:
                    continue
                # Check if mid appears in the first line pair
                if mid == p:
                    other = q
                elif mid == q:
                    other = p
                else:
                    continue
                # other is the point C
                inferred = canonical_cong(other, a, other, b)
                if state.has_fact(inferred):
                    continue
                yield RuleApplication(
                    step=Step(self.name, (fm, fp), inferred),
                )


# ── Isosceles triangle ───────────────────────────────────────────────


class IsoscelesBaseAngleRule(Rule):
    """Cong(A,B,C,D) with shared endpoint → base angle equality.

    Detects ALL shared-endpoint patterns in Cong(a,b,c,d) after
    canonical sorting (a≤b, c≤d):
      a==c: |ab|=|cd| with apex a=c  → ∠abd = ∠dba  (etc.)
      b==c: |ab|=|cd| with apex b=c
      b==d: |ab|=|cd| with apex b=d
      a==d: |ab|=|cd| with apex a=d

    Lean: ``isosceles_base_angle``.
    """
    name = "isosceles_base_angle"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("Cong"):
            if len(f.args) != 4:
                continue
            a, b, c, d = f.args
            # Try all 4 possible shared-endpoint positions
            for apex, p, q in (
                (a, b, d) if a == c else (None, None, None),
                (b, a, d) if b == c else (None, None, None),
                (b, a, c) if b == d else (None, None, None),
                (a, b, c) if a == d else (None, None, None),
            ):
                if apex is None or p == q:
                    continue
                inferred = canonical_eq_angle(apex, p, q, apex, q, p)
                if not state.has_fact(inferred):
                    yield RuleApplication(step=Step(self.name, (f,), inferred))


class CongPerpBisectorRule(Rule):
    """Cong(C,A,C,B) ∧ IsMidpoint(M,A,B) → Perpendicular(C,M,A,B).

    Converse of perp-bisector: if |CA|=|CB| and M=mid(AB), then CM⊥AB.
    Detects all shared-endpoint patterns (a==c, b==c, b==d, a==d).
    Lean: ``cong_perp_bisector``.
    """
    name = "cong_perp_bisector"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        congs = [f for f in state.by_predicate("Cong") if len(f.args) == 4]
        midpoints = [f for f in state.by_predicate("Midpoint") if len(f.args) == 3]
        for fc in congs:
            a, b, c, d = fc.args
            # Extract all (apex, p, q) with apex shared
            candidates = []
            if a == c:
                candidates.append((a, b, d))
            if b == c:
                candidates.append((b, a, d))
            if b == d:
                candidates.append((b, a, c))
            if a == d:
                candidates.append((a, b, c))
            for c_pt, a_pt, b_pt in candidates:
                if a_pt == b_pt:
                    continue
                ab_pair = tuple(sorted((a_pt, b_pt)))
                for fm in midpoints:
                    mid, ma, mb = fm.args
                    if tuple(sorted((ma, mb))) != ab_pair:
                        continue
                    inferred = canonical_perp(c_pt, mid, a_pt, b_pt)
                    if not state.has_fact(inferred):
                        yield RuleApplication(
                            step=Step(self.name, (fc, fm), inferred),
                        )


class ParallelAlternateAngleRule(Rule):
    """Parallel(A,B,C,D) ∧ Collinear(A,X,C) → EqAngle(B,A,X, D,C,X).

    Alternate interior angles: if AB∥CD and line AXC is a transversal,
    then ∠BAX = ∠DCX.  Lean: ``parallel_alternate_angle``.
    """
    name = "parallel_alternate_angle"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        parallels = [f for f in state.by_predicate("Parallel") if len(f.args) == 4]
        collinears = [f for f in state.by_predicate("Collinear") if len(f.args) == 3]
        for fp in parallels:
            a, b, c, d = fp.args
            for fc in collinears:
                pts = set(fc.args)
                # transversal must pass through one point from each parallel line
                inter_ab = {a, b} & pts
                inter_cd = {c, d} & pts
                if len(inter_ab) != 1 or len(inter_cd) != 1:
                    continue
                pa = inter_ab.pop()   # point on line AB
                pc = inter_cd.pop()   # point on line CD
                # The third point on the collinear line is the transversal crossing
                x_set = pts - {pa, pc}
                if len(x_set) != 1:
                    continue
                x = x_set.pop()
                # Other endpoints on parallel lines
                ob = b if pa == a else a
                od = d if pc == c else c
                inferred = canonical_eq_angle(ob, pa, x, od, pc, x)
                if not state.has_fact(inferred):
                    yield RuleApplication(
                        step=Step(self.name, (fp, fc), inferred),
                    )


class CyclicChordAngleRule(Rule):
    """Cyclic(A,B,C,D) → EqAngle(A,B,D, A,C,D).

    Angles subtended by the same chord AD from two other points on the
    circle (different permutation from CyclicEqAngleRule).
    Lean: ``cyclic_chord_angle``.
    """
    name = "cyclic_chord_angle"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("Cyclic"):
            if len(f.args) != 4:
                continue
            a, b, c, d = f.args
            # ∠ABD = ∠ACD (angles subtending chord AD)
            inferred = canonical_eq_angle(a, b, d, a, c, d)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


class MidsegmentCongRule(Rule):
    """Midsegment half-length:
    IsMidpoint(M,A,B) ∧ IsMidpoint(N,A,C) ∧ Cong(B,C,X,Y)
    → Cong(M,N,?,?) is hard; instead:
    IsMidpoint(M,A,B) ∧ IsMidpoint(N,A,C) → SimTri(A,M,N, A,B,C).

    The midsegment creates a similar triangle with ratio 1:2.
    Lean: ``midsegment_sim_tri``.
    """
    name = "midsegment_sim_tri"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        midpoints = [f for f in state.by_predicate("Midpoint") if len(f.args) == 3]
        for i, f1 in enumerate(midpoints):
            m1, a1, b1 = f1.args
            for f2 in midpoints[i + 1:]:
                m2, a2, b2 = f2.args
                shared = None
                other1 = other2 = None
                if a1 == a2:
                    shared, other1, other2 = a1, b1, b2
                elif a1 == b2:
                    shared, other1, other2 = a1, b1, a2
                elif b1 == a2:
                    shared, other1, other2 = b1, a1, b2
                elif b1 == b2:
                    shared, other1, other2 = b1, a1, a2
                if shared is None:
                    continue
                # △(shared, M1, M2) ~ △(shared, other1, other2)
                inferred = canonical_sim_tri(
                    shared, m1, m2, shared, other1, other2,
                )
                if not state.has_fact(inferred):
                    yield RuleApplication(
                        step=Step(self.name, (f1, f2), inferred),
                    )


class SimTriAngleRule(Rule):
    """SimTri(A,B,C,D,E,F) → EqAngle(B,A,C, E,D,F).

    Similar triangles have equal corresponding angles.
    Lean: ``sim_tri_angle``.
    """
    name = "sim_tri_angle"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("SimTri"):
            if len(f.args) != 6:
                continue
            a, b, c, d, e, ff = f.args
            # ∠BAC = ∠EDF
            inferred = canonical_eq_angle(b, a, c, e, d, ff)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


class SimTriCongRule(Rule):
    """SimTri(A,B,C,D,E,F) ∧ Cong(A,B,D,E) → Cong(A,C,D,F).

    If similar triangles have one pair of sides congruent,
    then corresponding sides are congruent (i.e. they're congruent triangles).
    Lean: ``sim_tri_cong``.
    """
    name = "sim_tri_cong"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        sim_tris = [f for f in state.by_predicate("SimTri") if len(f.args) == 6]
        congs = [f for f in state.by_predicate("Cong") if len(f.args) == 4]
        for fs in sim_tris:
            a, b, c, d, e, ff = fs.args
            for fc in congs:
                ca, cb, cc, cd = fc.args
                # Check if Cong matches AB=DE
                if tuple(sorted((ca, cb))) == tuple(sorted((a, b))) and \
                   tuple(sorted((cc, cd))) == tuple(sorted((d, e))):
                    inferred = canonical_cong(a, c, d, ff)
                    if not state.has_fact(inferred):
                        yield RuleApplication(
                            step=Step(self.name, (fs, fc), inferred),
                        )
                # Check if Cong matches BC=EF
                elif tuple(sorted((ca, cb))) == tuple(sorted((b, c))) and \
                     tuple(sorted((cc, cd))) == tuple(sorted((e, ff))):
                    inferred = canonical_cong(a, b, d, e)
                    if not state.has_fact(inferred):
                        yield RuleApplication(
                            step=Step(self.name, (fs, fc), inferred),
                        )


# ═══════════════════════════════════════════════════════════════════════
# NEW PREDICATES — additional rules for the 14 newly added geometric
# relations.  Each rule encodes a single inference schema.
# ═══════════════════════════════════════════════════════════════════════

# ── CongTri (triangle congruence) ────────────────────────────────────


class CongTriSideRule(Rule):
    """CongTri(A,B,C,D,E,F) → all three corresponding sides.

    Yields Cong(A,B,D,E), Cong(B,C,E,F), Cong(A,C,D,F)."""
    name = "congtri_side"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("CongTri"):
            if len(f.args) != 6:
                continue
            a, b, c, d, e, ff = f.args
            for p, q, r, s in ((a,b,d,e), (b,c,e,ff), (a,c,d,ff)):
                inferred = canonical_cong(p, q, r, s)
                if not state.has_fact(inferred):
                    yield RuleApplication(step=Step(self.name, (f,), inferred))


class CongTriAngleRule(Rule):
    """CongTri(A,B,C,D,E,F) → all three corresponding angles.

    Yields EqAngle(B,A,C, E,D,F), EqAngle(A,B,C, D,E,F), EqAngle(A,C,B, D,F,E)."""
    name = "congtri_angle"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("CongTri"):
            if len(f.args) != 6:
                continue
            a, b, c, d, e, ff = f.args
            # ∠A=∠D, ∠B=∠E, ∠C=∠F
            for v1, s1, s2, v2, t1, t2 in (
                (a, b, c,  d, e, ff),   # ∠BAC = ∠EDF
                (b, a, c,  e, d, ff),   # ∠ABC = ∠DEF
                (c, a, b,  ff, d, e),   # ∠ACB = ∠DFE
            ):
                inferred = canonical_eq_angle(s1, v1, s2, t1, v2, t2)
                if not state.has_fact(inferred):
                    yield RuleApplication(step=Step(self.name, (f,), inferred))


class CongTriFromSimCongRule(Rule):
    """SimTri(A,B,C,D,E,F) ∧ Cong(A,B,D,E) → CongTri(A,B,C,D,E,F).

    Similar triangles with one pair of corresponding sides congruent
    are congruent.  Lean: ``congtri_from_sim_cong``.
    """
    name = "congtri_from_sim_cong"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for fs in state.by_predicate("SimTri"):
            if len(fs.args) != 6:
                continue
            a, b, c, d, e, ff = fs.args
            target_cong = canonical_cong(a, b, d, e)
            if state.has_fact(target_cong):
                inferred = canonical_congtri(a, b, c, d, e, ff)
                if not state.has_fact(inferred):
                    yield RuleApplication(
                        step=Step(self.name, (fs, target_cong), inferred))


class CongTriEqAreaRule(Rule):
    """CongTri(A,B,C,D,E,F) → EqArea(A,B,C,D,E,F)."""
    name = "congtri_eqarea"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("CongTri"):
            if len(f.args) != 6:
                continue
            a, b, c, d, e, ff = f.args
            inferred = canonical_eqarea(a, b, c, d, e, ff)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


# ── Tangent (line tangent to circle) ─────────────────────────────────


class TangentPerpRadiusRule(Rule):
    """Tangent(A,B,O,P) → Perpendicular(O,P,A,B).

    The tangent is perpendicular to the radius at the point of tangency.
    """
    name = "tangent_perp_radius"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("Tangent"):
            if len(f.args) != 4:
                continue
            a, b, o, p = f.args
            inferred = canonical_perp(o, p, a, b)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


class TangentOnCircleRule(Rule):
    """Tangent(A,B,O,P) → OnCircle(O,P).

    The tangent point lies on the circle.
    """
    name = "tangent_oncircle"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("Tangent"):
            if len(f.args) != 4:
                continue
            _, _, o, p = f.args
            inferred = canonical_circle(o, p)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


# ── EqRatio (proportional segments) ──────────────────────────────────


class EqRatioFromSimTriRule(Rule):
    """SimTri(A,B,C,D,E,F) → EqRatio(A,B,D,E,A,C,D,F).

    Corresponding sides of similar triangles are proportional.
    """
    name = "eqratio_from_simtri"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("SimTri"):
            if len(f.args) != 6:
                continue
            a, b, c, d, e, ff = f.args
            inferred = canonical_eqratio(a, b, d, e, a, c, d, ff)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


class EqRatioSymRule(Rule):
    """EqRatio(A,B,C,D,E,F,G,H) → EqRatio(E,F,G,H,A,B,C,D)."""
    name = "eqratio_sym"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("EqRatio"):
            if len(f.args) != 8:
                continue
            a, b, c, d, e, ff, g, h = f.args
            inferred = canonical_eqratio(e, ff, g, h, a, b, c, d)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


class EqRatioTransRule(Rule):
    """EqRatio(A,B,C,D,E,F,G,H) ∧ EqRatio(E,F,G,H,I,J,K,L)
    → EqRatio(A,B,C,D,I,J,K,L).

    Transitivity of proportional segments.
    """
    name = "eqratio_trans"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        ratios = [f for f in state.by_predicate("EqRatio") if len(f.args) == 8]
        for f1 in ratios:
            tail1 = f1.args[4:]   # (E,F,G,H)
            for f2 in ratios:
                if f2 is f1:
                    continue
                head2 = f2.args[:4]  # (E,F,G,H)
                if tail1 != head2:
                    continue
                inferred = canonical_eqratio(
                    *f1.args[:4], *f2.args[4:])
                if not state.has_fact(inferred):
                    yield RuleApplication(
                        step=Step(self.name, (f1, f2), inferred))


# ── Between (ordered collinearity) ──────────────────────────────────


class BetweenCollinearRule(Rule):
    """Between(A,B,C) → Collinear(A,B,C)."""
    name = "between_collinear"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("Between"):
            if len(f.args) != 3:
                continue
            a, b, c = f.args
            inferred = canonical_collinear(a, b, c)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


class MidpointBetweenRule(Rule):
    """IsMidpoint(M,A,B) → Between(A,M,B)."""
    name = "midpoint_between"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("Midpoint"):
            if len(f.args) != 3:
                continue
            m, a, b = f.args
            inferred = canonical_between(a, m, b)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


# ── AngleBisect ──────────────────────────────────────────────────────


class AngleBisectEqAngleRule(Rule):
    """AngleBisect(A,P,B,C) → EqAngle(B,A,P, P,A,C).

    The bisector divides the angle into two equal parts.
    """
    name = "angle_bisect_eq_angle"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("AngleBisect"):
            if len(f.args) != 4:
                continue
            a, p, b, c = f.args
            inferred = canonical_eq_angle(b, a, p, p, a, c)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


class AngleBisectEqRatioRule(Rule):
    """AngleBisect(A,P,B,C) ∧ Between(B,P,C) → EqRatio(B,P,P,C,A,B,A,C).

    Angle bisector theorem: BP/PC = AB/AC.
    """
    name = "angle_bisect_eqratio"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for fab in state.by_predicate("AngleBisect"):
            if len(fab.args) != 4:
                continue
            a, p, b, c = fab.args
            target_bet = canonical_between(b, p, c)
            if state.has_fact(target_bet):
                inferred = canonical_eqratio(b, p, p, c, a, b, a, c)
                if not state.has_fact(inferred):
                    yield RuleApplication(
                        step=Step(self.name, (fab, target_bet), inferred))


# ── Concurrent ───────────────────────────────────────────────────────


class MediansConcurrentRule(Rule):
    """Medians are concurrent (centroid exists):
    IsMidpoint(D,B,C) ∧ IsMidpoint(E,A,C) ∧ IsMidpoint(F,A,B)
    → Concurrent(A,D, B,E, C,F).
    """
    name = "medians_concurrent"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        midpoints = [f for f in state.by_predicate("Midpoint")
                     if len(f.args) == 3]
        # We need 3 midpoints whose 'endpoints' partition into a triangle
        # M1 = mid(B,C), M2 = mid(A,C), M3 = mid(A,B)
        for i, f1 in enumerate(midpoints):
            m1, e1a, e1b = f1.args
            s1 = frozenset((e1a, e1b))
            for j, f2 in enumerate(midpoints):
                if j <= i:
                    continue
                m2, e2a, e2b = f2.args
                s2 = frozenset((e2a, e2b))
                shared_12 = s1 & s2
                if len(shared_12) != 1:
                    continue
                for k, f3 in enumerate(midpoints):
                    if k <= j:
                        continue
                    m3, e3a, e3b = f3.args
                    s3 = frozenset((e3a, e3b))
                    # All three vertex pairs should form a triangle
                    all_pts = s1 | s2 | s3
                    if len(all_pts) != 3:
                        continue
                    shared_13 = s1 & s3
                    shared_23 = s2 & s3
                    if len(shared_13) != 1 or len(shared_23) != 1:
                        continue
                    # Triangle vertices
                    verts = list(all_pts)
                    A, B, C = verts[0], verts[1], verts[2]
                    # Find which midpoint goes with which opposite side
                    mid_of = {}
                    for fm in (f1, f2, f3):
                        mm, ea, eb = fm.args
                        mid_of[frozenset((ea, eb))] = mm
                    # Median from A to mid(B,C), etc.
                    d = mid_of.get(frozenset((B, C)))
                    e = mid_of.get(frozenset((A, C)))
                    f_pt = mid_of.get(frozenset((A, B)))
                    if d is None or e is None or f_pt is None:
                        continue
                    inferred = canonical_concurrent(A, d, B, e, C, f_pt)
                    if not state.has_fact(inferred):
                        yield RuleApplication(
                            step=Step(self.name, (f1, f2, f3), inferred))


# ── Circumcenter ─────────────────────────────────────────────────────


class CircumcenterCongABRule(Rule):
    """Circumcenter(O,A,B,C) → Cong(O,A,O,B)."""
    name = "circumcenter_cong_ab"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("Circumcenter"):
            if len(f.args) != 4:
                continue
            o, a, b, c = f.args
            inferred = canonical_cong(o, a, o, b)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


class CircumcenterCongBCRule(Rule):
    """Circumcenter(O,A,B,C) → Cong(O,B,O,C)."""
    name = "circumcenter_cong_bc"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("Circumcenter"):
            if len(f.args) != 4:
                continue
            o, a, b, c = f.args
            inferred = canonical_cong(o, b, o, c)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


class CircumcenterOnCircleRule(Rule):
    """Circumcenter(O,A,B,C) → OnCircle(O,A)."""
    name = "circumcenter_oncircle"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("Circumcenter"):
            if len(f.args) != 4:
                continue
            o, a, _b, _c = f.args
            inferred = canonical_circle(o, a)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


# ── EqDist (equidistant) ─────────────────────────────────────────────


class EqDistFromCongRule(Rule):
    """Cong(P,A,P,B) → EqDist(P,A,B)  when two segments share endpoint P."""
    name = "eqdist_from_cong"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("Cong"):
            if len(f.args) != 4:
                continue
            a, b, c, d = f.args
            # After canonical_cong: a≤b, c≤d
            if a == c and b != d:
                inf = canonical_eqdist(a, b, d)
                if not state.has_fact(inf):
                    yield RuleApplication(step=Step(self.name, (f,), inf))
            if b == d and a != c:
                inf = canonical_eqdist(b, a, c)
                if not state.has_fact(inf):
                    yield RuleApplication(step=Step(self.name, (f,), inf))
            if a == d and b != c:
                inf = canonical_eqdist(a, b, c)
                if not state.has_fact(inf):
                    yield RuleApplication(step=Step(self.name, (f,), inf))
            if b == c and a != d:
                inf = canonical_eqdist(b, a, d)
                if not state.has_fact(inf):
                    yield RuleApplication(step=Step(self.name, (f,), inf))


class EqDistToCongRule(Rule):
    """EqDist(P,A,B) → Cong(P,A,P,B)."""
    name = "eqdist_to_cong"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("EqDist"):
            if len(f.args) != 3:
                continue
            p, a, b = f.args
            inferred = canonical_cong(p, a, p, b)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


# ── EqArea (equal triangle areas) ────────────────────────────────────


class EqAreaSymRule(Rule):
    """EqArea(A,B,C,D,E,F) → EqArea(D,E,F,A,B,C)."""
    name = "eqarea_sym"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("EqArea"):
            if len(f.args) != 6:
                continue
            a, b, c, d, e, ff = f.args
            inferred = canonical_eqarea(d, e, ff, a, b, c)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


# ── Harmonic (harmonic range) ────────────────────────────────────────


class HarmonicSwapRule(Rule):
    """Harmonic(A,B,C,D) → Harmonic(B,A,D,C).

    (A,B;C,D) = −1  ⟺  (B,A;D,C) = −1.
    """
    name = "harmonic_swap"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("Harmonic"):
            if len(f.args) != 4:
                continue
            a, b, c, d = f.args
            inferred = canonical_harmonic(b, a, d, c)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


class HarmonicCollinearRule(Rule):
    """Harmonic(A,B,C,D) → Collinear(A,C,D).

    All four harmonic points are collinear; we emit one triple.
    """
    name = "harmonic_collinear"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("Harmonic"):
            if len(f.args) != 4:
                continue
            a, _b, c, d = f.args
            inferred = canonical_collinear(a, c, d)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


# ── PolePolar ────────────────────────────────────────────────────────


class PolePolarPerpRule(Rule):
    """PolePolar(P,A,B,O) → Perpendicular(O,P,A,B).

    The line from centre O to pole P is perpendicular to the polar AB.
    """
    name = "pole_polar_perp"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("PolePolar"):
            if len(f.args) != 4:
                continue
            p, a, b, o = f.args
            inferred = canonical_perp(o, p, a, b)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


class PolePolarTangentRule(Rule):
    """PolePolar(P,A,B,O) ∧ OnCircle(O,A) → Tangent(P,A,O,A).

    If A is on the circle and on the polar of P, then line PA is tangent
    at A.
    """
    name = "pole_polar_tangent"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("PolePolar"):
            if len(f.args) != 4:
                continue
            p, a, b, o = f.args
            # Check if either point on the polar (a or b) is on the circle
            for pt in (a, b):
                oc = canonical_circle(o, pt)
                if state.has_fact(oc):
                    inferred = canonical_tangent(p, pt, o, pt)
                    if not state.has_fact(inferred):
                        yield RuleApplication(
                            step=Step(self.name, (f, oc), inferred))


# ── InvImage (circle inversion) ──────────────────────────────────────


class InversionCollinearRule(Rule):
    """InvImage(P',P,O,A) → Collinear(O,P,P').

    O, P, P' are collinear under inversion.
    """
    name = "inversion_collinear"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("InvImage"):
            if len(f.args) != 4:
                continue
            pp, p, o, _a = f.args
            inferred = canonical_collinear(o, p, pp)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


class InversionCircleFixedRule(Rule):
    """InvImage(P',P,O,A) ∧ OnCircle(O,P) → OnCircle(O,P').

    Points on the circle of inversion are fixed.
    """
    name = "inversion_circle_fixed"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("InvImage"):
            if len(f.args) != 4:
                continue
            pp, p, o, _a = f.args
            oc = canonical_circle(o, p)
            if state.has_fact(oc):
                inferred = canonical_circle(o, pp)
                if not state.has_fact(inferred):
                    yield RuleApplication(
                        step=Step(self.name, (f, oc), inferred))


# ── EqCrossRatio ─────────────────────────────────────────────────────


class EqCrossRatioSymRule(Rule):
    """EqCrossRatio(A,B,C,D,E,F,G,H) → EqCrossRatio(E,F,G,H,A,B,C,D)."""
    name = "cross_ratio_sym"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("EqCrossRatio"):
            if len(f.args) != 8:
                continue
            a, b, c, d, e, ff, g, h = f.args
            inferred = canonical_eq_cross_ratio(e, ff, g, h, a, b, c, d)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


class EqCrossRatioFromHarmonicRule(Rule):
    """Harmonic(A,B,C,D) ∧ Harmonic(E,F,G,H) → EqCrossRatio(A,B,C,D,E,F,G,H).

    Two harmonic ranges have equal cross-ratio (both = −1).
    """
    name = "cross_ratio_from_harmonic"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        harmonics = [f for f in state.by_predicate("Harmonic")
                     if len(f.args) == 4]
        for i, f1 in enumerate(harmonics):
            for f2 in harmonics[i + 1:]:
                inferred = canonical_eq_cross_ratio(
                    *f1.args, *f2.args)
                if not state.has_fact(inferred):
                    yield RuleApplication(
                        step=Step(self.name, (f1, f2), inferred))


# ── RadicalAxis ──────────────────────────────────────────────────────


class RadicalAxisPerpRule(Rule):
    """RadicalAxis(A,B,O1,O2) → Perpendicular(A,B,O1,O2).

    The radical axis is perpendicular to the line of centres.
    """
    name = "radical_axis_perp"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("RadicalAxis"):
            if len(f.args) != 4:
                continue
            a, b, o1, o2 = f.args
            inferred = canonical_perp(a, b, o1, o2)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


# ══════════════════════════════════════════════════════════════════════
#  Converse / production rules for input-only predicates
# ══════════════════════════════════════════════════════════════════════


class MidpointFromCongBetweenRule(Rule):
    """Cong(A,M,M,B) ∧ Between(A,M,B) → Midpoint(M,A,B).

    If M lies between A and B and |AM| = |MB|, then M is the midpoint.
    """
    name = "midpoint_from_cong_between"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for fb in state.by_predicate("Between"):
            if len(fb.args) != 3:
                continue
            a, m, b = fb.args
            # Check Cong(A,M,M,B)
            fc = canonical_cong(a, m, m, b)
            if not state.has_fact(fc):
                continue
            inferred = canonical_midpoint(m, a, b)
            if not state.has_fact(inferred):
                yield RuleApplication(
                    step=Step(self.name, (fb, fc), inferred))


class MidpointFromCollinearEqDistRule(Rule):
    """Collinear(A,M,B) ∧ EqDist(M,A,B) → Midpoint(M,A,B).

    Provides an additional midpoint production path independent from
    explicit Between(A,M,B), improving converse robustness.
    """
    name = "midpoint_from_collinear_eqdist"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        colls = [f for f in state.by_predicate("Collinear") if len(f.args) == 3]
        for fc in colls:
            a, m, b = fc.args
            fe = canonical_eqdist(m, a, b)
            if not state.has_fact(fe):
                continue
            inferred = canonical_midpoint(m, a, b)
            if not state.has_fact(inferred):
                yield RuleApplication(
                    step=Step(self.name, (fc, fe), inferred))


class CyclicFromEqAngleRule(Rule):
    """EqAngle(B,A,C,B,D,C) → Cyclic(A,B,C,D).

    Converse of inscribed angle theorem: if ∠BAC = ∠BDC then
    A, B, C, D are concyclic.
    """
    name = "cyclic_from_eq_angle"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("EqAngle"):
            if len(f.args) != 6:
                continue
            p, q, r, s, t, u = f.args
            # Pattern: EqAngle(B,A,C, B,D,C) — same first and third in each triple
            if p != s or r != u:
                continue
            # p=B, q=A, r=C, s=B, t=D, u=C
            b, a, c, d = p, q, r, t
            if len({a, b, c, d}) != 4:
                continue
            inferred = canonical_cyclic(a, b, c, d)
            if not state.has_fact(inferred):
                yield RuleApplication(
                    step=Step(self.name, (f,), inferred))


class CyclicFromChordEqAngleRule(Rule):
    """EqAngle(A,B,D,A,C,D) → Cyclic(A,B,C,D).

    Converse of cyclic_chord_angle, adds a second independent
    EqAngle→Cyclic production path.
    """
    name = "cyclic_from_chord_eq_angle"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("EqAngle"):
            if len(f.args) != 6:
                continue
            a1, b1, d1, a2, c2, d2 = f.args
            if a1 != a2 or d1 != d2:
                continue
            a, b, c, d = a1, b1, c2, d1
            if len({a, b, c, d}) != 4:
                continue
            inferred = canonical_cyclic(a, b, c, d)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


class CircumcenterFromEqDistRule(Rule):
    """EqDist(O,A,B) ∧ EqDist(O,B,C) → Circumcenter(O,A,B,C).

    If O is equidistant from A,B and from B,C, then O is the
    circumcentre of △ABC.
    """
    name = "circumcenter_from_eqdist"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        eqs = [f for f in state.by_predicate("EqDist") if len(f.args) == 3]
        for i, f1 in enumerate(eqs):
            o1, a1, b1 = f1.args
            for f2 in eqs[i + 1:]:
                o2, a2, b2 = f2.args
                if o1 != o2:
                    continue
                o = o1
                pts = {a1, b1, a2, b2}
                shared = {a1, b1} & {a2, b2}
                if len(pts) != 3 or len(shared) != 1:
                    continue
                tri = sorted(pts)
                inferred = canonical_circumcenter(o, tri[0], tri[1], tri[2])
                if not state.has_fact(inferred):
                    yield RuleApplication(
                        step=Step(self.name, (f1, f2), inferred))


class CircumcenterFromOnCircleTripleRule(Rule):
    """OnCircle(O,A) ∧ OnCircle(O,B) ∧ OnCircle(O,C) → Circumcenter(O,A,B,C)."""
    name = "circumcenter_from_oncircle"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        circles = [f for f in state.by_predicate("OnCircle") if len(f.args) == 2]
        by_center: dict[str, set[str]] = {}
        fact_index: dict[tuple[str, str], Fact] = {}
        for f in circles:
            o, p = f.args
            by_center.setdefault(o, set()).add(p)
            fact_index[(o, p)] = f

        for o, pts in by_center.items():
            pts_list = sorted(pts)
            if len(pts_list) < 3:
                continue
            for i in range(len(pts_list)):
                for j in range(i + 1, len(pts_list)):
                    for k in range(j + 1, len(pts_list)):
                        a, b, c = pts_list[i], pts_list[j], pts_list[k]
                        inferred = canonical_circumcenter(o, a, b, c)
                        if not state.has_fact(inferred):
                            yield RuleApplication(
                                step=Step(
                                    self.name,
                                    (fact_index[(o, a)], fact_index[(o, b)], fact_index[(o, c)]),
                                    inferred,
                                ),
                            )


class AngleBisectFromEqAngleRule(Rule):
    """EqAngle(B,A,P,P,A,C) → AngleBisect(A,P,B,C).

    If ∠BAP = ∠PAC, i.e. ray AP bisects ∠BAC, then AP is the
    angle bisector.
    """
    name = "angle_bisect_from_eq_angle"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("EqAngle"):
            if len(f.args) != 6:
                continue
            p, q, r, s, t, u = f.args
            # Pattern: EqAngle(B,A,P, P,A,C) — vertex A shared, P shared
            if q != t or r != s:
                continue
            # q=t=A (vertex), r=s=P (bisector point), p=B, u=C
            a, bp, b, c = q, r, p, u
            if len({a, bp, b, c}) != 4:
                continue
            inferred = canonical_angle_bisect(a, bp, b, c)
            if not state.has_fact(inferred):
                yield RuleApplication(
                    step=Step(self.name, (f,), inferred))


class AngleBisectFromEqRatioBetweenRule(Rule):
    """EqRatio(B,P,P,C,A,B,A,C) ∧ Between(B,P,C) → AngleBisect(A,P,B,C).

    Converse form of the angle bisector theorem, adds an independent
    production path for AngleBisect.
    """
    name = "angle_bisect_from_eqratio_between"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        ratios = [f for f in state.by_predicate("EqRatio") if len(f.args) == 8]
        for fr in ratios:
            b, p, p2, c, a, b2, a2, c2 = fr.args
            if p != p2 or a != a2 or b != b2 or c != c2:
                continue
            fb = canonical_between(b, p, c)
            if not state.has_fact(fb):
                continue
            if len({a, b, c, p}) != 4:
                continue
            inferred = canonical_angle_bisect(a, p, b, c)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (fr, fb), inferred))


class PolePolarFromTangentsRule(Rule):
    """Tangent(P,A,O,A) ∧ Tangent(P,B,O,B) → PolePolar(P,A,B,O).

    If PA and PB are both tangent to circle O at A and B respectively,
    then line AB is the polar of pole P w.r.t. circle O.
    """
    name = "pole_polar_from_tangents"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        tangents = [f for f in state.by_predicate("Tangent") if len(f.args) == 4]
        for i, f1 in enumerate(tangents):
            a1, b1, o1, p1 = f1.args
            for f2 in tangents[i + 1:]:
                a2, b2, o2, p2 = f2.args
                if o1 != o2:
                    continue
                o = o1
                # Both tangent from same external point P
                # Tangent is canonical_tangent(P,X,O,X) → sorted(P,X), O, X
                line1 = {a1, b1}
                line2 = {a2, b2}
                shared = line1 & line2
                if len(shared) != 1:
                    continue
                pole = shared.pop()
                pt1 = (line1 - {pole}).pop()  # tangent point 1
                pt2 = (line2 - {pole}).pop()  # tangent point 2
                # Verify tangent points: p1==pt1 and p2==pt2
                if p1 not in {pt1, pt2} or p2 not in {pt1, pt2}:
                    continue
                inferred = canonical_pole_polar(pole, pt1, pt2, o)
                if not state.has_fact(inferred):
                    yield RuleApplication(
                        step=Step(self.name, (f1, f2), inferred))


class PolePolarFromPerpOnCircleRule(Rule):
    """Perpendicular(O,P,A,B) ∧ OnCircle(O,A) ∧ OnCircle(O,B) → PolePolar(P,A,B,O).

    Adds a second production path for PolePolar and bridges
    PROJECTIVE-CIRCLE-LINE families.
    """
    name = "pole_polar_from_perp_oncircle"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        perps = [f for f in state.by_predicate("Perpendicular") if len(f.args) == 4]
        for fp in perps:
            o, p, a, b = fp.args
            oa = canonical_circle(o, a)
            ob = canonical_circle(o, b)
            if not state.has_fact(oa) or not state.has_fact(ob):
                continue
            inferred = canonical_pole_polar(p, a, b, o)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (fp, oa, ob), inferred))


class RadicalAxisFromCommonPointsRule(Rule):
    """OnCircle(O1,A) ∧ OnCircle(O2,A) ∧ OnCircle(O1,B) ∧ OnCircle(O2,B)
       → RadicalAxis(A,B,O1,O2).

    If A and B both lie on circles O1 and O2 (the common chord),
    then line AB is the radical axis.
    """
    name = "radical_axis_from_common_points"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        circles = [f for f in state.by_predicate("OnCircle") if len(f.args) == 2]
        # Group by point: which circles contain each point?
        from collections import defaultdict
        point_circles: dict[str, set] = defaultdict(set)
        point_facts: dict[tuple, Fact] = {}
        for f in circles:
            o, pt = f.args
            point_circles[pt].add(o)
            point_facts[(o, pt)] = f
        # Find pairs of points on the same two circles
        pts = [p for p, cs in point_circles.items() if len(cs) >= 2]
        for i, pa in enumerate(pts):
            for pb in pts[i + 1:]:
                common = point_circles[pa] & point_circles[pb]
                if len(common) < 2:
                    continue
                centers = sorted(common)
                for ci in range(len(centers)):
                    for cj in range(ci + 1, len(centers)):
                        o1, o2 = centers[ci], centers[cj]
                        inferred = canonical_radical_axis(pa, pb, o1, o2)
                        if not state.has_fact(inferred):
                            f1 = point_facts[(o1, pa)]
                            f2 = point_facts[(o2, pa)]
                            f3 = point_facts[(o1, pb)]
                            f4 = point_facts[(o2, pb)]
                            yield RuleApplication(
                                step=Step(self.name,
                                          (f1, f2, f3, f4), inferred))


class ConcurrentFromAngleBisectorsRule(Rule):
    """Three angle bisectors imply concurrency (incenter pattern).

    AngleBisect(A,P,B,C) ∧ AngleBisect(B,Q,C,A) ∧ AngleBisect(C,R,A,B)
    → Concurrent(A,P,B,Q,C,R)
    """
    name = "angle_bisectors_concurrent"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        bis = [f for f in state.by_predicate("AngleBisect") if len(f.args) == 4]
        for f1 in bis:
            a, p, b, c = f1.args
            # Search explicitly for the other two bisectors.
            for f2 in bis:
                if f2 is f1:
                    continue
                b2, q, c2, a2 = f2.args
                if (b2, c2, a2) != (b, c, a):
                    continue
                for f3 in bis:
                    if f3 is f1 or f3 is f2:
                        continue
                    c3, r, a3, b3 = f3.args
                    if (c3, a3, b3) != (c, a, b):
                        continue
                    inferred = canonical_concurrent(a, p, b, q, c, r)
                    if not state.has_fact(inferred):
                        yield RuleApplication(
                            step=Step(self.name, (f1, f2, f3), inferred),
                        )


class InvImageFromCollinearCongRule(Rule):
    """Collinear(O,P,P') ∧ OnCircle(O,A) ∧ EqRatio(O,P,O,A,O,A,O,P')
       → InvImage(P',P,O,A).

    If O,P,P' are collinear and |OP|·|OP'| = |OA|², then P' is the
    inversion of P w.r.t. circle (O,|OA|).

    Special easy case: if P is on circle (O,A), then P maps to itself.
    We handle: Collinear(O,P,P') ∧ OnCircle(O,P) ∧ Cong(O,P,O,P')
               → InvImage(P',P,O,P)  (self-inversion).
    """
    name = "inv_image_from_self"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for fc in state.by_predicate("Collinear"):
            if len(fc.args) != 3:
                continue
            pts = fc.args
            # Try each point as center O
            for idx in range(3):
                o = pts[idx]
                others = [pts[j] for j in range(3) if j != idx]
                p, pp = others[0], others[1]
                # Check OnCircle(O, P) — P is on the inversion circle
                oc = canonical_circle(o, p)
                if not state.has_fact(oc):
                    continue
                # Check Cong(O,P,O,P') — same distance → self-inversion
                cg = canonical_cong(o, p, o, pp)
                if not state.has_fact(cg):
                    continue
                inferred = canonical_inv_image(pp, p, o, p)
                if not state.has_fact(inferred):
                    yield RuleApplication(
                        step=Step(self.name, (fc, oc, cg), inferred))


class InvImageFromOnCircleFixedPointRule(Rule):
    """OnCircle(O,P) → InvImage(P,P,O,P).

    A point on the inversion circle is fixed under inversion.
    This provides an additional independent producer for InvImage.
    """
    name = "inv_image_from_oncircle_fixed"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("OnCircle"):
            if len(f.args) != 2:
                continue
            o, p = f.args
            inferred = canonical_inv_image(p, p, o, p)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


class CongTriFromThreeCongRule(Rule):
    """Cong(AB,DE) ∧ Cong(BC,EF) ∧ Cong(AC,DF) → CongTri(ABC,DEF)."""
    name = "congtri_from_three_cong"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        congs = [f for f in state.by_predicate("Cong") if len(f.args) == 4]
        by_pair: dict[tuple[str, str], list[Fact]] = {}
        for f in congs:
            a, b, c, d = f.args
            by_pair.setdefault(tuple(sorted((a, b))), []).append(f)
            by_pair.setdefault(tuple(sorted((c, d))), []).append(f)

        for f1 in congs:
            a, b, d, e = f1.args
            # Seek BC=EF and AC=DF
            key_bc = tuple(sorted((b,)))
            _ = key_bc  # avoid lint-style noise in dynamic pattern search
            for f2 in congs:
                if f2 is f1:
                    continue
                b2, c, e2, ff = f2.args
                if tuple(sorted((b2, e2))) != tuple(sorted((b, e))):
                    continue
                for f3 in congs:
                    if f3 is f1 or f3 is f2:
                        continue
                    a3, c3, d3, f3p = f3.args
                    if tuple(sorted((a3, d3))) != tuple(sorted((a, d))):
                        continue
                    if c3 != c or f3p != ff:
                        continue
                    inferred = canonical_congtri(a, b, c, d, e, ff)
                    if not state.has_fact(inferred):
                        yield RuleApplication(step=Step(self.name, (f1, f2, f3), inferred))


class EqDistFromIsoscelesAngleRule(Rule):
    """EqAngle(B,A,C,B,C,A) → EqDist(B,A,C).

    If base angles at A and C from apex B are equal, infer BA = BC.
    Adds an independent producer for EqDist.
    """
    name = "eqdist_from_isosceles_angle"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("EqAngle"):
            if len(f.args) != 6:
                continue
            s1, v1, s2, t1, v2, t2 = f.args
            # Pattern EqAngle(B,A,C, B,C,A)
            if s1 != t1 or v1 != t2 or s2 != v2:
                continue
            b, a, c = s1, v1, s2
            if len({a, b, c}) != 3:
                continue
            inferred = canonical_eqdist(b, a, c)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f,), inferred))


class HarmonicFromCrossRatioRule(Rule):
    """EqCrossRatio(A,B,C,D,E,F,G,H) ∧ Harmonic(E,F,G,H) → Harmonic(A,B,C,D)."""
    name = "harmonic_from_cross_ratio"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for f in state.by_predicate("EqCrossRatio"):
            if len(f.args) != 8:
                continue
            a, b, c, d, e, ff, g, h = f.args
            fh = canonical_harmonic(e, ff, g, h)
            if not state.has_fact(fh):
                continue
            inferred = canonical_harmonic(a, b, c, d)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (f, fh), inferred))


class CircleMetricAngleBridgeRule(Rule):
    """Circumcenter(O,A,B,C) ∧ Cong(A,B,A,C) → EqAngle(B,A,C,B,C,A).

    Explicit bridge across CIRCLE + METRIC -> ANGLE.
    """
    name = "circle_metric_angle_bridge"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for fcc in state.by_predicate("Circumcenter"):
            if len(fcc.args) != 4:
                continue
            _o, a, b, c = fcc.args
            fcg = canonical_cong(a, b, a, c)
            if not state.has_fact(fcg):
                continue
            inferred = canonical_eq_angle(b, a, c, b, c, a)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (fcc, fcg), inferred))


class ProjectiveCircleLineBridgeRule(Rule):
    """PolePolar(P,A,B,O) ∧ OnCircle(O,A) → Perpendicular(P,A,O,A).

    Explicit bridge across PROJECTIVE + CIRCLE -> LINE.
    """
    name = "projective_circle_line_bridge"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        for fpp in state.by_predicate("PolePolar"):
            if len(fpp.args) != 4:
                continue
            p, a, _b, o = fpp.args
            fo = canonical_circle(o, a)
            if not state.has_fact(fo):
                continue
            inferred = canonical_perp(p, a, o, a)
            if not state.has_fact(inferred):
                yield RuleApplication(step=Step(self.name, (fpp, fo), inferred))


class SimilarityMetricConcurrencyBridgeRule(Rule):
    """SimTri(A,B,C,D,E,F) ∧ Cong(B,C,E,F) ∧ Midpoint(M,B,C)
    → Concurrent(A,M,D,M,B,E).

    Explicit bridge across SIMILARITY + METRIC (+ MIDPOINT) -> CONCURRENCY.
    """
    name = "similarity_metric_concurrency_bridge"

    def apply(self, state: GeoState) -> Iterable[RuleApplication]:
        sims = [f for f in state.by_predicate("SimTri") if len(f.args) == 6]
        mids = [f for f in state.by_predicate("Midpoint") if len(f.args) == 3]
        for fs in sims:
            a, b, c, d, e, ff = fs.args
            fc = canonical_cong(b, c, e, ff)
            if not state.has_fact(fc):
                continue
            for fm in mids:
                m, x, y = fm.args
                if tuple(sorted((x, y))) != tuple(sorted((b, c))):
                    continue
                inferred = canonical_concurrent(a, m, d, m, b, e)
                if not state.has_fact(inferred):
                    yield RuleApplication(step=Step(self.name, (fs, fc, fm), inferred))


# ── Default rule set ─────────────────────────────────────────────────


def default_rules() -> List[Rule]:
    """Return all currently implemented deduction rules."""
    return [
        ParallelSymmetryRule(),
        ParallelTransitivityRule(),
        PerpSymmetryRule(),
        ParallelPerpTransRule(),
        # Collinear (perm/cycle rules removed — canonical sort makes them no-ops)
        # Midpoint
        MidpointCollinearRule(),
        MidpointCongRule(),
        MidsegmentParallelRule(),
        # Congruence
        CongSymmRule(),
        CongTransRule(),
        # Angle equality
        EqAngleSymmRule(),
        EqAngleTransRule(),
        # Cyclic (CyclicPermRule removed — canonical sort makes it a no-op)
        CyclicEqAngleRule(),
        # Cross-domain
        PerpBisectorCongRule(),
        # Triangle / circle rules
        IsoscelesBaseAngleRule(),
        CongPerpBisectorRule(),
        ParallelAlternateAngleRule(),
        CyclicChordAngleRule(),
        MidsegmentCongRule(),     # midsegment → SimTri
        SimTriAngleRule(),
        SimTriCongRule(),
        # ── NEW RULES ──
        # CongTri
        CongTriSideRule(),
        CongTriAngleRule(),
        CongTriFromSimCongRule(),
        CongTriEqAreaRule(),
        # Tangent
        TangentPerpRadiusRule(),
        TangentOnCircleRule(),
        # EqRatio
        EqRatioFromSimTriRule(),
        EqRatioSymRule(),
        EqRatioTransRule(),
        # Between
        BetweenCollinearRule(),
        MidpointBetweenRule(),
        # AngleBisect
        AngleBisectEqAngleRule(),
        AngleBisectEqRatioRule(),
        # Concurrent
        MediansConcurrentRule(),
        # Circumcenter
        CircumcenterCongABRule(),
        CircumcenterCongBCRule(),
        CircumcenterOnCircleRule(),
        # EqDist
        EqDistFromCongRule(),
        EqDistToCongRule(),
        # EqArea
        EqAreaSymRule(),
        # Harmonic
        HarmonicSwapRule(),
        HarmonicCollinearRule(),
        # PolePolar
        PolePolarPerpRule(),
        PolePolarTangentRule(),
        # InvImage
        InversionCollinearRule(),
        InversionCircleFixedRule(),
        # EqCrossRatio
        EqCrossRatioSymRule(),
        EqCrossRatioFromHarmonicRule(),
        # RadicalAxis
        RadicalAxisPerpRule(),
        # ── Converse / production rules ──
        MidpointFromCongBetweenRule(),
        MidpointFromCollinearEqDistRule(),
        CyclicFromEqAngleRule(),
        CyclicFromChordEqAngleRule(),
        CircumcenterFromEqDistRule(),
        CircumcenterFromOnCircleTripleRule(),
        AngleBisectFromEqAngleRule(),
        AngleBisectFromEqRatioBetweenRule(),
        PolePolarFromTangentsRule(),
        PolePolarFromPerpOnCircleRule(),
        RadicalAxisFromCommonPointsRule(),
        InvImageFromCollinearCongRule(),
        InvImageFromOnCircleFixedPointRule(),
        CongTriFromThreeCongRule(),
        EqDistFromIsoscelesAngleRule(),
        HarmonicFromCrossRatioRule(),
        ConcurrentFromAngleBisectorsRule(),
        # ── Bridge-strength enhancers (3-family+) ──
        CircleMetricAngleBridgeRule(),
        ProjectiveCircleLineBridgeRule(),
        SimilarityMetricConcurrencyBridgeRule(),
    ]
