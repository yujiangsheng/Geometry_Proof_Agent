"""dsl.py – Geometry domain-specific language: entities, predicates, facts, state.

Provides the foundational data types for the entire system:
  • Point / Line / Circle – geometric entities
  • Fact – a predicate applied to entity names (e.g. Parallel(A,B,C,D))
  • Goal / Step – proof-search primitives
  • GeoState – mutable fact database with *predicate-indexed* lookup
    for O(1) filtering (critical for rule matching performance)
  • Canonical constructors – normalise arguments so that logically
    equivalent facts always hash to the same Fact object

Author:  Jiangsheng Yu
License: MIT
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple


# ── Geometric entities ───────────────────────────────────────────────

@dataclass(frozen=True, order=True)
class Point:
    name: str


@dataclass(frozen=True, order=True)
class Line:
    a: Point
    b: Point

    def normalized(self) -> "Line":
        return self if self.a.name <= self.b.name else Line(self.b, self.a)


@dataclass(frozen=True, order=True)
class Circle:
    center: Point
    through: Point


# ── Core data types ──────────────────────────────────────────────────

@dataclass(frozen=True, order=True)
class Fact:
    """An atomic geometric proposition, e.g. Parallel(A, B, C, D)."""
    predicate: str
    args: Tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.predicate}({', '.join(self.args)})"

    def to_dict(self) -> dict:
        return {"predicate": self.predicate, "args": list(self.args)}

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(predicate=d["predicate"], args=tuple(d["args"]))


@dataclass(frozen=True)
class Goal:
    fact: Fact


@dataclass(frozen=True)
class Step:
    """One derivation step: rule applied to premises yielding a conclusion."""
    rule_name: str
    premise_facts: Tuple[Fact, ...]
    conclusion_fact: Fact

    def to_dict(self) -> dict:
        return {
            "rule_name": self.rule_name,
            "premises": [f.to_dict() for f in self.premise_facts],
            "conclusion": self.conclusion_fact.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        return cls(
            rule_name=d["rule_name"],
            premise_facts=tuple(Fact.from_dict(f) for f in d["premises"]),
            conclusion_fact=Fact.from_dict(d["conclusion"]),
        )


# ── GeoState with predicate index ────────────────────────────────────

class GeoState:
    """Mutable fact database with predicate-indexed fast lookup.

    Internally maintains both:
      • `facts: Set[Fact]` – for O(1) membership test
      • `_index: Dict[str, Set[Fact]]` – predicate → facts, for rule matching

    This avoids the O(n) linear scan that every rule previously required.
    """

    __slots__ = ("facts", "history", "_index")

    def __init__(
        self,
        facts: Set[Fact] | None = None,
        history: List[Step] | None = None,
    ) -> None:
        self.facts: Set[Fact] = facts if facts is not None else set()
        self.history: List[Step] = history if history is not None else []
        # Build index from initial facts
        self._index: Dict[str, Set[Fact]] = defaultdict(set)
        for f in self.facts:
            self._index[f.predicate].add(f)

    # ── Lookups ──────────────────────────────────────────────

    def has_fact(self, fact: Fact) -> bool:
        return fact in self.facts

    def by_predicate(self, predicate: str) -> Set[Fact]:
        """Return all facts with the given predicate (O(1) lookup)."""
        return self._index.get(predicate, set())

    # ── Mutations ────────────────────────────────────────────

    def add_fact(self, fact: Fact, via: Optional[Step] = None) -> bool:
        if fact in self.facts:
            return False
        self.facts.add(fact)
        self._index[fact.predicate].add(fact)
        if via is not None:
            self.history.append(via)
        return True

    def extend_facts(self, facts: Iterable[Fact]) -> int:
        added = 0
        for f in facts:
            if self.add_fact(f):
                added += 1
        return added

    # ── Iteration ────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.facts)

    def __iter__(self) -> Iterator[Fact]:
        return iter(self.facts)

    def __contains__(self, item: Fact) -> bool:
        return item in self.facts


# ── Canonical fact constructors ──────────────────────────────────────

def fact(predicate: str, *args: str) -> Fact:
    return Fact(predicate=predicate, args=tuple(args))


def canonical_parallel(a: str, b: str, c: str, d: str) -> Fact:
    """Parallel(sorted-pair, sorted-pair).  Pair *order* is preserved
    so that symmetry (swapping the two lines) yields a distinct Fact."""
    left = tuple(sorted((a, b)))
    right = tuple(sorted((c, d)))
    return Fact("Parallel", left + right)


def canonical_perp(a: str, b: str, c: str, d: str) -> Fact:
    """Perpendicular – same within-pair normalisation as Parallel."""
    left = tuple(sorted((a, b)))
    right = tuple(sorted((c, d)))
    return Fact("Perpendicular", left + right)


def canonical_collinear(a: str, b: str, c: str) -> Fact:
    return Fact("Collinear", tuple(sorted((a, b, c))))


def canonical_cyclic(a: str, b: str, c: str, d: str) -> Fact:
    return Fact("Cyclic", tuple(sorted((a, b, c, d))))


def canonical_midpoint(m: str, a: str, b: str) -> Fact:
    """IsMidpoint M A B – A,B sorted."""
    x, y = sorted((a, b))
    return Fact("Midpoint", (m, x, y))


def canonical_cong(a: str, b: str, c: str, d: str) -> Fact:
    """Cong(AB, CD) — segment congruence |AB|=|CD|.
    Within each pair, sort the two points; pair order preserved."""
    left = tuple(sorted((a, b)))
    right = tuple(sorted((c, d)))
    return Fact("Cong", left + right)


def canonical_eq_angle(a: str, b: str, c: str, d: str, e: str, f: str) -> Fact:
    """EqAngle(ABC, DEF)."""
    return Fact("EqAngle", (a, b, c, d, e, f))


def canonical_sim_tri(a: str, b: str, c: str, d: str, e: str, f: str) -> Fact:
    """SimTri(ABC, DEF) — triangles ABC ~ DEF (similar)."""
    return Fact("SimTri", (a, b, c, d, e, f))


def canonical_circle(o: str, a: str) -> Fact:
    """OnCircle(O, A) — A lies on circle centred at O with radius OA."""
    return Fact("OnCircle", (o, a))


# ── New predicates (Tier 1–3) ────────────────────────────────────────

def canonical_congtri(a: str, b: str, c: str,
                      d: str, e: str, f: str) -> Fact:
    """CongTri(ABC, DEF) — △ABC ≅ △DEF.  Vertex correspondence preserved."""
    return Fact("CongTri", (a, b, c, d, e, f))


def canonical_tangent(a: str, b: str, o: str, p: str) -> Fact:
    """Tangent(A,B,O,P) — line AB tangent to circle centred at O at point P."""
    left = tuple(sorted((a, b)))
    return Fact("Tangent", left + (o, p))


def canonical_eqratio(a: str, b: str, c: str, d: str,
                      e: str, f: str, g: str, h: str) -> Fact:
    """EqRatio(A,B,C,D,E,F,G,H) — |AB|/|CD| = |EF|/|GH|."""
    return Fact("EqRatio", (a, b, c, d, e, f, g, h))


def canonical_between(a: str, b: str, c: str) -> Fact:
    """Between(A,B,C) — B lies strictly between A and C on a line."""
    x, z = sorted((a, c))
    return Fact("Between", (x, b, z))


def canonical_angle_bisect(a: str, p: str, b: str, c: str) -> Fact:
    """AngleBisect(A,P,B,C) — ray AP bisects ∠BAC.  B,C sorted."""
    x, y = sorted((b, c))
    return Fact("AngleBisect", (a, p, x, y))


def canonical_concurrent(a: str, b: str, c: str, d: str,
                         e: str, f: str) -> Fact:
    """Concurrent(A,B,C,D,E,F) — lines AB, CD, EF are concurrent."""
    pairs = [tuple(sorted((a, b))),
             tuple(sorted((c, d))),
             tuple(sorted((e, f)))]
    pairs.sort()
    return Fact("Concurrent", pairs[0] + pairs[1] + pairs[2])


def canonical_circumcenter(o: str, a: str, b: str, c: str) -> Fact:
    """Circumcenter(O,A,B,C) — O is the circumcentre of △ABC."""
    tri = tuple(sorted((a, b, c)))
    return Fact("Circumcenter", (o,) + tri)


def canonical_eqdist(p: str, a: str, b: str) -> Fact:
    """EqDist(P,A,B) — |PA| = |PB|.  A,B sorted."""
    x, y = sorted((a, b))
    return Fact("EqDist", (p, x, y))


def canonical_eqarea(a: str, b: str, c: str,
                     d: str, e: str, f: str) -> Fact:
    """EqArea(A,B,C,D,E,F) — area(△ABC) = area(△DEF)."""
    return Fact("EqArea", (a, b, c, d, e, f))


def canonical_harmonic(a: str, b: str, c: str, d: str) -> Fact:
    """Harmonic(A,B,C,D) — (A,B;C,D) is a harmonic range."""
    return Fact("Harmonic", (a, b, c, d))


def canonical_pole_polar(p: str, a: str, b: str, o: str) -> Fact:
    """PolePolar(P,A,B,O) — P is pole of line AB w.r.t. circle O.  A,B sorted."""
    left = tuple(sorted((a, b)))
    return Fact("PolePolar", (p,) + left + (o,))


def canonical_inv_image(p_prime: str, p: str, o: str, a: str) -> Fact:
    """InvImage(P',P,O,A) — P' is inversion of P w.r.t. circle (O,|OA|)."""
    return Fact("InvImage", (p_prime, p, o, a))


def canonical_eq_cross_ratio(a: str, b: str, c: str, d: str,
                             e: str, f: str, g: str, h: str) -> Fact:
    """EqCrossRatio(A,B,C,D,E,F,G,H) — (A,B;C,D) = (E,F;G,H)."""
    return Fact("EqCrossRatio", (a, b, c, d, e, f, g, h))


def canonical_radical_axis(a: str, b: str, o1: str, o2: str) -> Fact:
    """RadicalAxis(A,B,O1,O2) — line AB is radical axis of circles O1, O2."""
    left = tuple(sorted((a, b)))
    right = tuple(sorted((o1, o2)))
    return Fact("RadicalAxis", left + right)
