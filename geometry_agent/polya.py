"""polya.py – Pólya plausible-reasoning agent: empirical conjecture testing.

Implements George Pólya's *patterns of plausible inference*:
  For a conjecture (assumptions ⊢ goal), randomly instantiate concrete
  numerical configurations.  If **any** instance satisfies all assumptions
  but violates the goal, the conjecture is **falsified** (counter-example).
  If *n* independent random instances all satisfy both assumptions and goal,
  we assign a **confidence score** that grows with *n*.

Confidence model (Bayesian-inspired)
-------------------------------------
We use a simple Laplace-succession formula:

    confidence = 1 − 1 / (k + 1)

where *k* is the number of *successful* trials (assumptions + goal both
satisfied).  This gives:
  •  0 trials →  0.00  (no evidence)
  •  1 trial  →  0.50
  •  5 trials →  0.83
  • 10 trials →  0.91
  • 20 trials →  0.95
  • 50 trials →  0.98

The agent does NOT prove anything; it filters conjectures **before** the
expensive symbolic search, saving compute on hopeless candidates and
prioritising likely-true conjectures.

Usage in the pipeline
---------------------
Insert between conjecture generation and beam search::

    conjecture = generate_conjecture(...)
    result = polya_test(conjecture.assumptions, conjecture.goal,
                        n_trials=30)
    if result.falsified:
        skip  # don't waste beam search on it
    elif result.confidence >= 0.90:
        prove(conjecture)  # high priority
    else:
        prove(conjecture)  # lower priority (fewer passing trials)

Author:  Jiangsheng Yu
License: MIT
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .dsl import Fact

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Result data class
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PolyaResult:
    """Result of Pólya plausible-reasoning testing.

    Attributes
    ----------
    falsified : bool
        True if a counter-example was found (conjecture is FALSE).
    counter_example : dict or None
        The coordinate assignment that falsified the conjecture.
    n_trials : int
        Total number of trial instantiations attempted.
    n_valid : int
        Trials where all assumptions were numerically satisfied
        (so the goal could be meaningfully tested).
    n_passed : int
        Trials where assumptions AND goal both held.
    n_failed : int
        Trials where assumptions held but goal did NOT (counter-examples).
    confidence : float
        Pólya confidence score in [0, 1).
    """
    falsified: bool = False
    counter_example: Optional[Dict[str, Tuple[float, float]]] = None
    n_trials: int = 0
    n_valid: int = 0
    n_passed: int = 0
    n_failed: int = 0
    confidence: float = 0.0

    @property
    def confidence_label(self) -> str:
        if self.falsified:
            return "已证伪 / Falsified"
        if self.confidence >= 0.95:
            return "高度可信 / Highly plausible"
        if self.confidence >= 0.85:
            return "较可信 / Plausible"
        if self.confidence >= 0.70:
            return "有一定可信度 / Somewhat plausible"
        if self.confidence >= 0.50:
            return "证据不足 / Weak evidence"
        return "未知 / Unknown"


# ═══════════════════════════════════════════════════════════════════════
# Numerical predicate checkers
# ═══════════════════════════════════════════════════════════════════════

# Tolerance for floating-point comparison.
# Pólya testing is *verification*, not proof — we allow pixel-level
# imprecision so that valid conjectures aren't falsified by
# floating-point rounding in constrained coordinate generation.
_EPS = 1e-4

# Maximum allowed coordinate magnitude — prevents OverflowError in
# expressions like x**2 when iterative constraint adjustments amplify
# coordinate values beyond float64 range.
_COORD_MAX = 1e100


def _clamp_coords(coords: 'Coords') -> 'Coords':
    """Clamp all coordinates to [-_COORD_MAX, _COORD_MAX]."""
    clamped = {}
    for p, (x, y) in coords.items():
        if not (math.isfinite(x) and math.isfinite(y)):
            # Replace NaN/Inf with random small values
            x = random.uniform(-10, 10) if not math.isfinite(x) else x
            y = random.uniform(-10, 10) if not math.isfinite(y) else y
        x = max(-_COORD_MAX, min(_COORD_MAX, x))
        y = max(-_COORD_MAX, min(_COORD_MAX, y))
        clamped[p] = (x, y)
    return clamped


def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _vec(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    return (p2[0] - p1[0], p2[1] - p1[1])


def _dot(u: Tuple[float, float], v: Tuple[float, float]) -> float:
    return u[0] * v[0] + u[1] * v[1]


def _cross(u: Tuple[float, float], v: Tuple[float, float]) -> float:
    return u[0] * v[1] - u[1] * v[0]


def _angle(a: Tuple[float, float], b: Tuple[float, float],
           c: Tuple[float, float]) -> float:
    """Angle ∠ABC in radians (at vertex B)."""
    u = _vec(b, a)
    v = _vec(b, c)
    dot = _dot(u, v)
    cross = _cross(u, v)
    return math.atan2(abs(cross), dot)


def _signed_angle(a: Tuple[float, float], b: Tuple[float, float],
                  c: Tuple[float, float]) -> float:
    """Signed angle ∠ABC at vertex B, in (−π, π]."""
    u = _vec(b, a)
    v = _vec(b, c)
    return math.atan2(_cross(u, v), _dot(u, v))


def _triangle_area(a: Tuple[float, float], b: Tuple[float, float],
                   c: Tuple[float, float]) -> float:
    """Signed area of triangle ABC (positive if counter-clockwise)."""
    return 0.5 * _cross(_vec(a, b), _vec(a, c))


Coords = Dict[str, Tuple[float, float]]


def _check_predicate(pred: str, args: Tuple[str, ...],
                     coords: Coords) -> Optional[bool]:
    """Check if a predicate holds numerically.

    Returns True/False if checkable, or None if the predicate is
    unsupported or the points are degenerate (e.g. coincident).
    """
    # Resolve coordinates; if any point is missing return None
    try:
        pts = [coords[a] for a in args]
    except KeyError:
        return None

    if pred == "Parallel":
        # Parallel(A,B,C,D): AB ∥ CD
        if len(args) != 4:
            return None
        a, b, c, d = pts
        u = _vec(a, b)
        v = _vec(c, d)
        if _dist(a, b) < _EPS or _dist(c, d) < _EPS:
            return None  # degenerate
        return abs(_cross(u, v)) < _EPS * max(_dist(a, b), _dist(c, d))

    if pred == "Perpendicular":
        # Perpendicular(A,B,C,D): AB ⊥ CD
        if len(args) != 4:
            return None
        a, b, c, d = pts
        u = _vec(a, b)
        v = _vec(c, d)
        if _dist(a, b) < _EPS or _dist(c, d) < _EPS:
            return None
        return abs(_dot(u, v)) < _EPS * max(_dist(a, b), _dist(c, d))

    if pred == "Collinear":
        # Collinear(A,B,C)
        if len(args) != 3:
            return None
        a, b, c = pts
        return abs(_cross(_vec(a, b), _vec(a, c))) < _EPS * max(
            _dist(a, b), _dist(a, c), 1e-12)

    if pred == "Cong":
        # Cong(A,B,C,D): |AB| = |CD|
        if len(args) != 4:
            return None
        a, b, c, d = pts
        d1 = _dist(a, b)
        d2 = _dist(c, d)
        if d1 < _EPS and d2 < _EPS:
            return None  # degenerate
        return abs(d1 - d2) < _EPS * max(d1, d2, 1.0)

    if pred == "Midpoint":
        # Midpoint(M,A,B): M is midpoint of AB
        if len(args) != 3:
            return None
        m, a, b = pts
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        return _dist(m, (mx, my)) < _EPS * max(_dist(a, b), 1.0)

    if pred == "EqAngle":
        # EqAngle(A,B,C,D,E,F): ∠ABC = ∠DEF
        if len(args) != 6:
            return None
        a, b, c, d, e, f = pts
        if _dist(b, a) < _EPS or _dist(b, c) < _EPS:
            return None
        if _dist(e, d) < _EPS or _dist(e, f) < _EPS:
            return None
        ang1 = _angle(a, b, c)
        ang2 = _angle(d, e, f)
        return abs(ang1 - ang2) < _EPS * 100  # angles need looser tolerance

    if pred == "Cyclic":
        # Cyclic(A,B,C,D): A,B,C,D concyclic
        if len(args) != 4:
            return None
        a, b, c, d = pts
        # Check via determinant (all 4 on the same circle)
        # Use the concyclic determinant test
        def _lift(p: Tuple[float, float]) -> Tuple[float, float, float]:
            return (p[0], p[1], p[0]**2 + p[1]**2)
        la, lb, lc, ld = _lift(a), _lift(b), _lift(c), _lift(d)
        # Build 4x4 determinant with row = (x, y, x²+y², 1)
        mat = [
            [la[0], la[1], la[2], 1],
            [lb[0], lb[1], lb[2], 1],
            [lc[0], lc[1], lc[2], 1],
            [ld[0], ld[1], ld[2], 1],
        ]
        det = _det4(mat)
        scale = max(abs(la[2]), abs(lb[2]), abs(lc[2]), abs(ld[2]), 1.0)
        return abs(det) < _EPS * scale * 100

    if pred == "SimTri":
        # SimTri(A,B,C,D,E,F): △ABC ∼ △DEF
        if len(args) != 6:
            return None
        a, b, c, d, e, f = pts
        sides1 = sorted([_dist(a, b), _dist(b, c), _dist(a, c)])
        sides2 = sorted([_dist(d, e), _dist(e, f), _dist(d, f)])
        if sides1[0] < _EPS or sides2[0] < _EPS:
            return None
        # Check ratios equal
        r1 = sides1[1] / sides1[0]
        r2 = sides1[2] / sides1[0]
        s1 = sides2[1] / sides2[0]
        s2 = sides2[2] / sides2[0]
        return abs(r1 - s1) < _EPS * 100 and abs(r2 - s2) < _EPS * 100

    if pred == "CongTri":
        # CongTri(A,B,C,D,E,F): △ABC ≅ △DEF
        if len(args) != 6:
            return None
        a, b, c, d, e, f = pts
        sides1 = sorted([_dist(a, b), _dist(b, c), _dist(a, c)])
        sides2 = sorted([_dist(d, e), _dist(e, f), _dist(d, f)])
        if sides1[0] < _EPS:
            return None
        return all(abs(s1 - s2) < _EPS * max(s1, 1.0)
                   for s1, s2 in zip(sides1, sides2))

    if pred == "Tangent":
        # Tangent(A,B,O,P): line AB tangent to circle O at P
        if len(args) != 4:
            return None
        a, b, o, p = pts
        # P on circle => |OP| = radius ... but we don't know radius.
        # Check: P on AB and OP ⊥ AB
        u = _vec(a, b)
        v = _vec(p, o)
        if _dist(a, b) < _EPS:
            return None
        # P on line AB?
        if abs(_cross(_vec(a, p), u)) > _EPS * _dist(a, b):
            return None  # P not on AB → degenerate
        # OP ⊥ AB?
        return abs(_dot(u, v)) < _EPS * max(_dist(a, b), _dist(p, o), 1.0)

    if pred == "EqRatio":
        # EqRatio(A,B,C,D,E,F,G,H): |AB|/|CD| = |EF|/|GH|
        if len(args) != 8:
            return None
        a, b, c, d, e, f, g, h = pts
        d1, d2 = _dist(a, b), _dist(c, d)
        d3, d4 = _dist(e, f), _dist(g, h)
        if d2 < _EPS or d4 < _EPS:
            return None
        return abs(d1 / d2 - d3 / d4) < _EPS * 100

    if pred == "Between":
        # Between(A,B,C): B between A and C
        if len(args) != 3:
            return None
        a, b, c = pts
        # B on AC and t ∈ (0,1)
        ac = _dist(a, c)
        if ac < _EPS:
            return None
        ab = _dist(a, b)
        bc = _dist(b, c)
        return abs(ab + bc - ac) < _EPS * ac

    if pred == "AngleBisect":
        # AngleBisect(A,P,B,C): ray AP bisects ∠BAC
        if len(args) != 4:
            return None
        a, p, b, c = pts
        if _dist(a, b) < _EPS or _dist(a, c) < _EPS or _dist(a, p) < _EPS:
            return None
        ang1 = _angle(b, a, p)
        ang2 = _angle(p, a, c)
        return abs(ang1 - ang2) < _EPS * 100

    if pred == "Circumcenter":
        # Circumcenter(O,A,B,C): O is circumcentre of △ABC
        if len(args) != 4:
            return None
        o, a, b, c = pts
        ra = _dist(o, a)
        rb = _dist(o, b)
        rc = _dist(o, c)
        if ra < _EPS:
            return None
        return abs(ra - rb) < _EPS * ra and abs(ra - rc) < _EPS * ra

    if pred == "EqDist":
        # EqDist(P,A,B): |PA| = |PB|
        if len(args) != 3:
            return None
        p, a, b = pts
        da = _dist(p, a)
        db = _dist(p, b)
        return abs(da - db) < _EPS * max(da, db, 1.0)

    if pred == "EqArea":
        # EqArea(A,B,C,D,E,F): area(△ABC) = area(△DEF)
        if len(args) != 6:
            return None
        a, b, c, d, e, f = pts
        area1 = abs(_triangle_area(a, b, c))
        area2 = abs(_triangle_area(d, e, f))
        return abs(area1 - area2) < _EPS * max(area1, area2, 1.0) * 100

    if pred == "Concurrent":
        # Concurrent(A,B,C,D,E,F): lines AB, CD, EF concurrent
        if len(args) != 6:
            return None
        a, b, c, d, e, f = pts
        # Find intersection of AB and CD, check if EF passes through it
        u = _vec(a, b)
        v = _vec(c, d)
        cr = _cross(u, v)
        if abs(cr) < _EPS:
            return None  # parallel or degenerate
        w = _vec(a, c)
        t = _cross(w, v) / cr
        ix = a[0] + t * u[0]
        iy = a[1] + t * u[1]
        # Check if (ix, iy) is on line EF
        ef = _vec(e, f)
        if _dist(e, f) < _EPS:
            return None
        return abs(_cross(_vec(e, (ix, iy)), ef)) < _EPS * _dist(e, f) * 100

    if pred == "Harmonic":
        # Harmonic(A,B,C,D): (A,B;C,D) = -1
        if len(args) != 4:
            return None
        a, b, c, d = pts
        # C and D on line AB, cross-ratio = -1
        # Use signed ratios: (AC/CB) / (AD/DB) = -1
        ab_vec = _vec(a, b)
        ab_len = _dist(a, b)
        if ab_len < _EPS:
            return None
        # Project C and D onto AB
        tc = _dot(_vec(a, c), ab_vec) / (ab_len ** 2)
        td = _dot(_vec(a, d), ab_vec) / (ab_len ** 2)
        # Cross ratio: (tc * (1-td)) / ((1-tc) * td) ... simplified
        if abs(1 - tc) < _EPS or abs(td) < _EPS or abs(1 - td) < _EPS or abs(tc) < _EPS:
            return None
        cr_val = (tc / (tc - 1)) * ((td - 1) / td)
        return abs(cr_val - 1.0) < _EPS * 100  # cross-ratio = -1 simplifies

    if pred == "OnCircle":
        # OnCircle(O,A): A on circle centred O (but radius unknown)
        # Without knowing the radius, can't verify meaningfully.
        return None

    if pred in ("PolePolar", "InvImage", "EqCrossRatio", "RadicalAxis"):
        # These advanced predicates need more context (circle radii, etc.)
        # Return None to skip — the trial is not counted.
        return None

    # Unknown predicate
    return None


def _det4(m: List[List[float]]) -> float:
    """4×4 determinant by cofactor expansion on first row."""
    def _det3(a: List[List[float]]) -> float:
        return (a[0][0] * (a[1][1]*a[2][2] - a[1][2]*a[2][1])
              - a[0][1] * (a[1][0]*a[2][2] - a[1][2]*a[2][0])
              + a[0][2] * (a[1][0]*a[2][1] - a[1][1]*a[2][0]))

    result = 0.0
    for j in range(4):
        minor = [[m[i][k] for k in range(4) if k != j] for i in range(1, 4)]
        sign = 1 if j % 2 == 0 else -1
        result += sign * m[0][j] * _det3(minor)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Random geometric configuration generators
# ═══════════════════════════════════════════════════════════════════════

def _random_coords(points: Sequence[str],
                   spread: float = 10.0) -> Coords:
    """Generate fully random coordinate assignment (uniform in [−spread, spread])."""
    return {p: (random.uniform(-spread, spread),
                random.uniform(-spread, spread)) for p in points}


def _presolve_perp_intersections(
    coords: Coords,
    assumptions: Sequence[Fact],
) -> Coords:
    """Analytically solve perpendicular-line intersection constraints.

    When two ``Perpendicular`` constraints share a point in their first
    line pair (e.g.  Perp(O, M₁, A, B) and Perp(O, M₂, C, D) both have
    O), the iterative adjustor oscillates because both constraints try to
    move the same point.  Here we detect such shared points and compute
    them as the intersection of the two perpendicular lines.

    This handles the common "double perpendicular bisector → circumcenter"
    pattern analytically.
    """
    coords = dict(coords)
    perps = [f for f in assumptions if f.predicate == "Perpendicular"]
    if len(perps) < 2:
        return coords

    # Build map: point → list of Perp facts where it appears in 1st pair
    point_perps: Dict[str, List[Fact]] = {}
    for fact in perps:
        a, b = fact.args[0], fact.args[1]
        point_perps.setdefault(a, []).append(fact)
        point_perps.setdefault(b, []).append(fact)

    for shared_pt, facts in point_perps.items():
        if len(facts) < 2:
            continue
        # Take the first two Perps sharing this point
        f1, f2 = facts[0], facts[1]
        a1, b1, c1, d1 = f1.args
        a2, b2, c2, d2 = f2.args

        # Identify the "other" point on each first line (the non-shared one)
        other1 = b1 if a1 == shared_pt else a1
        other2 = b2 if a2 == shared_pt else a2

        # Line 1: through `other1`, direction perpendicular to (c1,d1)
        # Line 2: through `other2`, direction perpendicular to (c2,d2)
        try:
            p_o1 = coords[other1]
            p_c1, p_d1 = coords[c1], coords[d1]
            p_o2 = coords[other2]
            p_c2, p_d2 = coords[c2], coords[d2]
        except KeyError:
            continue

        # Direction of line (c1,d1)
        dx1, dy1 = p_d1[0] - p_c1[0], p_d1[1] - p_c1[1]
        # Perpendicular direction
        perp1 = (-dy1, dx1)

        dx2, dy2 = p_d2[0] - p_c2[0], p_d2[1] - p_c2[1]
        perp2 = (-dy2, dx2)

        # Intersection of:
        #   line through other1 in direction perp1
        #   line through other2 in direction perp2
        # Parametric: other1 + t*perp1 = other2 + s*perp2
        # Solve: t*perp1 - s*perp2 = other2 - other1
        det = perp1[0] * (-perp2[1]) - perp1[1] * (-perp2[0])
        if abs(det) < 1e-10:
            continue  # parallel lines → skip

        rx = p_o2[0] - p_o1[0]
        ry = p_o2[1] - p_o1[1]
        t = (rx * (-perp2[1]) - ry * (-perp2[0])) / det

        ix = p_o1[0] + t * perp1[0]
        iy = p_o1[1] + t * perp1[1]

        if math.isfinite(ix) and math.isfinite(iy):
            coords[shared_pt] = (ix, iy)

    return coords


def _constrained_coords(
    points: Sequence[str],
    assumptions: Sequence[Fact],
    spread: float = 10.0,
    max_retries: int = 50,
) -> Optional[Coords]:
    """Generate random coordinates that satisfy all assumptions.

    Strategy: start with random coordinates, then iteratively adjust
    points to honour each assumption constraint.  Uses a two-phase
    approach:

    1. **Deterministic phase** — constraints like Midpoint, Circumcenter,
       Between compute their output points exactly from other points.
       These are processed first every iteration.
    2. **Iterative phase** — remaining constraints (Perpendicular, Cong,
       EqAngle, …) adjust coordinates, but are told which points are
       *protected* (involved in deterministic constraints) so they
       avoid moving those points and instead adjust free points.

    If the constraints can't be satisfied within *max_retries* adjustment
    passes, return None (the trial is skipped).
    """
    coords = _random_coords(points, spread)

    # --- Separate deterministic vs. iterative constraints ----------------
    _DET_PREDS = {"Midpoint", "Circumcenter", "Between"}
    det_facts: List[Fact] = []
    other_facts: List[Fact] = []
    det_outputs: Set[str] = set()         # OUTPUT points of deterministic constraints

    for fact in assumptions:
        if fact.predicate in _DET_PREDS:
            det_facts.append(fact)
            # Only protect the OUTPUT point (not the inputs):
            #   Midpoint(M, A, B) → protect M
            #   Circumcenter(O, A, B, C) → protect O
            #   Between(A, B, C) → protect B (the middle point)
            if fact.predicate in ("Midpoint", "Circumcenter"):
                det_outputs.add(fact.args[0])
            elif fact.predicate == "Between":
                det_outputs.add(fact.args[1])
        else:
            other_facts.append(fact)

    # Process order: deterministic first, then the rest
    ordered = det_facts + other_facts

    # Use a tighter internal tolerance so converged results easily
    # pass the strict re-verification in check_premise_consistency.
    global _EPS
    saved_eps = _EPS
    _EPS = saved_eps * 0.01  # 100× tighter than normal

    try:
        for _attempt in range(max_retries):
            # Phase 1: ALWAYS recompute deterministic constraints (Midpoint,
            # Circumcenter, Between) so their outputs reflect the latest
            # input coordinates.  This prevents stale values after other
            # constraints have been adjusted.
            for fact in det_facts:
                try:
                    coords = _adjust_for_constraint(coords, fact, spread)
                except (OverflowError, ValueError):
                    pass

            # Phase 1.5: Analytically solve perpendicular-line intersections.
            # This must run AFTER midpoints are recomputed so the perpendicular
            # lines pass through the correct midpoint positions.
            if _attempt == 0 or _attempt % 10 == 0:
                coords = _presolve_perp_intersections(coords, assumptions)

            # Phase 2: check ALL constraints and adjust non-deterministic ones
            all_ok = True
            for fact in ordered:
                result = _check_predicate(fact.predicate, fact.args, coords)
                if result is None:
                    continue  # unsupported predicate → skip
                if result:
                    continue  # already satisfied

                all_ok = False
                # For deterministic constraints they were already recomputed
                # above, so only adjust non-deterministic ones here.
                if fact.predicate not in _DET_PREDS:
                    try:
                        coords = _adjust_for_constraint(
                            coords, fact, spread, protected=frozenset(det_outputs))
                    except (OverflowError, ValueError):
                        pass  # constraint adjustment failed; skip

            # Clamp coordinates to prevent overflow in subsequent passes
            coords = _clamp_coords(coords)

            if all_ok:
                return coords
    finally:
        _EPS = saved_eps

    return None


def _adjust_for_constraint(
    coords: Coords,
    fact: Fact,
    spread: float,
    protected: frozenset = frozenset(),
) -> Coords:
    """Attempt to adjust coordinates to satisfy a single fact.

    Modifies the last point(s) in the fact's args to honour the constraint,
    unless that point is in *protected* — in which case an alternative
    (unprotected) point is adjusted instead.
    Returns a new dict (shallow copy).
    """
    coords = dict(coords)
    pred, args = fact.predicate, fact.args

    try:
        pts = [coords[a] for a in args]
    except KeyError:
        return coords

    if pred == "Parallel" and len(args) == 4:
        # Make line(a,b) ∥ line(c,d).
        # Default: adjust d.  If d is protected, try a, then c, then b.
        a, b, c, d = args
        if d not in protected:
            pa, pb, pc = coords[a], coords[b], coords[c]
            dx, dy = pb[0] - pa[0], pb[1] - pa[1]
            coords[d] = (pc[0] + dx, pc[1] + dy)
        elif a not in protected:
            pb, pc, pd = coords[b], coords[c], coords[d]
            dx, dy = pd[0] - pc[0], pd[1] - pc[1]
            coords[a] = (pb[0] - dx, pb[1] - dy)
        elif c not in protected:
            pa, pb, pd = coords[a], coords[b], coords[d]
            dx, dy = pb[0] - pa[0], pb[1] - pa[1]
            coords[c] = (pd[0] - dx, pd[1] - dy)
        elif b not in protected:
            pa, pc, pd = coords[a], coords[c], coords[d]
            dx, dy = pd[0] - pc[0], pd[1] - pc[1]
            coords[b] = (pa[0] + dx, pa[1] + dy)

    elif pred == "Perpendicular" and len(args) == 4:
        # Make line(a,b) ⊥ line(c,d).
        # Default: adjust d.  If d is protected, find the first
        # unprotected point among {a, c, b} and adjust it instead.
        a, b, c, d = args
        if d not in protected:
            pa, pb, pc = coords[a], coords[b], coords[c]
            dx, dy = pb[0] - pa[0], pb[1] - pa[1]
            coords[d] = (pc[0] - dy, pc[1] + dx)
        elif a not in protected:
            # A = B + rot90(D − C) so that AB ⊥ CD
            pb, pc, pd = coords[b], coords[c], coords[d]
            dx, dy = pd[0] - pc[0], pd[1] - pc[1]
            coords[a] = (pb[0] - dy, pb[1] + dx)
        elif c not in protected:
            # C = D − rot90(B − A) so that CD ⊥ AB
            pa, pb, pd = coords[a], coords[b], coords[d]
            dx, dy = pb[0] - pa[0], pb[1] - pa[1]
            coords[c] = (pd[0] + dy, pd[1] - dx)
        elif b not in protected:
            # B = A + rot90(D − C)
            pa, pc, pd = coords[a], coords[c], coords[d]
            dx, dy = pd[0] - pc[0], pd[1] - pc[1]
            coords[b] = (pa[0] - dy, pa[1] + dx)

    elif pred == "Collinear" and len(args) == 3:
        # Put C on line AB
        a, b, c = args
        pa, pb = coords[a], coords[b]
        t = random.uniform(0.2, 0.8)
        coords[c] = (pa[0] + t * (pb[0] - pa[0]),
                      pa[1] + t * (pb[1] - pa[1]))

    elif pred == "Midpoint" and len(args) == 3:
        # M is midpoint of AB
        m, a, b = args
        pa, pb = coords[a], coords[b]
        coords[m] = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)

    elif pred == "Cong" and len(args) == 4:
        # |AB| = |CD|: move D so |CD| = |AB|
        a, b, c, d = args
        d_ab = _dist(coords[a], coords[b])
        pc = coords[c]
        pd = coords[d]
        d_cd = _dist(pc, pd)
        if d_cd > _EPS:
            scale = d_ab / d_cd
            coords[d] = (pc[0] + scale * (pd[0] - pc[0]),
                         pc[1] + scale * (pd[1] - pc[1]))
        else:
            # D coincides with C; place D at distance d_ab from C
            angle = random.uniform(0, 2 * math.pi)
            coords[d] = (pc[0] + d_ab * math.cos(angle),
                         pc[1] + d_ab * math.sin(angle))

    elif pred == "EqAngle" and len(args) == 6:
        # ∠ABC = ∠DEF: rotate F around E so the angle matches
        a, b, c, d, e, f = args
        ang1 = _angle(coords[a], coords[b], coords[c])
        pe, pd_pt = coords[e], coords[d]
        # Place F so ∠DEF = ang1
        # Direction from E to D
        ed = _vec(pe, pd_pt)
        base_angle = math.atan2(ed[1], ed[0])
        r = _dist(pe, coords[f])
        if r < _EPS:
            r = random.uniform(1, 5)
        # Try both +ang1 and -ang1 (pick one randomly)
        sign = random.choice([-1, 1])
        target_angle = base_angle + sign * ang1
        coords[f] = (pe[0] + r * math.cos(target_angle),
                      pe[1] + r * math.sin(target_angle))

    elif pred == "Cyclic" and len(args) == 4:
        # Place all 4 on a circle: keep A,B,C, place D on circumcircle
        a, b, c, d = args
        pa, pb, pc = coords[a], coords[b], coords[c]
        center = _circumcenter_pt(pa, pb, pc)
        if center:
            r = _dist(center, pa)
            angle = random.uniform(0, 2 * math.pi)
            coords[d] = (center[0] + r * math.cos(angle),
                         center[1] + r * math.sin(angle))

    elif pred == "Circumcenter" and len(args) == 4:
        # O is circumcentre of △ABC
        o, a, b, c = args
        pa, pb, pc = coords[a], coords[b], coords[c]
        center = _circumcenter_pt(pa, pb, pc)
        if center:
            coords[o] = center

    elif pred == "Between" and len(args) == 3:
        # B between A and C
        a, b, c = args
        pa, pc = coords[a], coords[c]
        t = random.uniform(0.1, 0.9)
        coords[b] = (pa[0] + t * (pc[0] - pa[0]),
                      pa[1] + t * (pc[1] - pa[1]))

    elif pred == "EqDist" and len(args) == 3:
        # |PA| = |PB|: put P on perpendicular bisector of AB
        p, a, b = args
        pa_pt, pb_pt = coords[a], coords[b]
        mx = (pa_pt[0] + pb_pt[0]) / 2
        my = (pa_pt[1] + pb_pt[1]) / 2
        dx, dy = pb_pt[0] - pa_pt[0], pb_pt[1] - pa_pt[1]
        t = random.uniform(-3, 3)
        coords[p] = (mx - t * dy, my + t * dx)

    elif pred == "SimTri" and len(args) == 6:
        # △ABC ~ △DEF: scale + rotate △ABC around D
        a, b, c, d, e, f = args
        pa, pb, pc, pd = coords[a], coords[b], coords[c], coords[d]
        scale = random.uniform(0.5, 2.0)
        theta = random.uniform(0, 2 * math.pi)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        # Transform B-A and C-A, then place at D
        for src, tgt in [(pb, e), (pc, f)]:
            dx_s = src[0] - pa[0]
            dy_s = src[1] - pa[1]
            coords[tgt] = (pd[0] + scale * (cos_t * dx_s - sin_t * dy_s),
                           pd[1] + scale * (sin_t * dx_s + cos_t * dy_s))

    elif pred == "AngleBisect" and len(args) == 4:
        # Ray AP bisects ∠BAC: place P on bisector
        a, p, b, c = args
        pa_pt, pb_pt, pc_pt = coords[a], coords[b], coords[c]
        if _dist(pa_pt, pb_pt) > _EPS and _dist(pa_pt, pc_pt) > _EPS:
            ub = _vec(pa_pt, pb_pt)
            uc = _vec(pa_pt, pc_pt)
            nb = math.hypot(*ub)
            nc = math.hypot(*uc)
            bx = ub[0] / nb + uc[0] / nc
            by = ub[1] / nb + uc[1] / nc
            r = random.uniform(1, 5)
            norm = math.hypot(bx, by)
            if norm > _EPS:
                coords[p] = (pa_pt[0] + r * bx / norm,
                             pa_pt[1] + r * by / norm)

    elif pred == "EqArea" and len(args) == 6:
        # area(△ABC) = area(△DEF): adjust F
        a, b, c, d, e, f = args
        target_area = abs(_triangle_area(coords[a], coords[b], coords[c]))
        pd_pt, pe_pt = coords[d], coords[e]
        base = _dist(pd_pt, pe_pt)
        if base > _EPS:
            h = 2 * target_area / base
            # Perpendicular direction from DE
            de = _vec(pd_pt, pe_pt)
            perp = (-de[1], de[0])
            norm_p = math.hypot(*perp)
            if norm_p > _EPS:
                sign = random.choice([-1, 1])
                coords[f] = (
                    (pd_pt[0] + pe_pt[0]) / 2 + sign * h * perp[0] / norm_p,
                    (pd_pt[1] + pe_pt[1]) / 2 + sign * h * perp[1] / norm_p,
                )

    elif pred == "CongTri" and len(args) == 6:
        # △ABC ≅ △DEF: rigid motion of △ABC to D
        a, b, c, d, e, f = args
        pa, pb, pc, pd = coords[a], coords[b], coords[c], coords[d]
        theta = random.uniform(0, 2 * math.pi)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        for src, tgt in [(pb, e), (pc, f)]:
            dx_s = src[0] - pa[0]
            dy_s = src[1] - pa[1]
            coords[tgt] = (pd[0] + cos_t * dx_s - sin_t * dy_s,
                           pd[1] + sin_t * dx_s + cos_t * dy_s)

    elif pred == "Tangent" and len(args) == 4:
        # Tangent(A,B,O,P): line AB tangent to circle O at P
        # Place P on AB, then O so OP ⊥ AB
        a, b, o, p_name = args
        pa, pb = coords[a], coords[b]
        t = random.uniform(0.3, 0.7)
        pp = (pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1]))
        coords[p_name] = pp
        ab = _vec(pa, pb)
        r = random.uniform(1, 4)
        sign = random.choice([-1, 1])
        n = math.hypot(*ab)
        if n > _EPS:
            coords[o] = (pp[0] + sign * r * (-ab[1]) / n,
                         pp[1] + sign * r * ab[0] / n)

    elif pred == "EqRatio" and len(args) == 8:
        # |AB|/|CD| = |EF|/|GH|: adjust H
        a, b, c, d, e, f, g, h = args
        d1 = _dist(coords[a], coords[b])
        d2 = _dist(coords[c], coords[d])
        d3 = _dist(coords[e], coords[f])
        if d2 > _EPS:
            target_d4 = d3 * d2 / d1 if d1 > _EPS else d3
            pg = coords[g]
            ph = coords[h]
            d_curr = _dist(pg, ph)
            if d_curr > _EPS:
                scale = target_d4 / d_curr
                coords[h] = (pg[0] + scale * (ph[0] - pg[0]),
                             pg[1] + scale * (ph[1] - pg[1]))

    elif pred == "Concurrent" and len(args) == 6:
        # Lines AB, CD, EF concurrent: find intersection of AB & CD, 
        # place F so EF passes through it
        a, b, c, d, e, f = args
        pa, pb, pc, pd, pe = [coords[x] for x in [a, b, c, d, e]]
        u = _vec(pa, pb)
        v = _vec(pc, pd)
        cr = _cross(u, v)
        if abs(cr) > _EPS:
            w = _vec(pa, pc)
            t_val = _cross(w, v) / cr
            ix = pa[0] + t_val * u[0]
            iy = pa[1] + t_val * u[1]
            # Place F on ray from E through intersection
            ef_dir = _vec(pe, (ix, iy))
            r = random.uniform(0.5, 2.0)
            coords[f] = (pe[0] + r * ef_dir[0], pe[1] + r * ef_dir[1])

    return coords


def _circumcenter_pt(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """Circumcentre of triangle ABC, or None if degenerate."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    try:
        D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(D) < _EPS:
            return None
        ux = ((ax**2 + ay**2) * (by - cy) +
              (bx**2 + by**2) * (cy - ay) +
              (cx**2 + cy**2) * (ay - by)) / D
        uy = ((ax**2 + ay**2) * (cx - bx) +
              (bx**2 + by**2) * (ax - cx) +
              (cx**2 + cy**2) * (bx - ax)) / D
        if not (math.isfinite(ux) and math.isfinite(uy)):
            return None
        return (ux, uy)
    except (OverflowError, ValueError):
        return None


# ═══════════════════════════════════════════════════════════════════════
# Main Pólya testing function
# ═══════════════════════════════════════════════════════════════════════

def polya_test(
    assumptions: Sequence[Fact],
    goal: Fact,
    *,
    n_trials: int = 30,
    spread: float = 10.0,
    early_falsify: bool = True,
) -> PolyaResult:
    """Run Pólya plausible-reasoning test on a conjecture.

    Parameters
    ----------
    assumptions : list[Fact]
        The geometric hypotheses.
    goal : Fact
        The statement to test.
    n_trials : int
        Number of random instantiation attempts (default: 30).
    spread : float
        Coordinate range [−spread, spread] for random points.
    early_falsify : bool
        If True, stop immediately upon finding a counter-example.

    Returns
    -------
    PolyaResult
        Includes falsified flag, confidence score, and statistics.
    """
    # Collect all point names
    all_points: List[str] = []
    seen: Set[str] = set()
    for fact in list(assumptions) + [goal]:
        for arg in fact.args:
            if arg not in seen:
                seen.add(arg)
                all_points.append(arg)

    result = PolyaResult(n_trials=n_trials)

    for trial in range(n_trials):
        # Generate coordinates satisfying the assumptions
        try:
            coords = _constrained_coords(all_points, assumptions,
                                         spread=spread, max_retries=30)
        except (OverflowError, ValueError):
            continue
        if coords is None:
            # Could not satisfy assumptions → skip this trial
            continue

        # Double-check: verify ALL assumptions hold numerically
        assumptions_ok = True
        any_unsupported = False
        try:
            for fact in assumptions:
                check = _check_predicate(fact.predicate, fact.args, coords)
                if check is None:
                    any_unsupported = True
                    continue
                if not check:
                    assumptions_ok = False
                    break
        except (OverflowError, ValueError):
            continue

        if not assumptions_ok:
            continue  # constraint solver didn't fully converge

        # Assumptions satisfied → this is a valid trial
        result.n_valid += 1

        # Now check the goal
        try:
            goal_check = _check_predicate(goal.predicate, goal.args, coords)
        except (OverflowError, ValueError):
            continue
        if goal_check is None:
            # Goal predicate unsupported → can't evaluate; still
            # count as valid but don't count for/against.
            continue

        if goal_check:
            result.n_passed += 1
        else:
            result.n_failed += 1
            if early_falsify:
                result.falsified = True
                result.counter_example = coords
                result.confidence = 0.0
                logger.info(
                    "Pólya: conjecture FALSIFIED in trial %d "
                    "(counter-example found)", trial + 1)
                return result

    # Compute confidence score
    if result.n_failed > 0:
        result.falsified = True
        result.confidence = 0.0
    elif result.n_passed > 0:
        # Laplace succession: confidence = 1 − 1/(k+1)
        result.confidence = 1.0 - 1.0 / (result.n_passed + 1)
    else:
        result.confidence = 0.0  # no valid trials at all

    return result


def polya_test_two_stage(
    assumptions: Sequence[Fact],
    goal: Fact,
    *,
    fast_trials: int = 3,
    full_trials: int = 20,
    spread: float = 10.0,
) -> PolyaResult:
    """Two-stage Pólya test: fast rejection then confirmation.

    Stage 1: Run *fast_trials* trials with early_falsify=True.
             ~50% of false conjectures are caught in 1-2 trials.
    Stage 2: Only if Stage 1 passes, run remaining trials up to
             *full_trials* for proper confidence estimation.

    This typically saves ~40% of total Pólya compute compared to
    always running *full_trials* trials.
    """
    # Stage 1: fast rejection (1-3 trials)
    fast_result = polya_test(
        assumptions, goal,
        n_trials=fast_trials,
        spread=spread,
        early_falsify=True,
    )
    if fast_result.falsified:
        return fast_result

    # Stage 2: confirmation with remaining budget
    remaining = max(1, full_trials - fast_trials)
    full_result = polya_test(
        assumptions, goal,
        n_trials=remaining,
        spread=spread,
        early_falsify=True,
    )

    # Merge stats from both stages
    merged = PolyaResult(n_trials=fast_trials + remaining)
    merged.n_valid = fast_result.n_valid + full_result.n_valid
    merged.n_passed = fast_result.n_passed + full_result.n_passed
    merged.n_failed = fast_result.n_failed + full_result.n_failed
    merged.falsified = full_result.falsified
    merged.counter_example = full_result.counter_example

    if merged.n_failed > 0:
        merged.falsified = True
        merged.confidence = 0.0
    elif merged.n_passed > 0:
        merged.confidence = 1.0 - 1.0 / (merged.n_passed + 1)
    else:
        merged.confidence = 0.0

    return merged


# ═══════════════════════════════════════════════════════════════════════
# Premise consistency checker
# ═══════════════════════════════════════════════════════════════════════

def _check_structural_nondegeneracy(
    assumptions: Sequence[Fact],
    coords: Coords,
    min_sep: float = 0.005,
    min_triangle_area: float = 1e-4,
) -> bool:
    """Check structural non-degeneracy beyond point separation.

    Ensures:
      1. All distinctly named points are pairwise separated.
      2. Triangle vertices in Circumcenter / SimTri / CongTri / EqArea
         are non-collinear (triangle area > threshold).
      3. Line-defining pairs in Perpendicular / Parallel have positive
         length (the two points are separated).
      4. Angle vertices in EqAngle are separated from their ray points.

    Returns ``True`` if the configuration is non-degenerate.
    """
    # Check 1: pairwise point separation
    names = list(coords.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if _dist(coords[names[i]], coords[names[j]]) < min_sep:
                return False

    for fact in assumptions:
        p, args = fact.predicate, fact.args
        try:
            pts = [coords[a] for a in args]
        except KeyError:
            continue

        # Check 2: triangle non-collinearity
        if p == "Circumcenter" and len(args) == 4:
            # args = (O, A, B, C) — A, B, C must form a non-degenerate triangle
            a, b, c = pts[1], pts[2], pts[3]
            if abs(_triangle_area(a, b, c)) < min_triangle_area:
                return False

        # Check 3: line-defining pairs have positive length
        if p in ("Perpendicular", "Parallel") and len(args) == 4:
            if _dist(pts[0], pts[1]) < min_sep:
                return False
            if _dist(pts[2], pts[3]) < min_sep:
                return False

        # Check 4: angle vertices separated from ray endpoints
        if p == "EqAngle" and len(args) == 6:
            # ∠ABC = ∠DEF — B and E are vertices
            if _dist(pts[1], pts[0]) < min_sep or _dist(pts[1], pts[2]) < min_sep:
                return False
            if _dist(pts[4], pts[3]) < min_sep or _dist(pts[4], pts[5]) < min_sep:
                return False

        # Check: SimTri / CongTri / EqArea triangles non-degenerate
        if p in ("SimTri", "CongTri", "EqArea") and len(args) == 6:
            t1 = (pts[0], pts[1], pts[2])
            t2 = (pts[3], pts[4], pts[5])
            if abs(_triangle_area(*t1)) < min_triangle_area:
                return False
            if abs(_triangle_area(*t2)) < min_triangle_area:
                return False

    return True


def check_premise_consistency(
    assumptions: Sequence[Fact],
    n_trials: int = 80,
    spread: float = 10.0,
    min_point_sep: float = 0.005,
) -> bool:
    """Check whether the assumption set is jointly satisfiable.

    Generates random coordinate layouts and attempts to satisfy all
    premises simultaneously.  After convergence, every assumption is
    re-verified with a *tight* tolerance (``1e-5``), and the
    configuration is checked for **non-degeneracy** — all distinctly
    named points must have pairwise distance ``>= min_point_sep``,
    and structural non-degeneracy (triangle vertices non-collinear,
    line-defining pairs non-coincident, etc.).

    Parameters
    ----------
    assumptions : sequence of Fact
        The premise set to check.
    n_trials : int
        Number of random trials.
    spread : float
        Coordinate range for random initialisation.
    min_point_sep : float
        Minimum pairwise distance between named points.

    Returns
    -------
    bool
        ``True`` if at least one non-degenerate, tight-tolerance
        configuration was found; ``False`` otherwise.
    """
    global _EPS

    # Collect point names
    all_points: List[str] = []
    seen: Set[str] = set()
    for fact in assumptions:
        for arg in fact.args:
            if arg not in seen:
                seen.add(arg)
                all_points.append(arg)
    if len(all_points) < 2:
        return True  # trivially consistent

    saved_eps = _EPS
    n_valid = 0

    for _ in range(n_trials):
        # Phase 1: generate coords with normal (loose) tolerance
        try:
            coords = _constrained_coords(
                all_points, assumptions,
                spread=spread, max_retries=120,
            )
        except (OverflowError, ValueError):
            continue
        if coords is None:
            continue

        # Phase 2: structural non-degeneracy (points distinct,
        # triangles non-collinear, lines well-defined, etc.)
        if not _check_structural_nondegeneracy(
            assumptions, coords, min_sep=min_point_sep,
        ):
            continue

        # Phase 3: re-verify ALL assumptions with strict tolerance
        _EPS = 1e-5
        try:
            all_ok = True
            for fact in assumptions:
                check = _check_predicate(fact.predicate, fact.args, coords)
                if check is None:
                    continue  # unsupported predicate — skip
                if not check:
                    all_ok = False
                    break
        except (OverflowError, ValueError):
            all_ok = False
        finally:
            _EPS = saved_eps

        if all_ok:
            n_valid += 1
            if n_valid >= 2:
                return True  # high confidence

    return n_valid > 0


def verify_premises_strict(
    assumptions: Sequence[Fact],
    goal: Optional[Fact] = None,
    n_trials: int = 200,
    spread: float = 10.0,
    min_point_sep: float = 0.01,
    min_valid: int = 5,
) -> Tuple[bool, int, int]:
    """Strictly verify that premises are jointly satisfiable.

    A stronger version of :func:`check_premise_consistency` used as a
    final acceptance gate.  Differences:

      - Runs more trials (default 200 vs 80)
      - Uses tighter ``min_point_sep`` (0.01 vs 0.005)
      - Requires more valid configurations (default 5 vs 2)
      - If *goal* is provided, also verifies that the goal holds in
        every valid configuration (detects spurious proofs)
      - Checks structural non-degeneracy (non-collinear triangles, etc.)

    Returns
    -------
    (ok, n_valid, n_total)
        ok    : True if n_valid >= min_valid
        n_valid : number of non-degenerate configs that passed
        n_total : number of trials attempted
    """
    global _EPS

    # Collect point names
    all_points: List[str] = []
    seen: Set[str] = set()
    for fact in assumptions:
        for arg in fact.args:
            if arg not in seen:
                seen.add(arg)
                all_points.append(arg)
    if goal is not None:
        for arg in goal.args:
            if arg not in seen:
                seen.add(arg)
                all_points.append(arg)
    if len(all_points) < 2:
        return True, n_trials, n_trials

    saved_eps = _EPS
    n_valid = 0

    for trial in range(n_trials):
        try:
            coords = _constrained_coords(
                all_points, assumptions,
                spread=spread, max_retries=150,
            )
        except (OverflowError, ValueError):
            continue
        if coords is None:
            continue

        # Structural non-degeneracy
        if not _check_structural_nondegeneracy(
            assumptions, coords, min_sep=min_point_sep,
        ):
            continue

        # Strict re-verification of ALL premises
        _EPS = 5e-6  # strict but achievable (tighter than normal 1e-5)
        try:
            all_ok = True
            for fact in assumptions:
                check = _check_predicate(fact.predicate, fact.args, coords)
                if check is None:
                    continue
                if not check:
                    all_ok = False
                    break
            # If goal provided, also verify goal holds
            if all_ok and goal is not None:
                goal_check = _check_predicate(goal.predicate, goal.args, coords)
                if goal_check is not None and not goal_check:
                    all_ok = False
        except (OverflowError, ValueError):
            all_ok = False
        finally:
            _EPS = saved_eps

        if all_ok:
            n_valid += 1

    ok = n_valid >= min_valid
    return ok, n_valid, n_trials


# ═══════════════════════════════════════════════════════════════════════
# Batch testing: filter a list of conjectures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PolyaRankedConjecture:
    """A conjecture with its Pólya confidence attached."""
    assumptions: List[Fact]
    goal: Fact
    polya_result: PolyaResult
    original_data: Optional[dict] = None  # pass-through for pipeline data


def polya_filter(
    conjectures: Sequence[dict],
    *,
    n_trials: int = 30,
    min_confidence: float = 0.80,
    verbose: bool = False,
) -> Tuple[List[dict], List[dict]]:
    """Filter conjectures using Pólya testing.

    Parameters
    ----------
    conjectures : list[dict]
        Each dict must have 'assumptions' (list[Fact]) and 'goal' (Fact).
    n_trials : int
        Trials per conjecture.
    min_confidence : float
        Minimum Pólya confidence to pass.
    verbose : bool
        Print progress.

    Returns
    -------
    (passed, rejected) : tuple of two lists
        passed: conjectures with confidence ≥ min_confidence (sorted
                by confidence descending).
        rejected: falsified or low-confidence conjectures.
    """
    passed: List[dict] = []
    rejected: List[dict] = []

    for i, conj in enumerate(conjectures):
        assm = conj.get("assumptions", [])
        goal = conj.get("goal")
        if not assm or goal is None:
            rejected.append(conj)
            continue

        pr = polya_test(assm, goal, n_trials=n_trials)
        conj["polya_result"] = pr

        if pr.falsified:
            rejected.append(conj)
            if verbose:
                print(f"  ✗ 猜想#{i+1}: 已证伪 (反例在第{pr.n_valid}次试验)")
        elif pr.confidence >= min_confidence:
            passed.append(conj)
            if verbose:
                print(f"  ✓ 猜想#{i+1}: 置信度 {pr.confidence:.2f} "
                      f"({pr.n_passed}/{pr.n_valid} 通过) — {pr.confidence_label}")
        else:
            rejected.append(conj)
            if verbose:
                print(f"  ? 猜想#{i+1}: 置信度 {pr.confidence:.2f} "
                      f"({pr.n_passed}/{pr.n_valid} 通过) — 证据不足，跳过")

    # Sort passed by confidence (descending) to prioritise most likely
    passed.sort(key=lambda c: c.get("polya_result", PolyaResult()).confidence,
                reverse=True)

    return passed, rejected
