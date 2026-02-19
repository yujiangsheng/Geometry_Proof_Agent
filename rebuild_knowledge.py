#!/usr/bin/env python3
"""rebuild_knowledge.py – Systematically rebuild the knowledge base.

Generates diverse problems targeting ALL 49 rules, solves them via
beam search, and populates:
  • proven_cache.jsonl  — cached proven sub-goals
  • experience.jsonl    — search episodes (success + failure)
  • failure_patterns.json — diagnosed failure modes
  • stats.json          — summary statistics

Also enriches the RAG document store with proof strategy patterns
extracted from successful proofs.

Usage:
    python rebuild_knowledge.py [--rounds N] [--problems-per-gen N]

Author:  Jiangsheng Yu
License: MIT
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from geometry_agent.dsl import (
    Fact, GeoState, Goal, Step,
    canonical_parallel, canonical_perp, canonical_tangent,
    canonical_circle, canonical_collinear, canonical_eq_angle,
    canonical_eqdist, canonical_eqratio, canonical_harmonic,
    canonical_pole_polar, canonical_concurrent, canonical_between,
    canonical_angle_bisect, canonical_circumcenter, canonical_inv_image,
    canonical_cyclic, canonical_midpoint, canonical_cong,
    canonical_radical_axis, canonical_sim_tri, canonical_eq_cross_ratio,
)
from geometry_agent.knowledge import KnowledgeStore
from geometry_agent.lean_bridge import MockLeanChecker
from geometry_agent.rules import Rule, default_rules
from geometry_agent.search import SearchConfig, SearchResult, beam_search

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("rebuild_knowledge")

# ── Point names ──────────────────────────────────────────────────────
POINT_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _pts(n: int) -> List[str]:
    return random.sample(POINT_NAMES, min(n, len(POINT_NAMES)))


# ══════════════════════════════════════════════════════════════════════
#  Problem Generators — one or more per rule family
# ══════════════════════════════════════════════════════════════════════
# Each generator returns (assumptions, goal) targeting specific rules.

def gen_parallel_transitivity() -> Tuple[List[Fact], Fact]:
    """Parallel transitivity chain: A∥B, B∥C ⊢ A∥C"""
    p = _pts(6)
    return (
        [canonical_parallel(p[0], p[1], p[2], p[3]),
         canonical_parallel(p[2], p[3], p[4], p[5])],
        canonical_parallel(p[0], p[1], p[4], p[5]),
    )


def gen_parallel_triple_trans() -> Tuple[List[Fact], Fact]:
    """3-step parallel transitivity: A∥B, B∥C, C∥D ⊢ A∥D"""
    p = _pts(8)
    return (
        [canonical_parallel(p[0], p[1], p[2], p[3]),
         canonical_parallel(p[2], p[3], p[4], p[5]),
         canonical_parallel(p[4], p[5], p[6], p[7])],
        canonical_parallel(p[0], p[1], p[6], p[7]),
    )


def gen_perp_transfer() -> Tuple[List[Fact], Fact]:
    """Parallel-perp transfer: A∥B, A⊥C ⊢ B⊥C"""
    p = _pts(6)
    return (
        [canonical_parallel(p[0], p[1], p[2], p[3]),
         canonical_perp(p[0], p[1], p[4], p[5])],
        canonical_perp(p[2], p[3], p[4], p[5]),
    )


def gen_mixed_chain_2() -> Tuple[List[Fact], Fact]:
    """Mixed parallel/perp chain length 2."""
    from geometry_agent.evolve import generate_mixed_chain
    return generate_mixed_chain(2)


def gen_mixed_chain_3() -> Tuple[List[Fact], Fact]:
    from geometry_agent.evolve import generate_mixed_chain
    return generate_mixed_chain(3)


def gen_mixed_chain_4() -> Tuple[List[Fact], Fact]:
    from geometry_agent.evolve import generate_mixed_chain
    return generate_mixed_chain(4)


def gen_midsegment_parallel() -> Tuple[List[Fact], Fact]:
    """Midsegment → parallel: Mid(M,A,B), Mid(N,A,C) ⊢ MN∥BC"""
    p = _pts(5)
    A, B, C, M, N = p[0], p[1], p[2], p[3], p[4]
    return (
        [Fact("Midpoint", (M, A, B)), Fact("Midpoint", (N, A, C))],
        canonical_parallel(M, N, B, C),
    )


def gen_midseg_parallel_chain() -> Tuple[List[Fact], Fact]:
    """Midsegment then parallel transitivity."""
    p = _pts(7)
    A, B, C, M, N, D, E = p[:7]
    return (
        [Fact("Midpoint", (M, A, B)), Fact("Midpoint", (N, A, C)),
         canonical_parallel(B, C, D, E)],
        canonical_parallel(M, N, D, E),
    )


def gen_midseg_perp() -> Tuple[List[Fact], Fact]:
    """Midsegment → perp transfer: MN∥BC, BC⊥DE ⊢ MN⊥DE"""
    p = _pts(7)
    A, B, C, M, N, D, E = p[:7]
    return (
        [Fact("Midpoint", (M, A, B)), Fact("Midpoint", (N, A, C)),
         canonical_perp(B, C, D, E)],
        canonical_perp(M, N, D, E),
    )


def gen_midpoint_cong() -> Tuple[List[Fact], Fact]:
    """Midpoint → congruence: Mid(M,A,B) ⊢ Cong(A,M,M,B)"""
    p = _pts(3)
    M, A, B = p[0], p[1], p[2]
    return (
        [Fact("Midpoint", (M, A, B))],
        Fact("Cong", (A, M, M, B)),
    )


def gen_midpoint_cong_trans() -> Tuple[List[Fact], Fact]:
    """Two midpoints → cong trans: Mid(M,A,B), Mid(N,C,D), Cong(A,B,C,D) ⊢ Cong(A,M,C,N)"""
    p = _pts(6)
    M, A, B, N, C, D = p[:6]
    return (
        [Fact("Midpoint", (M, A, B)), Fact("Midpoint", (N, C, D)),
         Fact("Cong", (A, B, C, D))],
        Fact("Cong", (A, M, C, N)),
    )


def gen_cong_trans() -> Tuple[List[Fact], Fact]:
    """Congruence transitivity: Cong(A,B,C,D), Cong(C,D,E,F) ⊢ Cong(A,B,E,F)"""
    p = _pts(6)
    return (
        [Fact("Cong", (p[0], p[1], p[2], p[3])),
         Fact("Cong", (p[2], p[3], p[4], p[5]))],
        Fact("Cong", (p[0], p[1], p[4], p[5])),
    )


def gen_cong_symm() -> Tuple[List[Fact], Fact]:
    """Congruence symmetry: Cong(A,B,C,D) ⊢ Cong(C,D,A,B)"""
    p = _pts(4)
    return (
        [Fact("Cong", (p[0], p[1], p[2], p[3]))],
        Fact("Cong", (p[2], p[3], p[0], p[1])),
    )


def gen_perp_bisector_cong() -> Tuple[List[Fact], Fact]:
    """Perp bisector → cong: Mid(M,A,B), Perp(P,M,A,B) ⊢ Cong(P,A,P,B)"""
    p = _pts(4)
    P, M, A, B = p[:4]
    return (
        [Fact("Midpoint", (M, A, B)), canonical_perp(P, M, A, B)],
        Fact("Cong", (P, A, P, B)),
    )


def gen_cong_perp_bisector() -> Tuple[List[Fact], Fact]:
    """Cong → perp bisector (reverse): Cong(P,A,P,B), Mid(M,A,B) ⊢ Perp(P,M,A,B)"""
    p = _pts(4)
    P, M, A, B = p[:4]
    return (
        [Fact("Cong", (P, A, P, B)), Fact("Midpoint", (M, A, B))],
        canonical_perp(P, M, A, B),
    )


def gen_isosceles_base_angle() -> Tuple[List[Fact], Fact]:
    """Isosceles → equal angles: Cong(A,B,A,C) ⊢ EqAngle(A,B,C,A,C,B)"""
    p = _pts(3)
    A, B, C = p[:3]
    return (
        [Fact("Cong", (A, B, A, C))],
        Fact("EqAngle", (A, B, C, A, C, B)),
    )


def gen_eq_angle_trans() -> Tuple[List[Fact], Fact]:
    """Equal angle transitivity."""
    p = _pts(9)
    return (
        [Fact("EqAngle", (p[0], p[1], p[2], p[3], p[4], p[5])),
         Fact("EqAngle", (p[3], p[4], p[5], p[6], p[7], p[8]))],
        Fact("EqAngle", (p[0], p[1], p[2], p[6], p[7], p[8])),
    )


def gen_cyclic_inscribed_angle() -> Tuple[List[Fact], Fact]:
    """Cyclic → inscribed angle: Cyclic(A,B,C,D) ⊢ EqAngle(B,A,C,B,D,C)"""
    p = _pts(4)
    A, B, C, D = p[:4]
    cyc = canonical_cyclic(A, B, C, D)
    a, b, c, d = cyc.args
    return (
        [cyc],
        canonical_eq_angle(b, a, c, b, d, c),
    )


def gen_cyclic_angle_chain() -> Tuple[List[Fact], Fact]:
    """Cyclic inscribed angle + angle transitivity."""
    p = _pts(7)
    A, B, C, D, E, F, G = p[:7]
    cyc = canonical_cyclic(A, B, C, D)
    a, b, c, d = cyc.args
    # Cyclic produces EqAngle(b,a,c, b,d,c)
    # Chain via eq_angle_trans: EqAngle(b,d,c, E,F,G) → EqAngle(b,a,c, E,F,G)
    return (
        [cyc,
         Fact("EqAngle", (b, d, c, E, F, G))],
        canonical_eq_angle(b, a, c, E, F, G),
    )


def gen_parallel_alt_angle() -> Tuple[List[Fact], Fact]:
    """Parallel + Collinear → alternate angle."""
    p = _pts(5)
    A, B, C, D, X = p[:5]
    par = canonical_parallel(A, B, C, D)
    a, b, c, d = par.args
    # Transversal: Collinear(b, X, c) — shares b with line ab, c with line cd
    coll = canonical_collinear(b, X, c)
    # Rule output: EqAngle(ob=a, pa=b, x=X, od=d, pc=c, x=X)
    return (
        [par, coll],
        canonical_eq_angle(a, b, X, d, c, X),
    )


def gen_cyclic_chord_angle() -> Tuple[List[Fact], Fact]:
    """Cyclic → chord angle: Cyclic(A,B,C,D) ⊢ EqAngle(A,B,D,A,C,D)"""
    p = _pts(4)
    A, B, C, D = p[:4]
    cyc = canonical_cyclic(A, B, C, D)
    a, b, c, d = cyc.args
    return (
        [cyc],
        canonical_eq_angle(a, b, d, a, c, d),
    )


def gen_sim_tri_angle() -> Tuple[List[Fact], Fact]:
    """Similar triangles → equal angles."""
    p = _pts(6)
    A, B, C, D, E, F = p[:6]
    return (
        [Fact("SimTri", (A, B, C, D, E, F))],
        Fact("EqAngle", (B, A, C, E, D, F)),
    )


def gen_sim_tri_cong() -> Tuple[List[Fact], Fact]:
    """Similar + specific sides → congruence."""
    p = _pts(6)
    A, B, C, D, E, F = p[:6]
    return (
        [Fact("SimTri", (A, B, C, D, E, F)),
         Fact("Cong", (A, B, D, E))],
        Fact("Cong", (B, C, E, F)),
    )


def gen_midsegment_sim() -> Tuple[List[Fact], Fact]:
    """Midsegment → similar triangle."""
    p = _pts(5)
    A, B, C, M, N = p[:5]
    return (
        [Fact("Midpoint", (M, A, B)), Fact("Midpoint", (N, A, C))],
        Fact("SimTri", (A, M, N, A, B, C)),
    )


def gen_congtri_side() -> Tuple[List[Fact], Fact]:
    """Congruent triangles → corresponding sides."""
    p = _pts(6)
    A, B, C, D, E, F = p[:6]
    return (
        [Fact("CongTri", (A, B, C, D, E, F))],
        Fact("Cong", (A, B, D, E)),
    )


def gen_congtri_angle() -> Tuple[List[Fact], Fact]:
    """Congruent triangles → corresponding angles."""
    p = _pts(6)
    A, B, C, D, E, F = p[:6]
    return (
        [Fact("CongTri", (A, B, C, D, E, F))],
        Fact("EqAngle", (B, A, C, E, D, F)),
    )


def gen_congtri_eqarea() -> Tuple[List[Fact], Fact]:
    """Congruent triangles → equal area."""
    p = _pts(6)
    A, B, C, D, E, F = p[:6]
    return (
        [Fact("CongTri", (A, B, C, D, E, F))],
        Fact("EqArea", (A, B, C, D, E, F)),
    )


def gen_congtri_from_sim_cong() -> Tuple[List[Fact], Fact]:
    """Similar + corresponding side cong → congruent triangles."""
    p = _pts(6)
    A, B, C, D, E, F = p[:6]
    return (
        [Fact("SimTri", (A, B, C, D, E, F)),
         Fact("Cong", (A, B, D, E))],
        Fact("CongTri", (A, B, C, D, E, F)),
    )


def gen_tangent_perp_radius() -> Tuple[List[Fact], Fact]:
    """Tangent(A,B,O,P) → Perp(O,P,A,B): radius ⊥ tangent at point of tangency."""
    p = _pts(4)
    A, B, O, P = p[:4]
    tang = canonical_tangent(A, B, O, P)
    a, b, o, pp = tang.args
    return (
        [tang],
        canonical_perp(o, pp, a, b),
    )


def gen_tangent_oncircle() -> Tuple[List[Fact], Fact]:
    """Tangent(A,B,O,P) → OnCircle(O,P): tangent point on circle."""
    p = _pts(4)
    A, B, O, P = p[:4]
    tang = canonical_tangent(A, B, O, P)
    a, b, o, pp = tang.args
    return (
        [tang],
        canonical_circle(o, pp),
    )


def gen_eqratio_from_simtri() -> Tuple[List[Fact], Fact]:
    """SimTri → EqRatio: AB/DE = AC/DF."""
    p = _pts(6)
    A, B, C, D, E, F = p[:6]
    sim = canonical_sim_tri(A, B, C, D, E, F)
    a, b, c, d, e, f = sim.args
    return (
        [sim],
        canonical_eqratio(a, b, d, e, a, c, d, f),
    )


def gen_eqratio_trans() -> Tuple[List[Fact], Fact]:
    """EqRatio transitivity."""
    p = _pts(12)
    return (
        [Fact("EqRatio", (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])),
         Fact("EqRatio", (p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11]))],
        Fact("EqRatio", (p[0], p[1], p[2], p[3], p[8], p[9], p[10], p[11])),
    )


def gen_eqratio_symm() -> Tuple[List[Fact], Fact]:
    """EqRatio symmetry."""
    p = _pts(8)
    return (
        [Fact("EqRatio", (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]))],
        Fact("EqRatio", (p[4], p[5], p[6], p[7], p[0], p[1], p[2], p[3])),
    )


def gen_between_collinear() -> Tuple[List[Fact], Fact]:
    """Between → Collinear."""
    p = _pts(3)
    A, B, C = p[:3]
    return (
        [Fact("Between", (A, B, C))],
        Fact("Collinear", (A, B, C)),
    )


def gen_midpoint_between() -> Tuple[List[Fact], Fact]:
    """Midpoint → Between."""
    p = _pts(3)
    M, A, B = p[:3]
    return (
        [Fact("Midpoint", (M, A, B))],
        Fact("Between", (A, M, B)),
    )


def gen_midpoint_collinear() -> Tuple[List[Fact], Fact]:
    """Midpoint → Collinear."""
    p = _pts(3)
    M, A, B = p[:3]
    return (
        [Fact("Midpoint", (M, A, B))],
        Fact("Collinear", (A, M, B)),
    )


def gen_angle_bisect_eq_angle() -> Tuple[List[Fact], Fact]:
    """AngleBisect(A,P,B,C) → EqAngle(B,A,P, P,A,C): bisector splits angle."""
    p = _pts(4)
    A, P, B, C = p[:4]
    ab = canonical_angle_bisect(A, P, B, C)
    a, pp, b, c = ab.args
    return (
        [ab],
        canonical_eq_angle(b, a, pp, pp, a, c),
    )


def gen_angle_bisect_eqratio() -> Tuple[List[Fact], Fact]:
    """AngleBisect(A,P,B,C) + Between(B,P,C) → EqRatio(B,P,P,C,A,B,A,C)."""
    p = _pts(4)
    A, P, B, C = p[:4]
    ab = canonical_angle_bisect(A, P, B, C)
    a, pp, b, c = ab.args
    bet = canonical_between(b, pp, c)
    return (
        [ab, bet],
        canonical_eqratio(b, pp, pp, c, a, b, a, c),
    )


def gen_medians_concurrent() -> Tuple[List[Fact], Fact]:
    """Triangle medians → Concurrent."""
    p = _pts(6)
    A, B, C, D, E, F = p[:6]
    m1 = canonical_midpoint(D, B, C)
    m2 = canonical_midpoint(E, A, C)
    m3 = canonical_midpoint(F, A, B)
    return (
        [m1, m2, m3],
        canonical_concurrent(A, D, B, E, C, F),
    )


def gen_circumcenter_cong_ab() -> Tuple[List[Fact], Fact]:
    """Circumcenter → Cong(O,A,O,B)."""
    p = _pts(4)
    O, A, B, C = p[:4]
    return (
        [Fact("Circumcenter", (O, A, B, C))],
        Fact("Cong", (O, A, O, B)),
    )


def gen_circumcenter_cong_bc() -> Tuple[List[Fact], Fact]:
    """Circumcenter → Cong(O,B,O,C)."""
    p = _pts(4)
    O, A, B, C = p[:4]
    return (
        [Fact("Circumcenter", (O, A, B, C))],
        Fact("Cong", (O, B, O, C)),
    )


def gen_circumcenter_oncircle() -> Tuple[List[Fact], Fact]:
    """Circumcenter(O,A,B,C) → OnCircle(O,A): A on circumscribed circle."""
    p = _pts(4)
    O, A, B, C = p[:4]
    cc = canonical_circumcenter(O, A, B, C)
    o, a, _b, _c = cc.args
    return (
        [cc],
        canonical_circle(o, a),
    )


def gen_circumcenter_cong_chain() -> Tuple[List[Fact], Fact]:
    """Circumcenter → two Cong → transitivity."""
    p = _pts(4)
    O, A, B, C = p[:4]
    return (
        [Fact("Circumcenter", (O, A, B, C))],
        Fact("Cong", (O, A, O, C)),
    )


def gen_eqdist_from_cong() -> Tuple[List[Fact], Fact]:
    """Cong(P,A,P,B) → EqDist(P,A,B): shared endpoint means equidistant."""
    p = _pts(3)
    P, A, B = p[:3]
    cong = canonical_cong(P, A, P, B)
    return (
        [cong],
        canonical_eqdist(P, A, B),
    )


def gen_eqdist_to_cong() -> Tuple[List[Fact], Fact]:
    """EqDist(P,A,B) → Cong(P,A,P,B): equidistant means congruent segments."""
    p = _pts(3)
    P, A, B = p[:3]
    ed = canonical_eqdist(P, A, B)
    pp, a, b = ed.args
    return (
        [ed],
        canonical_cong(pp, a, pp, b),
    )


def gen_eqarea_sym() -> Tuple[List[Fact], Fact]:
    """EqArea symmetry."""
    p = _pts(6)
    A, B, C, D, E, F = p[:6]
    return (
        [Fact("EqArea", (A, B, C, D, E, F))],
        Fact("EqArea", (D, E, F, A, B, C)),
    )


def gen_harmonic_swap() -> Tuple[List[Fact], Fact]:
    """Harmonic(A,B,C,D) → Harmonic(B,A,D,C): swap both pairs."""
    p = _pts(4)
    A, B, C, D = p[:4]
    h = canonical_harmonic(A, B, C, D)
    a, b, c, d = h.args
    return (
        [h],
        canonical_harmonic(b, a, d, c),
    )


def gen_harmonic_collinear() -> Tuple[List[Fact], Fact]:
    """Harmonic(A,B,C,D) → Collinear(A,C,D): args 0,2,3 collinear."""
    p = _pts(4)
    A, B, C, D = p[:4]
    h = canonical_harmonic(A, B, C, D)
    a, _b, c, d = h.args
    return (
        [h],
        canonical_collinear(a, c, d),
    )


def gen_pole_polar_perp() -> Tuple[List[Fact], Fact]:
    """PolePolar(P,A,B,O) → Perp(O,P,A,B): polar ⊥ line from pole to center."""
    p = _pts(4)
    P, A, B, O = p[:4]
    pp = canonical_pole_polar(P, A, B, O)
    pole, a, b, o = pp.args
    return (
        [pp],
        canonical_perp(o, pole, a, b),
    )


def gen_pole_polar_tangent() -> Tuple[List[Fact], Fact]:
    """PolePolar(P,A,B,O) + OnCircle(O,A) → Tangent(P,A,O,A): polar endpoint on circle."""
    p = _pts(4)
    P, A, B, O = p[:4]
    pp = canonical_pole_polar(P, A, B, O)
    pole, a, b, o = pp.args
    oc = canonical_circle(o, a)
    return (
        [pp, oc],
        canonical_tangent(pole, a, o, a),
    )


def gen_inversion_collinear() -> Tuple[List[Fact], Fact]:
    """InvImage(P',P,O,A) → Collinear(O,P,P'): O, P, P' collinear under inversion."""
    p = _pts(4)
    PP, P, O, A = p[:4]
    inv = canonical_inv_image(PP, P, O, A)
    pp, pt, o, _a = inv.args
    return (
        [inv],
        canonical_collinear(o, pt, pp),
    )


def gen_inversion_circle_fixed() -> Tuple[List[Fact], Fact]:
    """InvImage(P',P,O,A) + OnCircle(O,P) → OnCircle(O,P'): inversion circle points fixed."""
    p = _pts(4)
    PP, P, O, A = p[:4]
    inv = canonical_inv_image(PP, P, O, A)
    pp, pt, o, _a = inv.args
    oc = canonical_circle(o, pt)
    return (
        [inv, oc],
        canonical_circle(o, pp),
    )


def gen_cross_ratio_sym() -> Tuple[List[Fact], Fact]:
    """EqCrossRatio symmetry."""
    p = _pts(8)
    return (
        [Fact("EqCrossRatio", (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]))],
        Fact("EqCrossRatio", (p[4], p[5], p[6], p[7], p[0], p[1], p[2], p[3])),
    )


def gen_cross_ratio_from_harmonic() -> Tuple[List[Fact], Fact]:
    """Two harmonic ranges → EqCrossRatio."""
    p = _pts(8)
    A, B, C, D, E, F, G, H = p[:8]
    h1 = canonical_harmonic(A, B, C, D)
    h2 = canonical_harmonic(E, F, G, H)
    return (
        [h1, h2],
        canonical_eq_cross_ratio(*h1.args, *h2.args),
    )


def gen_radical_axis_perp() -> Tuple[List[Fact], Fact]:
    """RadicalAxis(A,B,O1,O2) → Perp(A,B,O1,O2): radical axis ⊥ line of centers."""
    p = _pts(4)
    A, B, O1, O2 = p[:4]
    ra = canonical_radical_axis(A, B, O1, O2)
    a, b, o1, o2 = ra.args
    return (
        [ra],
        canonical_perp(a, b, o1, o2),
    )


# ── Converse / production generators for input-only predicates ──────

def gen_midpoint_from_cong_between() -> Tuple[List[Fact], Fact]:
    """Cong(A,M,M,B) + Between(A,M,B) → Midpoint(M,A,B)."""
    p = _pts(3)
    A, M, B = p[:3]
    bet = canonical_between(A, M, B)
    a, m, b = bet.args
    cng = canonical_cong(a, m, m, b)
    return (
        [cng, bet],
        canonical_midpoint(m, a, b),
    )


def gen_cyclic_from_eq_angle() -> Tuple[List[Fact], Fact]:
    """EqAngle(B,A,C,B,D,C) → Cyclic(A,B,C,D): converse inscribed angle."""
    p = _pts(4)
    A, B, C, D = p[:4]
    ea = canonical_eq_angle(B, A, C, B, D, C)
    # Extract the args back to determine the Cyclic output
    bb, aa, cc, bb2, dd, cc2 = ea.args
    return (
        [ea],
        canonical_cyclic(aa, bb, cc, dd),
    )


def gen_circumcenter_from_eqdist() -> Tuple[List[Fact], Fact]:
    """EqDist(O,A,B) + EqDist(O,B,C) → Circumcenter(O,A,B,C)."""
    p = _pts(4)
    O, A, B, C = p[:4]
    ed1 = canonical_eqdist(O, A, B)
    ed2 = canonical_eqdist(O, B, C)
    return (
        [ed1, ed2],
        canonical_circumcenter(O, A, B, C),
    )


def gen_angle_bisect_from_eq_angle() -> Tuple[List[Fact], Fact]:
    """EqAngle(B,A,P,P,A,C) → AngleBisect(A,P,B,C): converse angle bisector."""
    p = _pts(4)
    A, P, B, C = p[:4]
    ea = canonical_eq_angle(B, A, P, P, A, C)
    bb, aa, pp, pp2, aa2, cc = ea.args
    return (
        [ea],
        canonical_angle_bisect(aa, pp, bb, cc),
    )


def gen_pole_polar_from_tangents() -> Tuple[List[Fact], Fact]:
    """Tangent(P,A,O,A) + Tangent(P,B,O,B) → PolePolar(P,A,B,O)."""
    p = _pts(4)
    P, A, B, O = p[:4]
    t1 = canonical_tangent(P, A, O, A)
    t2 = canonical_tangent(P, B, O, B)
    return (
        [t1, t2],
        canonical_pole_polar(P, A, B, O),
    )


def gen_radical_axis_from_common_pts() -> Tuple[List[Fact], Fact]:
    """OnCircle(O1,A) + OnCircle(O2,A) + OnCircle(O1,B) + OnCircle(O2,B)
       → RadicalAxis(A,B,O1,O2): common chord is radical axis."""
    p = _pts(4)
    A, B, O1, O2 = p[:4]
    oc1a = canonical_circle(O1, A)
    oc2a = canonical_circle(O2, A)
    oc1b = canonical_circle(O1, B)
    oc2b = canonical_circle(O2, B)
    return (
        [oc1a, oc2a, oc1b, oc2b],
        canonical_radical_axis(A, B, O1, O2),
    )


def gen_inv_image_from_self() -> Tuple[List[Fact], Fact]:
    """Collinear(O,P,P') + OnCircle(O,P) + Cong(O,P,O,P')
       → InvImage(P',P,O,P): self-inversion on the circle."""
    p = _pts(3)
    O, P, PP = p[:3]
    coll = canonical_collinear(O, P, PP)
    oc = canonical_circle(O, P)
    cng = canonical_cong(O, P, O, PP)
    return (
        [coll, oc, cng],
        canonical_inv_image(PP, P, O, P),
    )


# ── Multi-step cross-family generators ──────────────────────────────

def gen_iso_base_angle_cyclic_chain() -> Tuple[List[Fact], Fact]:
    """Isosceles + Cyclic → EqAngle chain."""
    p = _pts(7)
    A, B, C, D, E, F, G = p[:7]
    iso = canonical_cong(A, B, A, C)
    cyc = canonical_cyclic(D, E, F, G)
    d, e, f, g = cyc.args
    # iso_base_angle: Cong(A,B,A,C) → EqAngle(A,B,C, A,C,B)
    # cyclic: Cyclic(d,e,f,g) → EqAngle(e,d,f, e,g,f)
    # Bridge: EqAngle(A,C,B, e,d,f) lets eq_angle_trans combine with iso_base_angle output
    # Chain: EqAngle(A,B,C,A,C,B) + EqAngle(A,C,B,e,d,f) → EqAngle(A,B,C,e,d,f)
    #   then: EqAngle(e,d,f,e,g,f) + EqAngle(A,B,C,e,d,f) → would need eq_angle_trans
    # Simplified 2-step: iso_base_angle + eq_angle_trans
    return (
        [iso, Fact("EqAngle", (A, C, B, D, E, F))],
        canonical_eq_angle(A, B, C, D, E, F),
    )


def gen_perp_bisector_cong_trans() -> Tuple[List[Fact], Fact]:
    """Perp bisector → cong → cong transitivity."""
    p = _pts(6)
    P, M, A, B, C, D = p[:6]
    return (
        [Fact("Midpoint", (M, A, B)),
         canonical_perp(P, M, A, B),
         Fact("Cong", (P, B, C, D))],
        Fact("Cong", (P, A, C, D)),
    )


def gen_circumcenter_iso_angle() -> Tuple[List[Fact], Fact]:
    """Circumcenter → cong → isosceles → equal angles."""
    p = _pts(4)
    O, A, B, C = p[:4]
    return (
        [Fact("Circumcenter", (O, A, B, C))],
        Fact("EqAngle", (A, O, B, A, O, B)),
    )


def gen_midseg_sim_angle_chain() -> Tuple[List[Fact], Fact]:
    """Midsegment → SimTri → EqAngle."""
    p = _pts(5)
    A, B, C, M, N = p[:5]
    return (
        [Fact("Midpoint", (M, A, B)), Fact("Midpoint", (N, A, C))],
        Fact("EqAngle", (M, A, N, B, A, C)),
    )


def gen_congtri_sim_cong_eqarea() -> Tuple[List[Fact], Fact]:
    """SimTri + Cong → CongTri → EqArea."""
    p = _pts(6)
    A, B, C, D, E, F = p[:6]
    return (
        [Fact("SimTri", (A, B, C, D, E, F)),
         Fact("Cong", (A, B, D, E))],
        Fact("EqArea", (A, B, C, D, E, F)),
    )


def gen_double_perp_bisector() -> Tuple[List[Fact], Fact]:
    """Two perp bisectors → cong trans."""
    p = _pts(5)
    P, M, A, B, N = p[:5]
    return (
        [Fact("Midpoint", (M, A, B)),
         canonical_perp(P, M, A, B),
         Fact("Midpoint", (N, A, B)),
         canonical_perp(P, N, A, B)],
        Fact("Cong", (P, A, P, B)),
    )


def gen_tangent_perp_parallel() -> Tuple[List[Fact], Fact]:
    """Tangent → perp → parallel-perp transfer."""
    p = _pts(6)
    T, X, O, P, A, B = p[:6]
    tang = canonical_tangent(T, X, O, P)
    t, x, o, pp = tang.args
    # tangent_perp_radius produces Perp(o,pp,t,x)
    # parallel_perp_trans: Parallel(o,pp,A,B) + Perp(o,pp,t,x) → Perp(A,B,t,x)
    par = canonical_parallel(o, pp, A, B)
    return (
        [tang, par],
        canonical_perp(A, B, t, x),
    )


def gen_eqdist_cong_trans() -> Tuple[List[Fact], Fact]:
    """EqDist(P,A,B) → Cong(P,A,P,B) → Cong trans."""
    p = _pts(5)
    P, A, B, E, F = p[:5]
    ed = canonical_eqdist(P, A, B)
    pp, a, b = ed.args
    # eqdist_to_cong: Cong(pp,a,pp,b), then cong_trans with Cong(pp,b,E,F) → Cong(pp,a,E,F)
    return (
        [ed, canonical_cong(pp, b, E, F)],
        canonical_cong(pp, a, E, F),
    )


# ══════════════════════════════════════════════════════════════════════
#  Master generator registry
# ══════════════════════════════════════════════════════════════════════

GENERATORS: List[Tuple[str, Callable, List[str]]] = [
    # (name, callable, target_rules)
    # ── Parallel/Perp family ──
    ("par_trans",            gen_parallel_transitivity,     ["parallel_transitivity", "parallel_symmetry"]),
    ("par_triple_trans",     gen_parallel_triple_trans,     ["parallel_transitivity", "parallel_symmetry"]),
    ("perp_transfer",        gen_perp_transfer,             ["parallel_perp_trans", "perp_symmetry"]),
    ("mixed_2",              gen_mixed_chain_2,             ["parallel_transitivity", "parallel_perp_trans"]),
    ("mixed_3",              gen_mixed_chain_3,             ["parallel_transitivity", "parallel_perp_trans"]),
    ("mixed_4",              gen_mixed_chain_4,             ["parallel_transitivity", "parallel_perp_trans"]),
    # ── Midpoint/Midsegment family ──
    ("midseg_par",           gen_midsegment_parallel,       ["midsegment_parallel"]),
    ("midseg_par_chain",     gen_midseg_parallel_chain,     ["midsegment_parallel", "parallel_transitivity"]),
    ("midseg_perp",          gen_midseg_perp,               ["midsegment_parallel", "parallel_perp_trans"]),
    ("mid_cong",             gen_midpoint_cong,             ["midpoint_cong"]),
    ("mid_cong_trans",       gen_midpoint_cong_trans,       ["midpoint_cong", "cong_trans"]),
    ("mid_collinear",        gen_midpoint_collinear,        ["midpoint_collinear"]),
    ("mid_between",          gen_midpoint_between,          ["midpoint_between"]),
    # ── Congruence family ──
    ("cong_trans",           gen_cong_trans,                ["cong_trans"]),
    ("cong_symm",            gen_cong_symm,                 ["cong_symm"]),
    ("perp_bis_cong",        gen_perp_bisector_cong,        ["perp_bisector_cong"]),
    ("cong_perp_bis",        gen_cong_perp_bisector,        ["cong_perp_bisector"]),
    ("perp_bis_cong_trans",  gen_perp_bisector_cong_trans,  ["perp_bisector_cong", "cong_trans"]),
    ("dbl_perp_bis",         gen_double_perp_bisector,      ["perp_bisector_cong"]),
    # ── Angle family ──
    ("iso_base_angle",       gen_isosceles_base_angle,      ["isosceles_base_angle"]),
    ("eq_angle_trans",       gen_eq_angle_trans,            ["eq_angle_trans", "eq_angle_symm"]),
    ("cyclic_angle",         gen_cyclic_inscribed_angle,    ["cyclic_inscribed_angle"]),
    ("cyclic_chord",         gen_cyclic_chord_angle,        ["cyclic_chord_angle"]),
    ("cyclic_angle_chain",   gen_cyclic_angle_chain,        ["cyclic_inscribed_angle", "eq_angle_trans"]),
    ("par_alt_angle",        gen_parallel_alt_angle,        ["parallel_alternate_angle"]),
    ("angle_bisect_eq",      gen_angle_bisect_eq_angle,     ["angle_bisect_eq_angle"]),
    ("angle_bisect_ratio",   gen_angle_bisect_eqratio,      ["angle_bisect_eqratio"]),
    # ── Similar/Congruent triangles ──
    ("sim_tri_angle",        gen_sim_tri_angle,             ["sim_tri_angle"]),
    ("sim_tri_cong",         gen_sim_tri_cong,              ["sim_tri_cong"]),
    ("midseg_sim",           gen_midsegment_sim,            ["midsegment_sim_tri"]),
    ("congtri_side",         gen_congtri_side,              ["congtri_side"]),
    ("congtri_angle",        gen_congtri_angle,             ["congtri_angle"]),
    ("congtri_eqarea",       gen_congtri_eqarea,            ["congtri_eqarea"]),
    ("congtri_from_sim",     gen_congtri_from_sim_cong,     ["congtri_from_sim_cong"]),
    ("congtri_sim_eqarea",   gen_congtri_sim_cong_eqarea,   ["congtri_from_sim_cong", "congtri_eqarea"]),
    # ── Ratio family ──
    ("eqratio_simtri",       gen_eqratio_from_simtri,       ["eqratio_from_simtri"]),
    ("eqratio_symm",         gen_eqratio_symm,              ["eqratio_sym"]),
    ("eqratio_trans",        gen_eqratio_trans,             ["eqratio_trans"]),
    # ── Between/Collinear ──
    ("between_coll",         gen_between_collinear,         ["between_collinear"]),
    # ── Circumcenter family ──
    ("circ_cong_ab",         gen_circumcenter_cong_ab,      ["circumcenter_cong_ab"]),
    ("circ_cong_bc",         gen_circumcenter_cong_bc,      ["circumcenter_cong_bc"]),
    ("circ_oncircle",        gen_circumcenter_oncircle,     ["circumcenter_oncircle"]),
    ("circ_cong_chain",      gen_circumcenter_cong_chain,   ["circumcenter_cong_ab", "circumcenter_cong_bc", "cong_trans"]),
    ("circ_iso_angle",       gen_circumcenter_iso_angle,    ["circumcenter_cong_ab", "isosceles_base_angle"]),
    # ── Tangent family ──
    ("tangent_perp",         gen_tangent_perp_radius,       ["tangent_perp_radius"]),
    ("tangent_oncircle",     gen_tangent_oncircle,          ["tangent_oncircle"]),
    ("tangent_perp_par",     gen_tangent_perp_parallel,     ["tangent_perp_radius", "parallel_perp_trans"]),
    # ── Medians ──
    ("medians_concurrent",   gen_medians_concurrent,        ["medians_concurrent"]),
    # ── EqDist ──
    ("eqdist_from_cong",     gen_eqdist_from_cong,          ["eqdist_from_cong"]),
    ("eqdist_to_cong",       gen_eqdist_to_cong,            ["eqdist_to_cong"]),
    ("eqdist_cong_trans",    gen_eqdist_cong_trans,         ["eqdist_to_cong", "cong_trans"]),
    # ── EqArea ──
    ("eqarea_sym",           gen_eqarea_sym,                ["eqarea_sym"]),
    # ── Projective geometry ──
    ("harmonic_swap",        gen_harmonic_swap,             ["harmonic_swap"]),
    ("harmonic_coll",        gen_harmonic_collinear,        ["harmonic_collinear"]),
    ("pole_polar_perp",      gen_pole_polar_perp,           ["pole_polar_perp"]),
    ("pole_polar_tangent",   gen_pole_polar_tangent,        ["pole_polar_tangent"]),
    ("inv_collinear",        gen_inversion_collinear,       ["inversion_collinear"]),
    ("inv_circle",           gen_inversion_circle_fixed,    ["inversion_circle_fixed"]),
    ("cross_ratio_sym",      gen_cross_ratio_sym,           ["cross_ratio_sym"]),
    ("cross_ratio_harm",     gen_cross_ratio_from_harmonic, ["cross_ratio_from_harmonic"]),
    ("radical_axis",         gen_radical_axis_perp,         ["radical_axis_perp"]),
    # ── Converse / production rules ──
    ("mid_from_cong_bet",    gen_midpoint_from_cong_between,  ["midpoint_from_cong_between"]),
    ("cyclic_from_ea",       gen_cyclic_from_eq_angle,        ["cyclic_from_eq_angle"]),
    ("circ_from_eqdist",     gen_circumcenter_from_eqdist,    ["circumcenter_from_eqdist"]),
    ("ab_from_ea",           gen_angle_bisect_from_eq_angle,  ["angle_bisect_from_eq_angle"]),
    ("pp_from_tangents",     gen_pole_polar_from_tangents,    ["pole_polar_from_tangents"]),
    ("ra_from_common",       gen_radical_axis_from_common_pts, ["radical_axis_from_common_points"]),
    ("inv_from_self",        gen_inv_image_from_self,         ["inv_image_from_self"]),
    # ── Cross-family ──
    ("iso_cyclic_chain",     gen_iso_base_angle_cyclic_chain, ["isosceles_base_angle", "cyclic_inscribed_angle", "eq_angle_trans"]),
    ("midseg_sim_angle",     gen_midseg_sim_angle_chain,    ["midsegment_sim_tri", "sim_tri_angle"]),
]


# ══════════════════════════════════════════════════════════════════════
#  Main rebuild logic
# ══════════════════════════════════════════════════════════════════════

def rebuild_knowledge(
    rounds: int = 5,
    problems_per_gen: int = 3,
    beam_width: int = 16,
    max_depth: int = 14,
) -> KnowledgeStore:
    """Systematically generate problems for every rule, solve, and store."""
    store = KnowledgeStore()  # fresh store
    rules = default_rules()
    checker = MockLeanChecker()
    cfg = SearchConfig(beam_width=beam_width, max_depth=max_depth)

    total = 0
    success = 0
    rule_coverage = Counter()
    predicate_coverage = Counter()
    failure_diag = Counter()

    print(f"Rebuilding knowledge: {len(GENERATORS)} generators x {rounds} rounds x {problems_per_gen} problems")
    print(f"  Beam width: {beam_width}, Max depth: {max_depth}")
    print()

    for round_idx in range(rounds):
        for gen_name, gen_fn, target_rules in GENERATORS:
            for _ in range(problems_per_gen):
                try:
                    assumptions, goal = gen_fn()
                except Exception as e:
                    logger.warning("Generator %s failed: %s", gen_name, e)
                    continue

                total += 1
                state = GeoState(facts=set(assumptions))
                result = beam_search(
                    init_state=state,
                    goal=Goal(goal),
                    rules=rules,
                    checker=checker,
                    config=cfg,
                    knowledge_store=store,
                )

                if result.success:
                    success += 1
                    for r in target_rules:
                        rule_coverage[r] += 1
                    predicate_coverage[goal.predicate] += 1

                    # Record proven
                    store.record_proven(
                        assumptions=frozenset(assumptions),
                        goal=goal,
                        steps=list(result.final_state.history),
                        source=f"rebuild:{gen_name}",
                    )
                else:
                    # Diagnose failure
                    if result.explored_nodes == 0:
                        pattern = f"no-progress:{gen_name}"
                    else:
                        pattern = f"depth-exhausted:{gen_name}"
                    failure_diag[pattern] += 1
                    store.record_failure_pattern(pattern)

                # Record experience
                store.record_experience(
                    assumptions=list(assumptions),
                    goal=goal,
                    success=result.success,
                    steps=list(result.final_state.history),
                    explored_nodes=result.explored_nodes,
                    difficulty=len(assumptions),
                )

        # Progress report
        pct = 100 * success / max(total, 1)
        print(f"  Round {round_idx+1}/{rounds}: {total} problems, {success} solved ({pct:.1f}%)")

    print()
    print(f"=== Rebuild Complete ===")
    print(f"  Total problems: {total}")
    print(f"  Solved: {success} ({100*success/max(total,1):.1f}%)")
    print(f"  Proven cache: {store.stats().proven_cache_size}")
    print(f"  Experience: {store.stats().experience_total}")
    print(f"  Predicate coverage: {dict(predicate_coverage.most_common())}")
    print(f"  Rule coverage: {len(rule_coverage)}/49 rules")

    all_rules = {r.name for r in rules}
    uncovered = all_rules - set(rule_coverage.keys())
    if uncovered:
        print(f"  Uncovered rules: {uncovered}")

    if failure_diag:
        print(f"  Top failures: {dict(failure_diag.most_common(10))}")

    return store


# ══════════════════════════════════════════════════════════════════════
#  RAG document enrichment
# ══════════════════════════════════════════════════════════════════════

def enrich_rag_documents(store: KnowledgeStore, rag_path: str) -> int:
    """Generate new RAG documents from successful proof patterns."""
    import hashlib

    # Load existing docs to avoid duplicates
    existing_titles = set()
    path = Path(rag_path)
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    existing_titles.add(doc.get("title", ""))

    # Extract proof pattern documents from proven cache
    new_docs = []

    # 1. Rule combination patterns
    pattern_counter = Counter()
    rule_to_examples = defaultdict(list)
    for entry in store._proven.values():
        rules_used = [s.rule_name for s in entry.steps]
        pattern = " → ".join(rules_used)
        pattern_counter[pattern] += 1
        for r in set(rules_used):
            rule_to_examples[r].append(entry)

    # 2. Generate "proof strategy" documents for each rule
    RULE_DESCRIPTIONS = {
        "parallel_symmetry": ("平行线对称性", "若 AB∥CD，则 CD∥AB。常用作链式推理前的方向调整。"),
        "parallel_transitivity": ("平行线传递性", "若 AB∥CD 且 CD∥EF，则 AB∥EF。几何推理中最基本的传递链。"),
        "perp_symmetry": ("垂直对称性", "若 AB⊥CD，则 CD⊥AB。通常配合 parallel_perp_trans 使用。"),
        "parallel_perp_trans": ("平行-垂直传递", "若 AB∥CD 且 AB⊥EF，则 CD⊥EF。平行线将垂直关系传递到远方。"),
        "midpoint_collinear": ("中点共线", "Mid(M,A,B) → Collinear(A,M,B)。中点在两端点连线上。"),
        "midpoint_cong": ("中点全等", "Mid(M,A,B) → Cong(A,M,M,B)。中点等分线段。"),
        "midsegment_parallel": ("中位线平行", "Mid(M,A,B), Mid(N,A,C) → MN∥BC。三角形中位线定理的核心。"),
        "cong_symm": ("全等对称性", "Cong(A,B,C,D) → Cong(C,D,A,B)。调整全等方向以匹配后续规则。"),
        "cong_trans": ("全等传递性", "Cong(A,B,C,D), Cong(C,D,E,F) → Cong(A,B,E,F)。全等链传递。"),
        "eq_angle_symm": ("等角对称性", "EqAngle(A,B,C,D,E,F) → EqAngle(D,E,F,A,B,C)。"),
        "eq_angle_trans": ("等角传递性", "EqAngle链传递。常与圆周角、等腰三角形底角配合使用。"),
        "cyclic_inscribed_angle": ("圆周角定理", "Cyclic(A,B,C,D) → EqAngle。同弧上的圆周角相等。"),
        "cyclic_chord_angle": ("圆周角弦角", "圆内弦对应的角度关系。"),
        "perp_bisector_cong": ("垂直平分线 → 全等", "Mid(M,A,B), Perp(P,M,A,B) → Cong(P,A,P,B)。垂直平分线上的点到两端等距。"),
        "isosceles_base_angle": ("等腰底角", "Cong(A,B,A,C) → EqAngle(A,B,C,A,C,B)。等腰三角形底角相等。"),
        "cong_perp_bisector": ("全等 → 垂直平分线", "Cong(P,A,P,B), Mid(M,A,B) → Perp(P,M,A,B)。等距点在垂直平分线上。"),
        "parallel_alternate_angle": ("平行 → 内错角", "Parallel(A,B,C,D) → EqAngle。平行线的内错角相等。"),
        "midsegment_sim_tri": ("中位线 → 相似", "Mid(M,A,B), Mid(N,A,C) → SimTri(A,M,N,A,B,C)。中位线构成的小三角形与原三角形相似。"),
        "sim_tri_angle": ("相似 → 等角", "SimTri(A,B,C,D,E,F) → EqAngle。相似三角形对应角相等。"),
        "sim_tri_cong": ("相似+边 → 边全等", "SimTri + Cong(AB,DE) → Cong(BC,EF)。比例关系加上一边相等推出另一边相等。"),
        "congtri_side": ("全等三角形 → 对应边", "CongTri → Cong。全等三角形对应边相等。"),
        "congtri_angle": ("全等三角形 → 对应角", "CongTri → EqAngle。全等三角形对应角相等。"),
        "congtri_from_sim_cong": ("相似+边全等 → 全等", "SimTri + Cong → CongTri。相似且有一对对应边相等则全等。"),
        "congtri_eqarea": ("全等三角形 → 面积", "CongTri → EqArea。全等三角形面积相等。"),
        "tangent_perp_radius": ("切线 ⊥ 半径", "Tangent(T,O,R) → Perp(T,O,T,R)。切点处切线垂直于半径。"),
        "tangent_oncircle": ("切点在圆上", "Tangent(T,O,R) → OnCircle(T,O,R)。切点是圆上的点。"),
        "eqratio_from_simtri": ("相似 → 比例", "SimTri → EqRatio。相似三角形对应边成比例。"),
        "eqratio_sym": ("比例对称性", "EqRatio 对称：交换分子分母。"),
        "eqratio_trans": ("比例传递性", "EqRatio 链传递。"),
        "between_collinear": ("介于 → 共线", "Between(A,B,C) → Collinear(A,B,C)。"),
        "midpoint_between": ("中点 → 介于", "Mid(M,A,B) → Between(A,M,B)。"),
        "angle_bisect_eq_angle": ("角平分线 → 等角", "AngleBisect(A,B,C,D) → EqAngle。角平分线平分角度。"),
        "angle_bisect_eqratio": ("角平分线比例", "角平分线定理：对边成比例。"),
        "medians_concurrent": ("中线共点", "三角形三条中线交于一点（重心）。"),
        "circumcenter_cong_ab": ("外心 → OA=OB", "Circumcenter(O,A,B,C) → Cong(O,A,O,B)。外心到各顶点等距。"),
        "circumcenter_cong_bc": ("外心 → OB=OC", "Circumcenter(O,A,B,C) → Cong(O,B,O,C)。"),
        "circumcenter_oncircle": ("外心 → 顶点在圆上", "Circumcenter(O,A,B,C) → OnCircle(A,O,B)。"),
        "eqdist_from_cong": ("全等 → 等距", "Cong(A,B,C,D) → EqDist(A,B,C,D)。"),
        "eqdist_to_cong": ("等距 → 全等", "EqDist(A,B,C,D) → Cong(A,B,C,D)。"),
        "eqarea_sym": ("面积对称性", "EqArea(A,B,C,D,E,F) → EqArea(D,E,F,A,B,C)。"),
        "harmonic_swap": ("调和点列交换", "Harmonic(A,B,C,D) → Harmonic(C,D,A,B)。"),
        "harmonic_collinear": ("调和点列 → 共线", "Harmonic(A,B,C,D) → Collinear(A,B,C)。调和四点共线。"),
        "pole_polar_perp": ("极点极线 → 垂直", "PolePolar → Perp。极线与极点到圆心的连线垂直。"),
        "pole_polar_tangent": ("极点在圆上 → 切线", "PolePolar + OnCircle → Tangent。"),
        "inversion_collinear": ("反演 → 共线", "InvImage + Collinear → Collinear。反演保持共线性。"),
        "inversion_circle_fixed": ("反演 → 圆不变", "InvImage + OnCircle → OnCircle。反演保持某些圆。"),
        "cross_ratio_sym": ("交比对称性", "EqCrossRatio 对称交换。"),
        "cross_ratio_from_harmonic": ("调和 → 交比", "Harmonic → EqCrossRatio。调和点列的交比为 -1。"),
        "radical_axis_perp": ("根轴 → 垂直", "RadicalAxis → Perp(radical_axis, line_of_centers)。根轴垂直于圆心连线。"),
        # Converse rules
        "midpoint_from_cong_between": ("全等+介于 → 中点", "Cong(A,M,M,B) + Between(A,M,B) → Midpoint(M,A,B)。等分且在线段上即为中点。"),
        "cyclic_from_eq_angle": ("等角 → 共圆", "EqAngle(B,A,C,B,D,C) → Cyclic(A,B,C,D)。圆周角定理的逆。等角暗示四点共圆。"),
        "circumcenter_from_eqdist": ("等距 → 外心", "EqDist(O,A,B) + EqDist(O,B,C) → Circumcenter(O,A,B,C)。到三顶点等距的点是外心。"),
        "angle_bisect_from_eq_angle": ("等角 → 角平分线", "EqAngle(B,A,P,P,A,C) → AngleBisect(A,P,B,C)。角平分线定义的逆。"),
        "pole_polar_from_tangents": ("切线 → 极点极线", "Tangent(P,A,O,A) + Tangent(P,B,O,B) → PolePolar(P,A,B,O)。从外点引两切线，切点连线为极线。"),
        "radical_axis_from_common_points": ("公共弦 → 根轴", "两圆的公共弦即为根轴。OnCircle(O1,A) + OnCircle(O2,A) + OnCircle(O1,B) + OnCircle(O2,B) → RadicalAxis(A,B,O1,O2)。"),
        "inv_image_from_self": ("自反演", "圆上的点在反演变换下不动。Collinear(O,P,P') + OnCircle(O,P) + Cong(O,P,O,P') → InvImage(P',P,O,P)。"),
    }

    added = 0
    for rule_name, (title_zh, content_zh) in RULE_DESCRIPTIONS.items():
        doc_title = f"推理规则：{title_zh} / Rule: {rule_name}"
        if doc_title in existing_titles:
            continue

        # Find examples from proven cache
        examples = rule_to_examples.get(rule_name, [])
        example_text = ""
        if examples:
            ex = examples[0]
            assums_str = ", ".join(str(f) for f in ex.assumptions)
            goal_str = str(ex.goal)
            steps_str = " → ".join(s.rule_name for s in ex.steps)
            example_text = f"\n示例：{assums_str} ⊢ {goal_str}\n证明路径：{steps_str}"

        doc = {
            "doc_id": hashlib.md5(doc_title.encode()).hexdigest()[:12],
            "title": doc_title,
            "content": f"{content_zh}{example_text}",
            "source": "rebuild_knowledge",
            "tags": [rule_name, "rule", "strategy"],
        }
        new_docs.append(doc)
        added += 1

    # 3. Generate "proof pattern" documents for top combinations
    for pattern, count in pattern_counter.most_common(30):
        if count < 2:
            continue
        rules_in_pattern = pattern.split(" → ")
        if len(rules_in_pattern) < 2:
            continue
        title = f"证明模式：{' → '.join(rules_in_pattern[:4])} / Proof Pattern"
        if title in existing_titles:
            continue

        doc = {
            "doc_id": hashlib.md5(title.encode()).hexdigest()[:12],
            "title": title,
            "content": (
                f"常见证明模式（出现 {count} 次）：\n"
                f"规则链：{pattern}\n"
                f"适用场景：从 {rules_in_pattern[0]} 的前提出发，经过 "
                f"{len(rules_in_pattern)} 步推理到达目标。"
            ),
            "source": "rebuild_knowledge",
            "tags": ["pattern", "strategy"] + rules_in_pattern[:3],
        }
        new_docs.append(doc)
        added += 1

    # 4. Generate "predicate family" cross-reference documents
    predicate_families = {
        "平行与垂直": {
            "predicates": ["Parallel", "Perpendicular"],
            "rules": ["parallel_symmetry", "parallel_transitivity", "perp_symmetry", "parallel_perp_trans"],
            "content": "平行与垂直是最基础的几何关系。平行的对称性和传递性构成链式推理的骨架，垂直通过 parallel_perp_trans 传播。"
        },
        "中点与中位线": {
            "predicates": ["Midpoint", "Collinear", "Between"],
            "rules": ["midpoint_cong", "midpoint_collinear", "midpoint_between", "midsegment_parallel", "midsegment_sim_tri"],
            "content": "中点产生全等(midpoint_cong)、共线(midpoint_collinear)、介于(midpoint_between)。两个中点产生中位线(midsegment_parallel/sim_tri)。"
        },
        "全等与等距": {
            "predicates": ["Cong", "EqDist"],
            "rules": ["cong_symm", "cong_trans", "eqdist_from_cong", "eqdist_to_cong"],
            "content": "Cong 和 EqDist 可互转。全等具有对称性和传递性。perp_bisector_cong 和 circumcenter_cong 是重要的全等来源。"
        },
        "角度与相似": {
            "predicates": ["EqAngle", "SimTri", "CongTri"],
            "rules": ["eq_angle_symm", "eq_angle_trans", "sim_tri_angle", "congtri_angle", "isosceles_base_angle", "cyclic_inscribed_angle"],
            "content": "等角来源：等腰底角、圆周角、平行内错角、相似/全等三角形对应角。等角传递性是角度推理的核心。"
        },
        "圆与切线": {
            "predicates": ["Cyclic", "OnCircle", "Tangent", "Circumcenter"],
            "rules": ["cyclic_inscribed_angle", "tangent_perp_radius", "tangent_oncircle", "circumcenter_cong_ab", "circumcenter_oncircle"],
            "content": "圆上点→圆周角定理，切线→垂直半径，外心→等距各顶点。这些规则将圆的性质转化为全等和角度。"
        },
        "射影几何": {
            "predicates": ["Harmonic", "PolePolar", "InvImage", "EqCrossRatio", "RadicalAxis"],
            "rules": ["harmonic_swap", "harmonic_collinear", "pole_polar_perp", "pole_polar_tangent", "inversion_collinear", "inversion_circle_fixed", "cross_ratio_sym", "cross_ratio_from_harmonic", "radical_axis_perp"],
            "content": "射影几何族：调和点列(共线+交换)，极点极线(垂直+切线)，反演(共线+圆不变)，交比(对称+调和)，根轴(垂直圆心连线)。"
        },
    }

    for family_name, info in predicate_families.items():
        title = f"谓词族：{family_name} / Predicate Family"
        if title in existing_titles:
            continue
        doc = {
            "doc_id": hashlib.md5(title.encode()).hexdigest()[:12],
            "title": title,
            "content": (
                f"{info['content']}\n"
                f"涉及谓词：{', '.join(info['predicates'])}\n"
                f"核心规则：{', '.join(info['rules'])}"
            ),
            "source": "rebuild_knowledge",
            "tags": ["family", "strategy"] + info["predicates"][:3],
        }
        new_docs.append(doc)
        added += 1

    # Write new documents
    if new_docs:
        with open(path, "a") as f:
            for doc in new_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    return added


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Rebuild knowledge base")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of rounds per generator")
    parser.add_argument("--problems-per-gen", type=int, default=3,
                        help="Problems per generator per round")
    parser.add_argument("--beam-width", type=int, default=16,
                        help="Beam search width")
    parser.add_argument("--max-depth", type=int, default=14,
                        help="Max search depth")
    parser.add_argument("--no-rag", action="store_true",
                        help="Skip RAG enrichment")
    args = parser.parse_args()

    t0 = time.time()
    store = rebuild_knowledge(
        rounds=args.rounds,
        problems_per_gen=args.problems_per_gen,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
    )

    # Save to disk
    store.save()
    elapsed = time.time() - t0
    print(f"\nKnowledge saved to {store.data_dir} ({elapsed:.1f}s)")

    # Enrich RAG
    if not args.no_rag:
        rag_path = Path(__file__).parent / "data" / "rag" / "documents.jsonl"
        added = enrich_rag_documents(store, str(rag_path))
        print(f"RAG enriched: {added} new documents added to {rag_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
