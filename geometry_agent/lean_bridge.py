"""
lean_bridge.py
──────────────
Lean 4 checker protocol, mock checker, and real Lean 4 process bridge
with parallel batch verification.

Provides:
  - ``CheckResult`` – step-level result dataclass.
  - ``LeanChecker`` – protocol for step-level checking.
  - ``MockLeanChecker`` – fast trust-the-rule-engine mode.
  - ``make_checker()`` – factory (lean vs mock).
  - ``ProcessLeanChecker`` – real Lean 4 subprocess bridge.
  - ``check_full_proof`` / ``check_steps_batch`` – convenience helpers.

Author:  Jiangsheng Yu
License: MIT
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from .dsl import Fact, GeoState, Step

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  CheckResult / LeanChecker protocol / MockLeanChecker
#  (formerly in lean_checker.py)
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    message: str = ""


class LeanChecker(Protocol):
    def check_step(self, state: GeoState, step: Step) -> CheckResult:
        ...


class MockLeanChecker:
    """Fast, trust-the-rule-engine mode (no Lean invocation)."""

    def check_step(self, state: GeoState, step: Step) -> CheckResult:
        if step.conclusion_fact in state.facts:
            return CheckResult(False, "fact already known")
        return CheckResult(True, "mock-accepted")


def make_checker(use_lean: bool = False, **kwargs) -> LeanChecker:
    """Factory: returns ProcessLeanChecker when `use_lean=True`,
    otherwise MockLeanChecker (fast, for development/large batches)."""
    if use_lean:
        return ProcessLeanChecker(**kwargs)  # type: ignore[return-value]
    return MockLeanChecker()  # type: ignore[return-value]

# ── Rule → Lean mapping ─────────────────────────────────────────────

PREDICATE_LEAN_NAME: Dict[str, str] = {
    "Parallel": "Parallel",
    "Perpendicular": "Perpendicular",
    "Collinear": "Collinear",
    "Cyclic": "Cyclic",
    "Midpoint": "IsMidpoint",
    "EqAngle": "EqAngle",
    "Cong": "Cong",
    "SimTri": "SimTri",
    "OnCircle": "OnCircle",
    # New predicates
    "CongTri": "CongTri",
    "Tangent": "Tangent",
    "EqRatio": "EqRatio",
    "Between": "Between",
    "AngleBisect": "AngleBisect",
    "Concurrent": "Concurrent",
    "Circumcenter": "Circumcenter",
    "EqDist": "EqDist",
    "EqArea": "EqArea",
    "Harmonic": "Harmonic",
    "PolePolar": "PolePolar",
    "InvImage": "InvImage",
    "EqCrossRatio": "EqCrossRatio",
    "RadicalAxis": "RadicalAxis",
}


@dataclass(frozen=True)
class RuleLeanSpec:
    """How a Python rule maps to its Lean axiom."""
    lean_lemma: str
    # Extract the explicit point arguments for the Lean proof term
    # from the tuple of premise Facts.
    point_extractor: Callable[[Tuple[Fact, ...]], List[str]]


def _pts_all(premises: Tuple[Fact, ...]) -> List[str]:
    """All args from the single premise."""
    return list(premises[0].args)


def _pts_trans(premises: Tuple[Fact, ...]) -> List[str]:
    """First premise args + non-shared tail of second premise."""
    return list(premises[0].args) + list(premises[1].args[2:])


def _pts_midsegment(premises: Tuple[Fact, ...]) -> List[str]:
    """MidsegmentParallel: IsMidpoint(M,A,B), IsMidpoint(N,A,C).
    Lean signature: midsegment_parallel M N A B C h0 h1.
    We need to figure out M, N, A (shared), B, C (others)."""
    m1, a1, b1 = premises[0].args
    m2, a2, b2 = premises[1].args
    # Find shared endpoint
    if a1 == a2:
        return [m1, m2, a1, b1, b2]
    elif a1 == b2:
        return [m1, m2, a1, b1, a2]
    elif b1 == a2:
        return [m1, m2, b1, a1, b2]
    elif b1 == b2:
        return [m1, m2, b1, a1, a2]
    return list(premises[0].args) + list(premises[1].args)


def _pts_medians_concurrent(premises: Tuple[Fact, ...]) -> List[str]:
    """MediansConcurrent: IsMidpoint(D,B,C), IsMidpoint(E,A,C), IsMidpoint(F,A,B).

    Lean signature:  medians_concurrent A B C D E F_pt h0 h1 h2
    We recover the triangle vertices (A,B,C) and midpoints (D,E,F) by
    finding which points appear as endpoints and which appear as midpoints.
    """
    # Collect all 3 midpoint facts — each is Midpoint(mid, p, q)
    mids = [(p[0], frozenset(p[1:])) for p in (pr.args for pr in premises)]
    # Triangle vertices are the union of all endpoints
    all_endpoints: set[str] = set()
    for _, seg in mids:
        all_endpoints |= seg
    # There should be exactly 3 vertices
    verts = sorted(all_endpoints)
    if len(verts) != 3:
        # Fallback: just dump all args
        return [a for pr in premises for a in pr.args]
    A, B, C = verts
    mid_of = {}
    for m, seg in mids:
        mid_of[seg] = m
    D = mid_of.get(frozenset((B, C)), "?")
    E = mid_of.get(frozenset((A, C)), "?")
    F = mid_of.get(frozenset((A, B)), "?")
    return [A, B, C, D, E, F]


def _pts_perp_bisector(premises: Tuple[Fact, ...]) -> List[str]:
    """PerpBisectorCong: IsMidpoint(M,A,B), Perpendicular(C,M,A,B).
    Lean signature: perp_bisector_cong M A B C h0 h1."""
    mid, a, b = premises[0].args
    p, q, _, _ = premises[1].args
    other = q if p == mid else p
    return [mid, a, b, other]


def _pts_eq_angle_trans(premises: Tuple[Fact, ...]) -> List[str]:
    """EqAngle trans: all 9 point args."""
    return list(premises[0].args) + list(premises[1].args[3:])


def _pts_cyclic_inscribed(premises: Tuple[Fact, ...]) -> List[str]:
    """Cyclic inscribed angle: all 4 cyclic points."""
    return list(premises[0].args)


def _pts_cong_perp_bisector(premises: Tuple[Fact, ...]) -> List[str]:
    """CongPerpBisector: Cong(C,A,C,B), IsMidpoint(M,A,B).
    Lean signature: cong_perp_bisector C A B M h0 h1."""
    ca, cb, _, cd = premises[0].args
    mid = premises[1].args[0]
    return [ca, cb, cd, mid]


def _pts_parallel_alt_angle(premises: Tuple[Fact, ...]) -> List[str]:
    """ParallelAlternateAngle: Parallel(A,B,C,D), Collinear(A,X,C).
    Lean signature: parallel_alternate_angle A B C D X h0 h1."""
    a, b, c, d = premises[0].args
    col_pts = set(premises[1].args)
    inter_ab = {a, b} & col_pts
    inter_cd = {c, d} & col_pts
    pa = inter_ab.pop() if inter_ab else a
    pc = inter_cd.pop() if inter_cd else c
    x = (col_pts - {pa, pc}).pop() if len(col_pts - {pa, pc}) == 1 else 'X'
    ob = b if pa == a else a
    od = d if pc == c else c
    return [pa, ob, pc, od, x]


def _pts_sim_tri(premises: Tuple[Fact, ...]) -> List[str]:
    """SimTri rules: all 6 points."""
    return list(premises[0].args)


def _pts_sim_tri_cong(premises: Tuple[Fact, ...]) -> List[str]:
    """SimTriCong: SimTri(A,B,C,D,E,F) + Cong(A,B,D,E)."""
    return list(premises[0].args)


RULE_LEAN_MAP: Dict[str, RuleLeanSpec] = {
    "parallel_symmetry": RuleLeanSpec(
        lean_lemma="parallel_symm",
        point_extractor=_pts_all,
    ),
    "parallel_transitivity": RuleLeanSpec(
        lean_lemma="parallel_trans",
        point_extractor=_pts_trans,
    ),
    "perp_symmetry": RuleLeanSpec(
        lean_lemma="perp_symm",
        point_extractor=_pts_all,
    ),
    "parallel_perp_trans": RuleLeanSpec(
        lean_lemma="parallel_perp_trans",
        point_extractor=_pts_trans,
    ),
    # Midpoint
    "midpoint_collinear": RuleLeanSpec(
        lean_lemma="midpoint_collinear",
        point_extractor=_pts_all,
    ),
    "midpoint_cong": RuleLeanSpec(
        lean_lemma="midpoint_cong",
        point_extractor=_pts_all,
    ),
    "midsegment_parallel": RuleLeanSpec(
        lean_lemma="midsegment_parallel",
        point_extractor=_pts_midsegment,
    ),
    # Congruence
    "cong_symm": RuleLeanSpec(
        lean_lemma="cong_symm",
        point_extractor=_pts_all,
    ),
    "cong_trans": RuleLeanSpec(
        lean_lemma="cong_trans",
        point_extractor=_pts_trans,
    ),
    # Angle equality
    "eq_angle_symm": RuleLeanSpec(
        lean_lemma="eq_angle_symm",
        point_extractor=_pts_all,
    ),
    "eq_angle_trans": RuleLeanSpec(
        lean_lemma="eq_angle_trans",
        point_extractor=_pts_eq_angle_trans,
    ),
    # Cyclic
    "cyclic_inscribed_angle": RuleLeanSpec(
        lean_lemma="cyclic_inscribed_angle",
        point_extractor=_pts_cyclic_inscribed,
    ),
    # Cross-domain
    "perp_bisector_cong": RuleLeanSpec(
        lean_lemma="perp_bisector_cong",
        point_extractor=_pts_perp_bisector,
    ),
    # Triangle / circle rules
    "isosceles_base_angle": RuleLeanSpec(
        lean_lemma="isosceles_base_angle",
        point_extractor=_pts_all,
    ),
    "cong_perp_bisector": RuleLeanSpec(
        lean_lemma="cong_perp_bisector",
        point_extractor=_pts_cong_perp_bisector,
    ),
    "parallel_alternate_angle": RuleLeanSpec(
        lean_lemma="parallel_alternate_angle",
        point_extractor=_pts_parallel_alt_angle,
    ),
    "cyclic_chord_angle": RuleLeanSpec(
        lean_lemma="cyclic_chord_angle",
        point_extractor=_pts_cyclic_inscribed,
    ),
    "midsegment_sim_tri": RuleLeanSpec(
        lean_lemma="midsegment_sim_tri",
        point_extractor=_pts_midsegment,
    ),
    "sim_tri_angle": RuleLeanSpec(
        lean_lemma="sim_tri_angle",
        point_extractor=_pts_sim_tri,
    ),
    "sim_tri_cong": RuleLeanSpec(
        lean_lemma="sim_tri_cong",
        point_extractor=_pts_sim_tri_cong,
    ),
    # ── NEW RULES ──
    # CongTri
    "congtri_side": RuleLeanSpec(
        lean_lemma="congtri_side",
        point_extractor=_pts_sim_tri,   # 6-point same as SimTri
    ),
    "congtri_angle": RuleLeanSpec(
        lean_lemma="congtri_angle",
        point_extractor=_pts_sim_tri,
    ),
    "congtri_from_sim_cong": RuleLeanSpec(
        lean_lemma="congtri_from_sim_cong",
        point_extractor=_pts_sim_tri_cong,
    ),
    "congtri_eqarea": RuleLeanSpec(
        lean_lemma="congtri_eqarea",
        point_extractor=_pts_sim_tri,
    ),
    # Tangent
    "tangent_perp_radius": RuleLeanSpec(
        lean_lemma="tangent_perp_radius",
        point_extractor=_pts_all,
    ),
    "tangent_oncircle": RuleLeanSpec(
        lean_lemma="tangent_oncircle",
        point_extractor=_pts_all,
    ),
    # EqRatio
    "eqratio_from_simtri": RuleLeanSpec(
        lean_lemma="eqratio_from_simtri",
        point_extractor=_pts_sim_tri,
    ),
    "eqratio_sym": RuleLeanSpec(
        lean_lemma="eqratio_sym",
        point_extractor=_pts_all,
    ),
    "eqratio_trans": RuleLeanSpec(
        lean_lemma="eqratio_trans",
        point_extractor=lambda ps: list(ps[0].args) + list(ps[1].args[4:]),
    ),
    # Between
    "between_collinear": RuleLeanSpec(
        lean_lemma="between_collinear",
        point_extractor=_pts_all,
    ),
    "midpoint_between": RuleLeanSpec(
        lean_lemma="midpoint_between",
        point_extractor=_pts_all,
    ),
    # AngleBisect
    "angle_bisect_eq_angle": RuleLeanSpec(
        lean_lemma="angle_bisect_eq_angle",
        point_extractor=_pts_all,
    ),
    "angle_bisect_eqratio": RuleLeanSpec(
        lean_lemma="angle_bisect_eqratio",
        point_extractor=lambda ps: list(ps[0].args),
    ),
    # Concurrent
    "medians_concurrent": RuleLeanSpec(
        lean_lemma="medians_concurrent",
        point_extractor=_pts_medians_concurrent,
    ),
    # Circumcenter
    "circumcenter_cong_ab": RuleLeanSpec(
        lean_lemma="circumcenter_cong_ab",
        point_extractor=_pts_all,
    ),
    "circumcenter_cong_bc": RuleLeanSpec(
        lean_lemma="circumcenter_cong_bc",
        point_extractor=_pts_all,
    ),
    "circumcenter_oncircle": RuleLeanSpec(
        lean_lemma="circumcenter_oncircle",
        point_extractor=_pts_all,
    ),
    # EqDist
    "eqdist_from_cong": RuleLeanSpec(
        lean_lemma="eqdist_from_cong",
        point_extractor=_pts_all,
    ),
    "eqdist_to_cong": RuleLeanSpec(
        lean_lemma="eqdist_to_cong",
        point_extractor=_pts_all,
    ),
    # EqArea
    "eqarea_sym": RuleLeanSpec(
        lean_lemma="eqarea_sym",
        point_extractor=_pts_all,
    ),
    # Harmonic
    "harmonic_swap": RuleLeanSpec(
        lean_lemma="harmonic_swap",
        point_extractor=_pts_all,
    ),
    "harmonic_collinear": RuleLeanSpec(
        lean_lemma="harmonic_collinear",
        point_extractor=_pts_all,
    ),
    # PolePolar
    "pole_polar_perp": RuleLeanSpec(
        lean_lemma="pole_polar_perp",
        point_extractor=_pts_all,
    ),
    "pole_polar_tangent": RuleLeanSpec(
        lean_lemma="pole_polar_tangent",
        point_extractor=lambda ps: list(ps[0].args),
    ),
    # InvImage
    "inversion_collinear": RuleLeanSpec(
        lean_lemma="inversion_collinear",
        point_extractor=_pts_all,
    ),
    "inversion_circle_fixed": RuleLeanSpec(
        lean_lemma="inversion_circle_fixed",
        point_extractor=lambda ps: list(ps[0].args),
    ),
    # EqCrossRatio
    "cross_ratio_sym": RuleLeanSpec(
        lean_lemma="cross_ratio_sym",
        point_extractor=_pts_all,
    ),
    "cross_ratio_from_harmonic": RuleLeanSpec(
        lean_lemma="cross_ratio_from_harmonic",
        point_extractor=lambda ps: list(ps[0].args) + list(ps[1].args),
    ),
    # RadicalAxis
    "radical_axis_perp": RuleLeanSpec(
        lean_lemma="radical_axis_perp",
        point_extractor=_pts_all,
    ),
}

# ── Helpers ──────────────────────────────────────────────────────────


def _fact_to_lean_prop(fact: Fact) -> str:
    lean_pred = PREDICATE_LEAN_NAME.get(fact.predicate, fact.predicate)
    return f"{lean_pred} {' '.join(fact.args)}"


def translate_step(step: Step) -> str:
    """Return a self-contained Lean4 source that checks one step."""
    spec = RULE_LEAN_MAP.get(step.rule_name)
    if spec is None:
        raise ValueError(f"No Lean translation for rule '{step.rule_name}'")

    # Collect point names
    points: set[str] = set()
    for f in step.premise_facts:
        points.update(f.args)
    points.update(step.conclusion_fact.args)

    lines: list[str] = ["import LeanGeo", ""]

    # Declare points
    for p in sorted(points):
        lines.append(f"axiom {p} : GPoint")
    lines.append("")

    # Declare premises
    hyp_names: list[str] = []
    for i, f in enumerate(step.premise_facts):
        h = f"h{i}"
        hyp_names.append(h)
        lines.append(f"axiom {h} : {_fact_to_lean_prop(f)}")
    lines.append("")

    # Proof term
    pt_args = spec.point_extractor(step.premise_facts)
    proof = f"{spec.lean_lemma} {' '.join(pt_args)} {' '.join(hyp_names)}"
    goal_prop = _fact_to_lean_prop(step.conclusion_fact)
    lines.append(f"theorem step_check : {goal_prop} :=")
    lines.append(f"  {proof}")
    lines.append("")
    return "\n".join(lines)


def translate_full_proof(
    assumptions: Sequence[Fact],
    steps: Sequence[Step],
    final_goal: Fact,
) -> str:
    """Return Lean4 source for the *entire* proof chain."""
    points: set[str] = set()
    for f in assumptions:
        points.update(f.args)
    for s in steps:
        for f in s.premise_facts:
            points.update(f.args)
        points.update(s.conclusion_fact.args)
    points.update(final_goal.args)

    lines: list[str] = ["import LeanGeo", ""]
    for p in sorted(points):
        lines.append(f"axiom {p} : GPoint")
    lines.append("")

    # Assumptions
    fact_hyp: Dict[Fact, str] = {}
    for i, f in enumerate(assumptions):
        h = f"hyp{i}"
        fact_hyp[f] = h
        lines.append(f"axiom {h} : {_fact_to_lean_prop(f)}")
    lines.append("")

    # Intermediate lemmas
    for idx, step in enumerate(steps):
        spec = RULE_LEAN_MAP.get(step.rule_name)
        if spec is None:
            raise ValueError(f"No Lean translation for rule '{step.rule_name}'")
        lem = f"lem{idx}"
        hyps = [fact_hyp[f] for f in step.premise_facts]
        pt_args = spec.point_extractor(step.premise_facts)
        proof = f"{spec.lean_lemma} {' '.join(pt_args)} {' '.join(hyps)}"
        prop = _fact_to_lean_prop(step.conclusion_fact)
        lines.append(f"theorem {lem} : {prop} :=")
        lines.append(f"  {proof}")
        lines.append("")
        fact_hyp[step.conclusion_fact] = lem

    # Final goal alias (optional, documents the target)
    if final_goal in fact_hyp:
        lines.append(f"-- Goal `{_fact_to_lean_prop(final_goal)}` proved as `{fact_hyp[final_goal]}`")
    else:
        lines.append(f"-- WARNING: final goal not reached in proof chain")
    lines.append("")
    return "\n".join(lines)


# ── Process-level checker ────────────────────────────────────────────


class ProcessLeanChecker:
    """Invoke Lean4 kernel via subprocess to verify each step."""

    def __init__(
        self,
        lean_project_dir: str | Path | None = None,
        lean_exe: str | None = None,
        timeout: int = 60,
    ):
        if lean_project_dir is None:
            # Default: the lean_geo directory next to this package
            lean_project_dir = (
                Path(__file__).resolve().parent.parent / "lean_geo"
            )
        self.project_dir = Path(lean_project_dir).resolve()
        self.lean_exe = lean_exe or os.path.expanduser("~/.elan/bin/lean")
        self.timeout = timeout

        self._lean_path: Optional[str] = None
        self._cache: Dict[Tuple[str, Tuple[Fact, ...], Fact], CheckResult] = {}

        # Eagerly resolve LEAN_PATH
        self._resolve_lean_path()

    # ── public API ───────────────────────────────────────────

    def check_step(self, state: GeoState, step: Step) -> CheckResult:
        key = (step.rule_name, step.premise_facts, step.conclusion_fact)
        if key in self._cache:
            return self._cache[key]

        try:
            source = translate_step(step)
        except ValueError as exc:
            res = CheckResult(False, str(exc))
            self._cache[key] = res
            return res

        res = self._run_lean(source)
        self._cache[key] = res
        return res

    def check_full_proof(
        self,
        assumptions: Sequence[Fact],
        steps: Sequence[Step],
        final_goal: Fact,
    ) -> CheckResult:
        try:
            source = translate_full_proof(assumptions, steps, final_goal)
        except ValueError as exc:
            return CheckResult(False, str(exc))
        return self._run_lean(source)

    def check_steps_batch(
        self,
        state: GeoState,
        steps: Sequence[Step],
        max_workers: int = 4,
    ) -> List[CheckResult]:
        """Check multiple steps in parallel using a thread pool.

        Each step is independently translated to a Lean source and verified.
        Results are returned *in order* matching the input steps.
        """
        if len(steps) <= 1 or max_workers <= 1:
            return [self.check_step(state, s) for s in steps]

        results: List[Optional[CheckResult]] = [None] * len(steps)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(self.check_step, state, step): idx
                for idx, step in enumerate(steps)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = CheckResult(False, f"batch-error: {exc}")
        return [r if r is not None else CheckResult(False, "missing") for r in results]

    # ── internals ────────────────────────────────────────────

    def _resolve_lean_path(self) -> None:
        """Run `lake print-paths` once to discover LEAN_PATH."""
        try:
            res = subprocess.run(
                [os.path.expanduser("~/.elan/bin/lake"), "print-paths"],
                capture_output=True,
                text=True,
                cwd=str(self.project_dir),
                timeout=120,
            )
            if res.returncode == 0:
                data = json.loads(res.stdout)
                paths = data.get("oleanPath", [])
                if paths:
                    self._lean_path = ":".join(paths)
                    logger.info("LEAN_PATH resolved: %s", self._lean_path)
                    return
        except Exception as exc:
            logger.warning("lake print-paths failed: %s", exc)

        # Fallback: guess build/lib
        fallback = str(self.project_dir / ".lake" / "build" / "lib")
        self._lean_path = fallback
        logger.info("LEAN_PATH fallback: %s", fallback)

    def _run_lean(self, source: str) -> CheckResult:
        check_dir = self.project_dir / "_check"
        check_dir.mkdir(exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            suffix=".lean", prefix="chk_", dir=str(check_dir)
        )
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(source)

            env = dict(os.environ)
            if self._lean_path:
                env["LEAN_PATH"] = self._lean_path

            proc = subprocess.run(
                [self.lean_exe, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            if proc.returncode == 0:
                return CheckResult(True, "lean-verified")
            else:
                msg = (proc.stderr or proc.stdout or "unknown error")[:800]
                return CheckResult(False, f"lean-rejected: {msg}")
        except subprocess.TimeoutExpired:
            return CheckResult(False, "lean-timeout")
        except Exception as exc:
            return CheckResult(False, f"lean-error: {exc}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
