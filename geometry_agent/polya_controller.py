"""polya_controller.py – Minimal Pólya 4-step controller for conjecture search.

Implements a lightweight, practical mapping of Pólya's core ideas:

1) Understand the problem
2) Devise a plan
3) Carry out the plan
4) Look back (reflection)

The controller is intentionally small and non-invasive so it can be
plugged into the existing heuristic pipeline without architectural churn.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .dsl import Fact


@dataclass(frozen=True)
class PolyaPlan:
    """Execution plan produced from a conjecture profile."""

    polya_trials: int
    polya_min_confidence: float
    premise_probe_trials: int
    fast_beam_width: int
    fast_max_depth: int
    deep_beam_width: int
    deep_max_depth: int


@dataclass(frozen=True)
class PhaseSchedule:
    """Cross-engine phase schedule for evolve_hybrid orchestration."""

    phase: str
    target_novel: int
    min_difficulty: float
    beam_width: int
    max_depth: int
    budget: int


# ── Predicate complexity profiles ────────────────────────────────────
# Maps predicate families to a complexity weight used for adaptive
# search resource allocation.
_PRED_COMPLEXITY: Dict[str, float] = {
    "Parallel": 0.3, "Perpendicular": 0.3, "Collinear": 0.2, "Between": 0.2,
    "Midpoint": 0.4, "AngleBisect": 0.6,
    "Cong": 0.5, "EqAngle": 0.7, "EqDist": 0.4, "EqArea": 0.8, "EqRatio": 0.9,
    "Cyclic": 0.8, "OnCircle": 0.5, "Circumcenter": 0.7, "Tangent": 0.8,
    "RadicalAxis": 0.9,
    "SimTri": 0.7, "CongTri": 0.7, "Concurrent": 0.8,
    "Harmonic": 1.0, "PolePolar": 1.0, "InvImage": 1.0, "EqCrossRatio": 1.0,
}


class PolyaController:
    """Minimal Pólya controller with adaptive planning and reflection."""

    _HIGH_TIER_PREDS = {
        "Cyclic", "Circle", "Tangent", "EqRatio", "EqArea",
        "Harmonic", "PolePolar", "EqCrossRatio", "RadicalAxis",
    }

    def __init__(self, knowledge_store: Optional["KnowledgeStore"] = None) -> None:  # type: ignore[name-defined]
        self.knowledge_store = knowledge_store
        self._success_by_strategy: Dict[str, int] = defaultdict(int)
        self._attempt_by_strategy: Dict[str, int] = defaultdict(int)
        self._fail_reasons: Dict[str, int] = defaultdict(int)
        # Track recent Pólya confidence values for adaptive calibration
        self._recent_confidences: List[float] = []
        self._max_recent = 50

    def _compute_complexity(self, assumptions: List[Fact], goal: Fact) -> Tuple[int, float]:
        """Compute discrete complexity level and continuous complexity score.

        Returns (discrete_level: 0-5, continuous_score: 0.0-1.0).
        """
        preds = {f.predicate for f in assumptions}
        preds.add(goal.predicate)

        n_assumptions = len(assumptions)
        n_preds = len(preds)
        has_high_tier = any(p in self._HIGH_TIER_PREDS for p in preds)

        # Discrete complexity (backward compatible)
        complexity = 0
        if n_assumptions >= 4:
            complexity += 1
        if n_assumptions >= 6:
            complexity += 1
        if n_preds >= 4:
            complexity += 1
        if has_high_tier:
            complexity += 1

        # Continuous score: weighted sum of predicate complexities
        all_facts = list(assumptions) + [goal]
        pred_scores = [_PRED_COMPLEXITY.get(f.predicate, 0.5) for f in all_facts]
        continuous = sum(pred_scores) / max(len(pred_scores), 1)
        # Scale by assumption count
        continuous = min(1.0, continuous * (1.0 + 0.1 * max(0, n_assumptions - 3)))

        return complexity, continuous

    def make_plan(self, assumptions: List[Fact], goal: Fact, strategy: str) -> PolyaPlan:
        """Step-1+2: understand + devise a plan from problem complexity.

        Uses fine-grained complexity analysis to allocate search resources:
        - Simple problems: narrow beam, shallow depth (fast confirmation)
        - Complex problems: wide beam, deep search (thorough exploration)
        - High-tier predicates: extra resources for difficult domains
        """
        self._attempt_by_strategy[strategy] += 1

        complexity, cont_score = self._compute_complexity(assumptions, goal)
        is_deep = strategy.startswith("deep:") or strategy.startswith("constructive:")

        if is_deep:
            complexity += 1

        # ── Adaptive confidence gate ──────────────────────────
        # Easier problems can be filtered harder; deep/complex ones
        # should tolerate lower initial confidence.
        min_conf = 0.55
        if complexity >= 3:
            min_conf = 0.45
        if is_deep:
            min_conf = 0.40

        # ── Pólya trial budget ────────────────────────────────
        # Two-stage: fast_trials handled by polya_test_two_stage,
        # this sets the total budget
        polya_trials = 10 + 2 * min(complexity, 4)
        premise_probe_trials = 6 if len(assumptions) >= 4 else 0
        if complexity >= 3:
            premise_probe_trials = 8

        # ── Three-tier search budget allocation ───────────────
        # Tier A: Simple (cont_score < 0.4) — fast confirmation
        # Tier B: Medium (0.4 <= cont_score < 0.7) — balanced
        # Tier C: Complex (cont_score >= 0.7) — thorough exploration
        if cont_score < 0.4:
            # Tier A: simple problems — narrow and shallow
            fast_beam = 48
            fast_depth = 10
            deep_beam = 96
            deep_depth = 18
        elif cont_score < 0.7:
            # Tier B: medium problems — balanced
            fast_beam = 64 + int(complexity * 16)
            fast_depth = 12 + complexity
            deep_beam = 160 + int(complexity * 20)
            deep_depth = 22 + complexity
        else:
            # Tier C: complex problems — wide and deep
            fast_beam = 96 + int(complexity * 24)
            fast_depth = 14 + complexity * 2
            deep_beam = 240 + int(complexity * 40)
            deep_depth = 28 + complexity * 2

        # Knowledge-guided boost: if we have success data for this
        # strategy, allocate more resources to profitable strategies
        if self.knowledge_store is not None:
            success_rate = self._strategy_success_rate(strategy)
            if success_rate > 0.3:
                # Successful strategy: invest more in deep search
                deep_beam = int(deep_beam * 1.3)
                deep_depth += 2

        return PolyaPlan(
            polya_trials=polya_trials,
            polya_min_confidence=min_conf,
            premise_probe_trials=premise_probe_trials,
            fast_beam_width=fast_beam,
            fast_max_depth=fast_depth,
            deep_beam_width=deep_beam,
            deep_max_depth=deep_depth,
        )

    def make_adaptive_plan(
        self,
        assumptions: List[Fact],
        goal: Fact,
        strategy: str,
        polya_confidence: float,
    ) -> PolyaPlan:
        """Re-plan search parameters using observed Pólya confidence.

        Called AFTER the Pólya test passes, to fine-tune beam search
        resources based on actual confidence rather than predictions.
        High confidence → can afford narrower beam (fast confirmation).
        Medium confidence → needs wider beam (uncertain proof path).
        """
        base_plan = self.make_plan(assumptions, goal, strategy)

        # Track confidence for calibration
        self._recent_confidences.append(polya_confidence)
        if len(self._recent_confidences) > self._max_recent:
            self._recent_confidences.pop(0)

        if polya_confidence >= 0.95:
            # Very high confidence: narrow fast search is sufficient
            return PolyaPlan(
                polya_trials=base_plan.polya_trials,
                polya_min_confidence=base_plan.polya_min_confidence,
                premise_probe_trials=base_plan.premise_probe_trials,
                fast_beam_width=max(32, base_plan.fast_beam_width // 2),
                fast_max_depth=max(8, base_plan.fast_max_depth - 2),
                deep_beam_width=base_plan.deep_beam_width,
                deep_max_depth=base_plan.deep_max_depth,
            )
        elif polya_confidence < 0.70:
            # Low confidence: invest more in deep search
            return PolyaPlan(
                polya_trials=base_plan.polya_trials,
                polya_min_confidence=base_plan.polya_min_confidence,
                premise_probe_trials=base_plan.premise_probe_trials,
                fast_beam_width=base_plan.fast_beam_width,
                fast_max_depth=base_plan.fast_max_depth,
                deep_beam_width=int(base_plan.deep_beam_width * 1.4),
                deep_max_depth=base_plan.deep_max_depth + 3,
            )
        else:
            return base_plan

    def _strategy_success_rate(self, strategy: str) -> float:
        """Compute success rate for a strategy, Laplace-smoothed."""
        s = self._success_by_strategy.get(strategy, 0)
        a = self._attempt_by_strategy.get(strategy, 0)
        return (s + 1) / (a + 2)

    def should_escalate(self, confidence: float, strategy: str) -> bool:
        """Step-3 policy: decide whether to escalate to deep search.

        Escalation criteria:
        - Deep/constructive strategies always escalate
        - High confidence (>= 0.85) warrants deeper exploration
        - When recent average confidence is high, raise the bar
        """
        if strategy.startswith("deep:") or strategy.startswith("constructive:"):
            return True

        # Adaptive threshold based on recent success distribution
        if self._recent_confidences:
            avg_conf = sum(self._recent_confidences) / len(self._recent_confidences)
            threshold = max(0.75, min(0.90, avg_conf))
        else:
            threshold = 0.85

        return confidence >= threshold

    def note_success(self, strategy: str) -> None:
        """Step-4 reflection: record successful execution."""
        self._success_by_strategy[strategy] += 1

    def note_failure(self, reason: str) -> None:
        """Step-4 reflection: record categorized failure."""
        self._fail_reasons[reason] += 1

    def summary(self) -> str:
        """Return compact reflection summary for verbose diagnostics."""
        attempts = sum(self._attempt_by_strategy.values())
        success = sum(self._success_by_strategy.values())
        parts = [f"Pólya控制器: 尝试={attempts}, 成功={success}"]
        if self._fail_reasons:
            top = sorted(self._fail_reasons.items(), key=lambda x: x[1], reverse=True)[:4]
            parts.append("失败Top=" + ", ".join(f"{k}:{v}" for k, v in top))
        return " | ".join(parts)

    def plan_phase(
        self,
        phase: str,
        remaining_target: int,
        min_difficulty: float,
    ) -> PhaseSchedule:
        """Plan coarse phase-level budget for GA/RLVR orchestration."""
        target = max(1, int(remaining_target))

        if phase == "ga":
            # GA benefits from broader search under higher difficulty bars.
            difficulty = max(1.5, min_difficulty - 0.4)
            beam = 120 if min_difficulty >= 5.0 else 96
            depth = 24 if min_difficulty >= 5.0 else 22
            budget = max(60, 80 + target * 20)
            return PhaseSchedule(
                phase=phase,
                target_novel=target,
                min_difficulty=difficulty,
                beam_width=beam,
                max_depth=depth,
                budget=budget,
            )

        if phase == "rlvr":
            # RLVR tends to require more episodes for harder thresholds.
            difficulty = min_difficulty
            beam = 40 if min_difficulty >= 5.0 else 32
            depth = 20 if min_difficulty >= 5.0 else 18
            budget = max(600, 700 + target * 220)
            return PhaseSchedule(
                phase=phase,
                target_novel=target,
                min_difficulty=difficulty,
                beam_width=beam,
                max_depth=depth,
                budget=budget,
            )

        # Fallback schedule for unknown phases.
        return PhaseSchedule(
            phase=phase,
            target_novel=target,
            min_difficulty=min_difficulty,
            beam_width=96,
            max_depth=22,
            budget=target * 100,
        )
