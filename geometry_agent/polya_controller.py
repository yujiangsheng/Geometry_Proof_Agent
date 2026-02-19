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
from typing import Dict, List, Optional

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

    def make_plan(self, assumptions: List[Fact], goal: Fact, strategy: str) -> PolyaPlan:
        """Step-1+2: understand + devise a plan from problem complexity."""
        self._attempt_by_strategy[strategy] += 1

        preds = {f.predicate for f in assumptions}
        preds.add(goal.predicate)

        n_assumptions = len(assumptions)
        n_preds = len(preds)
        has_high_tier = any(p in self._HIGH_TIER_PREDS for p in preds)

        complexity = 0
        if n_assumptions >= 4:
            complexity += 1
        if n_assumptions >= 6:
            complexity += 1
        if n_preds >= 4:
            complexity += 1
        if has_high_tier:
            complexity += 1
        if strategy.startswith("deep:"):
            complexity += 1

        # Adaptive confidence gate: easier problems can be filtered harder,
        # while deep/complex ones should tolerate lower initial confidence.
        min_conf = 0.55
        if complexity >= 3:
            min_conf = 0.45
        if strategy.startswith("deep:"):
            min_conf = 0.40

        # Pólya numeric trials and premise probe budget.
        polya_trials = 10 + 2 * min(complexity, 4)
        premise_probe_trials = 6 if n_assumptions >= 4 else 0
        if complexity >= 3:
            premise_probe_trials = 8

        # Search budget planning.
        fast_beam = 64 + complexity * 24
        fast_depth = 12 + complexity * 2
        deep_beam = 180 + complexity * 30
        deep_depth = 24 + complexity * 2

        return PolyaPlan(
            polya_trials=polya_trials,
            polya_min_confidence=min_conf,
            premise_probe_trials=premise_probe_trials,
            fast_beam_width=fast_beam,
            fast_max_depth=fast_depth,
            deep_beam_width=deep_beam,
            deep_max_depth=deep_depth,
        )

    def should_escalate(self, confidence: float, strategy: str) -> bool:
        """Step-3 policy: decide whether to escalate to deep search."""
        if strategy.startswith("deep:"):
            return True
        return confidence >= 0.85

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
