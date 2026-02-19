"""knowledge.py – Persistent, thread-safe knowledge store with guidance.

Stores all accumulated knowledge and experience to disk under a
dedicated ``data/knowledge/`` directory so that:
  • Knowledge survives across process restarts
  • Different training runs can inspect / merge prior experience
  • Deduplication is maintained across sessions (via fingerprints)

Directory layout (auto-created on first save)::

    data/knowledge/
    ├── proven_cache.jsonl   # one proven sub-goal per line
    ├── experience.jsonl     # one search-episode per line
    ├── failure_patterns.json
    └── stats.json

Design goals:
  • Proven sub-goal cache shared across search instances
  • Lemma / macro-rule registry with provenance tracking
  • Proof-trace experience buffer with fingerprint-based dedup
  • Statistics for the self-evolution evaluation loop
  • **Bidirectional guidance**: accumulated knowledge actively guides
    conjecture generation, rule selection, and evolution strategy.
    Knowledge ↔ evolution form a mutual promotion loop.

Author:  Jiangsheng Yu
License: MIT
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .dsl import Fact, Goal, Step
from .semantic import (
    compute_isomorphism_map,
    remap_fact,
    remap_step,
    semantic_proof_fingerprint,
    semantic_theorem_fingerprint,
    structural_theorem_fingerprint,
)

logger = logging.getLogger(__name__)

# Default data directory: <project_root>/data/knowledge
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge"


# ── Fingerprinting ───────────────────────────────────────────────────
# All fingerprinting is now **semantic** (isomorphism-invariant).
# A proof that differs only by point renaming is recognised as
# a duplicate — simple substitution never produces a "new" result.

def _fact_fingerprint(fact: Fact) -> str:
    """Legacy per-fact fingerprint (used only for failure pattern keys)."""
    return f"{fact.predicate}:{','.join(fact.args)}"


def _trace_fingerprint(
    assumptions: Sequence[Fact],
    goal: Fact,
    steps: Sequence[Step],
) -> str:
    """Semantic-level proof fingerprint: isomorphism-invariant.

    Two proofs that differ only in point naming produce the *same* hash.
    This ensures that simple substitution (renaming A→X, B→Y, …)
    never creates a "new" experience record.
    """
    return semantic_proof_fingerprint(assumptions, goal, steps)


def _goal_key(assumptions: FrozenSet[Fact], goal: Fact) -> str:
    """Semantic theorem key for the proven cache.

    Two theorems that differ only in point naming share the same key.
    """
    return semantic_theorem_fingerprint(assumptions, goal)


# ── Data classes ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProvenEntry:
    """A cached proven sub-goal with its proof trace."""
    goal: Fact
    assumptions: FrozenSet[Fact]
    steps: Tuple[Step, ...]
    timestamp: float
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "goal": self.goal.to_dict(),
            "assumptions": [f.to_dict() for f in sorted(self.assumptions)],
            "steps": [s.to_dict() for s in self.steps],
            "timestamp": self.timestamp,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProvenEntry":
        return cls(
            goal=Fact.from_dict(d["goal"]),
            assumptions=frozenset(Fact.from_dict(f) for f in d["assumptions"]),
            steps=tuple(Step.from_dict(s) for s in d["steps"]),
            timestamp=d.get("timestamp", 0.0),
            source=d.get("source", ""),
        )


@dataclass
class ExperienceRecord:
    """One episode of search: problem → result."""
    assumptions: List[Fact]
    goal: Fact
    success: bool
    steps: List[Step]
    explored_nodes: int
    difficulty: int = 0
    fingerprint: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "assumptions": [f.to_dict() for f in self.assumptions],
            "goal": self.goal.to_dict(),
            "success": self.success,
            "steps": [s.to_dict() for s in self.steps],
            "explored_nodes": self.explored_nodes,
            "difficulty": self.difficulty,
            "fingerprint": self.fingerprint,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExperienceRecord":
        return cls(
            assumptions=[Fact.from_dict(f) for f in d["assumptions"]],
            goal=Fact.from_dict(d["goal"]),
            success=d["success"],
            steps=[Step.from_dict(s) for s in d["steps"]],
            explored_nodes=d["explored_nodes"],
            difficulty=d.get("difficulty", 0),
            fingerprint=d.get("fingerprint", ""),
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class KnowledgeStats:
    proven_cache_size: int = 0
    proven_cache_hits: int = 0
    experience_total: int = 0
    experience_deduped: int = 0
    failed_patterns: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "proven_cache_size": self.proven_cache_size,
            "proven_cache_hits": self.proven_cache_hits,
            "experience_total": self.experience_total,
            "experience_deduped": self.experience_deduped,
            "failed_patterns": self.failed_patterns,
        }


# ── Knowledge Store ──────────────────────────────────────────────────

class KnowledgeStore:
    """Thread-safe knowledge accumulator with disk persistence and guidance.

    Used by all search instances / agents to:
      1. Cache proven sub-goals  → avoid re-proving the same thing
      2. Record experience traces → training data for policy learning
      3. Track failure patterns   → feed the Critic / Curriculum agent
      4. Persist to / load from ``data/knowledge/`` directory
      5. **Guide future decisions** → rule ordering, predicate coverage,
         proof templates, generator ranking (mutual promotion loop)
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._lock = threading.RLock()  # re-entrant for save() → stats()
        self.data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR

        # ── proven sub-goal cache ──
        self._proven: Dict[str, ProvenEntry] = {}
        self._proven_hits: int = 0

        # ── experience buffer ──
        self._experience: List[ExperienceRecord] = []
        self._experience_fps: Set[str] = set()
        self._dedup_skipped: int = 0

        # ── failure pattern counters ──
        self._failure_patterns: Dict[str, int] = defaultdict(int)

        # ── generator memory (cross-session failure/success tracking) ──
        # {generator_name: {"attempts": n, "successes": m,
        #                    "consecutive_failures": k}}
        self._generator_memory: Dict[str, Dict[str, int]] = {}

        # ── guidance cache (invalidated on new experience) ──
        self._guidance_cache: Dict[str, Any] = {}

    # ── Proven cache ─────────────────────────────────────────

    def lookup_proven(
        self, assumptions: FrozenSet[Fact], goal: Fact,
    ) -> Optional[ProvenEntry]:
        key = _goal_key(assumptions, goal)
        with self._lock:
            entry = self._proven.get(key)
            if entry is not None:
                self._proven_hits += 1
                logger.debug("KnowledgeStore HIT: %s", goal)
                # Remap point names if the cached entry uses different points
                entry = self._remap_entry_if_needed(entry, assumptions, goal)
            return entry

    def _remap_entry_if_needed(
        self,
        entry: ProvenEntry,
        query_assumptions: FrozenSet[Fact],
        query_goal: Fact,
    ) -> Optional[ProvenEntry]:
        """Remap a cached ProvenEntry if it uses different point names.

        When semantic dedup matches an isomorphic theorem (same structure,
        different point names), the cached proof steps reference the
        *original* names.  This method computes the isomorphism mapping
        and remaps the steps so they are consistent with the query.
        """
        # Quick check: if the assumptions already match, no remapping needed
        if entry.assumptions == query_assumptions and entry.goal == query_goal:
            return entry

        # Compute point-name mapping: cached → query
        source_facts = list(entry.assumptions) + [entry.goal]
        target_facts = list(query_assumptions) + [query_goal]
        mapping = compute_isomorphism_map(source_facts, target_facts)
        if mapping is None:
            logger.debug("Could not compute isomorphism mapping for cache hit")
            return None  # treat as cache miss; avoid mismatched proof reuse

        # Remap everything
        new_goal = remap_fact(entry.goal, mapping)
        new_assumptions = frozenset(remap_fact(f, mapping) for f in entry.assumptions)
        new_steps = tuple(remap_step(s, mapping) for s in entry.steps)

        return ProvenEntry(
            goal=new_goal,
            assumptions=new_assumptions,
            steps=new_steps,
            timestamp=entry.timestamp,
            source=entry.source,
        )

    def record_proven(
        self,
        assumptions: FrozenSet[Fact],
        goal: Fact,
        steps: Sequence[Step],
        source: str = "",
    ) -> bool:
        key = _goal_key(assumptions, goal)
        with self._lock:
            if key in self._proven:
                return False
            entry = ProvenEntry(
                goal=goal,
                assumptions=assumptions,
                steps=tuple(steps),
                timestamp=time.time(),
                source=source,
            )
            self._proven[key] = entry
            # Update structural fingerprint index for dedup
            if hasattr(self, "_structural_fps"):
                sfp = structural_theorem_fingerprint(assumptions, goal)
                self._structural_fps.add(sfp)
            logger.debug("KnowledgeStore STORE: %s  (%d steps)", goal, len(steps))
            return True

    # ── Experience buffer ────────────────────────────────────

    def record_experience(
        self,
        assumptions: List[Fact],
        goal: Fact,
        success: bool,
        steps: List[Step],
        explored_nodes: int,
        difficulty: int = 0,
    ) -> bool:
        fp = (_trace_fingerprint(assumptions, goal, steps)
              if steps
              else f"fail:{semantic_theorem_fingerprint(assumptions, goal)}")
        with self._lock:
            if fp in self._experience_fps:
                self._dedup_skipped += 1
                return False
            self._experience_fps.add(fp)
            self._experience.append(ExperienceRecord(
                assumptions=assumptions,
                goal=goal,
                success=success,
                steps=steps,
                explored_nodes=explored_nodes,
                difficulty=difficulty,
                fingerprint=fp,
                timestamp=time.time(),
            ))
            # Invalidate guidance cache on new experience
            self._guidance_cache.clear()
            return True

    def get_experience(self, *, success_only: bool = False) -> List[ExperienceRecord]:
        with self._lock:
            if success_only:
                return [e for e in self._experience if e.success]
            return list(self._experience)

    # ── Failure tracking ─────────────────────────────────────

    def record_failure_pattern(self, pattern: str) -> None:
        with self._lock:
            self._failure_patterns[pattern] += 1

    def top_failure_patterns(self, n: int = 10) -> List[Tuple[str, int]]:
        with self._lock:
            return sorted(self._failure_patterns.items(), key=lambda x: -x[1])[:n]

    # ── Statistics ───────────────────────────────────────────

    def stats(self) -> KnowledgeStats:
        with self._lock:
            return KnowledgeStats(
                proven_cache_size=len(self._proven),
                proven_cache_hits=self._proven_hits,
                experience_total=len(self._experience),
                experience_deduped=self._dedup_skipped,
                failed_patterns=dict(self._failure_patterns),
            )

    def summary(self) -> str:
        s = self.stats()
        lines = [
            f"KnowledgeStore [{self.data_dir}]",
            f"  Proven cache   : {s.proven_cache_size} entries, {s.proven_cache_hits} hits",
            f"  Experience buf : {s.experience_total} traces, {s.experience_deduped} deduped",
        ]
        top = self.top_failure_patterns(5)
        if top:
            lines.append(f"  Top failures   : {top}")
        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────

    def save(self) -> None:
        """Persist all data to ``self.data_dir``."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            # Proven cache
            proven_path = self.data_dir / "proven_cache.jsonl"
            with open(proven_path, "w") as fh:
                for entry in self._proven.values():
                    fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

            # Experience
            exp_path = self.data_dir / "experience.jsonl"
            with open(exp_path, "w") as fh:
                for rec in self._experience:
                    fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

            # Failure patterns
            fp_path = self.data_dir / "failure_patterns.json"
            with open(fp_path, "w") as fh:
                json.dump(dict(self._failure_patterns), fh, indent=2)

            # Stats snapshot
            st_path = self.data_dir / "stats.json"
            with open(st_path, "w") as fh:
                json.dump(self.stats().to_dict(), fh, indent=2)

            # Generator memory (cross-session)
            gm_path = self.data_dir / "generator_memory.json"
            with open(gm_path, "w") as fh:
                json.dump(self._generator_memory, fh, indent=2)

        logger.info("KnowledgeStore saved to %s", self.data_dir)

    def load(self) -> int:
        """Load previously persisted data.  Returns count of entries loaded.

        Existing in-memory data is *merged* (not replaced), with dedup.
        """
        loaded = 0

        # Proven cache
        proven_path = self.data_dir / "proven_cache.jsonl"
        if proven_path.exists():
            with open(proven_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    entry = ProvenEntry.from_dict(json.loads(line))
                    key = _goal_key(entry.assumptions, entry.goal)
                    with self._lock:
                        if key not in self._proven:
                            self._proven[key] = entry
                            loaded += 1

        # Experience
        exp_path = self.data_dir / "experience.jsonl"
        if exp_path.exists():
            with open(exp_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = ExperienceRecord.from_dict(json.loads(line))
                    with self._lock:
                        if rec.fingerprint and rec.fingerprint not in self._experience_fps:
                            self._experience_fps.add(rec.fingerprint)
                            self._experience.append(rec)
                            loaded += 1

        # Failure patterns
        fp_path = self.data_dir / "failure_patterns.json"
        if fp_path.exists():
            with open(fp_path) as fh:
                data = json.load(fh)
            with self._lock:
                for pat, cnt in data.items():
                    self._failure_patterns[pat] += cnt
                    loaded += 1

        # Generator memory (cross-session)
        gm_path = self.data_dir / "generator_memory.json"
        if gm_path.exists():
            with open(gm_path) as fh:
                gm_data = json.load(fh)
            with self._lock:
                for gen_name, mem in gm_data.items():
                    if gen_name not in self._generator_memory:
                        self._generator_memory[gen_name] = mem
                    else:
                        # Merge: accumulate counts, keep max consecutive_failures
                        existing = self._generator_memory[gen_name]
                        existing["attempts"] += mem.get("attempts", 0)
                        existing["successes"] += mem.get("successes", 0)
                        existing["consecutive_failures"] = max(
                            existing["consecutive_failures"],
                            mem.get("consecutive_failures", 0),
                        )
                    loaded += 1

        logger.info("KnowledgeStore loaded %d entries from %s", loaded, self.data_dir)
        return loaded

    # ── Bulk operations ──────────────────────────────────────

    def merge_from(self, other: "KnowledgeStore") -> int:
        added = 0
        with other._lock:
            proven_items = list(other._proven.items())
            experience_items = list(other._experience)
            failure_items = dict(other._failure_patterns)
        with self._lock:
            for key, entry in proven_items:
                if key not in self._proven:
                    self._proven[key] = entry
                    added += 1
            for rec in experience_items:
                if rec.fingerprint not in self._experience_fps:
                    self._experience_fps.add(rec.fingerprint)
                    self._experience.append(rec)
                    added += 1
                else:
                    self._dedup_skipped += 1
            for pat, cnt in failure_items.items():
                self._failure_patterns[pat] += cnt
        return added

    def clear(self) -> None:
        with self._lock:
            self._proven.clear()
            self._proven_hits = 0
            self._experience.clear()
            self._experience_fps.clear()
            self._dedup_skipped = 0
            self._failure_patterns.clear()
            self._generator_memory.clear()
            self._guidance_cache.clear()

    # ── Generator memory (cross-session) ─────────────────────

    def record_generator_outcome(
        self, gen_name: str, success: bool,
    ) -> None:
        """Track per-generator performance across sessions.

        Persisted via ``save()`` so cold-start generators are remembered
        even after process restarts.
        """
        with self._lock:
            if gen_name not in self._generator_memory:
                self._generator_memory[gen_name] = {
                    "attempts": 0,
                    "successes": 0,
                    "consecutive_failures": 0,
                }
            mem = self._generator_memory[gen_name]
            mem["attempts"] += 1
            if success:
                mem["successes"] += 1
                mem["consecutive_failures"] = 0
            else:
                mem["consecutive_failures"] += 1

    def is_generator_cold(
        self, gen_name: str, threshold: int = 8,
    ) -> bool:
        """Return True if *gen_name* has >= *threshold* consecutive failures.

        Used by the conjecture pipeline to skip generators that have been
        consistently unproductive, even across sessions.
        """
        with self._lock:
            mem = self._generator_memory.get(gen_name)
            if mem is None:
                return False
            return mem["consecutive_failures"] >= threshold

    def generator_memory_summary(self) -> Dict[str, Dict[str, int]]:
        """Return a snapshot of per-generator statistics."""
        with self._lock:
            return {k: dict(v) for k, v in self._generator_memory.items()}

    # ═══════════════════════════════════════════════════════════════════
    # Guidance API — knowledge actively informs the agent
    # ═══════════════════════════════════════════════════════════════════
    # These methods close the mutual promotion loop:
    #   Agent → knowledge (via record_proven / record_experience)
    #   Knowledge → agent (via guidance methods below)
    #
    #   • rule_success_profile()    → which rules succeed overall
    #   • suggest_rule_order()      → rank rules for a specific goal
    #   • predicate_coverage()      → detect under-explored areas
    #   • under_explored_predicates → focus evolution on gaps
    #   • proven_rule_chains()      → reusable proof templates
    #   • structural_dedup_check()  → catch predicate-family duplicates
    #   • difficulty_profile()      → adaptive difficulty estimation
    #   • generator_success_rates() → rank generators for evolution

    def _ensure_guidance_cache(self) -> None:
        """Lazily compute and cache guidance statistics.

        The cache is invalidated whenever new experience is recorded.
        """
        with self._lock:
            cache_key = len(self._experience)
            if self._guidance_cache.get("_version") == cache_key:
                return  # already up to date

            # ── rule → success/fail counts ─────────────────────
            rule_stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
            # ── rule → per-goal-predicate success rate ─────────
            rule_by_goal: Dict[str, Dict[str, List[int]]] = defaultdict(
                lambda: defaultdict(lambda: [0, 0])
            )
            # ── goal predicate → success/fail counts ──────────
            pred_stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
            # ── rule chains: rule sequences in successful proofs ──
            rule_chains: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
            # ── difficulty → success/fail counts ──────────────
            diff_stats: Dict[int, List[int]] = defaultdict(lambda: [0, 0])

            for rec in self._experience:
                goal_pred = rec.goal.predicate
                idx = 0 if rec.success else 1
                pred_stats[goal_pred][idx] += 1
                diff_stats[rec.difficulty][idx] += 1

                if rec.success and rec.steps:
                    chain = tuple(s.rule_name for s in rec.steps)
                    rule_chains[goal_pred].append(chain)
                    for step in rec.steps:
                        rule_stats[step.rule_name][0] += 1
                        rule_by_goal[step.rule_name][goal_pred][0] += 1
                elif rec.steps:
                    # Partial progress but ultimately failed
                    for step in rec.steps:
                        rule_stats[step.rule_name][1] += 1
                        rule_by_goal[step.rule_name][goal_pred][1] += 1

            self._guidance_cache["rule_stats"] = dict(rule_stats)
            self._guidance_cache["rule_by_goal"] = {
                k: dict(v) for k, v in rule_by_goal.items()
            }
            self._guidance_cache["pred_stats"] = dict(pred_stats)
            self._guidance_cache["rule_chains"] = dict(rule_chains)
            self._guidance_cache["diff_stats"] = dict(diff_stats)
            self._guidance_cache["_version"] = cache_key

    def rule_success_profile(self) -> Dict[str, Tuple[int, int, float]]:
        """Return ``{rule_name: (successes, failures, rate)}`` from experience.

        The rate is Laplace-smoothed to avoid zero-division.
        """
        self._ensure_guidance_cache()
        with self._lock:
            stats = self._guidance_cache.get("rule_stats", {})
            result: Dict[str, Tuple[int, int, float]] = {}
            for rule, (s, f) in stats.items():
                rate = (s + 1) / (s + f + 2)  # Laplace smoothing
                result[rule] = (s, f, rate)
            return result

    def suggest_rule_order(self, goal_pred: str) -> List[Tuple[str, float]]:
        """Rank rules by predicted usefulness for proving *goal_pred*.

        Returns ``[(rule_name, score), ...]`` in descending score order.
        Rules that historically appear in successful proofs of the same
        goal predicate are ranked highest.  This is the primary mechanism
        by which accumulated experience guides future search.
        """
        self._ensure_guidance_cache()
        with self._lock:
            rule_by_goal = self._guidance_cache.get("rule_by_goal", {})
            rule_stats = self._guidance_cache.get("rule_stats", {})
            scores: Dict[str, float] = {}

            for rule_name in set(rule_stats.keys()):
                # Goal-specific success rate (weight 3x)
                goal_data = rule_by_goal.get(rule_name, {}).get(goal_pred)
                if goal_data:
                    gs, gf = goal_data
                    goal_rate = (gs + 1) / (gs + gf + 2)
                else:
                    goal_rate = 0.5  # neutral if no data

                # Global success rate (weight 1x)
                global_data = rule_stats.get(rule_name, [0, 0])
                gs_g, gf_g = global_data
                global_rate = (gs_g + 1) / (gs_g + gf_g + 2)

                scores[rule_name] = goal_rate * 3.0 + global_rate

            return sorted(scores.items(), key=lambda x: -x[1])

    def predicate_coverage(self) -> Dict[str, Dict[str, int]]:
        """Return ``{predicate: {"success": n, "fail": m, "proven": p}}``.

        Useful for identifying under-explored predicates that the
        evolution loop should prioritise.
        """
        self._ensure_guidance_cache()
        with self._lock:
            pred_stats = self._guidance_cache.get("pred_stats", {})
            proven_by_pred: Dict[str, int] = defaultdict(int)
            for entry in self._proven.values():
                proven_by_pred[entry.goal.predicate] += 1

            result: Dict[str, Dict[str, int]] = {}
            all_preds = set(pred_stats.keys()) | set(proven_by_pred.keys())
            for pred in all_preds:
                s_f = pred_stats.get(pred, [0, 0])
                s, f = s_f[0], s_f[1]
                result[pred] = {
                    "success": s,
                    "fail": f,
                    "proven": proven_by_pred.get(pred, 0),
                }
            return result

    def under_explored_predicates(self, top_n: int = 5) -> List[str]:
        """Return predicates with lowest coverage, for targeted exploration."""
        cov = self.predicate_coverage()
        if not cov:
            return []
        scored = [
            (pred, data["success"] + data["proven"])
            for pred, data in cov.items()
        ]
        scored.sort(key=lambda x: x[1])
        return [p for p, _ in scored[:top_n]]

    def proven_rule_chains(
        self, goal_pred: str, max_chains: int = 10,
    ) -> List[Tuple[str, ...]]:
        """Return rule-name sequences from successful proofs of *goal_pred*.

        These chains serve as **proof templates** — search can prioritise
        rules that appear early in known-good chains, and conjecture
        generation can prefer bridge compositions that match templates.
        """
        self._ensure_guidance_cache()
        with self._lock:
            chains = self._guidance_cache.get("rule_chains", {})
            all_chains = chains.get(goal_pred, [])
            # Deduplicate and return most common ones
            from collections import Counter
            chain_counts = Counter(all_chains)
            return [c for c, _ in chain_counts.most_common(max_chains)]

    def structural_dedup_check(
        self,
        assumptions: FrozenSet[Fact],
        goal: Fact,
    ) -> bool:
        """Check if theorem is a structural (predicate-family) duplicate.

        Returns True if this theorem is merely a predicate-family
        substitution of something already in the proven cache.
        This catches e.g. Parallel↔Perpendicular swaps that are
        semantically distinct but structurally identical.
        """
        sfp = structural_theorem_fingerprint(assumptions, goal)
        with self._lock:
            if not hasattr(self, "_structural_fps"):
                # Build structural fingerprint index from proven cache
                self._structural_fps: Set[str] = set()
                for entry in self._proven.values():
                    fp = structural_theorem_fingerprint(
                        entry.assumptions, entry.goal
                    )
                    self._structural_fps.add(fp)
            return sfp in self._structural_fps

    def difficulty_profile(self) -> Dict[int, Tuple[int, int, float]]:
        """Return ``{difficulty: (success, fail, rate)}`` for adaptive control.

        The evolution loop can use this to decide whether to escalate
        difficulty (high solve rate) or stay at current level (low rate).
        """
        self._ensure_guidance_cache()
        with self._lock:
            diff_stats = self._guidance_cache.get("diff_stats", {})
            result: Dict[int, Tuple[int, int, float]] = {}
            for diff, (s, f) in diff_stats.items():
                rate = (s + 1) / (s + f + 2)
                result[diff] = (s, f, rate)
            return result

    def generator_success_rates(self) -> Dict[str, Tuple[int, int, float]]:
        """Return ``{generator_name: (success, fail, rate)}`` from experience.

        Generator names are extracted from failure-pattern keys that
        follow the ``gen_name:...`` convention.  This helps the evolution
        loop focus on generators that produce solvable problems.
        """
        with self._lock:
            gen_stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
            for pat, cnt in self._failure_patterns.items():
                # Pattern format: "gen_name:detail" or free-form
                if ":" in pat:
                    gen_name = pat.split(":")[0]
                    gen_stats[gen_name][1] += cnt

            result: Dict[str, Tuple[int, int, float]] = {}
            for gen, (s, f) in gen_stats.items():
                rate = (s + 1) / (s + f + 2)
                result[gen] = (s, f, rate)
            return result

    def bridge_success_rates(self) -> Dict[str, Tuple[int, int, float]]:
        """Return ``{rule_name: (success, fail, rate)}`` for rule bridges.

        Used by conjecture.py to weight bridge selection: prefer bridges
        that historically lead to provable theorems.
        """
        return self.rule_success_profile()

    def guidance_summary(self) -> str:
        """Human-readable summary of guidance statistics."""
        lines = [self.summary(), ""]

        # Rule success rates
        profile = self.rule_success_profile()
        if profile:
            top_rules = sorted(profile.items(), key=lambda x: -x[1][2])[:5]
            lines.append("  Top rules (by success rate):")
            for rule, (s, f, r) in top_rules:
                lines.append(f"    {rule}: {s}✓ {f}✗ ({r:.0%})")

        # Predicate coverage
        cov = self.predicate_coverage()
        if cov:
            under = self.under_explored_predicates(3)
            if under:
                lines.append(f"  Under-explored predicates: {', '.join(under)}")

        # Difficulty profile
        diff = self.difficulty_profile()
        if diff:
            lines.append("  Difficulty solve rates:")
            for d in sorted(diff.keys()):
                s, f, r = diff[d]
                lines.append(f"    Level {d}: {s}✓ {f}✗ ({r:.0%})")

        return "\n".join(lines)


# ── Module-level singleton ───────────────────────────────────────────

_global_store: Optional[KnowledgeStore] = None
_global_lock = threading.Lock()


def get_global_store(data_dir: str | Path | None = None) -> KnowledgeStore:
    """Return (or create) the process-wide singleton KnowledgeStore.

    On first call, attempts to ``load()`` any previously persisted data
    from the data directory.
    """
    global _global_store
    if _global_store is None:
        with _global_lock:
            if _global_store is None:
                store = KnowledgeStore(data_dir=data_dir)
                try:
                    store.load()
                except Exception as exc:
                    logger.warning("Could not load prior knowledge: %s", exc)
                _global_store = store
    return _global_store
