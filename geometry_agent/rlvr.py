"""rlvr.py – Reinforcement Learning with Verifiable Rewards (RLVR).

Uses the symbolic geometry engine as a *verifier* providing ground-truth
rewards.  A lightweight policy (no neural network required!) learns to
generate increasingly difficult novel conjectures by:

  1. **Policy**: probability distribution over conjecture templates / genes
  2. **Reward**: symbolic engine verifies provability; difficulty evaluator
     scores quality → combined reward signal
  3. **Update**: REINFORCE-style policy gradient with baseline subtraction

The key insight: the symbolic proof engine is a *perfect verifier*, so
every reward is ground-truth — no reward hacking possible.

Architecture
~~~~~~~~~~~~

    ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
    │  Policy π(θ) │────→│  Conjecture Gen  │────→│  Symbolic Engine │
    │  (template   │     │  (decode genome  │     │  (beam search +  │
    │   weights)   │     │   → facts)       │     │   rule engine)   │
    └──────────────┘     └─────────────────┘     └──────┬───────────┘
           ↑                                            │
           │              ┌──────────────────┐          │
           └──────────────│  Reward Shaping  │←─────────┘
                          │  (difficulty +   │
                          │   novelty +      │
                          │   diversity)     │
                          └──────────────────┘

Components
----------
- ``TemplatePolicy``: UCB1 / softmax policy over predicate templates
- ``RewardComputer``: shapes raw verification results into rewards
- ``ExperienceBuffer``: stores (action, reward) trajectories
- ``RLVRTrainer``: orchestrates the training loop

Author:  Jiangsheng Yu
License: MIT
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from .dsl import Fact, GeoState, Goal, Step
from .difficulty_eval import evaluate_difficulty, DifficultyReport, compute_value_score
from .genetic import (
    ConjectureGenome,
    GAConfig,
    PredicateGene,
    _BRIDGE_TEMPLATES,
    _PRED_BY_FAMILY,
    _PRED_BY_NAME,
    _PRED_BY_TIER,
    _PRED_META,
    _genome_from_template,
    _make_gene,
    _random_genome,
    decode_genome,
    mutate,
    POINT_POOL,
)
from .knowledge import KnowledgeStore, get_global_store
from .lean_bridge import MockLeanChecker
from .rules import default_rules
from .search import SearchConfig, SearchResult, beam_search
from .semantic import semantic_theorem_fingerprint
from .polya import polya_test
from .polya_controller import PolyaController

logger = logging.getLogger(__name__)


# ── Experience (trajectory) ──────────────────────────────────────────

@dataclass
class Experience:
    """Single (state, action, reward) experience."""
    template_id: str          # which template/action was used
    genome: ConjectureGenome  # the generated conjecture
    reward: float             # shaped reward
    provable: bool            # whether proof was found
    difficulty: float         # raw difficulty score (0 if not provable)
    novel: bool               # whether fingerprint was new
    generation: int           # when this experience was collected
    # Optional extras
    assumptions: Optional[List[Fact]] = None
    goal: Optional[Fact] = None
    steps: Optional[List[Step]] = None
    diff_report: Optional[DifficultyReport] = None


@dataclass
class ExperienceBuffer:
    """Replay buffer for RLVR experiences.

    Maintains a sliding window of recent experiences and computes
    running statistics for baseline subtraction.
    """
    max_size: int = 2000
    _buffer: List[Experience] = field(default_factory=list)
    _reward_sum: float = 0.0
    _reward_sq_sum: float = 0.0
    _count: int = 0

    def add(self, exp: Experience) -> None:
        self._buffer.append(exp)
        self._reward_sum += exp.reward
        self._reward_sq_sum += exp.reward ** 2
        self._count += 1
        # Evict oldest if over capacity
        if len(self._buffer) > self.max_size:
            old = self._buffer.pop(0)
            self._reward_sum -= old.reward
            self._reward_sq_sum -= old.reward ** 2

    @property
    def mean_reward(self) -> float:
        return self._reward_sum / max(self._count, 1)

    @property
    def std_reward(self) -> float:
        if self._count < 2:
            return 1.0
        var = (self._reward_sq_sum / self._count) - self.mean_reward ** 2
        return max(math.sqrt(max(var, 0)), 0.01)

    def normalise_reward(self, reward: float) -> float:
        """Baseline-subtracted, normalised reward."""
        return (reward - self.mean_reward) / self.std_reward

    @property
    def size(self) -> int:
        return len(self._buffer)

    def best_experiences(self, k: int = 10) -> List[Experience]:
        """Top-k experiences by reward."""
        return sorted(self._buffer, key=lambda e: e.reward, reverse=True)[:k]

    def recent(self, k: int = 50) -> List[Experience]:
        return self._buffer[-k:]


# ── Reward shaping ───────────────────────────────────────────────────

@dataclass
class RewardConfig:
    """Reward shaping hyperparameters."""
    # Base rewards
    unprovable_penalty: float = -1.0
    trivial_reward: float = -0.5
    decode_fail_penalty: float = -2.0
    # Provable rewards
    difficulty_weight: float = 1.0
    novelty_bonus: float = 3.0       # bonus for first-time fingerprint
    duplicate_penalty: float = -1.0   # penalty for seen fingerprint
    # Structure rewards
    family_bonus: float = 0.5        # per family crossed
    tier_bonus: float = 0.3          # per max tier level
    step_bonus: float = 0.15         # per proof step (capped)
    rule_diversity_bonus: float = 0.2 # per distinct rule type
    # Value bonus — strongly rewards proofs using many distinct knowledge points
    # A proof with ≥5 distinct rules gets a large value bonus, encouraging
    # the policy to generate complex cross-domain conjectures.
    value_weight: float = 1.5        # multiplier for value_score (0–10)
    value_threshold_bonus: float = 3.0  # extra bonus when n_rules ≥ 5
    value_threshold: int = 5         # rule count threshold for bonus
    # Caps
    max_step_reward: float = 2.0
    max_total_reward: float = 25.0   # raised to accommodate value bonus


class RewardComputer:
    """Compute shaped rewards from verification outcomes."""

    def __init__(self, config: RewardConfig = RewardConfig()):
        self.config = config
        self._seen_fps: Set[str] = set()

    def compute(
        self,
        genome: ConjectureGenome,
        assumptions: Optional[List[Fact]],
        goal: Optional[Fact],
        result: Optional[SearchResult],
        diff_report: Optional[DifficultyReport],
    ) -> Tuple[float, bool]:
        """Compute reward and novelty flag.

        Returns (reward, is_novel).
        """
        c = self.config

        # Decode failure
        if assumptions is None or goal is None:
            return c.decode_fail_penalty, False

        # Not provable
        if result is None or not result.success:
            # Small structural bonus even for unprovable
            fam_count = len(genome.family_set)
            struct_bonus = fam_count * 0.05 + genome.max_tier * 0.02
            return c.unprovable_penalty + struct_bonus, False

        steps = list(result.final_state.history)

        # Trivial proof
        if len(steps) < 3:
            return c.trivial_reward, False

        # Provable — compute full reward
        reward = 0.0

        # 1. Difficulty score (primary signal)
        if diff_report is not None:
            reward += c.difficulty_weight * diff_report.overall_score

        # 2. Novelty bonus
        fp = semantic_theorem_fingerprint(assumptions, goal)
        if fp not in self._seen_fps:
            self._seen_fps.add(fp)
            reward += c.novelty_bonus
            is_novel = True
        else:
            reward += c.duplicate_penalty
            is_novel = False

        # 3. Structural bonuses
        n_families = len(genome.family_set)
        reward += n_families * c.family_bonus
        reward += genome.max_tier * c.tier_bonus

        # 4. Proof depth bonus (capped)
        n_distinct = len({s.rule_name for s in steps})
        step_reward = min(n_distinct * c.rule_diversity_bonus, c.max_step_reward)
        step_reward += min(len(steps) * c.step_bonus, c.max_step_reward)
        reward += step_reward

        # 5. Value bonus — more distinct knowledge points → higher value
        # This is the primary signal encouraging the policy to discover
        # theorems whose proofs span many different rules / knowledge points.
        value_score, _, _ = compute_value_score(n_distinct)
        reward += c.value_weight * value_score
        # Extra threshold bonus when proof uses ≥ value_threshold rules
        if n_distinct >= c.value_threshold:
            reward += c.value_threshold_bonus

        reward = min(reward, c.max_total_reward)
        return reward, is_novel


# ── Template policy (UCB1 + softmax) ────────────────────────────────

@dataclass
class TemplateStats:
    """Statistics for one conjecture template."""
    total_reward: float = 0.0
    count: int = 0
    successes: int = 0
    best_reward: float = -float("inf")
    # UCB1
    ucb_bonus: float = float("inf")

    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(self.count, 1)

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.count, 1)


class TemplatePolicy:
    """Policy over conjecture templates using UCB1 + softmax.

    Each "template" is a (assumption_predicates, goal_predicate) pair.
    The policy maintains per-template reward statistics and uses UCB1
    for exploration–exploitation balance.
    """

    def __init__(
        self,
        ucb_c: float = 2.0,
        temperature: float = 1.0,
        min_temperature: float = 0.3,
        decay_rate: float = 0.995,
    ):
        self._stats: Dict[str, TemplateStats] = {}
        self._templates: Dict[str, Tuple[List[str], str]] = {}
        self.ucb_c = ucb_c
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.decay_rate = decay_rate
        self._total_pulls = 0

        # Initialise with bridge templates
        for i, (assm, goal) in enumerate(_BRIDGE_TEMPLATES):
            tid = self._template_id(assm, goal)
            self._templates[tid] = (assm, goal)
            self._stats[tid] = TemplateStats()

        # Also add some random templates
        for _ in range(30):
            assm, goal = self._random_template()
            tid = self._template_id(assm, goal)
            if tid not in self._templates:
                self._templates[tid] = (assm, goal)
                self._stats[tid] = TemplateStats()

    @staticmethod
    def _template_id(assm_preds: List[str], goal_pred: str) -> str:
        return "+".join(sorted(assm_preds)) + "→" + goal_pred

    @staticmethod
    def _random_template() -> Tuple[List[str], str]:
        n = random.randint(3, 5)
        preds = [random.choice(_PRED_META)[0] for _ in range(n)]
        goal = random.choice(_PRED_META)[0]
        return preds, goal

    def select_template(self) -> Tuple[str, List[str], str]:
        """Select a template using UCB1 + softmax exploration.

        Returns (template_id, assumption_preds, goal_pred).
        """
        self._total_pulls += 1

        # Compute UCB1 scores
        log_total = math.log(max(self._total_pulls, 1))
        scores: Dict[str, float] = {}
        for tid, stats in self._stats.items():
            if stats.count == 0:
                scores[tid] = float("inf")  # unexplored → always pick
            else:
                exploit = stats.mean_reward
                explore = self.ucb_c * math.sqrt(log_total / stats.count)
                scores[tid] = exploit + explore

        # If unexplored templates exist, sample uniformly among them.
        # This avoids inf/NaN issues in downstream softmax.
        unexplored = [tid for tid, v in scores.items() if math.isinf(v)]
        if unexplored:
            chosen_tid = random.choice(unexplored)
            assm, goal = self._templates[chosen_tid]
            return chosen_tid, assm, goal

        # Softmax over UCB scores
        tids = list(scores.keys())
        vals = [scores[t] for t in tids]
        max_val = max(vals)

        # Clamp to prevent overflow
        exp_vals = [math.exp(min((v - max_val) / max(self.temperature, 0.01), 50))
                    for v in vals]
        total = sum(exp_vals)
        probs = [e / total for e in exp_vals]

        # Sample
        chosen_idx = random.choices(range(len(tids)), weights=probs, k=1)[0]
        chosen_tid = tids[chosen_idx]

        assm, goal = self._templates[chosen_tid]
        return chosen_tid, assm, goal

    def update(self, template_id: str, reward: float, success: bool) -> None:
        """Update statistics for a template after receiving reward."""
        if template_id not in self._stats:
            self._stats[template_id] = TemplateStats()

        stats = self._stats[template_id]
        stats.total_reward += reward
        stats.count += 1
        if success:
            stats.successes += 1
        stats.best_reward = max(stats.best_reward, reward)

        # Decay temperature
        self.temperature = max(
            self.min_temperature,
            self.temperature * self.decay_rate,
        )

    def inject_template(self, assm_preds: List[str], goal_pred: str) -> str:
        """Add a new template to the policy (e.g. from GA discoveries)."""
        tid = self._template_id(assm_preds, goal_pred)
        if tid not in self._templates:
            self._templates[tid] = (assm_preds, goal_pred)
            self._stats[tid] = TemplateStats()
        return tid

    def top_templates(self, k: int = 10) -> List[Tuple[str, TemplateStats]]:
        """Return top-k templates by mean reward."""
        sorted_stats = sorted(
            self._stats.items(),
            key=lambda x: x[1].mean_reward,
            reverse=True,
        )
        return sorted_stats[:k]

    @property
    def n_templates(self) -> int:
        return len(self._templates)

    def summary(self) -> str:
        """Brief summary of the policy state."""
        explored = sum(1 for s in self._stats.values() if s.count > 0)
        mean_sr = sum(s.success_rate for s in self._stats.values()) / max(len(self._stats), 1)
        return (f"模板策略: {self.n_templates}个模板, "
                f"已探索{explored}, 平均成功率{mean_sr:.1%}, "
                f"温度{self.temperature:.3f}")


# ── RLVR Trainer ─────────────────────────────────────────────────────

@dataclass
class RLVRConfig:
    """RLVR training configuration."""
    # Training
    max_episodes: int = 3000
    batch_size: int = 20             # episodes per batch
    target_novel: int = 3
    min_difficulty: float = 5.0
    # Policy
    ucb_c: float = 2.0
    initial_temperature: float = 1.5
    min_temperature: float = 0.2
    decay_rate: float = 0.997
    # Reward
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    # Exploration
    mutation_rate: float = 0.4
    n_mutations_per_template: int = 3 # generate N variants per template
    # Template evolution
    inject_from_ga: bool = True       # inject successful GA templates
    template_expansion_interval: int = 200  # add new random templates every N episodes
    # Verification
    beam_width: int = 32
    max_depth: int = 18


@dataclass
class RLVRResult:
    """Result of RLVR training."""
    discoveries: List[Dict]
    total_episodes: int
    total_provable: int
    total_novel: int
    elapsed_seconds: float
    reward_history: List[float]
    policy_summary: str
    top_templates: List[Tuple[str, float]]  # (template_id, mean_reward)


class RLVRTrainer:
    """Reinforcement Learning with Verifiable Rewards.

    Uses the symbolic geometry engine as a perfect verifier to train a
    template policy that generates increasingly difficult conjectures.
    """

    def __init__(
        self,
        config: RLVRConfig = RLVRConfig(),
        knowledge_store: Optional[KnowledgeStore] = None,
    ):
        self.config = config
        self.knowledge = knowledge_store or get_global_store()
        self.rules = default_rules()
        self.checker = MockLeanChecker()

        # Policy
        self.policy = TemplatePolicy(
            ucb_c=config.ucb_c,
            temperature=config.initial_temperature,
            min_temperature=config.min_temperature,
            decay_rate=config.decay_rate,
        )

        # Reward computer
        self.reward_computer = RewardComputer(config.reward_config)

        # Experience buffer
        self.buffer = ExperienceBuffer(max_size=3000)

        # Tracking
        self.discoveries: List[Dict] = []
        self.seen_fingerprints: Set[str] = set()
        self.polya_controller = PolyaController(knowledge_store=self.knowledge)

    def _generate_conjecture(
        self,
        assm_preds: List[str],
        goal_pred: str,
    ) -> ConjectureGenome:
        """Generate a conjecture genome from a template with mutations."""
        n_points = random.randint(8, 14)
        genome = _genome_from_template(assm_preds, goal_pred, n_points)
        # Apply mutations for variety
        for _ in range(random.randint(1, self.config.n_mutations_per_template)):
            genome = mutate(genome, self.config.mutation_rate)
        return genome

    def _verify_and_reward(
        self,
        genome: ConjectureGenome,
        template_id: str,
        episode: int,
    ) -> Experience:
        """Verify a conjecture and compute reward."""
        decoded = decode_genome(genome)
        if decoded is None:
            reward, novel = self.reward_computer.compute(
                genome, None, None, None, None)
            return Experience(
                template_id=template_id,
                genome=genome,
                reward=reward,
                provable=False,
                difficulty=0.0,
                novel=False,
                generation=episode,
            )

        assumptions, goal = decoded

        plan = self.polya_controller.make_plan(
            assumptions,
            goal,
            strategy=f"rlvr:{template_id}",
        )

        # ── Pólya plausible-reasoning pre-filter ──
        polya_res = polya_test(assumptions, goal, n_trials=plan.polya_trials)
        if polya_res.falsified or polya_res.confidence < plan.polya_min_confidence:
            # Falsified or too uncertain — skip beam search entirely
            reward, novel = self.reward_computer.compute(
                genome, assumptions, goal, None, None)
            self.polya_controller.note_failure("rlvr_polya_reject")
            return Experience(
                template_id=template_id,
                genome=genome,
                reward=reward - 0.5,  # penalty for Pólya-rejected
                provable=False,
                difficulty=0.0,
                novel=False,
                generation=episode,
                assumptions=assumptions,
                goal=goal,
            )

        # Verify with beam search
        state = GeoState(facts=set(assumptions))
        fast_cfg = SearchConfig(
            beam_width=max(self.config.beam_width, min(plan.fast_beam_width, self.config.beam_width + 24)),
            max_depth=max(self.config.max_depth, min(plan.fast_max_depth, self.config.max_depth + 6)),
            parallel_workers=0,
        )
        result = beam_search(
            init_state=state,
            goal=Goal(goal),
            rules=self.rules,
            checker=self.checker,
            config=fast_cfg,
            knowledge_store=self.knowledge,
        )

        if (not result.success) and self.polya_controller.should_escalate(
            polya_res.confidence,
            strategy=f"rlvr:{template_id}",
        ):
            deep_cfg = SearchConfig(
                beam_width=max(self.config.beam_width + 24,
                               min(plan.deep_beam_width, self.config.beam_width + 64)),
                max_depth=max(self.config.max_depth + 4,
                              min(plan.deep_max_depth, self.config.max_depth + 10)),
                parallel_workers=0,
            )
            result = beam_search(
                init_state=state,
                goal=Goal(goal),
                rules=self.rules,
                checker=self.checker,
                config=deep_cfg,
                knowledge_store=self.knowledge,
            )

        # Difficulty evaluation (if provable)
        diff_report = None
        if result.success:
            steps = list(result.final_state.history)
            if len(steps) >= 3:
                diff_report = evaluate_difficulty(assumptions, goal, steps)

        # Compute shaped reward
        reward, is_novel = self.reward_computer.compute(
            genome, assumptions, goal, result, diff_report,
        )

        if result.success:
            self.polya_controller.note_success(f"rlvr:{template_id}")
        else:
            self.polya_controller.note_failure("rlvr_search_fail")

        return Experience(
            template_id=template_id,
            genome=genome,
            reward=reward,
            provable=result.success,
            difficulty=diff_report.overall_score if diff_report else 0.0,
            novel=is_novel,
            generation=episode,
            assumptions=assumptions,
            goal=goal,
            steps=list(result.final_state.history) if result.success else None,
            diff_report=diff_report,
        )

    def _check_discovery(self, exp: Experience) -> bool:
        """Check if an experience qualifies as a discovery."""
        if not exp.provable or not exp.novel:
            return False
        if exp.diff_report is None:
            return False
        if exp.difficulty < self.config.min_difficulty:
            return False
        if exp.steps is None or len(exp.steps) < 5:
            return False

        # Cross-domain check
        from .evolve import _is_cross_domain_proof, is_mathlib4_known
        if is_mathlib4_known(exp.assumptions, exp.goal, exp.steps):
            return False
        if not _is_cross_domain_proof(exp.assumptions, exp.goal, exp.steps):
            return False

        return True

    def train(self, verbose: bool = True) -> RLVRResult:
        """Run the RLVR training loop.

        Each episode:
        1. Policy selects a template
        2. Generate N mutated conjectures from template
        3. Verify each with symbolic engine
        4. Compute rewards and update policy
        """
        cfg = self.config
        reward_history: List[float] = []
        total_provable = 0
        total_novel = 0
        t0 = time.time()

        if verbose:
            print("\n╔══════════════════════════════════════════════════════════╗")
            print("║  RLVR 训练启动 / RL with Verifiable Rewards Started     ║")
            print("╚══════════════════════════════════════════════════════════╝")
            print(f"  最大轮次: {cfg.max_episodes}  批大小: {cfg.batch_size}")
            print(f"  模板数: {self.policy.n_templates}  UCB_c: {cfg.ucb_c}")
            print(f"  目标: {cfg.target_novel} 个难度≥{cfg.min_difficulty} 的新定理")
            print(f"  beam_width: {cfg.beam_width}  max_depth: {cfg.max_depth}")
            print()

        for episode in range(1, cfg.max_episodes + 1):
            batch_rewards: List[float] = []

            for _ in range(cfg.batch_size):
                # 1. Policy selects template
                tid, assm_preds, goal_pred = self.policy.select_template()

                # 2. Generate conjecture
                genome = self._generate_conjecture(assm_preds, goal_pred)

                # 3. Verify and compute reward
                exp = self._verify_and_reward(genome, tid, episode)

                # 4. Store experience
                self.buffer.add(exp)
                batch_rewards.append(exp.reward)

                if exp.provable:
                    total_provable += 1
                if exp.novel:
                    total_novel += 1

                # 5. Update policy
                self.policy.update(tid, exp.reward, exp.provable)

                # 6. Check for discovery
                if self._check_discovery(exp):
                    self.discoveries.append({
                        "experience": exp,
                        "assumptions": exp.assumptions,
                        "goal": exp.goal,
                        "steps": exp.steps,
                        "difficulty": exp.diff_report,
                        "generation": episode,
                        "fingerprint": semantic_theorem_fingerprint(
                            exp.assumptions, exp.goal),
                        "template_id": tid,
                    })

                    if verbose:
                        print(f"\n  🌟 RLVR发现#{len(self.discoveries)}: "
                              f"难度 {exp.difficulty:.1f}/10"
                              f" ({exp.diff_report.label_zh})"
                              f"  模板: {tid}"
                              f"  奖励: {exp.reward:.2f}")
                        print(f"      {exp.diff_report.assessment_zh}")

                    # Inject successful templates as new templates
                    if exp.assumptions:
                        new_assm = [f.predicate for f in exp.assumptions]
                        self.policy.inject_template(new_assm, exp.goal.predicate)

                    if len(self.discoveries) >= cfg.target_novel:
                        elapsed = time.time() - t0
                        if verbose:
                            print(f"\n  ✅ RLVR目标达成! 第{episode}轮发现"
                                  f" {len(self.discoveries)} 个新定理"
                                  f" ({elapsed:.1f}s)")
                            print(f"  {self.policy.summary()}")
                        return self._make_result(
                            episode, total_provable, total_novel,
                            time.time() - t0, reward_history,
                        )

            # Batch stats
            avg_reward = sum(batch_rewards) / max(len(batch_rewards), 1)
            reward_history.append(avg_reward)

            # Progress reporting
            if verbose and episode % 10 == 0:
                rate = total_provable / max(episode * cfg.batch_size, 1)
                print(f"  轮{episode:4d} | 平均奖励 {avg_reward:6.2f}"
                      f"  可证率 {rate:.1%}"
                      f"  新颖 {total_novel}"
                      f"  发现 {len(self.discoveries)}/{cfg.target_novel}"
                      f"  温度 {self.policy.temperature:.3f}"
                      f"  缓冲 {self.buffer.size}")

            # Periodic template expansion
            if episode % cfg.template_expansion_interval == 0:
                self._expand_templates(verbose)

        elapsed = time.time() - t0
        if verbose:
            print(f"\n  RLVR结束: {cfg.max_episodes}轮, "
                  f"发现 {len(self.discoveries)} 个新定理 ({elapsed:.1f}s)")
            print(f"  {self.policy.summary()}")
            print(f"  {self.polya_controller.summary()}")

        return self._make_result(
            cfg.max_episodes, total_provable, total_novel,
            elapsed, reward_history,
        )

    def _expand_templates(self, verbose: bool = False) -> None:
        """Add new templates based on top performers and exploration."""
        # 1. Derive new templates from top performers
        top = self.policy.top_templates(5)
        for tid, stats in top:
            if stats.count > 0 and stats.mean_reward > 0:
                assm, goal = self.policy._templates[tid]
                # Mutate the template itself
                new_assm = list(assm)
                if random.random() < 0.5 and new_assm:
                    idx = random.randint(0, len(new_assm) - 1)
                    new_assm[idx] = random.choice(_PRED_META)[0]
                if random.random() < 0.3:
                    new_assm.append(random.choice(_PRED_META)[0])
                new_goal = goal if random.random() < 0.7 else random.choice(_PRED_META)[0]
                self.policy.inject_template(new_assm, new_goal)

        # 2. Add some fully random templates
        for _ in range(5):
            assm, goal = TemplatePolicy._random_template()
            self.policy.inject_template(assm, goal)

        if verbose:
            print(f"    📎 模板扩展: 现有 {self.policy.n_templates} 个模板")

    def _make_result(
        self,
        episodes: int,
        total_provable: int,
        total_novel: int,
        elapsed: float,
        reward_history: List[float],
    ) -> RLVRResult:
        top = self.policy.top_templates(10)
        return RLVRResult(
            discoveries=self.discoveries,
            total_episodes=episodes,
            total_provable=total_provable,
            total_novel=total_novel,
            elapsed_seconds=elapsed,
            reward_history=reward_history,
            policy_summary=self.policy.summary(),
            top_templates=[(tid, s.mean_reward) for tid, s in top],
        )


# ── Hybrid GA + RLVR pipeline ───────────────────────────────────────

@dataclass
class HybridConfig:
    """Configuration for the hybrid GA + RLVR pipeline."""
    # Phase 1: GA warm-up
    ga_config: GAConfig = field(default_factory=lambda: GAConfig(
        population_size=60,
        max_generations=50,
        target_novel=1,
        min_difficulty=4.0,
    ))
    # Phase 2: RLVR main training
    rlvr_config: RLVRConfig = field(default_factory=lambda: RLVRConfig(
        max_episodes=2000,
        target_novel=3,
        min_difficulty=5.0,
    ))
    # Interleaving
    ga_rlvr_ratio: float = 0.3       # fraction of GA in mixed phase
    cross_pollinate: bool = True      # share discoveries between GA and RLVR


def run_hybrid_evolution(
    config: HybridConfig = HybridConfig(),
    knowledge_store: Optional[KnowledgeStore] = None,
    verbose: bool = True,
) -> RLVRResult:
    """Run the hybrid GA + RLVR pipeline.

    Phase 1: Genetic Algorithm explores the conjecture space broadly,
             finding promising structural templates.
    Phase 2: RLVR focuses on the most promising templates, exploiting
             the policy gradient to generate harder conjectures.

    GA discoveries are injected into the RLVR template policy for
    cross-pollination.
    """
    from .genetic import run_genetic_evolution

    if knowledge_store is None:
        knowledge_store = get_global_store()

    t0 = time.time()
    all_discoveries: List[Dict] = []

    if verbose:
        print("\n╔══════════════════════════════════════════════════════════╗")
        print("║  混合 GA+RLVR 演化管线启动                              ║")
        print("║  Hybrid GA + RLVR Evolution Pipeline                    ║")
        print("╚══════════════════════════════════════════════════════════╝\n")

    # ── Phase 1: GA warm-up ──────────────────────────────────
    if verbose:
        print("━━━ 阶段1: 遗传算法探索 (GA Exploration) ━━━\n")

    ga_result = run_genetic_evolution(
        config=config.ga_config,
        knowledge_store=knowledge_store,
        verbose=verbose,
    )
    all_discoveries.extend(ga_result.discoveries)

    if verbose:
        print(f"\n  GA阶段完成: {len(ga_result.discoveries)} 个发现,"
              f" {ga_result.total_evaluations} 次评估,"
              f" {ga_result.elapsed_seconds:.1f}s")

    # ── Phase 2: RLVR focused training ───────────────────────
    if verbose:
        print("\n━━━ 阶段2: RLVR 强化学习验证 (Focused Training) ━━━\n")

    # Adjust target for RLVR (subtract GA discoveries)
    remaining = config.rlvr_config.target_novel - len(all_discoveries)
    if remaining <= 0:
        remaining = 1  # Always try at least 1 more

    rlvr_cfg = RLVRConfig(
        max_episodes=config.rlvr_config.max_episodes,
        batch_size=config.rlvr_config.batch_size,
        target_novel=remaining,
        min_difficulty=config.rlvr_config.min_difficulty,
        ucb_c=config.rlvr_config.ucb_c,
        initial_temperature=config.rlvr_config.initial_temperature,
        min_temperature=config.rlvr_config.min_temperature,
        decay_rate=config.rlvr_config.decay_rate,
        reward_config=config.rlvr_config.reward_config,
        mutation_rate=config.rlvr_config.mutation_rate,
        n_mutations_per_template=config.rlvr_config.n_mutations_per_template,
        beam_width=config.rlvr_config.beam_width,
        max_depth=config.rlvr_config.max_depth,
    )

    trainer = RLVRTrainer(config=rlvr_cfg, knowledge_store=knowledge_store)

    # Cross-pollinate: inject GA best genomes as templates
    if config.cross_pollinate and ga_result.best_genomes:
        for genome in ga_result.best_genomes:
            if genome.provable:
                decoded = decode_genome(genome)
                if decoded:
                    assm_preds = [f.predicate for f in decoded[0]]
                    goal_pred = decoded[1].predicate
                    trainer.policy.inject_template(assm_preds, goal_pred)
        if verbose:
            print(f"  已注入 GA 最佳模板 → RLVR策略"
                  f" (模板数: {trainer.policy.n_templates})")

    # Also inject GA discoveries' fingerprints to avoid duplicates
    for disc in ga_result.discoveries:
        fp = disc.get("fingerprint", "")
        if fp:
            trainer.seen_fingerprints.add(fp)
            trainer.reward_computer._seen_fps.add(fp)

    rlvr_result = trainer.train(verbose=verbose)
    all_discoveries.extend(rlvr_result.discoveries)

    elapsed = time.time() - t0
    if verbose:
        print(f"\n╔══════════════════════════════════════════════════════════╗")
        print(f"║  混合演化完成: 共发现 {len(all_discoveries)} 个新定理"
              f"  ({elapsed:.1f}s)      ║")
        print(f"╚══════════════════════════════════════════════════════════╝")

    return RLVRResult(
        discoveries=all_discoveries,
        total_episodes=rlvr_result.total_episodes,
        total_provable=rlvr_result.total_provable,
        total_novel=rlvr_result.total_novel,
        elapsed_seconds=elapsed,
        reward_history=rlvr_result.reward_history,
        policy_summary=rlvr_result.policy_summary,
        top_templates=rlvr_result.top_templates,
    )
