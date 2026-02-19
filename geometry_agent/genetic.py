"""genetic.py – Genetic Algorithm for geometry conjecture generation.

Encodes geometry conjectures as *chromosomes* (structured genomes) and
evolves them through crossover, mutation, and selection.  The fitness
function uses the symbolic engine as a verifier: a conjecture is fit if
it is (a) provable, (b) non-trivial, and (c) scores high on difficulty.

Chromosome structure
~~~~~~~~~~~~~~~~~~~~
A chromosome is a ``ConjectureGenome`` containing:

  • assumption_genes : list of (predicate, arity, point_slots)
  • goal_gene        : (predicate, arity, point_slots)
  • point_pool       : shared point names (controls point overlap)
  • chain_depth      : expected proof depth

The GA generates Fact-level conjectures from genomes and evaluates
them via beam search + difficulty evaluator.

Author:  Jiangsheng Yu
License: MIT
"""

from __future__ import annotations

import copy
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from .dsl import Fact, GeoState, Goal, Step
from .difficulty_eval import evaluate_difficulty, DifficultyReport
from .knowledge import KnowledgeStore, get_global_store
from .lean_bridge import MockLeanChecker
from .rules import default_rules
from .search import SearchConfig, SearchResult, beam_search, default_scorer
from .polya import polya_test
from .polya_controller import PolyaController

logger = logging.getLogger(__name__)

# ── Predicate metadata ───────────────────────────────────────────────

# (predicate_name, arity, canonical_fn_name, concept_family, tier)
_PRED_META: List[Tuple[str, int, str, str, int]] = [
    # Tier 1 — LINE
    ("Parallel",      4, "canonical_parallel",      "LINE",       1),
    ("Perpendicular", 4, "canonical_perp",          "LINE",       1),
    ("Collinear",     3, "canonical_collinear",     "LINE",       1),
    ("Between",       3, "canonical_between",       "LINE",       1),
    # Tier 2 — MIDPOINT / ANGLE
    ("Midpoint",      3, "canonical_midpoint",      "MIDPOINT",   2),
    ("AngleBisect",   4, "canonical_angle_bisect",  "ANGLE",      2),
    # Tier 3 — METRIC
    ("Cong",          4, "canonical_cong",           "METRIC",     3),
    ("EqAngle",       6, "canonical_eq_angle",       "ANGLE",      3),
    ("EqDist",        3, "canonical_eqdist",         "METRIC",     3),
    ("EqArea",        6, "canonical_eqarea",         "METRIC",     3),
    ("EqRatio",       8, "canonical_eqratio",        "METRIC",     3),
    # Tier 4 — CIRCLE
    ("Cyclic",        4, "canonical_cyclic",         "CIRCLE",     4),
    ("OnCircle",      2, "canonical_circle",         "CIRCLE",     4),
    ("Circumcenter",  4, "canonical_circumcenter",   "CIRCLE",     4),
    ("Tangent",       4, "canonical_tangent",        "CIRCLE",     4),
    ("RadicalAxis",   4, "canonical_radical_axis",   "CIRCLE",     4),
    # Tier 5 — SIMILARITY
    ("SimTri",        6, "canonical_sim_tri",        "SIMILARITY", 5),
    ("CongTri",       6, "canonical_congtri",        "SIMILARITY", 5),
    ("Concurrent",    6, "canonical_concurrent",     "CONCURRENCY",5),
    # Tier 6 — PROJECTIVE
    ("Harmonic",      4, "canonical_harmonic",       "PROJECTIVE", 6),
    ("PolePolar",     4, "canonical_pole_polar",     "PROJECTIVE", 6),
    ("InvImage",      4, "canonical_inv_image",      "PROJECTIVE", 6),
    ("EqCrossRatio",  8, "canonical_eq_cross_ratio", "PROJECTIVE", 6),
]

_PRED_BY_NAME: Dict[str, Tuple[str, int, str, str, int]] = {
    m[0]: m for m in _PRED_META
}
_PRED_BY_FAMILY: Dict[str, List[str]] = {}
for _m in _PRED_META:
    _PRED_BY_FAMILY.setdefault(_m[3], []).append(_m[0])
_PRED_BY_TIER: Dict[int, List[str]] = {}
for _m in _PRED_META:
    _PRED_BY_TIER.setdefault(_m[4], []).append(_m[0])

POINT_POOL = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ── Known provable bridges (predicate pair → required bridge predicates) ──
# These encode domain knowledge about what assumption combos CAN lead
# to interesting proofs spanning multiple concept families.
_BRIDGE_TEMPLATES: List[Tuple[List[str], str]] = [
    # (assumption_predicates, goal_predicate)
    (["Midpoint", "Midpoint", "Parallel", "Perpendicular"], "Perpendicular"),
    (["Midpoint", "Midpoint", "Cong"],                      "Cong"),
    (["Cyclic", "EqAngle"],                                 "EqAngle"),
    (["Cyclic", "Cong"],                                    "EqAngle"),
    (["Midpoint", "Perpendicular", "Cong"],                 "Cong"),
    (["Midpoint", "Midpoint", "Cyclic"],                    "Parallel"),
    (["Cong", "Midpoint", "Parallel"],                      "Perpendicular"),
    (["Circumcenter", "Midpoint"],                          "Perpendicular"),
    (["Circumcenter", "Cong"],                              "EqAngle"),
    (["AngleBisect", "EqAngle"],                            "EqAngle"),
    (["AngleBisect", "Midpoint"],                           "EqRatio"),
    (["Midpoint", "Midpoint", "EqAngle"],                   "EqAngle"),
    (["Cyclic", "Cong", "EqAngle"],                         "EqAngle"),
    (["Cong", "Midpoint"],                                  "Perpendicular"),
    (["Tangent", "Parallel"],                               "Perpendicular"),
    (["PolePolar", "Parallel"],                             "Perpendicular"),
    (["RadicalAxis", "Parallel"],                           "Perpendicular"),
    (["Midpoint", "Midpoint", "Perpendicular", "Cong"],     "Cong"),
    (["Cyclic", "Midpoint", "Midpoint"],                    "Parallel"),
    (["Circumcenter", "Midpoint", "Parallel"],              "Perpendicular"),
    (["Harmonic", "Harmonic"],                              "EqCrossRatio"),
    (["Cong", "Cong", "Midpoint"],                          "Perpendicular"),
    (["Midpoint", "Midpoint", "Midpoint"],                  "Concurrent"),
    # Deep multi-family bridges
    (["Cyclic", "Cong", "Midpoint"],                        "Perpendicular"),
    (["Cyclic", "Cong", "Midpoint", "Parallel"],            "Perpendicular"),
    (["Circumcenter", "Midpoint", "Cong"],                  "EqAngle"),
    (["Cyclic", "Midpoint", "Perpendicular"],               "Cong"),
    (["Cong", "Cong", "Midpoint", "Parallel"],              "Perpendicular"),
    (["AngleBisect", "Cyclic", "EqAngle"],                  "EqAngle"),
    (["Circumcenter", "Midpoint", "Cyclic"],                "EqAngle"),
]


# ── Genome representation ────────────────────────────────────────────

@dataclass
class PredicateGene:
    """A single predicate in the assumption or goal."""
    predicate: str      # e.g. "Midpoint", "Cyclic"
    point_indices: List[int]  # indices into the genome's point pool

    def clone(self) -> "PredicateGene":
        return PredicateGene(self.predicate, list(self.point_indices))


@dataclass
class ConjectureGenome:
    """Genome encoding a geometry conjecture.

    The genome stores predicate types + point-index references.
    Actual point names are drawn from a shared pool at decode time.
    """
    assumption_genes: List[PredicateGene]
    goal_gene: PredicateGene
    n_points: int = 8         # size of point pool for this genome
    fitness: float = -1.0     # cached fitness
    difficulty_score: float = 0.0
    provable: bool = False
    generation: int = 0
    polya_confidence: float = 0.0  # Pólya plausible-reasoning confidence

    def clone(self) -> "ConjectureGenome":
        return ConjectureGenome(
            assumption_genes=[g.clone() for g in self.assumption_genes],
            goal_gene=self.goal_gene.clone(),
            n_points=self.n_points,
            fitness=self.fitness,
            difficulty_score=self.difficulty_score,
            provable=self.provable,
            generation=self.generation,
        )

    @property
    def family_set(self) -> Set[str]:
        """Concept families touched by this genome."""
        fams = set()
        for g in self.assumption_genes:
            meta = _PRED_BY_NAME.get(g.predicate)
            if meta:
                fams.add(meta[3])
        meta = _PRED_BY_NAME.get(self.goal_gene.predicate)
        if meta:
            fams.add(meta[3])
        return fams

    @property
    def max_tier(self) -> int:
        tiers = []
        for g in self.assumption_genes:
            meta = _PRED_BY_NAME.get(g.predicate)
            if meta:
                tiers.append(meta[4])
        meta = _PRED_BY_NAME.get(self.goal_gene.predicate)
        if meta:
            tiers.append(meta[4])
        return max(tiers) if tiers else 1


# ── Genome construction helpers ──────────────────────────────────────

def _make_gene(pred_name: str, n_points: int) -> PredicateGene:
    """Create a gene with random point indices for a given predicate."""
    meta = _PRED_BY_NAME[pred_name]
    arity = meta[1]
    indices = [random.randint(0, n_points - 1) for _ in range(arity)]
    return PredicateGene(pred_name, indices)


def _random_genome(
    min_assumptions: int = 3,
    max_assumptions: int = 6,
    min_families: int = 3,
    min_tier: int = 2,
) -> ConjectureGenome:
    """Create a random genome biased towards interesting conjectures."""
    n_assm = random.randint(min_assumptions, max_assumptions)
    n_points = random.randint(max(6, n_assm + 2), min(14, n_assm + 8))

    # Pick assumption predicates ensuring diversity
    chosen_preds: List[str] = []
    families_covered: Set[str] = set()
    max_tier_seen = 0

    # Phase 1: seed with bridge template (50% chance)
    if random.random() < 0.5 and _BRIDGE_TEMPLATES:
        template = random.choice(_BRIDGE_TEMPLATES)
        chosen_preds = list(template[0])
        goal_pred = template[1]
        for p in chosen_preds:
            meta = _PRED_BY_NAME.get(p)
            if meta:
                families_covered.add(meta[3])
                max_tier_seen = max(max_tier_seen, meta[4])
    else:
        goal_pred = None

    # Phase 2: fill remaining assumptions
    while len(chosen_preds) < n_assm:
        # Bias towards underrepresented families
        if len(families_covered) < min_families:
            uncovered = [f for f in _PRED_BY_FAMILY if f not in families_covered]
            if uncovered:
                fam = random.choice(uncovered)
                pred = random.choice(_PRED_BY_FAMILY[fam])
            else:
                pred = random.choice(_PRED_META)[0]
        elif max_tier_seen < min_tier:
            # Pick a higher-tier predicate
            high_tiers = [m[0] for m in _PRED_META if m[4] >= min_tier]
            pred = random.choice(high_tiers) if high_tiers else random.choice(_PRED_META)[0]
        else:
            pred = random.choice(_PRED_META)[0]

        chosen_preds.append(pred)
        meta = _PRED_BY_NAME.get(pred)
        if meta:
            families_covered.add(meta[3])
            max_tier_seen = max(max_tier_seen, meta[4])

    # Phase 3: choose goal predicate (if not from template)
    if goal_pred is None:
        # Goal should be from a different family than any single assumption
        assm_families = set()
        for p in chosen_preds:
            m = _PRED_BY_NAME.get(p)
            if m:
                assm_families.add(m[3])
        # Prefer goal from a family that appears in assumptions
        # (provability hint)
        candidates = [m[0] for m in _PRED_META if m[3] in assm_families]
        if not candidates:
            candidates = [m[0] for m in _PRED_META]
        goal_pred = random.choice(candidates)

    # Build genes
    assm_genes = [_make_gene(p, n_points) for p in chosen_preds]
    goal_gene = _make_gene(goal_pred, n_points)

    # Point sharing heuristic: ensure some point overlap between predicates
    # Share 1-2 points between adjacent assumptions
    for i in range(1, len(assm_genes)):
        if random.random() < 0.6:
            # Share a random point from previous gene
            prev = assm_genes[i - 1]
            shared_idx = random.choice(prev.point_indices)
            swap_pos = random.randint(0, len(assm_genes[i].point_indices) - 1)
            assm_genes[i].point_indices[swap_pos] = shared_idx

    # Share points with goal
    if assm_genes and random.random() < 0.7:
        donor = random.choice(assm_genes)
        shared_idx = random.choice(donor.point_indices)
        swap_pos = random.randint(0, len(goal_gene.point_indices) - 1)
        goal_gene.point_indices[swap_pos] = shared_idx

    return ConjectureGenome(
        assumption_genes=assm_genes,
        goal_gene=goal_gene,
        n_points=n_points,
    )


def _genome_from_template(
    assm_preds: List[str],
    goal_pred: str,
    n_points: int = 10,
) -> ConjectureGenome:
    """Create a genome from a bridge template with smart point allocation."""
    assm_genes = []
    used_indices: List[int] = []

    for pred in assm_preds:
        gene = _make_gene(pred, n_points)
        # Share some points with earlier genes
        if used_indices and random.random() < 0.65:
            shared = random.choice(used_indices)
            pos = random.randint(0, len(gene.point_indices) - 1)
            gene.point_indices[pos] = shared
        used_indices.extend(gene.point_indices)
        assm_genes.append(gene)

    goal_gene = _make_gene(goal_pred, n_points)
    # Goal usually shares points with assumptions
    if used_indices:
        for pos in range(len(goal_gene.point_indices)):
            if random.random() < 0.5:
                goal_gene.point_indices[pos] = random.choice(used_indices)

    return ConjectureGenome(
        assumption_genes=assm_genes,
        goal_gene=goal_gene,
        n_points=n_points,
    )


def _genome_from_deep_generator() -> Optional[ConjectureGenome]:
    """Create a genome from a random DEEP_GENERATOR result.

    This produces a genome that, when decoded, yields the EXACT same
    (assumptions, goal) the generator produced — guaranteeing a provable
    starting point.  The GA can then mutate/crossover from here to
    explore the neighbourhood of known-good conjectures.
    """
    from .conjecture import DEEP_GENERATORS
    gen_name, gen_fn = random.choice(DEEP_GENERATORS)
    try:
        assumptions, goal = gen_fn()
    except (ValueError, IndexError):
        return None

    # Build deterministic point ↔ index mapping via POINT_POOL
    all_points: Set[str] = set()
    for fact in assumptions:
        all_points.update(fact.args)
    all_points.update(goal.args)

    # Map each point to its position in POINT_POOL
    point_index: Dict[str, int] = {}
    for pt in all_points:
        try:
            idx = POINT_POOL.index(pt)
        except ValueError:
            return None  # point name not in pool (shouldn't happen)
        point_index[pt] = idx
    n_points = max(point_index.values()) + 1 if point_index else 10

    # Encode assumptions as genes
    assm_genes: List[PredicateGene] = []
    for fact in assumptions:
        indices = [point_index[p] for p in fact.args]
        assm_genes.append(PredicateGene(
            predicate=fact.predicate,
            point_indices=indices,
        ))

    # Encode goal
    goal_indices = [point_index[p] for p in goal.args]
    goal_gene = PredicateGene(
        predicate=goal.predicate,
        point_indices=goal_indices,
    )

    return ConjectureGenome(
        assumption_genes=assm_genes,
        goal_gene=goal_gene,
        n_points=n_points,
    )


# ── Decode genome → (assumptions, goal) facts ───────────────────────

def _decode_gene(gene: PredicateGene, point_names: List[str]) -> Optional[Fact]:
    """Convert a gene to a Fact using point names."""
    from . import dsl
    try:
        args = [point_names[i % len(point_names)] for i in gene.point_indices]
        # Check for degenerate cases: all points same
        if len(set(args)) < 2:
            return None
        # Use canonical constructor
        fn_name = _PRED_BY_NAME[gene.predicate][2]
        fn = getattr(dsl, fn_name)
        return fn(*args)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def decode_genome(genome: ConjectureGenome) -> Optional[Tuple[List[Fact], Fact]]:
    """Decode a genome into (assumptions, goal) suitable for beam search."""
    point_names = POINT_POOL[:genome.n_points]
    # NOTE: we do NOT shuffle — indices are meaningful and must produce
    # the same facts deterministically (especially for generator-seeded
    # genomes).  Variety comes from mutation and crossover.

    assumptions = []
    for gene in genome.assumption_genes:
        fact = _decode_gene(gene, point_names)
        if fact is not None:
            assumptions.append(fact)

    if len(assumptions) < 2:
        return None

    goal = _decode_gene(genome.goal_gene, point_names)
    if goal is None:
        return None

    # Skip if goal is already in assumptions
    if goal in set(assumptions):
        return None

    return assumptions, goal


# ── Genetic operators ────────────────────────────────────────────────

def crossover(parent1: ConjectureGenome, parent2: ConjectureGenome) -> ConjectureGenome:
    """Single-point crossover on assumption genes + goal from one parent."""
    g1 = parent1.clone()
    g2 = parent2.clone()

    # Mix assumptions: take some from each parent
    all_genes = g1.assumption_genes + g2.assumption_genes
    random.shuffle(all_genes)
    n_take = random.randint(
        min(3, len(all_genes)),
        min(6, len(all_genes)),
    )
    child_genes = all_genes[:n_take]

    # Goal from fitter parent (or random)
    if parent1.fitness >= parent2.fitness:
        child_goal = g1.goal_gene
    else:
        child_goal = g2.goal_gene

    n_points = max(g1.n_points, g2.n_points)

    return ConjectureGenome(
        assumption_genes=child_genes,
        goal_gene=child_goal,
        n_points=n_points,
    )


def mutate(genome: ConjectureGenome, rate: float = 0.3) -> ConjectureGenome:
    """Mutation operators applied with given probability."""
    g = genome.clone()

    # Mutation 1: change a predicate type
    if random.random() < rate and g.assumption_genes:
        idx = random.randint(0, len(g.assumption_genes) - 1)
        new_pred = random.choice(_PRED_META)[0]
        g.assumption_genes[idx] = _make_gene(new_pred, g.n_points)

    # Mutation 2: swap point indices
    if random.random() < rate and g.assumption_genes:
        idx = random.randint(0, len(g.assumption_genes) - 1)
        gene = g.assumption_genes[idx]
        if len(gene.point_indices) >= 2:
            i, j = random.sample(range(len(gene.point_indices)), 2)
            gene.point_indices[i], gene.point_indices[j] = \
                gene.point_indices[j], gene.point_indices[i]

    # Mutation 3: add an assumption
    if random.random() < rate * 0.5 and len(g.assumption_genes) < 7:
        new_pred = random.choice(_PRED_META)[0]
        g.assumption_genes.append(_make_gene(new_pred, g.n_points))

    # Mutation 4: remove an assumption
    if random.random() < rate * 0.3 and len(g.assumption_genes) > 3:
        idx = random.randint(0, len(g.assumption_genes) - 1)
        g.assumption_genes.pop(idx)

    # Mutation 5: change goal predicate
    if random.random() < rate * 0.4:
        new_pred = random.choice(_PRED_META)[0]
        g.goal_gene = _make_gene(new_pred, g.n_points)

    # Mutation 6: increase point sharing (provability boost)
    if random.random() < rate * 0.6 and len(g.assumption_genes) >= 2:
        src = random.choice(g.assumption_genes)
        dst = random.choice(g.assumption_genes)
        if src is not dst:
            shared = random.choice(src.point_indices)
            pos = random.randint(0, len(dst.point_indices) - 1)
            dst.point_indices[pos] = shared

    # Mutation 7: adjust n_points
    if random.random() < rate * 0.2:
        delta = random.choice([-1, 1])
        g.n_points = max(6, min(16, g.n_points + delta))

    g.fitness = -1.0  # invalidate cache
    return g


def tournament_select(
    population: List[ConjectureGenome],
    k: int = 3,
) -> ConjectureGenome:
    """Tournament selection: pick k random individuals, return fittest."""
    candidates = random.sample(population, min(k, len(population)))
    return max(candidates, key=lambda g: g.fitness)


# ── Fitness evaluation ───────────────────────────────────────────────

def evaluate_fitness(
    genome: ConjectureGenome,
    rules: Any = None,
    checker: Any = None,
    knowledge_store: Optional[KnowledgeStore] = None,
    seen_fingerprints: Optional[Set[str]] = None,
    polya_controller: Optional[PolyaController] = None,
) -> Tuple[float, Optional[SearchResult], Optional[DifficultyReport]]:
    """Evaluate conjecture fitness.

    Returns (fitness, search_result, difficulty_report).
    Fitness breakdown:
      - Not decodable:  -2.0
      - Not provable:   -1.0  (penalised but not eliminated)
      - Provable trivial: 0.0
      - Provable novel:  difficulty_score (1-10)
    """
    if rules is None:
        rules = default_rules()
    if checker is None:
        checker = MockLeanChecker()
    if knowledge_store is None:
        knowledge_store = get_global_store()
    if polya_controller is None:
        polya_controller = PolyaController(knowledge_store=knowledge_store)

    # Decode
    decoded = decode_genome(genome)
    if decoded is None:
        genome.fitness = -2.0
        genome.provable = False
        return -2.0, None, None

    assumptions, goal = decoded

    # ── Pólya Step 1+2: understand+plan ──
    plan = polya_controller.make_plan(
        assumptions,
        goal,
        strategy="ga:evaluate_fitness",
    )

    # ── Pólya plausible-reasoning pre-filter ──
    polya_res = polya_test(assumptions, goal, n_trials=plan.polya_trials)
    if polya_res.falsified or polya_res.confidence < plan.polya_min_confidence:
        genome.fitness = -1.5
        genome.provable = False
        polya_controller.note_failure("ga_polya_reject")
        return -1.5, None, None

    # Solve: staged search with adaptive budgets
    state = GeoState(facts=set(assumptions))
    fast_cfg = SearchConfig(
        beam_width=max(96, min(plan.fast_beam_width, 120)),
        max_depth=max(22, min(plan.fast_max_depth, 26)),
        parallel_workers=0,
    )
    result = beam_search(
        init_state=state,
        goal=Goal(goal),
        rules=rules,
        checker=checker,
        config=fast_cfg,
        knowledge_store=knowledge_store,
    )

    if (not result.success) and polya_controller.should_escalate(
        polya_res.confidence,
        strategy="ga:evaluate_fitness",
    ):
        deep_cfg = SearchConfig(
            beam_width=max(128, min(plan.deep_beam_width, 180)),
            max_depth=max(24, min(plan.deep_max_depth, 30)),
            parallel_workers=0,
        )
        result = beam_search(
            init_state=state,
            goal=Goal(goal),
            rules=rules,
            checker=checker,
            config=deep_cfg,
            knowledge_store=knowledge_store,
        )

    if not result.success:
        # Not provable — but Pólya-plausible: give partial credit
        # proportional to confidence so GA can explore further.
        genome.fitness = -1.0
        genome.provable = False
        fam_bonus = len(genome.family_set) * 0.05
        tier_bonus = genome.max_tier * 0.02
        polya_bonus = polya_res.confidence * 0.3  # up to +0.3 for high confidence
        genome.fitness = -1.0 + fam_bonus + tier_bonus + polya_bonus
        genome.polya_confidence = polya_res.confidence
        polya_controller.note_failure("ga_search_fail")
        return genome.fitness, None, None

    steps = list(result.final_state.history)
    genome.provable = True

    # Trivial check
    if len(steps) < 3:
        genome.fitness = 0.0
        return 0.0, result, None

    # Difficulty evaluation
    diff_report = evaluate_difficulty(assumptions, goal, steps)
    genome.difficulty_score = diff_report.overall_score

    # Fingerprint dedup bonus
    from .semantic import semantic_theorem_fingerprint
    fp = semantic_theorem_fingerprint(assumptions, goal)
    novelty_bonus = 0.0
    if seen_fingerprints is not None:
        if fp not in seen_fingerprints:
            novelty_bonus = 1.0  # reward novel structures
        else:
            novelty_bonus = -0.5  # penalise duplicates

    # Fitness = difficulty_score + novelty + diversity bonuses
    fitness = diff_report.overall_score + novelty_bonus
    # Bonus for crossing many families
    fitness += len(genome.family_set) * 0.3
    # Bonus for using high-tier predicates
    fitness += genome.max_tier * 0.2
    # Bonus for longer proofs (more knowledge applied)
    n_distinct = len({s.rule_name for s in steps})
    fitness += min(n_distinct * 0.2, 2.0)

    genome.fitness = fitness
    polya_controller.note_success("ga:evaluate_fitness")
    return fitness, result, diff_report


# ── Main GA loop ─────────────────────────────────────────────────────

@dataclass
class GAConfig:
    """Configuration for the genetic algorithm."""
    population_size: int = 80
    elite_count: int = 8
    tournament_k: int = 4
    crossover_rate: float = 0.7
    mutation_rate: float = 0.35
    max_generations: int = 150
    min_difficulty: float = 5.0
    target_novel: int = 3
    # Initialisation
    template_ratio: float = 0.4     # fraction seeded from bridge templates
    min_assumptions: int = 3
    max_assumptions: int = 6
    min_families: int = 3
    min_tier: int = 3


@dataclass
class GAResult:
    """Summary of GA evolution run."""
    best_genomes: List[ConjectureGenome]
    discoveries: List[Dict]
    generations_run: int
    total_evaluations: int
    elapsed_seconds: float
    fitness_history: List[float] = field(default_factory=list)


def run_genetic_evolution(
    config: GAConfig = GAConfig(),
    knowledge_store: Optional[KnowledgeStore] = None,
    verbose: bool = True,
) -> GAResult:
    """Run the genetic algorithm to discover novel geometry conjectures.

    The GA maintains a population of conjecture genomes and evolves them
    through crossover, mutation, and selection.  Fitness is determined by
    the symbolic engine (provability) and difficulty evaluator.
    """
    if knowledge_store is None:
        knowledge_store = get_global_store()

    rules = default_rules()
    checker = MockLeanChecker()
    polya_controller = PolyaController(knowledge_store=knowledge_store)
    seen_fingerprints: Set[str] = set()
    discoveries: List[Dict] = []
    fitness_history: List[float] = []
    total_evals = 0
    t0 = time.time()

    if verbose:
        print("\n╔══════════════════════════════════════════════════════════╗")
        print("║  遗传算法猜想生成启动 / Genetic Algorithm Started        ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"  种群大小: {config.population_size}  精英保留: {config.elite_count}")
        print(f"  交叉率: {config.crossover_rate}  变异率: {config.mutation_rate}")
        print(f"  目标: {config.target_novel} 个难度≥{config.min_difficulty} 的新定理")
        print()

    # ── Initialise population ────────────────────────────────
    population: List[ConjectureGenome] = []

    # Seed from deep generators (provable starting points)
    n_deep_seed = int(config.population_size * 0.5)
    for _ in range(n_deep_seed):
        g = _genome_from_deep_generator()
        if g is not None:
            population.append(g)

    # Seed from bridge templates
    n_template = int(config.population_size * config.template_ratio)
    for _ in range(n_template):
        if len(population) >= config.population_size:
            break
        tmpl = random.choice(_BRIDGE_TEMPLATES)
        g = _genome_from_template(tmpl[0], tmpl[1])
        population.append(g)

    # Fill remainder with random genomes
    while len(population) < config.population_size:
        g = _random_genome(
            min_assumptions=config.min_assumptions,
            max_assumptions=config.max_assumptions,
            min_families=config.min_families,
            min_tier=config.min_tier,
        )
        population.append(g)

    # ── Evolution loop ───────────────────────────────────────
    for gen in range(1, config.max_generations + 1):
        # Evaluate fitness for unevaluated genomes
        for genome in population:
            if genome.fitness < -1.5:  # needs evaluation
                fitness, result, diff_report = evaluate_fitness(
                    genome,
                    rules,
                    checker,
                    knowledge_store,
                    seen_fingerprints,
                    polya_controller,
                )
                total_evals += 1

                # Check for discovery
                if (result is not None
                    and diff_report is not None
                    and diff_report.overall_score >= config.min_difficulty):
                    decoded = decode_genome(genome)
                    if decoded:
                        assumptions, goal = decoded
                        from .semantic import semantic_theorem_fingerprint
                        fp = semantic_theorem_fingerprint(assumptions, goal)
                        if fp not in seen_fingerprints:
                            seen_fingerprints.add(fp)
                            steps = list(result.final_state.history)
                            discoveries.append({
                                "genome": genome.clone(),
                                "assumptions": assumptions,
                                "goal": goal,
                                "steps": steps,
                                "difficulty": diff_report,
                                "generation": gen,
                                "fingerprint": fp,
                            })
                            if verbose:
                                print(f"  🌟 GA发现#{len(discoveries)}: "
                                      f"难度 {diff_report.overall_score:.1f}/10"
                                      f" ({diff_report.label_zh})"
                                      f"  族={len(genome.family_set)}"
                                      f"  层级={genome.max_tier}"
                                      f"  步={len(steps)}")

                            if len(discoveries) >= config.target_novel:
                                elapsed = time.time() - t0
                                if verbose:
                                    print(f"\n  ✅ GA目标达成! 第{gen}代发现"
                                          f" {len(discoveries)} 个新定理"
                                          f" ({elapsed:.1f}s)")
                                return GAResult(
                                    best_genomes=sorted(
                                        population,
                                        key=lambda g: g.fitness,
                                        reverse=True,
                                    )[:10],
                                    discoveries=discoveries,
                                    generations_run=gen,
                                    total_evaluations=total_evals,
                                    elapsed_seconds=elapsed,
                                    fitness_history=fitness_history,
                                )

        # Sort by fitness
        population.sort(key=lambda g: g.fitness, reverse=True)
        best_fitness = population[0].fitness if population else 0
        avg_fitness = sum(g.fitness for g in population) / max(len(population), 1)
        fitness_history.append(best_fitness)

        provable_count = sum(1 for g in population if g.provable)
        high_diff = sum(1 for g in population if g.difficulty_score >= 3.0)

        if verbose and gen % 5 == 0:
            print(f"  代{gen:3d} | 最佳适应度 {best_fitness:6.2f}"
                  f"  平均 {avg_fitness:5.2f}"
                  f"  可证 {provable_count}/{len(population)}"
                  f"  难度≥3: {high_diff}"
                  f"  发现 {len(discoveries)}/{config.target_novel}")

        # ── Elitism ──────────────────────────────────────────
        next_gen: List[ConjectureGenome] = []
        elite = population[:config.elite_count]
        for e in elite:
            next_gen.append(e.clone())

        # ── Crossover + Mutation ─────────────────────────────
        while len(next_gen) < config.population_size:
            if random.random() < config.crossover_rate:
                p1 = tournament_select(population, config.tournament_k)
                p2 = tournament_select(population, config.tournament_k)
                child = crossover(p1, p2)
            else:
                parent = tournament_select(population, config.tournament_k)
                child = parent.clone()

            child = mutate(child, config.mutation_rate)
            child.generation = gen
            next_gen.append(child)

        # ── Immigration: inject fresh genomes (10%) ─────────────
        # Prefer deep-generator-seeded genomes for higher provability.
        n_immigrants = max(2, config.population_size // 10)
        for i in range(n_immigrants):
            idx = len(next_gen) - 1 - i
            if idx >= config.elite_count:
                # 60% deep generators, 20% bridge templates, 20% random
                r = random.random()
                if r < 0.6:
                    g = _genome_from_deep_generator()
                    if g is not None:
                        next_gen[idx] = g
                        continue
                if r < 0.8 and _BRIDGE_TEMPLATES:
                    tmpl = random.choice(_BRIDGE_TEMPLATES)
                    next_gen[idx] = _genome_from_template(tmpl[0], tmpl[1])
                else:
                    next_gen[idx] = _random_genome(
                        min_families=config.min_families,
                        min_tier=config.min_tier,
                    )

        population = next_gen[:config.population_size]

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  GA结束: {config.max_generations}代, "
              f"发现 {len(discoveries)} 个新定理 ({elapsed:.1f}s)")
        print(f"  {polya_controller.summary()}")

    return GAResult(
        best_genomes=sorted(population, key=lambda g: g.fitness, reverse=True)[:10],
        discoveries=discoveries,
        generations_run=config.max_generations,
        total_evaluations=total_evals,
        elapsed_seconds=elapsed,
        fitness_history=fitness_history,
    )
