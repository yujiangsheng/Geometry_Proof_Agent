# Usage Guide

Detailed usage examples and recipes for the Geometry Proof Agent v0.14.0.

---

## Table of Contents

- [Basic Usage](#basic-usage)
- [Programmatic API](#programmatic-api)
- [Knowledge-Guided Search](#knowledge-guided-search)
- [Theorem Discovery](#theorem-discovery)
- [Knowledge Guidance API](#knowledge-guidance-api)
- [Lean 4 Verification](#lean-4-verification)
- [LLM Integration](#llm-integration)
- [Adding New Rules](#adding-new-rules)
- [Reading the Output](#reading-the-output)
- [Advanced Recipes](#advanced-recipes)
- [Troubleshooting](#troubleshooting)

---

## Basic Usage

### Running the Demo

```bash
python -m geometry_agent.main
```

This runs two built-in demonstrations:

1. **Parallel transitivity**: AB ∥ CD, CD ∥ EF ⊢ AB ∥ EF
2. **Parallel-perpendicular transfer**: AB ∥ CD, CD ⊥ EF ⊢ AB ⊥ EF

Each demo shows:
- Natural language statement (中文)
- Layered architecture info (engine, verifier, LLM)
- Proof steps with rule names
- Lean 4 formal statement
- Verification status

### CLI Flags

```bash
python -m geometry_agent.main --lean      # Real Lean 4 verification
python -m geometry_agent.main --llm       # LLM narration (requires Ollama)
python -m geometry_agent.main -v          # Verbose logging (debug level)
python -m geometry_agent.main --hybrid    # Hybrid evolution (recommended)
```

---

## Programmatic API

### Solving a Custom Problem

```python
from geometry_agent.dsl import canonical_parallel, canonical_perp, GeoState, Goal
from geometry_agent.rules import default_rules
from geometry_agent.search import SearchConfig, beam_search
from geometry_agent.lean_bridge import MockLeanChecker
from geometry_agent.semantic import theorem_to_nl, proof_to_nl

# Define the problem: AB ∥ CD, CD ⊥ EF ⊢ AB ⊥ EF
assumptions = [
    canonical_parallel("A", "B", "C", "D"),
    canonical_perp("C", "D", "E", "F"),
]
goal = canonical_perp("A", "B", "E", "F")

# Solve with beam search
state = GeoState(facts=set(assumptions))
config = SearchConfig(beam_width=16, max_depth=10)
result = beam_search(
    init_state=state,
    goal=Goal(goal),
    rules=default_rules(),
    checker=MockLeanChecker(),
    config=config,
)

if result.success:
    print("Proved!")
    # Print proof steps
    for i, step in enumerate(result.final_state.history, 1):
        print(f"  {i}. {step.rule_name}: {step.conclusion_fact}")

    # Natural language proof
    nl = proof_to_nl(assumptions, result.final_state.history, goal, lang="zh")
    print(nl)
```

### Working with Facts

```python
from geometry_agent.dsl import (
    Fact,
    canonical_parallel,
    canonical_perp,
    canonical_midpoint,
    canonical_cong,
    canonical_eq_angle,
    canonical_circumcenter,
    canonical_cyclic,
)

# Create facts using canonical constructors (ensure consistent argument order)
par = canonical_parallel("A", "B", "C", "D")   # Parallel(A,B,C,D)
perp = canonical_perp("E", "F", "G", "H")      # Perpendicular(E,F,G,H)
mid = canonical_midpoint("M", "A", "B")         # Midpoint(M,A,B)
cong = canonical_cong("A", "B", "C", "D")       # Cong(A,B,C,D)
cc = canonical_circumcenter("O", "A", "B", "C") # Circumcenter(O,A,B,C)

# Or create directly
f = Fact(predicate="EqAngle", args=("A", "B", "C", "D", "E", "F"))

# Facts are frozen dataclasses — hashable and comparable
assert par == canonical_parallel("A", "B", "C", "D")
my_set = {par, perp, mid}
```

### Using the GeoState

```python
from geometry_agent.dsl import GeoState, Fact

state = GeoState(facts={
    Fact("Parallel", ("A", "B", "C", "D")),
    Fact("Perpendicular", ("C", "D", "E", "F")),
})

# Predicate-indexed lookup: O(1) instead of scanning all facts
parallels = state.by_predicate("Parallel")  # Set[Fact]
print(f"Parallel facts: {parallels}")

# Add a derived fact
new_fact = Fact("Perpendicular", ("A", "B", "E", "F"))
state.add_fact(new_fact)
```

### Generating NL and Lean4

```python
from geometry_agent.dsl import canonical_parallel, canonical_perp
from geometry_agent.semantic import (
    fact_to_nl,
    theorem_to_nl,
    theorem_to_lean,
    semantic_theorem_fingerprint,
    structural_theorem_fingerprint,
)

assumptions = [canonical_parallel("A", "B", "C", "D")]
goal = canonical_perp("A", "B", "E", "F")

# Natural language (中文 / English)
print(theorem_to_nl(assumptions, goal, lang="zh"))
# → 已知 直线 AB 平行于直线 CD，求证 直线 AB 垂直于直线 EF。

print(theorem_to_nl(assumptions, goal, lang="en"))
# → Given: line AB is parallel to line CD. Prove: line AB is perpendicular to line EF.

# Lean 4 formal statement
lean_code = theorem_to_lean(assumptions, goal, name="my_theorem")
print(lean_code)

# Fingerprinting (isomorphism-invariant)
fp = semantic_theorem_fingerprint(assumptions, goal)
sfp = structural_theorem_fingerprint(assumptions, goal)
```

---

## Knowledge-Guided Search

v0.12.0 introduces a knowledge-guided beam search that leverages
accumulated experience to improve search efficiency.

### Using Knowledge-Guided Scorer

```python
from geometry_agent.dsl import canonical_parallel, canonical_perp, GeoState, Goal
from geometry_agent.rules import default_rules
from geometry_agent.search import SearchConfig, beam_search, make_knowledge_scorer
from geometry_agent.lean_bridge import MockLeanChecker
from geometry_agent.knowledge import get_global_store

# Load the knowledge store (auto-loads from data/knowledge/)
store = get_global_store()

# Define problem
assumptions = [canonical_parallel("A", "B", "C", "D")]
goal = canonical_perp("A", "B", "E", "F")

# Create a knowledge-guided scorer
# This scorer adds:
#   +15 pts for rules with high success rates in past experience
#   +5 pts for rules appearing in known proof chains
scorer = make_knowledge_scorer(store, goal)

# Solve with knowledge-guided search
state = GeoState(facts=set(assumptions))
config = SearchConfig(beam_width=16, max_depth=10)
result = beam_search(
    init_state=state,
    goal=Goal(goal),
    rules=default_rules(),
    checker=MockLeanChecker(),
    config=config,
    knowledge_store=store,  # enables cache shortcuts + rule ordering
    scorer=scorer,           # experience-weighted scoring
)

if result.success:
    print(f"Proved in {result.explored_nodes} nodes, {result.cache_hits} cache hits")
```

### Automatic Knowledge-Guided Upgrade

When you pass a `knowledge_store` to `beam_search()`, the scorer is
automatically upgraded to include knowledge bonuses — no need to manually
create the scorer:

```python
# Simplified: just pass the knowledge store
result = beam_search(
    init_state=state,
    goal=Goal(goal),
    rules=default_rules(),
    checker=MockLeanChecker(),
    config=config,
    knowledge_store=store,  # scorer auto-upgraded with experience bonuses
)
```

---

## Theorem Discovery

### Quick Start: Hybrid Evolution

The recommended way to discover new theorems:

```bash
python run_evolve.py
```

Or with more control:

```bash
python -m geometry_agent.main --hybrid
```

### Quick Start: Fast Discovery Profile

For high-throughput theorem hunting, use short multi-seed rounds:

```bash
# Fast profile: multi-round heuristic discovery + cross-round dedup
python run_evolve.py --profile fast_discover --target-novel 8 --rounds 6 --per-round-target 3

# Keep existing HTML results (do not reset output/new_theorems.html)
python run_evolve.py --profile fast_discover --no-reset-html

# Reproducible run with custom seed
python run_evolve.py --profile fast_discover --base-seed 20260218
```

`run_evolve.py` profile behavior:
- `standard`: single run using `--mode hybrid|heuristic|ga|rlvr`
- `fast_discover`: multiple short heuristic runs, semantic dedup across rounds, early stop at `--target-novel`

### Pólya Four-Step Controller (v0.12.0+, enhanced v0.13.0)

Heuristic conjecture search now uses a lightweight Pólya-style control loop:

1. **Understand**: profile conjecture complexity from assumptions/goal predicates
2. **Plan**: adapt `polya_test` trials and fast/deep beam budgets per conjecture;
   boost `premise_probe_trials` to 30 for Cyclic or multi-Perp conjecture sets
3. **Carry out**: run staged search with plan-specific thresholds
4. **Look back**: record failure reasons and success counts for diagnostics

You can observe this in verbose logs as a `Pólya控制器` summary at the end
of heuristic / GA / RLVR runs. In `hybrid` mode, `evolve_hybrid()` also
prints phase-level Pólya scheduling decisions for GA and RLVR.

This runs three phases sequentially:
1. **Heuristic search** — bridge composition, backward chaining, deep generators, MCTS
2. **Genetic Algorithm** — population-based conjecture evolution
3. **RLVR** — reinforcement learning with symbolic verification as reward

### Programmatic Evolution

```python
from geometry_agent.evolve import evolve_hybrid

discoveries, conjectures = evolve_hybrid(
    target_novel=5,          # Find 5 novel theorems
    min_difficulty=5.0,      # Minimum difficulty 5.0/10
    use_lean=False,          # True to verify with real Lean 4
    use_llm=False,           # True for LLM narration
    verbose=True,            # Print progress
    mode="hybrid",           # "hybrid" | "ga" | "rlvr" | "heuristic"
)

for d in discoveries:
    print(f"Theorem: {d.nl_statement}")
    print(f"Difficulty: {d.difficulty_score:.1f}/10 ({d.difficulty_label_zh})")
    print(f"Proof ({d.n_steps} steps): {d.rule_types_used}")
    print(f"Lean code:\n{d.lean_code}")
    print()

# Unproven conjectures (Pólya-plausible but not yet symbolically proved)
for c in conjectures:
    print(f"Conjecture (confidence: {c.polya_confidence*100:.0f}%)")
    print(f"  Pólya: {c.polya_n_passed}/{c.polya_n_trials} trials passed")
```

### Classic Evolution Loop

```python
from geometry_agent.evolve import evolve

discoveries = evolve(
    max_generations=200,     # Maximum evolution rounds
    problems_per_gen=60,     # Problems per round
    min_steps=5,             # Minimum proof steps
    min_predicates=3,        # Minimum predicate types
    min_difficulty=5.0,      # Minimum difficulty score
    target_novel=3,          # Stop after N novel theorems
    use_lean=False,
    verbose=True,
)
```

### Heuristic Conjecture Search

```python
from geometry_agent.conjecture import generate_heuristic_conjectures, HeuristicConfig

config = HeuristicConfig(
    total_attempts=600,
    min_difficulty=4.0,
    target_novel=10,
    mcts_iterations=300,
    bridge_composition_weight=0.4,   # 40% budget on bridge composition
    backward_chaining_weight=0.3,    # 30% on backward chaining
    deep_generator_weight=0.3,       # 30% on deep generators
)

# Knowledge store is automatically used for experience-guided bridge selection
results = generate_heuristic_conjectures(config=config, verbose=True)
for r in results:
    print(f"Strategy: {r['strategy']}, Difficulty: {r['difficulty'].overall_score:.1f}")
```

### Genetic Algorithm

```python
from geometry_agent.genetic import run_genetic_evolution, GAConfig

config = GAConfig(
    population_size=100,
    max_generations=120,
    target_novel=5,
    min_difficulty=4.5,
    min_families=2,
    min_tier=2,
    mutation_rate=0.3,
    crossover_rate=0.7,
    elite_count=8,
    tournament_k=4,
)

result = run_genetic_evolution(config=config, verbose=True)
print(f"Discovered: {len(result.discoveries)} theorems")
```

### RLVR (Reinforcement Learning with Verifiable Rewards)

```python
from geometry_agent.rlvr import RLVRTrainer, RLVRConfig

config = RLVRConfig(
    max_episodes=1500,
    batch_size=15,
    min_difficulty=4.5,
    target_novel=5,
    ucb_c=2.0,
    initial_temperature=1.2,
    decay_rate=0.997,
    beam_width=32,
    max_depth=18,
)

trainer = RLVRTrainer(config=config)
result = trainer.train(verbose=True)
print(f"Discovered: {len(result.discoveries)} theorems")
```

---

## Knowledge Guidance API

v0.12.0 provides a rich guidance API that extracts actionable insights
from accumulated experience.

### Inspecting the Guidance Summary

```python
from geometry_agent.knowledge import get_global_store

store = get_global_store()

# Full guidance summary: rule success rates, under-explored predicates,
# difficulty solve rates, generator stats
summary = store.guidance_summary()
print(summary)
```

Example output:
```
── Rule Success Profile (top 10) ──
  medians_concurrent        : 95.0%
  cyclic_from_eq_angle      : 94.0%
  circumcenter_on_circle    : 92.0%
  ...

── Under-Explored Predicates ──
  Between, Harmonic, AngleBisect

── Difficulty Solve Rates ──
  Level 1: 64% | Level 2: 61% | Level 3: 42% | Level 4: 38%
```

### Individual Guidance Methods

```python
store = get_global_store()

# Per-rule success rates
profile = store.rule_success_profile()
for rule, rate in sorted(profile.items(), key=lambda x: -x[1])[:5]:
    print(f"  {rule}: {rate*100:.0f}%")

# Suggested rule ordering for a specific goal predicate
ordered = store.suggest_rule_order("Perpendicular")
print(f"Best rules for Perpendicular: {ordered[:5]}")

# Predicate coverage (how well each predicate is explored)
coverage = store.predicate_coverage()
for pred, pct in coverage.items():
    print(f"  {pred}: {pct*100:.0f}%")

# Under-explored predicates (< 5 success occurrences)
under = store.under_explored_predicates()
print(f"Under-explored: {under}")

# Proven rule chains (frequently occurring patterns)
chains = store.proven_rule_chains()
for chain, count in chains[:5]:
    print(f"  {chain}: {count}×")

# Difficulty profile (solve rate by level)
diff = store.difficulty_profile()
for level, rate in sorted(diff.items()):
    print(f"  Level {level}: {rate*100:.0f}%")

# Cross-session structural dedup check
is_dup = store.structural_dedup_check(assumptions, goal)
print(f"Duplicate: {is_dup}")
```

### Connecting Knowledge to Evolution

Evolution automatically uses knowledge guidance when a store is available:

```python
from geometry_agent.evolve import evolve_hybrid

# The knowledge store is automatically loaded and used for:
# - Adaptive difficulty (escalate if >80% solve rate, reduce if <20%)
# - Weighted generator selection (2.5x for under-explored predicates)
# - Cross-session structural dedup
# - Knowledge auto-save with guidance summary after each generation
discoveries, conjectures = evolve_hybrid(
    target_novel=3,
    min_difficulty=5.0,
    verbose=True,
)
```

---

## Lean 4 Verification

### Building the Lean Project

```bash
cd lean_geo
lake build
```

This compiles:
- `LeanGeo/Defs.lean` — Type definitions (Point, Line, Circle, predicates)
- `LeanGeo/Basic.lean` — Basic lemmas
- `LeanGeo/Rules.lean` — Lean axioms for legacy-complete rule coverage (newer Python rules may use fallback mapping)

### Running with Lean Verification

```bash
python -m geometry_agent.main --lean --hybrid
```

When `--lean` is enabled, each discovered theorem's Lean 4 code is
compiled against the `lean_geo` project to independently verify correctness.

### Programmatic Lean Bridge

```python
from geometry_agent.lean_bridge import ProcessLeanChecker

checker = ProcessLeanChecker(lean_project_dir="lean_geo")
result = checker.check_source("""
import LeanGeo

theorem my_thm (A B C D E F : Point)
    (h1 : Parallel A B C D)
    (h2 : Parallel C D E F) :
    Parallel A B E F := parallel_trans A B C D E F h1 h2
""")
print(f"OK: {result.ok}, Message: {result.message}")
```

---

## LLM Integration

### Model Auto-Detection

The system automatically detects the best available Ollama model:

```
Preference chain:
  qwen3:235b > qwen3-coder:30b > deepseek-r1:8b > qwen3-vl:8b > qwen2.5:7b-instruct
```

### Manual Model Selection

```bash
python -m geometry_agent.main --llm --model qwen3-coder:30b
```

### Programmatic LLM Usage

```python
from geometry_agent.llm import LLMClient, detect_best_model

# Auto-detect best model
model = detect_best_model() or "qwen3-coder:30b"
llm = LLMClient(model=model)

# Direct query
response = llm.chat("Explain the inscribed angle theorem in geometry.")
print(response)
```

### RAG Integration

```python
from geometry_agent.rag import get_rag

rag = get_rag(enable_web=True)  # Enable DuckDuckGo web search

# Query with retrieval augmentation
context = rag.retrieve("circumcenter equidistant property", top_k=5)
for doc in context:
    print(f"[{doc.source}] {doc.text[:100]}...")
```

---

## Adding New Rules

### Step 1: Define the Rule in Python

In `geometry_agent/rules.py`:

```python
class MyNewRule(Rule):
    """Brief description of what this rule does."""
    name = "my_new_rule"

    def try_apply(self, state: GeoState) -> List[Step]:
        """Apply this rule to all matching fact combinations."""
        results = []
        # Use predicate-indexed lookup for O(1) matching:
        for fact_a in state.by_predicate("Cong"):
            for fact_b in state.by_predicate("Midpoint"):
                if fact_a.args[0] == fact_b.args[1]:  # shared point
                    new_fact = Fact("Perpendicular", (
                        fact_b.args[0], fact_a.args[0],
                        fact_a.args[2], fact_a.args[3],
                    ))
                    if new_fact not in state.facts:
                        results.append(Step(
                            rule_name=self.name,
                            premise_facts=(fact_a, fact_b),
                            conclusion_fact=new_fact,
                        ))
        return results
```

### Step 2: Add the Lean 4 Axiom

In `lean_geo/LeanGeo/Rules.lean`:

```lean
axiom my_new_rule (A B C D M : Point)
    (h1 : Cong A B C D)
    (h2 : Midpoint M A B) :
    Perpendicular M A C D
```

### Step 3: Add the Lean Bridge Spec

In `geometry_agent/lean_bridge.py`:

```python
RuleLeanSpec("my_new_rule", "my_new_rule", ["Cong", "Midpoint"], "Perpendicular"),
```

### Step 4: Add NL Templates

In `geometry_agent/semantic.py`:

```python
# In _RULE_NL_ZH:
"my_new_rule": "我的新规则",

# In _RULE_NL_EN:
"my_new_rule": "my new rule",

# In _RULE_LEAN_NAME:
"my_new_rule": "my_new_rule",
```

### Step 5: Classify in Difficulty Evaluator

In `geometry_agent/difficulty_eval.py`:

```python
# Add to _PRED_FAMILY if new predicate types are involved
# Add to _PRED_TIER to set concept tier
# Add to _TRIVIAL_RULES ONLY if it's a pure symmetry/permutation rule
```

---

## Reading the Output

### HTML Output

Discovered theorems are saved to `output/new_theorems.html`.  Each theorem card shows:

- **Star rating** (1–5 stars, derived from difficulty score)
- **Formal statement**: assumptions ⊢ goal in predicate notation
- **Natural language** (中文): human-readable statement and proof
- **SVG diagram**: geometry visualization with labeled points and relationships
- **Lean 4 code**: complete theorem + proof in Lean 4 syntax
- **Difficulty assessment**: score, label, and detailed breakdown

### Programmatic Access

```python
from geometry_agent.html_export import HtmlExporter

exporter = HtmlExporter()
# Access previously discovered theorems
print(f"Total theorems: {exporter.count}")
print(f"Known fingerprints: {len(exporter._known_fingerprints)}")
```

### Difficulty Report

```python
from geometry_agent.difficulty_eval import evaluate_difficulty

report = evaluate_difficulty(assumptions, goal, steps)
print(f"Score: {report.overall_score}/10 ({report.label_zh})")
print(f"Substantive rules: {report.n_substantive_rules}")
print(f"Concept families: {report.n_concept_families}")
print(f"Knowledge density: {report.knowledge_density}")
print(f"Assessment: {report.assessment_zh}")
```

---

## Advanced Recipes

### Custom Beam Search Configuration

```python
from geometry_agent.search import SearchConfig

# Wide beam for difficult problems
config = SearchConfig(
    beam_width=64,        # Wider beam = more exploration
    max_depth=20,         # Deeper search
    parallel_workers=4,   # Parallel beam expansion
)
```

### Proof Pruning and Compression

```python
from geometry_agent.evolve import prune_proof, compress_proof

# Remove unused assumptions and dead steps
pruned_assm, pruned_steps = prune_proof(assumptions, goal, steps)

# Remove trivial symmetry steps (argument-order swaps)
compressed = compress_proof(pruned_steps)
print(f"Original: {len(steps)} → Pruned: {len(pruned_steps)} → Compressed: {len(compressed)}")
```

### Semantic Fingerprinting

v0.14.0 uses **symmetry-variant canonicalization**: each fingerprint enumerates
all predicate symmetry equivalences (e.g. `Cong(A,B,C,D)` ≡ `Cong(C,D,A,B)`)
× assumption permutations, picks the lexicographic minimum, and hashes it.
This gives true isomorphism-invariant dedup without relying on a fixed
normalisation order.

```python
from geometry_agent.semantic import (
    semantic_theorem_fingerprint,
    structural_theorem_fingerprint,
    compute_isomorphism_map,
)

# Check if two theorems are isomorphic (same up to point renaming)
# Now uses symmetry-variant canonicalization internally
fp1 = semantic_theorem_fingerprint(assm1, goal1)
fp2 = semantic_theorem_fingerprint(assm2, goal2)
is_same = fp1 == fp2

# Check if they are structural variants (same up to predicate-family swaps)
sfp1 = structural_theorem_fingerprint(assm1, goal1)
sfp2 = structural_theorem_fingerprint(assm2, goal2)
is_variant = sfp1 == sfp2
```

### Knowledge Store

```python
from geometry_agent.knowledge import get_global_store

store = get_global_store()

# Record an experience
store.record_experience(
    assumptions=assumptions,
    goal=goal,
    success=True,
    steps=steps,
    explored_nodes=42,
    difficulty=5,
)

# Check cache
cached = store.lookup_proven(assumptions, goal)
if cached:
    print("Already proven (cache hit)!")

# Save to disk
store.save()
print(store.summary())

# Guidance insights
print(store.guidance_summary())
```

### Pólya Plausible Reasoning

```python
from geometry_agent.polya import polya_test

result = polya_test(assumptions, goal, n_trials=50)

# Check whether conjecture is plausible under random numeric instantiations
print(f"Falsified: {result.falsified}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Passed: {result.n_passed}/{result.n_trials} trials")
print(f"Valid samples: {result.n_valid}")
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run from the project root: `cd Geometry_Proof_Agent` |
| `No discoveries` | Lower `min_difficulty` (e.g., 4.0) or increase `max_generations` |
| `Lean check fails` | Ensure `lean_geo` is built: `cd lean_geo && lake build` |
| `Ollama not found` | Install: `brew install ollama && ollama serve` |
| `No LLM detected` | Pull a model: `ollama pull qwen3-coder:30b` |
| `HTML file empty` | Results go to `output/new_theorems.html`; check `output/` directory |
| `ImportError: lean_checker` | Module was merged into `lean_bridge.py` in v0.11.0; use `from geometry_agent.lean_bridge import MockLeanChecker` |
| `Gate D rejects valid theorems` | Ensure you are running v0.13.0+; earlier versions had a thread-safety bug where `_EPS` was corrupted under `ThreadPoolExecutor` (fixed by thread-safe `eps` parameter) |
| `Pólya solver low pass rate` | v0.13.0’s `_smart_init_coords` dramatically improves pass rates for Cyclic/Cong/Midpoint constraints; upgrade if on v0.12.x |
