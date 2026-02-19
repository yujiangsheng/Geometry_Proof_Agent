# Architecture — Geometry Proof Agent v0.13.0

> **Version**: 0.13.0 · **Author**: Jiangsheng Yu · **License**: MIT  
> **18 Python modules** (~19,700 lines) + companion **Lean 4** project

Deep-dive into the system's design, module API, data flow,
and dependency structure.  v0.12.0 introduced a **mutual promotion loop**
where accumulated knowledge guides future conjecture generation, rule
selection, and search strategy, while new proofs enrich the knowledge base.
v0.13.0 adds **thread-safe Pólya agent**, **constraint-aware coordinate
initialisation**, **29 deep generators** (including 3 diversity generators),
and **adaptive Gate D** trial counts.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Five-Layer Architecture](#2-five-layer-architecture)
3. [Mutual Promotion Loop](#3-mutual-promotion-loop)
4. [Module Reference](#4-module-reference)
5. [Agent Inventory](#5-agent-inventory)
6. [Dependency Graph](#6-dependency-graph)
7. [Data Flow](#7-data-flow)
8. [Key Abstractions](#8-key-abstractions)
9. [Quality Pipeline](#9-quality-pipeline)
10. [Lean 4 Integration](#10-lean-4-integration)
11. [Extension Points](#11-extension-points)

---

## 1. Overview

The Geometry Proof Agent is a self-evolving system that discovers, proves,
and formally verifies novel geometry theorems.  It follows a **De Bruijn
criterion** pattern: a fast-but-untrusted Python symbolic engine generates
proof candidates, which are independently verified by a trustworthy Lean 4
kernel.

The system supports three workflows:

- **Proving** — given assumptions + goal → find and verify a proof
- **Evolution** — randomly generate conjectures → prove → filter novel
  theorems not in Lean mathlib4
- **Hybrid Discovery** — coordinate GA, RLVR, Heuristic, MCTS, and Pólya
  strategies to discover high-difficulty theorems

**Design principles**: de Bruijn separation (engine ≠ verifier),
predicate-indexed O(1) lookups, isomorphism-invariant fingerprinting,
knowledge-density gates, mutual promotion loop, and formal Lean 4 as the
trusted ground truth.

---

## 2. Five-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 5  │  pipeline · html_export · main · run_evolve         │
│           │                            Orchestration & Entry    │
├───────────┼─────────────────────────────────────────────────────┤
│  Layer 4  │  evolve · conjecture · genetic · rlvr · polya       │
│           │  polya_controller                       Discovery   │
├───────────┼─────────────────────────────────────────────────────┤
│  Layer 3  │  engine · search                        Reasoning   │
├───────────┼─────────────────────────────────────────────────────┤
│  Layer 2  │  lean_bridge · llm · rag                External    │
├───────────┼─────────────────────────────────────────────────────┤
│  Layer 1  │  dsl · rules · semantic · knowledge ·               │
│           │  difficulty_eval                        Foundation   │
└───────────┴─────────────────────────────────────────────────────┘
```

### Layer 1 — Foundation

| Module | Lines | Purpose |
|--------|------:|---------|
| `dsl.py` | 301 | Core data types: `Fact`, `Goal`, `Step`, `GeoState`, 24 canonical constructors |
| `rules.py` | 1808 | 69 deduction rules including converse/production and cross-family bridge enhancers |
| `semantic.py` | 1081 | Fingerprinting, NL generation (zh/en), Lean 4 codegen, SVG visualisation |
| `knowledge.py` | 840 | Thread-safe persistent store with **guidance API**: rule success profiles, proof chains, predicate coverage, difficulty profiles, structural dedup |
| `difficulty_eval.py` | 568 | Ceva-calibrated difficulty scoring (1–10) with concept-family analysis |

**Invariant**: Foundation modules depend only on `dsl.py` (or nothing).

### Layer 2 — External Interfaces

| Module | Lines | Purpose |
|--------|------:|---------|
| `lean_bridge.py` | 680 | `CheckResult`, `LeanChecker` protocol, `MockLeanChecker`, `ProcessLeanChecker`, batch verification |
| `llm.py` | 650 | Ollama REST client with model auto-detection and preference ranking |
| `rag.py` | 1095 | LocalRetriever (TF-IDF / Ollama embeddings) + WebSearchProvider |

**Invariant**: External modules wrap third-party tools. `llm.py` and
`rag.py` are leaf modules with zero internal dependencies.

### Layer 3 — Reasoning Core

| Module | Lines | Purpose |
|--------|------:|---------|
| `engine.py` | 554 | `SymbolicEngine` + `PythonSymbolicEngine`, `ProofVerifier` + `LeanProofVerifier` + `MockProofVerifier`, `ProofCertificate`, `VerificationResult` |
| `search.py` | 319 | **Knowledge-guided** parallel beam search: experience-weighted scorer (+15 pts rule bonus, +5 pts chain bonus), rule ordering by experience, cache shortcuts |

**De Bruijn separation**: The engine proposes proof steps; the verifier
independently checks them.  They share no mutable state.

### Layer 4 — Discovery Engines

| Module | Lines | Purpose |
|--------|------:|---------|
| `evolve.py` | 2921 | **Knowledge-adaptive** evolution loop: 38+ problem generators, adaptive difficulty, weighted generator selection, proof pruning, compression, adaptive Gate D trials (200/120), cross-session structural dedup |
| `conjecture.py` | 2483 | **Experience-guided** conjecture search: bridge composition, backward chaining, MCTS; 29 deep generators (incl. 3 diversity generators); experience-weighted bridge selection, under-explored predicate targeting |
| `genetic.py` | 889 | Genetic Algorithm: genome encoding, crossover, mutation, tournament |
| `rlvr.py` | 944 | REINFORCE-style policy gradient with symbolic engine as reward oracle |
| `polya.py` | 1629 | **Thread-safe** Pólya plausible-reasoning agent: constraint-aware coordinate initialisation (`_smart_init_coords`), thread-safe epsilon (no `global _EPS`), numerical conjecture validation |
| `polya_controller.py` | 330 | Pólya 4-step adaptive controller: conjecture profiling, plan generation, boosted `premise_probe_trials` for Cyclic/multi-Perp |

**Quality gates**: Every candidate theorem must pass ≥5 checks:
novelty fingerprint, structural fingerprint, knowledge-density ≥ 0.4,
minimum difficulty score, minimum distinct-rule count.

### Layer 5 — Orchestration

| Module | Lines | Purpose |
|--------|------:|---------|
| `pipeline.py` | 698 | Multi-agent pipeline: Parser → Planner → Proposer → Critic → Curriculum |
| `html_export.py` | 1333 | Incremental HTML writer with SVG diagrams and Lean 4 syntax highlighting |
| `main.py` | 331 | CLI (`argparse`): `--lean`, `--llm`, `--hybrid`, `--heuristic`, etc. |
| `run_evolve.py` | 176 | Profile-based runner (`standard` / `fast_discover`) for theorem discovery |

---

## 3. Mutual Promotion Loop

v0.12.0 establishes a bidirectional feedback loop between knowledge
accumulation and theorem discovery:

```
  ┌──────────────────────────────────────────────────────┐
  │            Mutual Promotion Loop (知识互促)             │
  │                                                      │
  │  演化 (Evolution)                                     │
  │    │                                                 │
  │    ▼                                                 │
  │  积累经验 (Accumulate Experience)                      │
  │    ├─ rule_success_profile()  → per-rule success %   │
  │    ├─ proven_rule_chains()    → known proof patterns  │
  │    ├─ predicate_coverage()    → coverage gaps         │
  │    ├─ difficulty_profile()    → solve rates by level  │
  │    └─ generator_success_rates() → generator stats    │
  │    │                                                 │
  │    ▼                                                 │
  │  引导发现 (Guide Discovery)                           │
  │    ├─ search.py: +15 pts rule-experience bonus       │
  │    ├─ search.py: +5 pts proof-chain bonus            │
  │    ├─ search.py: rule ordering by success rate       │
  │    ├─ conjecture.py: experience-weighted bridges     │
  │    ├─ conjecture.py: under-explored predicate focus  │
  │    ├─ evolve.py: adaptive difficulty (80%↑ / 20%↓)   │
  │    └─ evolve.py: 2.5× weight for under-explored gens │
  │    │                                                 │
  │    ▼                                                 │
  │  发现更多定理 → 积累更丰富经验 → 更好引导 → ...          │
  └──────────────────────────────────────────────────────┘
```

### Knowledge Guidance API (`knowledge.py`)

| Method | Returns |
|--------|---------|
| `rule_success_profile()` | Per-rule success rate (e.g., `medians_concurrent: 95%`) |
| `suggest_rule_order(goal_pred)` | Rules sorted by success rate for a given goal predicate |
| `predicate_coverage()` | Coverage percentage per predicate |
| `under_explored_predicates()` | Predicates with < 5 success occurrences |
| `proven_rule_chains()` | Frequently occurring rule chains in proven theorems |
| `structural_dedup_check(assm, goal)` | Cross-session duplicate detection |
| `difficulty_profile()` | Solve rate per difficulty level |
| `generator_success_rates()` | Per-generator success statistics |
| `bridge_success_rates()` | Bridge-rule success rates for conjecture guidance |
| `guidance_summary()` | Formatted summary of all guidance data |

### Semantic-Level Deduplication

Experience is accumulated at the **semantic level** to avoid redundancy:

1. **Proven cache** — keyed by semantic fingerprint (point-relabeling invariant)
2. **Experience buffer** — fingerprint-deduplicated traces
3. **Structural dedup** — rejects predicate-family variants cross-session
4. **Guidance cache** — auto-invalidated when new experience arrives

---

## 4. Module Reference

This section provides both **design narrative** and **API detail** for each
module, organised by layer (bottom-up).

---

### 4.1 `__init__.py` (72 lines) — Package Metadata

| Item | Detail |
|------|--------|
| `__version__` | `"0.13.0"` |
| `__author__` | `"Jiangsheng Yu"` |
| `__all__` | 18 module names exported |

Docstring describes the 5-layer architecture and mutual promotion loop.

**Dependencies**: None.

---

### 4.2 `dsl.py` (301 lines) — Domain-Specific Language

The canonical data model for the entire system.  All other modules depend on
the types defined here.

#### Classes

| Class | Role | Key Fields/Methods |
|-------|------|--------------------|
| `Point` | Geometric point entity | `name: str` |
| `Line` | Line entity | `p1, p2: str` |
| `Circle` | Circle entity | `center, radius_point: str` |
| `Fact` | Atomic geometric proposition | `predicate: str, args: Tuple[str, ...]` (frozen dataclass) |
| `Goal` | Wrapper for a target fact | `fact: Fact` |
| `Step` | Single deduction step | `rule_name: str, premise_facts: Tuple[Fact,...], conclusion_fact: Fact` |
| `GeoState` | Mutable fact database | `facts: Set[Fact]`, `history: Tuple[Step,...]`, `_index: Dict[str, Set[Fact]]` — O(1) predicate-indexed lookup via `add()`, `query()` |

```python
@dataclass(frozen=True)
class Fact:
    predicate: str        # e.g. "Parallel", "Cong", "EqAngle"
    args: Tuple[str, ...]

@dataclass(frozen=True)
class Step:
    rule_name: str
    premise_facts: Tuple[Fact, ...]
    conclusion_fact: Fact

class GeoState:
    facts: Set[Fact]
    history: List[Step]

    def by_predicate(self, p: str) -> Set[Fact]:
        """O(1) predicate-indexed lookup."""
```

#### Canonical Constructors (24 total)

`canonical_parallel`, `canonical_perp`, `canonical_collinear`, `canonical_cyclic`,
`canonical_midpoint`, `canonical_cong`, `canonical_eq_angle`, `canonical_sim_tri`,
`canonical_circle`, `canonical_congtri`, `canonical_tangent`, `canonical_eqratio`,
`canonical_between`, `canonical_angle_bisect`, `canonical_concurrent`,
`canonical_circumcenter`, `canonical_eqdist`, `canonical_eqarea`,
`canonical_harmonic`, `canonical_pole_polar`, `canonical_inv_image`,
`canonical_eq_cross_ratio`, `canonical_radical_axis`, `canonical_oncircle`

Each constructor sorts/normalises arguments to ensure structural equality.

23 core predicates (+ `Circle` alias for `OnCircle` in some docs/output): `Parallel`, `Perpendicular`, `Collinear`, `Cyclic`,
`Midpoint`, `Cong`, `EqAngle`, `SimTri`, `CongTri`, `Tangent`,
`EqRatio`, `Between`, `AngleBisect`, `Concurrent`, `Circumcenter`,
`EqDist`, `EqArea`, `Harmonic`, `PolePolar`, `InvImage`, `EqCrossRatio`,
`RadicalAxis`, `OnCircle`.

**Dependencies**: None (leaf module).

---

### 4.3 `rules.py` (1808 lines) — Deduction Rules

69 rules organised into 8 concept families, including converse/production
and cross-family bridge enhancer rules.

#### Base Types

| Type | Role |
|------|------|
| `Rule` (abstract) | `name: str`, `apply(state: GeoState) → List[RuleApplication]` |
| `RuleApplication` | `rule_name, premises: Tuple[Fact,...], conclusion: Fact` |

#### Rule Inventory (representative families; total 69 rules)

| Family | Rules | Count |
|--------|-------|------:|
| **Parallel / Perpendicular** | `parallel_symmetry`, `parallel_transitivity`, `perp_symmetry`, `parallel_perp_trans`, `perp_parallel_trans` | 5 |
| **Congruence / Distance** | `cong_symm`, `cong_trans`, `eq_dist_to_cong`, `eqdist_from_cong`, `eqdist_to_cong`, `cong_from_eqdist`, `perp_bisector_cong`, `cong_perp_bisector` | 8 |
| **Angle** | `eq_angle_symm`, `eq_angle_trans`, `isosceles_base_angle`, `parallel_alternate_angle`, `angle_bisect_eq_angle`, `angle_bisect_eqratio`, `cyclic_from_eq_angle` | 7 |
| **Triangle Similarity** | `midsegment_sim_tri`, `sim_tri_angle`, `sim_tri_cong`, `congtri_from_sim_cong` | 4 |
| **Midpoint / Between** | `midpoint_collinear`, `midpoint_cong`, `midsegment_parallel`, `between_collinear`, `midpoint_between` | 5 |
| **Circle / Cyclic** | `cyclic_inscribed_angle`, `cyclic_chord_angle`, `circumcenter_cong_ab`, `circumcenter_cong_bc`, `circumcenter_oncircle`, `tangent_perp_radius`, `tangent_oncircle`, `oncircle_from_circumcenter` | 8 |
| **Area / Bisector / CongTri** | `congtri_side`, `congtri_angle`, `congtri_eqarea`, `eqarea_sym`, `eqarea_from_congtri` | 5+ |
| **Projective** | `harmonic_swap`, `harmonic_collinear`, `pole_polar_perp`, `pole_polar_tangent`, `inversion_collinear`, `inversion_circle_fixed`, `cross_ratio_sym`, `cross_ratio_from_harmonic`, `radical_axis_perp`, `eqratio_from_simtri`, `eqratio_sym`, `eqratio_trans`, `medians_concurrent` | 13 |

Each rule implements `try_apply(state) → List[Step]` using predicate-indexed
lookups for efficient matching.

#### Representative Converse Rules

| Rule | Direction |
|------|-----------|
| `cong_from_eqdist` | EqDist → Cong (converse of `eqdist_from_cong`) |
| `eqarea_from_congtri` | CongTri → EqArea (converse of `congtri_eqarea`) |
| `cyclic_from_eq_angle` | EqAngle → Cyclic (converse of `cyclic_inscribed_angle`) |
| `oncircle_from_circumcenter` | Circumcenter → OnCircle (converse of `circumcenter_oncircle`) |
| `perp_parallel_trans` | Perp + Parallel → Perp (complement to `parallel_perp_trans`) |
| `collinear_from_between` | Between → Collinear (ensure Between→Collinear coverage) |
| `concurrent_from_medians` | Midpoints → Concurrent (converse flow of `medians_concurrent`) |

Additional rule families include production and bridge-enhancer rules (for
example circle-metric bridges, projective-circle-line bridges, and
similarity-metric-concurrency bridges).

#### Public Functions

| Function | Role |
|----------|------|
| `default_rules()` | Returns list of all 69 instantiated rules |

**Dependencies**: `dsl` (canonical constructors).

---

### 4.4 `semantic.py` (1081 lines) — Semantic Layer

Solves four problems: fingerprinting, natural language translation,
Lean 4 code generation, and visualisation.

#### Section 1 — Fingerprinting

| Function | Role |
|----------|------|
| `_canonical_relabel(facts)` | Structural relabelling: point names → P0, P1, ... |
| `compute_isomorphism_map(src_facts, dst_facts)` | Constraint propagation + backtracking for point-name bijection |
| `remap_fact(fact, mapping)` / `remap_step(step, mapping)` | Apply point-name mapping |
| `semantic_theorem_fingerprint(assumptions, goal)` | Theorem → canonical fingerprint (point-renaming invariant) |
| `structural_theorem_fingerprint(assumptions, goal)` | Theorem → fingerprint (point-renaming + predicate-family invariant) |
| `semantic_proof_fingerprint(assumptions, steps, goal)` | Full proof → canonical fingerprint |

#### Section 2 — Natural Language

| Function | Role |
|----------|------|
| `fact_to_nl(fact, lang)` | Fact → NL string (zh/en, 24 predicate templates each) |
| `theorem_to_nl(assumptions, goal, lang)` | Full theorem statement in NL |
| `proof_to_nl(assumptions, steps, goal, lang)` | Proof narrative in NL |

Translation maps: `_NL_TEMPLATES_ZH`, `_NL_TEMPLATES_EN` (predicate templates),
`_RULE_NL_ZH`, `_RULE_NL_EN` (legacy-major-rule coverage with fallback to raw rule name).

#### Section 3 — Lean 4 Code Generation

| Function | Role |
|----------|------|
| `theorem_to_lean(assumptions, goal, name?, with_proof?, proof_steps?)` | Complete `.lean` source |

Map: `_PRED_LEAN_NAME` (23 core predicates → Lean names; unknown predicates fall back to raw name).

#### Section 4 — Visualisation

| Function | Role |
|----------|------|
| `draw_geometry(facts, goal?, title?, output_dir?)` | matplotlib-based diagram with CJK fonts, colour-coded predicates |

**Dependencies**: `dsl`, `lean_bridge` (deferred for `RULE_LEAN_MAP`).

---

### 4.5 `knowledge.py` (840 lines) — Knowledge Store with Guidance API

Thread-safe persistent store backed by JSONL files, with a **guidance API**
that extracts actionable insights from accumulated experience:

```
data/knowledge/
├── proven_cache.jsonl      # (fingerprint → proof) cache (~200 entries)
├── experience.jsonl        # Search traces for learning (~330 entries)
├── failure_patterns.json   # Failed attempts for negative mining
└── stats.json              # Global statistics
```

#### Data Types

| Type | Key Fields |
|------|-----------|
| `ProvenEntry` | `fingerprint`, `assumptions`, `goal`, `steps`, `score` |
| `ExperienceRecord` | `fingerprint`, `assumptions`, `goal`, `success`, `steps`, `explored_nodes`, `difficulty`, `timestamp` |
| `KnowledgeStats` | `proven_count`, `experience_count`, `failure_pattern_count`, `cache_hit_count` |

#### Main Class: `KnowledgeStore`

Thread-safe (RLock).  Stores:
- **Proven cache**: `Dict[str, ProvenEntry]` keyed by semantic fingerprint
- **Experience buffer**: `Dict[str, ExperienceRecord]` with fingerprint dedup
- **Failure patterns**: `Counter[str]`
- **Guidance cache**: `Dict[str, Any]` auto-invalidated on new experience

| Method | Role |
|--------|------|
| `lookup_proven(assumptions, goal)` | Find cached sub-goal proof (with isomorphism remapping) |
| `record_proven(assumptions, goal, steps)` | Cache a proved sub-goal |
| `record_experience(...)` | Store search episode (auto-invalidates guidance cache) |
| `record_failure_pattern(pattern_key)` | Track common failure modes |
| `top_failure_patterns(n)` | Most common failures |
| `stats()` / `summary()` | Analytics |
| `save()` / `load()` | Persist to `data/knowledge/` (JSONL + JSON files) |
| `merge_from(other)` | Merge another store (for distributed training) |

#### Guidance API (v0.12.0)

| Method | Returns |
|--------|---------|
| `rule_success_profile()` | `Dict[str, float]` — per-rule success rate |
| `suggest_rule_order(goal_pred)` | `List[str]` — rules sorted by success rate for goal predicate |
| `predicate_coverage()` | `Dict[str, float]` — coverage per predicate |
| `under_explored_predicates()` | `List[str]` — predicates with < 5 successes |
| `proven_rule_chains()` | `List[Tuple[str, int]]` — rule chains by frequency |
| `structural_dedup_check(assm, goal)` | `bool` — cross-session structural duplicate check |
| `difficulty_profile()` | `Dict[int, float]` — solve rate per difficulty level |
| `generator_success_rates()` | `Dict[str, float]` — per-generator success rate |
| `bridge_success_rates()` | `Dict[str, float]` — bridge-rule success rates |
| `guidance_summary()` | `str` — formatted multi-section guidance report |

**Dependencies**: `dsl`, `semantic` (fingerprinting).

---

### 4.6 `difficulty_eval.py` (568 lines) — Difficulty Scoring

Fair, Ceva-calibrated difficulty evaluation for theorems.

#### Formula

$$\text{raw} = N_{\text{distinct}} \times Q \times A \times T \times D \times \rho$$

$$\text{score} = 1 + 9 \cdot \frac{\text{raw}}{10 + \text{raw}}$$

Where:
- $N_{\text{distinct}}$ = number of distinct substantive rules
- $Q$ = quality factor (cross-family bonuses)
- $A$ = auxiliary-point factor
- $T$ = concept-tier factor (1–6 tiers)
- $D$ = diversity factor (family transitions)
- $\rho$ = density factor = $0.7 + 0.3 \times \text{knowledge\_density}$

#### Constants

| Constant | Description |
|----------|------------|
| `_TRIVIAL_RULES` | ~10 symmetry/permutation rules excluded from substantive count |
| `_PRED_FAMILY` | 23 core predicates → 8 concept families |
| `_PRED_TIER` | Predicates → tiers 1–6 |
| `_LABELS_ZH` | 7 difficulty levels: 极易 / 简单 / 中等 / 中等偏难 / 较难 / 难 / 高级 |

#### Public Functions

| Function | Role |
|----------|------|
| `evaluate_difficulty(assumptions, goal, steps)` | → `DifficultyReport` |

**Dependencies**: `dsl` (type hints only).

---

### 4.7 `lean_bridge.py` (680 lines) — Lean 4 Bridge

Lean 4 checker protocol, mock implementation, and real process bridge.

#### Data Types

| Type | Role |
|------|------|
| `CheckResult` | `ok: bool`, `message: str` (frozen dataclass) |
| `LeanChecker` (Protocol) | `check_step(state, step) → CheckResult` |
| `MockLeanChecker` | Fast trust-the-engine mode: accepts if conclusion not already in state |
| `RuleLeanSpec` | Maps a Python rule to its Lean axiom: `lean_lemma: str`, `point_extractor: Callable` |
| `PREDICATE_LEAN_NAME` | Dict mapping 23 core DSL predicates (+ alias handling) → Lean type names |
| `RULE_LEAN_MAP` | Dict mapping legacy-covered rule names → `RuleLeanSpec` (with fallback for newer rules) |

#### Classes

| Class | Role |
|-------|------|
| `ProcessLeanChecker` | Subprocess-based Lean4 kernel invocation. Methods: `check_step()`, `check_full_proof()`, `check_steps_batch()` (parallel via ThreadPoolExecutor), `check_source()` |

#### Public Functions

| Function | Role |
|----------|------|
| `make_checker(use_lean=False)` | Factory → `LeanChecker` |
| `translate_step(step)` | Step → self-contained `.lean` source |
| `translate_full_proof(assumptions, steps, goal)` | Full proof → `.lean` source with intermediate lemmas |

**Dependencies**: `dsl`.

---

### 4.8 `llm.py` (650 lines) — LLM Client

LLM integration via Ollama REST API with auto-detection.

Auto-detects the best available Ollama model:

```
qwen3:235b > qwen3-coder:30b > deepseek-r1:8b > qwen3-vl:8b > qwen2.5:7b-instruct
```

#### Classes

| Class | Role |
|-------|------|
| `LLMResponse` | `content`, `model`, `duration`, `tokens`, `raw` |
| `ModelInfo` | `name`, `family`, `parameter_size`, `quantization`, `size_bytes` |
| `LLMClient` | `chat()`, `generate()`, `clear_history()`, `chat_with_rag()` |

#### Public Functions

| Function | Role |
|----------|------|
| `list_local_models()` | List Ollama models |
| `detect_best_model()` | Preference chain (see above) |
| `get_llm(model?)` | Singleton `LLMClient` |
| `narrate_theorem(...)` | LLM narration of a verified theorem |

**Dependencies**: None (standalone HTTP client).

---

### 4.9 `rag.py` (1095 lines) — Retrieval-Augmented Generation

Three-tier retrieval:
1. **Local vector store** — Ollama embeddings or TF-IDF fallback
2. **Web search** — DuckDuckGo (free) / SerpAPI / Bing
3. **Orchestrator** — `GeometryRAG` merges local + web results

#### Classes

| Class | Role |
|-------|------|
| `WebSearchProvider` | 3-tier cascade: SerpAPI → Bing → DuckDuckGo HTML scraping |
| `LocalRetriever` | Vector store with Ollama embeddings (or TF-IDF fallback). ~100 built-in documents |
| `GeometryRAG` | Orchestrator: local retrieval → threshold check → optional web fallback → merge/rank |

**Dependencies**: None (standalone retrieval module).

---

### 4.10 `engine.py` (554 lines) — De Bruijn Separation

Symbolic reasoning engine and proof verification — de Bruijn separation.

```
  SymbolicEngine (propose)      ProofVerifier (check)
  ─────────────────────         ────────────────────
  Apply rules to GeoState  →  ProofCertificate  →  Verify independently
  May use heuristics            Replay-based checking
  Untrusted                     Trusted (Lean 4 backend)
```

Both engine and verifier protocols live in `engine.py` but maintain strict
separation: they share no mutable state.

#### Classes

| Class | Role |
|-------|------|
| `ProofCertificate` | Self-contained proof object: `assumptions`, `goal`, `steps`, `success`, `explored_nodes`, `cache_hits`, `engine_name`, `engine_version`, `metadata`. Methods: `to_dict()`, `from_dict()`, `to_json()`, `from_json()`, `to_search_result()`, `from_search_result()`. **Contract between engine and verifier.** |
| `SymbolicEngine` (Protocol) | Abstract: `name`, `version`, `solve(assumptions, goal, **kw) → ProofCertificate` |
| `PythonSymbolicEngine` | Concrete implementation using `default_rules()` + `beam_search()` + `MockLeanChecker` |
| `VerificationResult` | `verified: bool`, `message: str`, `verifier_name: str`, `lean_source: str` |
| `ProofVerifier` (Protocol) | Abstract: `verify(certificate: ProofCertificate) → VerificationResult` |
| `LeanProofVerifier` | Translates certificate → `.lean` source → runs Lean4 kernel via `ProcessLeanChecker` |
| `MockProofVerifier` | Always returns `verified=True` (for dev/fast mode) |

#### Public Functions

| Function | Role |
|----------|------|
| `make_engine(kind="python")` | Factory → `SymbolicEngine` |
| `make_verifier(use_lean=False)` | Factory → `ProofVerifier` |

**Dependencies**: `dsl`, `knowledge`, `lean_bridge`, `rules`, `search`.

---

### 4.11 `search.py` (319 lines) — Knowledge-Guided Beam Search

Bounded-width BFS with knowledge-guided enhancements:
- `ThreadPoolExecutor` for parallel beam expansion
- Knowledge-store cache shortcuts (skip already-proven goals)
- **Experience-weighted scorer** (`make_knowledge_scorer()`):
  +15 pts for rules with high success rates, +5 pts for rules in known
  proof chains
- **Rule ordering by experience** — historically successful rules tried first
- Per-depth deduplication via state fingerprinting
- Configurable beam width, max depth, and scorer function

#### Classes

| Class | Key Fields |
|-------|-----------|
| `SearchConfig` | `beam_width=8`, `max_depth=8`, `parallel_workers` (0=auto) |
| `SearchResult` | `success`, `final_state: GeoState`, `explored_nodes`, `cache_hits` |

#### Public Functions

| Function | Signature | Role |
|----------|-----------|------|
| `beam_search()` | `(init_state, goal, rules, checker, config, knowledge_store?, scorer?) → SearchResult` | Core parallel beam search with knowledge cache shortcuts, rule ordering by experience, auto-upgrade to knowledge scorer |
| `make_knowledge_scorer()` | `(knowledge_store, goal) → Callable` | Build experience-weighted scorer: +15 pts for rules with high success rates, +5 pts for rules in proven chains |
| `default_scorer()` | `() → Callable` | Goal-directed proximity scorer |
| `clone_state()` | `(GeoState) → GeoState` | Deep-copy state for parallel branches |

**Dependencies**: `dsl`, `lean_bridge`, `rules`, `knowledge` (deferred import).

---

### 4.12 `evolve.py` (2921 lines) — Knowledge-Adaptive Evolution Loop

The largest module.  Drives theorem discovery in a knowledge-aware loop:

```
for each generation:
    1. Generate random problem instances (38+ generators)
       — generator selection weighted by under-explored predicates (2.5×)
       — difficulty adapts via solve-rate (escalate > 80%, reduce < 20%)
    2. Solve with knowledge-guided beam search
    3. Prune proof (backward BFS removes dead steps)
    4. Compress proof (remove trivial symmetry steps)
    5. Evaluate difficulty (reject if < threshold)
    6. Check novelty (semantic + structural fingerprint, cross-session dedup)
    7. Check knowledge density (reject if < 0.4)
    8. Record experience and failed attempts
    9. Export to HTML, auto-save knowledge with guidance summary
```

#### Novelty Filtering

| Component | Role |
|-----------|------|
| `KNOWN_AXIOM_PATTERNS` | ~50 single-step axiom signatures (trivial) |
| `MATHLIB4_KNOWN_FAMILIES` | ~15 known theorem families to exclude |
| `is_mathlib4_known(...)` | Combined novelty gate |
| `_is_cross_domain_proof(...)` | Requires ≥3 concept families or bridge rules |

#### Problem Generators (38+, difficulty 2–8)

Categories:
- **Chain generators**: `generate_mixed_chain`, `generate_reverse_chain`, `generate_zigzag`
- **Midpoint-based**: `generate_midsegment_perp`, `generate_double_midsegment`, `generate_midpoint_cong_chain`, `generate_triple_midpoint_parallel`
- **Cross-domain**: `generate_diamond`, `generate_perp_transfer_chain`, `generate_perp_bisector_chain`, `generate_cyclic_angle_chain`, `generate_midseg_perp_bisector`, `generate_cyclic_midseg_bridge`
- **Triangle/Circle**: `generate_isosceles_cyclic`, `generate_isosceles_perp_bisector`, `generate_sim_tri_angle_chain`, `generate_cyclic_isosceles_bridge`, `generate_two_triangle_sim`
- **Extended predicates**: `generate_congtri_sim_cong_chain`, `generate_tangent_perp_chain`, `generate_circumcenter_chain`, `generate_angle_bisect_chain`, `generate_pole_polar_perp_chain`, `generate_radical_axis_perp_chain`, `generate_inversion_collinear_chain`, `generate_harmonic_cross_ratio_chain`, `generate_eqdist_midpoint_chain`, `generate_eqarea_congtri_chain`, `generate_concurrent_medians`

#### Knowledge-Adaptive Features (v0.12.0+)

| Feature | Description |
|---------|-------------|
| **Adaptive difficulty** | Escalates if >80% solve rate, reduces if <20% |
| **Weighted generator selection** | Under-explored predicate generators get 2.5× weight |
| **Failed attempt recording** | Records unsuccessful attempts for learning |
| **Cross-session structural dedup** | `structural_dedup_check()` prevents rediscovery |
| **Knowledge auto-save** | Saves store with `guidance_summary()` after each generation |
| **Adaptive Gate D** (v0.13.0) | `_has_inconsistent_premises()` uses 200 trials for Cyclic/multi-Perp, 120 otherwise |

#### Main Functions

| Function | Role |
|----------|------|
| `evolve(...)` | Main knowledge-adaptive evolution loop |
| `evolve_hybrid(...)` | GA + Heuristic + RLVR pipeline with Pólya pre-filter. Returns `(discoveries, conjectures)` |

**Dependencies**: `dsl`, `knowledge`, `lean_bridge`, `rules`, `search`, `semantic`, `difficulty_eval`, `html_export`, `llm`, `conjecture`, `genetic`, `rlvr`, `polya`.

---

### 4.13 `conjecture.py` (2483 lines) — Experience-Guided Conjecture Search

Three strategies with configurable budget allocation, all guided by
accumulated experience:

#### Strategy 1: Bridge Composition (default 40%)

Combine two known lemmas through shared intermediate facts.  Bridge
selection is **experience-weighted**: bridges with high success rates get
+2.0 weight, under-explored predicates get +1.5, and chain frequency
contributes +0.5×count.

| Component | Description |
|-----------|-------------|
| `_RULE_BRIDGES` | 29 entries: `(input_preds) → (output_pred, rule_name)` |
| `_compose_bridges(...)` | Chain bridges end-to-end; experience-weighted selection |

#### Strategy 2: Backward Chaining (default 30%)

Start from a desired conclusion predicate and search backward.  Goal
predicate selection injects **under-explored predicates** for coverage
balance.

| Function | Description |
|----------|-------------|
| `backward_chain_conjecture(...)` | Goal → unfold backward via bridges; injects under-explored predicates |

#### Strategy 3: Deep Generators (default 30%)

**29 deep generators** spanning 4–5 concept families targeting
difficulty ≥ 5.0, organised in 5 tiers:

- **Original 10**: `gen_circumcenter_iso_perp_chain`, `gen_cyclic_iso_midpoint_perp`,
  `gen_double_midpoint_sim_angle`, `gen_circumcenter_midpoint_cong_angle`,
  `gen_perp_bisector_cyclic_bridge`, `gen_angle_bisect_cyclic_chain`,
  `gen_tangent_circumcenter_chain`, `gen_triple_midpoint_concurrent_cong`,
  `gen_pole_polar_midpoint_chain`, `gen_radical_axis_circumcenter`
- **Extended 5**: bridge extended, cross-domain, and similarity
- **Clean 6**: ultra-clean congruence/midpoint/cyclic combinations
- **Diversity 3** (v0.13.0): structurally distinct fingerprint generators
  - `gen_cong_trans_isosceles_angle` (Cong|Cong → EqAngle, METRIC → ANGLE)
  - `gen_double_cong_perp_bisector` (Cong|Cong|Midpoint → Perp, METRIC|MIDPOINT → LINE)
  - `gen_parallel_perp_transfer` (Circumcenter|Midpoint(BC) → Perp, CIRCLE|MIDPOINT → LINE)
- **Ultra-deep 5**: advanced generators spanning higher tiers

#### Strategy 4: MCTS Conjecture Search

| Class | Description |
|-------|-------------|
| `MCTSNode` | Tree node: `pred_path`, `visits`, `total_value`, `best_value`, `ucb1()` |
| `MCTSConjectureSearch` | MCTS over predicate space: select → expand → rollout → backpropagate |

#### Unified Entry Point

| Type | Description |
|------|-------------|
| `HeuristicConfig` | Budget allocation: bridge 40%, backward 30%, deep+MCTS 30% |
| `generate_heuristic_conjectures(config, knowledge_store, verbose)` | Allocates compute across all strategies; returns discovery dicts |

**Dependencies**: `dsl`, `difficulty_eval`, `knowledge`, `lean_bridge`, `rules`, `search`, `genetic` (for `_PRED_META`).

---

### 4.14 `genetic.py` (889 lines) — Genetic Algorithm

Conjectures encoded as chromosomes (`ConjectureGenome`):

```
Gene = (predicate, point_slots)
Genome = (assumption_genes, goal_gene)
```

Operators:
- **Crossover**: swap assumption genes between parents
- **Mutation**: random predicate/point perturbation (7 mutation operators)
- **Selection**: tournament selection with fitness = difficulty × novelty
- **Immigration**: inject fresh random genomes to prevent stagnation

#### Core GA Functions

| Function | Role |
|----------|------|
| `_random_genome()` | Random individual |
| `_genome_from_template()` | Seed from 31 bridge templates |
| `decode_genome(genome)` | Genome → `(assumptions, goal)` |
| `crossover(parent1, parent2)` | Single-point crossover |
| `mutate(genome, rate)` | 7 mutation operators |
| `tournament_select(pop, k)` | Tournament selection |
| `evaluate_fitness(genome, ...)` | Decode → beam_search → difficulty_eval → fitness |
| `run_genetic_evolution(config, ...)` | Full GA loop → `GAResult` |

**Dependencies**: `dsl`, `difficulty_eval`, `knowledge`, `lean_bridge`, `rules`, `search`, `semantic`.

---

### 4.15 `rlvr.py` (944 lines) — REINFORCE with Verifiable Rewards

The symbolic engine provides a perfect, deterministic reward signal:

$$R = \begin{cases}
\text{difficulty\_score} & \text{if proof found and novel} \\
-0.1 & \text{if proof found but trivial/duplicate} \\
0 & \text{if no proof found}
\end{cases}$$

Policy gradient update:

$$\nabla J(\theta) = \mathbb{E}\left[ \sum_t (R - b) \nabla \log \pi_\theta(a_t | s_t) \right]$$

No neural network required — the policy is a categorical distribution over
conjecture templates, updated via REINFORCE with baseline subtraction.

#### Classes

| Class | Role |
|-------|------|
| `Experience` | `assumptions`, `goal`, `steps`, `reward`, `template_name` |
| `ExperienceBuffer` | Sliding window with running mean/std for baseline subtraction |
| `RewardComputer` | Shaped reward: difficulty × weight + novelty bonus + cross-domain bonus |
| `TemplatePolicy` | UCB1 + softmax over conjecture templates with temperature decay |
| `RLVRTrainer` | Orchestrates: policy → select → generate → search → reward → update |

**Dependencies**: `dsl`, `difficulty_eval`, `genetic`, `knowledge`, `lean_bridge`, `rules`, `search`, `semantic`.

---

### 4.16 `polya.py` (912 lines) — Pólya Plausible-Reasoning Agent

Numerical pre-filter inspired by George Pólya's plausible reasoning:

1. **Coordinate instantiation** — randomly assign point coordinates
2. **Numerical evaluation** — check if assumptions and goal hold numerically
3. **Multi-trial validation** — repeat across N random configurations
4. **Confidence scoring** — conjectures passing ≥ threshold trials are
   considered plausible

This filters out geometrically unsound conjectures before expensive
symbolic search, dramatically reducing wasted computation.

**Dependencies**: `dsl` (type hints).

---

### 4.18 `pipeline.py` (698 lines) — Multi-Agent Orchestration

```
User Problem
    │
    ▼
ParserAgent ─────── parse NL → (assumptions, goal)
    │
    ▼
PlannerAgent ────── decompose into sub-goals
    │
    ▼
StepProposerAgent ─ propose proof steps (engine + LLM)
    │
    ▼
CriticReflectAgent ─ verify and repair steps
    │
    ▼
CurriculumAgent ──── schedule progressive difficulty
    │
    ▼
Output (certificate, NL, Lean 4)
```

#### Agent Classes (5 agents)

| Agent | Role |
|-------|------|
| `ParserAgent` | Parse NL / structured geometry problems |
| `PlannerAgent` | Configure search strategy (heuristic table + LLM hints) |
| `StepProposerAgent` | Holds rule inventory |
| `CriticReflectAgent` | Analyse results, diagnose failures (with RAG augmentation) |
| `CurriculumAgent` | Manage difficulty progression: <50 exp→2, <200→3, ≥200→5 |

#### Main Orchestrator: `GeometryPipeline`

| Method | Role |
|--------|------|
| `solve_layered(assumptions, goal)` | Full pipeline: Parse → Plan → Solve → Verify → Critic → Record |
| `solve_structured(assumptions, goal)` | Backward-compat: returns `SearchResult` |

**Dependencies**: `dsl`, `engine`, `knowledge`, `lean_bridge`, `llm`, `rag`, `rules`, `search`.

---

### 4.19 `html_export.py` (1333 lines) — HTML Output

Incremental writer producing a single self-contained HTML file:
- Dark-themed responsive design
- Inline SVG diagrams for each theorem
- Lean 4 code with syntax highlighting
- Star ratings (1–5) from difficulty scores
- Fingerprint deduplication (no duplicate theorem cards)

#### Components

| Component | Description |
|-----------|-------------|
| `_HTML_HEAD` | Dark-themed responsive HTML + CSS |
| `_COLOURS` | 24 predicate-specific SVG colours |
| `_generate_svg(facts, goal)` | Inline SVG diagram with predicate-specific rendering |
| `_highlight_lean(code)` | Regex-based Lean4 syntax highlighting |
| `_render_theorem_card(theorem, index)` | Full theorem card: SVG, NL, Lean4, stars |

#### Public Class

| Class | Role |
|-------|------|
| `HtmlExporter` | Incremental file writer with fingerprint-based cross-run dedup |

**Dependencies**: `dsl`, `evolve` (type hints for `NovelTheorem`).

---

### 4.20 `main.py` (331 lines) — CLI Entry Point

#### CLI Arguments

| Flag | Effect |
|------|--------|
| `--lean` | Enable Lean4 kernel verification |
| `--llm` | Enable LLM features |
| `--rag` | Enable RAG retrieval |
| `--evolve` | Self-evolution loop |
| `--hybrid` | Full hybrid pipeline (recommended) |
| `--heuristic` | Heuristic conjecture search only |
| `--ga` | Genetic Algorithm mode |
| `--rlvr` | RLVR mode |
| `--workers N` | Parallel workers |
| `--model NAME` | Force Ollama model |
| `-v` | Verbose logging |

**Dependencies**: `dsl`, `knowledge`, `llm`, `pipeline`, `semantic`, `engine`, `evolve`, `rag`.

---

## 5. Agent Inventory

| # | Agent | Location | Skill |
|---|-------|----------|-------|
| 1 | **ParserAgent** | `pipeline.py` | Parse NL/structured geometry problems |
| 2 | **PlannerAgent** | `pipeline.py` | Choose search configuration |
| 3 | **StepProposerAgent** | `pipeline.py` | Provide deduction rules |
| 4 | **CriticReflectAgent** | `pipeline.py` | Analyse results, diagnose failures |
| 5 | **CurriculumAgent** | `pipeline.py` | Adapt difficulty progression |
| 6 | **PythonSymbolicEngine** | `engine.py` | Symbolic reasoning |
| 7 | **LeanProofVerifier** | `engine.py` | Lean 4 verification |
| 8 | **MockProofVerifier** | `engine.py` | Fast dev-mode verification |
| 9 | **ProcessLeanChecker** | `lean_bridge.py` | Lean4 subprocess invocation |
| 10 | **MockLeanChecker** | `lean_bridge.py` | Trust-the-engine mode |
| 11 | **LLMClient** | `llm.py` | LLM intelligence |
| 12 | **GeometryRAG** | `rag.py` | Retrieval augmentation |
| 13 | **KnowledgeStore** | `knowledge.py` | Persistent cross-problem memory + guidance API |
| 14 | **DifficultyEvaluator** | `difficulty_eval.py` | Quality gatekeeper |
| 15 | **GAEvolver** | `genetic.py` | Genetic Algorithm conjecture search |
| 16 | **RLVRTrainer** | `rlvr.py` | RL with verifiable rewards |
| 17 | **MCTSConjectureSearch** | `conjecture.py` | MCTS conjecture exploration |
| 18 | **polya_test / Polya utilities** | `polya.py` | Numerical plausible-reasoning pre-filter |
| 19 | **HtmlExporter** | `html_export.py` | Rich output rendering |

---

## 6. Dependency Graph

```
                         ┌─────────┐
                         │  main   │
                         └────┬────┘
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
     ┌─────────┐       ┌──────────┐        ┌──────────────┐
     │ pipeline │       │  evolve  │        │ html_export  │
     └────┬────┘       └────┬─────┘        └──────────────┘
          │                 │
    ┌─────┼─────┐     ┌────┼────────┐
    ▼     ▼     ▼     ▼    ▼    ▼   ▼
 engine  rag   llm  genetic rlvr conjecture  polya
    │                  │     │      │
    │                  └──┬──┘      │
    │                     │         │
    ▼                     ▼         │
 search            difficulty_eval  │
    │                     │         │
    ▼                     │         │
 rules                   │         │
    │                     │         │
    ├─────────────────────┤         │
    ▼                     ▼         ▼
 semantic ◄───────────────────────────
    │
    ▼
 dsl.py  (leaf module, no dependencies)

Cross-cutting:
  knowledge.py ← search, engine, evolve, conjecture, genetic, rlvr, pipeline
  lean_bridge.py ← engine, evolve, semantic
```

### Import Rules

1. **Downward only** — A module may only import from layers below it.
2. **No cycles** — The dependency graph is a DAG.
3. **TYPE_CHECKING** — `difficulty_eval.py` and `html_export.py` use
   `TYPE_CHECKING` imports to avoid circular dependencies while retaining
   type annotations.

---

## 7. Data Flow

### 7.1 Single-Problem Proof Pipeline

```
User Input (NL or structured)
       │
       ▼
  ParserAgent ──→ Problem (assumptions + goal)
       │
       ▼
  PlannerAgent ──→ SearchConfig
       │
       ▼
    PythonSymbolicEngine.solve()
      │  ├─ default_rules() (69 rules) + beam_search()
       │  ├─ KnowledgeStore.lookup_proven()  ← cache shortcut
       │  ├─ make_knowledge_scorer()  ← experience-weighted scoring
       │  └─ MockLeanChecker per step
       │
       ▼
  ProofCertificate
       │
       ▼
  LeanProofVerifier.verify()  ← independent check
       │  ├─ lean_bridge.translate_full_proof()
       │  └─ ProcessLeanChecker._run_lean()
       │
       ▼
  CriticReflectAgent.analyze()
       │  ├─ success → record experience → invalidate guidance cache
       │  └─ failure → diagnose_with_llm() + RAG augmentation
       │
       ▼
  Output: NL theorem, NL proof, Lean4 code, diagram, LLM narration
```

### 7.2 Knowledge-Adaptive Evolution

```
┌────────────────────────────────────────────────────────────┐
│                evolve() loop — knowledge-adaptive           │
│                                                            │
│   ┌─ Load knowledge → guidance_summary()                   │
│   │  ├─ difficulty_profile() → set initial difficulty      │
│   │  ├─ under_explored_predicates() → weight generators    │
│   │  └─ rule_success_profile() → guide beam search         │
│   │                                                        │
│   ▼                                                        │
│   Generator Pool (30+ generators, difficulty 2-8)          │
│   × weighted by under-explored predicates (2.5×)           │
│         │                                                  │
│         ▼                                                  │
│   Generate (assumptions, goal)                             │
│         │                                                  │
│         ▼                                                  │
│   Pólya pre-filter (numerical validation)                  │
│         │                                                  │
│         ▼                                                  │
│   knowledge-guided beam_search()                           │
│   × experience-weighted scorer (+15 pts rule, +5 chain)    │
│   × rules ordered by success rate                          │
│         │                                                  │
│         ▼                                                  │
│   5 Novelty Checks + cross-session structural dedup        │
│         │                                                  │
│         ▼                                                  │
│   evaluate_difficulty() → score ≥ threshold?               │
│         │                                                  │
│         ▼                                                  │
│   ┌─────────────────────┐                                  │
│   │  NovelTheorem found │                                  │
│   └─────────┬───────────┘                                  │
│             │                                              │
│     ┌───────┼───────┬──────────┐                           │
│     ▼       ▼       ▼          ▼                           │
│   Lean4   NL+LLM  HTML     Knowledge                      │
│   verify  narrate  export   Store.record                   │
│                                 │                          │
│                                 ▼                          │
│                   guidance cache invalidated               │
│                   → better scoring next round              │
│                                                            │
│   Adaptive difficulty: escalate if >80%, reduce if <20%    │
│   Auto-save knowledge with guidance_summary() each gen     │
└────────────────────────────────────────────────────────────┘
```

### 7.3 Hybrid Evolution (Heuristic + GA + RLVR)

```
evolve_hybrid(mode="hybrid")
       │
       ├── Load knowledge store → guidance_summary()
       │
       ├──→ Phase 1: generate_heuristic_conjectures()
       │       ├─ Bridge Composition (40%) — experience-weighted
       │       ├─ Backward Chaining (30%) — under-explored predicates
       │       └─ Deep Generators + MCTS (30%)
       │
       ├──→ Phase 2: run_genetic_evolution()
       │       ├─ Population: 50% random + 50% template-seeded
       │       ├─ Selection: tournament
       │       ├─ Operators: crossover, 7 mutation types
       │       ├─ Elitism + immigration
       │       └─ Fitness = difficulty + provability + novelty
       │
       └──→ Phase 3: RLVRTrainer.train()
               ├─ TemplatePolicy: UCB1 + softmax
               ├─ Cross-pollination from GA discoveries
               ├─ RewardComputer: shaped rewards
               └─ Temperature decay for exploitation focus

All phases:
  → Shared KnowledgeStore (guidance available)
  → Shared fingerprint dedup (including structural)
  → Shared HtmlExporter (incremental output)
  → Pólya pre-filter for conjecture validation
```

---

## 8. Key Abstractions

### Protocols

The system uses Python `Protocol` classes for dependency inversion:

| Protocol | Implementations | Purpose |
|----------|----------------|---------|
| `SymbolicEngine` | `PythonSymbolicEngine` | Propose proof steps |
| `ProofVerifier` | `LeanProofVerifier`, `MockProofVerifier` | Check certificates |
| `LeanChecker` | `ProcessLeanChecker`, `MockLeanChecker` | Compile Lean 4 code |

### Canonical Constructors

Every predicate has a `canonical_*()` constructor that enforces a
deterministic argument order. This ensures that `Parallel(A,B,C,D)` and
`Parallel(C,D,A,B)` produce the same `Fact` object.

### Fingerprints

Three levels of fingerprinting:

| Level | Function | Invariant |
|-------|----------|-----------|
| Semantic | `semantic_theorem_fingerprint()` | Point renaming |
| Structural | `structural_theorem_fingerprint()` | Point renaming + predicate family |
| Proof | `semantic_proof_fingerprint()` | Point renaming + proof steps |

---

## 9. Quality Pipeline

Every candidate theorem passes through this pipeline before being accepted:

```
Candidate theorem (assumptions + goal + proof)
  │
  ├─ 1. Proof pruning       (remove dead steps, backward BFS)
  ├─ 2. Proof compression   (remove trivial symmetry steps)
  ├─ 3. Semantic fingerprint (reject isomorphic duplicates)
  ├─ 4. Structural fingerprint (reject predicate-swap variants)
  ├─ 5. Cross-session structural dedup (knowledge store)
  ├─ 6. Difficulty scoring   (reject if score < threshold)
  ├─ 7. Knowledge density    (reject if kd < 0.4)
  ├─ 8. Distinct-rule count  (reject if < min_predicates)
  └─ 9. Mathlib4 known check (reject if matches known library theorem)
        │
        ▼
  Accepted → NovelTheorem → HTML export + optional Lean 4 verify
                          + record experience (feeds back into guidance)
```

---

## 10. Lean 4 Integration

The `lean_geo/` directory is a self-contained Lake project:

```
lean_geo/
├── lakefile.toml          # Lake build config
├── lean-toolchain         # Lean 4 v4.16.0
├── LeanGeo/
│   ├── Defs.lean          # Point, Line, Circle, predicates
│   ├── Basic.lean         # Basic utility lemmas
│   └── Rules.lean         # Lean axioms (legacy-complete; newer rules may use fallback mapping)
└── _check/
    └── test_gen.lean      # Generated verification targets
```

The bridge translates Python proof objects into Lean 4 source:

```
Python (Fact, Step)  →  lean_bridge.py  →  .lean source  →  lake env lean  →  CheckResult
```

Each rule has a `RuleLeanSpec` mapping its Python name to the Lean 4 axiom
name, input/output predicate types, and argument patterns.

---

## 11. Extension Points

### Adding a New Predicate

1. Add canonical constructor in `dsl.py`
2. Add NL templates in `semantic.py` (`_PRED_NL_ZH`, `_PRED_NL_EN`)
3. Add Lean 4 definition in `lean_geo/LeanGeo/Defs.lean`
4. Add SVG rendering case in `html_export.py`
5. Classify in `difficulty_eval.py` (`_PRED_FAMILY`, `_PRED_TIER`)

### Adding a New Rule

1. Create `Rule` subclass in `rules.py`
2. Add Lean 4 axiom in `lean_geo/LeanGeo/Rules.lean`
3. Add `RuleLeanSpec` in `lean_bridge.py`
4. Add NL templates in `semantic.py` (`_RULE_NL_ZH`, `_RULE_NL_EN`)
5. Optionally classify in `difficulty_eval.py` (`_TRIVIAL_RULES`)

### Adding a New Discovery Engine

1. Create module in `geometry_agent/`
2. Import `dsl`, `rules`, `search`, `difficulty_eval` as needed
3. Return `List[NovelTheorem]` from the main entry point
4. Wire into `evolve_hybrid()` in `evolve.py`
5. Add CLI flag in `main.py`

### Adding a New Output Format

1. Create exporter module (like `html_export.py`)
2. Accept `NovelTheorem` objects
3. Call from `evolve.py` alongside `HtmlExporter`

---

*Architecture reference — v0.13.0*
