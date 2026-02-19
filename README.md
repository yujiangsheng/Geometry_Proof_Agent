# EasyGeometry: A Geometry Proof Agent

**Lean4-Verified Geometry Theorem Discovery & Proving System**

> **Author:** Jiangsheng Yu  ·  **License:** MIT  ·  **Version:** 0.14.0

A multi-agent framework that **discovers novel geometry theorems** through
symbolic search, verifies every derivation with the **Lean 4** kernel, and
explains results in natural language.  A **mutual promotion loop** lets
accumulated knowledge guide conjecture generation, rule selection, and
search strategy, while new proofs continuously enrich the knowledge base.

---

## Highlights

| Feature | Description |
|---------|-------------|
| **23 geometric predicates** | Parallel, Perpendicular, Collinear, Midpoint, Cong, EqAngle, Cyclic, SimTri, CongTri, Circumcenter, Tangent, EqRatio, Between, AngleBisect, Concurrent, EqDist, EqArea, Harmonic, PolePolar, InvImage, EqCrossRatio, RadicalAxis, OnCircle (`Circle` is an alias in some docs/output) |
| **69 deduction rules** | Expanded rule base with converse/production and cross-family bridge enhancers (Lean map fallback supported for new rules) |
| **5 discovery engines** | Heuristic search · Genetic Algorithm · RLVR · MCTS · Pólya plausible-reasoning |
| **29 deep generators** | Hand-crafted conjecture generators spanning 4–5 concept families, including 3 clean diversity generators for structurally distinct fingerprints |
| **Thread-safe Pólya agent** | Numerical pre-filter with constraint-aware coordinate initialisation (`_smart_init_coords`), fully thread-safe — no global-state mutation |
| **Mutual promotion loop** | Knowledge ↔ evolution: experience guides search/conjecture; new proofs enrich knowledge |
| **Knowledge-guided search** | Beam search scorer with rule-experience bonus (+15 pts) and proof-chain bonus (+5 pts) |
| **Adaptive Gate D** | Premise-consistency check with adaptive trial counts (200 for Cyclic/multi-Perp, 120 otherwise) |
| **Relay variable elimination** | Post-proof cleanup removes pass-through point renames for cleaner theorems |
| **Symmetry-variant fingerprinting** | Enumerates predicate symmetry equivalences × assumption permutations for true isomorphism-invariant dedup |
| **Anti-substitution filter** | Structural fingerprinting rejects trivial predicate-swap variants |
| **Proof compression** | Symmetry steps auto-removed; only substantive reasoning kept |
| **Fair difficulty scoring** | Ceva-calibrated 1–10 scale with density, diversity & tier factors |
| **Styled HTML export** | Dark-themed HTML with inline SVG geometry diagrams |
| **Local LLM support** | Ollama integration (Qwen / DeepSeek / Llama) for NL parsing & narration |
| **RAG retrieval** | Local vector store + web search for LLM augmentation |

---

## Architecture (5 Layers)

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 5 — Orchestration & Entry Points                      │
│  pipeline.py · html_export.py · main.py · run_evolve.py      │
│  Multi-agent pipeline · HTML export · CLI                     │
├──────────────────────────────────────────────────────────────┤
│  Layer 4 — Discovery Engines                                 │
│  evolve.py · conjecture.py · genetic.py · rlvr.py · polya.py  │
│  polya_controller.py                                         │
│  Knowledge-adaptive evolution · 29 deep generators            │
│  Genetic Algorithm · RLVR · Thread-safe Pólya agent           │
├──────────────────────────────────────────────────────────────┤
│  Layer 3 — Reasoning Core                                    │
│  engine.py · search.py                                       │
│  Symbolic engine (de Bruijn separation) ·                     │
│  Knowledge-guided parallel beam search                        │
├──────────────────────────────────────────────────────────────┤
│  Layer 2 — External Interfaces                               │
│  lean_bridge.py · llm.py · rag.py                            │
│  Lean 4 checker + bridge · Ollama LLM · RAG retrieval         │
├──────────────────────────────────────────────────────────────┤
│  Layer 1 — Foundation                                        │
│  dsl.py · rules.py · semantic.py · knowledge.py ·             │
│  difficulty_eval.py                                           │
│  23 predicates · 69 rules · fingerprinting · knowledge store  │
│  with guidance API · difficulty evaluation                    │
└──────────────────────────────────────────────────────────────┘

          ┌─────────────────────────────────────┐
          │   Mutual Promotion Loop (知识互促)    │
          │                                     │
          │  Evolution ──→ accumulate experience │
          │       ↑                ↓             │
          │  discover theorems ←── guide search  │
          │                       / conjecture   │
          │                       / evolution    │
          └─────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- **Python ≥ 3.9** (no external packages required for core functionality)
- **Lean 4** (optional — for kernel verification; see `lean_geo/`)
- **Ollama** (optional — for LLM features; `brew install ollama`)

### Installation

```bash
git clone https://github.com/yujiangsheng/Geometry_Proof_Agent.git
cd Geometry_Proof_Agent
```

No `pip install` needed — the project is self-contained pure Python.

### Basic Demo

```bash
# Run built-in demos (parallel transitivity, parallel-perp transfer)
python -m geometry_agent.main

# With verbose output
python -m geometry_agent.main -v
```

### Discover Novel Theorems

```bash
# Hybrid evolution (recommended): heuristic + GA + RLVR
python -m geometry_agent.main --hybrid

# Or use the profile-based runner (recommended for batch discovery):
python run_evolve.py

# Fast multi-seed short rounds (higher throughput):
python run_evolve.py --profile fast_discover --target-novel 8 --rounds 6 --per-round-target 3

# Keep existing HTML fingerprints (do not reset output/new_theorems.html):
python run_evolve.py --profile fast_discover --no-reset-html

# Pure heuristic search
python -m geometry_agent.main --heuristic

# Pure Genetic Algorithm
python -m geometry_agent.main --ga

# Pure RLVR (Reinforcement Learning with Verifiable Rewards)
python -m geometry_agent.main --rlvr

# Classic evolution loop
python -m geometry_agent.main --evolve
```

Results are written to `output/new_theorems.html` with styled HTML and SVG diagrams.

`run_evolve.py` profiles:
- `standard`: one direct run (supports `--mode hybrid|heuristic|ga|rlvr`)
- `fast_discover`: multi-round multi-seed heuristic discovery + cross-round dedup

### Enable Lean 4 Verification

```bash
# Install Lean 4 toolchain (if not already installed)
curl https://raw.githubusercontent.com/leanprover/elan/main/elan-init.sh -sSf | sh

# Build the Lean project
cd lean_geo && lake build && cd ..

# Run with real Lean verification
python -m geometry_agent.main --lean --hybrid
```

### Enable LLM Features

```bash
# Start Ollama (install: brew install ollama)
ollama serve

# Pull a recommended model
ollama pull qwen3-coder:30b

# Run with LLM narration
python -m geometry_agent.main --llm --hybrid

# Specify a custom model
python -m geometry_agent.main --llm --model deepseek-r1:8b

# Enable RAG (retrieval-augmented generation)
python -m geometry_agent.main --llm --rag --hybrid
```

---

## CLI Reference

```
python -m geometry_agent.main [OPTIONS]

Options:
  --lean          Enable Lean 4 kernel verification
  --llm           Enable local LLM (Ollama) for NL parsing & narration
  --model MODEL   Specify Ollama model (default: auto-detect)
  --rag           Enable RAG retrieval (local + web search)
  --evolve        Classic self-evolution loop
  --hybrid        Hybrid evolution: heuristic + GA + RLVR (recommended)
  --heuristic     Heuristic conjecture search only
  --ga            Genetic Algorithm evolution
  --rlvr          RLVR evolution
  --workers N     Parallel workers (0 = auto-detect)
  -v, --verbose   Verbose logging
```

```
python run_evolve.py [OPTIONS]

Options:
   --profile {standard,fast_discover}
   --target-novel N
   --min-difficulty X
   --mode {hybrid,heuristic,ga,rlvr}      # for standard profile
   --rounds N                              # for fast_discover
   --per-round-target N                    # for fast_discover
   --base-seed N                           # for fast_discover
   --reset-html / --no-reset-html
```

---

## How It Works

### 1. Mutual Promotion Loop (知识↔演化互促)

The core innovation of v0.12.0 (enhanced in v0.13.0): knowledge and
evolution form a self-reinforcing cycle.

```
  Evolution (演化)
      │
      ▼
  Accumulate experience (积累经验)
  ─ rule success profiles
  ─ proof chain patterns
  ─ difficulty solve rates
  ─ generator success rates
      │
      ▼
  Guide future discovery (引导发现)
  ─ knowledge-scored beam search (+15 pts rule bonus)
  ─ experience-weighted bridge selection
  ─ under-explored predicate targeting
  ─ adaptive difficulty control
      │
      ▼
  Discover more theorems → richer experience → better guidance → ...
```

### 2. Problem Generation

The system generates geometry problems of increasing complexity using 38+
**problem generators** (evolve.py, difficulty 2–8) plus **29 deep generators**
(conjecture.py, targeting difficulty ≥ 5.0):

- **Basic chains** — parallel / perpendicular transitivity chains
- **Mixed generators** — midpoint + congruence + angle combinations
- **Circle generators** — circumcenter, cyclic, tangent problems
- **Projective generators** — harmonic ranges, pole-polar, inversions
- **Diversity generators** (v0.13.0) — 3 new generators producing
  structurally distinct fingerprints:
  - `gen_cong_trans_isosceles_angle` (METRIC → ANGLE)
  - `gen_double_cong_perp_bisector` (METRIC|MIDPOINT → LINE)
  - `gen_parallel_perp_transfer` (CIRCLE|MIDPOINT → LINE)

Generator selection is **experience-weighted**: under-explored predicate
generators receive 2.5× sampling weight; difficulty adapts based on
solve-rate feedback (escalate if >80%, reduce if <20%).

### 3. Symbolic Search

Each problem is solved by **knowledge-guided beam search** (bounded-width
BFS) over 69 deduction rules.  The predicate-indexed `GeoState` ensures
O(1) rule matching.  The knowledge store provides:

- **Cached sub-goal shortcuts** — skip already-proven goals
- **Rule ordering by experience** — try historically successful rules first
- **Experience-weighted scoring** — +15 pts for rules with high success rates
- **Proof chain bonuses** — +5 pts for rules appearing in known proof chains

### 4. Pólya Plausible Reasoning (Numerical Pre-Filter)

Before attempting expensive symbolic proof, the **Pólya agent** validates
conjectures numerically:

1. **Constraint-aware coordinate initialisation** (`_smart_init_coords`,
   v0.13.0) — seeds points on a circle for `Cyclic`, analytically solves
   `Cyclic+Midpoint+Perp`, and uses symmetric placement for `Cyclic+Cong`.
   This boosts solver success from ~30% to ~100% on complex constraint
   combinations.
2. Evaluate assumptions and goal numerically with thread-safe per-call
   epsilon (no global state mutation)
3. Repeat across multiple random trials
4. **Adaptive Gate D** (premise-consistency check): 200 trials for
   Cyclic/multi-Perp constraints, 120 otherwise
5. Filter out conjectures that fail numerically (confidence < threshold)

The Pólya agent is fully **thread-safe** (v0.13.0): the former
`global _EPS` save/modify/restore pattern was replaced by a local `eps`
parameter, eliminating race conditions under `ThreadPoolExecutor`.

This dramatically reduces wasted search effort on unsound conjectures.

### 5. Proof Quality

Discovered proofs go through three quality filters:

| Filter | Description |
|--------|-------------|
| **Pruning** | Backward BFS removes unused assumptions and dead steps |
| **Compression** | Symmetry steps (argument swaps) are inlined and eliminated |
| **Density check** | Proofs with < 40% unique rules are rejected as repetitive |

### 6. Novelty Verification

Each theorem passes 6 novelty gates:

1. ≥ N substantive proof steps (default: 5)
2. ≥ M distinct predicate types (default: 3)
3. Not a known mathlib4 family (single-domain results excluded)
4. ≥ 3 distinct rule types used
5. Genuinely cross-domain (≥ 3 concept families or bridge rules)
6. Symmetry-variant fingerprint dedup — enumerates all predicate symmetry
   equivalences × assumption permutations for true isomorphism-invariant
   and anti-substitution dedup (cross-session structural dedup included)

### 7. Difficulty Evaluation

The **Ceva-calibrated** difficulty evaluator scores theorems on a 1–10 scale:

```
score = 1 + 9 × raw / (10 + raw)

raw = N_distinct × quality × aux × tier × diversity × density
```

| Factor | Formula | Rewards |
|--------|---------|---------|
| quality | 0.5 + 0.5 × (substantive / total) | Clean proofs without padding |
| aux | 1 + 0.3 × N_aux | Auxiliary point constructions |
| tier | 1 + 0.1 × (max_tier − 1) | Higher concept tiers (circle, projective) |
| diversity | 1 + 0.15 × (N_families − 1) | Cross-family breadth |
| density | 0.7 + 0.3 × (distinct_rules / total) | Non-repetitive reasoning |

Score labels: 极易(≤1.5) · 初级(≤2.5) · 简单(≤4.0) · 中等(≤5.5) · 较难(≤7.0) · 困难(≤8.5) · 高级(≤10)

### 8. Output

Discovered theorems are exported to `output/new_theorems.html` with:
- Formal statement (assumptions ⊢ goal)
- Natural language description (中文 / English)
- Step-by-step proof narration
- Lean 4 code
- Inline SVG geometry diagram
- Post-generation statement/proof consistency auto-repair
- Point-coverage safeguard for diagram labels
- Difficulty rating with star visualization

---

## Project Layout

```
Geometry_Proof_Agent/
├── README.md                      # This file
├── USAGE.md                       # Detailed usage examples & recipes
├── ARCHITECTURE.md                # Architecture deep-dive, API reference & design rationale
├── LICENSE                        # MIT License
├── run_evolve.py                  # Profile-based theorem discovery runner
│
├── geometry_agent/                # Main Python package (18 modules, ~19,900 lines)
│   ├── __init__.py                # Package metadata (v0.14.0)
│   ├── dsl.py                     # Domain-specific language: Fact, Step, GeoState (301)
│   ├── rules.py                   # 69 deduction rules with O(1) indexed matching (1808)
│   ├── search.py                  # Knowledge-guided parallel beam search (319)
│   ├── engine.py                  # Symbolic engine + proof verifier (de Bruijn) (554)
│   ├── lean_bridge.py             # Lean4 checker protocol + process bridge (680)
│   ├── llm.py                     # LLM client (Ollama) with auto-detection (650)
│   ├── rag.py                     # RAG: local vector store + web search (1095)
│   ├── pipeline.py                # Multi-agent orchestration (5 agents) (698)
│   ├── evolve.py                  # Knowledge-adaptive self-evolution + relay elimination (3060)
│   ├── conjecture.py              # 29 deep generators + MCTS conjecture search (2480)
│   ├── genetic.py                 # Genetic Algorithm for conjecture evolution (889)
│   ├── rlvr.py                    # RLVR (RL with Verifiable Rewards) (944)
│   ├── polya.py                   # Thread-safe Pólya agent + smart init (1646)
│   ├── polya_controller.py        # Pólya four-step adaptive controller (330)
│   ├── knowledge.py               # Persistent knowledge store with guidance API (840)
│   ├── semantic.py                # Fingerprints (symmetry-variant), NL, Lean4, viz (1272)
│   ├── difficulty_eval.py         # Fair difficulty evaluation agent (568)
│   ├── html_export.py             # Styled HTML export with SVG diagrams (1333)
│   └── main.py                    # CLI entry point (331)
│
├── lean_geo/                      # Lean 4 Lake project (v4.16.0)
│   ├── lakefile.toml              # Lake build config
│   ├── lean-toolchain             # Lean 4 v4.16.0
│   ├── LeanGeo.lean               # Top-level import
│   ├── Main.lean                  # Executable entry point
│   └── LeanGeo/
│       ├── Defs.lean              # Geometric type definitions
│       ├── Basic.lean             # Basic lemmas
│       └── Rules.lean             # Lean axioms (legacy-complete; new rules use fallback when missing)
│
├── data/knowledge/                # Auto-managed persistent knowledge
│   ├── proven_cache.jsonl         # Cached proven sub-goals (~200 entries)
│   ├── experience.jsonl           # Search episode traces (~330 entries)
│   ├── failure_patterns.json      # Failure type frequencies
│   └── stats.json                 # Runtime statistics
│
├── data/rag/                      # RAG vector store
│   └── documents.jsonl            # Local knowledge documents (~100 entries)
│
└── output/                        # Generated outputs
    └── new_theorems.html          # Discovered theorems (auto-generated)
```

---

## Concept Families & Tiers

The 23 core predicates (with `Circle` as alias output in some paths) are organized into **8 concept families** across **6 tiers**:

| Tier | Domain | Predicates |
|------|--------|------------|
| 1 | LINE | Parallel, Perpendicular, Collinear, Between |
| 2 | MIDPOINT / ANGLE | Midpoint, AngleBisect |
| 3 | METRIC | Cong, EqAngle, EqDist, EqArea, EqRatio |
| 4 | CIRCLE | Cyclic, OnCircle, Circumcenter, Tangent, RadicalAxis |
| 5 | SIMILARITY / CONCURRENCY | SimTri, CongTri, Concurrent |
| 6 | PROJECTIVE | Harmonic, PolePolar, InvImage, EqCrossRatio |

A theorem that spans **≥ 3 families** is considered **cross-domain** and scores
higher on the difficulty scale.

---

## Key Design Principles

1. **Lean only certifies truth; agents provide speed & intelligence.**
   Any guess or derivation must be Lean-verified before entering the knowledge base.

2. **Knowledge accumulates with semantic deduplication.**
   A global `KnowledgeStore` caches proven sub-goals and experience traces,
   fingerprint-deduplicated, shared across search trees.  A guidance API
   extracts actionable insights (rule rankings, proof chains, coverage gaps).

3. **Mutual promotion loop.**
   Knowledge guides conjecture generation, rule selection, search scoring,
   and difficulty targeting.  New proofs continuously enrich the store,
   forming a self-reinforcing cycle.

4. **Predicate-indexed state.**
   `GeoState` maintains an internal `Dict[str, Set[Fact]]` index so rules
   look up relevant facts in O(1) instead of scanning all facts.

5. **Proof conciseness over verbosity.**
   Trivial symmetry steps (argument-order swaps) are automatically compressed.
   Only substantive reasoning steps appear in the final proof.

6. **Anti-substitution novelty.**
   Structural fingerprinting ensures that theorems obtained by merely swapping
   one predicate for another within the same family (e.g., Parallel ↔ Perpendicular)
   are recognized as duplicates, including cross-session structural dedup.

7. **Layered architecture with de Bruijn separation.**
   Engine solves without Lean; Verifier independently certifies; LLM explains;
   RAG augments.  Each layer is optional and independently testable.

---

## Contributing

Contributions are welcome.  To add a new deduction rule:

1. **`rules.py`** — Subclass `Rule`, implement `try_apply()` with predicate-indexed matching
2. **`lean_geo/LeanGeo/Rules.lean`** — Add the corresponding Lean 4 axiom
3. **`lean_bridge.py`** — Add `RuleLeanSpec` entry for proof translation
4. **`semantic.py`** — Add NL templates (`_RULE_NL_ZH` / `_RULE_NL_EN`) and Lean name mapping
5. **`difficulty_eval.py`** — Classify the rule: assign family, tier; mark as trivial if symmetry-only

See [USAGE.md](USAGE.md) for detailed examples.

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{yu2025geometry_proof_agent,
  author  = {Yu, Jiangsheng},
  title   = {Geometry Proof Agent: Lean4-Verified Geometry Theorem Discovery},
  year    = {2025},
  url     = {https://github.com/yujiangsheng/Geometry_Proof_Agent},
  version = {0.14.0}
}
```
