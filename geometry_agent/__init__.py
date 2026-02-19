"""Geometry Proof Agent – Lean4-verified geometry proving system.

A self-evolving geometry proving framework with five-layer architecture
and a **mutual promotion loop**: knowledge accumulated during evolution
guides future conjecture generation, rule selection, and search strategy,
while new proofs continuously enrich the knowledge base.

  **Layer 1 – Foundation** (基础层):
    DSL data types, 56 deduction rules, semantic fingerprinting,
    knowledge store with guidance API, difficulty evaluation.
    Modules: ``dsl``, ``rules``, ``semantic``, ``knowledge``,
    ``difficulty_eval``.

  **Layer 2 – External Interfaces** (外部接口层):
    Lean 4 checker/bridge (includes CheckResult, LeanChecker protocol,
    MockLeanChecker, ProcessLeanChecker), Ollama LLM client, RAG retrieval.
    Modules: ``lean_bridge``, ``llm``, ``rag``.

  **Layer 3 – Reasoning Core** (推理核心层):
    Symbolic engine + proof verifier (de Bruijn separation),
    knowledge-guided parallel beam search.
    Modules: ``engine`` (includes ProofCertificate, SymbolicEngine,
    VerificationResult, ProofVerifier), ``search``.

  **Layer 4 – Discovery Engines** (发现引擎层):
    Knowledge-adaptive evolution, experience-guided conjecture search,
    genetic algorithm, RLVR (reinforcement learning with verifiable
    rewards), Pólya plausible-reasoning agent (numerical pre-filter).
    Modules: ``evolve``, ``conjecture``, ``genetic``, ``rlvr``, ``polya``.

  **Layer 5 – Orchestration & Entry Points** (编排与入口层):
    Multi-agent pipeline, HTML export with SVG diagrams, CLI.
    Modules: ``pipeline``, ``html_export``, ``main``.

  **Mutual Promotion Loop** (知识↔演化互促循环):
    演化→积累经验→引导搜索/猜想/演化→发现更多定理→积累更多经验
    Evolution → accumulate experience → guide search/conjecture/evolution
    → discover more theorems → accumulate richer experience → ...

Author:  Jiangsheng Yu
License: MIT
"""

__version__ = "0.12.0"
__author__ = "Jiangsheng Yu"
__license__ = "MIT"

__all__ = [
    # Layer 1 — Foundation
    "dsl",
    "rules",
    "semantic",
    "knowledge",
    "difficulty_eval",
    # Layer 2 — External Interfaces
    "lean_bridge",
    "llm",
    "rag",
    # Layer 3 — Reasoning Core
    "engine",
    "search",
    # Layer 4 — Discovery Engines
    "evolve",
    "conjecture",
    "genetic",
    "rlvr",
    "polya",
    "polya_controller",
    # Layer 5 — Orchestration & Entry Points
    "pipeline",
    "html_export",
]
