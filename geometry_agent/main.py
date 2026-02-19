"""main.py – CLI entry point for Geometry Proof Agent (v0.12.0).

Demonstrates the five-layer architecture with built-in demo problems
and provides access to all major features via command-line flags.

Demo problems
~~~~~~~~~~~~~
1. **Parallel transitivity**: AB ∥ CD, CD ∥ EF ⊢ AB ∥ EF
2. **Parallel-perpendicular transfer**: AB ∥ CD, CD ⊥ EF ⊢ AB ⊥ EF

CLI flags
~~~~~~~~~
--lean      Enable real Lean 4 verification (requires lean_geo build)
--llm       Enable LLM narration (requires Ollama)
--hybrid    Run hybrid theorem discovery (heuristic + GA + RLVR)
--heuristic Pure heuristic conjecture search
--ga        Genetic Algorithm evolution
--rlvr      RLVR evolution
--evolve    Classic self-evolution loop
-v          Verbose logging (debug level)

Each demo shows: NL statement, proof steps, Lean 4 code,
verification status, and optional LLM narration.

Author:  Jiangsheng Yu
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import random
import sys

from .dsl import canonical_parallel, canonical_perp
from .knowledge import get_global_store
from .llm import DEFAULT_MODEL, LLMClient, detect_best_model, get_llm
from .pipeline import GeometryPipeline
from .rules import default_rules
from .semantic import (
    draw_geometry,
    fact_to_nl,
    proof_to_nl,
    theorem_to_lean,
    theorem_to_nl,
)
from .engine import ProofCertificate, VerificationResult


def demo_parallel_chain(use_lean: bool, workers: int, llm: LLMClient | None = None, use_llm: bool = False) -> None:
    """AB ∥ CD, CD ∥ EF  ⊢  AB ∥ EF"""
    print("\n══ Demo 1: parallel transitivity ══")
    assumptions = [
        canonical_parallel("A", "B", "C", "D"),
        canonical_parallel("C", "D", "E", "F"),
    ]
    goal = canonical_parallel("A", "B", "E", "F")
    _solve_and_report(assumptions, goal, use_lean, workers, llm=llm, use_llm=use_llm)


def demo_parallel_perp(use_lean: bool, workers: int, llm: LLMClient | None = None, use_llm: bool = False) -> None:
    """AB ∥ CD, CD ⊥ EF  ⊢  AB ⊥ EF"""
    print("\n══ Demo 2: parallel + perpendicular transfer ══")
    assumptions = [
        canonical_parallel("A", "B", "C", "D"),
        canonical_perp("C", "D", "E", "F"),
    ]
    goal = canonical_perp("A", "B", "E", "F")
    _solve_and_report(assumptions, goal, use_lean, workers, llm=llm, use_llm=use_llm)


def demo_synth_batch(workers: int) -> None:
    """Generate & solve a batch of synthetic problems (using evolve generators)."""
    print("\n══ Demo 3: synthetic batch ══")
    from .evolve import generate_mixed_chain
    from .search import SearchConfig, beam_search
    from .lean_bridge import MockLeanChecker
    from .dsl import GeoState, Goal

    cfg = SearchConfig(beam_width=16, max_depth=12)
    checker = MockLeanChecker()
    ok = 0
    total = 50
    for _ in range(total):
        length = random.randint(2, 4)
        assumptions, goal = generate_mixed_chain(length)
        state = GeoState(facts=set(assumptions))
        result = beam_search(init_state=state, goal=Goal(goal),
                             rules=default_rules(), checker=checker, config=cfg)
        if result.success:
            ok += 1
    print(f"  Success rate: {ok}/{total} ({100*ok/total:.1f}%)")


def _solve_and_report(
    assumptions, goal, use_lean: bool, workers: int,
    llm: LLMClient | None = None, use_llm: bool = False,
) -> None:
    pipeline = GeometryPipeline(
        use_lean=use_lean, parallel_workers=workers,
        llm=llm, use_llm=use_llm,
    )

    # ── Layered architecture: Engine → Certificate → Verifier ──
    certificate, verification = pipeline.solve_layered(
        assumptions=list(assumptions), goal=goal,
    )
    result = certificate.to_search_result()

    # ── Natural language ──
    print()
    print("  [自然语言 / Natural Language]")
    print("  " + theorem_to_nl(assumptions, goal, lang="zh").replace("\n", "\n  "))
    print()

    # ── Layered architecture info ──
    print("  [分层架构 / Layered Architecture]")
    print(f"    Layer 1 ─ 符号引擎 : {certificate.engine_name} v{certificate.engine_version}")
    print(f"    Layer 2 ─ 校验器   : {verification.verifier_name}")
    if llm is not None:
        print(f"    Layer 3 ─ 大模型   : {llm.model}")
    elif use_llm:
        print(f"    Layer 3 ─ 大模型   : (auto-detect)")
    print()

    # ── Search result ──
    print("  Success       :", certificate.success)
    print("  Explored nodes:", certificate.explored_nodes)
    print("  Cache hits    :", certificate.cache_hits)
    print("  Derived facts :", len(result.final_state.facts))
    if result.final_state.history:
        print("  Proof steps:")
        for idx, step in enumerate(result.final_state.history, start=1):
            print(f"    {idx}. {step.rule_name}: {step.conclusion_fact}")

    # ── Verification result ──
    if certificate.success:
        status = "✅ VERIFIED" if verification.verified else "❌ REJECTED"
        print(f"\n  [Lean 校验 / Verification] {status}")
        if verification.message:
            print(f"    {verification.message}")

    # ── NL proof ──
    if certificate.success and result.final_state.history:
        print()
        print("  [证明过程 / Proof Narrative]")
        nl_proof = proof_to_nl(
            assumptions, result.final_state.history, goal, lang="zh",
        )
        print("  " + nl_proof.replace("\n", "\n  "))

    # ── LLM narration (verified theorem explained by LLM) ──
    if certificate.success and use_llm and hasattr(pipeline, '_last_llm_narration'):
        narration = pipeline._last_llm_narration
        if narration:
            print()
            print("  [大模型讲解 / LLM Narration]")
            for line in narration.splitlines():
                print(f"    {line}")

    # ── Lean4 theorem statement ──
    print()
    print("  [Lean4 精确表述 / Lean4 Formal Statement]")
    lean_src = theorem_to_lean(
        assumptions, goal,
        with_proof=certificate.success,
        proof_steps=result.final_state.history if certificate.success else None,
    )
    for line in lean_src.splitlines():
        print(f"    {line}")

    # ── Visualisation ──
    try:
        fig_path = draw_geometry(
            facts=list(assumptions),
            goal=goal,
            title=theorem_to_nl(assumptions, goal, lang="en").split("\n")[0],
        )
        if fig_path:
            print(f"\n  [几何图形 / Diagram] → {fig_path}")
    except Exception as exc:
        print(f"\n  (Diagram skipped: {exc})")

    # ── Lean full-chain verification (legacy, only if --lean) ──
    if use_lean and certificate.success and not verification.verified:
        print("  ── Fallback: explicit Lean verification ──")
        ck = pipeline.verify_full_proof(assumptions, result, goal)
        print(f"    Lean check: {'PASS' if ck.ok else 'FAIL'}  {ck.message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Geometry Proof Agent demo")
    parser.add_argument(
        "--lean", action="store_true",
        help="Enable real Lean4 kernel verification (requires lean4 installed)",
    )
    parser.add_argument(
        "--synth", action="store_true",
        help="Run synthetic data batch demo",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Number of parallel workers (0 = auto-detect CPU count)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"Ollama model name (default: auto-detect, recommended: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Enable LLM-powered NL parsing, diagnosis, and strategy hints",
    )
    parser.add_argument(
        "--rag", action="store_true",
        help="Enable RAG retrieval (local vector store + web search) for LLM augmentation",
    )
    parser.add_argument(
        "--evolve", action="store_true",
        help="Start self-evolution loop to discover novel theorems not in Lean4",
    )
    parser.add_argument(
        "--ga", action="store_true",
        help="Use Genetic Algorithm for conjecture generation",
    )
    parser.add_argument(
        "--rlvr", action="store_true",
        help="Use RLVR (Reinforcement Learning with Verifiable Rewards)",
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Hybrid evolution: heuristic + GA + RLVR pipeline",
    )
    parser.add_argument(
        "--heuristic", action="store_true",
        help="Use heuristic conjecture search only",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    # LLM setup
    llm: LLMClient | None = None
    use_llm = args.llm or args.model is not None
    use_rag = args.rag
    if use_llm:
        model_name = args.model
        if model_name is None:
            model_name = detect_best_model()
            if model_name:
                print(f"\n  Auto-detected best LLM: {model_name}")
            else:
                print(f"\n  No local LLM detected. Using default: {DEFAULT_MODEL}")
                model_name = DEFAULT_MODEL
        llm = LLMClient(model=model_name)
        print(f"  LLM ready: {llm.model}")

    # RAG setup (initialise lazily on first use)
    if use_rag:
        from .rag import get_rag
        rag = get_rag(enable_web=True)
        print(f"  RAG ready: {rag._local.doc_count} local docs")

    demo_parallel_chain(use_lean=args.lean, workers=args.workers, llm=llm, use_llm=use_llm)
    demo_parallel_perp(use_lean=args.lean, workers=args.workers, llm=llm, use_llm=use_llm)
    if args.synth:
        demo_synth_batch(workers=args.workers)

    # ── Self-evolution ──
    if args.evolve:
        from .evolve import evolve
        discoveries, conjectures = evolve(
            max_generations=100,
            problems_per_gen=60,
            min_steps=5,
            min_predicates=3,
            min_difficulty=3.0,
            target_novel=3,
            use_lean=args.lean,
            use_llm=use_llm,
            llm_model=args.model or (llm.model if llm else None),
            verbose=True,
        )
        if not discoveries and not conjectures:
            print("\n  未发现新定理或猜想。请增加代数或扩展规则集。")
        elif conjectures and not discoveries:
            print(f"\n  发现 {len(conjectures)} 个未证明猜想 (已通过Pólya合情推理检验)。")

    # ── GA / RLVR / Hybrid / Heuristic evolution ──
    ga_or_rlvr = args.ga or args.rlvr or args.hybrid or getattr(args, 'heuristic', False)
    if ga_or_rlvr:
        from .evolve import evolve_hybrid
        if args.hybrid:
            mode = "hybrid"
        elif args.ga:
            mode = "ga"
        elif args.rlvr:
            mode = "rlvr"
        else:
            mode = "heuristic"

        discoveries, conjectures = evolve_hybrid(
            target_novel=3,
            min_difficulty=3.0,
            use_lean=args.lean,
            use_llm=use_llm,
            llm_model=args.model or (llm.model if llm else None),
            verbose=True,
            mode=mode,
        )
        if not discoveries and not conjectures:
            print("\n  未发现新定理或猜想。")

    # Knowledge store: save to disk & print summary
    store = get_global_store()
    try:
        store.save()
    except Exception as exc:
        print(f"Warning: could not save knowledge: {exc}")
    print("\n" + store.summary())


if __name__ == "__main__":
    main()
