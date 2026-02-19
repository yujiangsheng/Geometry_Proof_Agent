#!/usr/bin/env python3
"""Run hybrid evolution to discover novel geometry theorems."""
import argparse
import logging
import os
import random
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s: %(message)s",
)

sys.path.insert(0, os.path.dirname(__file__))

from geometry_agent.evolve import evolve_hybrid

def _print_results(discoveries, conjectures) -> None:
    print(f"\n{'='*60}")
    print(f"Total discoveries: {len(discoveries)}")
    for i, d in enumerate(discoveries, 1):
        print(f"\n--- Theorem #{i} ---")
        print(f"  Difficulty: {d.difficulty_score:.1f}/10 ({d.difficulty_label_zh})")
        print(f"  Steps: {d.n_steps}")
        print(f"  Predicates: {d.predicate_types}")
        print(f"  Rules: {d.rule_types_used}")
        print(f"  Statement: {d.nl_statement}")

    if conjectures:
        print(f"\n{'='*60}")
        print(f"Unproven conjectures (Pólya-plausible): {len(conjectures)}")
        for i, c in enumerate(conjectures, 1):
            conf_pct = c.polya_confidence * 100
            print(f"\n--- Conjecture #{i} (confidence: {conf_pct:.0f}%) ---")
            print(f"  Pólya: {c.polya_n_passed}/{c.polya_n_trials} trials passed")
            print(f"  Predicates: {c.predicate_types}")
            print(f"  Statement: {c.nl_statement}")


def _run_standard(args):
    discoveries, conjectures = evolve_hybrid(
        target_novel=args.target_novel,
        min_difficulty=args.min_difficulty,
        use_lean=False,
        use_llm=False,
        verbose=True,
        mode=args.mode,
    )
    return discoveries, conjectures


def _run_fast_discover(args):
    """Fast theorem discovery mode.

    Strategy:
      - multiple short heuristic runs with different random seeds
      - cross-round dedup by semantic fingerprint
      - early stop when target_novel reached
    """
    target = args.target_novel
    rounds = max(1, args.rounds)
    per_round_target = max(1, args.per_round_target)
    base_seed = args.base_seed

    all_discoveries = []
    all_conjectures = []
    seen_fp = set()
    t0 = time.time()

    if args.reset_html:
        try:
            from geometry_agent.html_export import HtmlExporter
            html_path = HtmlExporter().path
            if html_path.exists():
                html_path.unlink()
            HtmlExporter()  # recreate skeleton
            print(f"reset html: {html_path}")
        except Exception as exc:
            print(f"warning: reset html failed: {exc}")

    print("\n" + "=" * 60)
    print("FAST DISCOVER MODE")
    print(f"target={target}, rounds={rounds}, per_round_target={per_round_target}")
    print("mode=heuristic (short multi-seed runs)")

    for ridx in range(rounds):
        seed = base_seed + ridx
        random.seed(seed)
        print(f"\n[Round {ridx + 1}/{rounds}] seed={seed}")

        discoveries, conjectures = evolve_hybrid(
            target_novel=per_round_target,
            min_difficulty=args.min_difficulty,
            use_lean=False,
            use_llm=False,
            verbose=True,
            mode="heuristic",
        )

        added = 0
        for d in discoveries:
            fp = getattr(d, "fingerprint", "")
            if fp and fp in seen_fp:
                continue
            if fp:
                seen_fp.add(fp)
            all_discoveries.append(d)
            added += 1
            if len(all_discoveries) >= target:
                break

        for c in conjectures:
            fp = getattr(c, "fingerprint", "")
            if fp and fp in seen_fp:
                continue
            all_conjectures.append(c)

        print(f"  round discoveries: {len(discoveries)} (added unique: {added})")
        print(f"  total unique discoveries: {len(all_discoveries)}/{target}")
        if len(all_discoveries) >= target:
            break

    elapsed = time.time() - t0
    print(f"\nFAST DISCOVER finished in {elapsed:.1f}s")
    return all_discoveries[:target], all_conjectures


def main() -> int:
    parser = argparse.ArgumentParser(description="Run theorem evolution")
    parser.add_argument(
        "--profile",
        choices=["standard", "fast_discover"],
        default="standard",
        help="Run profile",
    )
    parser.add_argument("--target-novel", type=int, default=5)
    parser.add_argument("--min-difficulty", type=float, default=1.5)
    parser.add_argument(
        "--mode",
        choices=["hybrid", "heuristic", "ga", "rlvr"],
        default="hybrid",
        help="Mode used by standard profile",
    )
    parser.add_argument("--rounds", type=int, default=5,
                        help="Rounds for fast_discover")
    parser.add_argument("--per-round-target", type=int, default=3,
                        help="Per-round target for fast_discover")
    parser.add_argument("--base-seed", type=int, default=20260218,
                        help="Base random seed for fast_discover")
    parser.add_argument(
        "--reset-html",
        dest="reset_html",
        action="store_true",
        help="Reset output/new_theorems.html before fast_discover runs (default)",
    )
    parser.add_argument(
        "--no-reset-html",
        dest="reset_html",
        action="store_false",
        help="Keep existing output/new_theorems.html fingerprints",
    )
    parser.set_defaults(reset_html=True)
    args = parser.parse_args()

    if args.profile == "fast_discover":
        discoveries, conjectures = _run_fast_discover(args)
    else:
        discoveries, conjectures = _run_standard(args)

    _print_results(discoveries, conjectures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
