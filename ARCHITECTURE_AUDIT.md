# Architecture Audit (Delta)

> Scope: keep only audit conclusions and action items.
> Canonical architecture reference: [ARCHITECTURE.md](ARCHITECTURE.md).
> Last updated: 2026-02-18

---

## 1) Why this file exists

`ARCHITECTURE.md` is now the single source of truth for full architecture and module inventory.
This file intentionally keeps only:

- audit findings,
- risk hotspots,
- cleanup decisions,
- follow-up actions.

This avoids long-term doc drift caused by duplicated full-architecture descriptions.

---

## 2) Current audit findings

### Strengths

- Layering is clear and mostly respected (`foundation -> interfaces -> reasoning -> discovery -> orchestration`).
- De Bruijn-style separation is preserved (symbolic engine vs verifier).
- Knowledge-guided loop is implemented and materially improves search guidance.
- Semantic/structural dedup is integrated across discovery and persistence.
- HTML export now includes stronger consistency safeguards (statement/proof/diagram alignment).

### Hotspots

- `geometry_agent/evolve.py` and `geometry_agent/conjecture.py` are still high-complexity modules.
- Discovery pipeline has many quality gates; observability is good but parameter coupling is non-trivial.
- Runtime-generated output (`output/new_theorems.html`) can become stale/noisy between runs without profile flags.

---

## 3) Redundancy cleanup decisions

- Full module-by-module duplication removed from this file.
- `ARCHITECTURE.md` remains the only full architecture document.
- This file is now a compact audit delta and maintenance checklist.

---

## 4) Maintenance checklist

When architecture changes, update in this order:

1. Update `ARCHITECTURE.md` (source of truth).
2. Update this file **only if** audit conclusions changed.
3. If CLI/runtime behavior changed, sync `README.md` and `USAGE.md` examples.

---

## 5) Next low-risk improvements

- Add lightweight per-module complexity/ownership table in `ARCHITECTURE.md`.
- Split some long functions in `conjecture.py` into smaller strategy helpers.
- Add a short “fresh run” recipe in docs to avoid stale HTML interpretation.

---

## 6) Verification status

- No code-path behavior changed by this document refactor.
- This update is documentation-only and removes cross-file duplication risk.
