# LeanGeo — Lean 4 Geometry Axiom Library

Companion Lean 4 project for the Geometry Proof Agent.  Provides formal
definitions and axiom declarations used for independent proof verification.

## Structure

```
lean_geo/
├── lakefile.toml          # Lake build configuration
├── lean-toolchain         # Lean 4 v4.16.0
├── LeanGeo.lean           # Root import file
├── Main.lean              # Lake entry point
├── LeanGeo/
│   ├── Defs.lean          # Type definitions (GPoint, predicates)
│   ├── Basic.lean         # Basic utility lemmas
│   └── Rules.lean         # 49 deduction rules as axioms
└── _check/
    └── test_gen.lean      # Generated verification targets
```

## Building

```bash
cd lean_geo
lake build
```

Requires Lean 4 v4.16.0 (specified in `lean-toolchain`).

## Usage from Python

The `geometry_agent.lean_bridge` module translates Python proof objects
into `.lean` source files and invokes `lake env lean` for verification:

```python
from geometry_agent.lean_bridge import ProcessLeanChecker

checker = ProcessLeanChecker(lean_project_dir="lean_geo")
result = checker.check_source(lean_source_code)
print(f"OK: {result.ok}")
```

## Adding a New Rule

1. Add the axiom declaration in `LeanGeo/Rules.lean`
2. Add the corresponding `RuleLeanSpec` in `geometry_agent/lean_bridge.py`
3. Add the `Rule` subclass in `geometry_agent/rules.py`

See [USAGE.md](../USAGE.md#adding-new-rules) for the complete checklist.