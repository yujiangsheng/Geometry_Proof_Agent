"""engine.py – Symbolic reasoning engine + proof verification layer (v0.12.0).

Layered architecture (de Bruijn criterion) within the 5-layer system::

    ┌─────────────────────────────────────────────────┐
    │              GeometryPipeline                    │
    │              (orchestrator)                      │
    └──────────┬──────────────────┬────────────────────┘
               │                  │
    ┌──────────▼──────────┐  ┌───▼──────────────────┐
    │   SymbolicEngine    │  │   ProofVerifier       │
    │   (Layer 1: 推理)   │  │   (Layer 2: 校验)    │
    ├─────────────────────┤  ├──────────────────────┤
    │ solve(assums, goal) │  │ verify(certificate)  │
    │  → ProofCertificate │  │  → VerificationResult│
    └──────────┬──────────┘  └───┬──────────────────┘
               │                  │
    ┌──────────┴──────────┐  ┌───┴──────────────────┐
    │ PythonSymbolicEngine│  │ LeanProofVerifier    │
    │ (rules + search)    │  │ (subprocess lean)    │
    └─────────────────────┘  ├─────────────────────┤
                              │ MockProofVerifier    │
                              │ (dev / test)         │
                              └─────────────────────┘

**Layer 1 – 符号引擎 (Symbolic Engine)**:
  Performs ALL deductive reasoning.  Produces a ``ProofCertificate``.

**Layer 2 – Lean 校验器 (Proof Verifier)**:
  Independently validates proof certificates.  Does NOT participate in
  reasoning – it only checks.

**ProofCertificate** is the formal contract between the two layers.

Author:  Jiangsheng Yu
License: MIT
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Protocol, Sequence, runtime_checkable

from .dsl import Fact, GeoState, Goal, Step
from .knowledge import KnowledgeStore, get_global_store
from .lean_bridge import MockLeanChecker
from .rules import Rule, default_rules
from .search import SearchConfig, SearchResult, beam_search

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  ProofCertificate — the contract between engine and verifier
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ProofCertificate:
    """Self-contained proof certificate produced by a symbolic engine.

    Contains everything a verifier needs to independently validate the
    proof, without access to the engine's internal state.

    Attributes
    ----------
    assumptions : list[Fact]
        The geometric premises (hypotheses).
    goal : Fact
        The statement to be proven.
    steps : list[Step]
        The derivation trace (each step: rule + premises → conclusion).
    success : bool
        Whether the engine found a valid proof.
    explored_nodes : int
        Number of search nodes explored (profiling).
    cache_hits : int
        Number of knowledge-cache hits (profiling).
    engine_name : str
        Identifier for the engine that produced this certificate.
    engine_version : str
        Version of the engine.
    timestamp : float
        When the certificate was produced.
    metadata : dict
        Arbitrary extra information (e.g. search config, timing).
    """
    assumptions: List[Fact]
    goal: Fact
    steps: List[Step]
    success: bool
    explored_nodes: int = 0
    cache_hits: int = 0
    engine_name: str = "unknown"
    engine_version: str = "0.0.0"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Serialisation ────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-friendly)."""
        return {
            "assumptions": [f.to_dict() for f in self.assumptions],
            "goal": self.goal.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "success": self.success,
            "explored_nodes": self.explored_nodes,
            "cache_hits": self.cache_hits,
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProofCertificate":
        """Deserialise from a plain dict."""
        return cls(
            assumptions=[Fact.from_dict(f) for f in d["assumptions"]],
            goal=Fact.from_dict(d["goal"]),
            steps=[Step.from_dict(s) for s in d.get("steps", [])],
            success=d["success"],
            explored_nodes=d.get("explored_nodes", 0),
            cache_hits=d.get("cache_hits", 0),
            engine_name=d.get("engine_name", "unknown"),
            engine_version=d.get("engine_version", "0.0.0"),
            timestamp=d.get("timestamp", 0.0),
            metadata=d.get("metadata", {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "ProofCertificate":
        return cls.from_dict(json.loads(s))

    # ── Backward-compatible conversion ───────────────────────

    def to_search_result(self) -> SearchResult:
        """Convert to ``SearchResult`` for backward compatibility.

        Reconstructs a ``GeoState`` from assumptions + steps.
        """
        state = GeoState(facts=set(self.assumptions), history=[])
        for step in self.steps:
            state.add_fact(step.conclusion_fact, via=step)
        return SearchResult(
            success=self.success,
            final_state=state,
            explored_nodes=self.explored_nodes,
            cache_hits=self.cache_hits,
        )

    @classmethod
    def from_search_result(
        cls,
        result: SearchResult,
        assumptions: List[Fact],
        goal: Fact,
        engine_name: str = "python",
        engine_version: str = "1.0.0",
    ) -> "ProofCertificate":
        """Create from an existing ``SearchResult``."""
        return cls(
            assumptions=assumptions,
            goal=goal,
            steps=list(result.final_state.history),
            success=result.success,
            explored_nodes=result.explored_nodes,
            cache_hits=result.cache_hits,
            engine_name=engine_name,
            engine_version=engine_version,
        )


# ═══════════════════════════════════════════════════════════════════════
#  SymbolicEngine — abstract protocol
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class SymbolicEngine(Protocol):
    """Abstract interface for any symbolic geometry reasoning engine.

    Implementations must be interchangeable: a ``PythonSymbolicEngine``
    and an ``OCamlSymbolicEngine`` can be used in the same pipeline.
    """

    @property
    def name(self) -> str:
        """Short identifier (e.g. ``"python"``, ``"ocaml"``)."""
        ...

    @property
    def version(self) -> str:
        """Semantic version string."""
        ...

    def solve(
        self,
        assumptions: List[Fact],
        goal: Fact,
        *,
        config: Optional[SearchConfig] = None,
    ) -> ProofCertificate:
        """Run proof search and return a self-contained certificate.

        The engine should NOT call into Lean during search.  Lean
        verification is performed *after* the engine returns, by a
        ``ProofVerifier`` in a separate layer.

        Parameters
        ----------
        assumptions : list[Fact]
            Geometric premises.
        goal : Fact
            Target proposition to prove.
        config : SearchConfig, optional
            Search hyper-parameters.  If None, the engine picks defaults.

        Returns
        -------
        ProofCertificate
            Self-contained proof (or failure indication).
        """
        ...


# ═══════════════════════════════════════════════════════════════════════
#  PythonSymbolicEngine — pure-Python implementation
# ═══════════════════════════════════════════════════════════════════════


class PythonSymbolicEngine:
    """Pure-Python symbolic engine using ``rules.py`` + ``search.py``.

    Performs ALL reasoning in Python without calling Lean.  Uses
    ``MockLeanChecker`` for internal step validation (trusts the rule
    engine).  The resulting ``ProofCertificate`` must be independently
    verified by a ``ProofVerifier``.

    This mirrors the standard *external prover + kernel verifier*
    pattern (de Bruijn criterion):  the engine may be fast but
    untrusted; the Lean kernel provides the trustworthy guarantee.

    Parameters
    ----------
    rules : list[Rule], optional
        Deduction rules. Defaults to ``default_rules()``.
    knowledge_store : KnowledgeStore, optional
        Proof cache for cross-problem reuse. Defaults to global store.
    parallel_workers : int
        CPU parallelism for beam expansion. 0 = auto.
    default_beam_width : int
        Default beam width when no ``SearchConfig`` is supplied.
    default_max_depth : int
        Default max depth when no ``SearchConfig`` is supplied.
    """

    _name = "python"
    _version = "1.0.0"

    def __init__(
        self,
        rules: Optional[List[Rule]] = None,
        knowledge_store: Optional[KnowledgeStore] = None,
        parallel_workers: int = 0,
        default_beam_width: int = 8,
        default_max_depth: int = 8,
    ) -> None:
        self.rules = rules or default_rules()
        self.store = knowledge_store or get_global_store()
        self.parallel_workers = parallel_workers
        self._default_bw = default_beam_width
        self._default_md = default_max_depth
        # Internal step checker — always mock (engine does not call Lean)
        self._step_checker = MockLeanChecker()

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def solve(
        self,
        assumptions: List[Fact],
        goal: Fact,
        *,
        config: Optional[SearchConfig] = None,
    ) -> ProofCertificate:
        """Run beam search and return a ProofCertificate."""
        if config is None:
            config = SearchConfig(
                beam_width=self._default_bw,
                max_depth=self._default_md,
                parallel_workers=self.parallel_workers,
            )

        t0 = time.time()
        state = GeoState(facts=set(assumptions), history=[])
        result = beam_search(
            init_state=state,
            goal=Goal(goal),
            rules=self.rules,
            checker=self._step_checker,
            config=config,
            knowledge_store=self.store,
        )
        elapsed = time.time() - t0

        cert = ProofCertificate.from_search_result(
            result=result,
            assumptions=assumptions,
            goal=goal,
            engine_name=self.name,
            engine_version=self.version,
        )
        cert.metadata["search_time_s"] = round(elapsed, 4)
        cert.metadata["beam_width"] = config.beam_width
        cert.metadata["max_depth"] = config.max_depth

        logger.info(
            "PythonEngine: %s in %.3fs  (nodes=%d, cache=%d)",
            "PROVED" if cert.success else "FAILED",
            elapsed, cert.explored_nodes, cert.cache_hits,
        )
        return cert


# ═══════════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════════


def make_engine(
    kind: str = "python",
    **kwargs,
) -> SymbolicEngine:
    """Create a symbolic engine by name.

    Parameters
    ----------
    kind : str
        ``"python"`` (only supported engine).
    **kwargs
        Forwarded to the engine constructor.

    Returns
    -------
    SymbolicEngine
    """
    if kind == "python":
        return PythonSymbolicEngine(**kwargs)  # type: ignore[return-value]
    else:
        raise ValueError(f"Unknown engine kind: {kind!r}  (expected 'python')")


# ═══════════════════════════════════════════════════════════════════════
#  VerificationResult
#  (formerly in verifier.py)
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of verifying a ``ProofCertificate``.

    Attributes
    ----------
    verified : bool
        ``True`` if the verifier accepts the proof.
    message : str
        Human-readable diagnostic (e.g. Lean error message on failure).
    verifier_name : str
        Which verifier produced this result.
    lean_source : str, optional
        The generated Lean 4 source code (for inspection / debugging).
    """
    verified: bool
    message: str = ""
    verifier_name: str = "unknown"
    lean_source: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict = {
            "verified": self.verified,
            "message": self.message,
            "verifier_name": self.verifier_name,
        }
        if self.lean_source is not None:
            d["lean_source"] = self.lean_source
        return d


# ═══════════════════════════════════════════════════════════════════════
#  ProofVerifier — abstract protocol
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class ProofVerifier(Protocol):
    """Abstract interface for proof verification."""

    @property
    def name(self) -> str:
        ...

    def verify(self, certificate: ProofCertificate) -> VerificationResult:
        ...


# ═══════════════════════════════════════════════════════════════════════
#  LeanProofVerifier — Lean 4 kernel verification
# ═══════════════════════════════════════════════════════════════════════


class LeanProofVerifier:
    """Verify proof certificates using the Lean 4 kernel."""

    _name = "lean4"

    def __init__(
        self,
        lean_project_dir: Optional[str | Path] = None,
        lean_exe: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self._lean_project_dir = lean_project_dir
        self._lean_exe = lean_exe
        self._timeout = timeout
        self._checker = None  # type: ignore

    @property
    def name(self) -> str:
        return self._name

    def _get_checker(self):
        if self._checker is None:
            from .lean_bridge import ProcessLeanChecker
            kwargs = {"timeout": self._timeout}
            if self._lean_project_dir:
                kwargs["lean_project_dir"] = self._lean_project_dir
            if self._lean_exe:
                kwargs["lean_exe"] = self._lean_exe
            self._checker = ProcessLeanChecker(**kwargs)
        return self._checker

    def verify(self, certificate: ProofCertificate) -> VerificationResult:
        if not certificate.success:
            return VerificationResult(
                verified=False,
                message="certificate indicates engine did not find a proof",
                verifier_name=self.name,
            )
        if not certificate.steps:
            return VerificationResult(
                verified=False,
                message="certificate has no proof steps",
                verifier_name=self.name,
            )
        from .lean_bridge import translate_full_proof
        try:
            lean_src = translate_full_proof(
                assumptions=certificate.assumptions,
                steps=certificate.steps,
                final_goal=certificate.goal,
            )
        except ValueError as exc:
            return VerificationResult(
                verified=False,
                message=f"translation error: {exc}",
                verifier_name=self.name,
            )
        checker = self._get_checker()
        check = checker._run_lean(lean_src)
        return VerificationResult(
            verified=check.ok,
            message=check.message,
            verifier_name=self.name,
            lean_source=lean_src,
        )

    def verify_step(
        self,
        step: Step,
        context_facts: Optional[FrozenSet[Fact]] = None,
    ) -> VerificationResult:
        checker = self._get_checker()
        dummy_state = GeoState(facts=set(context_facts or set()))
        check = checker.check_step(dummy_state, step)
        return VerificationResult(
            verified=check.ok,
            message=check.message,
            verifier_name=self.name,
        )


# ═══════════════════════════════════════════════════════════════════════
#  MockProofVerifier — fast mock for development / testing
# ═══════════════════════════════════════════════════════════════════════


class MockProofVerifier:
    """Always-pass verifier for development and batch training."""

    _name = "mock"

    @property
    def name(self) -> str:
        return self._name

    def verify(self, certificate: ProofCertificate) -> VerificationResult:
        if not certificate.success:
            return VerificationResult(
                verified=False,
                message="engine did not find proof (mock-verifier)",
                verifier_name=self.name,
            )
        return VerificationResult(
            verified=True,
            message="mock-verified",
            verifier_name=self.name,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Verifier Factory
# ═══════════════════════════════════════════════════════════════════════


def make_verifier(
    use_lean: bool = False,
    **kwargs,
) -> ProofVerifier:
    """Create a proof verifier.

    Parameters
    ----------
    use_lean : bool
        If ``True``, return a ``LeanProofVerifier``.
        If ``False``, return a ``MockProofVerifier``.
    """
    if use_lean:
        return LeanProofVerifier(**kwargs)  # type: ignore[return-value]
    return MockProofVerifier()  # type: ignore[return-value]
