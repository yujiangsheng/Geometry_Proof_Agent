"""pipeline.py – Multi-agent orchestration with layered engine/verifier.

Layered architecture:  *外部 Python/OCaml 引擎 + Lean 校验器*
(External symbolic engine + Lean verifier)

::

    ┌────────────────────────────────────────────────────┐
    │              GeometryPipeline (orchestrator)        │
    └──────────┬──────────────────────┬──────────────────┘
               │                      │
    ┌──────────▼──────────┐  ┌───────▼────────────────┐
    │  SymbolicEngine     │  │  ProofVerifier         │
    │  (Layer 1: 推理引擎) │  │  (Layer 2: Lean 校验)  │
    │  solve() →          │  │  verify() →            │
    │  ProofCertificate   │  │  VerificationResult    │
    └─────────────────────┘  └────────────────────────┘

Five specialised agents cooperate through a shared ``KnowledgeStore``:

+-------------------+----------------------------------------------------+
| Agent             | Responsibility                                     |
+===================+====================================================+
| ParserAgent       | Convert raw input into a typed ``Problem`` object  |
| PlannerAgent      | Decide search hyper-parameters per problem         |
| StepProposerAgent | Hold the rule inventory used during search         |
| CriticReflectAgent| Diagnose search failures and record patterns       |
| CurriculumAgent   | Rank problems by difficulty for self-training      |
+-------------------+----------------------------------------------------+

``GeometryPipeline`` wires them together in a single ``solve_structured``
call:  Parser → Planner → Engine.solve() → Verifier.verify() → Critic.

Author:  Jiangsheng Yu
License: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .dsl import Fact, GeoState, Goal
from .engine import PythonSymbolicEngine, ProofCertificate, SymbolicEngine, make_engine
from .knowledge import KnowledgeStore, get_global_store
from .lean_bridge import CheckResult, LeanChecker, make_checker
from .llm import (
    CRITIC_PROMPT,
    PARSE_NL_PROMPT,
    STRATEGY_PROMPT,
    LLMClient,
    LLMResponse,
    get_llm,
    narrate_theorem,
)
from .rag import GeometryRAG, RetrievalContext, get_rag
from .rules import Rule, default_rules
from .search import SearchConfig, SearchResult, beam_search
from .engine import (
    LeanProofVerifier,
    MockProofVerifier,
    ProofVerifier,
    VerificationResult,
    make_verifier,
)

logger = logging.getLogger(__name__)


@dataclass
class Problem:
    """Fully parsed geometry problem ready for the search engine.

    Attributes
    ----------
    assumptions : list[Fact]
        Given geometric facts (e.g. ``Parallel(A B C D)``).
    goal : Goal
        The statement to be proven.
    """
    assumptions: List[Fact]
    goal: Goal


# ── Agent: Parser ────────────────────────────────────────────────────

class ParserAgent:
    """Convert structured or natural-language input into a ``Problem``.

    **Skills**

    1. *Structured parsing* (``parse_structured``):
       Accepts a dict with ``"assumptions"`` (``list[Fact]``) and
       ``"goal"`` (``Fact``) keys and produces a ``Problem``.

    2. *Natural-language parsing* (``parse_nl``):
       Accepts a free-text geometry statement (e.g. "已知直线AB平行于
       直线CD，求证…") and uses a local LLM to extract structured
       ``Fact`` objects.

    3. *(Future) Diagram parsing*:
       Will accept a vectorised or raster diagram and extract geometric
       relations via a vision model.

    **Inputs / Outputs**

    * Input  : ``Dict`` with `assumptions` + `goal`, or ``str`` (natural language)
    * Output : ``Problem``
    """

    def __init__(self, llm: Optional[LLMClient] = None):
        self._llm = llm

    def parse_structured(self, payload: Dict) -> Problem:
        assumptions = payload["assumptions"]
        goal = payload["goal"]
        return Problem(assumptions=assumptions, goal=Goal(goal))

    def parse_nl(self, text: str) -> Problem:
        """Parse a natural-language geometry problem via LLM.

        Uses the local LLM to extract structured assumptions and goal
        from free-text input.  Requires an active Ollama instance.

        Parameters
        ----------
        text : str
            Natural-language geometry problem statement.

        Returns
        -------
        Problem

        Raises
        ------
        ValueError
            If the LLM output cannot be parsed into a valid Problem.
        ConnectionError
            If the LLM is not reachable.
        """
        llm = self._llm or get_llm()
        prompt = PARSE_NL_PROMPT.format(problem=text)
        resp = llm.chat(prompt, temperature=0.1)
        content = resp.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            content = "\n".join(lines).strip()

        import json as _json
        try:
            data = _json.loads(content)
        except _json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned invalid JSON: {content[:200]}…"
            ) from exc

        assumptions = [Fact.from_dict(f) for f in data["assumptions"]]
        goal_fact = Fact.from_dict(data["goal"])
        return Problem(assumptions=assumptions, goal=Goal(goal_fact))


# ── Agent: Planner ───────────────────────────────────────────────────

class PlannerAgent:
    """Select search hyper-parameters based on problem features.

    **Skills**

    1. *Complexity estimation*:
       Estimates problem difficulty from the assumption count and
       predicate diversity, then picks beam width and max depth.

    2. *Resource allocation*:
       Sets ``parallel_workers`` to utilise available CPU cores for
       beam expansion.

    3. *LLM strategy hints* (optional):
       When ``use_llm=True``, queries the local LLM for strategy
       recommendations on complex problems (> 6 assumptions).

    **Inputs / Outputs**

    * Input  : ``Problem``, optional ``workers`` count
    * Output : ``SearchConfig``

    **Heuristic table** (default, no LLM)

    +-------------------+------------+-----------+
    | # assumptions     | beam_width | max_depth |
    +===================+============+===========+
    | ≤ 6               | 8          | 6         |
    | > 6               | 16         | 10        |
    +-------------------+------------+-----------+
    """

    def __init__(self, llm: Optional[LLMClient] = None, use_llm: bool = False):
        self._llm = llm
        self._use_llm = use_llm

    def choose_search_config(
        self, problem: Problem, *, workers: int = 0,
    ) -> SearchConfig:
        fact_count = len(problem.assumptions)

        # Default heuristic
        if fact_count <= 6:
            cfg = SearchConfig(beam_width=8, max_depth=6, parallel_workers=workers)
        else:
            cfg = SearchConfig(beam_width=16, max_depth=10, parallel_workers=workers)

        # LLM hint for complex problems
        if self._use_llm and fact_count > 6:
            cfg = self._llm_hint(problem, cfg, workers)

        return cfg

    def _llm_hint(
        self, problem: Problem, default: SearchConfig, workers: int,
    ) -> SearchConfig:
        """Ask the LLM for strategy recommendations."""
        try:
            llm = self._llm or get_llm()
            predicates = sorted({f.predicate for f in problem.assumptions})
            prompt = STRATEGY_PROMPT.format(
                assumptions=", ".join(str(f) for f in problem.assumptions),
                goal=str(problem.goal.fact),
                n_facts=len(problem.assumptions),
                predicates=", ".join(predicates),
            )
            resp = llm.chat(prompt, temperature=0.1)
            content = resp.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                lines = [l for l in lines if not l.startswith("```")]
                content = "\n".join(lines).strip()
            import json as _json
            data = _json.loads(content)
            bw = int(data.get("beam_width", default.beam_width))
            md = int(data.get("max_depth", default.max_depth))
            # Clamp to sane ranges
            bw = max(4, min(bw, 64))
            md = max(4, min(md, 20))
            logger.info("LLM strategy hint: beam=%d depth=%d reason=%s",
                        bw, md, data.get("reason", "?"))
            return SearchConfig(beam_width=bw, max_depth=md, parallel_workers=workers)
        except Exception as exc:
            logger.warning("LLM strategy hint failed, using defaults: %s", exc)
            return default


# ── Agent: Step Proposer ─────────────────────────────────────────────

class StepProposerAgent:
    """Maintain the rule inventory used during forward-chaining search.

    **Skills**

    1. *Rule management*:
       Holds a list of ``Rule`` objects, each of which can generate
       candidate derivation steps from the current ``GeoState``.
       Default ruleset includes:

       - ``ParallelSymmetryRule``  — if Parallel(A,B,C,D) then Parallel(C,D,A,B)
       - ``ParallelTransitivityRule`` — transitive closure over Parallel
       - ``PerpSymmetryRule`` — if Perpendicular(A,B,C,D) then Perpendicular(C,D,A,B)
       - ``ParallelPerpTransRule`` — if Parallel(A,B,C,D) and Perp(A,B,E,F)
         then Perp(C,D,E,F)

    2. *(Future) Neural suggestion*:
       A trained policy network will rank / filter candidate steps
       before they enter the beam, dramatically pruning the search
       space.

    3. *(Future) Macro-rule application*:
       Will apply cached multi-step lemmas from ``KnowledgeStore`` as
       single expansion steps.

    **Inputs / Outputs**

    * Input  : optional ``rules`` override
    * Output : ``self.rules`` list consumed by ``beam_search``

    **Extension points**

    * Pass custom rules at construction time.
    * Add a ``score_candidates(state, candidates)`` method that prunes
      by a learned heuristic.
    """

    def __init__(self, rules: List[Rule] | None = None):
        self.rules = rules or default_rules()


# ── Agent: Critic / Reflect ──────────────────────────────────────────

class CriticReflectAgent:
    """Diagnose search failures and feed patterns to knowledge store.

    **Skills**

    1. *Post-mortem analysis*:
       After a search run completes, classifies the outcome:

       - ``"success"`` — goal reached
       - ``"no-search-progress"`` — zero nodes explored (likely an
         input or initialisation error)
       - ``"missing-rules-or-lemmas"`` — search explored nodes but
         could not reach the goal (the rule set may be incomplete)

    2. *Failure pattern recording*:
       Writes the diagnosed pattern into ``KnowledgeStore`` so that
       the ``CurriculumAgent`` can adjust difficulty and the evolution
       loop can decide which rules to add.

    3. *(Future) Proof repair suggestions*:
       Will propose auxiliary constructions or lemma candidates that
       could have bridged the gap, feeding them back to the Planner
       or StepProposer for retry.

    4. *(Future) Self-critique via LLM*:
       Will call a language model to explain *why* the proof failed
       and generate human-readable diagnostics.

    **Inputs / Outputs**

    * Input  : ``SearchResult``, target ``Fact``
    * Output : diagnostic string (``"success"`` | failure pattern)
    """

    def __init__(
        self,
        knowledge_store: Optional[KnowledgeStore] = None,
        llm: Optional[LLMClient] = None,
        use_llm: bool = False,
        use_rag: bool = False,
        rag: Optional[GeometryRAG] = None,
    ):
        self.store = knowledge_store
        self._llm = llm
        self._use_llm = use_llm
        self._use_rag = use_rag
        self._rag = rag

    def analyze(
        self,
        result: SearchResult,
        goal: Fact,
        assumptions: Optional[List[Fact]] = None,
        rules_desc: Optional[str] = None,
    ) -> str:
        if result.success:
            return "success"
        if result.explored_nodes == 0:
            pattern = "no-search-progress"
        else:
            pattern = "missing-rules-or-lemmas"
        if self.store is not None:
            self.store.record_failure_pattern(pattern)
        return pattern

    def diagnose_with_llm(
        self,
        result: SearchResult,
        goal: Fact,
        assumptions: List[Fact],
        rules_desc: str = "parallel_symmetry, parallel_transitivity, perp_symmetry, parallel_perp_trans",
    ) -> Optional[str]:
        """Use the LLM (optionally augmented with RAG) for failure diagnosis.

        When ``use_rag=True``, retrieves relevant geometry knowledge
        from the local RAG store and web search before prompting
        the LLM, providing richer context for analysis.

        Returns a natural-language analysis string, or ``None`` if
        the LLM is unavailable.
        """
        if not self._use_llm:
            return None
        try:
            llm = self._llm or get_llm()
            diagnosis = self.analyze(result, goal)

            base_prompt = CRITIC_PROMPT.format(
                assumptions=", ".join(str(f) for f in assumptions),
                goal=str(goal),
                nodes=result.explored_nodes,
                diagnosis=diagnosis,
                rules=rules_desc,
            )

            # RAG augmentation: retrieve relevant knowledge
            if self._use_rag:
                prompt = self._rag_augment_prompt(
                    base_prompt, goal, assumptions, diagnosis,
                )
            else:
                prompt = base_prompt

            resp = llm.chat(prompt, temperature=0.3)
            return resp.content.strip()
        except Exception as exc:
            logger.warning("LLM diagnosis failed: %s", exc)
            return None

    def _rag_augment_prompt(
        self,
        base_prompt: str,
        goal: Fact,
        assumptions: List[Fact],
        diagnosis: str,
    ) -> str:
        """Augment the critic prompt with RAG-retrieved context."""
        try:
            rag = self._rag or get_rag()
            ctx = rag.retrieve_for_failure(
                goal_predicate=goal.predicate,
                goal_args=goal.args,
                assumptions_preds=[f.predicate for f in assumptions],
                diagnosis=diagnosis,
            )
            if ctx.has_results:
                return ctx.augmented_prompt(base_prompt)
        except Exception as exc:
            logger.debug("RAG augmentation failed: %s", exc)
        return base_prompt


# ── Agent: Curriculum ────────────────────────────────────────────────

class CurriculumAgent:
    """Decide the difficulty distribution for self-training / evolution.

    **Skills**

    1. *Adaptive difficulty ramp*:
       Based on the total experience accumulated so far, suggests a
       target difficulty level for the next batch of synthetic problems.

       +-------------------+-------------------+
       | Experience count  | Suggested level   |
       +===================+===================+
       | < 50              | 2  (easy)         |
       | 50 – 199          | 3  (medium)       |
       | ≥ 200             | 5  (hard)         |
       +-------------------+-------------------+

    2. *(Future) Competence-based progression*:
       Will track per-predicate success rates and only increase
       difficulty for predicates the agent already handles well.

    3. *(Future) Failure-driven focus*:
       Will over-sample problem types that appear frequently in
       ``KnowledgeStore.top_failure_patterns()`` to target weaknesses.

    **Inputs / Outputs**

    * Input  : ``KnowledgeStore``
    * Output : integer difficulty level
    """

    def suggest_difficulty(self, store: KnowledgeStore) -> int:
        s = store.stats()
        if s.experience_total < 50:
            return 2
        if s.experience_total < 200:
            return 3
        return 5


# ── Pipeline ─────────────────────────────────────────────────────────

class GeometryPipeline:
    """End-to-end orchestration: Engine → Certificate → Verifier → Critic.

    Implements the *外部引擎 + Lean 校验器* (external engine + Lean
    verifier) layered architecture.  The symbolic engine does ALL
    reasoning in Python (or OCaml) without calling Lean.  The Lean
    verifier independently certifies the resulting proof.

    Wires five agents together with a shared ``KnowledgeStore`` so that
    every successful proof is cached (avoiding re-derivation) and every
    failure is diagnosed (driving curriculum evolution).

    **Lifecycle of ``solve_structured``**::

        1. Build Problem from assumptions + goal
        2. PlannerAgent → SearchConfig
        3. Engine.solve(…) → ProofCertificate          [Layer 1: 推理]
        4. Verifier.verify(cert) → VerificationResult  [Layer 2: 校验]
        5. CriticReflectAgent.analyze(…) → diagnosis
        6. KnowledgeStore.record_experience(…)
        7. Return SearchResult  (backward-compatible)

    **Lifecycle of ``solve_layered``**::

        Same as above, but returns ``(ProofCertificate,
        VerificationResult)`` – the full layered output.

    **Extension points**

    * Supply a custom ``engine`` (``PythonSymbolicEngine``,
      ``OCamlSymbolicEngine``, or your own implementation).
    * Supply a custom ``verifier`` (``LeanProofVerifier``,
      ``MockProofVerifier``).
    * Supply custom ``rules`` to change the deductive repertoire.
    * Supply a ``knowledge_store`` to share across multiple pipelines.
    """

    def __init__(
        self,
        checker: Optional[LeanChecker] = None,
        rules: List[Rule] | None = None,
        use_lean: bool = False,
        knowledge_store: Optional[KnowledgeStore] = None,
        parallel_workers: int = 0,
        engine: Optional[SymbolicEngine] = None,
        verifier: Optional[ProofVerifier] = None,
        llm: Optional[LLMClient] = None,
        use_llm: bool = False,
        use_rag: bool = False,
        rag: Optional[GeometryRAG] = None,
    ):
        self.store = knowledge_store or get_global_store()
        self._llm = llm
        self._use_llm = use_llm
        self._use_rag = use_rag
        self._rag = rag
        self.parser = ParserAgent(llm=llm)
        self.planner = PlannerAgent(llm=llm, use_llm=use_llm)
        self.proposer = StepProposerAgent(rules=rules)
        self.critic = CriticReflectAgent(
            knowledge_store=self.store, llm=llm, use_llm=use_llm,
            use_rag=use_rag, rag=rag,
        )
        self.curriculum = CurriculumAgent()
        self.parallel_workers = parallel_workers

        # ── Layer 1: Symbolic Engine ─────────────────────────
        # Default: PythonSymbolicEngine (pure Python, no Lean during search)
        if engine is not None:
            self.engine: SymbolicEngine = engine
        else:
            self.engine = PythonSymbolicEngine(
                rules=self.proposer.rules,
                knowledge_store=self.store,
                parallel_workers=parallel_workers,
            )

        # ── Layer 2: Proof Verifier ──────────────────────────
        # Default: MockProofVerifier unless use_lean=True
        if verifier is not None:
            self.verifier: ProofVerifier = verifier
        else:
            self.verifier = make_verifier(use_lean=use_lean)

        # Legacy checker (backward compat for direct check_step usage)
        self.checker: LeanChecker = checker or make_checker(use_lean=use_lean)

    # ── Layered API (new) ────────────────────────────────────

    def solve_layered(
        self,
        assumptions: List[Fact],
        goal: Fact,
    ) -> tuple[ProofCertificate, VerificationResult]:
        """Full layered pipeline: Engine → Certificate → Verifier.

        Returns
        -------
        (ProofCertificate, VerificationResult)
            The engine's proof certificate and the verifier's ruling.
        """
        problem = Problem(assumptions=assumptions, goal=Goal(goal))
        cfg = self.planner.choose_search_config(
            problem, workers=self.parallel_workers,
        )

        # ── Layer 1: symbolic engine produces a proof certificate ──
        certificate = self.engine.solve(assumptions, goal, config=cfg)

        # ── Layer 2: verifier independently checks the certificate ──
        verification = self.verifier.verify(certificate)

        # ── Post-processing: critic + experience recording ──
        result = certificate.to_search_result()
        diagnosis = self.critic.analyze(result, goal)

        # LLM-powered failure analysis (if enabled and proof failed)
        self._last_llm_diagnosis: Optional[str] = None
        if not certificate.success and self._use_llm:
            self._last_llm_diagnosis = self.critic.diagnose_with_llm(
                result, goal, assumptions,
            )

        # LLM-powered theorem narration (if enabled and proof verified)
        self._last_llm_narration: Optional[str] = None
        if certificate.success and self._use_llm:
            self._last_llm_narration = self._narrate(
                assumptions, goal, result, verification,
            )

        logger.info(
            "Layered solve – engine=%s verifier=%s  %s  (nodes=%d, cache=%d, verified=%s)",
            self.engine.name, self.verifier.name, diagnosis,
            certificate.explored_nodes, certificate.cache_hits,
            verification.verified,
        )

        self.store.record_experience(
            assumptions=assumptions,
            goal=goal,
            success=certificate.success,
            steps=list(result.final_state.history),
            explored_nodes=certificate.explored_nodes,
        )

        return certificate, verification

    # ── LLM narration helper ─────────────────────────────────

    def _narrate(
        self,
        assumptions: List[Fact],
        goal: Fact,
        result: SearchResult,
        verification: VerificationResult,
    ) -> Optional[str]:
        """Generate an LLM narration for a verified theorem."""
        from .semantic import fact_to_nl, proof_to_nl, theorem_to_lean

        try:
            # Build human-readable inputs
            assumptions_nl = "，".join(fact_to_nl(f, "zh") for f in assumptions)
            goal_nl = fact_to_nl(goal, "zh")

            steps = result.final_state.history
            if steps:
                step_lines = []
                for i, step in enumerate(steps, 1):
                    prems = "、".join(fact_to_nl(f, "zh") for f in step.premise_facts)
                    concl = fact_to_nl(step.conclusion_fact, "zh")
                    step_lines.append(f"  {i}. 由{prems}，得{concl}（规则：{step.rule_name}）")
                proof_steps_nl = "\n".join(step_lines)
            else:
                proof_steps_nl = "  （直接可得）"

            lean_code = theorem_to_lean(
                assumptions, goal,
                with_proof=True,
                proof_steps=steps if steps else None,
            )

            return narrate_theorem(
                assumptions_nl=assumptions_nl,
                goal_nl=goal_nl,
                proof_steps_nl=proof_steps_nl,
                lean_code=lean_code,
                verified=verification.verified,
                llm=self._llm,
            )
        except Exception as exc:
            logger.warning("LLM narration failed: %s", exc)
            return None

    # ── Backward-compatible API ──────────────────────────────

    def solve_structured(self, assumptions: List[Fact], goal: Fact) -> SearchResult:
        """Solve via the layered architecture, returning SearchResult.

        This method maintains backward compatibility: internally uses
        ``Engine.solve() → ProofCertificate → Verifier.verify()`` but
        returns the familiar ``SearchResult`` object.
        """
        certificate, verification = self.solve_layered(assumptions, goal)
        return certificate.to_search_result()

    def verify_full_proof(
        self,
        assumptions: Sequence[Fact],
        result: SearchResult,
        goal: Fact,
    ) -> CheckResult:
        """Run a *separate* Lean check over the entire proof chain.

        This is a legacy API.  Prefer ``solve_layered()`` which
        integrates verification into the pipeline automatically.
        """
        cert = ProofCertificate.from_search_result(
            result=result,
            assumptions=list(assumptions),
            goal=goal,
        )
        vr = LeanProofVerifier().verify(cert)
        return CheckResult(ok=vr.verified, message=vr.message)
