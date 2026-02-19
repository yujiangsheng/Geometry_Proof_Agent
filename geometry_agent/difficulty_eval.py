"""difficulty_eval.py – Fair difficulty evaluation agent for discovered theorems.

Scoring philosophy — Ceva-calibrated & fairness-oriented
---------------------------------------------------------
Calibrated so that well-known textbook theorems like Ceva's theorem
score **≈ 2/10**.  The formula now incorporates four independent
dimensions of mathematical value to produce a **fair** assessment:

Core formula
~~~~~~~~~~~~
    raw = N_distinct × quality × aux_factor × tier_factor × diversity_factor × density_factor
    score = 1 + 9 × raw / (10 + raw)       (saturation curve, 1–10)

where
    N_distinct       = # distinct substantive rules used (trivial rewrites excluded)
    quality          = 0.5 + 0.5 × nt_ratio   (penalises proof padding)
    aux_factor       = 1 + 0.3 × N_aux
    tier_factor      = 1 + 0.1 × (max_concept_tier − 1)
    diversity_factor = 1 + 0.15 × (N_families − 1)  (rewards cross-family breadth)
    density_factor   = 0.7 + 0.3 × knowledge_density  (penalises repetitive proofs)

Fairness properties
~~~~~~~~~~~~~~~~~~~
* A proof using 5 rules from 5 different families scores HIGHER than
  5 rules from 1 family (diversity_factor).
* A clean 5-step proof (5 substantive / 5 total) scores HIGHER than
  the same 5 rules padded with 10 trivial rewrites (quality).
* Proofs where every step uses a different rule score HIGHER than
  proofs repeating the same rule many times (density_factor).
* Proofs touching higher concept tiers (circle, projective) get a
  mild bonus reflecting extra conceptual depth (tier_factor).

Calibration examples
    Ceva (1 rule, 0 aux, tier 5, 2 fam):  raw ≈ 1.6  → score ≈ 2.3  初級
    5 rules clean, 3 fam, tier 3:         raw ≈ 7.8  → score ≈ 5.0  中等
    5 rules padded (50% trivial), 3 fam:  raw ≈ 5.8  → score ≈ 4.3  简单
    8 rules, 5 fam, tier 5, 1 aux:        raw ≈ 22   → score ≈ 7.2  较难

Author:  Jiangsheng Yu
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

if TYPE_CHECKING:
    from .dsl import Fact, Step

# ── Trivial / tautological rules (同义反复) ──────────────────────────
# Rules that carry zero mathematical content — they merely restate,
# reformat, or definitionally unpack what is already known.
# Three categories:
#   1. Symmetry/permutation  (8) — swap argument halves
#   2. Transitivity          (5) — chain identical relations
#   3. Definitional          (24) — unpack / repack a predicate definition
# Total: 37 trivial out of 56 rules; the remaining 19 are substantive.
# Names MUST match the `name` attribute of the corresponding Rule class.

_TRIVIAL_RULES: Set[str] = {
    # ── Symmetry / permutation rules (zero content) ──
    "parallel_symmetry",       # AB∥CD → CD∥AB
    "perp_symmetry",           # AB⊥CD → CD⊥AB
    "eq_angle_symm",           # EqAngle swap
    "cong_symm",               # Cong swap
    "harmonic_swap",           # Harmonic pairs swap
    "eqarea_sym",              # EqArea swap
    "eqratio_sym",             # EqRatio swap
    "cross_ratio_sym",         # EqCrossRatio swap
    # ── Transitivity rules (trivial one-step transfer) ──
    # Proofs that ONLY chain transitivity produce difficulty ≈ 0.
    "parallel_transitivity",   # AB∥CD ∧ CD∥EF → AB∥EF
    "parallel_perp_trans",     # AB∥CD ∧ CD⊥EF → AB⊥EF
    "cong_trans",              # Cong(A,B,C,D) ∧ Cong(C,D,E,F) → Cong(A,B,E,F)
    "eq_angle_trans",          # EqAngle transitivity
    "eqratio_trans",           # EqRatio transitivity
    # ── Definitional unpacking (同义反复 / tautological) ──────────────
    # Single-predicate rules that merely extract a property already
    # encoded in the definition of the input predicate.  Zero content.
    # Format conversions:
    "eqdist_from_cong",        # Cong(P,A,P,B) → EqDist(P,A,B)
    "eqdist_to_cong",          # EqDist(P,A,B) → Cong(P,A,P,B)
    # Midpoint / Between unpacks:
    "midpoint_collinear",      # Midpoint(M,A,B) → Collinear(A,M,B)
    "midpoint_between",        # Midpoint(M,A,B) → Between(A,M,B)
    "midpoint_cong",           # Midpoint(M,A,B) → Cong(A,M,M,B)
    "between_collinear",       # Between(A,B,C) → Collinear(A,B,C)
    # Circumcenter unpacks:
    "circumcenter_cong_ab",    # Circumcenter(O,A,B,C) → Cong(O,A,O,B)
    "circumcenter_cong_bc",    # Circumcenter(O,A,B,C) → Cong(O,B,O,C)
    "circumcenter_oncircle",   # Circumcenter(O,A,B,C) → OnCircle(O,A)
    # Tangent unpacks:
    "tangent_perp_radius",     # Tangent(A,B,O,P) → Perp(O,P,A,B)
    "tangent_oncircle",        # Tangent(A,B,O,P) → OnCircle(O,P)
    # Angle bisector unpack:
    "angle_bisect_eq_angle",   # AngleBisect(A,P,B,C) → EqAngle(B,A,P,P,A,C)
    # Projective unpacks:
    "harmonic_collinear",      # Harmonic(A,B,C,D) → Collinear(A,C,D)
    "inversion_collinear",     # InvImage(P',P,O,A) → Collinear(O,P,P')
    "pole_polar_perp",         # PolePolar(P,A,B,O) → Perp(O,P,A,B)
    "radical_axis_perp",       # RadicalAxis(A,B,O1,O2) → Perp(A,B,O1,O2)
    # Congruent-triangle unpacks:
    "congtri_side",            # CongTri → Cong (corresponding sides)
    "congtri_angle",           # CongTri → EqAngle (corresponding angles)
    "congtri_eqarea",          # CongTri → EqArea
    # Similar-triangle unpacks:
    "sim_tri_angle",           # SimTri → EqAngle (corresponding angles)
    "eqratio_from_simtri",     # SimTri → EqRatio (proportional sides)
    # ── Reverse-definitional packing ─────────────────────────────────
    # Re-pack defining properties into a compound predicate.
    "midpoint_from_cong_between",   # Between + Cong → Midpoint
    "circumcenter_from_eqdist",     # EqDist + EqDist → Circumcenter
    "angle_bisect_from_eq_angle",   # EqAngle → AngleBisect
}

# ── Geometric concept families ───────────────────────────────────────

_PRED_FAMILY: Dict[str, str] = {
    "Parallel":      "LINE",
    "Perpendicular": "LINE",
    "Collinear":     "LINE",
    "Between":       "LINE",
    "Midpoint":      "MIDPOINT",
    "AngleBisect":   "ANGLE",
    "Cong":          "METRIC",
    "EqAngle":       "ANGLE",
    "EqDist":        "METRIC",
    "EqRatio":       "METRIC",
    "EqArea":        "METRIC",
    "Cyclic":        "CIRCLE",
    "OnCircle":      "CIRCLE",
    "Circumcenter":  "CIRCLE",
    "Tangent":       "CIRCLE",
    "RadicalAxis":   "CIRCLE",
    "SimTri":        "SIMILARITY",
    "CongTri":       "SIMILARITY",
    "Concurrent":    "CONCURRENCY",
    "Harmonic":      "PROJECTIVE",
    "PolePolar":     "PROJECTIVE",
    "InvImage":      "PROJECTIVE",
    "EqCrossRatio":  "PROJECTIVE",
}

# ── Concept tier (for depth / tier_factor scoring) ───────────────────
_PRED_TIER: Dict[str, int] = {
    # Tier 1 — basic line relations
    "Collinear":     1,
    "Parallel":      1,
    "Perpendicular": 1,
    "Between":       1,
    # Tier 2 — midpoint / construction
    "Midpoint":      2,
    "AngleBisect":   2,
    # Tier 3 — metric / angle
    "Cong":          3,
    "EqAngle":       3,
    "EqDist":        3,
    "EqArea":        3,
    "EqRatio":       3,
    # Tier 4 — circle
    "Cyclic":        4,
    "OnCircle":      4,
    "Circumcenter":  4,
    "Tangent":       4,
    "RadicalAxis":   4,
    # Tier 5 — similarity / congruence / concurrency
    "SimTri":        5,
    "CongTri":       5,
    "Concurrent":    5,
    # Tier 6 — projective geometry
    "Harmonic":      6,
    "PolePolar":     6,
    "InvImage":      6,
    "EqCrossRatio":  6,
}

# ── Difficulty labels ────────────────────────────────────────────────

_LABELS_ZH: List[Tuple[float, str, str]] = [
    (1.5, "极易",     "Trivial"),
    (2.5, "初级",     "Elementary"),
    (4.0, "简单",     "Easy"),
    (5.5, "中等",     "Medium"),
    (7.0, "较难",     "Challenging"),
    (8.5, "困难",     "Hard"),
    (10., "高级",     "Advanced"),
]


def _label_for_score(score: float) -> Tuple[str, str]:
    """Return (zh_label, en_label) for a given numeric score."""
    for threshold, zh, en in _LABELS_ZH:
        if score <= threshold:
            return zh, en
    return _LABELS_ZH[-1][1], _LABELS_ZH[-1][2]


# ── Helper: extract all points from a Fact ───────────────────────────

def _points_of_fact(f: "Fact") -> Set[str]:
    """All point names in a Fact (assumes all args are point names)."""
    return set(f.args)


# ── Counting helpers ─────────────────────────────────────────────────

def _count_substantive_rules(steps: List["Step"]) -> Tuple[Set[str], int]:
    """Return (set of substantive rule names, count of substantive steps).

    Substantive = not in _TRIVIAL_RULES.
    """
    sub_rules: Set[str] = set()
    sub_steps = 0
    for s in steps:
        if s.rule_name not in _TRIVIAL_RULES:
            sub_rules.add(s.rule_name)
            sub_steps += 1
    return sub_rules, sub_steps


def _count_auxiliary_points(
    assumptions: List["Fact"],
    goal: "Fact",
    steps: List["Step"],
) -> Tuple[Set[str], int]:
    """Points in the proof that are NOT in the theorem statement."""
    stmt_points: Set[str] = _points_of_fact(goal)
    for f in assumptions:
        stmt_points |= _points_of_fact(f)

    proof_points: Set[str] = set()
    for s in steps:
        proof_points |= _points_of_fact(s.conclusion_fact)
        for p in s.premise_facts:
            proof_points |= _points_of_fact(p)

    aux = proof_points - stmt_points
    return aux, len(aux)


def _count_concept_families(
    assumptions: List["Fact"],
    goal: "Fact",
    steps: List["Step"],
) -> Tuple[Set[str], int]:
    """Count distinct concept families touched in the proof."""
    families: Set[str] = set()

    def _add(f: "Fact") -> None:
        fam = _PRED_FAMILY.get(f.predicate, f.predicate)
        families.add(fam)

    _add(goal)
    for f in assumptions:
        _add(f)
    for s in steps:
        _add(s.conclusion_fact)
        for p in s.premise_facts:
            _add(p)
    return families, len(families)


def _family_transitions(steps: List["Step"]) -> int:
    """Count cross-family transitions in the proof."""
    n = 0
    for s in steps:
        if s.rule_name in _TRIVIAL_RULES:
            continue
        concl_fam = _PRED_FAMILY.get(s.conclusion_fact.predicate,
                                     s.conclusion_fact.predicate)
        prem_fams = {
            _PRED_FAMILY.get(p.predicate, p.predicate)
            for p in s.premise_facts
        }
        if concl_fam not in prem_fams:
            n += 1
    return n


def _max_concept_tier(
    assumptions: List["Fact"],
    goal: "Fact",
    steps: List["Step"],
) -> int:
    """Highest concept tier reached in the proof."""
    preds: Set[str] = {goal.predicate}
    for f in assumptions:
        preds.add(f.predicate)
    for s in steps:
        preds.add(s.conclusion_fact.predicate)
    return max((_PRED_TIER.get(p, 1) for p in preds), default=1)


# ── Evaluation dataclass ─────────────────────────────────────────────

@dataclass
class DifficultyReport:
    """Full difficulty assessment of a theorem."""
    overall_score: float            # 1.0 – 10.0
    label_zh: str                   # e.g. "困难"
    label_en: str                   # e.g. "Hard"

    # Core indicators
    n_substantive_rules: int = 0    # distinct non-trivial rules
    n_substantive_steps: int = 0    # non-trivial proof steps
    n_auxiliary_points: int = 0     # points introduced in proof
    n_family_transitions: int = 0   # cross-family bridges
    n_concept_families: int = 0     # distinct families touched
    max_concept_tier: int = 0       # highest concept tier
    nontrivial_ratio: float = 0.0   # substantive / total steps
    knowledge_density: float = 0.0  # distinct_rules / total_steps

    # Raw score (before saturation mapping)
    raw_score: float = 0.0

    # Legacy (for display compatibility)
    n_steps: int = 0
    n_preds: int = 0
    n_rules: int = 0

    # Textual assessment
    assessment_zh: str = ""
    assessment_en: str = ""

    # Stars for HTML rendering (1-5)
    stars: int = 1

    def summary_zh(self) -> str:
        return (f"难度评分 {self.overall_score:.1f}/10 ({self.label_zh})"
                f" | {self.n_substantive_rules}种知识"
                f" {self.n_auxiliary_points}辅助点"
                f" tier{self.max_concept_tier}")

    def summary_en(self) -> str:
        return (f"Difficulty {self.overall_score:.1f}/10 ({self.label_en})"
                f" | {self.n_substantive_rules} rules,"
                f" {self.n_auxiliary_points} aux pts,"
                f" tier {self.max_concept_tier}")


# ── Main evaluator ───────────────────────────────────────────────────

def evaluate_difficulty(
    assumptions: List["Fact"],
    goal: "Fact",
    steps: List["Step"],
) -> DifficultyReport:
    """Evaluate the difficulty of a discovered theorem.

    Scoring formula (Ceva-calibrated):

        raw = N_distinct × (1 + 0.3 × N_aux) × tier_factor
        score = 1 + 9 × raw / (10 + raw)

    where tier_factor = 1 + 0.1 × (max_tier − 1).

    Ceva's theorem (1 distinct rule, tier 3) ≈ 2.0/10.
    """
    n_total = len(steps)

    # ── Core indicators ──
    sub_rules, sub_steps = _count_substantive_rules(steps)
    _, n_aux = _count_auxiliary_points(assumptions, goal, steps)
    families, n_families = _count_concept_families(assumptions, goal, steps)
    n_trans = _family_transitions(steps)
    tier = _max_concept_tier(assumptions, goal, steps)
    nt_ratio = sub_steps / max(n_total, 1)
    knowledge_density = len(sub_rules) / max(n_total, 1)

    n_distinct = len(sub_rules)

    # ── Raw score (fairness-oriented) ──
    # quality: penalises padding with trivial rewrites (0.5–1.0)
    quality = 0.5 + 0.5 * nt_ratio
    # aux_factor: rewards auxiliary constructions
    aux_factor = 1.0 + 0.3 * n_aux
    # tier_factor: mild bonus for higher concept tiers
    tier_factor = 1.0 + 0.1 * (tier - 1)
    # diversity_factor: rewards spanning multiple concept families
    diversity_factor = 1.0 + 0.15 * max(0, n_families - 1)
    # density_factor: penalises proofs that repeat the same rule many
    # times (low knowledge density).  A proof where every step uses a
    # different rule is maximally dense (1.0); one that just repeats
    # two rules 10 times each is penalised (0.7).
    # density_factor penalises proofs that repeat the same rule many times.
    # 0.7 base + 0.3 × kd: at kd=0 (all steps use same rule) → 0.7;
    # at kd=1.0 (every step uses a unique rule) → 1.0.
    density_factor = 0.7 + 0.3 * knowledge_density
    raw = n_distinct * quality * aux_factor * tier_factor * diversity_factor * density_factor

    # ── Saturation mapping → 1–10 ──
    # Michaelis-Menten style curve: score = 1 + 9 × raw / (10 + raw)
    #   raw=0 → 1.0  (minimum)
    #   raw=10 → 5.5  (half-max point — calibration anchor)
    #   raw→∞ → 10.0  (asymptotic maximum — never quite reached)
    if raw <= 0:
        overall = 1.0
    else:
        overall = 1.0 + 9.0 * raw / (10.0 + raw)
    overall = round(min(10.0, overall), 1)

    label_zh, label_en = _label_for_score(overall)
    stars = max(1, min(5, round(overall / 2)))

    # ── All predicates / rules for legacy fields ──
    all_preds: Set[str] = {goal.predicate}
    all_rules: Set[str] = set()
    for f in assumptions:
        all_preds.add(f.predicate)
    for s in steps:
        all_preds.add(s.conclusion_fact.predicate)
        all_rules.add(s.rule_name)

    # ── Textual assessment ──
    zh_parts: List[str] = []
    en_parts: List[str] = []

    # Distinct knowledge
    zh_parts.append(
        f"使用{n_distinct}种不同知识点"
        f"（{n_total}步证明中{sub_steps}步实质推理）"
    )
    en_parts.append(
        f"{n_distinct} distinct rules"
        f" ({sub_steps} substantive / {n_total} total steps)"
    )

    # Auxiliary points
    if n_aux == 0:
        zh_parts.append("无辅助点")
        en_parts.append("no auxiliary points")
    elif n_aux <= 2:
        zh_parts.append(f"引入{n_aux}个辅助点")
        en_parts.append(f"{n_aux} auxiliary point(s)")
    else:
        zh_parts.append(f"引入{n_aux}个辅助点（辅助构造复杂）")
        en_parts.append(f"{n_aux} auxiliary points (complex construction)")

    # Concept diversity
    if n_families >= 4:
        zh_parts.append(f"跨越{n_families}个概念族（知识面广）")
        en_parts.append(f"spans {n_families} concept families (broad)")
    elif n_families >= 2:
        zh_parts.append(f"涉及{n_families}个概念族")
        en_parts.append(f"{n_families} concept families")
    else:
        zh_parts.append("单一概念族")
        en_parts.append("single concept family")

    # Proof quality
    if nt_ratio >= 0.9:
        zh_parts.append("推理紧凑高效")
        en_parts.append("compact and efficient reasoning")
    elif nt_ratio < 0.5:
        zh_parts.append("含较多重写步骤")
        en_parts.append("contains many rewrite steps")

    # Knowledge density
    if knowledge_density >= 0.8:
        zh_parts.append("知识密度高（每步运用不同的知识点）")
        en_parts.append("high knowledge density (each step uses a different rule)")
    elif knowledge_density < 0.4:
        zh_parts.append("知识密度较低（有较多重复性推理步骤）")
        en_parts.append("low knowledge density (many repetitive reasoning steps)")

    # Concept depth
    tier_desc_zh = {
        1: "基础线关系",
        2: "中点/角平分线",
        3: "全等·等角·度量",
        4: "圆的性质",
        5: "三角形相似/全等",
        6: "射影几何",
    }
    tier_desc_en = {
        1: "basic line relations",
        2: "midpoint/angle bisector",
        3: "congruence/angle/metric",
        4: "circle properties",
        5: "triangle similarity/congruence",
        6: "projective geometry",
    }
    zh_parts.append(f"最高概念层级：{tier_desc_zh.get(tier, f'tier{tier}')}")
    en_parts.append(f"highest tier: {tier_desc_en.get(tier, f'tier {tier}')}")

    # Overall quality note
    if overall >= 7.0:
        zh_parts.append("综合难度高，涉及多领域深度推理")
        en_parts.append("high difficulty, deep cross-domain reasoning")
    elif overall >= 5.0:
        zh_parts.append("达到中等以上难度")
        en_parts.append("medium+ difficulty")
    elif overall >= 3.0:
        zh_parts.append("组合了若干知识点，但整体较简单")
        en_parts.append("combines several facts but overall simple")
    else:
        zh_parts.append("基础结果，难度较低")
        en_parts.append("basic result, low difficulty")

    assessment_zh = "；".join(zh_parts) + "。"
    assessment_en = "; ".join(en_parts) + "."

    return DifficultyReport(
        overall_score=overall,
        label_zh=label_zh,
        label_en=label_en,
        n_substantive_rules=n_distinct,
        n_substantive_steps=sub_steps,
        n_auxiliary_points=n_aux,
        n_family_transitions=n_trans,
        n_concept_families=n_families,
        max_concept_tier=tier,
        nontrivial_ratio=round(nt_ratio, 2),
        knowledge_density=round(knowledge_density, 3),
        raw_score=round(raw, 2),
        n_steps=n_total,
        n_preds=len(all_preds),
        n_rules=len(all_rules),
        assessment_zh=assessment_zh,
        assessment_en=assessment_en,
        stars=stars,
    )


# ── Value scoring (based on rule diversity) ──────────────────────────


def compute_value_score(n_rule_types: int) -> tuple[float, str, str]:
    """Compute a theorem's value score based on distinct knowledge points used.

    The more distinct rules a proof uses, the more valuable it is —
    it demonstrates deeper, cross-domain reasoning.

    Returns (score, label_zh, label_en) where score is in [0, 10].

    Calibration:
        1 rule  → 1.0  基础
        2 rules → 2.3  一般
        3 rules → 3.7  较有价值
        5 rules → 5.5  有价值
        7 rules → 7.0  高价值
        10 rules → 8.5  极高价值
    """
    if n_rule_types <= 0:
        return (0.0, "", "")

    # Saturation curve: value = 10 × n / (n + 8)
    # Grows quickly at first, saturates towards 10
    raw = float(n_rule_types)
    score = 10.0 * raw / (raw + 8.0)
    score = round(min(10.0, max(0.0, score)), 1)

    if score >= 8.0:
        label_zh, label_en = "极高价值", "Exceptional"
    elif score >= 6.5:
        label_zh, label_en = "高价值", "High Value"
    elif score >= 5.0:
        label_zh, label_en = "有价值", "Valuable"
    elif score >= 3.5:
        label_zh, label_en = "较有价值", "Moderate"
    elif score >= 2.0:
        label_zh, label_en = "一般", "Basic"
    else:
        label_zh, label_en = "基础", "Elementary"

    return (score, label_zh, label_en)
