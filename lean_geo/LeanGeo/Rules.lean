/-
  LeanGeo/Rules.lean
  Deduction rules as axioms.
  Each axiom corresponds to exactly one Python-side `Rule`.
  Replace with proved lemmas once mathlib integration is done.
-/
import LeanGeo.Defs

-- ───── Parallel ──────────────────────────────────────────────
axiom parallel_symm :
  ∀ (A B C D : GPoint), Parallel A B C D → Parallel C D A B

axiom parallel_trans :
  ∀ (A B C D E F : GPoint),
    Parallel A B C D → Parallel C D E F → Parallel A B E F

-- ───── Perpendicular ────────────────────────────────────────
axiom perp_symm :
  ∀ (A B C D : GPoint), Perpendicular A B C D → Perpendicular C D A B

axiom parallel_perp_trans :
  ∀ (A B C D E F : GPoint),
    Parallel A B C D → Perpendicular C D E F → Perpendicular A B E F

-- ───── Collinear ────────────────────────────────────────────
axiom collinear_perm12 :
  ∀ (A B C : GPoint), Collinear A B C → Collinear B A C

axiom collinear_perm23 :
  ∀ (A B C : GPoint), Collinear A B C → Collinear A C B

axiom collinear_cycle :
  ∀ (A B C : GPoint), Collinear A B C → Collinear B C A

-- ───── Midpoint → Collinear ─────────────────────────────────
axiom midpoint_collinear :
  ∀ (M A B : GPoint), IsMidpoint M A B → Collinear A M B

-- ───── Cyclic ───────────────────────────────────────────────
axiom cyclic_perm :
  ∀ (A B C D : GPoint), Cyclic A B C D → Cyclic B C D A

axiom cyclic_inscribed_angle :
  ∀ (A B C D : GPoint), Cyclic A B C D → EqAngle B A C B D C

-- ───── Congruence ───────────────────────────────────────────
axiom cong_symm :
  ∀ (A B C D : GPoint), Cong A B C D → Cong C D A B

axiom cong_trans :
  ∀ (A B C D E F : GPoint),
    Cong A B C D → Cong C D E F → Cong A B E F

-- ───── Midpoint → Cong ──────────────────────────────────────
axiom midpoint_cong :
  ∀ (M A B : GPoint), IsMidpoint M A B → Cong A M M B

-- ───── Midsegment theorem ───────────────────────────────────
axiom midsegment_parallel :
  ∀ (M N A B C : GPoint),
    IsMidpoint M A B → IsMidpoint N A C → Parallel M N B C

-- ───── EqAngle ──────────────────────────────────────────────
axiom eq_angle_symm :
  ∀ (A B C D E F : GPoint),
    EqAngle A B C D E F → EqAngle D E F A B C

axiom eq_angle_trans :
  ∀ (A B C D E F G H I : GPoint),
    EqAngle A B C D E F → EqAngle D E F G H I → EqAngle A B C G H I

-- ───── Perpendicular bisector ───────────────────────────────
axiom perp_bisector_cong :
  ∀ (M A B C : GPoint),
    IsMidpoint M A B → Perpendicular C M A B → Cong C A C B

-- ───── Isosceles triangle ───────────────────────────────────
axiom isosceles_base_angle :
  ∀ (A B C : GPoint),
    Cong A B A C → EqAngle A B C A C B

-- ───── Converse of perpendicular bisector ───────────────────
axiom cong_perp_bisector :
  ∀ (C A B M : GPoint),
    Cong C A C B → IsMidpoint M A B → Perpendicular C M A B

-- ───── Parallel alternate interior angles ───────────────────
axiom parallel_alternate_angle :
  ∀ (A B C D X : GPoint),
    Parallel A B C D → Collinear A X C → EqAngle B A X D C X

-- ───── Cyclic chord angle ───────────────────────────────────
axiom cyclic_chord_angle :
  ∀ (A B C D : GPoint), Cyclic A B C D → EqAngle A B D A C D

-- ───── Midsegment similar triangle ─────────────────────────
axiom midsegment_sim_tri :
  ∀ (M N A B C : GPoint),
    IsMidpoint M A B → IsMidpoint N A C → SimTri A M N A B C

-- ───── Similar triangle → equal angles ─────────────────────
axiom sim_tri_angle :
  ∀ (A B C D E F : GPoint),
    SimTri A B C D E F → EqAngle B A C E D F

-- ───── Similar triangle + congruent side → congruent triangle
axiom sim_tri_cong :
  ∀ (A B C D E F : GPoint),
    SimTri A B C D E F → Cong A B D E → Cong A C D F

-- ═══════════════════════════════════════════════════════════════════
-- NEW RULES for the 14 newly added geometric relations.
-- ═══════════════════════════════════════════════════════════════════

-- ───── CongTri (triangle congruence) ────────────────────────────
axiom congtri_side :
  ∀ (A B C D E F : GPoint),
    CongTri A B C D E F → Cong A B D E

axiom congtri_angle :
  ∀ (A B C D E F : GPoint),
    CongTri A B C D E F → EqAngle B A C E D F

axiom congtri_from_sim_cong :
  ∀ (A B C D E F : GPoint),
    SimTri A B C D E F → Cong A B D E → CongTri A B C D E F

axiom congtri_eqarea :
  ∀ (A B C D E F : GPoint),
    CongTri A B C D E F → EqArea A B C D E F

-- ───── Tangent (line tangent to circle) ─────────────────────────
axiom tangent_perp_radius :
  ∀ (A B O P : GPoint),
    Tangent A B O P → Perpendicular O P A B

axiom tangent_oncircle :
  ∀ (A B O P : GPoint),
    Tangent A B O P → OnCircle O P

-- ───── EqRatio (proportional segments) ──────────────────────────
axiom eqratio_from_simtri :
  ∀ (A B C D E F : GPoint),
    SimTri A B C D E F → EqRatio A B D E A C D F

axiom eqratio_sym :
  ∀ (A B C D E F G H : GPoint),
    EqRatio A B C D E F G H → EqRatio E F G H A B C D

axiom eqratio_trans :
  ∀ (A B C D E F G H I J K L : GPoint),
    EqRatio A B C D E F G H →
    EqRatio E F G H I J K L →
    EqRatio A B C D I J K L

-- ───── Between (ordered collinearity) ───────────────────────────
axiom between_collinear :
  ∀ (A B C : GPoint), Between A B C → Collinear A B C

axiom midpoint_between :
  ∀ (M A B : GPoint), IsMidpoint M A B → Between A M B

-- ───── AngleBisect ──────────────────────────────────────────────
axiom angle_bisect_eq_angle :
  ∀ (A P B C : GPoint),
    AngleBisect A P B C → EqAngle B A P P A C

axiom angle_bisect_eqratio :
  ∀ (A P B C : GPoint),
    AngleBisect A P B C → Between B P C →
    EqRatio B P P C A B A C

-- ───── Concurrent (medians) ─────────────────────────────────────
axiom medians_concurrent :
  ∀ (A B C D E F_pt : GPoint),
    IsMidpoint D B C → IsMidpoint E A C → IsMidpoint F_pt A B →
    Concurrent A D B E C F_pt

-- ───── Circumcenter ─────────────────────────────────────────────
axiom circumcenter_cong_ab :
  ∀ (O A B C : GPoint),
    Circumcenter O A B C → Cong O A O B

axiom circumcenter_cong_bc :
  ∀ (O A B C : GPoint),
    Circumcenter O A B C → Cong O B O C

axiom circumcenter_oncircle :
  ∀ (O A B C : GPoint),
    Circumcenter O A B C → OnCircle O A

-- ───── EqDist (equidistant) ─────────────────────────────────────
axiom eqdist_from_cong :
  ∀ (P A B : GPoint),
    Cong P A P B → EqDist P A B

axiom eqdist_to_cong :
  ∀ (P A B : GPoint),
    EqDist P A B → Cong P A P B

-- ───── EqArea ───────────────────────────────────────────────────
axiom eqarea_sym :
  ∀ (A B C D E F : GPoint),
    EqArea A B C D E F → EqArea D E F A B C

-- ───── Harmonic ─────────────────────────────────────────────────
axiom harmonic_swap :
  ∀ (A B C D : GPoint),
    Harmonic A B C D → Harmonic B A D C

axiom harmonic_collinear :
  ∀ (A B C D : GPoint),
    Harmonic A B C D → Collinear A C D

-- ───── PolePolar ────────────────────────────────────────────────
axiom pole_polar_perp :
  ∀ (P A B O : GPoint),
    PolePolar P A B O → Perpendicular O P A B

axiom pole_polar_tangent :
  ∀ (P A B O : GPoint),
    PolePolar P A B O → OnCircle O A → Tangent P A O A

-- ───── InvImage (circle inversion) ──────────────────────────────
axiom inversion_collinear :
  ∀ (P' P O A : GPoint),
    InvImage P' P O A → Collinear O P P'

axiom inversion_circle_fixed :
  ∀ (P' P O A : GPoint),
    InvImage P' P O A → OnCircle O P → OnCircle O P'

-- ───── EqCrossRatio ─────────────────────────────────────────────
axiom cross_ratio_sym :
  ∀ (A B C D E F G H : GPoint),
    EqCrossRatio A B C D E F G H → EqCrossRatio E F G H A B C D

axiom cross_ratio_from_harmonic :
  ∀ (A B C D E F G H : GPoint),
    Harmonic A B C D → Harmonic E F G H →
    EqCrossRatio A B C D E F G H

-- ───── RadicalAxis ──────────────────────────────────────────────
axiom radical_axis_perp :
  ∀ (A B O₁ O₂ : GPoint),
    RadicalAxis A B O₁ O₂ → Perpendicular A B O₁ O₂
