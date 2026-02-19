/-
  LeanGeo/Defs.lean
  Abstract Euclidean-geometry predicates.
  Milestone A: axiomatised (swap for mathlib-backed defs later).
-/

axiom GPoint : Type

-- line(A,B) ∥ line(C,D)
axiom Parallel : GPoint → GPoint → GPoint → GPoint → Prop

-- A, B, C are collinear
axiom Collinear : GPoint → GPoint → GPoint → Prop

-- A, B, C, D are concyclic
axiom Cyclic : GPoint → GPoint → GPoint → GPoint → Prop

-- line(A,B) ⊥ line(C,D)
axiom Perpendicular : GPoint → GPoint → GPoint → GPoint → Prop

-- M is the midpoint of segment AB
axiom IsMidpoint : GPoint → GPoint → GPoint → Prop

-- |AB| = |CD| (segment congruence)
axiom Cong : GPoint → GPoint → GPoint → GPoint → Prop

-- ∠ABC = ∠DEF
axiom EqAngle : GPoint → GPoint → GPoint → GPoint → GPoint → GPoint → Prop

-- △ABC ~ △DEF (similar triangles)
axiom SimTri : GPoint → GPoint → GPoint → GPoint → GPoint → GPoint → Prop

-- A lies on circle centred at O
axiom OnCircle : GPoint → GPoint → Prop

-- ───── New predicates (Tier 1–3) ─────────────────────────────────

-- △ABC ≅ △DEF (triangle congruence, vertex correspondence preserved)
axiom CongTri : GPoint → GPoint → GPoint → GPoint → GPoint → GPoint → Prop

-- line(A,B) is tangent to circle centred at O at point P
axiom Tangent : GPoint → GPoint → GPoint → GPoint → Prop

-- |AB|/|CD| = |EF|/|GH| (proportional segments)
axiom EqRatio : GPoint → GPoint → GPoint → GPoint →
                GPoint → GPoint → GPoint → GPoint → Prop

-- B lies strictly between A and C on a line
axiom Between : GPoint → GPoint → GPoint → Prop

-- ray AP bisects ∠BAC
axiom AngleBisect : GPoint → GPoint → GPoint → GPoint → Prop

-- lines AB, CD, EF are concurrent
axiom Concurrent : GPoint → GPoint → GPoint → GPoint → GPoint → GPoint → Prop

-- O is the circumcentre of △ABC
axiom Circumcenter : GPoint → GPoint → GPoint → GPoint → Prop

-- |PA| = |PB| (equidistant)
axiom EqDist : GPoint → GPoint → GPoint → Prop

-- area(△ABC) = area(△DEF)
axiom EqArea : GPoint → GPoint → GPoint → GPoint → GPoint → GPoint → Prop

-- (A,B;C,D) is a harmonic range
axiom Harmonic : GPoint → GPoint → GPoint → GPoint → Prop

-- P is the pole of line AB w.r.t. circle centred at O
axiom PolePolar : GPoint → GPoint → GPoint → GPoint → Prop

-- P' is the inversion of P w.r.t. circle (O, |OA|)
axiom InvImage : GPoint → GPoint → GPoint → GPoint → Prop

-- (A,B;C,D) = (E,F;G,H) (equal cross-ratios)
axiom EqCrossRatio : GPoint → GPoint → GPoint → GPoint →
                     GPoint → GPoint → GPoint → GPoint → Prop

-- line AB is the radical axis of circles centred at O₁, O₂
axiom RadicalAxis : GPoint → GPoint → GPoint → GPoint → Prop
