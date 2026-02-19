import LeanGeo

variable (A : GPoint)
variable (B : GPoint)
variable (C : GPoint)
variable (D : GPoint)
variable (E : GPoint)
variable (F : GPoint)

theorem geo_theorem (h0 : Parallel A B C D) (h1 : Parallel C D E F) : Parallel A B E F :=
  let s0 := parallel_trans A B C D E F h0 h1
  s0
