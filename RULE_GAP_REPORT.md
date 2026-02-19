# Rule Gap Report (Geometry Knowledge)

Date: 2026-02-18

This report evaluates the current deduction-rule system from three angles:
1. Predicate reachability
2. Converse/bidirectional coverage
3. Cross-family bridge strength

## 1) Score Summary

- Predicate reachability: **100.0 / 100** (23/23 predicates have at least one production rule)
- Converse coverage: **28.6 / 100** (12 bidirectional predicate pairs / 42 observed pairs; baseline 16.7)
- Cross-family bridge strength: **63.8 / 100** (11 strong bridge rules among 69 total; baseline 35.7)

Interpretation:
- Coverage is broad (no completely unreachable predicate).
- Converse closure improved substantially, but still has room to grow.
- Cross-family bridges are now strong and materially denser than baseline.

## 1.1 Checklist Completion (patched)

Status: **completed** (P1/P2/P3 all implemented in `rules.py`).

Post-patch metrics:

- Rule count: **69** (from 56)
- Predicate reachability: **100.0 / 100** (unchanged, full)
- Converse coverage: **28.6 / 100** (from 16.7)
- Cross-family bridge strength: **63.8 / 100** (from 35.7)
- Strong bridge rules (>=3 families): **11/69** (from 5/56)

P1 brittle-predicate completion check:
- AngleBisect: 2 producers
- Circumcenter: 2 producers
- Concurrent: 3 producers
- CongTri: 2 producers
- Cyclic: 2 producers
- EqDist: 3 producers
- Harmonic: 2 producers
- InvImage: 2 producers
- Midpoint: 2 producers
- PolePolar: 2 producers

## 2) Findings

### A. Reachability is complete
All 23 predicates currently used by the engine can be produced by at least one rule.
No hard-dead predicate was detected.

### B. Converse-like rules exist but are not dense
Detected 12 explicit converse-like rule names:
- angle_bisect_from_eq_angle
- circumcenter_from_eqdist
- congtri_from_sim_cong
- cross_ratio_from_harmonic
- cyclic_from_eq_angle
- eqdist_from_cong
- eqdist_to_cong
- eqratio_from_simtri
- inv_image_from_self
- midpoint_from_cong_between
- pole_polar_from_tangents
- radical_axis_from_common_points

Predicate-level bidirectional edge coverage is now **28.6%** (up from 16.7%).

### C. Bridge strength is improved
Strong bridge coverage is now **11/69** (touching >=3 concept families), up from 5/56.
Bridge-oriented additions in P3 significantly improved cross-family connectivity.

## 3) Thin Spots (by production redundancy)

No former brittle predicate remains at a single producer in the patched rule set.
Current low-redundancy frontier is predicates with exactly **two** producers.

Implication:
- Reachability is no longer single-rule fragile for the previous P1 list.
- Robustness can still improve by increasing independent derivation paths beyond two.

## 4) Priority Recommendations

### Priority P1 (highest)
Consolidate and stress-test new converse/bridge rules under larger evolution runs,
and monitor whether they are selected in successful proof chains.

### Priority P2
Continue increasing converse closure on high-frequency edges:
- EqAngle <-> Cyclic
- Cong <-> EqDist
- Midpoint <-> Between/Collinear/Cong combinations

### Priority P3
Add more 3-family+ bridge rules that connect:
- CIRCLE <-> METRIC <-> ANGLE
- PROJECTIVE <-> CIRCLE <-> LINE
- SIMILARITY <-> METRIC <-> CONCURRENCY

## 5) Conclusion

Are the rules “comprehensive”?
- **Engineering perspective:** mostly yes (broad and usable).
- **Formal completeness perspective:** not yet.

Current system is strong for theorem discovery, with major gap closure complete;
remaining work is optimization of converse density and further bridge redundancy.
