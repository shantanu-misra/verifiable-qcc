# dfthcsos (Benchmark III, FeMoCo angle-loading QROMs)

Score = `lambda_eff * T_total * Q_total` (`data/verifiers/dfthcsos/SCORING_README.md`). Baseline 3,355,599,588
(lambda_eff 49.301369, T 93,237, Q 730; brot 12, J 8, ten plain QROMs of 3,974 Toffolis). Everything below that is
a metric was computed with the verifier's own functions (`tasks/dfthcsos/dfthc_tools.py` imports
`data/verifiers/dfthcsos/scoreinfra/*` read-only and reproduces the sample's 49.301369 / 18.640 mHa exactly);
scores quoted as verified come from `runs/dfthcsos_*.json`.

## 1. What the cost is made of

`T = ANGLES (forward + reverse, simulator count) + outer 846 + inner + 2*75*(2 brot - 1)`,
`Q = 90 persistent + J*brot + max(outer_temp 370, inner_temp, angles_temp)`.

| brot | inner Toffolis | givens | T_heur | inner temp | temp used |
|---|---|---|---|---|---|
| 12 | 9,461 | 3,450 | 13,757 | 544 | 544 |
| 10 | 9,461 | 2,850 | 13,157 | 544 | 544 |
| 8 | 9,461 | 2,250 | 12,557 | 544 | 544 |
| 7 | 16,989 | 1,950 | 19,785 | 273 | **370** (outer) |
| 6 | 16,989 | 1,650 | 19,485 | 273 | 370 |

The "cliff" at brot <= 7 is the inner QROAM dropping one level once `brot * 75 < 540`; it saves 174 temp
qubits but costs +7.5k inner Toffolis, and the outer PREPARE's 370 temporaries are the floor (not 272 as the
forum table assumed). Below 7 nothing improves.

## 2. Circuit: two-bank XOR QROM (`build_qrom.py`)

Address `a = 2g + a0`. One `unary_iteration` over the 11-bit group `g` (1,988 groups) XORs row `2g` into the word
registers and `row[2g] ^ row[2g+1]` into a W = J*brot scratch register; W Toffolis `CCX(a0, S_i, word_i)` pick the
odd row; kickmix's measurement-based `del_lookup` erases the scratch (88 Toffolis for 2^11 addresses). Measured per
circuit: 1,990 + W + 88 Toffolis, 12 + 2W + 11 qubits (119 at W = 48), simulator count = static count. The 128-qubit
cap gives W <= 52: J = 4 @ brot 12, 5 @ 10, 6 @ 8, 7 @ 7. More banks need 3W scratch and lose (J <= 3).

| design | K | ANGLES T (2 passes) | T | Q | cost @ lambda_eff 49.30 | ratio |
|---|---|---|---|---|---|---|
| baseline J8 brot12 | 10 | 79,480 | 93,237 | 730 | 3.356e9 | 1.00 |
| 2-bank J4 brot12 | 19 | 80,788 | 94,545 | 682 | 3.179e9 | 1.056 |
| 2-bank J5 brot10 | 15 | 63,840 | 76,997 | 684 | 2.597e9 | 1.29 |
| 2-bank J6 brot8 | 13 | 55,276 | 67,833 | 682 | 2.281e9 | 1.47 |
| 2-bank J7 brot7 | 11 | 46,794 | 66,579 | 509 | 1.671e9 | 2.01 |

So the width lever is binary in practice: brot 10 -> 8 is worth 13 %, brot 8 -> 7 is worth 36 %.

## 3. Accuracy: why plain rounding fails below brot 12

Baseline factorisation, verifier metrics (one-body rows = eigenvectors of h1' of the rounded two-body X, rounded):

| | error_1 | error_2 | Lambda | E_gap | lambda_eff |
|---|---|---|---|---|---|
| unrounded | 0 | **16.133** | 308.4485 | 3.958019 | 49.254686 |
| brot 12 | 3.185 | 18.640 | 308.4559 | 3.965477 | 49.301369 |
| brot 11 | 6.480 | 25.035 | | 3.968 | 49.319 |
| brot 10 | 13.969 | 40.015 | | 4.017 | 49.622 |
| brot 9 | 25.160 | 76.672 | | 4.222 | 50.883 |
| brot 8 | 52.351 | 155.383 | | 4.841 | 54.512 |
| brot 7 | 117.042 | 302.705 | | 6.330 | 62.407 |

(mHa). The intrinsic two-body error already uses 16.1 of the 20 mHa; rounding adds ~9.4 mHa in quadrature per
halving of the step. Quantisation noise also inflates the negative spectrum of h1' (negmass 11.276 at baseline), which
raises E_gap and lambda_eff (+27 % at brot 7). E_gap must stay > 3.914898: negmass may not drop below 11.251.
Lambda_2 = 295.6 of Lambda; the per-rho O(60) gauge (w -> wO, W -> O^T W leaves everything but Lambda_2 invariant)
is already better in the baseline (eigen gauge 328, random 353), so no free Lambda there.

The Givens tree is not a lever: the baseline's binomial tree already has the smallest noise amplification
(mean sum_t r_t^2 = 6.3, where r_t is the parent amplitude rotated by edge t; k-ary heaps 7.0-7.4, path/star 37).

## 4. Quantisation-aware refit (`refit.py`)

Exact verifier objective on the 2,926-pair symmetric matricisation (H = M M^T with M = s(X^(rho c)), 0.45 s per
loss+gradient natively; gradients checked against finite differences). Findings, in order:

- Continuous L-BFGS on (angles, weights) from the baseline: 16.133 -> 15.69 mHa after 400 iterations. The baseline
  factorisation has essentially no slack; the intrinsic error stays ~15.7 mHa, so quantisation may add only ~12 mHa
  in quadrature.
- Weight-only refit against frozen brot-10 words: 40.0 -> 30.4 mHa (and lambda_eff 49.6 -> 50.5: negmass and
  Lambda drift up). Not enough on its own.
- STE Adam (lr 2e-4 on angles) diverged (40 -> 62 mHa): the quantised forward pass makes normalised steps useless.
- Exact coordinate descent on the words (best of 150 +-1 single-word moves per vector, rank-2 residual update,
  ~2 min per sweep): 62 -> 39 -> 31 mHa in two sweeps; the strongest single tool.
- Progressive quantisation (freeze 1/16 of the rows at their rounded words, re-optimise the rest + weights with
  30 L-BFGS iterations, repeat): the error creeps from 16.2 to 22.1 mHa over 16 batches at brot 10 (it holds the
  frozen rows' noise in the free rows); polish then gives 20.8 after one sweep.
- One-body rows: plain rounding of the h1' eigenvectors gives 14 / 52 / 117 mHa at brot 10 / 8 / 7. Encoding them
  in decreasing |lambda| with error feedback (row b targets e_b - sum_a (lambda_a/lambda_b) d_a[b] e_a, so only
  lambda_min(a, b) times the rounding error survives per pair) gives 5.0 / 24 / 50 mHa, and coordinate descent with
  continuous "retarget" moves (v' = normalise(v - s/lambda P_perp E v), rounded) 4.2 / 12.6 / 27.8 mHa after 8
  sweeps. error_1 is therefore not the binding gate at brot 8; at brot 7 it is still 1.4x over.

- Unfreeze/refreeze rounds (1/8 of the rows made continuous again, re-optimised, re-frozen in 4 sub-batches):
  19.98 -> 19.81 mHa over 8 rounds; polish sweeps on top: 19.35, 19.06. The words are what limit the error now.
- brot 8 with the same pipeline (32 batches): 60.1 mHa after the progressive pass, 46.8 after three polish sweeps and
  a weight refit, 42.1 after a second round (error_1 18.8 with the feedback encoder). The quantisation noise is 4x
  brot 10's and the compensation mechanisms recover ~2x: **brot 8 (and 7) are out of reach with the baseline
  factorisation**; the 2x route needs a factorisation trained for coarse angles (lower intrinsic error, sparser rows).
- lambda_eff lever: the refit lets negmass (the negative spectrum of h1') drift up, which raises E_gap and lambda_eff.
  A linear negmass penalty (alpha 1e-4) cut lambda_eff 49.98 -> 48.54 for +0.05 mHa but overshot the E_gap floor
  (3.840 < 3.914898); an L1 penalty on w (mu) lowers Lambda_2 slightly but raises negmass, net worse. Steering E_gap to
  a target just above the floor with a quadratic penalty alpha (negmass - target)^2 lands exactly: E_gap 3.918016,
  lambda_eff 49.023 (-1.9 %), error_2 +0.06 mHa. The E_gap floor is what bounds lambda_eff for this factorisation.
  Pitfall: the word polish linearises the penalty; with the quadratic target a coefficient linearised once per sweep
  went stale after negmass drifted and one sweep traded 18 -> 102 mHa of real error for phantom penalty gain. The
  polish now re-linearises every 200 vectors, and the final run polishes without the penalty and re-steers E_gap in
  the weight refit afterwards.
- Final brot-10 pipeline: prog (16 batches) -> 2 polish sweeps -> w refit -> 8 refine rounds -> 2 x (3 polish sweeps +
  w refit) -> E_gap-targeted w refit -> 2 polish sweeps (2 moves per vector) -> targeted w refit -> one-body feedback +
  8 CD sweeps: error_2 18.079 mHa, error_1 2.335 mHa, E_gap 3.918090, lambda_eff 49.0228 (about 75 min of native CPU).

## 5. Results (verified, `runs/dfthcsos_*.json`)

| circuits | brot | J | words | lambda_eff | T | Q | score | ratio | run |
|---|---|---|---|---|---|---|---|---|---|
| `dfthcsos_3178937006_028667e2` | 12 | 4 | baseline (plain rounding) | 49.301369 | 94,545 | 682 | 3,178,937,005.85 | 1.0556 | `dfthcsos_3178937006_028667e2_full.json` |

| `dfthcsos_2632125690_c68ca4a8` | 10 | 5 | refit v1 (prog 16 batches + 2 polish sweeps + w refit) | 49.977752 | 76,997 | 684 | 2,632,125,689.76 | 1.2749 | `dfthcsos_2632125690_c68ca4a8_full.json` |
| `dfthcsos_2581833307_5ff495ce` | 10 | 5 | refit v2 (+ refine, polish, E_gap-targeted w refit) | 49.022820 | 76,997 | 684 | 2,581,833,307.18 | **1.2997** | `dfthcsos_2581833307_5ff495ce_full.json` |

Row 1: error_1 3.185 mHa, error_2 18.640 mHa, 19 circuits of 119 qubits and 2,126 Toffolis (80,788 over both passes),
0 failures on 254,464 shots. Only the circuit changed; this is the pipeline check.
Row 2: error_1 10.706 mHa, error_2 **19.982 mHa** (0.02 mHa under the gate; the metric is a deterministic function of the
files, so this is not a seed risk, but the margin is thin), E_gap 4.072132, Lambda 308.7275; 15 circuits of 123 qubits and
2,128 Toffolis (63,840 both passes); 0 failures. lambda_eff is 1.4 % above the baseline's because the refit lets negmass and
Lambda drift up; T is 17 % lower (fewer, cheaper lookups), Q 6 % lower (50 instead of 96 persistent word qubits).
Row 3 (**current best**): error_1 2.335 mHa, error_2 18.079 mHa (1.9 mHa margin), E_gap 3.918090 (floor 3.914898),
Lambda 308.6438, lambda_eff 49.0228 (0.6 % below the baseline's 49.3014); same 15 circuits of 123 qubits / 2,128 Toffolis
(the words differ). 0 failures on 254,464 shots. Bundle for upload with
`scripts/bundle.sh <invsqrt.kmx> <dqi.kmb> circuits/dfthcsos/dfthcsos_2581833307_5ff495ce`.
The public brot-10 design reports 2.345e9 with lambda_eff 47.5 from its own factorisation; at our lambda_eff the same
T and Q would give 2.60e9, so the circuit side is at the public level and the remaining 10 % is the factorisation.

## 6. What would move it further

- A factorisation trained for coarse angles (dfthc-sos Task 1, the starter kit's JAX trainer): lower intrinsic error
  and/or sparser rows would open brot 8 (1.47x) and brot 7 (2.0x); the public lambda_eff 47.5 is also a training result.
  Everything here (circuits, refit pipeline, one-body encoder) carries over unchanged to a new `submission.npz`.
- Circuit side is near its floor for this shape: ~1,990 (unary iteration over A/2 groups) + W + 88 per circuit; only
  a smaller A (fewer two-body vectors) or wider banks (impossible under the 128-qubit cap for W >= 35) would cut it.

## 7. Retarget (2026-10-25): load fewer vectors

The public frontier (`docs/public_sota.md`, 2026-10-25: 8.09e8, 4.15x, self-reported) does not come from better QROM
circuitry: it loads 572 vectors (single-group THC R 1 / B 496 / C 384) instead of 3,976. Cost model (verifier
formula, lambda_eff 47.5 assumed, ANGLES = K x 2 x (A - 1) with one unary-iteration Toffoli per address, circuit
qubits = selection + J brot + 10 iteration ancillas <= 128):

| shape | A | brot / J | K | T (heuristic) | Q | cost | ratio |
|---|---|---|---|---|---|---|---|
| (65,60,45) | 3,976 | 10 / 10 | 8 | 74,608 (11,008) | 734 | 2.60e9 | 1.29 |
| (24,60,30) | 1,516 | 10 / 10 | 8 | 30,056 (5,816) | 726 | 1.04e9 | 3.24 |
| (6,144,127) | 940 | 10 / 10 | 8 | 23,546 (8,522) | 793 | 8.87e8 | 3.78 |
| (1,496,384) | 572 | 10 / 10 | 8 | 20,146 (11,010) | 823 | 7.88e8 | 4.26 |
| (1,496,384) | 572 | 11 / 9 | 9 | 21,588 (11,310) | 822 | 8.43e8 | 3.98 |
| (1,496,256) | 572 | 10 / 10 | 8 | 17,930 (8,794) | 822 | 7.00e8 | 4.79 |
| (1,400,320) | 476 | 10 / 10 | 8 | 16,446 (8,846) | 823 | 6.43e8 | 5.22 |
| (2,250,200) | 576 | 10 / 10 | 8 | 17,000 (7,800) | 789 | 6.37e8 | 5.27 |

With J = 10 at brot 10 (100 word qubits) only a plain QROM fits in 128 qubits (10 + 100 + 10); the 2-bank trick needs
2W. T is ~1 Toffoli per address per circuit, so the lever is A = 76 + R B and, through the inner PREPARE, R C.

Accuracy limits of the target T = tei + shift (2,926 x 2,926 in the pair basis, PSD to 3e-4, ||T||_F 12.18 Ha):
the best rank-k approximation leaves 20 / 15 / 10 / 5 mHa at k = 290 / 307 / 330 / 384, so rank(Z) <= C must be
>= ~330 with quantisation in mind, and the public C = 384 sits at the 5 mHa rank. The baseline's own 3,900 vectors
are useless as THC points (random or pivoted-QR subsets of 500-1,200 leave more than 0.5 Ha with Z free): the points
must be fitted anew. `thc_fit.py`: eigen-dyad initialisation (dyads of the eigenvectors of the dominant
eigen-matrices of T), exact least-squares Z for fixed points (PSD rank-C truncation -> w), L-BFGS on (theta, w, W)
with the E_gap target and Lambda terms; ~40 ms per step at (1, 496, 384) natively.

The lambda_eff floor, derived: E_gap = E_UPPER - ecore + eta beta_1 + 0.5 sum_c W_c^2 + 2 negmass = -559.13 + 0.5 sum W^2 +
2 negmass (the baseline carries 0.5 sum W^2 = 540.5 Ha), so the strict floor forces 0.5 sum W^2 >= 563.05 - 2 negmass.
With single-signed weight columns, (|W_c + s_c| + ||w_c||_1)^2 >= W_c^2 (equality iff W_c <= -s_c), hence
Lambda_2 >= 0.25 sum W^2 >= 281.5 - negmass, and Lambda = Lambda_1 + Lambda_2 >= 281.5 + posmass(h1'). At the E_gap floor
lambda_eff >= sqrt(3.914898 * 559.13) = 46.79: the number the public notebooks quote. Reaching it needs E_gap at the
floor, h1' with no positive eigenvalues, every weight column of one sign and W_c <= -sum_b w_bc; our gauge-optimised
baseline-shaped family (Lambda 290) is 3 % above it.

Lookup for the small tables: no ANF compiler is needed. `build_qrom.py --banks 1` (one kickmix unary iteration over all
A addresses, CX loads) gives, for A = 572 / J 10 / brot 10, 8 circuits of 120 qubits and 574 Toffolis (4,592 per pass,
9,184 both passes, i.e. the public 20,286 - 11,310 heuristic); A = 476: 476 Toffolis, 118 qubits.

First fits (eigen-dyad init, exact least-squares Z, 3 x 300 L-BFGS iterations on (theta, w) with Z re-solved between
rounds): B 496 / C 384: 627 -> 376 -> 312 mHa; B 400 / C 320: 798 -> 356 mHa (an earlier "Z free" column here was
computed with a wrong formula and has been removed; ||T - P T P||^2 = ||T||^2 - ||Q^T T Q||^2). Z* = A T A^T is PSD
(T is) and its spectrum tail is tiny (rank-320 truncation of the B 400 Z* drops 0.004 in Frobenius), so the points
are the whole problem; the projection error is the right objective for them (variable projection, `varpro_points`).

Point optimisation: L-BFGS on the Givens angles converges to poor stationary points (B 400: stalled at 198.7 mHa after
1,500 iterations; B 496: 207 -> 147 after 1,000), because the angle parametrisation is badly conditioned (root angles
move whole subtrees). Variable projection directly on the point coordinates u (unconstrained: the scale of u_b is
absorbed by Z, rows renormalised at the end; `varpro_u`) drops the same B 400 state 198.7 -> 142.6 mHa in 100
evaluations. Starts: dyad ranking 628 mHa at B 496, pivoted-QR "ISDF" on the same candidates 1,245, random 10,387.

Direct-coordinate variable projection: B 400: 198.7 -> 94.1 mHa in 1,500 iterations (then ~0.7 mHa per 100); B 496:
207 -> 83.9 in 2,000 (~0.5 per 100); greedy growth 120 -> 504 with 60 iterations per step ends worse (158). The
residual of the B 400 fit is spread over ~120 eigen-directions (top 50 carry 59 of 94 mHa): it is not a few missing
points. The unregularised least-squares Z is badly conditioned (cond(U) 1,118, Z eigenvalues up to 1,161, trace 3,029
vs 28.8 for the baseline's weights, 49 point pairs with |cos| > 0.9), which gives Lambda ~ 29,000 (Lambda_2 29,373):
such a factorisation is useless for the score even if its error were fine.

**Why the public THC recipe penalises D(w).** With tr T = 53.3 Ha fixed (sum_c ||X_c||_F^2 = tr T for any decomposition),
C ~ 400 columns have ||X_c||_F ~ 0.37, and the E_gap floor fixes sum_c W_c^2 ~ 1,080 (W_rms 1.68). The one-norm
||w_c||_1 >= ||X_c||_F (equality only for identical same-signed dyads; the baseline's tightness is 4.7), so with
indefinite columns Lambda_2 ~ 1/4 sum_c (1.7 + 1.68)^2 ~ 1,100 and lambda_eff ~ 95. Only when every column is
single-signed (X^c PSD) and W_c <= -tr X^c does the term collapse to W_c^2 and Lambda_2 to the 270 floor, whatever
the weights' size. The baseline does not need this (only 3 % of its 3,900 columns are single-signed, its X^rc are
fully indefinite) because its columns are tiny (||X||_F 0.046 vs W_rms 0.53); a THC factorisation with ~400 columns
does. A score-worthy THC fit is therefore a constrained SOS-THC problem: T = sum_c X_c (x) X_c with PSD X_c,
plus the offsets and the error gate.

Probes of the constrained problem (B 400 points from the direct VarPro fit):
- Joint L-BFGS with mu Lambda_2 (mu 1e-5 .. 1e-3) and the E_gap target from the unregularised weights: E_gap steers
  to 3.918 fine, but Lambda stays at ~28,600 (mu 1e-3 trades +30 mHa of error for -10 of Lambda). The least-squares
  weights are a deep basin; the regularised problem cannot be reached from them by a penalty.
- SOS-constrained parametrisation (w = v*v, W_c = -tr X_c - t_c^2, `lbfgs_sos`): Lambda is 282-285 and lambda_eff
  47.2 **by construction** at E_gap 3.92-3.93, i.e. the floor structure is compatible with the gates; but starting
  from |w| of the unconstrained fit the error is 2.6 Ha and falls only to 1.23 Ha in 600 evaluations. The PSD-column
  decomposition needs its own initialisation (it is not a sign flip of the eigen-decomposition: cross terms).
- Unregularised B 496 asymptote: 74.5 mHa after 6,000 iterations (81 at 3,000; 0.16 mHa per 100 at the end): heading to ~70, not 15.

Assessment (2026-10-25): the vector-count lever is real and the circuit side is ready (plain QROM at 574 Toffolis /
120 qubits per circuit), but a from-scratch THC/SOS factorisation that clears error_2 <= 20 mHa with Lambda ~ 290 is
a multi-day optimisation project on this host (better optimiser: Gauss-Newton/LM on the variable-projection residual;
an initialisation with PSD columns; 10^5-step refits), with an uncertain outcome; the public results came from GPU
training (or a public dataset). Options: (a) stop at the verified 1.38x family (training branch) / 1.30x (this branch);
(b) continue the SOS-THC fit (days); (c) the group-dropping route R 65 -> ~26 on the baseline structure (the training
session measured 26-55 mHa after short refits for R 62-57; the public recipe uses 100k+ Adam steps per stage, ~12 h
per stage here); (d) a public factorisation dataset (the user's call).

## 8. SOS-THC continuation (2026-10-25, user's choice: continue)

Structure of the baseline that a THC fit must copy (all numbers exact):
- offsets: W_rc = -k tr X_rc with k = 1.434 (corr(W, -s) = 0.9966, s_rc = sum_b w_rbc = tr X_rc for unit rows); the
  one-body absorption sum W X has trace -748.6 against tr h1_base = -758.3, leaving tr h1' = -9.68;
- sum_c tr(X_c)^2 = s(I)^T T s(I) = K_T = 521.85 is fixed by T for *any* decomposition, so 0.5 sum W^2 = 0.5 k^2 K_T
  (540.5 at k 1.434) and E_gap = -559.13 + 260.9 k^2 + 2 negmass(h1_base + k sum_c s_c X_c): k is the one scalar
  that sets E_gap;
- columns nearly single-signed in the weighted sense: sum_c ||w_c||_1 / sum_c |s_c| = 1.08, so
  Lambda_2 = 1/4 sum_c (||w_c||_1 + |W_c + s_c|)^2 = 1/4 [611.6 + 464.9 + 105.8] = 295.6, 9 % above 1/4 sum W^2.
With PSD columns (w >= 0) and W_c = -k s_c the term is exactly k^2 s_c^2 and Lambda_2 = 1/4 k^2 K_T = 268.6: the floor.
Design for THC: w = v*v, W_c = -s_c - t_c^2 (t initialised at sqrt((k-1) s_c)), points shared; initialise from the
baseline itself by consolidating its 3,900 points (k-means on dyads) and its 3,900 columns (symmetric NMF of
Z = w'w'^T), then L-BFGS and Gauss-Newton (`gn_varpro`: Kaufman Jacobian, CG on the normal equations).

THC probes closed: the Gauss-Newton Jacobian is right (the finite-difference mismatch scaled linearly with the step),
and 40 LM iterations move the B 496 fit 74.45 -> 74.33 mHa: a genuine flat minimum. Consolidating the baseline's
points by k-means (B 400) costs 7.8 Ha before any refit (the groups' points are not shared directions), and compressing
its 3,900 columns by symmetric NMF to 384 loses 8 Ha for a 0.75 % relative error in Z: no usable THC start from the
baseline either. THC from scratch is parked.

**Within-group point pruning** (keep the baseline's 65 groups, fewer points each) is priced far better than I had
assumed, because at J 10 the lookups are ~16 A Toffolis over both passes and the inner PREPARE drops a QROAM level
at B <= 31 (temporaries 544 -> 511, b2 17 -> 16):

| shape | A | T (heur) | Q | ratio @ 47.5 |
|---|---|---|---|---|
| (65,60,45) | 3,976 | 74,608 (11,008) | 734 | 1.29 (verified 1.38 uses the 2-bank J 5 family: same T, Q 684) |
| (65,48,45) | 3,196 | 62,032 | 734 | 1.55 |
| (65,40,45) | 2,676 | 53,648 | 734 | 1.79 |
| (65,31,45) | 2,091 | 43,919 (10,479) | 699 | 2.30 |
| (65,24,45) | 1,636 | 36,583 | 699 | 2.76 |
| (65,16,45) | 1,116 | 28,135 | 667 | 3.76 |
| (65,15,30) | 1,051 | 24,792 (7,992) | 662 | 4.30 |
| (39,31,45) | 1,285 | 28,364 | 695 | 3.58 |

(For A <= ~2,100 the 2-bank J 5 family costs about the same: 15 A + 4,140 Toffolis vs 16 A, Q -50.) `prune.py` keeps
the B' highest-energy points per group and refits (u, w, W) with the full loss through `THCProblem`.

Within-group pruning is destructive: (65, 48, 45) from the baseline costs 1.19 Ha cold (20 % of the points, each
group's 60 dyads are a near-basis for its own columns), against 57 mHa for dropping three whole groups (5 % of the
points) measured by the training session: ~5x gentler per removed point. Switched to **group dropping** with long
joint refits (u, w, W; E_gap target 3.918; `prune.py --R`): 65 -> 60 -> 55 -> ... -> 26, groups ranked by their
contribution ||sum_c X_rc (x) X_rc||_F, 1,500 L-BFGS iterations per stage (~15 min at R 60).
