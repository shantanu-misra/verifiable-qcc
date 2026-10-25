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
