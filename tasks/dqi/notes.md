# dqi — working notes

Contract, cost formula and the verified baseline (2.444e13, Q 1,893, T 5,725,108) are in `docs/dqi.md`.
Everything below is measured with `operation_statistics()` on small builder circuits or by walking the
regenerated reference `.kmx`; these are **estimates**, never scores (CLAUDE.md rule 1).

## 1. kickmix GF(2¹²) primitive costs (`measure_primitives.py`, kickmix 0.1 at the pinned submodule)

| primitive | Toffoli-like | extra qubits | notes |
|---|---|---|---|
| `init_gf2_mul(x, y)` / `ixor_gf2_mul` (uncontrolled) | 63 | 12 (target) | 481 CX; `del_gf2_mul` is 0 Toffolis (HMR + CZ fixups) |
| `ixor_gf2_mul(..., control=q)` | 75 | 12 scratch | = 63 + 12: product into scratch, controlled XOR out, measure scratch away |
| `init_gf2_mul` + `del_gf2_mul`, both controlled | 150 | 24 | controlled del is not free |
| `init_gf2_inverse` (auto-clean) | 315 | 72 | Itoh–Tsujii, 5 multiplications; scaffold 48 + target 12 (+12 transient) |
| `init_gf2_inverse_with_scaffold` + `del_…_with_scaffold` | 315 | 60 | del is free while the chain is live |
| `init_gf2_div` (auto-clean) / `ixor_gf2_div` | 378 | 72 / 60 | inverse + 1 multiplication |
| `gf2_imul(const)` / `gf2_idiv(const)` | 0 | 0 | ≈ 33 CX each: GF(2)-linear |
| `gf2_iadd(reg, control=q)` | 12 | 0 | 12 CCX |
| `gf2_phase_by_product(x, y, mask)` | 0 | 0 | 144 CZ: ⟨x·y, mask⟩ kickback is free |
| `flip_if_equal(x, 0)` (12-bit) | 12 | 12 transient | 11 would do with an AND ladder; `flip_if_equal(x, y)` also 12 |
| `flip_if_less_than` 8-bit (reg / const) | 8 / 7 | 7 | |
| `iadd(1)` 8-bit (plain / controlled) | 6 / 9 | 7 | |
| `cswap(c, reg12, reg12)` | 12 CSWAP | 0 | |
| `multi_controlled_x` (12 controls) | 11 | 10 | |
| `init_lookup` + `del_lookup`, k-bit address → 12 bits | 17 / 72 / 144 / 279 for k = 4/6/7/8 | k | `del_lookup` raises for k ≥ 10 (needs more clean qubits than it allocates) |
| `init_lookup`, 12-bit address → 12 bits | 4,094 | 12 | plain unary iteration, 2^k − 2 |
| `unary_iteration` over 7-bit address (full sweep) | 126 | 7 | 2^k − 2 |
| `left_rotate(reg12, 4-bit quantum shift)` | 26 CSWAP | 0 | |

Takeaways: a quantum × quantum GF(2¹²) product costs 63 Toffolis and the inverse 315 (5 products); constant
multiplications and the ⟨product, mask⟩ phase kickback are free; a controlled register XOR is 12.

## 2. Reference circuit breakdown (`analyze_reference.py` on the regenerated `.kmx`, 86,105,159 lines)

Regenerating with `data/verifiers/dqi/dqi_reference_circuit.py` reproduces the sample's 5,725,461 static
Toffolis and Q = 1,893 (the verifier executed 5,725,108 per shot: only the `is_error` guard skips, M_j = 0).

| stage | Toffoli-like | share | per call |
|---|---|---|---|
| initial_shift (69 controlled lane rotations of reg_a) | 30,912 | 0.5 % | 448 per step, 28,980 of them CSWAP |
| forward_eea (222 synchronized-EEA steps) | 2,109,450 | 36.8 % | 9,502 per step |
| chien_forney_mbuc (4,095 positions) | 1,444,737 | 25.2 % | 351 per position |
| reverse_eea (222 steps, mirror) | 2,109,450 | 36.8 % | 9,502 per step |
| uncompute_initial_shift | 30,912 | 0.5 % | |

Inside one EEA step (9,502): `multiply_and_add_or_sub` 7,267 (two scaffolded divisions 378 each, then a unary-iteration
sweep over all 73 slots doing a controlled `ixor_gf2_mul` (75) + CSWAP q_i↔a_i (12) per slot ≈ 6,500), `shift_b_array_right`
1,069 (72 register CSWAPs + cursor logic), `swap_a_and_b` 893 (full A↔B register CSWAP, 876 + 17), `update_flag` 148,
`update_metadata` 56, `update_halting_counter` 48. 222 steps = 3(n+1)+9 is a worst-case bound on EEA sub-steps; every
step runs all 73 slots whether or not the polynomials are that long: this is the 40× that BM removes.

Inside one Forney position (351): `init_gf2_inverse_with_scaffold` of σ' 315 + `flip_if_equal(σ, 0)` 12 + `init_and`/`del_and`
of the 12-bit `located` register 12 + 12. The Chien update (`gf2_imul` by 69 constants) and the ⟨ω·σ'⁻¹, M_j⟩ kickback are
free (14.3 M CX over the sweep). 4,095 × 351 = 1,437,345.

### Register map (Q = 1,893) and live ranges

| block | qubits | live |
|---|---|---|
| reg_a[3..72] = S0..S69 (syndromes) | 840 | whole circuit (restored at the end) |
| reg_a[0..2] (3 extra polynomial slots) | 36 | from the setup to the end |
| reg_b[0..72] (second EEA polynomial array, becomes σ / σ' / ω) | 876 | forward_eea → reverse_eea |
| degree counters n_A n_B n_a n_b n_q (8 bits each), flag, 2-bit counter, 8-bit halting counter | 51 | setup → end |
| transient workspace inside an EEA step: q_i 12, division scaffold 60+12, 4 unary cursors 4×8, is_c3/is_hc_zero/ctrl×4/aux/mul/both/nf… | ≤ 90 | within a step |

Live-qubit profile from first/last mention: 11 before `initial_shift`, 869 after it (syndromes + reg_a[0..2] − unmentioned
lanes), 1,893 from the end of `forward_eea` through `reverse_eea` (the builder recycles freed ids, so every id is in use at
the EEA peak), back to 869 after `reverse_eea`. 1,893 − 840 = 1,053 workspace qubits: 876 are the second polynomial array,
87 counters/flags, ≤ 90 EEA-step scratch. The Forney sweep itself only touches 953 ids (σ, σ', ω slots + 72 inverse scratch),
but reg_b/reg_a stay live underneath it, so Q is set by the EEA, not by the sweep.

## 3. Classical model (`model_bm.py`) and design facts

BM over GF(2¹²) (`berlekamp_massey`) + Chien/Forney (`chien_forney`) reproduces e and the phase parity on verifier-sampled
shots: 0 mismatches / 2,048 (seed 1); 10,240-shot run in `build/model_bm_10240.log`. Observed: final L = k always (deg C = L,
no degree drops), the number of length changes = k, ≈ 7.9 zero discrepancies per shot.

Test distribution facts that the design may exploit:
- weight k: P(k=35) = 2.8 %, P(k≥34) = 12 %, P(k≤20) = 5.5e-5, P(k≤23) ≈ 1.8e-3. Nothing above k = 20 is rare enough to fail on.
- failure budget: R(ε⁺) with 0/1/2/5/10/25/50 failed shots of 102,400 = 1.1913 / 1.1967 / 1.2014 / 1.2131 / 1.2290 / 1.2659 / 1.3134.
  One failure costs +0.45 %, ten +3.2 %: inputs with probability ≲ 1e-5 per shot may be left undecoded if handling them
  costs more than ~1 % of T (a GF(2¹²) coefficient is zero w.p. 2.4e-4 → ~25 shots: too frequent to ignore).

State-size facts (why Q ≈ 1,800 in every published design): BM needs S (70 slots of 12) read-only plus C and B (36 slots
each) = 142 slots ≈ 1,704 qubits + scratch; Sugiyama/EEA holds two remainder/multiplier rows ≈ 142 slots. (σ, ω) alone
is 71 slots, and S = ω/σ (power series) is recoverable from it, but the forward computation still peaks at S + C + B.
Erasing syndromes outside BM's 36-wide window via the backward recurrence (needs C_L ≠ 0, true here) costs ≈ L products
+ 1 inversion per step, i.e. doubles forward/reverse T for Q ≈ 1,300–1,400: roughly break-even on Q²T. The win must come
from a cheaper T per step or a smaller window.

## 4. Design: windowed Berlekamp–Massey (decision 2026-10-24)

Reversibility without junk (checked with `build/probe_bm.py` on 512 shots, 31,759 BM steps): a BM step
`C ← C ⊕ λ x^m B`, λ = d/b, is undone from the post-state alone. Length change (B ← C_old, b ← d): the new term's top
lands at index L_new where C_old is zero (15,874/15,878; the rest have deg B < L_prev), so λ = C_new[m]⊕C_old[m] is a
single XOR and B_old = (C_new ⊕ C_old)/(λ x^m). No length change: λ = C_new[m] ⊕ B[m] when m < m_prev (generic
alternating pattern: m = 1, m_prev = 2). Reverse BM therefore needs no stored discrepancies and costs about half the
forward pass (as the public 92k vs 148k suggests).

Window: BM at step n only reads S_{n−L..n}, L ≤ 35, so S_j (j ≤ 35) is dead after step j + 35. Erase it there with the
backward recurrence S_j = C_L^{-1} Σ_{k<L} C_k S_{j+L−k} (deg C = L always in the model): L−1 products + 1 inversion
+ 1 product ≈ 2.3k Toffolis, 35 erasures per pass. Live state during BM: S window 35 slots + C 36 + B 35 = 106 slots
(1,272 qubits) + d, b, λ 36 + L, m counters ≈ 12 + inversion scratch ≈ 30 (tower) → **Q ≈ 1,350–1,400**.

Toffoli estimate per pass (kickmix 63/product; Σ_n L_n ≈ 1,209 for k ≈ 31): discrepancies 76k + updates 76k +
divisions d/b 70 × ~80 (tower) 6k + erasures 35 × 2.3k 80k → forward ≈ 240k; reverse (no discrepancy recompute) ≈ 160k;
sweep 4,095 × (zero test 11 + tower inverse ≈ 70–90) ≈ 330–410k. **T ≈ 730–810k**, cost ≈ 1.19 × 1,400² × 770k ≈ 1.8e12
(≈ 13.5× baseline; public 10.6×). Levers after a first verified circuit: own Karatsuba GF(2¹²) multiplier (63 → 54),
tighter tower inverse (public reports 69), sharing the C_L inverse across erasures.

Tower arithmetic measured in kickmix: GF(2³) mul 7 / inv 7, GF(2⁴) mul 9 / inv 18, GF(2⁶) mul 21 / inv 63 (24 extra
qubits). GF(2¹²) = GF(2⁶)[u]/(u²+u+ν): inverse = (a+b+bu)·N⁻¹, N = a²+ab+νb²: 1 GF(64) mul + GF(64) inverse
(via GF(8): 4 GF(8) muls ≈ 28) + 2 GF(64) muls ≈ 91 with kickmix primitives; the basis change is GF(2)-linear (free).
Field: modulus 0x1053, primitive element 2 (x).

### 4b. Corrections after probing (build/probe_bm*.py)

- The one-XOR recovery of λ only holds for the first no-length-change step of a division; the general rule is the
  power-series quotient ν_j = [A2/B]_j = (A2[j] ⊕ Σ_{i<j} ν_i B[j−i]) / B[0] (valid for j < m1, the shift at the last
  length change). Over 4,096 shots every no-change step had j ≤ 2 (the only j ≥ m1 case is the fixed step n = 1, where
  the "B_old" is the classical initial 1); zero-run ≥ 2 inputs (probability ≈ 4e-6/shot) would be mis-decoded — accepted.
- L_n ≤ ⌊n/2⌋ + 1 on all 2,048 probed shots → static loop bounds min(35, ⌊n/2⌋ + 2) for the discrepancy/update products.
- Replacing a syndrome register by its discrepancy breaks BM (0/256), so d must be erased each step (MBUC or ν·b) and
  recomputed in the reverse pass: reverse ≈ forward in products.
- Windowed syndrome erasure during BM is impractical: the backward recurrence anchors at C_L ↔ S_{n−L}, a quantum index,
  while the discrepancy anchors at C_0 ↔ S_n; no single layout aligns both classically. Peak live set = S 70 + C 36 +
  B 35 = 141 slots ≈ 1,692 qubits + scratch: **Q ≈ 1,800 is structural for BM/EEA**, matching the public circuit.
- ω: computed from σ and S_1..S_35, which are then erased via S_i = σ_0⁻¹(ω_{i−1} ⊕ Σ_k σ_k S_{i−k}) (classical alignment),
  so the sweep's live set is S_36..S_70 + σ + B + ω = 141 slots too. Horiguchi–Koetter (e_j ∝ 1/(B(X_j⁻¹)σ'(X_j⁻¹)))
  would avoid ω but costs an extra product per position (+217k) vs ω's 2 × 630 products (154k): not worth it.

v1 plan: kickmix primitives everywhere (63/product, 315/inverse) to get a verified circuit first; then tower inverse in the
sweep, own Karatsuba multiplier, tighter scratch.

## 5. Build log (Phase 3)

Generator `build_bm.py` (+ `tower.py`, `gfmul.py`). Toy instances GF(2⁴) n=8 m=15 and GF(2⁶) n=12 m=63 are checked
end to end against the classical phase parity (`build/test_full.py`, 0 errors). Full-scale circuits are ≈ 420–440 MB `.kmb`
(42–58 M instructions), so they are kept out of git (`circuits/dqi/.gitignore`); the generator and the verifier JSON
(with the circuit's sha256) are the committed evidence, and `python tasks/dqi/build_bm.py circuits/dqi/<name>.kmb`
regenerates the file byte-for-byte.

| version | change | static Toffolis | Q | official run |
|---|---|---|---|---|
| v1 | kickmix primitives only | 2,092,417 | 1,903 | smoke 2,560 shots PASS, 0/0 errors |
| v2 | tower inverse GF(2¹²)→GF(2⁶) (126/inverse) | 1,265,164 | 1,885 | **full PASS: score 5.3545e12, T 1,265,001.4, 0/0** |
| v3 | two-level tower →GF(2³) (91/inverse) | 1,111,969 | 1,885 | **full PASS: score 4.7062e12, T 1,111,842.0, 0/0** |
| v4 | 54-Toffoli Karatsuba products (gfmul.py) | 990,379 | 1,886 | smoke PASS 0/0 |
| v5 | recursion temporaries erased by measurement | 967,699 | 1,886 | smoke PASS 0/0 |
| v6 | 11-Toffoli zero test, inverse gated by the zero flag (no `located` AND) | 933,539 | 1,886 | **full PASS: score 3.9552e12 (6.18×), T 933,432.0, 0/0** — `runs/dqi_3955239639608_7d3775fc_full.json` |
| v7 | gating pushed down to the GF(2³) inverse (3 Toffolis) | 921,254 | 1,886 | **full PASS: score 3.9032e12 (6.26×), T 921,150.0, 0/0** — `runs/dqi_3903196955376_bdf68ad7_full.json` |

| v8 | zero-test ladder kept live across the sweep inverse | 880,304 | 1,886 | **full PASS: score 3.7297e12 (6.55×), T 880,209.8, 0/0** — `runs/dqi_3729720521808_0cb5eafc_full.json` |
| v9 | + live ladders in the BM recursion | 871,064 | 1,916 | rejected: Q² loss exceeds the T gain |
| v10 | v8 + two-copy B-slot selection | 875,264 | 1,886 | **full PASS with 1 value error**: score 3.7254e12 (6.56×), T 875,169.7, R 1.1967 — `runs/dqi_3725395305991_c3edfdbf_full.json`. The failing shot (replayed with `build/find_failing_shot.py`) has BM history `… z z L n1 n2 n3 …`: two consecutive zero discrepancies then a length change with shift j = 3, the accepted class outside the j ≤ 2 quotient recursion (≈ 3.6e-6/shot, 0.37 expected per 102,400). Extending the recursion to j = 3 would cost ≈ 42k Toffolis (4.8 %) to save 0.17 % of R: not worth it. |

Measured per-pass costs (v1 analyzer, kickmix products at 63): discrepancy 87.8k, update 92.0k, erase_nu 78.1k,
cswap 31.1k, quotient 26.5k; omega swap 79.7k forward / 40.0k reverse.

v6 per-stage (analyzer, static): bm_forward 232,326 (discrepancy 75,276, update 78,840, erase_nu 44,170, cswap 18,360,
quotient 9,310, L/flags 2,660), omega_swap 68,119, sweep 438,165 (107/position = zero test 2 × 11 + gated inverse 85),
omega_unswap 34,099, bm_reverse 160,830 (no discrepancy: erased by measurement). Public: BM 241k total, sweep 328k.
v8: the zero-test AND ladder stays live across the inverse so the flag is uncomputed with one Toffoli (−10/position).
v9: same for the three zero tests of the quotient recursion; B-slot selection with two controlled copies instead of
three (B_k = A1[k] ⊕ f1(A1[k]⊕A1[k+1]) ⊕ f2(A1[k]⊕A1[k+2]), valid because slots below the shift are zero).
Remaining overhead vs the public BM (≈ 144k): erase_nu 2 × 44k, cswap 2 × 18k, quotient 2 × 9k — the price of the
role-swap formulation; the products themselves (54/product) are within 6 % of a 51-Toffoli M(6)=17 multiplier.
