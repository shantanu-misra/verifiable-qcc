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
| v11 | recursion inverse without a kept scaffold (rebuilt to uncompute, +46 Toffolis/inverse, −30 qubits) | 881,284 | **1,856** | **full PASS: score 3.6160e12 (6.76×), T 881,190.1, 0/0** — `runs/dqi_3616031951334_7a56548d_full.json` |
| v12 | 17-product GF(2⁶) formula found by random search → 51-Toffoli GF(2¹²) products | 846,251 | 1,856 | **full PASS with 1 value error: score 3.4882e12 (7.01×), T 846,160.4, R 1.1967** — `runs/dqi_3488231896097_75a84b53_full.json` — the failing shot replays as `… z z L n1 n2 n3 …`, the accepted j = 3 class, not a multiplier fault |
| v13 | ω accumulated in place in S_i (two-product rescaling by σ̃₀ via s⁻¹), no erase-S step | 817,775 | 1,856 | **full PASS: score 3.3554e12 (7.28×), T 817,684.2, 0/0** — `runs/dqi_3355430905041_6ea37ae1_full.json` |

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

### Round 2 summary (2026-10-24)

v10 → v13: 875,264 → 817,775 static Toffolis (−6.6 %), Q 1,886 → 1,856 (−1.6 %), verified cost 3.725e12 → 3.355e12
(−9.9 %; 6.56× → 7.28× below baseline). Rejected: live zero-test ladders inside the BM recursion (v9: −9k Toffolis but
+30 qubits at the peak). Not pursued: Karatsuba short product for ω (superseded by the in-place accumulation), caching
σ₀⁻¹ across a division (+24 qubits at the peak outweigh the saved inverse), tighter static bounds (+2/+3 margins cover
one or two mid-run zero discrepancies; a tighter bound would fail ≈ 1e-4 of shots).
Where the remaining T sits (v13, analyzer): sweep 372,645 (4,095 × 91: zero test 12 + gated tower inverse 79);
bm_forward 222,294 (discrepancy 71,094, update 74,460, erase_nu 43,120, cswap 18,360, quotient 8,890); bm_reverse
154,770 (same minus the discrepancy); omega_inplace 34,033 each way. The public circuit's 583k differs mainly in the BM passes (241k vs ≈ 360k): the quotient
erasure, the role swap and the per-step division are the price of this reversible formulation.

## 6. Live-qubit profile of v13 and the Q question (round 3, 2026-10-25)

`live_profile.py` logs `CircuitBuilder.num_allocated_qubits` at every alloc/free (189,677 samples) — exact, since
kickmix reuses freed ids and `num_qubits` = peak allocation here (1,856 = 1,856).

| phase | max live | composition at the max |
|---|---|---|
| bm_forward / bm_reverse | **1,856** in all 140 steps, inside `erase_nu` | base 1,723 (S 840, A2 36×12, A1 36×12, b 12, L 7) + recursion scratch 133 (ν 12, B0..B2 36, inv 12, q0,t1,q1,t2,q2 60, flags/zero bits 13) |
| sweep | 1,792 | base + inverse scaffold and ladder (69) |
| omega_inplace (both) | 1,766 | base + s⁻¹, p, products (43) |

So the peak is a flat plateau set by the BM working set, not a spike: every step reaches it because A2 and A1 are
allocated at their full 36-slot bound (needed at the end, L = 35) and all 70 syndromes stay live.

Break-even rule (R fixed): Q′²·T′ < 1,856² × 817,684 = 2.82e12.

| candidate | Q′ | T′ | Q′²T′ / 2.82e12 | verdict |
|---|---|---|---|---|
| recursion scratch: recompute t1/q1 instead of keeping them (−24 qubits, +~200 Toffolis/step) | 1,832 | 846k | 1.008 | reject (loss) |
| recursion: controlled products instead of B1/B2 registers (−24, +~380/step) | 1,832 | 871k | 1.04 | reject |
| recursion: only j ≤ 1 branches (−24, no T cost) | 1,832 | 818k | 0.97 | reject: 0.7 % of shots fail (R ≈ 1.9) |
| A2/A1 at true degree (lazy slots) | 1,856 | 818k | 1.00 | no gain: the peak is at the end of BM where both are full |
| free B or σ during the sweep | 1,856 | — | 1.00 | the sweep (1,792) is not the peak |
| erase dead syndromes S_j, j ≤ n−35, during BM | ≈1,480 | ≥ 1.8M | ≥ 1.4 | reject: the recurrence that gives S_{n−35} anchors at C_L (quantum offset); classical alignment needs a controlled rotation of a 36-slot array by L (≈ 2.6k CSWAP, twice) per step |
| erase S_{n−L} (classically anchored at position n) when dead | — | — | — | reject on identity grounds: *which* syndrome is dead is quantum (depends on L), and Q counts ids, so a register whose content is zero only for some shots cannot be reused by another value — only a classically known id can be freed |
| in-place Euclid (reference layout) | 1,893 | — | — | two 73-slot rows: worse than BM's 141 slots |
| recompute S from (σ, ω) and run the passes without S | — | — | — | BM must read S every step in both passes; the sweep, which could run without S, is not the peak |

Conclusion: with the syndromes live, Q ≥ 1,723 + minimal scratch for any BM/EEA formulation, and the recursion's
133 scratch qubits cannot be traded for Toffolis at a profit (24 qubits = 2.6 % of Q², less than the recomputation
costs). A Q of 1,300 needs a decoder that *consumes* the syndromes while finding σ; the two obstacles are quantum
alignment (the recurrence relating a dead syndrome to live ones is indexed by L) and the id argument above. Nothing
here clears the break-even rule; round 3 is therefore a negative result on Q for this architecture.

## 7. In-place (syndrome-consuming) decoder: research result (round 4, 2026-10-25)

Facts from the classical model (`probe_dead.py`, 4,096 shots): step n′ reads S[n′−L_{n′−1} .. n′]; the smallest index any
later step reads is ≥ min(n+1−L_n, L_n−1) (282,624/282,624 checks). Classical lower envelope: at step n the syndromes
S[0 .. ⌊n/2⌋−3] are dead for every sampled shot. Interior coefficients C_c (0 < c < L) vanish with rate 1.8e-4.

What an in-place scheme must do: zero a dead syndrome S[j] at step n by XORing into it a function of *other live*
registers. The only available relations are the recurrences D(p) = Σ_k C_k S[p−k] = 0, L+1 ≤ p ≤ n, in which S[j]
carries coefficient C_{p−j}. Three obstacles, each fatal on its own:

1. **Identity of the freed register.** Q counts ids, so the freed register must be classically known (fixed j_n) and
   must be zero for *every* shot when another value takes its ids. Any scheme that erases "S[n−L]" or gates the
   erasure on a quantum flag frees different registers in different shots and cannot reuse them.
2. **Alignment.** With a classical j, the relation anchored at the top coefficient (c = L, the only interior-or-top
   coefficient that is guaranteed nonzero besides C_0) has p = j + L, a quantum position: either C or the syndrome
   window must be rotated by L (6 controlled rotations of a 36-slot array ≈ 2.6k CSWAP, twice per step), i.e.
   ≥ 5k Toffolis/step → T ≈ 1.6M at Q ≈ 1,500: Q²T ≈ 3.6e12 > 2.82e12. A top-aligned copy of C avoids the rotation
   but costs 36 slots and a second product set, more than the ≤ 32 slots it frees.
3. **Order.** The classically aligned relation anchored at C_0 (p = j) expresses S[j] through *older* syndromes, so
   erasures would have to run from newer to older while syndromes die from older to newer; any other classical
   anchor c (0 < c < L) also reaches older indices (p − L < j) unless c = L, and its coefficient vanishes with
   probability 1.8e-4 per erasure (≈ 0.6 % of shots over 35 erasures → R ≈ 1.8). Two alternative anchors cost a
   second product set.

Costed variants (all per BM step, both passes): rotate-by-L erasure ≈ 5k → fails; relation at position n with the
new C plus selection of C_L, the prefix and the target by unary iteration over L (≈ 3 × 432 + 36 products + inverse
≈ 3.3k) → Q ≈ 1,500, T ≈ 1.24M, Q²T ≈ 2.8e12 — break-even at best, and it violates obstacle 1 (the erased index is
n−L). Euclid-style states are 141 slots at every step (notes §4b); PGZ/Hankel elimination cannot be done in place
because syndromes are shared between matrix entries. No candidate clears the rule, so nothing is built.

Verdict: for BM/EEA decoders the syndromes stay live and Q ≥ 1,723 + scratch. A genuinely smaller Q needs a decoder
whose state is the error list rather than (S, σ, B) — not a variant of BM — and we have no such algorithm.

## 8. Toffolis at fixed Q (round 5, 2026-10-25): candidates against Q′²T′ < 2.82e12

v13 per BM step (forward): discrepancy 1,016 (≈ 20 products), update 1,064, erase_nu 616 (inverse 125, 6 products
306, B-slot copies 72, ν selection 36, three zero tests 66, flags ≈ 10), cswap 262, division 127 (inverse 79 + product
51), L/flag bookkeeping ≈ 40. Reverse: the same without the discrepancy. Sweep per position 91: zero test 12 (11 + 1),
gated tower inverse 79 (GF(64) product 17, GF(8) tower 27 incl. 3-Toffoli gate, two GF(64) products 34).

| candidate | ΔT (both passes) | ΔQ | verdict |
|---|---|---|---|
| division-free step by storing B pre-scaled (B/b, so ν = d and b ≡ 1) | −130/step but the lc re-scaling B ← C_old/d costs L products (+1,064/step) | 0 | reject |
| two row operations instead of cswap + division (C ⊕= d·X, then X ⊕= lc·d⁻¹·C: b stays 1 forever, no swap) | +1,064 −262 −51 −125 = +626/step | −12 | reject |
| inversionless (riBM) update C ← b·C ⊕ d·x^m B | in-place scaling of ≈ L slots: +2 products per slot | 0 | reject |
| cache 1/B[0] (and 1/C[0]) to drop the recursion's inverse | −125/step but needs 1/ν per lc (+79) and 2 extra registers | +24 | reject (Q² +2.6 % > T gain) |
| only j ≤ 1 branches, ν₂ parked in one junk register | −200/step, +12 qubits, ≈ 10 shots with two j=2 events fail | +12 | reject (R +3 %) |
| tighter static bounds (+1/+2 instead of +2/+3) | −14k | 0 | reject: ≈ 1e-4 of shots (two mid-run zeros) would fail, R +3 % |
| Karatsuba short product for ω | −38k | needs the 35 outputs in separate registers: +420 | reject |
| **2-bit saturating shift counter m instead of zero tests of A1[0..2]** (f0 = lc, f1 = ¬lc∧m=1, f2 = ¬lc∧m=2; lc erased by m = 1; no wrap flag) | ≈ −90/step → −12.6k | +2 | **accept** |
| **GF(8) inverse as x²·x⁴ (one 6-Toffoli product of linear images) instead of kickmix's 7** | −1 per inverse: −4.4k | 0 | **accept** (marginal) |
| cheaper 12-bit zero test than 11 + 1 | none known | — | — |
| sharing products across sweep positions | none: each position's σ′ is an unrelated field element | — | — |

Where the public BM is cheaper (≈ 120k over both passes): their step has no quotient erasure and roughly half our
swap; we have not found a formulation without the ν erasure for no-length-change steps (the junk is intrinsic:
after C ⊕= ν x^m B the old C is gone and ν is only recoverable from (C, B) by the power-series recursion).
Built as v14 (the shift counter was dropped: its reset at a length change is irreversible, so the lc erasure reuses the
recursion's zero flag and gate instead, −29/step; leaf GF(2³) inverse as N²·N⁴ with N⁴ kept as the 3-qubit scaffold,
−1 per inverse): 810,176 static Toffolis, Q 1,856 — **full PASS: score 3.3243e12 (7.35×), T 810,085.9, 0/0** —
`runs/dqi_3324250406015_ee5f2540_full.json`.

### 8b. Formulations without a per-step quotient erasure (round 6, 2026-10-25) — none clears the rule

The junk is intrinsic to a no-length-change step: C ← C ⊕ ν x^m B destroys C_old, and ν is then a function of
(C_new, B) only through the power-series quotient (the recursion). Every alternative either moves the junk or pays
for it elsewhere:

| formulation | per-step effect vs v14 | ΔQ | verdict |
|---|---|---|---|
| keep ν as junk during the division, erase the whole quotient at the next length change (depth-δ recursion once per division) | the recursion is still present in every step (static circuit): no change | 0 | reject |
| skip slot m in the update and carry the term ν·B[0] implicitly (one extra product per discrepancy), materialise at the next lc | δ = 2 support needs two junk registers and the depth-2 recursion anyway: ≈ 626 vs 606 | +12..+24 | reject |
| erase ν₁ by a depth-1 recursion at j = 1 and ν₂ by a depth-2 one at j = 2 (separate machinery) | both run every step: 269 + 606 | 0 | reject |
| implicit C = X ⊕ Q(x)·C_old during a division (discrepancies of X and of C_old kept, no C update) | needs disc(C_old, ·) at δ+1 positions and the materialisation of both C′ and C″ at the lc: ≈ 6L vs 4L+24 products per two steps | 0 | reject |
| two row operations at the lc so that C stays unscaled (C₀ = 1, b ≡ 1) and the recursion needs no inverse | second row op = L products (+1,064), −262 cswap −51 −116 | −12 | reject (+635/step) |
| store κ = 1/B[0] once per division (recursion inverse-free) | κ must be erased at the next lc: another inverse of a quantum-indexed slot | +12 | reject |
| make B[0] equal to b so one inverse serves division and recursion | impossible: B[0]/b = 1/d_true is scale-invariant | — | reject |
| avoid δ = 2 divisions (the j = 2 branch costs ≈ 200/step) by splitting a late length change | not an algebraic identity of BM; zero discrepancies are data | — | reject |
| cheaper selections (rotate A1[0..4] by j instead of controlled copies) | 120 CSWAP vs 72 copies | 0 | reject |

Conclusion: within scaled-BM with a physically shifted B, the per-step erasure (inverse + 6 products + selections,
≈ 600 Toffolis) is minimal for j ≤ 2 support, and the public circuit's cheaper BM step must rest on a bookkeeping we
have not identified. T work on the BM passes stops here; v14 stands.

## 9. In-place continued-fraction decoder (round 7, 2026-10-25)

Credits: the in-place J-fraction chart is due to SomeJoint / Agniv Banerjee (public Apache-2.0 notebook "Quantum
Circuits 8.95 LB cost breakdown"); gap pointers, paired extraction and lifetime scheduling to Phuc Thai, ViberTeam /
Artem Nazemtsev and Ahmed Berat Özer; the classical mask projection to manwithacat. We read the rendered public
notebooks and wrote our own model and circuit from the algorithm (no code copied).

Algorithm (model_cf.py). f(x) = Σ S_{i+1} x^i. Level: pivot c = f[0] (gap order r if f starts at x^r), divide the
remaining words by c, invert the unit series in place, read A = 1 + a_1 x (+ … for a gap) and continue on the words
after A: f = x^r c / (A + x^{r+2} g). 35 ordinary levels turn the 70 syndrome words into 35 (c, a) pairs **in place**.
Convergents: q′ = A q + x^{r+2} p, p′ = x^r c q, bottom-up and also in place (old q cells become p′, old p cells
become q′, (c, a) become (p′₀, q′₁)); then q = σ (q₀ = 1), p = ω, and the Forney sweep is unchanged. Everything is
reversed afterwards. The unit-series inverse is the Frobenius chain a ← a·a^(d) for d = 2, 4, 16 (exponent
1 → 3 → 15 → 255 ≡ −1 mod x^70): a[k] ^= a[i]^d · a[k − d i] with k descending, each term a product of two still-
unmodified cells (a[i]^d is a free linear twist).

Model results: 0 mismatches of (p, q) against BM's (σ, ω) on 512 shots (20,480 running); first-order gaps (zero
pivot ⇔ BM zero discrepancy) in ≈ 0.8 % of shots, higher orders not seen yet. Gaps need pointer bookkeeping (public:
two 6-bit pointers, exact for ≤ 2 first-order gaps, third gap approximated at ≈ 8e-8/shot).

One-way product counts (ordinary chart): reciprocal chain 21,450 naive, scaling 1,225 (one product + free erase per
word), conversion 1,785; inversions 70. Estimates against Q′²T′ < 2.79e12 (v14):
- naive shears: T ≈ 2 × 24,460 × 51 + 140 × 79 + 4,095 × 90 ≈ 2.87 M at Q ≈ 920 → 2.4e12 (−13 %).
- Karatsuba on the cross blocks (≈ 2× fewer reciprocal products): T ≈ 1.7 M → ≈ 1.4e12 (−48 %).
- evaluation-rank blocks as in the public design (≈ 4.8k reciprocal products): T ≈ 1.3 M → ≈ 1.1e12.
Build order: scaling kernel → Frobenius-chain reciprocal (naive, then blocked) → conversion → gap flags → Forney on
(p, q) → full decoder.

### 9b. Kernels verified on toy fields (build_cf.py)

Extraction (in place, round trip restores), conversion ((P, Q) match the model) and the full decoder are exact on
GF(2⁴)/GF(2⁶) toy instances except on gap inputs (zero pivot with a nonzero tail): phase errors occur only there.
A term a[i]^d·a[i] (k = (d+1)i) needs a temporary copy because the multiplier takes two registers.

Gap design (first order, own formulation). A gap level f = x·c/(1 + a₁x + x²h), h = a₂ + x·g, is the *ordinary* level
applied to f/x followed by ordinary levels on h — so no "skip" slot is needed: detect g = [w₀ = 0 ∧ w₁ ≠ 0] (two zero
tests), rotate the active window left by one (controlled CSWAP chain, 12·(n−1)), run the ordinary level, and keep a
2-bit count s of borrowed zero cells now sitting at the end of the window. Scaling leaves those zeros zero by itself;
the reciprocal's updates of the top s cells are routed through a delta register and applied only when the cell is
real ([k < n − s], 12 Toffolis per d-pass per protected cell), delta erased by measurement. Conversion treats each
shifted level as p′ = x·c·q (one extra coefficient, taken from the borrowed zero by a controlled rotation). Second-
order gaps (w₀ = w₁ = 0, w₂ ≠ 0, ≈ 2e-6/shot) stay approximate, as in the public design.

### 9c. Gap handling as built (one first-order gap, pointer + rotation)

Full-scale ordinary chart (no gap handling, `dqi_cf1`): smoke run 2,560 shots → 0 value errors, 12 phase errors (all
gap shots), Q 921, T ≈ 2.82 M (statistics; not a score). Model facts (20,480 verifier shots, `model_cf_gaps.py`):
weight histogram peaks at 31–32, weight 35 is 2.7 %; gap shots 0.70 % (one gap 0.70 %, two gaps 0.01 %).

Design (the "plain" scheme, exact for weight ≤ 34): at level k, g = [w₀ = 0 ∧ w₁ ≠ 0 ∧ no gap yet]; if g, rotate the
window left by one (the zero goes to cell 69, which stays a protected zero) and write k+1 into a 6-qubit pointer; g is
erased by recomputing [ptr = k+1]. The ordinary level then runs on f/x; the reciprocal's writes to the last cell are
gated by F = [ptr = 0] (product into a temporary, 12 CCX, free erase), scaling a zero cell is harmless. Conversion is
the ordinary arithmetic; at the gap level P' = x·cQ is realised by rotating the P cell list right by one (controlled
on [ptr = k+1]); the spare top cell of Q supplies the zero. The pointer is freed after the reverse extraction.
Cell model `model_cf_gaps.py --cells`: exact on all shots except weight 35 with a gap (3 of 4 such shots: the J-fraction
of f/x needs 71 coefficients) and two-gap shots; expected ≈ 20–25 failures per 102,400 → R ≈ 1.25–1.27.
Toy circuits (GF(2⁴) n=8, GF(2⁶) n=12): cells+pointer match the model on 384 shots; full decode has phase errors only
on model-wrong inputs (6/256 and 0/128), 0 value errors.

Verifier failure table (N = 102,400, Clopper–Pearson at 2⁻¹²⁸): failures 0→R 1.1913, 1→1.1967, 5→1.2131, 10→1.229,
20→1.2548, 25→1.2659, 50→1.3134. So the weight-35 gap shots (≈19 per run) cost ≈ 5 % — worth an exact gap level
(A of degree 2, second rotation parking a₂, ≈ 100 k Toffolis) only while T ≳ 2.5 M.

**v15 = `dqi_3042909973384_df7d3863.kmb` (build_cf.py, gaps on): full PASS, score 3.0429e12 (8.03×), Q 927,
T 2,888,132.43, 0 value / 9 phase failures (ε⁺ 1.16e-3, R 1.2261)** — `runs/dqi_3042909973384_df7d3863_full.json`.
−8.5 % vs v14 with the naive reciprocal chain; the T budget is now the shears (≈ 2.2 M of 2.89 M), then the sweep (0.37 M).
Q profile (live_profile on m=63): sweep/forney_step peaks at 927, extraction 903, conversion 902, base 858 = 840 cells +
6 pointer + 12 for the constant-one register of A2[0] (removable: fold q₀ = 1 into the sweep's sums).

## 10. Round 8 estimates (v15 → rank-reduced shears), before building

**Where the 2,888,222 static Toffolis sit** (analyzer on the m=63 build, sweep scaled ×65; `build/dqi_cf2_m63.kmx`):

| phase | per direction | both | of which |
|---|---|---|---|
| extraction (35 levels) | 1,188,368 | 2,376,736 | reciprocal chain 21,450 products × 51 = 1,094 k; scaling 1,225 × 51 = 62 k; gated end-cell writes ≈ 12 k; rotations 14.7 k CSWAP; inverses + flags ≈ 6 k |
| conversion | 71,468 | 142,936 | 1,190 products × 51 = 61 k; P-list rotations 7.1 k CSWAP; inverses 4 k |
| sweep (4,095 positions) | — | 368,550 | ≈ 90 per position (tower inverse 78, zero test 11, …) |

So the shears are 76 % of T. Everything else is 0.51 M.

**Shear reduction (modelled in `build/shear_count.py`, checked algorithm in `build/tmvp_eval.py`).** Pass d of the
Frobenius chain is a ← a·(1 + Σ_i a_i^d x^{di}) mod x^L, i.e. d independent unit-triangular Toeplitz products, one per
residue class mod d. In place, a class vector of n cells is updated by the recursion Tri(n) = TMVP(⌊n/2⌋×⌈n/2⌉) +
Tri(⌈n/2⌉) + Tri(⌊n/2⌋) (hi part first, reading the untouched lo part; hi/lo of the whole window first, then recurse into
lo, so all coefficients a_i, i < L/d, are read before they change). A dense p×q Toeplitz block costs p+q−1 GF(4096)
products by evaluation/interpolation (transposed Toom–Cook: y = E_pᵀ((Iᵀt) ⊙ (E_q rev v)), numerically verified for
sizes up to 35×34); every linear part (GF(4096) constants, Frobenius of the operand sums, output fan-out) is CX-only,
so a block needs three scratch cells (operand sums X, Y, product Z erased for free by `del_gf2_mul`). Tri(n) comes out
at ≈ n log₂n − n (balanced split is optimal under this recurrence; the short-product rank 2n−3 is out of reach in place).

| strategy | products one way | shear Toffolis both ways (51/product) | T estimate |
|---|---|---|---|
| naive (v15) | 21,450 | 2,188 k | 2,888 k |
| Karatsuba blocks (0/1 constants) | 11,744 | 1,198 k | 1,898 k |
| evaluation-rank blocks | 8,564 | 874 k | **≈ 1,574 k** |
| + 42-Toffoli multiplier on all 10,979 products/way | 8,564 | 719 k | ≈ 1,380 k |

Per level L = 70 the three passes (d = 2, 4, 16) go from 1,156 / 561 / 116 to 289 / 222 / 95 products. Q: X, Y, Z replace
tmp + gtmp, so the extraction peak moves 903 → ≈ 915, still under the sweep's 927 — no Q change. Circuit size: the
constants cost ≈ 6–9 k CX per product, ≈ +100 M ops (≈ 200 M total, ~1.8 GB .kmb); v15's full run took 2 min 50 s (bit-sliced batches), so runtime is no concern; the build was ≈ 10 min.
Estimated cost at Q 927, T 1.57 M, R 1.226: 1.66e12 (14.7×); with R 1.191 and the 42-multiplier ≈ 1.41e12 (17.3×).
Break-even rule R′Q′²T′ < 3.043e12 is cleared by a factor ≈ 1.9.

**Kernel status (built after the estimate).** `tmvp_block` / `tri_block` / `pass_fast` in build_cf.py (constants from
`tmvp.py`); the hi/lo split must be ⌈L/2⌉ so every coefficient a_i, i ≤ (L−1)/d, stays in the untouched lo half, and
within a block the triangular part is applied to the old values before the hi += T·lo accumulation. Checks: reciprocal
alone exact for n = 2..12 (GF(2⁶)) and n = 35, 69 (GF(2¹²)); full window n = 69: 30,804 Toffolis vs 93,483 naive (×3.03);
toy decoders GF(2⁴) n=8 / GF(2⁶) n=12 unchanged (phase errors only on model-wrong inputs, 0 value errors).

**v16 = `dqi_1666385650802_08243c62.kmb` (build_cf.py, fast shears + one-gap handling): full PASS, score 1.6664e12
(14.67×), Q 927, T 1,589,677.71, 0 value / 7 phase failures (R 1.2199)** — `runs/dqi_1666385650802_08243c62_full.json`.
−45 % vs v15; static count 1,589,768 (estimate 1,574 k); 92.4 M ops / 860 MB (no growth: fewer products' internal CX
offset the constants); smoke 2,560 shots 0/0. Remaining T: shears ≈ 0.87 M, sweep 0.37 M, scaling + conversion 0.25 M,
gap handling + inverses ≈ 0.1 M.
