# Public state of the art (as of 2026-10-24) and our gap

Source: three public Kaggle notebooks read in the browser on 2026-10-24 — kirilltym "QROM + Tower42" (v10 public
5.64746, newer archive projects 5.7247), rudrabhaskar "Highest Public Score | Direct Polynomial Phasing" (5.54184),
alexchilton "Where the Toffolis Go: Anatomy + a Cheaper invsqrt" (5.75601). **Every number below is what those
notebooks report about themselves; none has been reproduced by our verifier runs** (rule 1). The public median
(5.64) is this portfolio; ~half the field submits it.

## Per task

| task | public best cost | ratio vs baseline | Q | Toffolis | our best | our ratio |
|---|---|---|---|---|---|---|
| invsqrt | 165,026 (alexchilton v6) | 12.58× | 93 | 1,470 (+44 rotations) | 488,144 | 4.25× |
| dqi | 2.297e12 (kirilltym, from Kozykorpesh/k256 lineage) | 10.64× | 1,818 | 583,505 | baseline | 1.0× |
| dfthcsos | 2.345e9 (kirilltym v10 banked QROM) | 1.43× | 681 (block encoding) | 72,514 | baseline | 1.0× |

Log-score shares of the public portfolio: invsqrt ≈ 48 %, dqi ≈ 45 %, chemistry ≈ 7 % (1.43× is only ×1.13 on the
final score). Matching the portfolio on all three gives ≈ 5.7; the top 50 (> 8.0) need a > 500× product and
#1 (16.6) ≈ 4,600×, which a forum thread reads as ~20× on invsqrt and/or dqi beyond the public circuits.

## invsqrt: the public design (Sumit Chongder → anazemcev → Rudra → kirilltym → Alex)

Readable generators are public (Apache-2.0 labelled; the original author's licence line is "not asserted").
Ideas, in order of value to us (our circuit: Q 147, 3,074 Toffolis; 2,016 of them in `_add_row` Horner rows):

1. **Signed-digit Horner rows without partial products.** Write δ = Σ_i (2d_i − 1)·2^(i−22) (centred digits ±1);
   then every row is an *unconditional* add of ±coef·2^k, with the sign applied by CX-conjugating the operand from
   d_i (Y − A = ~(~Y + A)). A row costs one Gidney adder (n − 1 Toffolis, carries measured away), no AND gates, no
   product register. Their Horner + shifter total is 1,201 Toffolis vs our 2,016 + 290·2 denormalize.
2. **Mantissa in [1,2) with a parity flag** (five binary normalise stages, shifts 16/8/4/2/1; flags = shift count),
   QROM of 256 entries addressed by 7 mantissa bits + parity (128 segments per parity), quadratic per segment fitted
   by weighted minimax (Lawson). Coefficients XORed straight into the working registers.
3. **Static high word of y from the QROM** (8 bits): the rows touch only a 21-bit low register; the high word is
   phased immediately after the lookup, erased by measurement-based unlookup of just those bits, and its qubits
   freed before the arithmetic ("static-high phase checkpoint"; the two diagonal phases add, no join needed).
4. **Narrow registers**: c2 8–9 bits, derivative split into a 12-bit residual + 7-bit QROM-supplied coarse part,
   top-carry reuse (n − 2 clean carries), x[21] (always 1 after normalisation, never read) borrowed as a carry wire.
   Budget: 22 x + 5 flags + 8 c2 + 12 + 7 + 21 low + 18 carries = 93.
5. Small ones: moves into still-|0⟩ shifter qubits as temp-AND + CNOT instead of CSWAP (−20), one prefix-AND chain
   for all "top k bits zero" flags (−6), one coefficient bit fewer (F2 10 → 9) at worst err/tol 0.945.

Reported dead ends (Alex): cubic × 32 segments (−207 lookup, +450 rows), 64 or 256 segments, phase-gradient
register (same 33 rotations + 30 qubits), in-place Cuccaro adder (−19 Q, +560 Toffolis), spending more margin on
rotations (≈ −11 Toffoli-eq). Alex calls 93 q / 1,470 "near the floor of the design"; one qubit ≈ 1 % of cost,
one Toffoli ≈ 0.05 %.

## dqi: the public design

Berlekamp–Massey replaces the reference's forward + reverse extended Euclid. Reported stage counts (Toffolis):
bm_forward 147,915 (70 iterations × 2,114), monicize/ω/ω-restore/unmonicize 1,536/6,174/6,174/1,536, **sweep
327,600 = 4,095 positions × 80** (zero test 11 + GF(2¹²) inversion as a GF(64)/GF(8) tower 69; the ⟨ω·inv, M_j⟩
kickback is CZ-only, free), bm_reverse 92,570. Classical-constant GF multiplications are GF(2)-linear → 0 Toffolis.
**Q stayed at 1,818 (reference 1,893)**: live qubits sit at 96–100 % in every stage, so the next factor must come
from fewer live polynomial registers, and Q enters squared. The public dqi circuit is a hash-pinned binary whose
original generator (k256) is no longer public; prize eligibility of such inputs is an open question on the forum.

## dfthcsos: the public design

Banked XOR QROM over GF(2)-compressed sparse-SOS factors (J = 5, brot = 10, 15 ANGLE circuits, 31,807 Toffolis
total vs 39,740) → 1.43×. Cost-model table from Alex (accuracy not checked): the same shape at brot 7 would be
2.96× because temp qubits fall from 543 to 272 ("cost cliff" at brot ≤ 7), but the public angle words sit at
19.98 mHa of the 20 mHa budget and plain rounding to fewer bits fails (188 mHa at brot 9); a quantisation-aware
refit with the starter kit's JAX trainer is the untested 2× he points at.

## Gap arithmetic for us (our numbers, from our verified scores)

- Current LB 1.62036 = (4.2544)^(1/3). Matching public invsqrt only → 2.33. + public-level dqi (10.6×) → 5.1.
- Top 50 needs e.g. invsqrt 15× · dqi 30× · chem 1.15×; top 10 e.g. 20× · 50× · 1.4×.
- dqi is the lever: 10× is already public with Q untouched; Q² means a 1,818 → 1,200 Q cut alone is 2.3×.
- invsqrt: adopting ideas 1–4 in our own generator should take us from 3,074/147 to the public ~1,470/93 band
  (×1.44 on the final score); passing it needs a new idea, not tuning.
