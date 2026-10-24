# dqi (combinatorial optimisation) — paper Benchmark II

Authoritative spec: `data/verifiers/dqi/verify_submission_dqi.py` (its docstring restates the contract).
Paper: arXiv 2610.23832 §III.B, Eq. 43; background arXiv 2408.08292, 2510.10967.
Reference generator: `data/verifiers/dqi/dqi_reference_circuit.py`.

## Contract

- Reed–Solomon code with m = **4095** positions, n = **70** syndromes, b = **12** (GF(2¹²) via `km.GF2Field(12)`,
  primitive element α), correcting up to l = n // 2 = **35** errors; r = |F_i| = 2016 accepted values per constraint.
- Register schema, pairwise disjoint, nothing else allowed:
  `S0..S69` — 12 qubits each (`km.q`, little-endian), holding syndromes S_i = Σ_j e_j·α^{(i+1)·j};
  `E0..E4094` — 12 classical bits each (`km.b`), holding the MBUC record M_0..M_4094.
- Per shot: restore `S0..S69`, return every other qubit to |0⟩, and multiply the global phase by
  **(−1)^{Σ_j popcount(e_j ∧ M_j)}** for the withheld error e. **`Z_POW` gates are forbidden** (rejected statically).
- Test distribution: error weight k ∈ {0..35} with probability w_k² (principal eigenvector of the DQI tridiagonal
  matrix; mean weight 31.05), k uniform positions, uniform non-zero values, uniform M_j. **102,400 shots**
  (400 × 256); the input RNG and `HMR`/`R` outcomes are seeded from SHA-256 of the canonical `.kmb` (+ `--salt`).
- Failures are allowed: k = `value_errors` (syndrome or scratch wrong) + `phase_errors` (value right, sign wrong).
  ε⁺ = one-sided Clopper-Pearson upper bound on the failure rate at δ = 2⁻¹²⁸; **ε⁺ ≥ 1/16 ⇒ FAIL**.

## Cost

`score = R(ε⁺) × Q² × T` — lower is better; leaderboard contribution = 24,439,463,583,672.93 / score.

- **R(ε) = ln 2 / −ln(1/2 + 2√ε)**, R(0) = 1. With zero failures ε⁺ = 1 − 2^(−128/N) ≈ 8.66e-4, so
  **R ≥ 1.19126 for any submission**: a perfect decoder gains nothing on R; a few failures cost little
  (ε⁺ ≈ 1.2e-3 at k = 1).
- **Q = `circuit.num_qubits`** (distinct qubit ids mentioned; the 49,154 classical bits are free). Q enters
  squared, so a qubit saved is worth about twice a proportional Toffoli saving.
- **T** = mean Toffoli-like (`CCX`/`CCZ`/`CSWAP`) gates executed per shot, including cleanup; the static
  `max_toffoli_count` is reported alongside.

## Running

```bash
scripts/verify.sh dqi circuits/dqi/<file>.kmb                   # 102,400 shots; ~3.5 min, --max-workers 2
scripts/verify.sh dqi circuits/dqi/<file>.kmb --max-workers 2   # default is 4; 2 was <2 GB RSS total here
```
Flags: `--format`, `--m` (4095), `--n` (70), `--degree` (12), `--r` (2016), `--num-shots` (102,400, multiple of
`--batch-size` 256), `--log2-delta` (−128), `--salt`, `--seed` (debug only), `--max-workers` (4).
Loading the 886 MB baseline `.kmb` (86 M instructions) dominates small runs.

| field (PASS) | meaning |
|---|---|
| `score` | R × Q² × T |
| `num_qubits`, `num_bits` | Q (scored), classical bits (free) |
| `toffoli_count`, `max_toffoli_count` | mean executed per shot, static count |
| `value_errors`, `phase_errors`, `empirical_error_rate` | failure tallies, k / N |
| `effective_error_rate`, `repetition_factor` | ε⁺, R(ε⁺) |
| `shot_operation_counts` | total executed count per op type across all shots |
| `mean_error_weight`, `m`, `n`, `degree`, `r`, `log2_delta` | instance parameters |
| `reaction_depth`, `num_ops`, `verified_shots`, `circuit_sha256` | diagnostics |

FAIL → exit 1 and `{"status": "FAIL", "error": …}`: `Registers must be exactly S0..S{n-1} and E0..E{m-1}.`,
`Z_POW gates are not allowed in Benchmark II.`, `Effective error rate … >= 1/16 …`, width/overlap errors.

## Baseline — full official run, 2026-10-23 (`runs/dqi_baseline_full.json`)

| field | value |
|---|---|
| status | PASS, 102,400 shots, 3 min 26 s with `--max-workers 2` (Docker, M-series) |
| **score** | **24,439,463,583,672.93 (2.444e13)** |
| Q / bits | 1,893 / 49,154 |
| T / shot | 5,725,108.382 (static 5,725,461) |
| failures | 0 value, 0 phase → ε⁺ = 8.6606e-4, R = 1.1912604648 |
| ops / reaction depth | 86,051,014 / 170,757 |
| sha256 | 959471d05a26373b4f14ca0cae2d8639ea25f18170d9f6532139aa51ccc04995 |

Recomputed from the JSON: 1.1912604648115832 × 1893² × 5725108.382109375 = 24439463583672.93 = reported score
(exact), and R(ε⁺) from the formula matches `repetition_factor` exactly.
