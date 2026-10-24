# invsqrt (fusion) — paper Benchmark I

Authoritative spec: `data/verifiers/invsqrt/verify_submission_invsqrt.py` (its docstring restates the contract).
Paper: arXiv 2610.23832 §III.A. Reference generator: `data/verifiers/invsqrt/nr_reference_circuit.py`.

## Contract

- Map `|x⟩|0…0⟩ → exp(2πi·β/√x) |x⟩|0…0⟩` with β = 1 for **every** x in [1, max_x],
  max_x = 3·(2^n_p − 2)² = **3,133,452** (n_p = 10; x = |ν − ν′|² for momenta with n_p-bit components).
- Register schema: exactly one register, named `x`, of width 2·n_p + 2 = **22 distinct qubits** (`km.q`,
  little-endian). No other registers. Every other qubit id the circuit mentions is scratch and must be |0⟩ at the end.
- Phase check: tracked phase θ must satisfy |θ − β/√x| ≤ **tol(x) = 2⁻²⁴·β/√x turns** (mod 1), compared in 64-bit
  fixed point. **Zero failures allowed** — the first failing shot aborts the run with `FAIL`.
- Shots: every x is run `shots_per_x` = 64 times → **200,540,928 shots**, in batches of 256, chunks of 64 batches
  across all CPU cores. `HMR`/`R` outcomes are seeded from SHA-256 of the canonical `.kmb` bytes (+ `--salt`).

## Cost

`score = Q × C` — lower is better; leaderboard contribution = 2,076,774.23 / score.

- **Q = `circuit.num_qubits`**: the number of distinct qubit ids the circuit mentions, not peak allocation.
  Reusing ids after `R`/`HMR` lowers Q; allocating a fresh id for each ancilla raises it.
- **C = Toffolis + T/4**, where Toffolis = mean `CCX`/`CCZ`/`CSWAP` executed per simulated shot (so branches skipped
  by measurement outcomes are not charged) and
  T = (#Z_POW with T angles) + n_rot × (0.56·log2(1/ε) + 5.3).
  Z_POW angle classes (`z_pow_class`): exponent ≡ 0 mod 1/2 → Clifford, free; ≡ 0 mod 1/4 → T, 1 T each;
  anything else → rotation, priced by mixed fallback synthesis (arXiv 2203.10064).
- **ε = π · min_x(tol(x) − |error(x)|) / n_rot**: the phase-tolerance margin the simulation leaves over, split
  evenly across the n_rot rotations (n_rot = `num_z_pow(min_clifford_hierarchy_level=4)`). Using more of the
  tolerance makes every rotation dearer; using all of it fails with `No tolerance left for rotation synthesis error`.

## Running

```bash
scripts/verify.sh invsqrt circuits/invsqrt/<file>.kmx                 # full run, ~4 min in Docker on M-series
scripts/verify.sh invsqrt circuits/invsqrt/<file>.kmx --max-x 4096    # smoke test (<1 s); NOT a reportable score
```
Flags: `--format {auto,kmb,kmx}`, `--n-p`, `--beta`, `--precision-bits` (24), `--shots-per-x` (64), `--eta` (218),
`--width`, `--max-x`, `--system-qubits`, `--salt`, `--seed` (debug only), `--batch-size` (256), `--max-workers`.
Output is JSON on stdout; `scripts/verify.sh` also saves it under `runs/`.

| field (PASS) | meaning |
|---|---|
| `score` | Q × C |
| `num_qubits`, `toffoli_equivalent` | Q, C |
| `toffoli_count` | mean Toffoli-like gates executed per shot |
| `max_toffoli_count` | static count from `operation_statistics()` |
| `t_count`, `rotation_count`, `rotation_eps` | T-angle Z_POWs, n_rot, ε |
| `worst_error_over_tolerance`, `worst_x` | max |error|/tol over all shots and the x where it happened |
| `reaction_depth`, `num_ops`, `verified_shots`, `circuit_sha256` | diagnostics |

FAIL → exit 1 and `{"status": "FAIL", "error": …}` with one of: `x was not restored.`,
`Scratch qubit q<N> did not return to |0>.`, `x=<N>: phase error … exceeds tolerance …`,
`Invalid register schema: […]`, `Expected 22 x qubits, got …`. Each was exercised deliberately in
`results/verifier_checks.md` (`scripts/verifier_checks.py`).

## Baseline — full official run, 2026-10-23 (`runs/invsqrt_baseline_full.json`)

| field | value |
|---|---|
| status | PASS, 200,540,928 shots, 4 min 9 s (Docker, M-series) |
| **score** | **2,076,774.233715282** |
| Q | 260 |
| Toffolis / shot | 7,735.0 (static 7,735) |
| T / rotations / ε | 1 / 37 / 1.529e-12 → 27.1 T per rotation |
| C | 7,987.593206597238 |
| worst error/tol | 0.477 at x = 2,588,411 |
| ops / reaction depth | 62,160 / 4,005 |
| sha256 | e2bf8eb64350a4460a271fc7f1d591ef125b1e276377e9a0f64b96ab3ca321e1 |

Recomputed from the JSON: 260 × 7987.593206597238 = 2076774.233715282 = reported score (exact); C rebuilt from
`toffoli_count + (t_count + rotation_count·(0.56·log2(1/rotation_eps) + 5.3))/4` is also exact.

Slack worth knowing: the baseline uses 48% of its phase tolerance at the worst x, and the 37 rotations cost
~250 of the 7,988 Toffoli-equivalents. Static and executed Toffoli counts coincide, so no measurement-dependent
branching is being exploited yet.
