# Submissions: packaging, metric-only checks, upload record

## Packaging

`scripts/bundle.sh <invsqrt circuit|baseline> <dqi circuit|baseline> [out.zip]` → `submissions/submission.zip`
(default name; Kaggle accepts any) plus a timestamped sidecar `submissions/submission_<utc>.txt` with the zip and
circuit hashes. Required layout (Overview › Submission File): `invsqrt.kmb|kmx`, `dqi.kmb|kmx`,
`dfthcsos/angle_k_0..angle_k_<n-1>.kmb|kmx` (gapless) + `dfthcsos/submission.npz`. `.kmx` is accepted for circuits.
Upload by hand on the competition page (Submit Prediction); 5 per day, 2 final picks; scoring takes ~45 min.

## What the metric checks that the standalone verifiers do not

Read from the official metric notebook (`metric/google-quantum-ai-metric`, 2026-10-24). A failure here rejects the
whole submission before any simulation, so a local `"status": "PASS"` is necessary, not sufficient.

- **Exactly one** `invsqrt.*`, one `dqi.*` and one `dfthcsos/` directory anywhere in the zip (searched
  recursively, `__MACOSX` ignored); a stray backup copy is a rejection. Every member must be a regular file.
- **Largest qubit or bit id** must be below `MAX_QUBIT_OR_BIT_ID = 2**30 // 1536 = 699,050` per circuit
  (kickmix itself would accept far larger ids). Baselines stay below 50,000; the hosts say the cap can be raised.
- **Registers**: at most the benchmark's maximum (invsqrt: 1 — `x`; dqi: 70 + 4095), and **every register must be
  named** via its `REGISTER` line (the dfthcsos verifier crashes on unnamed ones).
- Only instructions that `Circuit.operation_statistics()` knows are allowed (`unsupported instructions …`).
- `dfthcsos/`: `angle_k_0 … angle_k_<n-1>` each exactly once; `submission.npz` arrays must be plain little-endian
  number arrays of the specified numpy kinds/shapes (floats for the factorization, integer `tree_edges`), `.npy`
  format 1.0 only, total bytes capped — the standalone verifier is more lenient (accepts integer `H1_I = 0` or
  big-endian). The provided baseline passes.
- Scoring: per leaderboard split (public ≈ one half, private the other) each benchmark is verified once with that
  split's own ≥ 32-char secret salt merged into the circuit hash; `baseline_cost` is the reference submission's
  cost under the same salt; the split's score is `prod(baseline_cost / cost) ** (1/3)`. A failed verification
  surfaces as `The <benchmark> submission failed verification: …`; non-positive cost is rejected too.

Our generated circuits: single named register `x`, ids < 200 — all of the above hold; `scripts/bundle.sh` keeps the
layout. If a generator ever allocates ids sparsely, check the id cap, not just `num_qubits`.
- **Format trap** (reported by a public notebook that lost a submission to it): `Circuit.to_file(path)` writes
  *text* unless `format='kmb'` is passed, whatever the suffix; the standalone verifier auto-detects, but the metric
  picks the format from the suffix and a text file named `.kmb` fails the whole submission with no message. We
  write `.kmx`; if we ever ship `.kmb`, always pass `format='kmb'`.

## Upload record

| date (UTC) | zip sha256 | invsqrt | dqi | dfthcsos | local expected | public LB | rank |
|---|---|---|---|---|---|---|---|
| 2026-10-24 | 225d0487… | `invsqrt_488144_1b46306c.kmx` (488,144.07) | baseline | baseline | 1.6204 | **1.62036** | 392 / 521 |
| 2026-10-24 | 10c86bf6… | `invsqrt_165584_2d7c9f6e.kmx` (165,583.96) | `dqi_3725395305991_c3edfdbf.kmb` (3.7254e12, 1 failure locally) | baseline | 4.3494 (4.356 if the salted dqi run has 0 failures) | **4.35605** | n/a (top score 18.86345) |

The second upload scored 4.35605 against a local 4.3494: the difference is exactly the dqi repetition factor going
from R(1 failure) = 1.1967 to R(0 failures) = 1.1913, i.e. the salted shot stream drew none of the rare j = 3 inputs
(`tasks/dqi/notes.md` §5). The first upload's public score equalled the local prediction to five digits, so the salted leaderboard runs agree with our
local verifier runs (the per-salt `baseline_cost` for invsqrt is evidently the same as our unsalted one; the
verifier runs every x, so only `HMR`/`R` outcomes depend on the seed).

## Leaderboard snapshot, 2026-10-24 (public, `kaggle competitions leaderboard --download`)

521 scored teams. #1 16.62, #5 13.98, #10 11.27, #20 9.35, #50 8.01, #100 6.38, #200 5.79, #250 5.66,
median 5.64, #300 4.36, 45 teams at exactly 1.0. The dense band at 5.6–5.7 is the public-notebook portfolio
(e.g. "Verified portfolio: best public circuit per task" = 5.63791, "QROM + Tower42" = 5.64746), i.e. roughly a
180× product of ratios available from public circuits alone; the top 50 are above 8 (> 500× product).
