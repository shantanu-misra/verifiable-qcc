# verifiable-qcc — Kaggle: Google Quantum AI, Verifiable Quantum Circuit Compilation

Competition window: 2026-10-22 → 2027-01-20 (final submission, 23:59 UTC). Entry/team-merge
deadline 2027-01-13. 5 submissions/day, 2 final picks. Winners must open-source under Apache 2.0.

## Goal and tasks

Submit kickmix circuits that pass Google's verifier while minimising cost. Leaderboard score is the
geometric mean of per-task improvement over the reference: `Score = (∏_b cost_ref_b / cost_b)^(1/3)`,
so the baseline scores exactly 1.0 and higher is better.

- **invsqrt** (fusion, paper Benchmark I): map `|x⟩|0⟩ → e^{2πi/√x}|x⟩|0⟩` for every x in 1..3,133,452 held in
  a 22-qubit register named `x`; 64 tests per x; phase error ≤ 2⁻²⁴/√x turns, zero failures allowed. **Priority 1.**
- **dqi** (optimisation, Benchmark II): reversible RS(4095,4025) decoder over GF(2¹²); syndromes in registers
  `S0..S69`, classical record `M` in `E0..E4094`; must restore syndromes and apply phase `(−1)^{Σ_j popcount(e_j ∧ M_j)}`;
  102,400 shots; failures allowed but raise cost. **Priority 2.**
- **dfthcsos** (chemistry, Benchmark III): FeMoCo angle-loading QROMs + factorisation in `submission.npz`.
  **Submit the provided baseline unchanged. No effort here.**

## Cost formulas (from the Kaggle Evaluation page; Q = qubits, T = Toffolis, per shot)

| task | cost |
|---|---|
| invsqrt | `Q × C`, `C = Toffolis + (T gates)/4`. Clifford Z_POW free; T-angle = 1 T; finer rotations priced analytically at `0.56·log2(1/ε) + 5.3` T each (paper §III.A). Baseline: Q=260, 7,735 Toffolis, C=7,988, Q×C ≈ 2.08e6. |
| dqi | `R(ε⁺) × Q² × T`, `R(ε) = ln2 / (−ln(1/2 + 2√ε))`; ε⁺ = one-sided Clopper-Pearson upper bound at δ=2⁻¹²⁸ (≈88.7/N with 0 failures → R≈1.19); ε⁺ ≥ 1/16 ⇒ no finite score. Baseline: Q=1,893, T=5,725,461. |
| dfthcsos | `λ_eff × T × Q` over the whole block encoding, `λ_eff = √(E_gap(2Λ − E_gap))`; rejected if any ANGLE circuit > 128 qubits, Λ ≤ 49.0 or E_gap ≤ 3.914898 Ha, or encoded H > 20 mHa from reference. |

Confirmed from `data/verifiers/*/verify_submission_*.py`: **Q = `circuit.num_qubits`** (every qubit id the circuit
mentions, not peak allocation — reuse ids!), T = mean Toffoli-like (CCX/CCZ/CSWAP) executed per simulated shot.
invsqrt: register schema must be exactly one register `x` of 22 distinct qubits; everything else is scratch.
dqi: registers exactly `S0..S69` (12 qubits each) and `E0..E4094` (12 classical bits each); **Z_POW is forbidden**.
The `data/verifiers/` directory is the authoritative spec; the docstrings at the top of each script restate the contract.

## Directory layout (proposed)

```
CLAUDE.md
Dockerfile                 # builds kickmix from the pinned Qualtran commit + verifier deps
docker-compose.yml         # mounts repo + data/ into the container
data/                      # Kaggle competition data, NOT committed (gitignored)
  sample_submission/       # invsqrt.kmb dqi.kmb dfthcsos/{angle_k_0..9.kmb,submission.npz}
  verifiers/               # official verifiers + reference-circuit scripts (read-only, never edit)
  kickmix/                 # kickmix.so built for the Kaggle image (x86_64 linux only)
third_party/Qualtran/      # git submodule pinned (see Build); only kickmix/ and docs/kickmix/ matter
runs/                      # raw verifier JSON outputs, committed (small) — the evidence behind every score
tasks/invsqrt/             # generators (Python, km.CircuitBuilder) + notes.md
tasks/dqi/
tasks/dfthcsos/            # baseline passthrough only
circuits/<task>/           # committed .kmx/.kmb outputs, named <task>_<cost>_<shortsha>.kmx
submissions/               # zipped bundles actually uploaded, with the Kaggle score in a sidecar .txt
scripts/verify.sh          # thin wrapper: run official verifier in Docker on one circuit
scripts/bundle.sh          # assemble submission.zip from circuits/ + baseline dfthcsos
```

## Build and run (Docker)

Qualtran ships no Dockerfile; ours is `Dockerfile` + `docker-compose.yml` (service `dev`, repo mounted at `/work`,
forced `linux/amd64`). It `pip install`s `third_party/Qualtran/kickmix` (pybind11 2.12.0, C++20, Python ≥ 3.10,
numpy) from the submodule pinned at **`638777fac4379bc62e2a1e4a52cf1a1753895d48`** (2026-10-22, kickmix 0.1).
```bash
git submodule update --init --depth 1              # once, after clone
docker compose build                                # ~minutes under emulation on Apple Silicon
docker compose run --rm -T dev python -c "import kickmix as km; print(km.__file__)"
docker compose run --rm -T dev python <script.py>   # anything else; cwd is /work
```
Verified 2026-10-23: the source build reproduces the paper's baseline stats exactly (see Cost table), so it is our
reference simulator. The Kaggle-shipped `data/kickmix/kickmix.so` needs `GLIBCXX_3.4.32` (gcc ≥ 13) and does not
load on bookworm; TODO only if we ever see a discrepancy: rebuild on a trixie/ubuntu-24.04 base and compare.
Upstream dev flow for reference (`docs/kickmix/dev.md`): Bazel 7.2.0, `bazel test :all`, `bazel run kickmix:kickmix`
for the C++ CLI, `PYTHONPATH=kickmix pytest kickmix/src`. A PyPI `kickmix` is announced but unconfirmed.

## Running the verifier and reading output

Official verifiers (what the leaderboard runs): `data/verifiers/invsqrt/verify_submission_invsqrt.py`,
`data/verifiers/dqi/verify_submission_dqi.py`, `data/verifiers/dfthcsos/verify.py` (+ `SCORING_README.md`).
Seeds = SHA-256 of the canonical `.kmb` bytes (+ `--salt`, private on the leaderboard), so a local pass is strong
evidence, not identity; `--seed N` overrides for debugging only.
```bash
scripts/verify.sh invsqrt circuits/invsqrt/<file>.kmx            # full run: 200,540,928 shots (~10 min in Docker on M-series)
scripts/verify.sh invsqrt circuits/invsqrt/<file>.kmx --max-x 4096   # smoke test (<1 s); NOT a reportable score
scripts/verify.sh dqi     circuits/dqi/<file>.kmb                # 102,400 shots; --max-workers 4 default, ~6 GB RAM each
```
`scripts/verify.sh` wraps `docker compose run … python <verifier> <circuit> [args]` and saves stdout to `runs/`.
Output is JSON. PASS → `{"status":"PASS","score":…,"num_qubits":…,"toffoli_count":…,` plus for invsqrt
`"toffoli_equivalent"` (=C), `"t_count"`, `"rotation_count"`, `"rotation_eps"`, `"worst_error_over_tolerance"`,
`"worst_x"`; for dqi `"value_errors"`, `"phase_errors"`, `"effective_error_rate"` (=ε⁺), `"repetition_factor"`
(=R), `"num_bits"`; both give `"reaction_depth"`, `"num_ops"`, `"circuit_sha256"`. FAIL → exit 1 and
`{"status":"FAIL","error":"x=729: phase error … exceeds tolerance …"}` / `"Scratch qubit q17 did not return to |0>"`
/ `"Registers must be exactly S0..S{n-1} and E0..E{m-1}."` etc. The reported `score` is the per-task cost (lower is
better); the leaderboard number is `(∏ baseline/score)^(1/3)` across the three tasks.
Baseline verified locally 2026-10-23 (`--max-x 4096`): invsqrt score 2,069,865.06, C=7,961.02, Q=260 — use the
full-run figure in `runs/invsqrt_baseline_full.json` as the denominator.
Useful kickmix-native checks (confirmed APIs) that are NOT the verifier:
```python
import kickmix as km
c = km.Circuit.from_file("foo.kmx")           # or format="kmb"; c.to_file(path, format="kmb") to save
s = c.operation_statistics()                   # s.num_toffoli_like, s.num_t_like, s.num_non_clifford,
print(c.num_qubits, s.num_toffoli_like)        # s.num_z_pow(min_clifford_hierarchy_level=4), s["CCX"]
sim = km.Simulator(batch_size=256, seed=1); sim.use_same_registers_as(c); sim.do(c)
sim.read_within_shot("x", 0, out=int); sim.read_shot_phase(0)   # Fraction of a turn
```
C++ CLI subcommands: `circuit`, `diagram`, `count_operations [--in F] [--out F] [--groups] [--magic]`,
`count_operations_sampled`, `kmb2kmx`, `kmx2kmb`, `sample [--shots N] [--init S] [--decimal|--hex] [--in F]`, `spin`.
`count_operations --magic` counts CCX+CCZ+CSWAP+Z_POW (incl. `_IF` forms) — not the scored Toffoli count.

Kickmix gate set: X CX CCX SWAP CSWAP · Z CZ CCZ Z_POW NEG · R HMR · BIT_INVERT BIT_STORE0/1 ·
PUSH_CONDITION/POP_CONDITION, per-op `if bN` · REGISTER APPEND_TO_REGISTER INFO_TAG DEBUG_PRINT.
`R` on a |1⟩ qubit randomises the sign, so dirty ancillas fail verification. Z_POW exponent t phases by e^{iπt},
kept to 2⁻¹²⁷ precision (never round through a double).

## Rules (non-negotiable)

1. **Never report a score that was not reproduced by running the official verifier from `data/verifiers/` in this
   repo's Docker container.** `operation_statistics()` numbers are estimates; say so when quoting them.
2. **Never modify anything under `data/verifiers/` or `data/kickmix/`.** Treat them as read-only; if a verifier
   seems wrong, note it in `tasks/<task>/notes.md` and raise it on the Kaggle forum, do not patch.
3. **Every improved circuit is committed** to `circuits/<task>/` with the commit message carrying the verified
   numbers, e.g. `invsqrt: Q=240 T=7100 C=7102.3 cost=1.70e6 (verifier pass, 0 failures)`. Generator code goes in
   the same commit. Submissions uploaded to Kaggle get the public LB score recorded in `submissions/`.
4. Lower is better per task; the leaderboard is a geometric mean, so a 2× win on one task is worth as much as
   2× on any other. dfthcsos stays at baseline (ratio 1.0).
5. Follow the kickmix contract: inputs restored, all scratch qubits back to |0⟩, phase within tolerance. Any
   `R`/`HMR` trick is allowed only if the verifier accepts it.
6. No private code sharing outside the team; anything shared must go to the public forum (Kaggle rule C.5.d).

## Links

- Competition: https://www.kaggle.com/competitions/google-verifiable-quantum-circuit-compilation
  (Data page lists `verifiers/`, `sample_submission/`, `kickmix/`; Rules page for licensing/limits)
- dfthcsos starter kit (not needed unless we revisit chemistry):
  https://www.kaggle.com/datasets/sohier/verifiable-circuit-compilation-starter-kit
- Qualtran repo: https://github.com/quantumlib/Qualtran — kickmix source: `kickmix/`
- kickmix docs: https://github.com/quantumlib/Qualtran/tree/main/docs/kickmix
  (`kickmix_format.md`, `kickmix_binary_format.md`, `kickmix_instructions.md`, `api_reference.md`, `dev.md`, `conventions.md`)
- Companion paper (Khattar, Rubin, Gidney, Broughton, Low, Babbush): https://arxiv.org/abs/2610.23832
  (HTML: https://arxiv.org/html/2610.23832 — §III.A invsqrt, §III.B dqi, Eq. 43 R(ε), Eq. 44 dfthcsos)
- Announcement: https://opensource.googleblog.com/2026/10/announcing-verifiable-quantum-circuit-compilation-competition-on-kaggle.html
- Prior kickmix paper (ECC cost estimate): https://arxiv.org/abs/2603.28846
