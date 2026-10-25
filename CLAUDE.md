# verifiable-qcc — Kaggle: Google Quantum AI, Verifiable Quantum Circuit Compilation

Window 2026-10-22 → 2027-01-20 (final submission 23:59 UTC); entry/team-merge deadline 2027-01-13; 5 submissions/day,
2 final picks; winners open-source under Apache 2.0. This file is an index: task detail is in `docs/`, evidence in `runs/`.

## Goal and tasks

Submit kickmix circuits that pass Google's verifier while minimising cost. Leaderboard score is the geometric
mean of per-task improvement over the reference, `Score = (∏_b cost_ref_b / cost_b)^(1/3)`: the baseline scores
1.0, higher is better, and a 2× win on any one task is worth the same as on any other.

| task | one line | priority | detail |
|---|---|---|---|
| **invsqrt** (fusion, Benchmark I) | phase `\|x⟩` by e^{2πi/√x} for every x in 1..3,133,452; cost Q×C | 1 | `docs/invsqrt.md` |
| **dqi** (optimisation, Benchmark II) | reversible RS(4095,4025) syndrome decoder over GF(2¹²); cost R(ε⁺)×Q²×T | 2 | `docs/dqi.md` |
| **dfthcsos** (chemistry, Benchmark III) | FeMoCo angle-loading QROMs; cost λ_eff×T×Q | 3 | `tasks/dfthcsos/notes.md` (spec: `data/verifiers/dfthcsos/SCORING_README.md`) |

Verified baselines (official verifiers, this repo's Docker image, 2026-10-23): invsqrt **2,076,774.23**, dqi **2.444e13**,
dfthcsos **3,355,599,588.32** — `runs/*_baseline_full.json`, recomputed in `results/verifier_checks.md`. In invsqrt/dqi
**Q = `circuit.num_qubits`** (every distinct qubit id mentioned, not peak allocation — reuse ids), T = mean Toffoli-like gates
executed per shot. dfthcsos scores a *family* of QROM circuits plus the factorisation they load (`tasks/dfthcsos/notes.md`:
cost model, the 128-qubit cap, why the angle width needs a quantisation-aware refit). `data/verifiers/` is the spec.

## Directory layout

```
CLAUDE.md                  # this index
docs/{invsqrt,dqi}.md      # contract, cost formula, verifier flags/output fields, verified baseline per task
docs/submission.md         # zip layout, metric-only checks, upload record + leaderboard snapshots
Dockerfile, docker-compose.yml   # linux/amd64 image, kickmix built from the submodule; repo mounted at /work
data/                      # Kaggle competition data, gitignored: sample_submission/, verifiers/ (read-only), kickmix/
third_party/Qualtran/      # git submodule, pinned; only kickmix/ and docs/kickmix/ matter
runs/                      # raw verifier JSON outputs (committed): the evidence behind every score
results/                   # verifier negative checks and other recorded experiments
tasks/<task>/              # generators (Python, km.CircuitBuilder) + notes.md; dfthcsos: build_qrom.py + refit.py (native numpy)
circuits/<task>/           # committed .kmx/.kmb outputs, named <task>_<cost>_<shortsha>.kmx (dfthcsos: a directory)
submissions/               # zipped bundles actually uploaded + sidecar .txt with the Kaggle score
scripts/                   # verify.sh (official verifier in Docker → runs/), bundle.sh (zip; 3rd arg = dfthcsos dir), verifier_checks.py
```

## Build and run (Docker)

Qualtran ships no Dockerfile; ours `pip install`s `third_party/Qualtran/kickmix` (pybind11 2.12.0, C++20,
Python ≥ 3.10, numpy; plus scipy for the dfthcsos verifier) from the submodule pinned at **`638777fac4379bc62e2a1e4a52cf1a1753895d48`** (2026-10-22,
kickmix 0.1), forced to `linux/amd64`. The source build reproduces the paper's baseline stats exactly.
```bash
git submodule update --init --depth 1              # once, after clone
docker compose build                                # minutes under emulation on Apple Silicon
docker compose run --rm -T dev python <script.py>   # cwd is /work; -T for non-interactive use
```
A git worktree has no `data/` or submodule checkout: put `VQCC_DATA=/abs/path/to/verifiable-qcc/data` in a gitignored `.env`
and build the image with the main checkout as context (`docker build --platform linux/amd64 -t verifiable-qcc:dev -f Dockerfile <main checkout>`).
The dfthcsos metric math (no kickmix) also runs natively: `tasks/dfthcsos/dfthc_tools.py` needs only numpy/scipy.
The Kaggle-shipped `data/kickmix/kickmix.so` needs `GLIBCXX_3.4.32` (gcc ≥ 13) and does not load on bookworm;
TODO only if a discrepancy ever appears: rebuild on a trixie/ubuntu-24.04 base and compare.

## Verifier

```bash
scripts/verify.sh invsqrt circuits/invsqrt/<file>.kmx [--max-x 4096]   # full ≈4 min; --max-x = smoke test only
scripts/verify.sh dqi     circuits/dqi/<file>.kmb [--max-workers 2]     # ≈3.5 min
scripts/verify.sh dfthcsos circuits/dfthcsos/<dir>                      # dir = angle_k_*.kmb + submission.npz, ≈4 min
```
Wraps `docker compose run … python data/verifiers/<task>/verify_submission_<task>.py <circuit> [args]` and saves
the JSON to `runs/`. PASS → `{"status":"PASS","score":…}` (score = per-task cost, lower is better); FAIL → exit 1
and `{"status":"FAIL","error":…}`. Seeds come from SHA-256 of the canonical `.kmb` plus a salt that is private on
the leaderboard, so a local pass is strong evidence, not identity. Fields: `docs/<task>.md`; packaging, metric-only rejections, uploads: `docs/submission.md`.

## kickmix cheat sheet

```python
import kickmix as km
c = km.Circuit.from_file("foo.kmx")            # or format="kmb"; c.to_file(path, format="kmb") to save
s = c.operation_statistics()                    # s.num_toffoli_like, s.num_t_like, s.num_z_pow(min_clifford_hierarchy_level=4)
print(c.num_qubits, c.num_bits, len(c))         # scored Q, free classical bits, instruction count
sim = km.Simulator(batch_size=256, seed=1); sim.use_same_registers_as(c); sim.do(c)
sim.read_within_shot("x", 0, out=int); sim.read_shot_phase(0)   # Fraction of a turn
```
Gate set: X CX CCX SWAP CSWAP · Z CZ CCZ Z_POW NEG · R HMR · BIT_INVERT BIT_STORE0/1 · PUSH/POP_CONDITION,
per-op `if bN` · REGISTER APPEND_TO_REGISTER INFO_TAG DEBUG_PRINT. `R` on |1⟩ randomises the sign, so dirty
ancillas fail. Z_POW exponent t phases by e^{iπt} at 2⁻¹²⁷ precision — never round through a double.

## Rules (non-negotiable)

1. **Never report a score that was not reproduced by running the official verifier from `data/verifiers/` in this
   repo's Docker container.** `operation_statistics()` numbers and `--max-x` smoke runs are estimates; say so.
2. **Never modify anything under `data/verifiers/` or `data/kickmix/`.** If a verifier seems wrong, note it in
   `tasks/<task>/notes.md` and raise it on the Kaggle forum; do not patch.
3. **Every improved circuit is committed** to `circuits/<task>/` with the verified numbers in the message, e.g.
   `invsqrt: Q=240 T=7100 C=7102.3 cost=1.70e6 (verifier pass, 0 failures)`, generator code in the same commit.
   Uploaded submissions get their public LB score recorded in `submissions/`.
4. Follow the kickmix contract: inputs restored, scratch back to |0⟩, phase within tolerance. `R`/`HMR` tricks are
   fine only if the verifier accepts them.
5. No private code sharing outside the team; anything shared goes to the public forum (Kaggle rule C.5.d).

## Links

- Competition: https://www.kaggle.com/competitions/google-verifiable-quantum-circuit-compilation
- Qualtran repo: https://github.com/quantumlib/Qualtran — kickmix source `kickmix/`, format/instruction/API/dev docs
  `docs/kickmix/*.md` (same paths inside `third_party/Qualtran/`)
- Companion paper (Khattar, Rubin, Gidney, Broughton, Low, Babbush): https://arxiv.org/abs/2610.23832 —
  HTML https://arxiv.org/html/2610.23832 (§III.A invsqrt, §III.B dqi, Eq. 43 R(ε), Eq. 44 dfthcsos)
- Announcement: https://opensource.googleblog.com/2026/10/announcing-verifiable-quantum-circuit-compilation-competition-on-kaggle.html
- Prior kickmix paper (ECC): https://arxiv.org/abs/2603.28846 · dfthcsos starter kit (unused):
  https://www.kaggle.com/datasets/sohier/verifiable-circuit-compilation-starter-kit
