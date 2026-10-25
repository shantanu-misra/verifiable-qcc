"""Native (kickmix-free) helpers for the dfthcsos task: reference data, the verifier's own
metric functions, Givens-tree encode/decode of angle words, the heuristic cost model.

Runs on the host with numpy/scipy (fast) and inside Docker alike. The metric code is the
official verifier's (`data/verifiers/dfthcsos/scoreinfra/*`, imported read-only), so the
numbers here are the numbers the verifier computes from decoded words; only the circuit
simulation (Toffoli count, contract checks) needs kickmix and the official run.

    DATA = $VQCC_DATA or <repo>/data
    ref = load_reference_cached()           # FCIDUMP -> dict (cached as build/dfthcsos_ref.npz)
    base = load_baseline()                  # baseline submission.npz + fixed shifts -> Factorization
    th = decompose(vectors, edges)          # theta/pi per Givens edge, exact round trip
    words = quantize(th, brot); v = decode(words, brot, edges)
    m = evaluate_words(ref, fac, words, brot)   # error_1, error_2, Lambda, E_gap, lambda_eff
    est = estimate_cost(m['lambda_eff'], brot, J, angles_toffoli)  # heuristic model + ANGLES
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.environ.get("VQCC_DATA", os.path.join(REPO, "data"))
VERIFIER_DIR = os.path.join(DATA, "verifiers", "dfthcsos")
SAMPLE_DIR = os.path.join(DATA, "sample_submission", "dfthcsos")
BUILD = os.path.join(REPO, "build")
N = 76
E_UPPER = -22140.27335
KEEP_BITS = 10


def _load_module(name: str, path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The verifier's metric/cost modules, loaded by file so that no kickmix import is triggered.
dm = _load_module("dfthcsos_dfthc_math", os.path.join(VERIFIER_DIR, "scoreinfra", "dfthc_math.py"))
rf = _load_module("dfthcsos_reference", os.path.join(VERIFIER_DIR, "scoreinfra", "reference.py"))
cp = _load_module("dfthcsos_costs_prepare", os.path.join(VERIFIER_DIR, "scoreinfra", "costs_prepare.py"))


# ---------------------------------------------------------------------------------------------
# Reference integrals and the factorization


def load_reference_cached(cache: str | None = None) -> dict:
    """The verifier's `load_reference` of the packaged FCIDUMP, cached as an .npz (tei float64 267 MB)."""
    cache = cache or os.path.join(BUILD, "dfthcsos_ref.npz")
    if os.path.exists(cache):
        with np.load(cache) as z:
            ref = {k: z[k] for k in z.files}
    else:
        ref = rf.load_reference(rf.check_fcidump(os.path.join(VERIFIER_DIR, rf.FCIDUMP_NAME)))
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, **{k: np.asarray(v) for k, v in ref.items()})
    ref["electrons"] = int(ref["electrons"])
    ref["orbitals"] = int(ref["orbitals"])
    ref["ecore"] = float(ref["ecore"])
    return ref


@dataclass
class Factorization:
    """w (R,B,C), W (R,C), S (N,N), beta_1 as the verifier derives them, plus the tree and U."""

    w: np.ndarray
    W: np.ndarray
    S: np.ndarray
    beta_1: float
    edges: np.ndarray
    U: np.ndarray
    H1_I: np.ndarray
    Hsym_lower: np.ndarray

    @property
    def shape(self):
        return self.w.shape

    def npz_arrays(self, U: np.ndarray | None = None, angle_words: np.ndarray | None = None) -> dict:
        """The arrays of a submission.npz: W=w, H2_I = W + sum_b w, the fixed shift entries, tree."""
        out = {
            "U": np.asarray(self.U if U is None else U, dtype=np.float64),
            "W": np.asarray(self.w, dtype=np.float64),
            "H2_I": np.asarray(self.W + self.w.sum(axis=1), dtype=np.float64),
            "H1_I": np.asarray(self.H1_I),
            "Hsym_lower": np.asarray(self.Hsym_lower),
            "tree_edges": np.asarray(self.edges, dtype=np.int64),
        }
        if angle_words is not None:
            out["angle_words"] = np.asarray(angle_words, dtype=np.int64)
        return out


def factorization_from_params(params: dict, edges: np.ndarray) -> Factorization:
    fixed = rf.load_fixed_shifts(os.path.join(VERIFIER_DIR, rf.FIXED_SHIFTS_NAME))
    w = np.asarray(params["W"], dtype=np.float64)
    W = np.asarray(params["H2_I"], dtype=np.float64) - w.sum(axis=1)
    S = np.asarray(dm.symmetric_from_lower(fixed["Hsym_lower"], N), dtype=np.float64)
    beta_1 = float(fixed["H1_I"]) - 0.5 * float(np.trace(S))
    return Factorization(w=w, W=W, S=S, beta_1=beta_1, edges=np.asarray(edges, dtype=np.int64),
                         U=np.asarray(params["U"], dtype=np.float64), H1_I=fixed["H1_I"],
                         Hsym_lower=fixed["Hsym_lower"])


def load_baseline() -> tuple[Factorization, np.ndarray]:
    """The provided baseline factorization and its 12-bit angle_words (3976, 75)."""
    with np.load(os.path.join(SAMPLE_DIR, "submission.npz")) as z:
        params = {k: z[k] for k in ("U", "W", "H2_I", "H1_I", "Hsym_lower")}
        edges = z["tree_edges"]
        words = z["angle_words"].astype(np.int64)
    return factorization_from_params(params, edges), words


def load_submission(path: str) -> tuple[Factorization, np.ndarray | None]:
    with np.load(path) as z:
        params = {k: z[k] for k in ("U", "W", "H2_I", "H1_I", "Hsym_lower")}
        edges = z["tree_edges"]
        words = z["angle_words"].astype(np.int64) if "angle_words" in z.files else None
    return factorization_from_params(params, edges), words


# ---------------------------------------------------------------------------------------------
# Givens tree encode / decode


def decompose(vectors: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """theta/pi (rows, N-1) with reconstruct_u_vectors(theta/pi, edges) == unit rows of `vectors`.

    Reverse of the verifier's forward pass: processing the edges last-to-first, rotate the child
    amplitude into the parent (theta = atan2(x_c, x_p), the parent becomes hypot >= 0), so the
    root ends at +1.
    """
    x = np.array(vectors, dtype=np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    th = np.zeros((x.shape[0], len(edges)))
    for t in range(len(edges) - 1, -1, -1):
        p, c = edges[t]
        th[:, t] = np.arctan2(x[:, c], x[:, p]) / np.pi
        x[:, p] = np.hypot(x[:, p], x[:, c])
        x[:, c] = 0.0
    return th


def quantize(theta_over_pi: np.ndarray, brot: int) -> np.ndarray:
    """Round theta/pi to brot-bit words (theta/pi = word / 2**(brot-1) mod 2)."""
    return np.rint(np.asarray(theta_over_pi) * (1 << (brot - 1))).astype(np.int64) % (1 << brot)


def decode(words: np.ndarray, brot: int, edges: np.ndarray) -> np.ndarray:
    return dm.reconstruct_u_vectors(np.asarray(words) / (1 << (brot - 1)), edges)


def h1_prime(ref: dict, fac: Factorization, u2: np.ndarray) -> np.ndarray:
    """h1_prime built from two-body unit rows u2 (R, B, N), exactly as the verifier does."""
    X = dm.reconstruct_x(u2, fac.w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    h1_thc = ref["oei"] - 0.5 * np.einsum("prrq->pq", ref["tei"])
    return dm.build_h1_prime(h1_thc, fac.S, fac.beta_1, ref["electrons"], fac.W, X)


def one_body_rows(ref: dict, fac: Factorization, u2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvectors (rows) and eigenvalues of h1_prime for two-body rows u2 (R, B, N)."""
    lam, vecs = np.linalg.eigh(h1_prime(ref, fac, u2))
    return vecs.T.copy(), lam


# ---------------------------------------------------------------------------------------------
# Metrics and cost


def evaluate_words(ref: dict, fac: Factorization, words: np.ndarray, brot: int) -> dict:
    """The verifier's metrics of an integer word table (N + R*B, N - 1)."""
    return dm.evaluate_dfthc_angle_words(words, fac.w, fac.W, fac.S, fac.beta_1, ref, brot=brot,
                                         tree_edges=fac.edges, energy_upper=E_UPPER)


def evaluate_vectors(ref: dict, fac: Factorization, u_all: np.ndarray) -> dict:
    return dm.evaluate_dfthc_vectors(u_all, fac.w, fac.W, fac.S, fac.beta_1, ref, energy_upper=E_UPPER)


def passes(m: dict) -> bool:
    return bool(m["error_1"] <= 0.02 and m["error_2"] <= 0.02 and m["gap_in_domain"]
                and np.isfinite(m["lambda_eff"]) and m["Lambda"] > 49.0 and m["E_gap"] > 3.914898)


def fmt(m: dict) -> str:
    return (f"e1 {1e3*m['error_1']:7.3f} mHa  e2 {1e3*m['error_2']:7.3f} mHa  Lambda {m['Lambda']:.4f}  "
            f"E_gap {m['E_gap']:.6f}  lambda_eff {m['lambda_eff']:.6f}  {'PASS' if passes(m) else 'FAIL'}")


def heuristic(brot: int, shape=(65, 60, 60)) -> dict:
    R, B, C = shape
    mdl = cp.SpinFreeData_V1(R=R, B=B, C=C, N=N, chi=KEEP_BITS, beta=brot)
    o, i = mdl.outer_state_prep_qubit_cost(), mdl.inner_state_prep_qubit_cost()
    return {"toffoli": mdl.outer_state_prep() + mdl.inner_state_prep() + mdl.givens_rotations(),
            "persistent": o["persistent"] + i["persistent"], "temporary": max(o["temporary"], i["temporary"])}


def estimate_cost(lambda_eff: float, brot: int, J: int, angles_toffoli: float, angles_qubits: int,
                  shape=(65, 60, 60)) -> dict:
    """score = lambda_eff * T * Q with the verifier's formula; angles_toffoli = forward + reverse."""
    h = heuristic(brot, shape)
    selection = (N + shape[0] * shape[1] - 1).bit_length()
    angles_temp = angles_qubits - selection - J * brot
    T = angles_toffoli + h["toffoli"]
    Q = h["persistent"] + J * brot + max(h["temporary"], angles_temp)
    return {"T": T, "Q": Q, "score": lambda_eff * T * Q, "ratio": 3355599588.3155365 / (lambda_eff * T * Q)}
