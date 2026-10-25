"""Within-group point pruning of the DF-THC factorisation: (65, 60, C) -> (65, B', C).

The lookup cost is ~proportional to A = 76 + R B and the inner PREPARE drops a QROAM level at B <= 31 (its temporary
qubits then fall below the outer PREPARE's 370, like the brot <= 7 cliff), so fewer points per group pay twice:
(65, 31, 45) is worth 2.3x, (65, 24, 45) 2.8x, (65, 16, 45) 3.8x at brot 10 / J 10 if the accuracy survives (cost
table in notes.md §8). Each stage keeps the B' points of largest weight energy sum_c w_bc^2 in every group and
re-optimises points, weights and offsets with the full loss (error_2^2 + a_gap (E_gap - target)^2 + mu Lambda_2)
through `thc_fit.THCProblem`, which handles any (R, B, C).

    python tasks/dfthcsos/prune.py --init baseline --B 40 --iters 500 --out build/prune_B40.npz
    python tasks/dfthcsos/prune.py --init build/prune_B40.npz --B 32 --iters 800 --out build/prune_B32.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dfthc_tools as dt  # noqa: E402
import refit as rf  # noqa: E402
import thc_fit as tf  # noqa: E402

N = dt.N
log = rf.log


def load_state(path: str, base: dt.Factorization):
    """u (R, B, N) unit rows, w (R, B, C), W (R, C) from 'baseline' or a prune/refit state file."""
    if path == "baseline":
        u = base.U / np.linalg.norm(base.U, axis=2, keepdims=True)
        return u, base.w.copy(), base.W.copy()
    with np.load(path) as z:
        if "u" in z.files and z["u"].ndim == 3:
            return z["u"], z["w"], z["W"]
        # a refit.py state (theta, w) with the baseline's W
        theta = z["theta"]
        w = z["w"]
        R, B, C = w.shape
        u = dt.decode(dt.quantize(theta, 20), 20, base.edges) if theta.max() <= 2 else theta
        return u.reshape(R, B, N), w, base.W.copy()


def prune_points(u: np.ndarray, w: np.ndarray, B_new: int):
    R, B, C = w.shape
    energy = np.sum(w ** 2, axis=2)  # (R, B)
    keep = np.argsort(-energy, axis=1)[:, :B_new]
    keep.sort(axis=1)
    u2 = np.take_along_axis(u, keep[:, :, None], axis=1)
    w2 = np.take_along_axis(w, keep[:, :, None], axis=1)
    return u2, w2, keep


def group_contributions(prob, u: np.ndarray, w: np.ndarray) -> np.ndarray:
    """||sum_c X_rc (x) X_rc||_F per group = ||M_r M_r^T||_F with M_r = s(X_rc) rows (R,)."""
    R, B, C = w.shape
    Us = prob.sym.of_outer(u.reshape(-1, N)).reshape(R, B, -1)
    M = np.matmul(w.transpose(0, 2, 1), Us)  # (R, C, NPAIR)
    return np.array([np.linalg.norm(M[r] @ M[r].T) for r in range(R)])


def prune_groups(prob, u: np.ndarray, w: np.ndarray, W: np.ndarray, R_new: int):
    """Keep the R_new groups with the largest contribution to the two-body operator."""
    contrib = group_contributions(prob, u, w)
    keep = np.sort(np.argsort(-contrib)[:R_new])
    log(f"dropping groups {sorted(set(range(len(contrib))) - set(keep.tolist()))} (contributions "
        f"{np.round(np.sort(contrib)[:len(contrib)-R_new], 4).tolist()} of max {contrib.max():.3f})")
    return u[keep], w[keep], W[keep]


def prune_columns(w: np.ndarray, W: np.ndarray, C_new: int):
    R, B, C = w.shape
    energy = np.sum(w ** 2, axis=1)  # (R, C)
    keep = np.argsort(-energy, axis=1)[:, :C_new]
    keep.sort(axis=1)
    w2 = np.take_along_axis(w, keep[:, None, :], axis=2)
    W2 = np.take_along_axis(W, keep, axis=1)
    return w2, W2


def normalise_state(uu: np.ndarray, w: np.ndarray):
    """Unit rows with the norms absorbed into the weights: s(u u^T) = |u|^2 s(n n^T), so w_bc -> w_bc |u_b|^2."""
    R, B, C = w.shape
    norms = np.linalg.norm(uu.reshape(R * B, N), axis=1)
    u = (uu.reshape(R * B, N) / norms[:, None]).reshape(R, B, N)
    w = w * (norms ** 2).reshape(R, B, 1)
    return u, w


def make_problem(ref, base, u, w, W, egap_target: float, a_gap: float) -> tf.THCProblem:
    R, B, C = w.shape
    fac = dt.Factorization(w=w, W=W, S=base.S, beta_1=base.beta_1, edges=base.edges, U=u, H1_I=base.H1_I,
                           Hsym_lower=base.Hsym_lower)
    prob = tf.THCProblem(ref, fac, a_gap=a_gap, egap_target=egap_target)
    prob.set_W(W)
    return prob


def report(prob, u, w, W, label: str):
    r = prob.forward_backward_u(u.reshape(-1, N), w, 0.0, 0.0, False, W=W)
    log(f"{label}: e2 {1e3*r['error_2']:.3f} mHa  E_gap {r['E_gap']:.5f}  negmass {r['negmass']:.3f}  "
        f"Lambda {r['Lambda']:.3f} (L2 {r['lambda2']:.3f})  lam_eff {r['lambda_eff']:.4f}")
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="baseline")
    ap.add_argument("--B", type=int, default=None, help="points per group to keep (default: unchanged)")
    ap.add_argument("--R", type=int, default=None, help="groups to keep (default: unchanged)")
    ap.add_argument("--C", type=int, default=None, help="columns per group to keep (default: unchanged)")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--a-gap", type=float, default=10.0)
    ap.add_argument("--egap-target", type=float, default=3.918)
    ap.add_argument("--w-first", type=int, default=0, help="iterations on (w, W) only before the joint refit")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    ref = dt.load_reference_cached()
    base, _ = dt.load_baseline()
    u, w, W = load_state(args.init, base)
    prob = make_problem(ref, base, u, w, W, args.egap_target, args.a_gap)
    report(prob, u, w, W, f"init {args.init} {w.shape}")
    if args.C and args.C < w.shape[2]:
        w, W = prune_columns(w, W, args.C)
    if args.R and args.R < w.shape[0]:
        u, w, W = prune_groups(prob, u, w, W, args.R)
    if args.B and args.B < w.shape[1]:
        u, w, keep = prune_points(u, w, args.B)
    prob = make_problem(ref, base, u, w, W, args.egap_target, args.a_gap)
    report(prob, u, w, W, f"pruned to {w.shape}")
    uu = u.reshape(-1, N)
    if args.w_first:
        uu, w, W, r = tf.lbfgs_joint_u(prob, uu, w, W, opt_u=False, opt_w=True, opt_W=True, mu=args.mu,
                                       maxiter=args.w_first, tag=f"[{w.shape} w,W]")
    uu, w, W, r = tf.lbfgs_joint_u(prob, uu, w, W, opt_u=True, opt_w=True, opt_W=True, mu=args.mu,
                                   maxiter=args.iters, tag=f"[{w.shape}]")
    u, w = normalise_state(uu, w)
    r = report(prob, u, w, W, f"final {w.shape}")
    np.savez(args.out, u=u, w=w, W=W, theta=dt.decompose(u.reshape(-1, N), base.edges),
             metrics=np.array([r["error_2"], r["E_gap"], r["Lambda"], r["lambda_eff"]]))
    log(f"saved {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
