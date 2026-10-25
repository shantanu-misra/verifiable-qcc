"""Quantise a continuous (R, B, C) factorisation state (u, w, W from prune.py / thc_fit.py) into a submission:
plain rounding -> progressive freezing -> word polish + weight refit -> E_gap re-steer on W -> one-body rows ->
exact verifier metrics -> submission.npz arrays (float32 like the sample).

    python tasks/dfthcsos/quantize_state.py --state build/prune_B32.npz --brot 10 --out build/q_B32.npz
    python tasks/dfthcsos/build_qrom.py build/q_B32_sub/submission.npz build/q_B32_sub -J 10 --banks 1 --check
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
import prune as pr  # noqa: E402

N = dt.N
log = rf.log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True)
    ap.add_argument("--brot", type=int, default=10)
    ap.add_argument("--egap-target", type=float, default=3.918)
    ap.add_argument("--prog-batches", type=int, default=16)
    ap.add_argument("--prog-iters", type=int, default=30)
    ap.add_argument("--polish", type=int, default=2)
    ap.add_argument("--polish-rounds", type=int, default=2)
    ap.add_argument("--refine-rounds", type=int, default=0)
    ap.add_argument("--w-iters", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    ref = dt.load_reference_cached()
    base, _ = dt.load_baseline()
    u, w, W = pr.load_state(args.state, base)
    R, B, C = w.shape
    brot = args.brot
    prob = pr.make_problem(ref, base, u, w, W, args.egap_target, a_gap=10.0)
    pr.report(prob, u, w, W, f"continuous {w.shape}")
    theta = dt.decompose(u.reshape(-1, N), base.edges)
    words2 = rf.quantize_theta(theta, brot)[0]
    m, _ = rf.exact_eval(prob, words2, w, brot, one_body_sweeps=0)
    log(f"plain rounding brot {brot}: {dt.fmt(m)}")
    theta_q, w = rf.progressive(prob, theta, w, brot, batches=args.prog_batches, iters=args.prog_iters)
    words2 = rf.quantize_theta(theta_q, brot)[0]
    m, _ = rf.exact_eval(prob, words2, w, brot, one_body_sweeps=0)
    log(f"after progressive: {dt.fmt(m)}")
    for rnd in range(args.polish_rounds):
        words2 = rf.polish(prob, words2, w, brot, sweeps=args.polish, max_moves=2)
        theta_q = words2 / (1 << (brot - 1))
        theta_q, w, _ = rf.lbfgs(prob, theta_q, w, opt_theta=False, opt_w=True, maxiter=args.w_iters, tag=f"[w {rnd}]")
        m, _ = rf.exact_eval(prob, words2, w, brot, one_body_sweeps=0)
        log(f"after polish round {rnd}: {dt.fmt(m)}")
    if args.refine_rounds:
        theta_q, w = rf.refine(prob, theta_q, w, brot, rounds=args.refine_rounds)
        words2 = rf.quantize_theta(theta_q, brot)[0]
        words2 = rf.polish(prob, words2, w, brot, sweeps=args.polish, max_moves=2)
        theta_q = words2 / (1 << (brot - 1))
        theta_q, w, _ = rf.lbfgs(prob, theta_q, w, opt_theta=False, opt_w=True, maxiter=args.w_iters, tag="[w final]")
    # E_gap re-steer on the offsets with the final words/weights (W only; error_2 is independent of W)
    import thc_fit as tf
    u_q = rf.givens_forward(theta_q, base.edges)[0]
    uu, w, W, r = tf.lbfgs_joint_u(prob, u_q, w, W, opt_u=False, opt_w=False, opt_W=True, maxiter=200, tag="[W final]")
    prob.set_W(W)
    m, words = rf.exact_eval(prob, words2, w, brot, one_body_sweeps=8, verbose=True)
    log(f"final: {dt.fmt(m)}")
    sub = {
        "U": np.asarray(u_q.reshape(R, B, N), dtype=np.float32),
        "W": np.asarray(w, dtype=np.float32),
        "H2_I": np.asarray(W + w.sum(axis=1), dtype=np.float32),
        "H1_I": np.asarray(base.H1_I),
        "Hsym_lower": np.asarray(base.Hsym_lower),
        "tree_edges": np.asarray(base.edges, dtype=np.int64),
        "angle_words": np.asarray(words, dtype=np.int64),
    }
    np.savez(args.out, theta=theta_q, w=w, W=W, brot=np.int64(brot), **{f"sub_{k}": v for k, v in sub.items()},
             **{f"metric_{k}": np.float64(v) for k, v in m.items() if k != "gap_in_domain"})
    sub_dir = args.out[:-4] + "_sub"
    os.makedirs(sub_dir, exist_ok=True)
    np.savez(os.path.join(sub_dir, "submission.npz"), **sub)
    log(f"saved {args.out} and {sub_dir}/submission.npz ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
