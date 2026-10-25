"""Single-group THC-style factorisation of FeMoCo76o113e for the dfthcsos task.

The verifier accepts any (R, B, C). With R = 1 the two-body operator is tensor hypercontraction,
H = U Z U^T with U = [s(u_b u_b^T)]_b (2,926 pairs x B) and Z = w w^T (B x B, rank <= C), and the family
loads only 76 + B vectors instead of 3,976: the ANGLES Toffolis and the inner PREPARE both shrink with B and C
(cost model in notes.md §7). The public frontier (self-reported) is R 1 / B 496 / C 384.

Pieces:
  * `THCProblem(Problem)`: free identity offsets W (C,) with the E_gap formula
        E_gap = E_UPPER - ecore + eta beta_1 + 0.5 sum W^2 + 2 negmass(h1'),
    loss = error_2^2 + a_gap (E_gap - target)^2 + mu Lambda_2 (+ alpha negmass), gradients for theta, w, W.
  * `init_dyads`: starting points from the eigen-structure of the target: the eigen-matrices V_k of T
    (2,926 x 2,926 PSD, rank ~400 at 5 mHa) are decomposed, their eigenvector dyads e e^T ranked by
    sigma_k lambda_i^2 and the top B distinct directions kept.
  * `solve_Z`: for fixed points the best Z is the projection (U^T U)^-1 U^T T U (U^T U)^-1; its PSD rank-C
    truncation gives w.
  * `fit`: alternate solve_Z with L-BFGS on (theta, w, W).
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

N = dt.N
NPAIR = rf.NPAIR
E_GAP_FLOOR = 3.914898
log = rf.log


class THCProblem(rf.Problem):
    """refit.Problem with free offsets W and the E_gap / Lambda terms of the score in the loss."""

    def __init__(self, ref, fac, a_gap: float = 0.0, egap_target: float = 3.918):
        super().__init__(ref, fac)
        self.a_gap, self.egap_target = a_gap, egap_target
        self.c_gap = ref["ecore"] - ref["electrons"] * fac.beta_1  # E_sos = c_gap - 0.5 sum W^2 - 2 negmass

    def set_W(self, W: np.ndarray) -> None:
        self.Wmat = np.asarray(W, dtype=np.float64).reshape(self.R, self.C)
        self.esos_const = self.c_gap - 0.5 * float(np.sum(self.Wmat ** 2))

    def forward_backward_u(self, u, w, alpha=0.0, mu=0.0, need_grad=True, W=None):
        if W is not None:
            self.set_W(W)
        w_shape = w.shape
        w = w.reshape(self.R, self.B, self.C)
        r = super().forward_backward_u(u, w, alpha, mu, need_grad)
        if need_grad:
            r["dw"] = r["dw"].reshape(w_shape)
        return self._extra_terms(r, u, w, w_shape, mu, need_grad, W, theta_q=None)

    def forward_backward(self, theta_q, w, alpha=0.0, mu=0.0, need_grad=True, W=None):
        if W is not None:
            self.set_W(W)
        w_shape = w.shape
        w = w.reshape(self.R, self.B, self.C)
        r = super().forward_backward(theta_q, w, alpha, mu, need_grad)
        if need_grad:
            r["dw"] = r["dw"].reshape(w_shape)
        return self._extra_terms(r, r["u"], w, w_shape, mu, need_grad, W, theta_q=theta_q)

    def extra_dM(self, out):
        if not self.a_gap or not np.isfinite(out["E_gap"]):
            return None
        dgap = out["E_gap"] - self.egap_target
        a_lin = 2.0 * self.a_gap * dgap * 2.0  # dL/dE_gap * dE_gap/dnegmass
        lam, V = out["lam"], out["V"]
        neg = lam < 0
        sP = self.sym.of_matrix(V[:, neg] @ V[:, neg].T)
        return a_lin * np.einsum("rc,s->rcs", self.Wmat, sP).reshape(self.R * self.C, NPAIR)

    def _extra_terms(self, r, u, w, w_shape, mu, need_grad, W, theta_q):
        if self.a_gap:
            dgap = r["E_gap"] - self.egap_target
            r["loss"] += self.a_gap * dgap * dgap
        if not need_grad:
            return r
        # gradients of the E_gap term: through negmass (as alpha_lin) and directly through W
        Wm = self.Wmat
        M = r["M"].reshape(self.R, self.C, NPAIR)
        lam, V = r["lam"], r["V"]
        neg = lam < 0
        sP = self.sym.of_matrix(V[:, neg] @ V[:, neg].T)  # s(P_neg)
        dnm_dW = np.einsum("rcs,s->rc", M, sP)  # d negmass / d W_rc = <P_neg, X_rc>
        dW = np.zeros_like(Wm)
        if self.a_gap:
            dgap = r["E_gap"] - self.egap_target
            coef = 2.0 * self.a_gap * dgap  # dL/dE_gap
            dW += coef * (Wm + 2.0 * dnm_dW)  # the negmass part through M is in extra_dM
        if mu:
            wB = Wm + w.sum(axis=1)
            tot = np.abs(w).sum(axis=1) + np.abs(wB)
            dW += mu * 0.5 * tot * np.sign(wB)
        r["dW"] = dW.reshape(self.Wmat.shape if W is None else np.shape(W))
        return r


# ---------------------------------------------------------------------------------------------


def target_eig(prob, k: int):
    ev, V = np.linalg.eigh(prob.T)
    idx = np.argsort(-ev)[:k]
    return ev[idx], V[:, idx]


def init_dyads(prob, B: int, k_factors: int = 400, dedupe: float = 0.98, seed: int = 0) -> np.ndarray:
    """B unit vectors: dominant eigenvector dyads of the top eigen-matrices of T."""
    sig, V = target_eig(prob, k_factors)
    cands, scores = [], []
    for k in range(k_factors):
        Vk = prob.sym.to_matrix(V[:, k])
        lam, E = np.linalg.eigh(Vk)
        for i in range(N):
            cands.append(E[:, i])
            scores.append(sig[k] * lam[i] ** 2)
    cands, scores = np.array(cands), np.array(scores)
    order = np.argsort(-scores)
    chosen = []
    for i in order:
        c = cands[i]
        if chosen and np.max(np.abs(np.array(chosen) @ c)) > dedupe:
            continue
        chosen.append(c)
        if len(chosen) == B:
            break
    return np.array(chosen)


def solve_Z(prob, u: np.ndarray, C: int):
    """Least-squares Z for fixed points, its PSD rank-C factor w (B, C) and the two errors."""
    Us = prob.sym.of_outer(u)  # (B, NPAIR)
    Q, Rm = np.linalg.qr(Us.T)  # Us.T = Q R, Q (NPAIR, B)
    TQ = prob.T @ Q
    core = Q.T @ TQ  # (B, B)
    err_free = np.sqrt(max(np.sum(prob.T ** 2) - np.sum(core ** 2), 0.0))  # ||T - P T P||^2 = ||T||^2 - ||Q^T T Q||^2
    Rinv = np.linalg.inv(Rm)
    Z = Rinv @ core @ Rinv.T
    Z = 0.5 * (Z + Z.T)
    lam, E = np.linalg.eigh(Z)
    idx = np.argsort(-lam)[:C]
    lam_c = np.clip(lam[idx], 0, None)
    w = E[:, idx] * np.sqrt(lam_c)[None, :]
    H = (Us.T @ w) @ (w.T @ Us)
    err_c = float(np.linalg.norm(H - prob.T))
    return w, err_free, err_c, lam


def make_fac(base: dt.Factorization, w: np.ndarray, W: np.ndarray, u: np.ndarray) -> dt.Factorization:
    B, C = w.shape
    return dt.Factorization(w=w.reshape(1, B, C), W=W.reshape(1, C), S=base.S, beta_1=base.beta_1, edges=base.edges,
                            U=u.reshape(1, B, N), H1_I=base.H1_I, Hsym_lower=base.Hsym_lower)


def lbfgs_thc(prob: THCProblem, theta, w, W, *, opt_theta=True, opt_w=True, opt_W=True, alpha=0.0, mu=0.0,
              maxiter=200, tag=""):
    from scipy.optimize import minimize

    shapes = {"theta": theta.shape, "w": w.shape, "W": W.shape}
    parts = []
    if opt_theta:
        parts.append(("theta", theta.ravel(), 1.0))
    if opt_w:
        parts.append(("w", (w / rf.W_SCALE).ravel(), rf.W_SCALE))
    if opt_W:
        parts.append(("W", W.ravel(), 1.0))
    x0 = np.concatenate([p[1] for p in parts])
    state = {"calls": 0, "t0": time.time()}

    def unpack(x):
        vals = {"theta": theta, "w": w, "W": W}
        off = 0
        for name, arr, scale in parts:
            vals[name] = x[off:off + arr.size].reshape(shapes[name]) * scale
            off += arr.size
        return vals

    def fun(x):
        v = unpack(x)
        r = prob.forward_backward(v["theta"], v["w"], alpha, mu, True, W=v["W"])
        state["calls"] += 1
        g = []
        for name, arr, scale in parts:
            g.append((r["d" + name] * scale).ravel())
        if state["calls"] % 50 == 1:
            log(f"  lbfgs{tag} call {state['calls']:5d}  e2 {1e3*r['error_2']:8.3f} mHa  E_gap {r['E_gap']:.5f}  "
                f"Lambda {r['Lambda']:.3f}  lam_eff {r['lambda_eff']:.4f}  ({time.time()-state['t0']:.0f}s)")
        return r["loss"], np.concatenate(g)

    res = minimize(fun, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "maxcor": 30, "ftol": 1e-15, "gtol": 1e-13})
    v = unpack(res.x)
    r = prob.forward_backward(v["theta"], v["w"], alpha, mu, False, W=v["W"])
    log(f"  lbfgs{tag} done ({res.nit} it, {state['calls']} calls): e2 {1e3*r['error_2']:.3f} mHa  E_gap {r['E_gap']:.5f}  "
        f"Lambda {r['Lambda']:.3f}  lam_eff {r['lambda_eff']:.4f}")
    return v["theta"], v["w"], v["W"], r


# ---------------------------------------------------------------------------------------------
# Variable projection on the points: f(u) = min_Z ||U Z U^T - T||^2 with Z eliminated exactly.
# At the optimal Z the partial derivative in Z vanishes, so df/dU is the ordinary gradient with Z = Z*(U)
# (Golub-Pereyra); Z* = A T A^T with A = (U^T U)^-1 U^T is PSD because T is, so no truncation is involved.


def varpro_value_grad(prob, theta: np.ndarray, ridge: float = 0.0):
    u, cache = rf.givens_forward(theta, prob.edges)
    Us = prob.sym.of_outer(u)  # (B, NPAIR)
    Q, Rm = np.linalg.qr(Us.T)  # (NPAIR, B), (B, B)
    TQ = prob.T @ Q
    core = Q.T @ TQ
    f = float(np.sum(prob.T * prob.T) - np.sum(core * core))  # ||T - P T P||^2, P = Q Q^T
    # Z* = R^-1 core R^-T ; residual Res = Q core Q^T - T ; dL/dUs = 4 Z* Us Res
    Rinv = np.linalg.inv(Rm)
    Z = Rinv @ core @ Rinv.T
    ZUs = Z @ Us  # (B, NPAIR)
    # Res @ ZUs^T without forming Res: (Q core Q^T - T) ZUs^T = Q core (Q^T ZUs^T) - T ZUs^T
    QtZ = Q.T @ ZUs.T  # (B, B)
    dUs = 4.0 * (Q @ (core @ QtZ) - prob.T @ ZUs.T).T  # (B, NPAIR)
    du = prob.sym.grad_outer(dUs, u)
    dtheta = rf.givens_backward(du, cache, prob.edges)
    return f, dtheta, Z


def varpro_points(prob, theta: np.ndarray, maxiter: int = 500, tag: str = "", log=log):
    from scipy.optimize import minimize

    shape = theta.shape
    state = {"calls": 0, "t0": time.time()}

    def fun(x):
        f, g, _ = varpro_value_grad(prob, x.reshape(shape))
        state["calls"] += 1
        if state["calls"] % 50 == 1:
            log(f"  varpro{tag} call {state['calls']:5d}  projection error {1e3*np.sqrt(max(f, 0)):8.3f} mHa  ({time.time()-state['t0']:.0f}s)")
        return f, g.ravel()

    res = minimize(fun, theta.ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxcor": 30, "ftol": 1e-15, "gtol": 1e-13})
    theta = res.x.reshape(shape)
    f, _, Z = varpro_value_grad(prob, theta)
    log(f"  varpro{tag} done ({res.nit} it, {state['calls']} calls, {res.message}): projection error {1e3*np.sqrt(max(f, 0)):.3f} mHa")
    return theta, Z, float(np.sqrt(max(f, 0)))


def init_isdf(prob, B: int, k_factors: int = 450, seed: int = 0) -> np.ndarray:
    """ISDF-style points without a grid: candidate 'grid points' are the eigenvectors of the dominant
    eigen-matrices of T; the B of them whose dyads best span the dominant pair space are chosen by a
    column-pivoted QR of G = diag(sqrt(sigma)) V^T D, D = [s(e e^T)] over the candidates."""
    from scipy.linalg import qr

    sig, V = target_eig(prob, k_factors)
    cands = []
    for k in range(k_factors):
        lam, E = np.linalg.eigh(prob.sym.to_matrix(V[:, k]))
        cands.append(E.T)  # rows: eigenvectors
    cands = np.concatenate(cands, axis=0)  # (k_factors * N, N)
    D = prob.sym.of_outer(cands)  # (ncand, NPAIR)
    G = (np.sqrt(np.clip(sig, 0, None))[:, None] * V.T) @ D.T  # (k_factors, ncand)
    _, _, piv = qr(G, mode="economic", pivoting=True)
    chosen = []
    for i in piv:
        c = cands[i]
        if chosen and np.max(np.abs(np.array(chosen) @ c)) > 0.999:
            continue
        chosen.append(c)
        if len(chosen) == B:
            break
    return np.array(chosen)


def projection_error(prob, u: np.ndarray) -> float:
    Us = prob.sym.of_outer(u)
    Q, _ = np.linalg.qr(Us.T)
    core = Q.T @ (prob.T @ Q)
    return float(np.sqrt(max(np.sum(prob.T ** 2) - np.sum(core ** 2), 0.0)))


# ---------------------------------------------------------------------------------------------
# From points to a submission: weights, offsets, quantised words


def weights_from_Z(Z: np.ndarray, C: int) -> np.ndarray:
    lam, E = np.linalg.eigh(Z)
    idx = np.argsort(-lam)[:C]
    return E[:, idx] * np.sqrt(np.clip(lam[idx], 0, None))[None, :]


def solve_offsets(prob: THCProblem, theta: np.ndarray, w: np.ndarray, W0: np.ndarray, *, egap_target: float = 3.918,
                  a_gap: float = 10.0, mu: float = 0.0, iters: int = 300, tag: str = "[W]"):
    """Offsets W for fixed points/weights: minimise a_gap (E_gap - target)^2 + mu Lambda_2 (the error does not
    depend on W).  Start: least squares W with sum_c W_c X^c ~ h1_base (negmass ~ 0), then L-BFGS."""
    prob.a_gap, prob.egap_target = a_gap, egap_target
    th, ww, W, r = lbfgs_thc(prob, theta, w, W0, opt_theta=False, opt_w=False, opt_W=True, mu=mu, maxiter=iters, tag=tag)
    return W, r


def ls_offsets(prob: THCProblem, u: np.ndarray, w: np.ndarray, frac: float = 1.0) -> np.ndarray:
    """Least-squares W with sum_c W_c X^c ~ frac * h1_base (in the pair basis)."""
    Us = prob.sym.of_outer(u)
    M = (w.T @ Us)  # (C, NPAIR) = s(X^c)
    target = prob.sym.of_matrix(prob.h1_base) * frac
    W, *_ = np.linalg.lstsq(M.T, target, rcond=None)
    return W


def submission_arrays(base: dt.Factorization, u: np.ndarray, w: np.ndarray, W: np.ndarray, words: np.ndarray) -> dict:
    B, C = w.shape
    return {
        "U": np.asarray(u.reshape(1, B, N), dtype=np.float32),
        "W": np.asarray(w.reshape(1, B, C), dtype=np.float32),
        "H2_I": np.asarray((W + w.sum(axis=0)).reshape(1, C), dtype=np.float32),
        "H1_I": np.asarray(base.H1_I),
        "Hsym_lower": np.asarray(base.Hsym_lower),
        "tree_edges": np.asarray(base.edges, dtype=np.int64),
        "angle_words": np.asarray(words, dtype=np.int64),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", required=True, help="state .npz with theta (B, 75) (and optionally Z)")
    ap.add_argument("--C", type=int, default=None, help="weight rank (default B)")
    ap.add_argument("--brot", type=int, default=10)
    ap.add_argument("--egap-target", type=float, default=3.918)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--joint-iters", type=int, default=0, help="L-BFGS on (theta, w, W) with the full loss before quantising")
    ap.add_argument("--prog-batches", type=int, default=16)
    ap.add_argument("--prog-iters", type=int, default=30)
    ap.add_argument("--polish", type=int, default=2)
    ap.add_argument("--polish-rounds", type=int, default=1)
    ap.add_argument("--refine-rounds", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    ref = dt.load_reference_cached()
    base, _ = dt.load_baseline()
    prob0 = rf.Problem(ref, base)
    z = np.load(args.points)
    theta = z["theta"]
    B = theta.shape[0]
    C = args.C or B
    u = rf.givens_forward(theta, base.edges)[0]
    if "Z" in z.files and z["Z"].shape == (B, B):
        Z = z["Z"]
    else:
        _, _, Z = varpro_value_grad(prob0, theta)
    w = weights_from_Z(Z, C)
    log(f"points B {B}, C {C}: projection error {1e3*projection_error(prob0, u):.3f} mHa")
    W = ls_offsets(prob0, u, w)
    fac = make_fac(base, w, W, u)
    prob = THCProblem(ref, fac, a_gap=10.0, egap_target=args.egap_target)
    prob.set_W(W)
    r = prob.forward_backward(theta, w, need_grad=False)
    log(f"rank-{C} weights, LS offsets: e2 {1e3*r['error_2']:.3f} mHa  E_gap {r['E_gap']:.4f}  negmass {r['negmass']:.3f}  "
        f"Lambda {r['Lambda']:.2f}  lam_eff {r['lambda_eff']:.3f}")
    W, r = solve_offsets(prob, theta, w, W, egap_target=args.egap_target, a_gap=10.0, mu=args.mu, iters=300)
    prob.set_W(W)
    if args.joint_iters:
        theta, w, W, r = lbfgs_thc(prob, theta, w, W, opt_theta=True, opt_w=True, opt_W=True, mu=args.mu,
                                   maxiter=args.joint_iters, tag="[joint]")
        prob.set_W(W)
    # quantisation (refit pipeline on this problem; W stays fixed inside it)
    prob.a_gap = 0.0
    words2 = rf.quantize_theta(theta, args.brot)[0]
    m, _ = rf.exact_eval(prob, words2, w.reshape(1, B, C), args.brot, one_body_sweeps=0)
    log(f"plain rounding brot {args.brot}: {dt.fmt(m)}")
    w3 = w.reshape(1, B, C)
    theta_q, w3 = rf.progressive(prob, theta, w3, args.brot, batches=args.prog_batches, iters=args.prog_iters)
    words2 = rf.quantize_theta(theta_q, args.brot)[0]
    for rnd in range(args.polish_rounds):
        words2 = rf.polish(prob, words2, w3, args.brot, sweeps=args.polish, max_moves=2)
        theta_q = words2 / (1 << (args.brot - 1))
        theta_q, w3, _ = rf.lbfgs(prob, theta_q, w3, opt_theta=False, opt_w=True, maxiter=60, tag=f"[w {rnd}]")
    if args.refine_rounds:
        theta_q, w3 = rf.refine(prob, theta_q, w3, args.brot, rounds=args.refine_rounds)
        words2 = rf.quantize_theta(theta_q, args.brot)[0]
        words2 = rf.polish(prob, words2, w3, args.brot, sweeps=args.polish, max_moves=2)
        theta_q = words2 / (1 << (args.brot - 1))
        theta_q, w3, _ = rf.lbfgs(prob, theta_q, w3, opt_theta=False, opt_w=True, maxiter=60, tag="[w final]")
    # re-steer E_gap with the final weights (W only), then exact metrics
    w = w3.reshape(B, C)
    W, r = solve_offsets(prob, theta_q, w, W, egap_target=args.egap_target, a_gap=10.0, mu=args.mu, iters=200, tag="[W final]")
    prob.set_W(W)
    m, words = rf.exact_eval(prob, words2, w3, args.brot, one_body_sweeps=8, verbose=True)
    log(f"final: {dt.fmt(m)}")
    u_q = rf.givens_forward(theta_q, base.edges)[0]
    arrays = submission_arrays(base, u_q, w, W, words)
    np.savez(args.out, theta=theta_q, w=w, W=W, brot=np.int64(args.brot), **{f"sub_{k}": v for k, v in arrays.items()})
    log(f"saved {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------
# Variable projection directly on the point coordinates u (B, N): the scale of u_b is absorbed by Z,
# so u is unconstrained; much better conditioned than the Givens angles.


def varpro_u_value_grad(prob, u: np.ndarray):
    Us = prob.sym.of_outer(u)
    Q, Rm = np.linalg.qr(Us.T)
    core = Q.T @ (prob.T @ Q)
    f = float(np.sum(prob.T * prob.T) - np.sum(core * core))
    Rinv = np.linalg.inv(Rm)
    Z = Rinv @ core @ Rinv.T
    ZUs = Z @ Us
    QtZ = Q.T @ ZUs.T
    dUs = 4.0 * (Q @ (core @ QtZ) - prob.T @ ZUs.T).T
    return f, prob.sym.grad_outer(dUs, u), Z


def varpro_u(prob, u: np.ndarray, maxiter: int = 500, tag: str = "", log=log):
    from scipy.optimize import minimize

    shape = u.shape
    state = {"calls": 0, "t0": time.time()}

    def fun(x):
        f, g, _ = varpro_u_value_grad(prob, x.reshape(shape))
        state["calls"] += 1
        if state["calls"] % 50 == 1:
            log(f"  varpro_u{tag} call {state['calls']:5d}  projection error {1e3*np.sqrt(max(f, 0)):8.3f} mHa  ({time.time()-state['t0']:.0f}s)")
        return f, g.ravel()

    res = minimize(fun, u.ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxcor": 30, "ftol": 1e-15, "gtol": 1e-13})
    u = res.x.reshape(shape)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    f, _, Z = varpro_u_value_grad(prob, u)
    log(f"  varpro_u{tag} done ({res.nit} it, {state['calls']} calls, {res.message}): projection error {1e3*np.sqrt(max(f, 0)):.3f} mHa")
    return u, Z, float(np.sqrt(max(f, 0)))


def grow_points(prob, u: np.ndarray, B_target: int, step: int = 8, iters: int = 60, log=log):
    """Greedy growth: add `step` points at a time (the top eigenvector dyads of the residual's dominant
    eigen-matrices), re-optimise all points with varpro_u for `iters` iterations."""
    u = u.copy()
    while u.shape[0] < B_target:
        Us = prob.sym.of_outer(u)
        Q, _ = np.linalg.qr(Us.T)
        core = Q.T @ (prob.T @ Q)
        Res = prob.T - Q @ core @ Q.T  # T - P T P
        ev, V = np.linalg.eigh(Res)
        idx = np.argsort(-np.abs(ev))[:step]
        new = []
        for k in idx:
            lam, E = np.linalg.eigh(prob.sym.to_matrix(V[:, k]))
            new.append(E[:, np.argmax(np.abs(lam))])
        u = np.concatenate([u, np.array(new)], axis=0)
        u, Z, err = varpro_u(prob, u, maxiter=iters, tag=f"[grow B{u.shape[0]}]")
        log(f"grow: B {u.shape[0]}  projection error {1e3*err:.3f} mHa")
    return u


def lbfgs_joint_u(prob: THCProblem, u, w, W, *, opt_u=True, opt_w=True, opt_W=True, mu=0.0, maxiter=300, tag="",
                  log=log):
    """L-BFGS on point coordinates u (unnormalised), weights w (B, C) and offsets W (C,) with the full loss."""
    from scipy.optimize import minimize

    parts = []
    if opt_u:
        parts.append(("u", u, 1.0))
    if opt_w:
        parts.append(("w", w, rf.W_SCALE))
    if opt_W:
        parts.append(("W", W, 1.0))
    x0 = np.concatenate([(arr / sc).ravel() for _, arr, sc in parts])
    state = {"calls": 0, "t0": time.time()}

    def unpack(x):
        vals = {"u": u, "w": w, "W": W}
        off = 0
        for name, arr, sc in parts:
            vals[name] = x[off:off + arr.size].reshape(arr.shape) * sc
            off += arr.size
        return vals

    def fun(x):
        v = unpack(x)
        r = prob.forward_backward_u(v["u"], v["w"], 0.0, mu, True, W=v["W"])
        state["calls"] += 1
        g = np.concatenate([(r["d" + name] * sc).ravel() for name, _, sc in parts])
        if state["calls"] % 50 == 1:
            log(f"  joint{tag} call {state['calls']:5d}  e2 {1e3*r['error_2']:8.3f} mHa  E_gap {r['E_gap']:.4f}  "
                f"Lambda {r['Lambda']:.2f} (L2 {r['lambda2']:.2f})  lam_eff {r['lambda_eff']:.3f}  loss {r['loss']:.4e}  ({time.time()-state['t0']:.0f}s)")
        return r["loss"], g

    res = minimize(fun, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "maxcor": 30, "ftol": 1e-15, "gtol": 1e-13})
    v = unpack(res.x)
    r = prob.forward_backward_u(v["u"], v["w"], 0.0, mu, False, W=v["W"])
    log(f"  joint{tag} done ({res.nit} it, {state['calls']} calls, {res.message}): e2 {1e3*r['error_2']:.3f} mHa  "
        f"E_gap {r['E_gap']:.4f}  Lambda {r['Lambda']:.2f} (L2 {r['lambda2']:.2f})  lam_eff {r['lambda_eff']:.3f}")
    return v["u"], v["w"], v["W"], r


# ---------------------------------------------------------------------------------------------
# SOS-constrained THC: w = v*v (every column single-signed, X^c PSD) and W_c = -sum_b w_bc - t_c^2
# (W_c <= -tr X^c), so Lambda_2 = 1/4 sum W_c^2 exactly; the loss keeps only error_2^2 + a_gap (E_gap - target)^2.


def lbfgs_sos(prob: THCProblem, u, v, t, *, maxiter=300, tag="", log=log, opt_u=True):
    from scipy.optimize import minimize

    shapes = (u.shape, v.shape, t.shape)
    n_u = u.size if opt_u else 0
    x0 = np.concatenate([u.ravel() if opt_u else np.empty(0), v.ravel(), t.ravel()])
    state = {"calls": 0, "t0": time.time()}

    def unpack(x):
        uu = x[:n_u].reshape(shapes[0]) if opt_u else u
        vv = x[n_u:n_u + v.size].reshape(shapes[1])
        tt = x[n_u + v.size:].reshape(shapes[2])
        return uu, vv, tt

    def fun(x):
        uu, vv, tt = unpack(x)
        ww = vv * vv
        WW = -ww.sum(axis=0) - tt * tt
        r = prob.forward_backward_u(uu, ww, 0.0, 0.0, True, W=WW)
        state["calls"] += 1
        dW = r["dW"]
        dv = 2.0 * vv * (r["dw"] - dW[None, :])  # dW/dw_bc = -1
        dt_ = -2.0 * tt * dW
        g = np.concatenate([r["du"].ravel() if opt_u else np.empty(0), dv.ravel(), dt_.ravel()])
        if state["calls"] % 50 == 1:
            log(f"  sos{tag} call {state['calls']:5d}  e2 {1e3*r['error_2']:8.3f} mHa  E_gap {r['E_gap']:.4f}  "
                f"Lambda {r['Lambda']:.2f} (L2 {r['lambda2']:.2f})  lam_eff {r['lambda_eff']:.3f}  ({time.time()-state['t0']:.0f}s)")
        return r["loss"], g

    res = minimize(fun, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "maxcor": 30, "ftol": 1e-15, "gtol": 1e-13})
    uu, vv, tt = unpack(res.x)
    ww = vv * vv
    WW = -ww.sum(axis=0) - tt * tt
    r = prob.forward_backward_u(uu, ww, 0.0, 0.0, False, W=WW)
    log(f"  sos{tag} done ({res.nit} it, {state['calls']} calls, {res.message}): e2 {1e3*r['error_2']:.3f} mHa  "
        f"E_gap {r['E_gap']:.4f}  Lambda {r['Lambda']:.2f}  lam_eff {r['lambda_eff']:.3f}")
    return uu, vv, tt, r


# ---------------------------------------------------------------------------------------------
# Levenberg-Marquardt / Gauss-Newton on the variable-projection residual with the Kaufman Jacobian
# (Z held at its optimum during each step), normal equations solved by CG with matrix-free products.
#   r(u) = T - U^T Z U,  J d = -(dU^T Z U + U^T Z dU),  dU_b = s(u_b d_b^T + d_b u_b^T),
#   J^T S = -2 grad_outer(Z U S, u)  (S symmetric).


def _dU(sym, u, d):
    """Directional derivative of s(u u^T) rows along d: s(u d^T + d u^T)."""
    return (u[:, sym.P] * d[:, sym.Q] + d[:, sym.P] * u[:, sym.Q]) * sym.scale


def gn_varpro(prob, u: np.ndarray, iters: int = 30, cg_iters: int = 20, lam: float = 1e-3, tag: str = "", log=log):
    sym = prob.sym
    T = prob.T
    f, g, Z = varpro_u_value_grad(prob, u)
    log(f"  gn{tag} start: projection error {1e3*np.sqrt(f):.3f} mHa")
    t0 = time.time()
    for it in range(iters):
        Us = sym.of_outer(u)
        ZU = Z @ Us  # (B, NPAIR)
        H = Us.T @ ZU
        r = T - H
        gradf = 4.0 * sym.grad_outer(ZU @ (H - T), u)  # = 2 J^T r
        rhs = -0.5 * gradf  # -J^T r

        def Jd(d):
            dU = _dU(sym, u, d)
            A = dU.T @ ZU
            return -(A + A.T)

        def JtS(S):
            return -2.0 * sym.grad_outer(ZU @ S, u)

        def JtJ(d):
            return JtS(Jd(d)) + lam * d

        # CG
        x = np.zeros_like(u)
        res = rhs - JtJ(x)
        p = res.copy()
        rs = np.sum(res * res)
        for k in range(cg_iters):
            Ap = JtJ(p)
            alpha = rs / max(np.sum(p * Ap), 1e-300)
            x += alpha * p
            res -= alpha * Ap
            rs_new = np.sum(res * res)
            if np.sqrt(rs_new) < 1e-10 * np.sqrt(np.sum(rhs * rhs)):
                break
            p = res + (rs_new / rs) * p
            rs = rs_new
        # trial step with VarPro re-solve
        u_try = u + x
        f_try, g_try, Z_try = varpro_u_value_grad(prob, u_try)
        if f_try < f:
            u, f, g, Z = u_try, f_try, g_try, Z_try
            lam = max(lam / 3.0, 1e-12)
            accepted = True
        else:
            lam *= 10.0
            accepted = False
        if it % 1 == 0:
            log(f"  gn{tag} it {it:3d}  projection error {1e3*np.sqrt(f):8.3f} mHa  lam {lam:.1e}  {'ok' if accepted else 'rejected'}  "
                f"cg {k+1}  ({time.time()-t0:.0f}s)")
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    f, _, Z = varpro_u_value_grad(prob, u)
    return u, Z, float(np.sqrt(max(f, 0)))


def symmetric_nmf(Z: np.ndarray, C: int, iters: int = 2000, seed: int = 0, log=log, tag: str = "") -> np.ndarray:
    """Nonnegative V (B, C) with V V^T ~ Z (entries of Z clipped at 0), multiplicative updates."""
    rng = np.random.default_rng(seed)
    Zp = np.clip(Z, 0, None)
    B = Z.shape[0]
    # init from the top-C eigenpairs (absolute values) scaled to the right norm
    lam, E = np.linalg.eigh(Zp)
    idx = np.argsort(-lam)[:C]
    V = np.abs(E[:, idx]) * np.sqrt(np.clip(lam[idx], 1e-12, None))[None, :]
    V += 1e-3 * V.mean() * rng.random(V.shape)
    beta = 0.5
    for it in range(iters):
        ZV = Zp @ V
        VVtV = V @ (V.T @ V)
        V = V * (1 - beta + beta * ZV / np.maximum(VVtV, 1e-300))
        if it % 500 == 0 or it == iters - 1:
            log(f"  snmf{tag} it {it:5d}: ||Z - VV^T|| / ||Z|| = {np.linalg.norm(Zp - V @ V.T) / np.linalg.norm(Zp):.4e}")
    return V


# ---------------------------------------------------------------------------------------------
# Levenberg-Marquardt on the full (u, w, W) problem for any (R, B, C): residuals R_H = T - H (Frobenius) and
# r_gap = sqrt(a_gap) (E_gap - target); matrix-free J / J^T products, CG on the damped normal equations.


def gn_joint(prob: THCProblem, u: np.ndarray, w: np.ndarray, W: np.ndarray, iters: int = 20, cg_iters: int = 20,
             lam: float = 1e-2, tag: str = "", log=log, opt_W: bool = True):
    sym = prob.sym
    R, B, C = w.shape
    sq = np.sqrt(prob.a_gap) if prob.a_gap else 0.0
    f_prev = None
    t0 = time.time()
    for it in range(iters):
        r = prob.forward_backward_u(u, w, 0.0, 0.0, True, W=W)
        f = r["loss"]
        if f_prev is None:
            log(f"  gnj{tag} start: e2 {1e3*r['error_2']:.3f} mHa  E_gap {r['E_gap']:.4f}  loss {f:.4e}")
        Us = r["Us"].reshape(R, B, -1)
        M = r["M"].reshape(R, C, -1)
        lam_h, V = r["lam"], r["V"]
        neg = lam_h < 0
        sP = sym.of_matrix(V[:, neg] @ V[:, neg].T)  # (NPAIR,)
        sPX = M @ sP  # (R, C) = <P_neg, X_rc>
        Wm = prob.Wmat

        def Jd(du, dw, dW):
            dUs = _dU(sym, u, du).reshape(R, B, -1)
            dM = np.matmul(dw.transpose(0, 2, 1), Us) + np.matmul(w.transpose(0, 2, 1), dUs)  # (R, C, NPAIR)
            dM2 = dM.reshape(R * C, -1)
            A = dM2.T @ r["M"]
            JH = -(A + A.T)
            dgap = 0.0
            if sq:
                dgap = float(np.sum(Wm * dW)) + 2.0 * (float(np.sum(dW * sPX)) + float(np.sum(Wm[:, :, None] * dM * sP)))
            return JH, sq * dgap

        def JtS(S, rho):
            SM = -2.0 * (r["M"] @ S).reshape(R, C, -1)  # adjoint of J_H on dM
            if sq:
                SM += (rho * sq * 2.0) * Wm[:, :, None] * sP[None, None, :]
            dw_adj = np.matmul(Us, SM.transpose(0, 2, 1))  # (R, B, C)
            dUs_adj = np.matmul(w, SM).reshape(R * B, -1)
            du_adj = sym.grad_outer(dUs_adj, u)
            dW_adj = (rho * sq) * (Wm + 2.0 * sPX) if sq else np.zeros_like(Wm)
            return du_adj, dw_adj, dW_adj

        def JtJ(du, dw, dW):
            S, rho = Jd(du, dw, dW)
            a, b, c = JtS(S, rho)
            return a + lam * du, b + lam * dw * (rf.W_SCALE ** -2) * 0 + lam * dw, (c + lam * dW) if opt_W else np.zeros_like(dW)

        rhs = (-0.5 * r["du"], -0.5 * r["dw"], (-0.5 * r["dW"]) if opt_W else np.zeros_like(W))
        x = tuple(np.zeros_like(v) for v in rhs)
        res = tuple(b - a for a, b in zip(JtJ(*x), rhs))
        p = tuple(v.copy() for v in res)
        rs = sum(np.sum(v * v) for v in res)
        rhs_norm = np.sqrt(sum(np.sum(v * v) for v in rhs))
        for k in range(cg_iters):
            Ap = JtJ(*p)
            pAp = sum(np.sum(a * b) for a, b in zip(p, Ap))
            alpha = rs / max(pAp, 1e-300)
            x = tuple(a + alpha * b for a, b in zip(x, p))
            res = tuple(a - alpha * b for a, b in zip(res, Ap))
            rs_new = sum(np.sum(v * v) for v in res)
            if np.sqrt(rs_new) < 1e-8 * rhs_norm:
                break
            p = tuple(a + (rs_new / rs) * b for a, b in zip(res, p))
            rs = rs_new
        u_t, w_t, W_t = u + x[0], w + x[1], (W + x[2]) if opt_W else W
        r_t = prob.forward_backward_u(u_t, w_t, 0.0, 0.0, False, W=W_t)
        if r_t["loss"] < f:
            u, w, W = u_t, w_t, W_t
            lam = max(lam / 3.0, 1e-10)
            ok = True
        else:
            lam *= 10.0
            ok = False
        f_prev = f
        log(f"  gnj{tag} it {it:3d}  e2 {1e3*(r_t if ok else r)['error_2']:8.3f} mHa  E_gap {(r_t if ok else r)['E_gap']:.4f}  "
            f"lam {lam:.1e} {'ok' if ok else 'rejected'} cg {k+1}  ({time.time()-t0:.0f}s)")
    prob.set_W(W)
    return u, w, W
