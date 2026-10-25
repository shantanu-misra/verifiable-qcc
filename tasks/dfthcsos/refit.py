"""Quantisation-aware refit of the dfthcsos factorisation for a target angle width brot.

The verifier decodes brot-bit angle words on the Givens tree into unit vectors u_b^rho, builds
X^(rho c) = sum_b w_bc u_b u_b^T and scores error_2 = ||sum_rc X (x) X - (tei + shift)||_F against
a 20 mHa limit.  The provided baseline factorisation has an intrinsic error_2 of 16.1 mHa, so plain
rounding of its angles fails below brot 12 (25 mHa at 11, 40 at 10, 303 at 7).  This module refits
the two-body words and weights *at the quantised grid* so that the decoded words pass:

  1. `qat`     straight-through-estimator gradient descent (Adam) on latent angles theta/pi and the
               weights w: the forward pass uses the rounded words, the gradient is applied to the
               latent angles.  Loss = error_2^2 + alpha * negmass(h1') + mu * Lambda_2, where
               negmass = -sum of the negative eigenvalues of h1' (it sets E_gap and Lambda_1, hence
               lambda_eff) and Lambda_2 is the two-body one-norm.
  2. `polish`  exact coordinate descent on the integer words: for every two-body vector all
               150 single-word moves (+-1) are scored with the exact change of error_2^2 (rank-2
               update of the residual) and the best improving one is applied.
  3. `one_body_words`  the 76 one-body rows: eigenvectors of h1' built from the quantised two-body
               X, rounded and then polished by coordinate descent on error_1 = ||sum lambda_j v_j v_j^T - h1'||_F.

All metrics are evaluated with the verifier's own functions (dfthc_tools), with the weights rounded
to float32 as they will be stored.  Everything works on the symmetric-pair matricisation
(2926 = 76*77/2 pairs, off-diagonal pairs scaled by sqrt 2), which is exact for Frobenius norms.

    python tasks/dfthcsos/refit.py --brot 10 --steps 400 --out build/refit_b10.npz
    python tasks/dfthcsos/refit.py --brot 7 --init build/refit_b8.npz --steps 1000 --polish 3 --out build/refit_b7.npz
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

N = dt.N
NPAIR = N * (N + 1) // 2
PI = np.pi


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------------------------
# Symmetric-pair basis


class Sym:
    """s(X)_I = X_pq * (sqrt2 if p < q else 1) for pairs I = (p <= q): <X, Y>_F = s(X).s(Y)."""

    def __init__(self):
        self.P, self.Q = np.triu_indices(N)
        self.scale = np.where(self.P == self.Q, 1.0, np.sqrt(2.0))
        self.idx = np.zeros((N, N), dtype=np.int64)
        self.idx[self.P, self.Q] = np.arange(NPAIR)
        self.idx[self.Q, self.P] = np.arange(NPAIR)

    def of_outer(self, u: np.ndarray) -> np.ndarray:
        """s(u u^T) for rows u (m, N) -> (m, NPAIR)."""
        return u[:, self.P] * u[:, self.Q] * self.scale

    def of_matrix(self, X: np.ndarray) -> np.ndarray:
        return X[..., self.P, self.Q] * self.scale

    def grad_outer(self, g: np.ndarray, u: np.ndarray) -> np.ndarray:
        """d <g, s(u u^T)> / du for rows: Gamma_pq = g_I * scale / 2 (off-diag) or g_I (diag); 2 Gamma u."""
        coef = g * np.where(self.P == self.Q, 1.0, self.scale / 2.0)
        G = coef[..., self.idx]  # (m, N, N) symmetric gather
        return 2.0 * np.matmul(G, u[..., :, None])[..., 0]

    def to_matrix(self, z: np.ndarray) -> np.ndarray:
        """Inverse of of_matrix for one vector z (NPAIR,) -> symmetric (N, N)."""
        X = np.zeros((N, N))
        X[self.P, self.Q] = z / self.scale
        X[self.Q, self.P] = z / self.scale
        return X


# ---------------------------------------------------------------------------------------------
# Givens chain, with the adjoint


def givens_forward(theta_over_pi: np.ndarray, edges: np.ndarray):
    """Decode rows of theta/pi into unit vectors, keeping the pre-rotation pairs for the adjoint."""
    m = theta_over_pi.shape[0]
    v = np.zeros((m, N))
    v[:, 0] = 1.0
    cs = np.cos(PI * theta_over_pi)
    sn = np.sin(PI * theta_over_pi)
    saved = np.empty((len(edges), 2, m))
    for t, (p, c) in enumerate(edges):
        vp, vc = v[:, p].copy(), v[:, c].copy()
        saved[t, 0], saved[t, 1] = vp, vc
        v[:, p] = cs[:, t] * vp - sn[:, t] * vc
        v[:, c] = sn[:, t] * vp + cs[:, t] * vc
    return v, (cs, sn, saved)


def givens_backward(g: np.ndarray, cache, edges: np.ndarray) -> np.ndarray:
    """dL/d(theta/pi) given dL/dv (m, N)."""
    cs, sn, saved = cache
    g = g.copy()
    dth = np.empty((g.shape[0], len(edges)))
    for t in range(len(edges) - 1, -1, -1):
        p, c = edges[t]
        vp, vc = saved[t]
        gp, gc = g[:, p].copy(), g[:, c].copy()
        dth[:, t] = gp * (-sn[:, t] * vp - cs[:, t] * vc) + gc * (cs[:, t] * vp - sn[:, t] * vc)
        g[:, p] = cs[:, t] * gp + sn[:, t] * gc
        g[:, c] = -sn[:, t] * gp + cs[:, t] * gc
    return dth * PI


# ---------------------------------------------------------------------------------------------
# The problem


class Problem:
    def __init__(self, ref: dict, fac: dt.Factorization):
        self.ref, self.fac = ref, fac
        self.sym = Sym()
        self.edges = fac.edges
        self.R, self.B, self.C = fac.w.shape
        self.Wmat = fac.W  # (R, C), fixed
        tei = ref["tei"]
        S = fac.S
        # Target in the pair basis: T_s[I(pq), I(rs)] = (tei + shift)[p,q,r,s] * scale_I * scale_J
        P, Q, sc = self.sym.P, self.sym.Q, self.sym.scale
        T = tei[P[:, None], Q[:, None], P[None, :], Q[None, :]].copy()  # (NPAIR, NPAIR)
        diag_pairs = np.where(P == Q)[0]
        # + 0.5 S[p,q] delta_rs  and  + 0.5 delta_pq S[r,s]
        T[:, diag_pairs] += 0.5 * S[P, Q][:, None]
        T[diag_pairs, :] += 0.5 * S[P, Q][None, :]
        self.T = T * sc[:, None] * sc[None, :]
        h1_thc = ref["oei"] - 0.5 * np.einsum("prrq->pq", tei)
        self.h1_base = h1_thc - 0.5 * ref["electrons"] * S + fac.beta_1 * np.eye(N)
        self.esos_const = ref["ecore"] - ref["electrons"] * fac.beta_1 - 0.5 * np.sum(fac.W ** 2)

    # -- pieces
    def lambda2(self, w: np.ndarray) -> float:
        return float(0.25 * np.sum((np.abs(w).sum(axis=1) + np.abs(self.Wmat + w.sum(axis=1))) ** 2))

    def lambda2_grad(self, w: np.ndarray) -> np.ndarray:
        wB = self.Wmat + w.sum(axis=1)
        tot = np.abs(w).sum(axis=1) + np.abs(wB)  # (R, C)
        return 0.5 * tot[:, None, :] * (np.sign(w) + np.sign(wB)[:, None, :])

    def h1_prime_from_M(self, M: np.ndarray, w: np.ndarray) -> np.ndarray:
        z = np.einsum("rc,rcs->s", self.Wmat, M.reshape(self.R, self.C, NPAIR))
        return self.h1_base - self.sym.to_matrix(z)

    negmass_target: float | None = None  # when set, the alpha term is alpha * (negmass - target)**2

    def negmass_for_egap(self, e_gap: float) -> float:
        """negmass that gives E_gap = e_gap (E_gap = E_UPPER - esos_const + 2 negmass)."""
        return 0.5 * (e_gap - (dt.E_UPPER - self.esos_const))

    def forward_backward(self, theta_q: np.ndarray, w: np.ndarray, alpha: float = 0.0, mu: float = 0.0,
                         need_grad: bool = True) -> dict:
        """Loss and gradients at quantised angles theta_q (theta/pi, (R*B, N-1)) and weights w.

        alpha weighs negmass linearly, or (negmass - negmass_target)**2 when `negmass_target` is set."""
        u, cache = givens_forward(theta_q, self.edges)
        out = self.forward_backward_u(u, w, alpha, mu, need_grad)
        if need_grad:
            out["dtheta"] = givens_backward(out["du"], cache, self.edges)
        return out

    def forward_backward_u(self, u: np.ndarray, w: np.ndarray, alpha: float = 0.0, mu: float = 0.0,
                           need_grad: bool = True) -> dict:
        """Same loss at point coordinates u (R*B, N) (rows need not be normalised: the verifier normalises;
        here they are used as given), gradient `du` instead of `dtheta`."""
        Us = self.sym.of_outer(u)  # (R*B, NPAIR)
        Usr = Us.reshape(self.R, self.B, NPAIR)
        M = np.matmul(w.transpose(0, 2, 1), Usr).reshape(self.R * self.C, NPAIR)
        Res = M.T @ M
        Res -= self.T
        L2 = float(np.sum(Res * Res))
        h1p = self.h1_prime_from_M(M, w)
        lam, V = np.linalg.eigh(h1p)
        neg = lam < 0
        negmass = float(-lam[neg].sum())
        lam1 = float(np.abs(lam).sum())
        lam2 = self.lambda2(w)
        Lambda = lam1 + lam2
        E_sos = self.esos_const - 2.0 * negmass
        E_gap = dt.E_UPPER - E_sos
        lam_eff = float(np.sqrt(E_gap * (2 * Lambda - E_gap))) if 0 < E_gap < 2 * Lambda else float("nan")
        if self.negmass_target is None:
            pen, alpha_lin = alpha * negmass, alpha
        else:
            pen = alpha * (negmass - self.negmass_target) ** 2
            alpha_lin = 2.0 * alpha * (negmass - self.negmass_target)
        out = {"L2": L2, "error_2": np.sqrt(L2), "negmass": negmass, "Lambda": Lambda, "E_gap": E_gap,
               "lambda_eff": lam_eff, "lambda2": lam2, "loss": L2 + pen + mu * lam2, "alpha_lin": alpha_lin,
               "u": u, "Us": Us, "M": M, "Res": Res, "lam": lam, "V": V}
        if not need_grad:
            return out
        dM = 4.0 * (M @ Res)  # (R*C, NPAIR)
        if alpha_lin:
            Pneg = V[:, neg] @ V[:, neg].T
            dM += alpha_lin * np.einsum("rc,s->rcs", self.Wmat, self.sym.of_matrix(Pneg)).reshape(self.R * self.C, NPAIR)
        dMr = dM.reshape(self.R, self.C, NPAIR)
        dw = np.matmul(Usr, dMr.transpose(0, 2, 1))
        if mu:
            dw += mu * self.lambda2_grad(w)
        dUs = np.matmul(w, dMr).reshape(self.R * self.B, NPAIR)
        out["du"], out["dw"] = self.sym.grad_outer(dUs, u), dw
        return out


# ---------------------------------------------------------------------------------------------
# Words <-> latent angles, exact evaluation


def quantize_theta(theta: np.ndarray, brot: int) -> tuple[np.ndarray, np.ndarray]:
    words = dt.quantize(theta, brot)
    return words, words / (1 << (brot - 1))


def one_body_feedback(lam: np.ndarray, V: np.ndarray, brot: int, edges: np.ndarray, cap: float = 0.5,
                      total_cap: float = 0.6) -> np.ndarray:
    """Sequential error-feedback encoding of the eigenvectors (columns of V) as brot-bit words.

    To first order the one-body error is E_ab = lambda_a d_a[b] + lambda_b d_b[a] (d_j = v_j - e_j in the
    eigenbasis).  Encoding the rows in decreasing |lambda| and giving row b the target
    e_b - sum_{a done} (lambda_a / lambda_b) d_a[b] e_a cancels the error the earlier rows left in
    row/column b, so only lambda_min(a,b) * (rounding error) survives.  Corrections are clipped to +-cap per
    coordinate and total_cap in norm (the verifier must still match row b to e_b by overlap).
    """
    n = len(lam)
    order = np.argsort(-np.abs(lam))
    coords = np.zeros((n, n))  # v_j in the eigenbasis
    words = np.zeros((n, n - 1), dtype=np.int64)
    done = []
    for b in order:
        target = np.zeros(n)
        target[b] = 1.0
        if done and abs(lam[b]) > 1e-9:
            a = np.array(done)
            corr = -(lam[a] / lam[b]) * coords[a, b]
            corr = np.clip(corr, -cap, cap)
            nrm = np.linalg.norm(corr)
            if nrm > total_cap:
                corr *= total_cap / nrm
            target[a] = corr
        target /= np.linalg.norm(target)
        words[b] = dt.quantize(dt.decompose((V @ target)[None, :], edges), brot)[0]
        coords[b] = V.T @ dt.decode(words[b][None, :], brot, edges)[0]
        done.append(b)
    return words


def one_body_words(prob: Problem, words2: np.ndarray, w: np.ndarray, brot: int, sweeps: int = 2,
                   verbose: bool = True, feedback: bool = True) -> tuple[np.ndarray, float]:
    """Words of the 76 one-body rows for quantised two-body words: eigenvectors of h1' encoded with
    error feedback (or plain rounding, whichever is better), then coordinate descent on
    error_1 = ||sum_j lambda_j v_j v_j^T - h1'||_F (+-1, +-2 moves)."""
    fac = dt.Factorization(w=w, W=prob.Wmat, S=prob.fac.S, beta_1=prob.fac.beta_1, edges=prob.edges,
                           U=prob.fac.U, H1_I=prob.fac.H1_I, Hsym_lower=prob.fac.Hsym_lower)
    u2 = dt.decode(words2, brot, prob.edges).reshape(prob.R, prob.B, N)
    h1p = dt.h1_prime(prob.ref, fac, u2)
    lam, V = np.linalg.eigh(h1p)

    def err_of(words1):
        v = dt.decode(words1, brot, prob.edges)
        E = np.einsum("j,jp,jq->pq", lam, v, v) - h1p
        return v, E, float(np.linalg.norm(E))

    words1 = dt.quantize(dt.decompose(V.T, prob.edges), brot)
    v, E, err = err_of(words1)
    if verbose:
        print(f"  one-body: rounded error_1 {1e3*err:.3f} mHa", flush=True)
    if feedback:
        best = (err, words1, v, E)
        for cap, tcap in ((0.1, 0.2), (0.25, 0.4), (0.5, 0.6), (1.0, 0.9)):
            wf = one_body_feedback(lam, V, brot, prob.edges, cap=cap, total_cap=tcap)
            vf, Ef, ef = err_of(wf)
            if verbose:
                print(f"  one-body: feedback cap {cap}/{tcap} error_1 {1e3*ef:.3f} mHa", flush=True)
            if ef < best[0]:
                best = (ef, wf, vf, Ef)
        err, words1, v, E = best
    order = np.argsort(-np.abs(lam))
    modulus = 1 << brot
    for sweep in range(sweeps):
        improved = 0
        for j in order:
            if abs(lam[j]) < 1e-12:
                continue
            cand = np.repeat(words1[j][None, :], 4 * (N - 1), axis=0)
            t_idx = np.repeat(np.arange(N - 1), 4)
            d = np.tile(np.array([1, -1, 2, -2]), N - 1)
            cand[np.arange(len(cand)), t_idx] = (cand[np.arange(len(cand)), t_idx] + d) % modulus
            base = E - lam[j] * np.outer(v[j], v[j])
            # continuous retarget moves: v' = normalize(v_j - s/lam_j * P_perp base v_j) for a few step sizes,
            # rounded; the first-order optimum of ||base + lam_j v v^T|| near v_j
            g = base @ v[j]
            g -= (g @ v[j]) * v[j]
            targets = [v[j] - (sc / lam[j]) * g for sc in (0.25, 0.5, 0.75, 1.0, 1.5)]
            targets = np.array([t / np.linalg.norm(t) for t in targets])
            cand = np.concatenate([cand, dt.quantize(dt.decompose(targets, prob.edges), brot)], axis=0)
            vj_new = dt.decode(cand, brot, prob.edges)
            # ||base + lam_j v' v'^T||^2 = ||base||^2 + 2 lam_j v'^T base v' + lam_j^2
            quad = np.einsum("ip,pq,iq->i", vj_new, base, vj_new)
            vals = np.sum(base * base) + 2 * lam[j] * quad + lam[j] ** 2
            i = int(np.argmin(vals))
            if vals[i] < err ** 2 - 1e-14:
                words1[j] = cand[i]
                v[j] = vj_new[i]
                E = base + lam[j] * np.outer(v[j], v[j])
                err = float(np.sqrt(vals[i]))
                improved += 1
        if verbose:
            print(f"  one-body sweep {sweep}: {improved} moves, error_1 {1e3*err:.3f} mHa", flush=True)
        if not improved:
            break
    return words1, err


def exact_eval(prob: Problem, words2: np.ndarray, w: np.ndarray, brot: int, one_body_sweeps: int = 2,
               verbose: bool = False) -> tuple[dict, np.ndarray]:
    """Verifier metrics of the full word table with w rounded to float32 (as stored)."""
    w32 = np.asarray(np.asarray(w, dtype=np.float32), dtype=np.float64)
    words1, _ = one_body_words(prob, words2, w32, brot, sweeps=one_body_sweeps, verbose=verbose)
    words = np.concatenate([words1, words2], axis=0)
    fac = dt.Factorization(w=w32, W=prob.Wmat, S=prob.fac.S, beta_1=prob.fac.beta_1, edges=prob.edges,
                           U=prob.fac.U, H1_I=prob.fac.H1_I, Hsym_lower=prob.fac.Hsym_lower)
    # W is stored through H2_I = W + sum w in float32; mimic that rounding too
    H2_I = np.asarray(np.asarray(fac.W + w32.sum(axis=1), dtype=np.float32), dtype=np.float64)
    fac.W = H2_I - w32.sum(axis=1)
    return dt.evaluate_words(prob.ref, fac, words, brot), words


# ---------------------------------------------------------------------------------------------
# QAT (Adam + STE)


class Adam:
    def __init__(self, shape, lr, b1=0.9, b2=0.999, eps=1e-12):
        self.m, self.v, self.t = np.zeros(shape), np.zeros(shape), 0
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps

    def step(self, x, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return x - self.lr * mh / (np.sqrt(vh) + self.eps)


def qat(prob: Problem, theta: np.ndarray, w: np.ndarray, brot: int | None, steps: int, lr_theta: float,
        lr_w: float, alpha: float = 0.0, mu: float = 0.0, eval_every: int = 50, log=log,
        lr_decay: float = 1.0) -> tuple[np.ndarray, np.ndarray, list]:
    """Adam on (latent theta/pi, w); the forward pass uses the brot-quantised angles (STE) when brot is set."""
    theta, w = theta.copy(), w.copy()
    opt_t, opt_w = Adam(theta.shape, lr_theta), Adam(w.shape, lr_w)
    history = []
    t0 = time.time()
    for step in range(steps + 1):
        theta_q = quantize_theta(theta, brot)[1] if brot else theta
        r = prob.forward_backward(theta_q, w, alpha, mu, need_grad=step < steps)
        if step % eval_every == 0 or step == steps:
            msg = (f"step {step:5d}  e2 {1e3*r['error_2']:8.3f} mHa  negmass {r['negmass']:.4f}  "
                   f"Lambda {r['Lambda']:.3f}  E_gap {r['E_gap']:.5f}  lam_eff {r['lambda_eff']:.4f}  "
                   f"loss {r['loss']:.6e}  ({time.time()-t0:.0f}s)")
            log(msg)
            history.append({k: r[k] for k in ("error_2", "negmass", "Lambda", "E_gap", "lambda_eff", "loss")})
        if step == steps:
            break
        theta = opt_t.step(theta, r["dtheta"])
        if lr_w > 0:
            w = opt_w.step(w, r["dw"])
        opt_t.lr *= lr_decay
        opt_w.lr *= lr_decay
    return theta, w, history


# ---------------------------------------------------------------------------------------------
# L-BFGS on the continuous variables (optionally with some vectors frozen at quantised words)

W_SCALE = 2.5e-3  # typical |w|: L-BFGS sees w / W_SCALE so that angle and weight coordinates are comparable


def lbfgs(prob: Problem, theta: np.ndarray, w: np.ndarray, *, opt_theta: bool = True, opt_w: bool = True,
          frozen: np.ndarray | None = None, alpha: float = 0.0, mu: float = 0.0, maxiter: int = 200,
          log=log, tag: str = "") -> tuple[np.ndarray, np.ndarray, dict]:
    """Minimise loss over the continuous theta (rows not in `frozen`) and/or w with scipy L-BFGS-B.

    `frozen` is a boolean mask over the R*B two-body rows whose theta stay fixed (already quantised)."""
    from scipy.optimize import minimize

    theta0, w0 = theta.copy(), w.copy()
    free = np.ones(theta.shape[0], dtype=bool) if frozen is None else ~frozen
    n_t = int(free.sum()) * theta.shape[1] if opt_theta else 0
    n_w = w.size if opt_w else 0
    x0 = np.concatenate([theta0[free].ravel() if opt_theta else np.empty(0), (w0 / W_SCALE).ravel() if opt_w else np.empty(0)])
    state = {"calls": 0, "t0": time.time(), "best": None}

    def unpack(x):
        th = theta0.copy()
        if opt_theta:
            th[free] = x[:n_t].reshape(-1, theta.shape[1])
        ww = w0 if not opt_w else x[n_t:].reshape(w.shape) * W_SCALE
        return th, ww

    def fun(x):
        th, ww = unpack(x)
        r = prob.forward_backward(th, ww, alpha, mu)
        state["calls"] += 1
        g = []
        if opt_theta:
            g.append(r["dtheta"][free].ravel())
        if opt_w:
            g.append((r["dw"] * W_SCALE).ravel())
        if state["calls"] % 25 == 1:
            log(f"  lbfgs{tag} call {state['calls']:4d}  e2 {1e3*r['error_2']:8.3f} mHa  negmass {r['negmass']:.4f}  "
                f"Lambda {r['Lambda']:.3f}  lam_eff {r['lambda_eff']:.4f}  ({time.time()-state['t0']:.0f}s)")
        return r["loss"], np.concatenate(g)

    res = minimize(fun, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "maxcor": 20, "ftol": 1e-14, "gtol": 1e-12})
    th, ww = unpack(res.x)
    r = prob.forward_backward(th, ww, alpha, mu, need_grad=False)
    log(f"  lbfgs{tag} done ({res.nit} it, {state['calls']} calls, {res.message}): e2 {1e3*r['error_2']:.3f} mHa  "
        f"negmass {r['negmass']:.4f}  Lambda {r['Lambda']:.3f}  lam_eff {r['lambda_eff']:.4f}")
    return th, ww, r


def progressive(prob: Problem, theta: np.ndarray, w: np.ndarray, brot: int, batches: int = 16,
                iters: int = 30, alpha: float = 0.0, mu: float = 0.0, seed: int = 0, log=log,
                final_iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Progressive quantisation: freeze the rows batch by batch at their rounded words and re-optimise
    the still-continuous rows and the weights after each batch (error diffusion across vectors)."""
    theta, w = theta.copy(), w.copy()
    m = theta.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    frozen = np.zeros(m, dtype=bool)
    for k, rows in enumerate(np.array_split(perm, batches)):
        theta[rows] = quantize_theta(theta[rows], brot)[1]
        frozen[rows] = True
        r = prob.forward_backward(theta, w, alpha, mu, need_grad=False)
        log(f"progressive batch {k+1}/{batches}: frozen {frozen.sum()}  e2 after rounding {1e3*r['error_2']:.3f} mHa")
        last = frozen.all()
        theta, w, r = lbfgs(prob, theta, w, opt_theta=not last, opt_w=True, frozen=frozen, alpha=alpha, mu=mu,
                            maxiter=final_iters if last else iters, log=log, tag=f"[prog {k+1}]")
    return theta, w


def refine(prob: Problem, theta: np.ndarray, w: np.ndarray, brot: int, rounds: int = 8, fraction: float = 0.125,
           sub_batches: int = 4, iters_free: int = 30, iters_sub: int = 15, alpha: float = 0.0, mu: float = 0.0,
           seed: int = 1, log=log) -> tuple[np.ndarray, np.ndarray]:
    """Unfreeze/refreeze rounds on words already on the grid: a random `fraction` of the rows is made
    continuous and re-optimised (with w) against all the frozen rows, then frozen again progressively."""
    theta, w = theta.copy(), w.copy()
    m = theta.shape[0]
    rng = np.random.default_rng(seed)
    for rnd in range(rounds):
        rows = rng.choice(m, int(fraction * m), replace=False)
        frozen = np.ones(m, dtype=bool)
        frozen[rows] = False
        theta, w, r = lbfgs(prob, theta, w, opt_theta=True, opt_w=True, frozen=frozen, alpha=alpha, mu=mu,
                            maxiter=iters_free, log=log, tag=f"[refine {rnd+1} free]")
        for k, sub in enumerate(np.array_split(rng.permutation(rows), sub_batches)):
            theta[sub] = quantize_theta(theta[sub], brot)[1]
            frozen[sub] = True
            last = frozen.all()
            theta, w, r = lbfgs(prob, theta, w, opt_theta=not last, opt_w=True, frozen=frozen, alpha=alpha, mu=mu,
                                maxiter=iters_sub, log=log, tag=f"[refine {rnd+1}.{k+1}]")
        log(f"refine round {rnd+1}/{rounds}: e2 {1e3*r['error_2']:.3f} mHa  lam_eff {r['lambda_eff']:.4f}")
    return theta, w


# ---------------------------------------------------------------------------------------------
# Exact coordinate descent on the two-body words


def polish(prob: Problem, words2: np.ndarray, w: np.ndarray, brot: int, sweeps: int = 1, alpha: float = 0.0,
           moves=(1, -1), log=log, order_seed: int = 0, max_moves: int = 1) -> np.ndarray:
    """Greedy best single-word move per vector per sweep (up to `max_moves` accepted in a row per vector),
    scored with the exact change of error_2^2 (plus alpha * first-order change of negmass)."""
    words2 = words2.copy()
    modulus = 1 << brot
    sym = prob.sym
    theta_q = words2 / (1 << (brot - 1))
    r = prob.forward_backward(theta_q, w, alpha, 0.0, need_grad=False)
    Us, M, Res = r["Us"], r["M"].reshape(prob.R, prob.C, NPAIR), r["Res"]
    L2 = r["L2"]
    log(f"polish start: e2 {1e3*np.sqrt(L2):.3f} mHa")
    rng = np.random.default_rng(order_seed)
    nm = len(moves)
    t_idx = np.repeat(np.arange(N - 1), nm)
    d_arr = np.tile(np.array(moves), N - 1)
    ncand = len(t_idx)
    for sweep in range(sweeps):
        t0 = time.time()
        # refresh P_neg for the first-order negmass term
        neg_vec, alpha_lin = None, 0.0

        def relinearise():
            # first-order negmass term; with a quadratic target the coefficient depends on the current negmass,
            # so it is refreshed every 200 vectors (a stale coefficient lets the sweep trade error for penalty)
            h1p = prob.h1_prime_from_M(M.reshape(prob.R * prob.C, NPAIR), w)
            lam, V = np.linalg.eigh(h1p)
            negmass = float(-lam[lam < 0].sum())
            return (sym.of_matrix(V[:, lam < 0] @ V[:, lam < 0].T),
                    alpha if prob.negmass_target is None else 2.0 * alpha * (negmass - prob.negmass_target))

        accepted, gain = 0, 0.0
        for ib, b in enumerate(rng.permutation(prob.R * prob.B)):
          if alpha and ib % 200 == 0:
              neg_vec, alpha_lin = relinearise()
          for _rep in range(max_moves):
            rho, bb = divmod(b, prob.B)
            wb = w[rho, bb, :]  # (C,)
            y = wb @ M[rho]  # (NPAIR,) = sum_c w_bc M[rho, c]
            k = float(wb @ wb)
            cand = np.repeat(words2[b][None, :], ncand, axis=0)
            cand[np.arange(ncand), t_idx] = (cand[np.arange(ncand), t_idx] + d_arr) % modulus
            u_new = dt.decode(cand, brot, prob.edges)
            delta = sym.of_outer(u_new) - Us[b][None, :]  # (ncand, NPAIR)
            RD = Res @ delta.T  # (NPAIR, ncand)
            yRd = y @ RD
            dRd = np.einsum("is,si->i", delta, RD)
            yd = delta @ y
            dd = np.einsum("is,is->i", delta, delta)
            yy = float(y @ y)
            dH2 = 2 * yy * dd + 2 * yd ** 2 + k * k * dd ** 2 + 4 * k * yd * dd
            dL = 4 * yRd + 2 * k * dRd + dH2
            if alpha_lin:
                dL = dL + alpha_lin * float(prob.Wmat[rho] @ wb) * (delta @ neg_vec)
            i = int(np.argmin(dL))
            if dL[i] >= -1e-15:
                break
            dlt = delta[i]
            words2[b] = cand[i]
            Us[b] += dlt
            M[rho] += np.outer(wb, dlt)
            Res += np.outer(y, dlt) + np.outer(dlt, y) + k * np.outer(dlt, dlt)
            L2 += 4 * yRd[i] + 2 * k * dRd[i] + dH2[i]
            accepted += 1
            gain += dL[i]
        log(f"polish sweep {sweep}: {accepted} moves, e2 {1e3*np.sqrt(max(L2, 0)):.3f} mHa "
            f"(dL {gain:.3e}, {time.time()-t0:.0f}s)")
        if not accepted:
            break
    # recompute exactly (the incremental L2 accumulates rounding)
    r = prob.forward_backward(words2 / (1 << (brot - 1)), w, need_grad=False)
    log(f"polish end: e2 {1e3*r['error_2']:.3f} mHa  negmass {r['negmass']:.4f}  lam_eff {r['lambda_eff']:.4f}")
    return words2


# ---------------------------------------------------------------------------------------------
# State files


def save_state(path: str, prob: Problem, theta: np.ndarray, w: np.ndarray, brot: int, words: np.ndarray | None,
               metrics: dict | None) -> None:
    arrays = {"theta": theta, "w": w, "brot": np.int64(brot)}
    if words is not None:
        arrays["angle_words"] = words
        u_all = dt.decode(words, brot, prob.edges)
        sub = prob.fac.npz_arrays(U=u_all[N:].reshape(prob.R, prob.B, N), angle_words=words)
        sub["W"] = np.asarray(w, dtype=np.float32)
        sub["H2_I"] = np.asarray(prob.fac.W + w.sum(axis=1), dtype=np.float32)
        sub["U"] = np.asarray(sub["U"], dtype=np.float32)
        arrays.update({f"sub_{k}": v for k, v in sub.items()})
    if metrics is not None:
        arrays.update({f"metric_{k}": np.float64(v) for k, v in metrics.items() if k != "gap_in_domain"})
    np.savez(path, **arrays)


def write_submission_npz(state_path: str, out_path: str) -> dict:
    with np.load(state_path) as z:
        sub = {k[4:]: z[k] for k in z.files if k.startswith("sub_")}
    np.savez(out_path, **sub)
    return sub


def load_state(path: str):
    with np.load(path) as z:
        return z["theta"], z["w"], int(z["brot"])


# ---------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brot", type=int, required=True)
    ap.add_argument("--init", default=None, help="state .npz from an earlier run (default: the baseline)")
    ap.add_argument("--mode", choices=("prog", "refine", "qat", "polish", "wfit", "eval"), default="prog")
    ap.add_argument("--rounds", type=int, default=8, help="refine: unfreeze/refreeze rounds")
    ap.add_argument("--fraction", type=float, default=0.125, help="refine: fraction of rows per round")
    ap.add_argument("--batches", type=int, default=16, help="prog: number of freeze batches")
    ap.add_argument("--iters", type=int, default=30, help="prog: L-BFGS iterations per batch; wfit: iterations")
    ap.add_argument("--steps", type=int, default=300, help="qat: Adam steps")
    ap.add_argument("--lr-theta", type=float, default=1e-5)
    ap.add_argument("--lr-w", type=float, default=1e-7)
    ap.add_argument("--alpha", type=float, default=0.0, help="weight of negmass(h1') in the loss")
    ap.add_argument("--mu", type=float, default=0.0, help="weight of Lambda_2 in the loss")
    ap.add_argument("--egap-target", type=float, default=None,
                    help="steer E_gap to this value (Ha, > 3.914898): alpha then weighs (negmass - target)^2")
    ap.add_argument("--polish", type=int, default=0, help="coordinate-descent sweeps (after the main phase)")
    ap.add_argument("--polish-rounds", type=int, default=1, help="rounds of (polish sweeps + w refit)")
    ap.add_argument("--polish-alpha", type=float, default=None, help="polish: negmass weight (default: --alpha)")
    ap.add_argument("--polish-moves", type=int, default=1, help="polish: word deltas +-1..+-this")
    ap.add_argument("--polish-max-moves", type=int, default=1, help="polish: accepted moves in a row per vector")
    ap.add_argument("--one-body-sweeps", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    ref = dt.load_reference_cached()
    fac, base_words = dt.load_baseline()
    prob = Problem(ref, fac)
    if args.init:
        theta, w, _ = load_state(args.init)
    else:
        theta = dt.decompose(fac.U.reshape(-1, N), fac.edges)
        w = fac.w.copy()
    brot = args.brot
    if args.egap_target is not None:
        prob.negmass_target = prob.negmass_for_egap(args.egap_target)
        log(f"E_gap target {args.egap_target} -> negmass target {prob.negmass_target:.6f}")
    words2 = quantize_theta(theta, brot)[0]
    m, _ = exact_eval(prob, words2, w, brot, one_body_sweeps=0)
    log(f"start (plain rounding of init): {dt.fmt(m)}")

    if args.mode == "prog":
        theta, w = progressive(prob, theta, w, brot, batches=args.batches, iters=args.iters, alpha=args.alpha, mu=args.mu)
        words2 = quantize_theta(theta, brot)[0]
    elif args.mode == "refine":
        theta = quantize_theta(theta, brot)[1]
        theta, w = refine(prob, theta, w, brot, rounds=args.rounds, fraction=args.fraction, iters_free=args.iters,
                          iters_sub=max(10, args.iters // 2), alpha=args.alpha, mu=args.mu)
        words2 = quantize_theta(theta, brot)[0]
    elif args.mode == "qat":
        theta, w, _ = qat(prob, theta, w, brot, args.steps, args.lr_theta, args.lr_w, args.alpha, args.mu)
        words2 = quantize_theta(theta, brot)[0]
    elif args.mode == "wfit":
        theta = quantize_theta(theta, brot)[1]
        theta, w, _ = lbfgs(prob, theta, w, opt_theta=False, opt_w=True, alpha=args.alpha, mu=args.mu, maxiter=args.iters, tag="[wfit]")
        words2 = quantize_theta(theta, brot)[0]
    if args.mode != "eval":
        m, _ = exact_eval(prob, words2, w, brot, one_body_sweeps=0)
        log(f"after {args.mode}: {dt.fmt(m)}")
    for rnd in range(args.polish_rounds if args.polish else 0):
        mv = tuple(d for k in range(1, args.polish_moves + 1) for d in (k, -k))
        pa = args.alpha if args.polish_alpha is None else args.polish_alpha
        words2 = polish(prob, words2, w, brot, sweeps=args.polish, alpha=pa, moves=mv, max_moves=args.polish_max_moves)
        theta = words2 / (1 << (brot - 1))
        theta, w, _ = lbfgs(prob, theta, w, opt_theta=False, opt_w=True, alpha=args.alpha, mu=args.mu, maxiter=args.iters, tag=f"[w after polish {rnd}]")
        m, _ = exact_eval(prob, words2, w, brot, one_body_sweeps=0)
        log(f"after polish round {rnd}: {dt.fmt(m)}")
    m, words = exact_eval(prob, words2, w, brot, one_body_sweeps=args.one_body_sweeps, verbose=True)
    log(f"final: {dt.fmt(m)}")
    save_state(args.out, prob, theta, w, brot, words, m)
    log(f"saved {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
