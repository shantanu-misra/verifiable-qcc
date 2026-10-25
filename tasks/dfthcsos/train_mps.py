"""Adam training of the grouped DF-THC factorisation in PyTorch (MPS or CPU), with progressive group dropping.

Variables: points u (R*B, N) (unnormalised: the dyad scale is absorbed by the weights and divided out when a
state is saved), weights w (R, B, C), offsets W (R, C).  Loss (float32 on the device):
    error_2^2 + a_gap (E_gap - target)^2 + beta D(w) + mu Lambda_2
with D(w) = 1/2 sum_rc [(sum_b |w|)^2 - (sum_b w)^2] (zero iff every column is single-signed) and the verifier's
E_gap = E_UPPER - ecore + eta beta_1 + 1/2 sum W^2 + 2 negmass(h1'), h1' = h1_base - sum_rc W_rc X_rc (eigvalsh on a
76 x 76 matrix).  Every `eval_every` steps the state is evaluated in float64 with the verifier's own metric code
(dfthc_tools / refit.Problem) and the best state of the stage is kept.  Between stages the groups with the smallest
contribution ||sum_c X_rc (x) X_rc||_F are dropped.  Each stage checkpoints to <out-prefix>R<R>.npz (u normalised,
w rescaled, W), loadable by prune.load_state / quantize_state.py.

    python3 tasks/dfthcsos/train_mps.py --device mps --bench 100
    python3 tasks/dfthcsos/train_mps.py --device mps --schedule 60,55,48,42,36,32,28,26 --steps 100000 --out-prefix build/mps_
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dfthc_tools as dt  # noqa: E402
import refit as rf  # noqa: E402
import prune as pr  # noqa: E402

N = dt.N
log = rf.log


class Trainer:
    def __init__(self, ref, base, u, w, W, device: str, a_gap: float, egap_target: float, beta: float, mu: float):
        self.ref, self.base = ref, base
        self.dev = torch.device(device)
        self.dtype = torch.float32
        sym = rf.Sym()
        self.P = torch.tensor(sym.P, device=self.dev)
        self.Q = torch.tensor(sym.Q, device=self.dev)
        self.scale = torch.tensor(sym.scale, device=self.dev, dtype=self.dtype)
        prob0 = rf.Problem(ref, base)
        self.T = torch.tensor(prob0.T, device=self.dev, dtype=self.dtype)
        self.h1_base = torch.tensor(prob0.h1_base, device=self.dev, dtype=self.dtype)
        self.c_gap = float(dt.E_UPPER - ref["ecore"] + ref["electrons"] * base.beta_1)  # E_gap = c_gap + 0.5 sum W^2 + 2 negmass
        self.sym_np = sym
        self.a_gap, self.egap_target, self.beta, self.mu = a_gap, egap_target, beta, mu
        self.set_state(u, w, W)

    def set_state(self, u, w, W):
        R, B, C = w.shape
        self.R, self.B, self.C = R, B, C
        self.u = torch.tensor(np.asarray(u).reshape(R * B, N), device=self.dev, dtype=self.dtype, requires_grad=True)
        self.w = torch.tensor(np.asarray(w), device=self.dev, dtype=self.dtype, requires_grad=True)
        self.W = torch.tensor(np.asarray(W), device=self.dev, dtype=self.dtype, requires_grad=True)

    def forward(self, need_parts: bool = False):
        u, w, W = self.u, self.w, self.W
        R, B, C = self.R, self.B, self.C
        Us = (u[:, self.P] * u[:, self.Q] * self.scale).reshape(R, B, -1)  # (R, B, NPAIR)
        M = torch.bmm(w.transpose(1, 2), Us)  # (R, C, NPAIR)
        M2 = M.reshape(R * C, -1)
        H = M2.T @ M2
        L2 = ((H - self.T) ** 2).sum()
        # one-body absorption and E_gap
        z = (W.reshape(R * C, 1) * M2).sum(dim=0) / self.scale  # unscaled pair vector
        h1 = self.h1_base.clone()
        h1 = h1.index_put((self.P, self.Q), self.h1_base[self.P, self.Q] - z)
        h1 = h1.index_put((self.Q, self.P), self.h1_base[self.Q, self.P] - z)
        lam = torch.linalg.eigvalsh(h1)
        negmass = -torch.clamp(lam, max=0.0).sum()
        E_gap = self.c_gap + 0.5 * (W ** 2).sum() + 2.0 * negmass
        s = w.sum(dim=1)  # (R, C)
        l1 = w.abs().sum(dim=1)
        D = 0.5 * ((l1 ** 2) - (s ** 2)).sum()
        lam2 = 0.25 * ((l1 + (W + s).abs()) ** 2).sum()
        loss = L2 + self.a_gap * (E_gap - self.egap_target) ** 2 + self.beta * D + self.mu * lam2
        if need_parts:
            Lambda = lam.abs().sum() + lam2
            return loss, dict(L2=L2, E_gap=E_gap, negmass=negmass, D=D, lam2=lam2, Lambda=Lambda,
                              lam_eff=torch.sqrt(torch.clamp(E_gap * (2 * Lambda - E_gap), min=0.0)), M=M.detach())
        return loss, None

    def numpy_state(self):
        u = self.u.detach().cpu().numpy().astype(np.float64)
        w = self.w.detach().cpu().numpy().astype(np.float64)
        W = self.W.detach().cpu().numpy().astype(np.float64)
        un, wn = pr.normalise_state(u, w)
        return un, wn, W

    def evaluate64(self):
        """Verifier metrics in float64 (continuous rows)."""
        u, w, W = self.numpy_state()
        prob = pr.make_problem(self.ref, self.base, u, w, W, self.egap_target, 0.0)
        r = prob.forward_backward_u(u.reshape(-1, N), w, 0.0, 0.0, False, W=W)
        return {k: float(r[k]) for k in ("error_2", "E_gap", "negmass", "Lambda", "lambda2", "lambda_eff")}

    def drop_groups(self, R_new: int):
        with torch.no_grad():
            _, parts = self.forward(need_parts=True)
            M = parts["M"]
            contrib = torch.stack([torch.linalg.norm(M[r] @ M[r].T) for r in range(self.R)]).cpu().numpy()
        keep = np.sort(np.argsort(-contrib)[:R_new])
        log(f"dropping {self.R - R_new} groups (contributions {np.round(np.sort(contrib)[:self.R-R_new], 4).tolist()})")
        u = self.u.detach().cpu().numpy().reshape(self.R, self.B, N)[keep]
        w = self.w.detach().cpu().numpy()[keep]
        W = self.W.detach().cpu().numpy()[keep]
        self.set_state(u, w, W)

    def train_stage(self, steps: int, lr0: float, lr1: float, eval_every: int, tag: str, w_lr_scale: float = 1.0):
        params = [{"params": [self.u], "lr": lr0}, {"params": [self.w], "lr": lr0 * w_lr_scale}, {"params": [self.W], "lr": lr0}]
        opt = torch.optim.Adam(params)
        best = None
        t0 = time.time()
        for step in range(steps + 1):
            frac = step / max(steps, 1)
            lr = lr1 + 0.5 * (lr0 - lr1) * (1 + np.cos(np.pi * frac))  # cosine decay lr0 -> lr1
            for g, sc in zip(opt.param_groups, (1.0, w_lr_scale, 1.0)):
                g["lr"] = lr * sc
            if step % eval_every == 0 or step == steps:
                m = self.evaluate64()
                dt_s = time.time() - t0
                log(f"{tag} step {step:7d}  lr {lr:.2e}  e2 {1e3*m['error_2']:8.3f} mHa  E_gap {m['E_gap']:.4f}  "
                    f"Lambda {m['Lambda']:.2f}  lam_eff {m['lambda_eff']:.3f}  ({dt_s:.0f}s, {step/max(dt_s,1e-9):.1f} steps/s)")
                if best is None or m["error_2"] < best[0]:
                    best = (m["error_2"], self.numpy_state(), m)
            if step == steps:
                break
            opt.zero_grad(set_to_none=True)
            loss, _ = self.forward()
            loss.backward()
            opt.step()
        return best


def save_state(path: str, u, w, W, metrics: dict, base):
    np.savez(path, u=u, w=w, W=W, theta=dt.decompose(u.reshape(-1, N), base.edges),
             metrics=np.array([metrics["error_2"], metrics["E_gap"], metrics["Lambda"], metrics["lambda_eff"]]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="baseline")
    ap.add_argument("--C", type=int, default=45)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--schedule", default="60,55,48,42,36,32,28,26", help="group counts after each drop")
    ap.add_argument("--steps", type=int, default=100000, help="Adam steps per stage")
    ap.add_argument("--steps-first", type=int, default=None, help="steps for the first stage (default: --steps)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr-end", type=float, default=2e-5)
    ap.add_argument("--w-lr-scale", type=float, default=1.0)
    ap.add_argument("--a-gap", type=float, default=10.0)
    ap.add_argument("--egap-target", type=float, default=3.918)
    ap.add_argument("--beta", type=float, default=0.0, help="weight of the semidefiniteness penalty D(w)")
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--bench", type=int, default=0, help="time this many steps and exit")
    ap.add_argument("--out-prefix", default="build/mps_")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ref = dt.load_reference_cached()
    base, _ = dt.load_baseline()
    u, w, W = pr.load_state(args.init, base)
    if args.C and args.C < w.shape[2]:
        w, W = pr.prune_columns(w, W, args.C)
    tr = Trainer(ref, base, u, w, W, args.device, args.a_gap, args.egap_target, args.beta, args.mu)
    m = tr.evaluate64()
    log(f"init {args.init} {w.shape} on {args.device}: e2 {1e3*m['error_2']:.3f} mHa  E_gap {m['E_gap']:.4f}  Lambda {m['Lambda']:.2f}")
    if args.bench:
        opt = torch.optim.Adam([tr.u, tr.w, tr.W], lr=1e-6)
        for _ in range(3):
            opt.zero_grad(); loss, _ = tr.forward(); loss.backward(); opt.step()
        if args.device == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        for _ in range(args.bench):
            opt.zero_grad(); loss, _ = tr.forward(); loss.backward(); opt.step()
        if args.device == "mps":
            torch.mps.synchronize()
        dt_s = time.time() - t0
        log(f"bench {args.device}: {args.bench} Adam steps in {dt_s:.1f}s = {args.bench/dt_s:.2f} steps/s ({1e3*dt_s/args.bench:.0f} ms/step)")
        return
    schedule = [int(x) for x in args.schedule.split(",") if x]
    for i, R_new in enumerate(schedule):
        if R_new < tr.R:
            tr.drop_groups(R_new)
            m = tr.evaluate64()
            log(f"after drop -> R {tr.R}: e2 {1e3*m['error_2']:.3f} mHa  E_gap {m['E_gap']:.4f}")
        steps = args.steps_first if (i == 0 and args.steps_first) else args.steps
        best = tr.train_stage(steps, args.lr, args.lr_end, args.eval_every, tag=f"[R{tr.R}]", w_lr_scale=args.w_lr_scale)
        err, (u, w, W), m = best
        out = f"{args.out_prefix}R{tr.R}.npz"
        save_state(out, u, w, W, m, base)
        log(f"stage R {tr.R} best e2 {1e3*err:.3f} mHa  E_gap {m['E_gap']:.4f}  Lambda {m['Lambda']:.2f}  lam_eff {m['lambda_eff']:.3f} -> {out}")
        tr.set_state(u, w, W)


if __name__ == "__main__":
    main()
