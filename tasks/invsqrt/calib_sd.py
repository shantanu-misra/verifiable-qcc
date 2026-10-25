"""LP calibration of the Z_POW angles of a signed-digit invsqrt circuit (idea credited to lightningv08
and mizeroluckygall on Kaggle; implementation ours).

The circuit's phase is sum_j theta_j * bit_j(x) over the Z_POW positions of the two shifted words
(high word: whi + 10 positions, low word: LB - GY + 10 positions). The exact angles are powers of two;
positions whose angle is Clifford (Z, S) or T are kept exact. The remaining ("rotation") angles are
re-optimised as integers in units of 2^-64 turns so that the smallest absolute margin
min_x (tol(x) - |err(x)|) is maximised (that margin sets the synthesis eps and so the T-count per
rotation), optionally after dropping the lowest-weight rotations altogether (fewer Z_POWs).
Cutting-plane LP over the worst-margin inputs; every candidate is re-checked exactly on all x.

Usage (inside Docker):
    python tasks/invsqrt/calib_sd.py --table sd_tables/<cfg>.json [--drop 0 1 2] [--out angles.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'data', 'verifiers', 'invsqrt'))
import model_sd as ms
import build_sd as bs
import verify_submission_invsqrt as verifier

MASK64 = (1 << 64) - 1


def positions(cfg: ms.Config, lay: dict) -> list[tuple[str, int, int]]:
    """(word, index j, exponent e) for every Z_POW position; phase of a set bit is 2^e turns."""
    pos = []
    for j in range(lay['whi'] + ms.N_P):
        e = cfg.S - cfg.FY - cfg.SB + j
        if e < 0:
            pos.append(('high', j, e))
    for j in range(cfg.LB - cfg.GY + ms.N_P):
        e = cfg.GY - cfg.FY - cfg.SB + j
        if e < 0:
            pos.append(('low', j, e))
    return pos


def word_bits(cfg: ms.Config, tables: dict, lay: dict, xs: np.ndarray) -> dict[str, np.ndarray]:
    """Shifted high and low words for every x (uint64)."""
    dec = ms.decompose(cfg, xs)
    _, Y = ms.phase_words(cfg, tables, lay, xs)
    seg = dec['seg']
    hi_tab = np.zeros(256, dtype=np.int64)
    lo_tab = np.zeros(256, dtype=np.int64)
    for s_ in tables:
        hi_tab[s_], lo_tab[s_] = lay['hi'][s_], lay['lo'][s_]
    H = hi_tab[seg]
    L = Y - (H << cfg.S)
    assert (L >= 0).all() and (L < 2**cfg.LB).all()
    s = dec['s'].astype(np.uint64)
    return dict(high=H.astype(np.uint64) << s, low=(L >> cfg.GY).astype(np.uint64) << s)


def phase_from_angles(words: dict, pos: list, angles: list[int]) -> np.ndarray:
    acc = np.zeros(len(words['low']), dtype=np.uint64)
    for (w, j, _), a in zip(pos, angles):
        a &= MASK64
        if a:
            acc += ((words[w] >> np.uint64(j)) & np.uint64(1)) * np.uint64(a)
    return acc


def evaluate(words, pos, angles, target, tol):
    ph = phase_from_angles(words, pos, angles)
    err = (ph - target).view(np.int64)
    used = np.abs(err).view(np.uint64)
    ratio = used / tol
    margin = (tol.astype(np.float64) - used.astype(np.float64))
    return err, float(ratio.max()), int(np.argmax(ratio)), float(margin.min())


def rotation_t(n_rot: int, margin_units: float) -> float:
    eps = math.pi * margin_units / 2**64 / n_rot
    return n_rot * (0.56 * math.log2(1 / eps) + 5.3)


def calibrate(cfg, tables, lay, drop: int = 0, iters: int = 15, active0: int = 20000, verbose=True):
    xs = np.arange(1, ms.MAX_X + 1, dtype=np.int64)
    spec = verifier.ProblemSpecInvSqrt.for_n_p(ms.N_P)
    target, tol = verifier.reference_values(spec, xs.astype(np.uint64))
    words = word_bits(cfg, tables, lay, xs)
    pos = positions(cfg, lay)
    exact = [1 << (64 + e) for (_, _, e) in pos]
    rot_idx = [i for i, (_, _, e) in enumerate(pos) if e <= -3]            # T and finer
    rot_idx = [i for i in rot_idx if pos[i][2] < -3]                        # keep T exact; rotations only
    # drop the lowest-weight rotations (smallest e) first
    order = sorted(rot_idx, key=lambda i: pos[i][2])
    dropped, var = order[:drop], order[drop:]
    angles = list(exact)
    for i in dropped:
        angles[i] = 0
    err, worst, wx, margin = evaluate(words, pos, angles, target, tol)
    if verbose:
        print(f'drop={drop}: start worst {worst:.4f} margin {margin:.3e} units ({len(var)} rotations)', flush=True)
    bits = np.stack([((words[pos[i][0]] >> np.uint64(pos[i][1])) & np.uint64(1)).astype(np.float64) for i in var], axis=1)
    tolf, errf = tol.astype(np.float64), err.astype(np.float64)
    act = np.argsort(tolf - np.abs(errf))[:active0]
    best = None
    S = 2.0**24
    for it in range(iters):
        A, e, tl = bits[act], errf[act] / S, tolf[act] / S
        n = len(var)
        c = np.zeros(n + 1)
        c[-1] = -1                                                           # maximise m
        one = np.ones(len(act))
        A_ub = np.r_[np.c_[A, one], np.c_[-A, one]]
        b_ub = np.r_[tl - e, tl + e]
        bounds = [(-2.0**14, 2.0**14)] * n + [(None, None)]
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if r.x is None:
            print('LP failed', r.message)
            break
        d = np.round(r.x[:n] * S).astype(np.int64)
        tot = errf + bits @ d.astype(np.float64)
        slack = tolf - np.abs(tot)
        m = slack.min()
        if verbose:
            print(f'  it {it}: lp margin {r.x[-1] * S:.3e}, full-domain margin {m:.3e}, worst ratio {(np.abs(tot) / tolf).max():.4f}', flush=True)
        if best is None or m > best[0]:
            best = (m, d.copy())
        new = np.argsort(slack)[:5000]
        act2 = np.union1d(act, new)
        if len(act2) == len(act):
            break
        act = act2
    m, d = best
    for k, i in enumerate(var):
        angles[i] = int(exact[i]) + int(d[k])
    err, worst, wx, margin = evaluate(words, pos, angles, target, tol)
    return dict(angles=angles, pos=pos, worst=worst, worst_x=int(xs[wx]), margin=margin, n_rot=len(var),
                rot_T=rotation_t(len(var), margin) if margin > 0 else None, dropped=len(dropped))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--table', required=True)
    ap.add_argument('--drop', nargs='*', type=int, default=[0])
    ap.add_argument('--out', help='write the best angle set as JSON')
    args = ap.parse_args()
    cfg, tables, lay = bs.load(args.table)
    t0 = time.time()
    # baseline
    xs = np.arange(1, ms.MAX_X + 1, dtype=np.int64)
    spec = verifier.ProblemSpecInvSqrt.for_n_p(ms.N_P)
    target, tol = verifier.reference_values(spec, xs.astype(np.uint64))
    words = word_bits(cfg, tables, lay, xs)
    pos = positions(cfg, lay)
    exact = [1 << (64 + e) for (_, _, e) in pos]
    _, worst, wx, margin = evaluate(words, pos, exact, target, tol)
    n_rot = sum(1 for (_, _, e) in pos if e < -3)
    print(f'exact angles: {len(pos)} Z_POWs, {n_rot} rotations, worst {worst:.4f} at x={xs[wx]}, margin {margin:.3e}, '
          f'rotation T {rotation_t(n_rot, margin):.1f} ({rotation_t(n_rot, margin) / 4:.1f} Toffoli-eq)', flush=True)
    best = None
    for drop in args.drop:
        res = calibrate(cfg, tables, lay, drop=drop)
        ok = res['worst'] < 1 and res['margin'] > 0
        print(f'drop={drop}: worst {res["worst"]:.4f} at x={res["worst_x"]}, margin {res["margin"]:.3e}, rotations {res["n_rot"]}, '
              f'rotation T {res["rot_T"] if res["rot_T"] else float("nan"):.1f} '
              f'({(res["rot_T"] or 0) / 4:.1f} Toffoli-eq) {"OK" if ok else "FAIL"} [{time.time() - t0:.0f}s]', flush=True)
        if ok and (best is None or res['rot_T'] < best['rot_T']):
            best = res
    if args.out and best:
        with open(args.out, 'w') as f:
            json.dump(dict(table=os.path.basename(args.table), dropped=best['dropped'], worst=best['worst'],
                           worst_x=best['worst_x'], margin=best['margin'], n_rot=best['n_rot'], rot_T=best['rot_T'],
                           angles=[dict(word=w, j=j, e=e, units=str(a)) for (w, j, e), a in zip(best['pos'], best['angles'])]), f, indent=1)
        print('wrote', args.out)


if __name__ == '__main__':
    main()
