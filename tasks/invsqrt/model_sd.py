"""Classical fixed-point model for the signed-digit Horner invsqrt design (tasks/invsqrt/notes.md §10).

Ideas re-implemented from the public Kaggle notebooks (Sumit Chongder, anazemcev, Rudra Bhaskar,
kirilltym, Alex Chilton); all code here is our own.

Pipeline for one input x (22 bits):
  normalisation  'pairs' : s = leading zero pairs, x_norm = x << 2s, M = x_norm / 2^20 in [1, 4),
                           y = 2 / sqrt(M) in (1, 2],  1/sqrt(x) = y * 2^(s - 11);  segment = x_norm >> 14.
                 'binary': lz = leading zeros, x_norm = x << lz, M = x_norm / 2^21 in [1, 2),
                           e = lz & 1, s = lz >> 1, y = 2^((e-1)/2) / sqrt(M) in (0.5, 1],
                           1/sqrt(x) = y * 2^(s - 10);  segment = (e, (x_norm >> 14) & 127).
  digits         d_i = bit i of x_norm for i < 14, sig_i = 2 d_i - 1 (centred, +-1),
                 u = sum_i sig_i 2^(i - DU) = M - M_mid   (DU = 21 'pairs', 22 'binary').
  quadratic      y ~= c0 - c1 u + c2 u^2 per segment, integers K2 = c2 2^F2, K1 = c1 2^F1, K0 = c0 2^FY.
  Horner rows    R = K1 - sum_{i in rows2} sig_i rd(K2, sh2_i),   sh2_i = F2 + DU - i - F1
                 Q = sum_{i < 14} sig_i rd(R, sh1_i),             sh1_i = F1 + DU - i - FY
                 Y = K0 - Q
                 rd(v, s) = round-to-nearest shift: (v >> s) + bit(v, s-1); exactly what an adder of
                 v >> s with carry-in bit(v, s-1) computes. A row with sh2_i >= width(K2) is dropped.
  phase word     ((Y >> GY) << s) mod 2^F,  F = FY - GY + SB (SB = 11 'pairs', 10 'binary').
Optional layouts that do not change the arithmetic: a high/low split of K0 (Y = L + 2^S H, L in
[0, 2^LB); the two phases add) and a derivative split R = base 2^(RB-1) + low (low in [0, 2^RB)).

Coefficients are fitted per segment: weighted least squares with Lawson re-weighting (near-minimax
relative error in float) as the seed, then a greedy integer search on (K2, K1) with K0 re-optimised
exactly, all against the exact integer pipeline. The full check compares the phase word of every
x in 1..3,133,452 with the verifier's own `reference_values` (read-only import).

Usage (inside Docker):
    python tasks/invsqrt/model_sd.py                       # default configs
    python tasks/invsqrt/model_sd.py --configs binary:10:19:28:2 pairs:11:21:30:0   # norm:F2:F1:FY:GY
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'data', 'verifiers', 'invsqrt'))
import verify_submission_invsqrt as verifier  # read-only

N_P = 10
WIDTH = 22
MAX_X = 3 * (2**N_P - 2)**2
DB = 14  # digit positions


@dataclasses.dataclass(frozen=True)
class Config:
    norm: str = 'binary'   # 'pairs' | 'binary'
    F2: int = 10           # fractional bits of K2
    F1: int = 19           # fractional bits of K1 / R
    FY: int = 28           # fractional bits of K0 / Y
    GY: int = 2            # low bits of Y not phased
    drop2: tuple = ()      # extra R rows (digit indices) dropped
    LB: int | None = 21    # low-register width for the high/low split (None: no split)
    S: int | None = 20     # split position (K0 = H 2^S + k0lo)
    RB: int | None = 12    # residual width for the derivative split (None: no split)

    @property
    def DU(self) -> int:
        return 21 if self.norm == 'pairs' else 22

    @property
    def SB(self) -> int:
        return 11 if self.norm == 'pairs' else 10

    @property
    def F(self) -> int:
        return self.FY - self.GY + self.SB

    def sh2(self, i: int) -> int:
        return self.F2 + self.DU - i - self.F1

    def sh1(self, i: int) -> int:
        return self.F1 + self.DU - i - self.FY

    def rows2(self, w2: int) -> list[int]:
        """Digits whose K2 row survives (shifted K2 is not identically zero), high digit first."""
        return [i for i in reversed(range(DB)) if i not in self.drop2 and self.sh2(i) < w2]

    def name(self) -> str:
        split = f'_L{self.LB}S{self.S}' if self.LB else '_nosplit'
        ds = f'_RB{self.RB}' if self.RB else ''
        dr = ('_drop' + ''.join(map(str, self.drop2))) if self.drop2 else ''
        return f'sd_{self.norm}_F2{self.F2}_F1{self.F1}_FY{self.FY}_GY{self.GY}{dr}{split}{ds}'


# ---------------------------------------------------------------------------------------------
# Decomposition of x.

def decompose(cfg: Config, xs: np.ndarray) -> dict:
    xs = np.asarray(xs, dtype=np.int64)
    lz = WIDTH - 1 - np.floor(np.log2(xs)).astype(np.int64)
    if cfg.norm == 'pairs':
        s = lz // 2
        x_norm = xs << (2 * s)
        seg = x_norm >> DB                    # 8 bits, values 64..255
    else:
        s = lz >> 1
        x_norm = xs << lz
        seg = ((x_norm >> DB) & 127) + ((lz & 1) << 7)   # 7 mantissa bits + parity
    return dict(lz=lz, s=s, x_norm=x_norm, seg=seg, d=x_norm & (2**DB - 1))


def segment_geometry(cfg: Config, seg: int) -> tuple[float, float, float]:
    """(M_lo, M_mid, scale) of a segment: y = scale / sqrt(M), M = M_mid + u."""
    if cfg.norm == 'pairs':
        m_lo, width, scale = seg * 2.0**-6, 2.0**-6, 2.0
    else:
        e, a = seg >> 7, seg & 127
        m_lo, width, scale = 1 + a * 2.0**-7, 2.0**-7, 2.0**((e - 1) / 2)
    m_mid = m_lo + (2**DB - 1) * 2.0**(-cfg.DU)
    return m_lo, m_mid, scale


def u_values(cfg: Config, d: np.ndarray) -> np.ndarray:
    """u = sum sig_i 2^(i-DU) as float."""
    sig = 2 * ((d[:, None] >> np.arange(DB)) & 1) - 1
    return sig @ (2.0 ** (np.arange(DB) - cfg.DU))


# ---------------------------------------------------------------------------------------------
# Exact integer pipeline.

def rd(v, sh: int):
    if sh < 0:
        return v << (-sh)
    if sh == 0:
        return v
    return (v >> sh) + ((v >> (sh - 1)) & 1)


def pipeline(cfg: Config, w2: int, K2: int, K1: int, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (R, Q) for digit words d (int64 arrays)."""
    sig = [2 * ((d >> i) & 1) - 1 for i in range(DB)]
    R = np.full_like(d, K1)
    for i in cfg.rows2(w2):
        R = R - sig[i] * rd(K2, cfg.sh2(i))
    Q = np.zeros_like(d)
    for i in reversed(range(DB)):
        Q = Q + sig[i] * rd(R, cfg.sh1(i))
    return R, Q


# ---------------------------------------------------------------------------------------------
# Fitting.

def lawson_fit(u: np.ndarray, f: np.ndarray, deg: int = 2, iters: int = 60) -> np.ndarray:
    """Near-minimax polynomial for relative error: IRLS with Lawson weights. Returns coefficients
    a_0..a_deg of f ~= sum a_j u^j."""
    V = np.vander(u, deg + 1, increasing=True)
    rel = 1 / f
    w = np.ones_like(u)
    coef = None
    for _ in range(iters):
        sw = rel * np.sqrt(w)
        coef = np.linalg.lstsq(V * sw[:, None], f * sw, rcond=None)[0]
        err = np.abs((V @ coef - f) * rel)
        w = w * err
        w /= w.sum()
    return coef


def best_k0(tail: np.ndarray, target: np.ndarray, gy: int) -> tuple[int, float]:
    """Integer K0 minimising max |((K0 - Q) >> GY << GY) - target| / target  (convex-ish; search near midpoint)."""
    # Y = K0 - Q; phased value = (Y >> gy) << gy. Work with floats; probe a small window.
    diff = -tail  # Y = K0 + diff
    mid = (target - diff)
    c = int(round((mid.max() + mid.min()) / 2))
    best = (math.inf, c)
    for k0 in range(c - 2 * 2**gy, c + 2 * 2**gy + 1):
        y = ((k0 + diff).astype(np.int64) >> gy) << gy
        err = float(np.max(np.abs(y - target) / target))
        if err < best[0]:
            best = (err, k0)
    return best[1], best[0]


def fit_segment(cfg: Config, w2: int, seg: int, d: np.ndarray) -> tuple[tuple[int, int, int], float]:
    m_lo, m_mid, scale = segment_geometry(cfg, seg)
    u = u_values(cfg, d)
    m = m_mid + u
    f = scale / np.sqrt(m)
    target = f * 2.0**cfg.FY
    a = lawson_fit(u, f)
    K2c, K1c = int(round(a[2] * 2**cfg.F2)), int(round(-a[1] * 2**cfg.F1))

    def score(k2, k1):
        _, Q = pipeline(cfg, w2, k2, k1, d)
        return best_k0(Q, target, cfg.GY)

    best = None
    for k2 in range(max(0, K2c - 1), K2c + 2):
        for k1 in range(K1c - 2, K1c + 3):
            k0, err = score(k2, k1)
            if best is None or err < best[0]:
                best = (err, k2, k1, k0)
    # Greedy refinement.
    improved = True
    while improved:
        improved = False
        err0, k2, k1, k0 = best
        for dk2, dk1 in ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 4), (0, -4), (0, 16), (0, -16)):
            n2, n1 = k2 + dk2, k1 + dk1
            if n2 < 0:
                continue
            nk0, err = score(n2, n1)
            if err < err0:
                best, improved = (err, n2, n1, nk0), True
                break
    err, k2, k1, k0 = best
    return (k2, k1, k0), err


def fit_all(cfg: Config, verbose: bool = False) -> tuple[dict[int, tuple[int, int, int]], float]:
    d = np.arange(2**DB, dtype=np.int64)
    w2 = cfg.F2  # K2 = c2 2^F2 with c2 < 1  -> at most F2 bits; refined below from the actual table
    segs = range(64, 256) if cfg.norm == 'pairs' else range(256)
    tables, worst = {}, 0.0
    for seg in segs:
        coefs, err = fit_segment(cfg, w2, seg, d)
        tables[seg] = coefs
        worst = max(worst, err)
        if verbose:
            print(f'  seg {seg}: err/2^-24 = {err * 2**24:.3f} coefs {coefs}', flush=True)
    return tables, worst


# ---------------------------------------------------------------------------------------------
# Register layout derived from the tables (widths the circuit must use).

def layout(cfg: Config, tables: dict) -> dict:
    d = np.arange(2**DB, dtype=np.int64)
    w2 = max(t[0] for t in tables.values()).bit_length()
    rmax, qmin, qmax, ymax = 0, {}, {}, 0
    for seg, (K2, K1, K0) in tables.items():
        R, Q = pipeline(cfg, w2, K2, K1, d)
        assert R.min() >= 0 and K0 - Q.max() >= 0, seg
        rmax = max(rmax, int(R.max()), K1)
        qmin[seg], qmax[seg] = int(Q.min()), int(Q.max())
        ymax = max(ymax, K0 - qmin[seg])
    lay = dict(w2=w2, w1=rmax.bit_length(), wy=ymax.bit_length(), rows2=cfg.rows2(w2))
    if cfg.LB:
        lo, hi = {}, {}
        for seg, (K2, K1, K0) in tables.items():
            base = K0 % (1 << cfg.S)
            while base < qmax[seg]:
                base += 1 << cfg.S
            assert base - qmin[seg] < (1 << cfg.LB), f'segment {seg}: low register would overflow LB={cfg.LB}'
            lo[seg], hi[seg] = base, (K0 - base) >> cfg.S
            assert hi[seg] >= 0
        lay.update(lo=lo, hi=hi, whi=max(1, max(hi.values()).bit_length()))
    if cfg.RB:
        coarse, resid = {}, {}
        for seg, (K2, K1, K0) in tables.items():
            R, _ = pipeline(cfg, w2, K2, K1, d)
            base = int(R.min()) >> (cfg.RB - 1)
            low = R - (base << (cfg.RB - 1))
            assert (low >= 0).all() and (low < (1 << cfg.RB)).all(), f'segment {seg}: residual width {cfg.RB} too small'
            resid[seg] = K1 - (base << (cfg.RB - 1))
            assert 0 <= resid[seg] < (1 << cfg.RB)
            coarse[seg] = base
        lay.update(coarse=coarse, resid=resid, wco=max(1, (max(coarse.values()) + 1).bit_length()))
        assert lay['wco'] + cfg.RB - 1 >= lay['w1']
    return lay


# ---------------------------------------------------------------------------------------------
# Full-domain check against the verifier's numbers, and cost estimate.

def phase_words(cfg: Config, tables: dict, lay: dict, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dec = decompose(cfg, xs)
    packed = np.zeros((256, 3), dtype=np.int64)
    for seg, c in tables.items():
        packed[seg] = c
    K2, K1, K0 = (packed[dec['seg'], j] for j in range(3))
    d = dec['d']
    sig = [2 * ((d >> i) & 1) - 1 for i in range(DB)]
    R = K1.copy()
    for i in lay['rows2']:
        R = R - sig[i] * rd(K2, cfg.sh2(i))
    Q = np.zeros_like(d)
    for i in reversed(range(DB)):
        Q = Q + sig[i] * rd(R, cfg.sh1(i))
    Y = K0 - Q
    assert (Y >= 0).all() and (R >= 0).all()
    word = ((Y >> cfg.GY).astype(np.uint64) << dec['s'].astype(np.uint64)) & np.uint64(2**cfg.F - 1)
    return word, Y


def full_check(cfg: Config, tables: dict, lay: dict, xs, target_u64, tol_u64) -> dict:
    word, _ = phase_words(cfg, tables, lay, xs)
    theta = word << np.uint64(64 - cfg.F)
    err = (theta - target_u64).view(np.int64)
    used = np.abs(err).view(np.uint64)
    ratio = used / tol_u64
    i = int(ratio.argmax())
    return dict(max_ratio=float(ratio[i]), worst_x=int(xs[i]), frac_over_0_5=float((ratio > 0.5).mean()),
                frac_over_0_8=float((ratio > 0.8).mean()), pass_all=bool((used <= tol_u64).all()),
                min_margin_units=int((tol_u64 - used).view(np.int64).min()))


def cost_estimate(cfg: Config, lay: dict) -> dict:
    """Static Toffoli estimate per direction (adder of n bits = n - 1) and a rough Q."""
    w1 = lay['w1']
    inner_width = cfg.RB if cfg.RB else w1
    rows2 = len(lay['rows2'])
    outer_width = cfg.LB if cfg.LB else lay['wy']
    toff_rows = rows2 * (inner_width - 1) + DB * (outer_width - 1)
    merge = (lay['wco'] - 1) if cfg.RB else 0
    lookup = 254 + 25 + (15 if cfg.LB else 0)
    n_rot = (cfg.F - 3) if not cfg.LB else ((outer_width - cfg.GY + N_P) + (lay['whi'] + N_P) - 3)
    q = WIDTH + (5 if cfg.norm == 'binary' else 10) + lay['w2'] + (cfg.RB + lay['wco'] if cfg.RB else w1) \
        + outer_width + (outer_width - 2)
    return dict(rows_per_dir=toff_rows, merge=merge, lookup=lookup, n_rot=n_rot, Q_rough=q)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--configs', nargs='*', help='norm:F2:F1:FY:GY[:drop2 digits][:LB:S:RB]')
    ap.add_argument('--tables-dir', default=os.path.join(HERE, 'sd_tables'))
    ap.add_argument('--out', default=os.path.join(HERE, 'sd_results.md'))
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    def parse(c: str) -> Config:
        p = c.split(':')
        norm, F2, F1, FY, GY = p[0], int(p[1]), int(p[2]), int(p[3]), int(p[4])
        drop = tuple(int(ch) for ch in p[5]) if len(p) > 5 and p[5] not in ('', '-') else ()
        if len(p) > 8:
            LB, S, RB = (None if p[6] == '-' else int(p[6])), (None if p[7] == '-' else int(p[7])), (None if p[8] == '-' else int(p[8]))
        else:
            LB, S, RB = (None, None, None) if norm == 'pairs' else (21, 20, 12)
        return Config(norm, F2, F1, FY, GY, drop, LB, S, RB)

    default = ['pairs:13:23:29:0', 'pairs:12:22:29:0', 'binary:10:19:28:2:5', 'binary:9:19:28:2:5', 'binary:10:19:28:2',
               'binary:10:18:27:1:5', 'binary:9:18:27:1:5', 'binary:10:19:27:1:5']
    cfgs = [parse(c) for c in (args.configs or default)]
    t0 = time.time()
    xs = np.arange(1, MAX_X + 1, dtype=np.int64)
    spec = verifier.ProblemSpecInvSqrt.for_n_p(N_P)
    target_u64, tol_u64 = verifier.reference_values(spec, xs.astype(np.uint64))
    os.makedirs(args.tables_dir, exist_ok=True)
    lines = ['| config | fit worst (model, all δ) | max err/tol (all x) | worst x | frac>0.5 | frac>0.8 | pass | rows2 | w2/w1/wy | rows Toff/dir | n_rot | Q rough |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    print(f'setup {time.time() - t0:.1f}s', flush=True)
    for cfg in cfgs:
        t1 = time.time()
        tables, fit_worst = fit_all(cfg, args.verbose)
        try:
            lay = layout(cfg, tables)
        except AssertionError as e:
            print(f'{cfg.name()}: layout failed: {e}')
            lines.append(f'| {cfg.name()} | {fit_worst * 2**24:.3f} | layout failed: {e} | | | | | | | | | |')
            continue
        res = full_check(cfg, tables, lay, xs, target_u64, tol_u64)
        cost = cost_estimate(cfg, lay)
        with open(os.path.join(args.tables_dir, cfg.name() + '.json'), 'w') as f:
            json.dump(dict(config=dataclasses.asdict(cfg), tables={str(k): v for k, v in tables.items()},
                           layout={k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in lay.items()},
                           result=res, cost=cost), f)
        lines.append(f'| {cfg.name()} | {fit_worst * 2**24:.3f} | **{res["max_ratio"]:.3f}** | {res["worst_x"]} | {res["frac_over_0_5"]:.4f} | '
                     f'{res["frac_over_0_8"]:.5f} | {"PASS" if res["pass_all"] else "FAIL"} | {len(lay["rows2"])} | '
                     f'{lay["w2"]}/{lay["w1"]}/{lay["wy"]} | {cost["rows_per_dir"]} | {cost["n_rot"]} | {cost["Q_rough"]} |')
        print(f'{cfg.name()}: fit worst {fit_worst * 2**24:.3f}, max err/tol {res["max_ratio"]:.3f} at x={res["worst_x"]} '
              f'(>0.5 {res["frac_over_0_5"]:.4f}, >0.8 {res["frac_over_0_8"]:.5f}, pass={res["pass_all"]}) '
              f'rows2={len(lay["rows2"])} w2/w1/wy={lay["w2"]}/{lay["w1"]}/{lay["wy"]} '
              f'{"whi=" + str(lay["whi"]) if cfg.LB else ""} {"wco=" + str(lay["wco"]) if cfg.RB else ""} '
              f'rows/dir={cost["rows_per_dir"]} n_rot={cost["n_rot"]} Q~{cost["Q_rough"]} [{time.time() - t1:.0f}s]', flush=True)
    with open(args.out, 'w') as f:
        f.write('# Signed-digit design: classical model results (static estimates; not verifier output)\n\n' + '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
