"""Classical fixed-point model for invsqrt candidate #1: QROM-indexed piecewise polynomial.

Replaces the reference's Horner-cubic + Newton step with ONE piecewise polynomial in
`delta`, whose coefficients are looked up (QROM) by the top `k` bits of `x_norm`:

    normalise as the reference: s = leading zero pairs, x_norm = x << 2s, M = x_norm / 2^20 in [1, 4)
    address a = x_norm >> (22 - k)            (3 * 2^(k-2) pieces actually occur)
    delta    = (x_norm mod 2^(22-k)) / 2^20   truncated to its top `d` bits
    p  ~= 2 / sqrt(M)  via Horner with alternating signs and positive integer coefficients:
         deg 2:  r1 = c1 - T(c2 * delta),  p = c0 - T(r1 * delta)
         deg 3:  r1 = c2 - T(c3 * delta),  r2 = c1 - T(r1 * delta),  p = c0 - T(r2 * delta)
    y1 = p << s  (mod 2^F), F = wp + n_p + 1;  phase = y1 / 2^F turns.

`T(.)` has exactly the reference's `_add_row` semantics: one row per delta bit, each row
shifted and floored to the output register's LSB, rows summed, then *subtracted* from the
accumulator (`isub`), so no separate sign-correction stage exists (candidate #7 is built in).
Register widths: `c2`, `c3` have `wc` fractional bits and no integer bit (< 1), `c1`/`r*`
have `wc` fractional + 1 integer bit, `c0`/`p` have `wp` fractional + 1 integer bit.

Coefficients are fitted PER PIECE for minimax *relative* error of the fixed-point pipeline
itself (integer local search seeded by a float Chebyshev interpolant), on the x_norm values
that actually occur. Errors are then checked against the official verifier's own
`reference_values` (target and tol in 2^-64-turn units), i.e. the exact pass/fail test.

Also prints a STATIC cost estimate per direction (Toffolis: lookup 2^k - 2, measured
del_lookup, `_add_row` rows = ANDs + (adder length - 1)) and an estimated Q. These are
estimates, not verifier output; nothing here is a score.

Usage (inside Docker, repo at /work):
    python tasks/invsqrt/model_pwquad.py                 # default sweep, writes tasks/invsqrt/pwquad_results.md + tables
    python tasks/invsqrt/model_pwquad.py --configs 2:8:26:29:14 3:6:28:29:16   # deg:k:wc:wp:d
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'verifiers', 'invsqrt'))
import verify_submission_invsqrt as verifier  # read-only use of the official verifier module

N_P = 10
WIDTH = 2 * N_P + 2  # 22
MAX_X = 3 * (2**N_P - 2)**2
DEL_LOOKUP_COST = {4: 3, 5: 6, 6: 10, 7: 17, 8: 25, 9: 41, 10: 56}  # measured in Docker for ~84-bit outputs
F_OF = lambda wp: wp + N_P + 1


# ---------------------------------------------------------------------------------------------
# Fixed-point pipeline (vectorised over x_norm values of one piece or over all x).

def row_shifts(d: int, k: int, mult_frac: int, out_frac: int) -> np.ndarray:
    """Exponent e_m = out_frac - mult_frac - m for delta bits m = k-1 .. k-2+d (weights 2^-m)."""
    ms = np.arange(k - 1, k - 1 + d)
    return out_frac - mult_frac - ms


def delta_bits(x_norm: np.ndarray, k: int, d: int) -> np.ndarray:
    """Matrix (n, d) of delta bits, column j = bit of weight 2^-(k-1+j), i.e. x_norm bit 21-k-j."""
    cols = [(x_norm >> (WIDTH - 1 - k - j)) & 1 for j in range(d)]
    return np.stack(cols, axis=1).astype(np.int64)


def truncated_product(bits: np.ndarray, mult: np.ndarray | int, shifts: np.ndarray) -> np.ndarray:
    """T(mult * delta): sum over rows of bit_m * floor(mult * 2^e_m), mult a scalar or (n,) vector."""
    mult = np.asarray(mult, dtype=np.int64)
    if mult.ndim == 0:
        rows = np.where(shifts >= 0, mult << np.maximum(shifts, 0), mult >> np.maximum(-shifts, 0))
        return bits @ rows
    pos = np.maximum(shifts, 0)[None, :]
    neg = np.maximum(-shifts, 0)[None, :]
    rows = np.where(shifts[None, :] >= 0, mult[:, None] << pos, mult[:, None] >> neg)
    return (rows * bits).sum(axis=1)


class Config:
    def __init__(self, deg: int, k: int, wc: int, wp: int, d: int):
        assert deg in (2, 3) and 2 <= k <= 11 and 1 <= d <= WIDTH - k
        self.deg, self.k, self.wc, self.wp, self.d = deg, k, wc, wp, d
        self.sh_cc = row_shifts(d, k, wc, wc)  # intermediate steps: wc -> wc
        self.sh_cp = row_shifts(d, k, wc, wp)  # last step: wc -> wp

    def name(self) -> str:
        return f'deg{self.deg}_k{self.k}_wc{self.wc}_wp{self.wp}_d{self.d}'

    def evaluate(self, bits: np.ndarray, coefs: list[int]) -> np.ndarray:
        """p in units of 2^-wp for the given delta-bit matrix and integer coefficients [c0, c1, c2(, c3)]."""
        if self.deg == 2:
            c0, c1, c2 = coefs
            r1 = c1 - truncated_product(bits, c2, self.sh_cc)
            return c0 - truncated_product(bits, r1, self.sh_cp)
        c0, c1, c2, c3 = coefs
        r1 = c2 - truncated_product(bits, c3, self.sh_cc)
        r2 = c1 - truncated_product(bits, r1, self.sh_cc)
        return c0 - truncated_product(bits, r2, self.sh_cp)

    def tail(self, bits: np.ndarray, coefs_wo_c0: list[int]) -> np.ndarray:
        """Everything except the final c0, so c0 can be optimised by a cheap vector add."""
        return self.evaluate(bits, [0, *coefs_wo_c0])

    # Static cost model (see tasks/invsqrt/notes.md section 2 for the measured primitives).
    def cost(self) -> dict:
        def rows(mult_len: int, tgt_len: int, shifts: np.ndarray) -> tuple[int, int]:
            ands = adds = 0
            for e in shifts.tolist():
                first = max(0, -e)
                if first >= mult_len:
                    continue
                ands += mult_len - first
                low = max(e, 0)
                adds += tgt_len - low - 1
            return ands, adds
        wc, wp = self.wc, self.wp
        a1, s1 = rows(wc, wc + 1, self.sh_cc)                 # c2 (wc bits) * delta into c1/r1 (wc+1 bits)
        steps = [(a1, s1)]
        if self.deg == 3:
            steps.append(rows(wc + 1, wc + 1, self.sh_cc))    # r1 * delta into c1/r2
        steps.append(rows(wc + 1, wp + 1, self.sh_cp))        # r * delta into c0/p (wp+1 bits)
        ands, adds = (sum(t[0] for t in steps), sum(t[1] for t in steps))
        lookup = 2**self.k - 2 + DEL_LOOKUP_COST.get(self.k, round(2.5 * 2**(self.k / 2)))
        per_dir = ands + adds + lookup
        F = F_OF(wp)
        n_rot = F - 3
        # Q: x, y1 (F), flags (10), c2/c3 (wc), c1/r1 (wc+1) [+ r2 for deg 3], scratch (max row ANDs),
        # adder carry workspace (max adder length - 1), lookup workspace (k).
        q = (WIDTH + F + N_P + wc + (wc + 1) + (wc + 1) * (self.deg == 3)
             + (wp + 1) + wp + self.k)
        est_c = 2 * per_dir + 230 + 600 + n_rot * 6.8
        est_c_barrel = 2 * per_dir + 230 + 320 + n_rot * 6.8
        return dict(ands=ands, adds=adds, lookup=lookup, per_direction=per_dir, n_rot=n_rot, Q=q,
                    C_est=round(est_c), C_est_barrel=round(est_c_barrel),
                    QC_est=round(q * est_c), QC_est_barrel=round(q * est_c_barrel))


# ---------------------------------------------------------------------------------------------
# Fitting.

def float_seed(cfg: Config, m_lo: float, m_hi: float) -> list[int]:
    """Chebyshev interpolant of 2/sqrt(M) on [m_lo, m_hi] in delta = M - m_lo, scaled to integers."""
    deg = cfg.deg
    j = np.arange(deg + 1)
    nodes = (m_lo + m_hi) / 2 + (m_hi - m_lo) / 2 * np.cos((2 * j + 1) * np.pi / (2 * (deg + 1)))
    a = np.polynomial.polynomial.polyfit(nodes - m_lo, 2 / np.sqrt(nodes), deg)
    # Alternating signs: p = a0 + a1 d + a2 d^2 + a3 d^3 with a1 < 0, a2 > 0, a3 < 0.
    mags = [abs(a[i]) for i in range(deg + 1)]
    c = [round(mags[0] * 2**cfg.wp)] + [round(mags[i] * 2**cfg.wc) for i in range(1, deg + 1)]
    return c


def best_c0(tail: np.ndarray, target: np.ndarray) -> tuple[int, float]:
    """Integer c0 minimising max |(c0 + tail - target) / target| (convex in c0: check around the midpoint)."""
    diff = tail - target  # float64
    mid = -(diff.max() + diff.min()) / 2
    best = (math.inf, 0)
    for c0 in range(int(math.floor(mid)) - 2, int(math.ceil(mid)) + 3):
        err = np.abs((c0 + diff) / target).max()
        best = min(best, (err, c0))
    return best[1], best[0]


def fit_piece(cfg: Config, bits: np.ndarray, target: np.ndarray, seed: list[int]) -> tuple[list[int], float]:
    """Greedy integer local search on the non-c0 coefficients with c0 re-optimised each time."""
    rest = list(seed[1:])
    def score(r):
        t = cfg.tail(bits, r).astype(np.float64)
        return best_c0(t, target)
    c0, best = score(rest)
    steps = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    improved = True
    while improved:
        improved = False
        for i in range(len(rest)):
            for st in steps:
                for sgn in (1, -1):
                    cand = list(rest)
                    cand[i] += sgn * st
                    if cand[i] < 0:
                        continue
                    c0c, e = score(cand)
                    if e < best:
                        rest, c0, best, improved = cand, c0c, e, True
                        break
                if improved:
                    break
            if improved:
                break
    return [c0, *rest], best


def fit_all(cfg: Config, x_norm_all: np.ndarray, verbose: bool = False) -> dict[int, list[int]]:
    """Fits every occurring piece; returns {address: [c0, c1, c2(, c3)]}."""
    k = cfg.k
    addr = x_norm_all >> (WIDTH - k)
    tables: dict[int, list[int]] = {}
    for a in np.unique(addr).tolist():
        xn = x_norm_all[addr == a]
        bits = delta_bits(xn, k, cfg.d)
        target = 2 / np.sqrt(xn / 2.0**(WIDTH - 2)) * 2.0**cfg.wp  # exact p in units of 2^-wp
        m_lo = a * 2.0**(2 - k)
        seed = float_seed(cfg, m_lo, m_lo + 2.0**(2 - k))
        coefs, err = fit_piece(cfg, bits, target, seed)
        tables[a] = coefs
        if verbose:
            print(f'  piece {a}: rel err 2^{math.log2(max(err, 1e-300)):.2f} coefs {coefs}', flush=True)
    return tables


# ---------------------------------------------------------------------------------------------
# Full-domain check against the verifier's numbers.

def full_check(cfg: Config, tables: dict[int, list[int]], xs: np.ndarray, x_norm: np.ndarray, s: np.ndarray,
               target_u64: np.ndarray, tol_u64: np.ndarray) -> dict:
    k, wp = cfg.k, cfg.wp
    F = F_OF(wp)
    addr = x_norm >> (WIDTH - k)
    p = np.zeros(len(xs), dtype=np.int64)
    overflow = False
    for a, coefs in tables.items():
        sel = addr == a
        bits = delta_bits(x_norm[sel], k, cfg.d)
        pv = cfg.evaluate(bits, coefs)
        if (pv < 0).any() or (pv >= 2**(wp + 1)).any():
            overflow = True
        p[sel] = pv
    y1 = (p.astype(np.uint64) << s.astype(np.uint64)) & np.uint64(2**F - 1)
    theta = y1 << np.uint64(64 - F)
    err = (theta - target_u64).view(np.int64)
    used = np.abs(err).view(np.uint64)
    ratio = used / tol_u64
    i = int(ratio.argmax())
    return dict(max_ratio=float(ratio[i]), worst_x=int(xs[i]), frac_over_0_5=float((ratio > 0.5).mean()),
                frac_over_0_8=float((ratio > 0.8).mean()), pass_all=bool((used <= tol_u64).all()),
                overflow=overflow, min_margin_units=int((tol_u64 - used).view(np.int64).min()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--configs', nargs='*', help='deg:k:wc:wp:d entries; default sweep if omitted')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'pwquad_results.md'))
    ap.add_argument('--tables-dir', default=os.path.join(os.path.dirname(__file__), 'pwquad_tables'))
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    default = ['2:8:28:29:14', '2:8:26:29:14', '2:8:24:29:14', '2:8:22:29:13', '2:8:24:30:14', '2:8:22:30:14',
               '2:9:26:29:13', '2:9:24:29:13', '2:9:22:29:12', '2:9:22:30:13',
               '3:6:28:29:16', '3:6:26:29:16', '3:6:24:30:16', '3:7:24:29:15']
    cfgs = [Config(*map(int, c.split(':'))) for c in (args.configs or default)]

    t0 = time.time()
    xs = np.arange(1, MAX_X + 1, dtype=np.int64)
    s = (WIDTH - (np.floor(np.log2(xs)).astype(np.int64) + 1)) // 2
    x_norm = xs << (2 * s)
    x_norm_unique = np.unique(x_norm)
    spec = verifier.ProblemSpecInvSqrt.for_n_p(N_P)
    target_u64, tol_u64 = verifier.reference_values(spec, xs.astype(np.uint64))
    print(f'setup {time.time() - t0:.1f}s: {len(xs)} x, {len(x_norm_unique)} distinct x_norm', flush=True)

    os.makedirs(args.tables_dir, exist_ok=True)
    lines = ['| deg | k | wc | wp | d | max err/tol | worst x | frac>0.5 | frac>0.8 | pass | Toff/dir (ANDs+adds+lookup) | n_rot | Q est | C est (unary / barrel) | Q·C est (barrel) |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for cfg in cfgs:
        t1 = time.time()
        tables = fit_all(cfg, x_norm_unique, verbose=args.verbose)
        res = full_check(cfg, tables, xs, x_norm, s, target_u64, tol_u64)
        cost = cfg.cost()
        with open(os.path.join(args.tables_dir, cfg.name() + '.json'), 'w') as f:
            json.dump({'config': vars(cfg) | {'sh_cc': None, 'sh_cp': None}, 'tables': tables, 'result': res, 'cost': cost}, f,
                      default=lambda o: None)
        row = (f'| {cfg.deg} | {cfg.k} | {cfg.wc} | {cfg.wp} | {cfg.d} | **{res["max_ratio"]:.3f}** | {res["worst_x"]} | '
               f'{res["frac_over_0_5"]:.4f} | {res["frac_over_0_8"]:.5f} | {"PASS" if res["pass_all"] and not res["overflow"] else "FAIL"}'
               f'{" (overflow)" if res["overflow"] else ""} | {cost["per_direction"]} ({cost["ands"]}+{cost["adds"]}+{cost["lookup"]}) | '
               f'{cost["n_rot"]} | {cost["Q"]} | {cost["C_est"]} / {cost["C_est_barrel"]} | {cost["QC_est_barrel"]:,} |')
        lines.append(row)
        print(f'{cfg.name()}: max err/tol {res["max_ratio"]:.3f} at x={res["worst_x"]} '
              f'(>0.5: {res["frac_over_0_5"]:.4f}, >0.8: {res["frac_over_0_8"]:.5f}, pass={res["pass_all"]}, '
              f'overflow={res["overflow"]}) toff/dir={cost["per_direction"]} Q={cost["Q"]} '
              f'C_est={cost["C_est"]}/{cost["C_est_barrel"]} [{time.time() - t1:.0f}s]', flush=True)
    with open(args.out, 'w') as f:
        f.write('# Candidate #1 classical model results (static estimates; not verifier output)\n\n'
                f'Reference baseline for comparison: Q=260, C=7,987.6, Q·C=2,076,774 (verified).\n\n' + '\n'.join(lines) + '\n')
    print(f'wrote {args.out} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
