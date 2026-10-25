"""Classical model of an in-place continued-fraction (J-fraction) key-equation solver over GF(2^12).

The syndrome series f(x) = sum_i S_{i+1} x^i (70 words) is expanded level by level:
    f = x^r c / (A(x) + x^{r+2} g(x)),   A = 1 + a_1 x + ... + a_{r+1} x^{r+1},
where r >= 0 is the gap order (r = 0 ordinary: two words (c, a) per level). Each level divides the active words
by the pivot c, inverts the unit series in place, reads off (c, A) and continues on the remaining words, so the 70
words are transformed in place into the fraction data. The convergents give (p, q) = (omega, sigma) with q(0) = 1:
    q' = A q + x^{r+2} p,  p' = x^r c q   (bottom-up over the levels).
Algorithm family: SomeJoint / Agniv Banerjee (public Apache-2.0 notebook "8.95 LB cost breakdown"), paired extraction
and pointer bookkeeping ideas: Phuc Thai, ViberTeam / Artem Nazemtsev, Ahmed Berat Ozer; mask projection: manwithacat.
This file is our own implementation written from the algorithm description.

Usage (inside Docker): python tasks/dqi/model_cf.py [shots=20480] [seed=3]
"""
from __future__ import annotations

import collections
import sys

import numpy as np

sys.path.insert(0, 'data/verifiers/dqi')
sys.path.insert(0, 'tasks/dqi')
from model_bm import ORDER, TAB, gf_inv, gf_mul, LOG, EXP  # noqa: E402
from model_bm import berlekamp_massey  # noqa: E402
from verify_submission_dqi import ProblemSpecDQI, sample_shots  # noqa: E402


def gf_frob(a: int, d: int) -> int:
    """a^d for d a power of two (Frobenius), 0 -> 0."""
    return int(EXP[(LOG[a] * d) % ORDER]) if a else 0


def unit_inverse_direct(u: list[int]) -> list[int]:
    """1/u mod x^n for a unit series u (u[0] = 1), standard recursion."""
    n = len(u)
    b = [0] * n
    b[0] = 1
    for k in range(1, n):
        acc = 0
        for j in range(1, k + 1):
            if u[j]:
                acc ^= gf_mul(u[j], b[k - j])
        b[k] = acc
    return b


def frobenius_chain(a: list[int]) -> int:
    """In-place unit-series inverse by multiplying with Frobenius powers: a <- a * a^(d) for d = 2, 4, 16, ...
    (exponents 1 -> 3 -> 15 -> 255 >= 2^7 - 1, and u^(2^k - 1) = u^-1 mod x^n for 2^k > n). Returns the number of
    quantum-x-quantum products the schedule uses (each XOR of a[i]^d * a[k - d i] into a[k])."""
    n = len(a)
    assert a[0] == 1
    d, exponent, products = 2, 1, 0
    while exponent < n - 1:
        for k in range(n - 1, 0, -1):
            for i in range(1, k // d + 1):
                j = k - d * i
                if j == 0:
                    a[k] ^= gf_frob(a[i], d)          # times a[0] = 1: linear, free in a circuit
                else:
                    a[k] ^= gf_mul(gf_frob(a[i], d), a[j]); products += 1
        exponent *= 1 + d
        d *= d
    return products


def extract(s: list[int], max_gap: int | None = None):
    """Levels (r, c, A) of the J-fraction of the series s (in place semantics), plus the leftover tail."""
    active = s[:]
    layers = []
    while any(active):
        r = next(i for i, x in enumerate(active) if x)
        if (max_gap is not None and r > max_gap) or 2 * r + 2 > len(active):
            return layers, active
        c = active[r]
        ci = gf_inv(c)
        unit = [1] + [gf_mul(x, ci) for x in active[r + 1:]]
        inv = unit_inverse_direct(unit)
        layers.append((r, c, inv[:r + 2]))          # A = 1 + a_1 x + ... + a_{r+1} x^{r+1}
        active = inv[r + 2:]
    return layers, []


def convert(layers):
    """(p, q) with q(0) = 1 from the levels: q' = A q + x^{r+2} p, p' = x^r c q."""
    p, q = [], [1]
    for r, c, A in reversed(layers):
        qq = [0] * (len(q) + r + 1)
        for i, x in enumerate(A):
            for j, y in enumerate(q):
                qq[i + j] ^= gf_mul(x, y)
        for j, x in enumerate(p):
            qq[j + r + 2] ^= x
        p, q = [0] * r + [gf_mul(c, x) for x in q], qq
    return p, q


def main() -> None:
    shots = int(sys.argv[1]) if len(sys.argv) > 1 else 20480
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    spec = ProblemSpecDQI.benchmark_instance(num_shots=shots)
    rng = np.random.default_rng(seed)
    # check the Frobenius chain against the direct inverse on random unit series of every even length
    rr = np.random.default_rng(1)
    for n in range(2, 71, 2):
        u = [1] + [int(v) for v in rr.integers(0, 4096, n - 1)]
        a = u[:]
        frobenius_chain(a)
        assert a == unit_inverse_direct(u), n
    stats = collections.Counter(); gaps_hist = collections.Counter(); bad = 0; done = 0
    while done < shots:
        batch = min(256, shots - done)
        syn, mbuc, ph = sample_shots(spec, TAB, rng, batch)
        for t in range(batch):
            S = [int(v) for v in syn[t]]
            layers, tail = extract(S)
            gaps = [r for r, _, _ in layers if r]
            gaps_hist[(len(gaps), max(gaps, default=0))] += 1
            if tail and any(tail):
                stats['nonzero_tail'] += 1; bad += 1; continue
            p, q = convert(layers)
            C, st = berlekamp_massey(S)
            L = st['L']
            ok = q == C[:L + 1] + [0] * max(0, len(q) - L - 1) if len(q) >= L + 1 else False
            # omega = sigma * f mod x^L
            omega = [0] * L
            for i in range(L):
                acc = 0
                for k in range(i + 1):
                    acc ^= gf_mul(C[k], S[i - k]) if k < len(C) else 0
                omega[i] = acc
            pp = p + [0] * max(0, L - len(p))
            ok = ok and pp[:L] == omega and all(v == 0 for v in pp[L:])
            stats['ok' if ok else 'mismatch'] += 1; bad += not ok
            stats['levels_' + str(len(layers))] += 1
        done += batch
    print(f'shots={shots} mismatches={bad}  outcomes={dict((k, v) for k, v in stats.items() if not k.startswith("levels_"))}')
    print('gaps (count, max order) histogram:', dict(sorted(gaps_hist.items())))
    # one-way product counts of the ordinary chart (no gaps): reciprocal chain, scaling, conversion
    rec = sum(frobenius_chain([1] + [0] * (n - 1)) for n in range(2, 71, 2))
    print(f'one-way products: reciprocal chain (naive) {rec}, scaling {sum(2 * (n - 1) for n in range(2, 71, 2)) // 2}, '
          f'conversion {sum(3 * k for k in range(35))}')


if __name__ == '__main__':
    main()
