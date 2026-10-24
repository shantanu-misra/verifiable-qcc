"""Classical model of a Berlekamp-Massey + Chien/Forney MBUC decoder over GF(2^12).

Checks, on shots drawn with the verifier's own `sample_shots`, that BM recovers the error word and
the target phase parity sum_j popcount(e_j & M_j) mod 2. Also records per-shot statistics that the
circuit design depends on (final L, number of length changes, zero discrepancies, leading-coefficient
degree drops). Pure Python/numpy; nothing here is a score.

Usage (inside Docker): python tasks/dqi/model_bm.py [num_shots=10240] [seed=1]
"""
from __future__ import annotations

import collections
import sys

import numpy as np

sys.path.insert(0, 'data/verifiers/dqi')
import kickmix as km  # noqa: E402
from verify_submission_dqi import FieldTables, ProblemSpecDQI, sample_shots  # noqa: E402

FIELD = km.GF2Field(12)
TAB = FieldTables(FIELD)
ORDER = TAB.order  # 4095
LOG, EXP = TAB.log, TAB.antilog


def gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return int(EXP[(LOG[a] + LOG[b]) % ORDER])


def gf_inv(a: int) -> int:
    assert a != 0
    return int(EXP[(-LOG[a]) % ORDER])


def gf_pow_alpha(e: int) -> int:
    return int(EXP[e % ORDER])


def berlekamp_massey(S: list[int]) -> tuple[list[int], dict]:
    """Returns the connection polynomial C (C[0] = 1, len L+1) for S_1..S_N and trace statistics."""
    N = len(S)
    C, B = [1], [1]
    L, m, b = 0, 1, 1
    stats = collections.Counter()
    for n in range(N):
        d = S[n]
        for i in range(1, L + 1):
            d ^= gf_mul(C[i], S[n - i])
        if d == 0:
            m += 1
            stats['d_zero'] += 1
            continue
        coef = gf_mul(d, gf_inv(b))
        T = C[:]
        shifted = [0] * m + B
        new_len = max(len(C), len(shifted))
        C = C + [0] * (new_len - len(C))
        for i, v in enumerate(shifted):
            C[i] ^= gf_mul(coef, v)
        if 2 * L <= n:
            L = n + 1 - L
            B, b, m = T, d, 1
            stats['length_change'] += 1
        else:
            m += 1
            stats['no_length_change'] += 1
    C = C[:L + 1] + [0] * max(0, L + 1 - len(C))
    stats['L'] = L
    stats['deg_C'] = max([i for i, v in enumerate(C) if v] or [0])
    return C, dict(stats)


def chien_forney(C: list[int], S: list[int], m: int) -> dict[int, int]:
    """Error word {position: value} from sigma = C via Chien search + Forney (syndromes S_1.. with X_j = alpha^j)."""
    L = len(C) - 1
    omega = [0] * L  # omega = sigma * S(x) mod x^L, S(x) = sum S_{i+1} x^i
    for i in range(L):
        acc = 0
        for k in range(i + 1):
            acc ^= gf_mul(C[k], S[i - k])
        omega[i] = acc
    errors = {}
    for j in range(m):
        xinv = gf_pow_alpha(-j)  # X_j^{-1} = alpha^{-j}
        sig = 0
        for i in range(L, -1, -1):
            sig = gf_mul(sig, xinv) ^ C[i]
        if sig != 0:
            continue
        # sigma'(x) = sum over odd i of C[i] x^{i-1}
        dsig = 0
        for i in range(L, 0, -1):
            dsig = gf_mul(dsig, xinv) ^ (C[i] if i % 2 == 1 else 0)
        # wait: Horner above multiplies dsig by xinv for every i, i.e. computes sum C_i x^{i-1} over odd i
        om = 0
        for i in range(L - 1, -1, -1):
            om = gf_mul(om, xinv) ^ omega[i]
        errors[j] = gf_mul(om, gf_inv(dsig))
    return errors


def main() -> None:
    num_shots = int(sys.argv[1]) if len(sys.argv) > 1 else 10240
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    spec = ProblemSpecDQI.benchmark_instance(num_shots=num_shots)
    rng = np.random.default_rng(seed)
    agg = collections.Counter()
    bad = 0
    hist_L = collections.Counter()
    hist_lc = collections.Counter()
    done = 0
    while done < num_shots:
        batch = min(256, num_shots - done)
        syn, mbuc, phases = sample_shots(spec, TAB, rng, batch)
        for s in range(batch):
            S = [int(v) for v in syn[s]]
            C, st = berlekamp_massey(S)
            errors = chien_forney(C, S, spec.m)
            # Re-encode the recovered error word and compare syndromes; compare phase parity.
            re = [0] * spec.n
            for j, e in errors.items():
                for i in range(spec.n):
                    re[i] ^= gf_mul(e, gf_pow_alpha((i + 1) * j))
            parity = sum(int(e & int(mbuc[s, j])).bit_count() for j, e in errors.items()) & 1
            ok = (re == S) and (parity == int(phases[s])) and (len(errors) == st['L'])
            bad += not ok
            hist_L[st['L']] += 1
            hist_lc[st.get('length_change', 0)] += 1
            agg['d_zero'] += st.get('d_zero', 0)
            agg['deg_drop'] += st['deg_C'] != st['L']
        done += batch
    print(f'shots={num_shots} mismatches={bad}  mean zero-discrepancies/shot={agg["d_zero"] / num_shots:.3f}  '
          f'shots with deg C < L: {agg["deg_drop"]}')
    print('L histogram:', dict(sorted(hist_L.items())))
    print('length-change count histogram:', dict(sorted(hist_lc.items())))


if __name__ == '__main__':
    main()
