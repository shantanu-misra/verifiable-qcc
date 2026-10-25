"""Circuit-faithful model of the ordinary chart with one-gap handling (rotation + protected end cell + P-list shift).
Compares against the exact J-fraction model (model_cf) on verifier-sampled shots."""
import sys, collections
sys.path.insert(0, 'tasks/dqi'); sys.path.insert(0, 'data/verifiers/dqi')
import numpy as np
from model_bm import TAB, gf_inv, gf_mul
from model_cf import extract, convert, unit_inverse_direct
from verify_submission_dqi import ProblemSpecDQI, sample_shots

def extract_rotate(s, protect=True, max_gaps=2):
    w = s[:]; n = len(w); parked = 0; levels = []
    for start in range(0, n, 2):
        act = w[start:]; L = len(act); real = L - parked
        g = real >= 2 and act[0] == 0 and act[1] != 0 and len([1 for x in levels if x[0]]) < max_gaps
        if g:
            act = act[1:] + [act[0]]; parked += 1; real -= 1
        c = act[0] or 1
        ci = gf_inv(c)
        unit = [1] + [gf_mul(x, ci) for x in act[1:]]
        inv = unit_inverse_direct(unit)
        if protect:
            inv = inv[:real] + act[real:]
        w[start:] = [act[0]] + inv[1:]
        levels.append((g, act[0], inv[1] if L > 1 else 0))
    return levels

def convert_rotate(levels):
    p, q = [], [1]
    for g, c, a in reversed(levels):
        qq = [0] * max(len(q) + 1, len(p) + 2)
        for j, y in enumerate(q):
            qq[j] ^= y; qq[j + 1] ^= gf_mul(a, y)
        for j, x in enumerate(p):
            qq[j + 2] ^= x
        p, q = [0] * g + [gf_mul(c, x) for x in q], qq
    while p and p[-1] == 0: p.pop()
    while q and q[-1] == 0: q.pop()
    return p, q

def main():
    shots = int(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    spec = ProblemSpecDQI.benchmark_instance(num_shots=shots)
    rng = np.random.default_rng(seed)
    wt = collections.Counter(); res = collections.Counter(); done = 0
    while done < shots:
        batch = min(256, shots - done)
        syn, _, _ = sample_shots(spec, TAB, rng, batch)
        for t in range(batch):
            S = [int(v) for v in syn[t]]
            layers, tail = extract(S)
            p, q = convert(layers)
            while p and p[-1] == 0: p.pop()
            e = len(q) - 1; wt[e] += 1
            gaps = sum(1 for r, _, _ in layers if r)
            if gaps:
                p2, q2 = convert_rotate(extract_rotate(S))
                ok = (p2, q2) == (p, q) and not (tail and any(tail))
                res[(gaps, e, 'ok' if ok else 'BAD')] += 1
        done += batch
    print('weight histogram:', dict(sorted(wt.items())))
    print('gap shots (gaps, e, result):', dict(sorted(res.items())))

if __name__ == "__main__" and not (len(sys.argv) > 3 and sys.argv[3] == "cells"):
    main()


def extract_cells(s, max_gaps=1):
    """Cell-faithful extraction: returns (cells, flags) where cells is the 70-cell state after extraction."""
    w = s[:]; n = len(w); parked = 0; flags = []
    for start in range(0, n, 2):
        L = n - start; real = L - parked
        g = real >= 2 and w[start] == 0 and w[start + 1] != 0 and sum(flags) < max_gaps
        if g:
            w[start:] = w[start + 1:] + [w[start]]; parked += 1; real -= 1
        flags.append(int(g))
        c = w[start] or 1
        ci = gf_inv(c)
        unit = [1] + [gf_mul(x, ci) for x in w[start + 1:start + real]]
        inv = unit_inverse_direct(unit)
        w[start + 1:start + real] = inv[1:]
    return w, flags


def convert_cells(w, flags):
    """In-place cell conversion with the gap rule: same arithmetic, P' list rotated right by one when flagged."""
    w = w[:]; n = len(w)
    P, Q = [], []                                   # lists of cell indices
    for start in reversed(range(0, n, 2)):
        c, a = start, start + 1
        oldP, oldQ = P, Q; k = len(oldP)
        for j in range(2, k + 2):
            if j <= k:
                w[oldP[j - 2]] ^= w[oldQ[j - 1]]
            w[oldP[j - 2]] ^= gf_mul(w[a], w[oldQ[j - 2]])
        if k:
            w[a] ^= w[oldQ[0]]
        cv = w[c] or 1
        for q in oldQ:
            w[q] = gf_mul(w[q], cv)
        P, Q = [c] + oldQ, [a] + oldP
        if flags[start // 2]:                       # P' = x * (c Q): rotate the contents of the P list right by one
            vals = [w[i] for i in P]
            vals = [vals[-1]] + vals[:-1]
            for i, v in zip(P, vals):
                w[i] = v
    p = [w[i] for i in P]; q = [1] + [w[i] for i in Q]
    while p and p[-1] == 0: p.pop()
    while q and q[-1] == 0: q.pop()
    return p, q


def main2():
    shots = int(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    spec = ProblemSpecDQI.benchmark_instance(num_shots=shots)
    rng = np.random.default_rng(seed)
    res = collections.Counter(); done = 0
    while done < shots:
        batch = min(256, shots - done)
        syn, _, _ = sample_shots(spec, TAB, rng, batch)
        for t in range(batch):
            S = [int(v) for v in syn[t]]
            layers, tail = extract(S)
            p, q = convert(layers)
            while p and p[-1] == 0: p.pop()
            e = len(q) - 1
            w, flags = extract_cells(S)
            p2, q2 = convert_cells(w, flags)
            ok = (p2, q2) == (p, q) and not (tail and any(tail))
            gaps = [r for r, _, _ in layers if r]
            res[(len(gaps), max(gaps, default=0), 'e35' if e == 35 else 'e<35', 'ok' if ok else 'BAD')] += 1
        done += batch
    print('cell model (gap count, max gap order, weight, result):', dict(sorted(res.items())))

if __name__ == '__main__' and len(sys.argv) > 3 and sys.argv[3] == 'cells':
    main2()
