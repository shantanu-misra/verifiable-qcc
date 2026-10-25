"""For sampled shots: at each BM step n (after its update), the smallest syndrome index any later step reads
(min over n' > n of n' - L_{n'}), compared with the formula min(n + 1 - L_n, L_n - 1), and how many syndromes are
provably dead for *every* shot at each n (the classical lower envelope)."""
import sys, collections; sys.path.insert(0,'data/verifiers/dqi'); sys.path.insert(0,'tasks/dqi')
import numpy as np
from model_bm import gf_mul, gf_inv, TAB
from verify_submission_dqi import ProblemSpecDQI, sample_shots
N = 4096
spec = ProblemSpecDQI.benchmark_instance(num_shots=N); rng = np.random.default_rng(21)
syn, mbuc, ph = sample_shots(spec, TAB, rng, N)
formula_ok = 0; total = 0
env = [10**9] * 70          # over shots: min of (min future read index) at step n
interior_zero = 0; interior_total = 0
for s in range(N):
    S = [int(v) for v in syn[s]]; C=[1]; B=[1]; L=0; m=1; b=1; Ls = []; Cs = []
    for n in range(70):
        d = S[n]
        for i in range(1, L+1): d ^= gf_mul(C[i], S[n-i])
        if d:
            coef = gf_mul(d, gf_inv(b)); T = C[:]; sh = [0]*m + B; C = C + [0]*max(0, len(sh)-len(C))
            for i, v in enumerate(sh): C[i] ^= gf_mul(coef, v)
            if 2*L <= n: L = n+1-L; B, b, m = T, d, 1
            else: m += 1
        else: m += 1
        Ls.append(L); Cs.append(C[:L+1])
    reads = [n - Ls[n-1] if n else 0 for n in range(70)]   # step n reads S[n - L_{n-1} .. n]
    for n in range(69):
        fut = min(reads[n+1:]); total += 1
        formula_ok += (fut == min(n + 1 - Ls[n], Ls[n] - 1)) or (fut >= min(n + 1 - Ls[n], Ls[n] - 1))
        env[n] = min(env[n], fut)
    # interior coefficient zero rate (C_c for 1 <= c <= L-1) at step 40
    C40 = Cs[40]; interior_total += max(0, len(C40) - 2); interior_zero += sum(1 for v in C40[1:-1] if v == 0)
print('future-read index >= min(n+1-L, L-1) in', formula_ok, 'of', total, 'step checks')
print('classical dead envelope (number of syndromes dead for every shot) at n = 20,30,...,69:', [env[n] for n in (20, 30, 40, 50, 60, 69)])
print('interior coefficient zero rate:', interior_zero / interior_total)
