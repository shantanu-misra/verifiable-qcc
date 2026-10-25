"""In-place continued-fraction (J-fraction) DQI decoder built with km.CircuitBuilder.

Our own implementation of the algorithm in tasks/dqi/notes.md §9 (chart due to SomeJoint / Agniv Banerjee;
gap and pairing ideas Phuc Thai, ViberTeam, Ahmed Berat Özer). The 70 syndrome registers are the only polynomial
storage: extraction turns them into (c, a) pairs level by level, conversion into (p, q) = (omega, sigma), the sweep
phases, and both maps are reversed.

Usage (inside Docker): python tasks/dqi/build_cf.py out.kmb [--n 70 --degree 12 --m 4095]
"""
from __future__ import annotations

import kickmix as km

from build_bm import BMDecoder          # reuses the sweep, GF products and tower inverse
from tower import apply_gf2_matrix


class CFDecoder(BMDecoder):
    def __init__(self, builder, n, field, m_positions):
        super().__init__(builder, n, field, m_positions)
        self.frob = {}

    # ---- helpers ------------------------------------------------------------------------------------
    def frob_matrix(self, d: int):
        """GF(2)-matrix of x -> x^d (d a power of two) in the polynomial basis."""
        if d not in self.frob:
            gf, deg = self.gf, self.deg
            cols = [gf.pow(1 << i, d) for i in range(deg)]
            self.frob[d] = [[(cols[i] >> r) & 1 for i in range(deg)] for r in range(deg)]
        return self.frob[d]

    def frob_inplace(self, reg, d: int):
        apply_gf2_matrix(self.b, self.frob_matrix(d), reg)

    def unfrob_inplace(self, reg, d: int):
        from tower import gf2_matrix_inverse
        apply_gf2_matrix(self.b, gf2_matrix_inverse(self.frob_matrix(d)), reg)

    def inverse_init(self, x):
        if self.tower is not None:
            return self.tower.init_inverse_with_scaffold(self.b, x)
        return self.b.init_gf2_inverse_with_scaffold(x, field=self.gf)

    def inverse_del(self, x, inv, sc):
        if self.tower is not None:
            self.tower.del_inverse_with_scaffold(self.b, x, target=inv, scaffold=sc); self.b.free(inv)
        else:
            self.b.del_gf2_inverse_with_scaffold(x, target=inv, scaffold=sc, field=self.gf); self.b.free(sc); self.b.free(inv)

    # ---- kernels ------------------------------------------------------------------------------------
    def scale_words(self, pivot, words, *, divide: bool):
        """words[i] <- words[i] * pivot^(-1 if divide else 1), in place: one product + free erase per word.
        A zero pivot is treated as 1 (reversible: the zero flag toggles bit 0) — the ordinary chart's
        approximation for gap inputs."""
        b = self.b
        if not words:
            return
        z = b.alloc_qubits(1)[0]
        self.is_zero(pivot, z)
        b.cx(z, pivot[0])
        inv, sc = self.inverse_init(pivot)
        factor, undo = (inv, pivot) if divide else (pivot, inv)
        tmp = b.alloc_qubits(self.deg)
        for w in words:
            self.xor_prod(w, factor, tmp)            # tmp = w * factor
            b.swap(w, tmp)                            # w <- w * factor, tmp = old w = w_new * undo
            b.del_gf2_mul(w, undo, target=tmp, field=self.gf)
        b.free(tmp)
        self.inverse_del(pivot, inv, sc)
        b.cx(z, pivot[0])
        self.is_zero(pivot, z)
        b.free(z)

    def reciprocal(self, words, *, reverse: bool = False):
        """In-place unit-series inverse of [1] + words (the leading 1 is implicit): a <- a * a^(d), d = 2, 4, 16..."""
        b = self.b
        n = len(words) + 1                            # a[0] = 1 implicit, a[k] = words[k - 1]
        d, exponent, factors = 2, 1, []
        while exponent < n - 1:
            factors.append(d); exponent *= 1 + d; d *= d
        ops = []
        for d in factors:
            for k in range(n - 1, 0, -1):
                for i in range(1, k // d + 1):
                    ops.append((k, i, k - d * i, d))
        tmp = None
        for k, i, j, d in (reversed(ops) if reverse else ops):
            if j == i:                                # a[i]^d * a[i]: needs a copy, the multiplier takes two registers
                if tmp is None:
                    tmp = b.alloc_qubits(self.deg)
                b.cx(words[i - 1], tmp); self.frob_inplace(tmp, d)
                self.xor_prod(tmp, words[i - 1], words[k - 1])
                self.unfrob_inplace(tmp, d); b.cx(words[i - 1], tmp)
                continue
            self.frob_inplace(words[i - 1], d)
            if j == 0:
                b.cx(words[i - 1], words[k - 1])      # times a[0] = 1
            else:
                self.xor_prod(words[i - 1], words[j - 1], words[k - 1])
            self.unfrob_inplace(words[i - 1], d)
        if tmp is not None:
            b.free(tmp)

    def extract(self, words, *, reverse: bool = False):
        levels = range(0, len(words), 2)
        for start in (reversed(levels) if reverse else levels):
            active = words[start:]
            with self.b.mark('cf_level'):
                if reverse:
                    self.reciprocal(active[1:], reverse=True)
                    self.scale_words(active[0], active[1:], divide=False)
                else:
                    self.scale_words(active[0], active[1:], divide=True)
                    self.reciprocal(active[1:])

    def convert(self, words, *, reverse: bool = False):
        """Bottom-up in place: for level (c, a) with current (P, Q): Q' = (1 + a x) Q + x^2 P, P' = c Q.
        Returns (P, Q) cell lists (Q without its implicit leading 1)."""
        b = self.b
        P, Q, stages = [], [], []
        for start in reversed(range(0, len(words), 2)):
            c, a = words[start], words[start + 1]
            stages.append((c, a, list(P), list(Q)))
            P, Q = [c] + Q, [a] + P
        for c, a, oldP, oldQ in (reversed(stages) if reverse else stages):
            k = len(oldP)                             # len(oldQ) == k as well
            with b.mark('cf_convert'):
                if reverse:
                    self.scale_words(c, oldQ, divide=True)
                    if k:
                        b.cx(oldQ[0], a)
                # q'_j for j = 2..k+1 lands in oldP[j-2]: += q_j (j <= k) and a * q_{j-1}
                for j in range(2, k + 2):
                    if j <= k:
                        b.cx(oldQ[j - 1], oldP[j - 2])
                    self.xor_prod(a, oldQ[j - 2], oldP[j - 2])
                if not reverse:
                    if k:
                        b.cx(oldQ[0], a)              # q'_1 = a + q_1
                    self.scale_words(c, oldQ, divide=False)   # p'_j = c q_j
        return P, Q

    # ---- full decoder ---------------------------------------------------------------------------------
    def build(self, S, E):
        b = self.b
        words = list(S)
        with b.mark('cf_extract'):
            self.extract(words)
        with b.mark('cf_to_poly'):
            P, Q = self.convert(words)
        one = b.alloc_qubits(self.deg); b.x(one[0])
        self.A2 = [one] + Q + [None] * (self.slots - 1 - len(Q))
        self.W = P + [None] * (self.t - len(P))
        with b.mark('sweep'):
            self.sweep(E)
        b.x(one[0]); b.free(one)
        with b.mark('cf_from_poly'):
            self.convert(words, reverse=True)
        with b.mark('cf_unextract'):
            self.extract(words, reverse=True)


def build_circuit(n: int, degree: int, m: int) -> km.Circuit:
    gf = km.GF2Field(degree)
    b = km.CircuitBuilder()
    S = [b.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    E = [b.create_classical_register(degree, name=f'E{j}') for j in range(m)]
    CFDecoder(b, n, gf, m).build(S, E)
    return b.finish_circuit()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('output'); ap.add_argument('--n', type=int, default=70); ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--m', type=int, default=4095)
    a = ap.parse_args()
    c = build_circuit(a.n, a.degree, a.m)
    c.to_file(a.output, format='kmb' if a.output.endswith('.kmb') else 'kmx')
    st = c.operation_statistics()
    print(f'wrote {a.output}: toffoli_like={st.num_toffoli_like} num_qubits={c.num_qubits} ops={len(c)}')
