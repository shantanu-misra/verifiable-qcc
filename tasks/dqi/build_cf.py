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
    def __init__(self, builder, n, field, m_positions, gaps=True):
        super().__init__(builder, n, field, m_positions)
        self.frob = {}
        self.gaps = gaps          # handle one first-order gap exactly (pointer + rotation), see notes §9c
        self.ptr = None           # pointer register: 1-based index of the gap level, 0 = none

    # ---- gap handling ------------------------------------------------------------------------------
    def eq_const(self, reg, const, target):
        """target ^= (reg == const)."""
        for i in range(len(reg)):
            if (const >> i) & 1:
                self.b.x(reg[i])
        self.is_zero(reg, target)
        for i in range(len(reg)):
            if (const >> i) & 1:
                self.b.x(reg[i])

    def rotate(self, cells, ctrl, *, left):
        """Cyclic rotation of the cell contents by one position, controlled on ctrl (left: cells[0] -> end)."""
        pairs = [(i - 1, i) for i in range(1, len(cells))]
        for i, j in (pairs if left else reversed(pairs)):
            self.b.cswap(ctrl, cells[i], cells[j])

    def gated_xor(self, flag, src, dst):
        for s, d in zip(src, dst):
            self.b.ccx(flag, s, d)

    def flag_acquire(self):
        """F = [no gap so far] = 'the last window cell is a real coefficient'."""
        F = self.b.alloc_qubits(1)[0]; self.is_zero(self.ptr, F); return F

    def flag_release(self, F):
        self.is_zero(self.ptr, F); self.b.free(F)

    def _gap_flag(self, active, R1, g):
        """g ^= [active[0] == 0 and active[1] != 0 and R1]."""
        b = self.b
        z0, z1, t = b.alloc_qubits(3)
        self.is_zero(active[0], z0); self.is_zero(active[1], z1)
        b.x(z1); b.ccx(z0, z1, t); b.ccx(t, R1, g); b.ccx(z0, z1, t); b.x(z1)
        self.is_zero(active[1], z1); self.is_zero(active[0], z0)
        b.free(z0); b.free(z1); b.free(t)

    def gap_enter(self, active, k):
        """Level k: if the pivot is zero (and no gap yet) rotate the zero to the end and record k+1 in the pointer."""
        b = self.b
        R1, g = b.alloc_qubits(2)
        self.is_zero(self.ptr, R1)
        self._gap_flag(active, R1, g)
        for i in range(len(self.ptr)):
            if ((k + 1) >> i) & 1:
                b.cx(g, self.ptr[i])
        self.is_zero(self.ptr, R1); b.cx(g, R1)          # R1 -> 0
        self.rotate(active, g, left=True)
        self.eq_const(self.ptr, k + 1, g)                 # g -> 0
        b.free(R1); b.free(g)

    def gap_leave(self, active, k):
        """Inverse of gap_enter."""
        b = self.b
        R1, g = b.alloc_qubits(2)
        self.eq_const(self.ptr, k + 1, g)
        self.rotate(active, g, left=False)
        b.cx(g, R1); self.is_zero(self.ptr, R1)          # R1 = old R1
        for i in range(len(self.ptr)):
            if ((k + 1) >> i) & 1:
                b.cx(g, self.ptr[i])
        self._gap_flag(active, R1, g)                     # g -> 0
        self.is_zero(self.ptr, R1)                        # R1 -> 0
        b.free(R1); b.free(g)

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

    def reciprocal(self, words, *, reverse: bool = False, gate=None):
        """In-place unit-series inverse of [1] + words (the leading 1 is implicit): a <- a * a^(d), d = 2, 4, 16...
        With `gate`, writes to the last cell happen only if gate is set (the cell may be a parked zero)."""
        b = self.b
        gtmp = [None]

        def write(x, y, k):                            # a[k] ^= x * y (y None: times 1)
            target = words[k - 1]
            if gate is not None and k == len(words):
                if y is None:
                    self.gated_xor(gate, x, target)
                else:
                    if gtmp[0] is None:
                        gtmp[0] = b.alloc_qubits(self.deg)
                    self.xor_prod(x, y, gtmp[0])
                    self.gated_xor(gate, gtmp[0], target)
                    b.del_gf2_mul(x, y, target=gtmp[0], field=self.gf)
            elif y is None:
                b.cx(x, target)
            else:
                self.xor_prod(x, y, target)
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
                write(tmp, words[i - 1], k)
                self.unfrob_inplace(tmp, d); b.cx(words[i - 1], tmp)
                continue
            self.frob_inplace(words[i - 1], d)
            write(words[i - 1], None if j == 0 else words[j - 1], k)
            self.unfrob_inplace(words[i - 1], d)
        if tmp is not None:
            b.free(tmp)
        if gtmp[0] is not None:
            b.free(gtmp[0])

    def extract(self, words, *, reverse: bool = False):
        levels = range(0, len(words), 2)
        for start in (reversed(levels) if reverse else levels):
            active = words[start:]
            k = start // 2
            with self.b.mark('cf_level'):
                if reverse:
                    F = self.flag_acquire() if self.gaps else None
                    self.reciprocal(active[1:], reverse=True, gate=F)
                    self.scale_words(active[0], active[1:], divide=False)
                    if self.gaps:
                        self.flag_release(F); self.gap_leave(active, k)
                else:
                    if self.gaps:
                        self.gap_enter(active, k)
                    F = self.flag_acquire() if self.gaps else None
                    self.scale_words(active[0], active[1:], divide=True)
                    self.reciprocal(active[1:], gate=F)
                    if self.gaps:
                        self.flag_release(F)

    def convert(self, words, *, reverse: bool = False):
        """Bottom-up in place: for level (c, a) with current (P, Q): Q' = (1 + a x) Q + x^2 P, P' = c Q.
        Returns (P, Q) cell lists (Q without its implicit leading 1)."""
        b = self.b
        P, Q, stages = [], [], []
        for start in reversed(range(0, len(words), 2)):
            c, a = words[start], words[start + 1]
            stages.append((start // 2, c, a, list(P), list(Q)))
            P, Q = [c] + Q, [a] + P
        for lvl, c, a, oldP, oldQ in (reversed(stages) if reverse else stages):
            k = len(oldP)                             # len(oldQ) == k as well
            with b.mark('cf_convert'):
                if reverse:
                    if self.gaps and k:
                        self.shift_p([c] + oldQ, lvl, left=True)
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
                    if self.gaps and k:
                        self.shift_p([c] + oldQ, lvl, left=False)   # gap level: P' = x c Q
        return P, Q

    def shift_p(self, cells, lvl, *, left):
        """Rotate the P cell list by one, controlled on [this level was the gap level]."""
        g = self.b.alloc_qubits(1)[0]
        self.eq_const(self.ptr, lvl + 1, g)
        self.rotate(cells, g, left=left)
        self.eq_const(self.ptr, lvl + 1, g)
        self.b.free(g)

    # ---- full decoder ---------------------------------------------------------------------------------
    def build(self, S, E):
        b = self.b
        words = list(S)
        if self.gaps:
            self.ptr = b.alloc_qubits(max(3, (len(S) // 2).bit_length()))
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
        if self.gaps:
            b.free(self.ptr); self.ptr = None


def build_circuit(n: int, degree: int, m: int, gaps: bool = True) -> km.Circuit:
    gf = km.GF2Field(degree)
    b = km.CircuitBuilder()
    S = [b.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    E = [b.create_classical_register(degree, name=f'E{j}') for j in range(m)]
    CFDecoder(b, n, gf, m, gaps=gaps).build(S, E)
    return b.finish_circuit()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('output'); ap.add_argument('--n', type=int, default=70); ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--m', type=int, default=4095); ap.add_argument('--no-gaps', action='store_true')
    a = ap.parse_args()
    c = build_circuit(a.n, a.degree, a.m, gaps=not a.no_gaps)
    c.to_file(a.output, format='kmb' if a.output.endswith('.kmb') else 'kmx')
    st = c.operation_statistics()
    print(f'wrote {a.output}: toffoli_like={st.num_toffoli_like} num_qubits={c.num_qubits} ops={len(c)}')
