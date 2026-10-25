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
from tmvp import tmvp_algorithm


class CFDecoder(BMDecoder):
    def __init__(self, builder, n, field, m_positions, gaps=True, fast=True):
        super().__init__(builder, n, field, m_positions)
        self.frob = {}
        self.fast = fast          # rank-reduced shears (notes §10) instead of the naive chain
        self._cm, self._alg = {}, {}
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

    # ---- rank-reduced shears (notes §10) ---------------------------------------------------------------
    def cmul_xor(self, dst, src, lam):
        """dst ^= lam * src (GF(2)-linear in src: CX only)."""
        if lam == 0:
            return
        if lam == 1:
            self.b.cx(src, dst); return
        M = self._cm.get(lam)
        if M is None:
            cols = [self.gf.mul(lam, 1 << c) for c in range(self.deg)]
            M = self._cm[lam] = [[c for c in range(self.deg) if (cols[c] >> r) & 1] for r in range(self.deg)]
        for r in range(self.deg):
            for c in M[r]:
                self.b.cx(src[c], dst[r])

    def root(self, lam, d):
        """lam^(1/d) for d a power of two (Frobenius is a bijection)."""
        s = d.bit_length() - 1
        return self.gf.pow(lam, 1 << (self.deg - s)) if lam > 1 else lam

    def gated_write(self, flag, src, dst, lam=1):
        """dst ^= flag * lam * src with one scratch qubit (12 CCX)."""
        b = self.b
        t = b.alloc_qubits(1)[0]
        if lam == 1:
            for r in range(self.deg):
                b.ccx(flag, src[r], dst[r])
        else:
            cols = [self.gf.mul(lam, 1 << c) for c in range(self.deg)]
            for r in range(self.deg):
                cs = [c for c in range(self.deg) if (cols[c] >> r) & 1]
                for c in cs: b.cx(src[c], t)
                b.ccx(flag, t, dst[r])
                for c in cs: b.cx(src[c], t)
        b.free(t)

    def tmvp_block(self, rows, cols, coef, d, gate, sc):
        """rows/cols: [(cell, position)] ascending; rows[k] ^= sum_j Frob_d(coef(m_k - m'_j)) * cols[j]; coef(i) -> cell a_i.
        gate: flag qubit gating writes to the protected cell self._protected (or None). sc = (X, Y, Z) scratch cells."""
        b = self.b
        p, q = len(rows), len(cols)
        if p == 0 or q == 0:
            return
        X, Y, Z = sc
        def out(cell, src, lam):
            if gate is not None and cell is self._protected:
                self.gated_write(gate, src, cell, lam)
            else:
                self.cmul_xor(cell, src, lam)
        if p == 1 or q == 1:                           # naive: p*q products = rank, no constants
            for rc, m in rows:
                for cc, mm in cols:
                    a = coef(m - mm)
                    b.cx(a, X); self.frob_inplace(X, d)
                    if gate is not None and rc is self._protected:
                        self.xor_prod(X, cc, Z); self.gated_write(gate, Z, rc); b.del_gf2_mul(X, cc, target=Z, field=self.gf)
                    else:
                        self.xor_prod(X, cc, rc)
                    self.unfrob_inplace(X, d); b.cx(a, X)
            return
        key = (p, q)
        if key not in self._alg:
            self._alg[key] = tmvp_algorithm(p, q, self.gf)
        alpha, beta, gamma = self._alg[key]
        i_min = rows[0][1] - cols[-1][1]
        N = p + q - 1
        for m in range(N):
            for n in range(N):
                self.cmul_xor(X, coef(i_min + n), self.root(alpha[m][n], d))
            self.frob_inplace(X, d)
            for j in range(q):
                self.cmul_xor(Y, cols[j][0], beta[m][j])
            self.xor_prod(X, Y, Z)
            for k in range(p):
                out(rows[k][0], Z, gamma[m][k])
            b.del_gf2_mul(X, Y, target=Z, field=self.gf)
            for j in range(q):
                self.cmul_xor(Y, cols[j][0], beta[m][j])
            self.unfrob_inplace(X, d)
            for n in range(N):
                self.cmul_xor(X, coef(i_min + n), self.root(alpha[m][n], d))

    def tri_block(self, items, coef, d, gate, sc, reverse):
        """In-place unit-triangular Toeplitz update of a class vector: items[k] ^= sum_{j<k} Frob_d(coef(m_k - m_j)) items[j]."""
        n = len(items)
        if n <= 1:
            return
        lo, hi = items[:n // 2], items[n // 2:]
        if not reverse:                               # hi <- Tri(hi) on the old values first, then hi += T lo, then lo
            self.tri_block(hi, coef, d, gate, sc, False)
            self.tmvp_block(hi, lo, coef, d, gate, sc)
            self.tri_block(lo, coef, d, gate, sc, False)
        else:
            self.tri_block(lo, coef, d, gate, sc, True)
            self.tmvp_block(hi, lo, coef, d, gate, sc)
            self.tri_block(hi, coef, d, gate, sc, True)

    def pass_fast(self, words, d, gate, sc, reverse):
        """a <- a * (1 + sum_i a_i^d x^{di}) mod x^L in place (a_0 = 1 implicit), hi/lo recursion over residue classes."""
        b = self.b
        X = sc[0]
        coef = lambda i: words[i - 1]
        def cell(k): return words[k - 1]
        def rec(L):
            if L <= 2:
                return
            H = (L + 1) // 2                          # all coefficients a_i, i <= (L-1)/d, lie below H
            if reverse:
                rec(H)
            for r in range(d):
                lo = [(cell(k), k // d) for k in range(1, H) if k % d == r]
                hi = [(cell(k), k // d) for k in range(H, L) if k % d == r]
                if not hi:
                    continue
                def a0_column():                      # column a_0 = 1 (class 0): hi[m] ^= Frob_d(a_m), free
                    if r != 0:
                        return
                    for rc, m in hi:
                        a = coef(m); b.cx(a, X); self.frob_inplace(X, d)
                        if gate is not None and rc is self._protected:
                            self.gated_write(gate, X, rc)
                        else:
                            b.cx(X, rc)
                        self.unfrob_inplace(X, d); b.cx(a, X)
                if not reverse:
                    self.tri_block(hi, coef, d, gate, sc, False)
                    a0_column(); self.tmvp_block(hi, lo, coef, d, gate, sc)
                else:
                    self.tmvp_block(hi, lo, coef, d, gate, sc); a0_column()
                    self.tri_block(hi, coef, d, gate, sc, True)
            if not reverse:
                rec(H)
        rec(len(words) + 1)

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
        if self.fast:
            sc = [b.alloc_qubits(self.deg) for _ in range(3)]
            self._protected = words[-1] if words else None
            for d in (reversed(factors) if reverse else factors):
                self.pass_fast(words, d, gate, sc, reverse)
            for r in sc:
                b.free(r)
            return
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
        self.A2 = [None] + Q + [None] * (self.slots - 1 - len(Q))    # q_0 = 1 implicit (sweep tests acc == 1)
        self.W = P + [None] * (self.t - len(P))
        with b.mark('sweep'):
            self.sweep(E)
        with b.mark('cf_from_poly'):
            self.convert(words, reverse=True)
        with b.mark('cf_unextract'):
            self.extract(words, reverse=True)
        if self.gaps:
            b.free(self.ptr); self.ptr = None


def build_circuit(n: int, degree: int, m: int, gaps: bool = True, fast: bool = True) -> km.Circuit:
    gf = km.GF2Field(degree)
    b = km.CircuitBuilder()
    S = [b.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    E = [b.create_classical_register(degree, name=f'E{j}') for j in range(m)]
    CFDecoder(b, n, gf, m, gaps=gaps, fast=fast).build(S, E)
    return b.finish_circuit()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('output'); ap.add_argument('--n', type=int, default=70); ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--m', type=int, default=4095); ap.add_argument('--no-gaps', action='store_true'); ap.add_argument('--naive', action='store_true')
    a = ap.parse_args()
    c = build_circuit(a.n, a.degree, a.m, gaps=not a.no_gaps, fast=not a.naive)
    c.to_file(a.output, format='kmb' if a.output.endswith('.kmb') else 'kmx')
    st = c.operation_statistics()
    print(f'wrote {a.output}: toffoli_like={st.num_toffoli_like} num_qubits={c.num_qubits} ops={len(c)}')
