"""Berlekamp-Massey + Chien/Forney MBUC decoder for Benchmark II, built with km.CircuitBuilder.

Design (tasks/dqi/notes.md §4): the connection polynomial C lives in `A2` (t+1 slots, scaled so its constant term is a
quantum scalar), the auxiliary polynomial x^m B in `A1` (t+1 slots, rotated up one slot per step), discrepancies are
computed into `d`, the quotient scalar nu = d'/b' is erased after the update by the power-series recursion
nu_j = [A2 / B]_j (j <= 2), and syndromes stay live. Phase 2 (sweep) and the omega swap follow in later pieces.

Usage (inside Docker): python tasks/dqi/build_bm.py out.kmb [--n 70 --degree 12 --m 4095]
"""
from __future__ import annotations

import kickmix as km

from gfmul import GFMul
from tower import Tower12


class BMDecoder:
    def __init__(self, builder: km.CircuitBuilder, n: int, field: km.GF2Field, m_positions: int):
        self.b = builder
        self.n = n
        self.t = n // 2
        self.gf = field
        self.deg = field.degree
        self.m = m_positions
        self.slots = self.t + 1  # C has degree <= t
        self.tower = Tower12() if self.deg == 12 else None
        self.mul = GFMul(field) if self.deg in (3, 6, 12) else None

    # ---- helpers -------------------------------------------------------------------------------
    def disc_bound(self, n: int) -> int:
        """Highest C slot index that can be nonzero at step n (L_n <= n//2 + 1 observed, +1 margin)."""
        return min(self.t, n, n // 2 + 2)

    def upd_bound(self, n: int) -> int:
        """Highest A1 slot that can be nonzero at step n when d != 0 (top = n + 1 - L_old <= n//2 + 3)."""
        return min(self.t, n + 1, n // 2 + 3)

    def swap_bound(self, n: int) -> int:
        """Highest slot that can be nonzero in A1 or A2 at a length change at step n."""
        return max(self.disc_bound(n), self.upd_bound(n))

    def inv_init(self, x):
        """(inv, scaffold) with inv = x^-1 (0 -> 0); tower-field version for GF(2^12), kickmix otherwise."""
        if self.tower is not None:
            return self.tower.init_inverse_with_scaffold(self.b, x)
        return self.b.init_gf2_inverse_with_scaffold(x, field=self.gf)

    def inv_del(self, x, inv, scaffold):
        if self.tower is not None:
            self.tower.del_inverse_with_scaffold(self.b, x, target=inv, scaffold=scaffold)
            self.b.free(inv)
        else:
            self.b.del_gf2_inverse_with_scaffold(x, target=inv, scaffold=scaffold, field=self.gf)
            self.b.free(scaffold); self.b.free(inv)

    def xor_prod(self, x, y, target, control=True):
        if self.mul is not None and control is True:
            self.mul.ixor_mul(self.b, x, y, target)
        else:
            self.b.ixor_gf2_mul(x, y, target=target, field=self.gf, control=control)

    def is_zero_open(self, reg, target):
        """target ^= (reg == 0) with len(reg) - 1 Toffolis; returns the live AND ladder for `is_zero_close`."""
        b = self.b
        n = len(reg)
        b.x(reg)
        anc = b.alloc_qubits(n - 2)
        b.init_and(reg[0], reg[1], anc[0])
        for i in range(2, n - 1):
            b.init_and(anc[i - 2], reg[i], anc[i - 1])
        b.ccx(anc[n - 3], reg[n - 1], target)
        b.x(reg)
        return anc

    def is_zero_close(self, reg, target, anc):
        """Uncompute target (1 Toffoli) and release the ladder (0 Toffolis)."""
        b = self.b
        n = len(reg)
        b.x(reg)
        b.ccx(anc[n - 3], reg[n - 1], target)
        for i in range(n - 2, 1, -1):
            b.del_and(anc[i - 2], reg[i], anc[i - 1])
        b.del_and(reg[0], reg[1], anc[0])
        b.free(anc)
        b.x(reg)

    def is_zero(self, reg, target):
        """target ^= (reg == 0) with len(reg) - 1 Toffolis (AND ladder on the inverted bits, uncomputed by measurement)."""
        b = self.b
        n = len(reg)
        if n < 3:
            b.flip_if_equal(reg, 0, target=target); return
        b.x(reg)
        anc = b.alloc_qubits(n - 2)
        b.init_and(reg[0], reg[1], anc[0])
        for i in range(2, n - 1):
            b.init_and(anc[i - 2], reg[i], anc[i - 1])
        b.ccx(anc[n - 3], reg[n - 1], target)
        for i in range(n - 2, 1, -1):
            b.del_and(anc[i - 2], reg[i], anc[i - 1])
        b.del_and(reg[0], reg[1], anc[0])
        b.free(anc)
        b.x(reg)

    def ctrl_copy(self, flag, src, dst):
        """dst ^= flag & src (12 Toffolis), MBUC-erasable via `unctrl_copy`."""
        self.b.ccx(flag, src, dst)

    def mbuc_erase_ctrl_copies(self, dst, terms):
        """Erase dst = XOR_k flag_k & src_k by X-measurement + CZ fixups (0 Toffolis)."""
        bits = self.b.alloc_bits(len(dst))
        self.b.hmr(dst, bits)
        for flag, src in terms:
            for q, bit in zip(src, bits):
                self.b.push_condition(bit)
                self.b.cz(flag, q)
                self.b.pop_condition()
        self.b.free(bits)

    def mbuc_erase_sum_of_products(self, dst, pairs):
        """Erase dst = XOR_k x_k * y_k (GF products) by X-measurement + bilinear phase fixups (0 Toffolis)."""
        bits = self.b.alloc_bits(len(dst))
        self.b.hmr(dst, bits)
        for x, y in pairs:
            if y is None:   # a plain copied register term: phase (-1)^<m, x>
                for q, bit in zip(x, bits):
                    self.b.push_condition(bit); self.b.z(q); self.b.pop_condition()
            else:
                self.b.gf2_phase_by_product(x, y, mask=bits, field=self.gf)
        self.b.free(bits)

    # ---- registers -----------------------------------------------------------------------------
    def alloc_state(self):
        b = self.b
        self.A2 = [b.alloc_qubits(self.deg) for _ in range(self.slots)]  # scaled C
        self.A1 = [b.alloc_qubits(self.deg) for _ in range(self.slots)]  # x^m B, rotated
        self.breg = b.alloc_qubits(self.deg)                               # b (previous discrepancy), init 1
        self.Lreg = b.alloc_qubits(max(1, self.t.bit_length() + 1))        # L
        b.x(self.A2[0][0])   # C = 1
        b.x(self.A1[1][0])   # x^1 * B, B = 1
        b.x(self.breg[0])    # b = 1

    def free_state(self):
        b = self.b
        b.x(self.A2[0][0]); b.x(self.A1[1][0]); b.x(self.breg[0])
        for r in (*self.A2, *self.A1, self.breg, self.Lreg):
            b.free(r)

    # ---- one BM step -----------------------------------------------------------------------------
    def discrepancy(self, n: int, d, S):
        for i in range(self.disc_bound(n) + 1):
            self.xor_prod(self.A2[i], S[n - i], d)

    def disc_pairs(self, n: int, S):
        return [(self.A2[i], S[n - i]) for i in range(self.disc_bound(n) + 1)]

    def rotate_A1(self, up: bool):
        """Cyclic rotation of the slot array A1 by one slot (free SWAPs)."""
        order = range(self.slots - 1, 0, -1) if up else range(1, self.slots)
        for i in order:
            self.b.swap(self.A1[i], self.A1[i - 1])

    def lc_flag(self, n: int, d, lc):
        """lc = (d != 0) and (L <= n // 2)."""
        b = self.b
        z = b.alloc_qubits(1)[0]
        cond = b.alloc_qubits(1)[0]
        self.is_zero(d, z)
        b.x(z)
        b.flip_if_less_than(self.Lreg, n // 2 + 1, target=cond)
        b.ccx(z, cond, lc)
        b.flip_if_less_than(self.Lreg, n // 2 + 1, target=cond)
        b.x(z)
        self.is_zero(d, z)
        b.free(cond); b.free(z)

    def update_L(self, n: int, lc):
        """if lc: L <- n + 1 - L  (self-inverse)."""
        b = self.b
        tmp = b.alloc_qubits(len(self.Lreg))
        b.cx(self.Lreg, tmp)
        b.x(tmp); b.iadd(1, target=tmp)          # tmp = -L
        b.iadd(n + 1, target=tmp)                # tmp = n + 1 - L
        b.cswap(lc, self.Lreg, tmp)
        b.isub(n + 1, target=tmp); b.isub(1, target=tmp); b.x(tmp)   # tmp = L (new)
        b.cx(self.Lreg, tmp)                     # tmp = 0
        b.free(tmp)

    def quotient_recursion(self, n: int, nu, wrap):
        """nu ^= selected quotient coefficient of A2 / B, where B = A1 shifted down by j (j = 0, 1, 2 selected by
        zero tests of A1[0..2]); gated by `wrap` = 0 (L > n - t). Also used by the reverse pass."""
        b = self.b
        A1, A2 = self.A1, self.A2
        if self.slots < 5:
            raise ValueError('need at least 5 slots')
        # flags: f0 = A1[0] != 0 ; f1 = A1[0] == 0 & A1[1] != 0 ; f2 = A1[0] == 0 & A1[1] == 0 & A1[2] != 0 ; all & ~wrap
        z0, z1, z2 = b.alloc_qubits(3)
        self.is_zero(A1[0], z0); self.is_zero(A1[1], z1); self.is_zero(A1[2], z2)   # (live ladders here cost +30 qubits at the Q peak)
        # ok = (not wrap) and (2L > n): a step with 2L <= n and no length change had d = 0 (tail of zero
        # discrepancies, where j >= m1 and the recursion is invalid), so nothing must be XORed into nu there.
        cond = b.alloc_qubits(1)[0]
        b.flip_if_less_than(self.Lreg, n // 2 + 1, target=cond)   # cond = L <= n // 2
        b.x(cond); b.x(wrap)
        nw = b.alloc_qubits(1)[0]
        b.init_and(cond, wrap, nw)                                # nw = (2L > n) and not wrap
        b.x(wrap); b.x(cond)
        f0, f1, f2 = b.alloc_qubits(3)
        b.x(z0); b.init_and(z0, nw, f0); b.x(z0)                 # f0 = ~z0 & nw
        g = b.alloc_qubits(1)[0]
        b.init_and(z0, nw, g)                                     # g = z0 & nw
        b.x(z1); b.init_and(g, z1, f1); b.x(z1)                   # f1 = g & ~z1
        h = b.alloc_qubits(1)[0]
        b.init_and(g, z1, h)                                      # h = z0 & z1 & nw
        b.x(z2); b.init_and(h, z2, f2); b.x(z2)                   # f2 = h & ~z2
        # selected B[k] = A1[j + k]: slots below j are zero, so B0 = A1[0] + f1 A1[1] + f2 A1[2];
        # for k >= 1 use B_k = A1[k] + f1 (A1[k] + A1[k+1]) + f2 (A1[k] + A1[k+2]) (two controlled copies each)
        B0, B1, B2 = (b.alloc_qubits(self.deg) for _ in range(3))
        sel = []
        for k, dst in enumerate((B0, B1, B2)):
            b.cx(A1[k], dst)
            terms = []
            for flag, off in ((f1, 1), (f2, 2)):
                if k:
                    b.cx(A1[k], A1[k + off])          # A1[k+off] <- A1[k] + A1[k+off]
                self.ctrl_copy(flag, A1[k + off], dst)
                terms.append((flag, A1[k + off], k))
                if k:
                    b.cx(A1[k], A1[k + off])
            sel.append((dst, terms))
        if self.tower is not None:
            inv = self.tower.init_inverse_compact(b, B0)        # no scaffold kept: this is the circuit's qubit peak
        else:
            inv, scaffold = self.inv_init(B0)
        q0 = b.alloc_qubits(self.deg); self.xor_prod(A2[0], inv, q0)
        t1 = b.alloc_qubits(self.deg); b.cx(A2[1], t1); self.xor_prod(q0, B1, t1)
        if n == 1:
            b.x(t1[0])          # first division: X = x * B_init = x, X[1] = 1
        q1 = b.alloc_qubits(self.deg); self.xor_prod(t1, inv, q1)
        t2 = b.alloc_qubits(self.deg); b.cx(A2[2], t2); self.xor_prod(q0, B2, t2); self.xor_prod(q1, B1, t2)
        if n == 3:
            b.x(t2[0])          # first division with S_1 = 0: X = x^2, X[2] = 1
        q2 = b.alloc_qubits(self.deg); self.xor_prod(t2, inv, q2)
        self.ctrl_copy(f0, q0, nu); self.ctrl_copy(f1, q1, nu); self.ctrl_copy(f2, q2, nu)
        # uncompute: products by measurement (del_gf2_mul), copy-plus-products by HMR + Z/CZ fixups (0 Toffolis)
        b.del_gf2_mul(t2, inv, target=q2, field=self.gf); b.free(q2)
        if n == 3:
            b.x(t2[0])
        self.mbuc_erase_sum_of_products(t2, [(q0, B2), (q1, B1), (A2[2], None)]); b.free(t2)
        b.del_gf2_mul(t1, inv, target=q1, field=self.gf); b.free(q1)
        if n == 1:
            b.x(t1[0])
        self.mbuc_erase_sum_of_products(t1, [(q0, B1), (A2[1], None)]); b.free(t1)
        b.del_gf2_mul(A2[0], inv, target=q0, field=self.gf); b.free(q0)
        if self.tower is not None:
            self.tower.del_inverse_compact(b, B0, inv); b.free(inv)
        else:
            self.inv_del(B0, inv, scaffold)
        for dst, terms in sel:
            k = terms[0][2]
            for flag, src, _ in terms:
                if k:
                    b.cx(A1[k], src)
            bits = b.alloc_bits(self.deg)
            b.hmr(dst, bits)
            for q, bit in zip(A1[k], bits):           # plain copy term
                b.push_condition(bit); b.z(q); b.pop_condition()
            for flag, src, _ in terms:
                for q, bit in zip(src, bits):
                    b.push_condition(bit); b.cz(flag, q); b.pop_condition()
            b.free(bits)
            for flag, src, _ in terms:
                if k:
                    b.cx(A1[k], src)
            b.free(dst)
        b.x(z2); b.del_and(h, z2, f2); b.x(z2); b.del_and(g, z1, h)
        b.x(z1); b.del_and(g, z1, f1); b.x(z1); b.del_and(z0, nw, g)
        b.x(z0); b.del_and(z0, nw, f0); b.x(z0)
        b.free(km.array([f0, f1, f2, g, h]))
        self.is_zero(A1[2], z2); self.is_zero(A1[1], z1)
        b.free(km.array([z1, z2]))
        return z0, nw, cond                     # kept for the caller's lc erasure; released by `recursion_close`

    def recursion_close(self, n: int, wrap, z0, nw, cond):
        b = self.b
        b.x(cond); b.x(wrap); b.del_and(cond, wrap, nw); b.x(wrap); b.x(cond); b.free(nw)
        b.flip_if_less_than(self.Lreg, n // 2 + 1, target=cond); b.free(cond)
        self.is_zero(self.A1[0], z0); b.free(z0)

    def wrap_flag(self, n: int, wrap):
        """wrap = (L <= n - t): A1 may have wrapped around; then d = 0 and nothing is erased."""
        if n - self.t >= 0:
            self.b.flip_if_less_than(self.Lreg, n - self.t + 1, target=wrap)

    def forward_step(self, n: int, S):
        b = self.b
        with b.mark('bm_step'):
            d = b.alloc_qubits(self.deg)
            with b.mark('discrepancy'):
                self.discrepancy(n, d, S)
            lc = b.alloc_qubits(1)[0]
            self.lc_flag(n, d, lc)
            with b.mark('cswap'):
                for i in range(self.swap_bound(n) + 1):
                    b.cswap(lc, self.A1[i], self.A2[i])
                b.cswap(lc, d, self.breg)
            with b.mark('quotient'):
                binv, sc = self.inv_init(self.breg)
                nu = b.alloc_qubits(self.deg)
                self.xor_prod(d, binv, nu)
                self.inv_del(self.breg, binv, sc)
            with b.mark('update'):
                for i in range(self.upd_bound(n) + 1):
                    self.xor_prod(nu, self.A1[i], self.A2[i])
            b.del_gf2_mul(nu, self.breg, target=d, field=self.gf)   # d' = nu * b'
            b.free(d)
            self.update_L(n, lc)
            wrap = b.alloc_qubits(1)[0]
            self.wrap_flag(n, wrap)
            with b.mark('erase_nu'):
                z0, nw, cond = self.quotient_recursion(n, nu, wrap)
            b.free(nu)
            # erase lc: lc ^= (A1[0] != 0) & (2L > n) & ~wrap   (reuses the recursion's z0 and nw)
            b.x(z0); b.ccx(z0, nw, lc); b.x(z0)
            self.recursion_close(n, wrap, z0, nw, cond)
            b.free(lc)
            self.wrap_flag(n, wrap)
            b.free(wrap)
            if n + 1 < self.n:
                self.rotate_A1(up=True)

    def reverse_step(self, n: int, S):
        """Exact mirror of `forward_step`."""
        b = self.b
        with b.mark('bm_step_rev'):
            if n + 1 < self.n:
                self.rotate_A1(up=False)
            wrap = b.alloc_qubits(1)[0]
            self.wrap_flag(n, wrap)
            lc = b.alloc_qubits(1)[0]
            nu = b.alloc_qubits(self.deg)
            with b.mark('erase_nu'):
                z0, nw, cond = self.quotient_recursion(n, nu, wrap)
            b.x(z0); b.ccx(z0, nw, lc); b.x(z0)
            self.recursion_close(n, wrap, z0, nw, cond)
            self.wrap_flag(n, wrap)
            b.free(wrap)
            self.update_L(n, lc)
            d = b.alloc_qubits(self.deg)
            self.xor_prod(nu, self.breg, d)                       # d' = nu * b'
            with b.mark('update'):
                for i in range(self.upd_bound(n) + 1):
                    self.xor_prod(nu, self.A1[i], self.A2[i])
            with b.mark('quotient'):
                binv, sc = self.inv_init(self.breg)
                self.xor_prod(d, binv, nu)
                self.inv_del(self.breg, binv, sc)
            b.free(nu)
            with b.mark('cswap'):
                b.cswap(lc, d, self.breg)
                for i in range(self.swap_bound(n) + 1):
                    b.cswap(lc, self.A1[i], self.A2[i])
            self.lc_flag(n, d, lc)
            b.free(lc)
            with b.mark('discrepancy_mbuc'):
                self.mbuc_erase_sum_of_products(d, self.disc_pairs(n, S))
            b.free(d)

    # ---- omega in place of S_1..S_t -----------------------------------------------------------------
    # omega_i = sum_{k<=i} sigma_k S[i-k] (scaled sigma = A2) is accumulated *in* S[i]'s register, top-down:
    # S[i] <- A2[0] * S[i] (in-place scaling via s^-1: two products), then S[i] ^= sum_{k>=1} A2[k] S[i-k].
    def alloc_omega(self, S):
        self.W = [S[i] for i in range(self.t)]

    def _scale_inplace(self, x, s, sinv):
        """x <- s * x using p = s*x; x ^= sinv*p; swap(p, x) (2 products, p transient)."""
        b = self.b
        p = b.alloc_qubits(self.deg)
        self.xor_prod(s, x, p)
        self.xor_prod(sinv, p, x)          # x -> 0
        b.swap(p, x)
        b.free(p)

    def omega_swap_forward(self, S):
        b, t = self.b, self.t
        with b.mark('omega_inplace'):
            if self.tower is not None:
                sinv = self.tower.init_inverse_compact(b, self.A2[0])
            else:
                sinv, sc = self.inv_init(self.A2[0])
            for i in range(t - 1, -1, -1):
                self._scale_inplace(S[i], self.A2[0], sinv)
                for k in range(1, i + 1):
                    self.xor_prod(self.A2[k], S[i - k], S[i])
            if self.tower is not None:
                self.tower.del_inverse_compact(b, self.A2[0], sinv); b.free(sinv)
            else:
                self.inv_del(self.A2[0], sinv, sc)

    def omega_swap_reverse(self, S):
        b, t = self.b, self.t
        with b.mark('omega_inplace_rev'):
            if self.tower is not None:
                sinv = self.tower.init_inverse_compact(b, self.A2[0])
            else:
                sinv, sc = self.inv_init(self.A2[0])
            for i in range(t):
                for k in range(1, i + 1):
                    self.xor_prod(self.A2[k], S[i - k], S[i])
                self._scale_inplace(S[i], sinv, self.A2[0])
            if self.tower is not None:
                self.tower.del_inverse_compact(b, self.A2[0], sinv); b.free(sinv)
            else:
                self.inv_del(self.A2[0], sinv, sc)

    # ---- Chien search + Forney phase kickback -------------------------------------------------------
    def sweep(self, E):
        """For every position j (X_j = alpha^j): if sigma(X_j^-1) = 0, phase by <omega(X_j^-1)/sigma'(X_j^-1), M_j>."""
        b, gf, t = self.b, self.gf, self.t
        alpha = gf.primitive_element
        order = (1 << self.deg) - 1
        A2, W = self.A2, self.W
        odd = [A2[k] for k in range(1, t + 1, 2)]
        even = [A2[k] for k in range(2, t + 1, 2)]
        z = b.alloc_qubits(1)[0]
        located = b.alloc_qubits(self.deg)
        is_err = b.alloc_bits(1)
        step = b.mark('forney_step')
        for j in range(self.m):
            with step:
                b.bit_store(is_err[0], True, control=E[j])
                b.push_condition(is_err[0])
                # sums in place: A2[1] <- T = sum of odd slots; A2[0] <- sigma(X^-1) = A2[0] + T + even; W[0] <- omega(X^-1)
                for r in odd[1:]:
                    b.cx(r, A2[1])
                for r in even:
                    b.cx(r, A2[0])
                b.cx(A2[1], A2[0])
                for r in W[1:]:
                    b.cx(r, W[0])
                ladder = self.is_zero_open(A2[0], z)
                if self.tower is not None:
                    inv, sc = self.tower.init_inverse_with_scaffold(b, A2[1], gate=z)   # z / T
                else:
                    inv, sc = self.inv_init(A2[1])
                b.gf2_imul(gf.pow(alpha, (-j) % order), target=inv, field=gf) if j % order else None
                if self.tower is not None:
                    b.gf2_phase_by_product(W[0], inv, mask=E[j], field=gf)
                else:
                    b.init_and(z, W[0], located)
                    b.gf2_phase_by_product(located, inv, mask=E[j], field=gf)
                    b.del_and(z, W[0], located)
                if j % order:
                    b.gf2_idiv(gf.pow(alpha, (-j) % order), target=inv, field=gf)
                self.inv_del(A2[1], inv, sc)
                self.is_zero_close(A2[0], z, ladder)
                for r in W[1:]:
                    b.cx(r, W[0])
                b.cx(A2[1], A2[0])
                for r in even:
                    b.cx(r, A2[0])
                for r in odd[1:]:
                    b.cx(r, A2[1])
                b.pop_condition()
                b.bit_store(is_err[0], False)
                if j + 1 < self.m:
                    for k in range(1, t + 1):
                        b.gf2_imul(gf.pow(alpha, (-k) % order), target=A2[k], field=gf)
                    for i in range(1, t):
                        b.gf2_imul(gf.pow(alpha, (-i) % order), target=W[i], field=gf)
        # undo the Chien rotations (m - 1 steps)
        for k in range(1, t + 1):
            b.gf2_idiv(gf.pow(alpha, (-k * (self.m - 1)) % order), target=A2[k], field=gf)
        for i in range(1, t):
            b.gf2_idiv(gf.pow(alpha, (-i * (self.m - 1)) % order), target=W[i], field=gf)
        b.free(is_err); b.free(located); b.free(z)

    def build(self, S, E):
        b = self.b
        self.alloc_state()
        with b.mark('bm_forward'):
            for i in range(self.n):
                self.forward_step(i, S)
        self.alloc_omega(S)
        self.omega_swap_forward(S)
        with b.mark('sweep'):
            self.sweep(E)
        self.omega_swap_reverse(S)
        with b.mark('bm_reverse'):
            for i in reversed(range(self.n)):
                self.reverse_step(i, S)
        self.free_state()


def build_circuit(n: int, degree: int, m: int) -> km.Circuit:
    gf = km.GF2Field(degree)
    b = km.CircuitBuilder()
    S = [b.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    E = [b.create_classical_register(degree, name=f'E{j}') for j in range(m)]
    BMDecoder(b, n, gf, m).build(S, E)
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
