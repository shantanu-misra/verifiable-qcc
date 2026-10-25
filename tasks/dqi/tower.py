"""GF(2^12) inverse via the tower GF(2^12) = GF(2^6)[u]/(u^2 + u + nu).

(a + b u)^-1 = (a + b + b u) * N^-1 with N = a^2 + a b + nu b^2 in GF(2^6). Basis changes between kickmix's
polynomial basis (modulus 0x1053) and the tower basis {v^i, u v^i} are GF(2)-linear (CNOT networks, 0 Toffolis).
Cost: 3 GF(2^6) products + 1 GF(2^6) inverse (kickmix: 21 and 63 Toffolis) = 126 Toffolis instead of 315.
"""
from __future__ import annotations

import kickmix as km

from gfmul import GFMul


def gf2_matrix_inverse(M: list[list[int]]) -> list[list[int]]:
    n = len(M)
    A = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(M)]
    r = 0
    for c in range(n):
        piv = next((i for i in range(r, n) if A[i][c]), None)
        if piv is None:
            raise ValueError('singular')
        A[r], A[piv] = A[piv], A[r]
        for i in range(n):
            if i != r and A[i][c]:
                A[i] = [x ^ y for x, y in zip(A[i], A[r])]
        r += 1
    return [row[n:] for row in A]


def apply_gf2_matrix(builder: km.CircuitBuilder, M: list[list[int]], reg) -> None:
    """In-place y <- M y on the qubits `reg` (bit i of y is reg[i]) using CX and SWAP only."""
    n = len(M)
    A = [row[:] for row in M]
    # Reduce A to the identity by row operations applied to the *inverse* sequence: we record elementary ops
    # that turn A into I (left-multiplying), then apply their inverses in reverse order to the register.
    ops = []  # ('swap', i, j) or ('add', src, dst): row dst ^= row src
    for c in range(n):
        piv = next((i for i in range(c, n) if A[i][c]), None)
        if piv is None:
            raise ValueError('singular matrix')
        if piv != c:
            A[c], A[piv] = A[piv], A[c]; ops.append(('swap', c, piv))
        for i in range(n):
            if i != c and A[i][c]:
                A[i] = [x ^ y for x, y in zip(A[i], A[c])]; ops.append(('add', c, i))
    # Now E_k ... E_1 A = I, so A = E_1^-1 ... E_k^-1; y <- A y applies E_k^-1 first (each E is its own inverse).
    for op in reversed(ops):
        if op[0] == 'swap':
            builder.swap(reg[op[1]], reg[op[2]])
        else:
            builder.cx(reg[op[1]], reg[op[2]])


class Tower:
    """GF(2^(2k)) = GF(2^k)[u]/(u^2 + u + nu) inverse; the GF(2^k) inverse is kickmix's or another Tower."""

    def __init__(self, degree: int, sub: 'Tower | None' = None):
        self.gf = km.GF2Field(degree)
        k = degree // 2
        self.k = k
        self.g = km.GF2Field(k)
        self.sub = sub
        g = self.g
        def trace(x):
            t = 0; y = x
            for _ in range(k):
                t ^= y; y = g.square(y)
            return t
        self.nu = next(x for x in range(1, 1 << k) if trace(x) == 1)
        def tmul(p, q):
            a, b = p; c, d = q
            bd = g.mul(b, d)
            return (g.mul(a, c) ^ g.mul(self.nu, bd), g.mul(a, d) ^ g.mul(b, c) ^ bd)
        mod = self.gf.modulus
        def eval_mod(beta):
            acc = (0, 0); pw = (1, 0)
            for i in range(degree + 1):
                if (mod >> i) & 1:
                    acc = (acc[0] ^ pw[0], acc[1] ^ pw[1])
                pw = tmul(pw, beta)
            return acc
        beta = next((a, b) for a in range(1 << k) for b in range(1 << k) if (a, b) != (0, 0) and eval_mod((a, b)) == (0, 0))
        cols = []; pw = (1, 0)
        for i in range(degree):
            cols.append([(pw[0] >> j) & 1 for j in range(k)] + [(pw[1] >> j) & 1 for j in range(k)]); pw = tmul(pw, beta)
        self.M = [[cols[i][r] for i in range(degree)] for r in range(degree)]
        self.Minv = gf2_matrix_inverse(self.M)
        self.mul = GFMul(g) if k in (3, 6, 12) else None
        self.sq = [[(g.square(1 << i) >> r) & 1 for i in range(k)] for r in range(k)]
        self.nusq = [[(g.mul(self.nu, g.square(1 << i)) >> r) & 1 for i in range(k)] for r in range(k)]

    def to_tower(self, b, reg):
        apply_gf2_matrix(b, self.M, reg)

    def to_poly(self, b, reg):
        apply_gf2_matrix(b, self.Minv, reg)

    def _mul(self, b, x, y, target):
        if self.mul is not None:
            self.mul.ixor_mul(b, x, y, target)
        else:
            b.ixor_gf2_mul(x, y, target=target, field=self.g)

    def _norm_linear(self, b, a, bb, N):
        for i in range(self.k):
            for r in range(self.k):
                if self.sq[r][i]:
                    b.cx(a[i], N[r])
                if self.nusq[r][i]:
                    b.cx(bb[i], N[r])

    _LEAF = None

    @classmethod
    def _leaf_table(cls):
        """x^-1 in GF(2^3) as a quadratic map: 3 ANDs x_i x_j plus linear and constant terms per output bit."""
        if cls._LEAF is None:
            g = km.GF2Field(3)
            inv = [0] + [g.pow(x, 6) for x in range(1, 8)]
            pairs = [(0, 1), (0, 2), (1, 2)]
            def ev(x, c, d, e):
                v = e
                for k, (i, j) in enumerate(pairs):
                    if (c >> k) & 1: v ^= ((x >> i) & 1) & ((x >> j) & 1)
                for j in range(3):
                    if (d >> j) & 1: v ^= (x >> j) & 1
                return v
            table = []
            for r in range(3):
                sol = next((c, d, e) for c in range(8) for d in range(8) for e in range(2)
                           if all(ev(x, c, d, e) == (inv[x] >> r) & 1 for x in range(8)))
                table.append(sol)
            cls._LEAF = (pairs, table)
        return cls._LEAF

    def _leaf_inv_init(self, b, N):
        """(inv, None): inv = N^-1 in GF(2^3) with 3 Toffolis (x_i x_j terms), erased later by measurement."""
        pairs, table = self._leaf_table()
        inv = b.alloc_qubits(3); anc = b.alloc_qubits(1)[0]; bit = b.alloc_bits(1)
        for k, (i, j) in enumerate(pairs):
            b.ccx(N[i], N[j], anc)
            for r, (c, d, e) in enumerate(table):
                if (c >> k) & 1:
                    b.cx(anc, inv[r])
            b.hmr(anc, bit[0]); b.push_condition(bit[0]); b.cz(N[i], N[j]); b.pop_condition()
        for r, (c, d, e) in enumerate(table):
            for j in range(3):
                if (d >> j) & 1:
                    b.cx(N[j], inv[r])
            if e:
                b.x(inv[r])
        b.free(anc); b.free(bit)
        return inv, None

    def _leaf_inv_del(self, b, N, inv, t):
        """Erase inv = N^-1 with 0 Toffolis: X-measure each bit, fix the phase with CZ/Z on N (quadratic form)."""
        pairs, table = self._leaf_table()
        bits = b.alloc_bits(3)
        for r, (c, d, e) in enumerate(table):
            if e:
                b.x(inv[r])                   # remove the constant so the fixup is a pure quadratic form
            b.hmr(inv[r], bits[r])
            b.push_condition(bits[r])
            for k, (i, j) in enumerate(pairs):
                if (c >> k) & 1:
                    b.cz(N[i], N[j])
            for j in range(3):
                if (d >> j) & 1:
                    b.z(N[j])
            b.pop_condition()
        b.free(bits); b.free(inv)

    def _sub_inv_init(self, b, N, gate=None):
        """(Ninv, scaffold) for the subfield; with `gate`, Ninv = gate * N^-1 (gating is pushed to the smallest field)."""
        if self.sub is not None:
            return self.sub.init_inverse_with_scaffold(b, N, gate=gate)
        if self.k == 3:
            inv, t = self._leaf_inv_init(b, N)
            if gate is None:
                return inv, (('leaf', t), None, None)
            g_inv = b.alloc_qubits(3)
            b.ccx(gate, inv, g_inv)
            return g_inv, (('leaf', t), inv, gate)
        inv, sc = b.init_gf2_inverse_with_scaffold(N, field=self.g)
        if gate is None:
            return inv, (sc, None, None)
        g_inv = b.alloc_qubits(len(N))
        b.ccx(gate, inv, g_inv)
        return g_inv, (sc, inv, gate)

    def _sub_inv_del(self, b, N, Ninv, sc):
        if self.sub is not None:
            self.sub.del_inverse_with_scaffold(b, N, target=Ninv, scaffold=sc); b.free(Ninv)
        else:
            sc0, inv, gate = sc
            if inv is not None:
                b.del_and(gate, inv, Ninv); b.free(Ninv); Ninv = inv
            if isinstance(sc0, tuple) and sc0[0] == 'leaf':
                self._leaf_inv_del(b, N, Ninv, sc0[1])
            else:
                b.del_gf2_inverse_with_scaffold(N, target=Ninv, scaffold=sc0, field=self.g); b.free(sc0); b.free(Ninv)

    def init_inverse_with_scaffold(self, b: km.CircuitBuilder, x, *, target=None, gate=None):
        """target := x^-1 (0 -> 0), or gate * x^-1 when a control qubit `gate` is given (k Toffolis extra).
        x is left in the polynomial basis. Returns (target, scaffold)."""
        g, k = self.g, self.k
        self.to_tower(b, x)
        a, bb = x[:k], x[k:]
        ab = b.alloc_qubits(k)
        self._mul(b, a, bb, ab)
        N = b.alloc_qubits(k)
        b.cx(ab, N)
        self._norm_linear(b, a, bb, N)
        Ninv, sc = self._sub_inv_init(b, N, gate)      # gated inside the subfield when gate is given
        Ng = None
        use = Ninv
        if target is None:
            target = b.alloc_qubits(2 * k)
        lo, hi = target[:k], target[k:]
        b.cx(bb, a)
        self._mul(b, a, use, lo)
        b.cx(bb, a)
        self._mul(b, bb, use, hi)
        self.to_poly(b, target)
        self.to_poly(b, x)
        return target, (ab, N, Ninv, sc, Ng, gate)

    def _erase_scaffold(self, b, x, scaffold):
        """Erase the scaffold (ab, N, Ninv and the subfield's) with 0 Toffolis while x is intact (x in poly basis)."""
        ab, N, Ninv, sc, Ng, gate = scaffold
        k = self.k
        self.to_tower(b, x)
        a, bb = x[:k], x[k:]
        self._sub_inv_del(b, N, Ninv, sc)
        self._norm_linear(b, a, bb, N)
        b.cx(ab, N)
        b.del_gf2_mul(a, bb, target=ab, field=self.g)
        b.free(N); b.free(ab)
        self.to_poly(b, x)

    def init_inverse_compact(self, b: km.CircuitBuilder, x):
        """inv = x^-1 with no scaffold kept (scaffold erased for free); uncompute with `del_inverse_compact`."""
        inv, sc = self.init_inverse_with_scaffold(b, x)
        self._erase_scaffold(b, x, sc)
        return inv

    def del_inverse_compact(self, b: km.CircuitBuilder, x, inv):
        """Rebuild the scaffold (ab and N^-1, ~46 Toffolis), erase inv's two products, erase the scaffold again."""
        g, k = self.g, self.k
        self.to_tower(b, x)
        a, bb = x[:k], x[k:]
        ab = b.alloc_qubits(k)
        self._mul(b, a, bb, ab)
        N = b.alloc_qubits(k)
        b.cx(ab, N)
        self._norm_linear(b, a, bb, N)
        Ninv, sc = self._sub_inv_init(b, N, None)
        self.to_poly(b, x)
        self.del_inverse_with_scaffold(b, x, target=inv, scaffold=(ab, N, Ninv, sc, None, None))

    def del_inverse_with_scaffold(self, b: km.CircuitBuilder, x, *, target, scaffold):
        g, k = self.g, self.k
        ab, N, Ninv, sc, Ng, gate = scaffold
        use = Ng if Ng is not None else Ninv
        self.to_tower(b, x)
        self.to_tower(b, target)
        a, bb = x[:k], x[k:]
        lo, hi = target[:k], target[k:]
        b.del_gf2_mul(bb, use, target=hi, field=g)
        b.cx(bb, a)
        b.del_gf2_mul(a, use, target=lo, field=g)
        b.cx(bb, a)
        self._sub_inv_del(b, N, Ninv, sc)
        self._norm_linear(b, a, bb, N)
        b.cx(ab, N)
        b.del_gf2_mul(a, bb, target=ab, field=g)
        b.free(N); b.free(ab)
        self.to_poly(b, x)


def Tower12() -> Tower:
    """GF(2^12) -> GF(2^6) -> GF(2^3) tower: 15-AND GF(64) products, 3-AND GF(8) inverse (66 Toffolis ungated)."""
    return Tower(12, sub=Tower(6))


if __name__ == '__main__':
    import random
    tw = Tower12()
    print('nu =', tw.nu, 'sub nu =', tw.sub.nu)
    b = km.CircuitBuilder()
    x = b.create_quantum_register(12, name='x')
    inv, sc = tw.init_inverse_with_scaffold(b, x)
    y = b.create_quantum_register(12, name='y')
    b.ixor_gf2_mul(x, inv, target=y, field=tw.gf)      # y = x * inv (should be 1 for x != 0)
    tw.del_inverse_with_scaffold(b, x, target=inv, scaffold=sc)
    c = b.finish_circuit(); st = c.operation_statistics()
    print('tower inverse+del circuit: toffoli', st.num_toffoli_like, '(incl. 63 for the check product), qubits', c.num_qubits)
    sim = km.Simulator(batch_size=256, seed=1); sim.use_same_registers_as(c)
    xs = [random.randrange(1, 4096) for _ in range(255)] + [0]
    for s, v in enumerate(xs): sim.write_within_shot('x', s, v)
    sim.do(c)
    bad = sum(sim.read_within_shot('y', s, out=int) != (1 if xs[s] else 0) for s in range(256))
    junk = sum(sim.read_within_shot(inv, s, out=int) != 0 for s in range(256))
    print('x*inv != 1 in', bad, 'shots; inv not cleared in', junk, 'shots; phase flips', int(sim.read_phase_flipped_across_shots().sum()))
    # gated inverse: gate=1 -> inverse, gate=0 -> zero output
    b = km.CircuitBuilder()
    x = b.create_quantum_register(12, name='x'); gq = b.create_quantum_register(1, name='g')
    inv, sc = tw.init_inverse_with_scaffold(b, x, gate=gq[0])
    y = b.create_quantum_register(12, name='y'); b.ixor_gf2_mul(x, inv, target=y, field=tw.gf)
    tw.del_inverse_with_scaffold(b, x, target=inv, scaffold=sc)
    c = b.finish_circuit(); sim = km.Simulator(batch_size=256, seed=2); sim.use_same_registers_as(c)
    xs = [random.randrange(1, 4096) for _ in range(256)]
    for s, v in enumerate(xs): sim.write_within_shot('x', s, v); sim.write_within_shot('g', s, s % 2)
    sim.do(c)
    bad = sum(sim.read_within_shot('y', s, out=int) != (s % 2) for s in range(256))
    print('gated: wrong in', bad, 'shots; toffoli', c.operation_statistics().num_toffoli_like - 63, '; phase flips', int(sim.read_phase_flipped_across_shots().sum()))
    # compact inverse: scaffold erased after use, rebuilt to uncompute
    b = km.CircuitBuilder(); x = b.create_quantum_register(12, name='x')
    inv = tw.init_inverse_compact(b, x); y = b.create_quantum_register(12, name='y'); b.ixor_gf2_mul(x, inv, target=y, field=tw.gf)
    tw.del_inverse_compact(b, x, inv)
    c = b.finish_circuit(); sim = km.Simulator(batch_size=256, seed=5); sim.use_same_registers_as(c)
    xs = [random.randrange(1, 4096) for _ in range(255)] + [0]
    for s, v in enumerate(xs): sim.write_within_shot('x', s, v)
    sim.do(c)
    bad = sum(sim.read_within_shot('y', s, out=int) != (1 if xs[s] else 0) for s in range(256))
    junk = sum(sim.read_within_shot(inv, s, out=int) != 0 for s in range(256))
    print('compact: wrong', bad, 'inv not cleared', junk, 'toffoli', c.operation_statistics().num_toffoli_like - 63, 'qubits', c.num_qubits, 'phase flips', int(sim.read_phase_flipped_across_shots().sum()))
