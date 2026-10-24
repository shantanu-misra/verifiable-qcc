"""GF(2^12) product target ^= x * y with 54 Toffolis (recursive Karatsuba 12 -> 6 -> 3, M(3) = 6).

Each AND term is  (L . x) & (R . y)  for GF(2)-linear forms L, R; it is computed into one reusable ancilla with a
single CCX after bringing the forms into place with CNOTs, fanned out by CNOTs into the output bits it feeds, and
erased by X-measurement + CZ fixup (0 Toffolis). The polynomial product (23 bits) is reduced mod the field
polynomial by a linear map folded into the fan-out. kickmix's `ixor_gf2_mul` costs 63.
"""
from __future__ import annotations

import kickmix as km


def karatsuba_terms(n: int, lo: int = 0) -> list[tuple[list[int], list[int], list[int]]]:
    """AND terms for the polynomial product of two n-bit polynomials (n = 3 * 2^k).

    Returns a list of (L, R, outs): L and R are lists of input bit indices (relative to `lo`) whose XORs are
    multiplied, and `outs` the product-coefficient indices (relative to 2*lo) the term contributes to."""
    if n == 3:
        a = [lo, lo + 1, lo + 2]
        return [([a[0]], [a[0]], [0, 1, 2]),        # a0b0 -> c0, c1, c2
                ([a[1]], [a[1]], [1, 2, 3]),        # a1b1 -> c1, c2, c3
                ([a[2]], [a[2]], [2, 3, 4]),        # a2b2 -> c2, c3, c4
                ([a[0], a[1]], [a[0], a[1]], [1]),  # (a0+a1)(b0+b1) -> c1
                ([a[0], a[2]], [a[0], a[2]], [2]),  # (a0+a2)(b0+b2) -> c2
                ([a[1], a[2]], [a[1], a[2]], [3])]  # (a1+a2)(b1+b2) -> c3
    h = n // 2
    terms = []
    # low = A0 B0 -> c[0..], high = A1 B1 -> c[2h..], mid = (A0+A1)(B0+B1) + low + high -> c[h..]
    for L, R, outs in karatsuba_terms(h, lo):
        terms.append((L, R, sorted(set(outs) | set(o + h for o in outs))))
    for L, R, outs in karatsuba_terms(h, lo + h):
        terms.append((L, R, sorted(set(o + 2 * h for o in outs) | set(o + h for o in outs))))
    for L, R, outs in karatsuba_terms(h, lo):
        Lm = [i for i in L] + [i + h for i in L]
        Rm = [i for i in R] + [i + h for i in R]
        terms.append((Lm, Rm, [o + h for o in outs]))
    return terms


class GFMul:
    def __init__(self, field: km.GF2Field):
        self.gf = field
        self.n = field.degree
        assert self.n in (3, 6, 12)
        self.terms = karatsuba_terms(self.n)
        # reduction: polynomial coefficient c_k (k < 2n-1) -> field element x^k mod modulus
        self.red = [field.mod(1 << k) if hasattr(field, 'mod') else None for k in range(2 * self.n - 1)]
        if self.red[0] is None:
            raise RuntimeError('field.mod missing')

    def ixor_mul(self, b: km.CircuitBuilder, x, y, target, anc=None, bits=None) -> None:
        own = anc is None
        if own:
            anc = b.alloc_qubits(1)[0]
            bits = b.alloc_bits(1)
        for L, R, outs in self.terms:
            # bring the linear forms into x[L[0]] and y[R[0]]
            for i in L[1:]:
                b.cx(x[i], x[L[0]])
            for i in R[1:]:
                b.cx(y[i], y[R[0]])
            b.ccx(x[L[0]], y[R[0]], anc)
            mask = 0
            for o in outs:
                mask ^= self.red[o]
            for k in range(self.n):
                if (mask >> k) & 1:
                    b.cx(anc, target[k])
            b.hmr(anc, bits[0])
            b.push_condition(bits[0]); b.cz(x[L[0]], y[R[0]]); b.pop_condition()
            for i in R[1:]:
                b.cx(y[i], y[R[0]])
            for i in L[1:]:
                b.cx(x[i], x[L[0]])
        if own:
            b.free(anc); b.free(bits)


if __name__ == '__main__':
    import random
    for deg in (3, 6, 12):
        gf = km.GF2Field(deg); mul = GFMul(gf)
        b = km.CircuitBuilder()
        x = b.create_quantum_register(deg, name='x'); y = b.create_quantum_register(deg, name='y'); t = b.create_quantum_register(deg, name='t')
        mul.ixor_mul(b, x, y, t)
        c = b.finish_circuit(); st = c.operation_statistics()
        sim = km.Simulator(batch_size=256, seed=2); sim.use_same_registers_as(c); vals = []
        for s in range(256):
            xv, yv, tv = random.randrange(1 << deg), random.randrange(1 << deg), random.randrange(1 << deg); vals.append((xv, yv, tv))
            sim.write_within_shot('x', s, xv); sim.write_within_shot('y', s, yv); sim.write_within_shot('t', s, tv)
        sim.do(c)
        bad = sum(sim.read_within_shot('t', s, out=int) != (vals[s][2] ^ gf.mul(vals[s][0], vals[s][1])) for s in range(256))
        side = sum(sim.read_within_shot('x', s, out=int) != vals[s][0] or sim.read_within_shot('y', s, out=int) != vals[s][1] for s in range(256))
        print(f'GF(2^{deg}): toffoli {st.num_toffoli_like}, wrong products {bad}/256, inputs disturbed {side}, phase flips {int(sim.read_phase_flipped_across_shots().sum())}, qubits {c.num_qubits}')
