"""42-AND bilinear algorithm for GF(4096) multiplication: GF(4096) = GF(8)[y]/(g), Toom-Cook 4-way at the seven points
{0, 1, γ, γ², γ³, γ⁴, ∞} of GF(8) ∪ {∞}, each point product a 6-AND GF(8) multiplication. Every basis change,
evaluation and interpolation is GF(2)-linear, so the result is a list of 42 terms (xbits, ybits, out_mask):
target ^= mask whenever (XOR of x[xbits]) AND (XOR of y[ybits]) = 1. Built and verified numerically (python mul42.py)."""
from __future__ import annotations

import kickmix as km


def _solve_coords(basis, elem, gf):
    """Coordinates (bits) of elem in the GF(2)-basis `basis` (list of field elements), by Gaussian elimination."""
    n = len(basis)
    rows = [[(basis[c] >> r) & 1 for c in range(n)] + [(elem >> r) & 1] for r in range(gf.degree)]
    # eliminate
    piv_cols = []
    r0 = 0
    for c in range(n):
        piv = next((r for r in range(r0, len(rows)) if rows[r][c]), None)
        if piv is None:
            continue
        rows[r0], rows[piv] = rows[piv], rows[r0]
        for r in range(len(rows)):
            if r != r0 and rows[r][c]:
                rows[r] = [a ^ b for a, b in zip(rows[r], rows[r0])]
        piv_cols.append(c); r0 += 1
    coords = [0] * n
    for i, c in enumerate(piv_cols):
        coords[c] = rows[i][n]
    return coords


def build_terms(gf=None):
    gf = gf or km.GF2Field(12)
    assert gf.degree == 12
    alpha = gf.primitive_element
    gamma = gf.pow(alpha, 585)                       # generator of GF(8)*
    G8 = [0] + [gf.pow(gamma, i) for i in range(7)]
    gb = [1, gamma, gf.mul(gamma, gamma)]            # GF(2)-basis of GF(8)
    # GF(8) multiplication as 6 bilinear terms on gamma-coordinates: polynomial product mod m_gamma (cubic)
    from gfmul import karatsuba_terms
    m3 = karatsuba_terms(3)                          # (L, R, outs) with outs in 0..4 (polynomial coefficients)
    red3 = [_solve_coords(gb, gf.pow(gamma, o), gf) for o in range(5)]   # gamma^o in the gamma-basis
    g8_terms = []
    for L, R, outs in m3:
        mask = [0, 0, 0]
        for o in outs:
            mask = [a ^ b for a, b in zip(mask, red3[o])]
        g8_terms.append((L, R, mask))
    # GF(8)-basis {1, alpha, alpha^2, alpha^3} of GF(4096); GF(2)-basis e_{ij} = gamma^j alpha^i (index 3i + j)
    ab = [gf.pow(alpha, i) for i in range(4)]
    basis12 = [gf.mul(gb[j], ab[i]) for i in range(4) for j in range(3)]
    def coords12(x): return _solve_coords(basis12, x, gf)
    def g8_coords(x): return _solve_coords(gb, x, gf)            # x in GF(8)
    def from_g8(c): return gf.mul(c[0], 1) ^ gf.mul(c[1], gb[1]) ^ gf.mul(c[2], gb[2]) if False else (c[0] * gb[0]) ^ (c[1] * gb[1]) ^ (c[2] * gb[2])
    # minimal polynomial g(y) of alpha over GF(8): prod (y - alpha^(8^j))
    poly = [1]
    for j in range(4):
        r = gf.pow(alpha, 8 ** j)
        poly = [(poly[i - 1] if i else 0) ^ gf.mul(r, poly[i]) if i < len(poly) else poly[i - 1] for i in range(len(poly) + 1)]
    assert len(poly) == 5 and poly[4] == 1 and all(p in G8 for p in poly)
    # y^m mod g for m = 0..6 as GF(8) combos of y^0..y^3
    ymod = [[1 if i == m else 0 for i in range(4)] for m in range(4)]
    for m in range(4, 7):
        prev = ymod[m - 1]                            # y * prev
        shifted = [0] + prev[:3]
        top = prev[3]
        ymod.append([shifted[i] ^ gf.mul(top, poly[i]) for i in range(4)])
    pts = [0, 1, gamma, gf.pow(gamma, 2), gf.pow(gamma, 3), gf.pow(gamma, 4)]   # 6 finite points + infinity
    # interpolation: d(y) of degree <= 6; d_6 = w_inf; the finite values give the degree-5 remainder
    V = [[gf.pow(p, i) if i else 1 for i in range(6)] for p in pts]
    from tmvp import gf2_matrix_inverse
    Vinv = gf2_matrix_inverse(V, gf)
    terms = []
    for pi in range(7):
        # evaluation of a(y) = sum c_i y^i at point pi: linear map from the 12 coordinate bits to GF(8) (3 bits)
        def eval_rows(pi=pi):
            rows = []                                 # rows[r] = set of coordinate indices (3i+j) feeding bit r of a(p)
            for r in range(3):
                rows.append(set())
            for i in range(4):
                for j in range(3):
                    if pi < 6:
                        val = gf.mul(gb[j], gf.pow(pts[pi], i) if i else 1)
                    else:
                        val = gb[j] if i == 3 else 0
                    for r, bit in enumerate(g8_coords(val)):
                        if bit:
                            rows[r] ^= {3 * i + j}
            return rows
        rows = eval_rows()
        # output map: a product value w (GF(8), 3 bits) at point pi -> 12-bit contribution in the standard basis
        def out_vec(wbits):
            w = from_g8(wbits)
            d = [0] * 7
            if pi == 6:
                d[6] = w
                # the finite points' values must be corrected by -w*p^6 before interpolation
                corr = [gf.mul(w, gf.pow(p, 6)) if p else 0 for p in pts]
                for m in range(6):
                    acc = 0
                    for k in range(6):
                        acc ^= gf.mul(Vinv[m][k], corr[k])
                    d[m] = acc
            else:
                for m in range(6):
                    d[m] = gf.mul(Vinv[m][pi], w)
            e = [0] * 4
            for m in range(7):
                for i in range(4):
                    e[i] ^= gf.mul(d[m], ymod[m][i])
            x = 0
            for i in range(4):
                x ^= gf.mul(e[i], ab[i])
            return x
        outs = [out_vec([1 if r == k else 0 for r in range(3)]) for k in range(3)]
        for L, R, mask in g8_terms:
            xb = set()
            for r in L: xb ^= rows[r]
            yb = set()
            for r in R: yb ^= rows[r]
            m12 = 0
            for k in range(3):
                if mask[k]: m12 ^= outs[k]
            terms.append((sorted(xb), sorted(yb), m12))
    # the terms act on coordinates; compose with the basis change x -> coords12(x) (linear): a bit of coords is a
    # linear form of the standard bits.  Row c of C = coords12 matrix: coords12(x)[c] = sum_s C[c][s] x_s.
    C = [coords12(1 << s) for s in range(12)]         # C[s][c]: coordinate c of basis vector s
    final = []
    for xb, yb, m12 in terms:
        xs = [s for s in range(12) if sum(C[s][c] for c in xb) & 1]
        ys = [s for s in range(12) if sum(C[s][c] for c in yb) & 1]
        if xs and ys and m12:
            final.append((xs, ys, m12))
    return final


def check(terms, gf, trials=2000):
    import random
    for _ in range(trials):
        a, b = random.randrange(4096), random.randrange(4096)
        acc = 0
        for xs, ys, m in terms:
            if (sum((a >> s) & 1 for s in xs) & 1) and (sum((b >> s) & 1 for s in ys) & 1):
                acc ^= m
        if acc != gf.mul(a, b):
            return False
    return True


if __name__ == '__main__':
    gf = km.GF2Field(12)
    terms = build_terms(gf)
    print(f'terms={len(terms)} verified={check(terms, gf)} avg|L|={sum(len(t[0]) for t in terms)/len(terms):.1f} avg|R|={sum(len(t[1]) for t in terms)/len(terms):.1f} avg|mask|={sum(bin(t[2]).count("1") for t in terms)/len(terms):.1f}')
