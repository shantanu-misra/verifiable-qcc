"""Rank-(p+q-1) bilinear algorithm for the Toeplitz matrix-vector product over GF(2^deg) (transposed Toom-Cook).
y_k = sum_j t_{k-j+q-1} v_j  (k < p, j < q; t has p+q-1 entries)  =  E_p^T ((I^T t) * (E_q rev(v))).
Checked numerically in build/tmvp_eval.py. Only the products cost Toffolis; all constants are GF(2)-linear maps."""
from __future__ import annotations


def gf2_matrix_inverse(M, gf):
    n = len(M); A = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(M)]
    for c in range(n):
        piv = next(r for r in range(c, n) if A[r][c])
        A[c], A[piv] = A[piv], A[c]
        inv = gf.invert(A[c][c]); A[c] = [gf.mul(inv, x) for x in A[c]]
        for r in range(n):
            if r != c and A[r][c]:
                f = A[r][c]; A[r] = [x ^ gf.mul(f, y) for x, y in zip(A[r], A[c])]
    return [row[n:] for row in A]


def tmvp_algorithm(p: int, q: int, gf):
    """(alpha, beta, gamma): y_k = sum_m gamma[m][k] * (sum_n alpha[m][n] t_n) * (sum_j beta[m][j] v_j), m < p+q-1."""
    N = p + q - 1
    pts = [gf.pow(gf.primitive_element, i) for i in range(N - 1)] + [0]
    E = [[1 if j == 0 else gf.pow(x, j) for j in range(N)] for x in pts]
    I = gf2_matrix_inverse(E, gf)
    alpha = [[I[n][m] for n in range(N)] for m in range(N)]
    beta = [[E[m][q - 1 - j] for j in range(q)] for m in range(N)]
    gamma = [[E[m][k] for k in range(p)] for m in range(N)]
    return alpha, beta, gamma
