# dqi — working notes

Contract, cost formula and the verified baseline (2.444e13, Q 1,893, T 5,725,108) are in `docs/dqi.md`.
Everything below is measured with `operation_statistics()` on small builder circuits or by walking the
regenerated reference `.kmx`; these are **estimates**, never scores (CLAUDE.md rule 1).

## 1. kickmix GF(2¹²) primitive costs (`measure_primitives.py`, kickmix 0.1 at the pinned submodule)

| primitive | Toffoli-like | extra qubits | notes |
|---|---|---|---|
| `init_gf2_mul(x, y)` / `ixor_gf2_mul` (uncontrolled) | 63 | 12 (target) | 481 CX; `del_gf2_mul` is 0 Toffolis (HMR + CZ fixups) |
| `ixor_gf2_mul(..., control=q)` | 75 | 12 scratch | = 63 + 12: product into scratch, controlled XOR out, measure scratch away |
| `init_gf2_mul` + `del_gf2_mul`, both controlled | 150 | 24 | controlled del is not free |
| `init_gf2_inverse` (auto-clean) | 315 | 72 | Itoh–Tsujii, 5 multiplications; scaffold 48 + target 12 (+12 transient) |
| `init_gf2_inverse_with_scaffold` + `del_…_with_scaffold` | 315 | 60 | del is free while the chain is live |
| `init_gf2_div` (auto-clean) / `ixor_gf2_div` | 378 | 72 / 60 | inverse + 1 multiplication |
| `gf2_imul(const)` / `gf2_idiv(const)` | 0 | 0 | ≈ 33 CX each: GF(2)-linear |
| `gf2_iadd(reg, control=q)` | 12 | 0 | 12 CCX |
| `gf2_phase_by_product(x, y, mask)` | 0 | 0 | 144 CZ: ⟨x·y, mask⟩ kickback is free |
| `flip_if_equal(x, 0)` (12-bit) | 12 | 12 transient | 11 would do with an AND ladder; `flip_if_equal(x, y)` also 12 |
| `flip_if_less_than` 8-bit (reg / const) | 8 / 7 | 7 | |
| `iadd(1)` 8-bit (plain / controlled) | 6 / 9 | 7 | |
| `cswap(c, reg12, reg12)` | 12 CSWAP | 0 | |
| `multi_controlled_x` (12 controls) | 11 | 10 | |
| `init_lookup` + `del_lookup`, k-bit address → 12 bits | 17 / 72 / 144 / 279 for k = 4/6/7/8 | k | `del_lookup` raises for k ≥ 10 (needs more clean qubits than it allocates) |
| `init_lookup`, 12-bit address → 12 bits | 4,094 | 12 | plain unary iteration, 2^k − 2 |
| `unary_iteration` over 7-bit address (full sweep) | 126 | 7 | 2^k − 2 |
| `left_rotate(reg12, 4-bit quantum shift)` | 26 CSWAP | 0 | |

Takeaways: a quantum × quantum GF(2¹²) product costs 63 Toffolis and the inverse 315 (5 products); constant
multiplications and the ⟨product, mask⟩ phase kickback are free; a controlled register XOR is 12.
