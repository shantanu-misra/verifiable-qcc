"""Static Toffoli / qubit costs of kickmix GF(2^12) primitives (estimates, not scores).

Builds tiny circuits around one primitive each and reports `operation_statistics()`
and the number of workspace qubits the builder allocated. Run inside Docker:
    docker compose run --rm -T dev python tasks/dqi/measure_primitives.py
"""
import kickmix as km

gf = km.GF2Field(12)
D = 12


def measure(name, fn, n_inputs=2, widths=None):
    b = km.CircuitBuilder()
    widths = widths or [D] * n_inputs
    regs = [b.create_quantum_register(w, name=f'in{i}') for i, w in enumerate(widths)]
    base = b.num_allocated_qubits
    fn(b, *regs)
    c = b.finish_circuit()
    s = c.operation_statistics()
    toff = s.num_toffoli_like
    ccx = s.get('CCX', 0); ccz = s.get('CCZ', 0); csw = s.get('CSWAP', 0)
    cx = s.get('CX', 0); hmr = s.get('HMR', 0); cz = s.get('CZ', 0)
    print(f'{name:<48} toff={toff:>5} (CCX {ccx:>4} CCZ {ccz:>3} CSWAP {csw:>4})  CX={cx:>5} CZ={cz:>4} HMR={hmr:>4}  '
          f'qubits={c.num_qubits:>4} (inputs {base}, extra {c.num_qubits - base})')
    return c


def t_init_mul(b, x, y):
    t = b.init_gf2_mul(x, y, field=gf)
    b.del_gf2_mul(x, y, target=t, field=gf)


def t_init_mul_only(b, x, y):
    b.init_gf2_mul(x, y, field=gf)


def t_ixor_mul(b, x, y, t):
    b.ixor_gf2_mul(x, y, target=t, field=gf)


def t_ixor_mul_ctrl(b, x, y, t, c):
    b.ixor_gf2_mul(x, y, target=t, field=gf, control=c[0])


def t_init_mul_ctrl(b, x, y, c):
    t = b.init_gf2_mul(x, y, field=gf, control=c[0])
    b.del_gf2_mul(x, y, target=t, field=gf, control=c[0])


def t_inv(b, x):
    b.init_gf2_inverse(x, field=gf)


def t_inv_scaffold(b, x):
    inv, sc = b.init_gf2_inverse_with_scaffold(x, field=gf)
    print('   scaffold len', len(sc))
    b.del_gf2_inverse_with_scaffold(x, target=inv, scaffold=sc, field=gf)


def t_div(b, x, y):
    b.init_gf2_div(x, y, field=gf)


def t_div_scaffold(b, x, y):
    q, sc = b.init_gf2_div_with_scaffold(x, y, field=gf)
    print('   scaffold len', len(sc))
    b.del_gf2_div_with_scaffold(x, y, target=q, scaffold=sc, field=gf)


def t_ixor_div(b, x, y, t):
    b.ixor_gf2_div(x, y, target=t, field=gf)


def t_imul_const(b, x):
    b.gf2_imul(gf.primitive_element, target=x, field=gf)
    b.gf2_imul(gf.pow(gf.primitive_element, 1234), target=x, field=gf)


def t_iadd_ctrl(b, x, y, c):
    b.gf2_iadd(x, target=y, field=gf, control=c[0])


def t_phase_by_product(b, x, y):
    m = b.create_classical_register(D, name='mask')
    b.gf2_phase_by_product(x, y, mask=m, field=gf)


def t_flip_if_equal_zero(b, x, t):
    b.flip_if_equal(x, 0, target=t[0])


def t_flip_if_equal_regs(b, x, y, t):
    b.flip_if_equal(x, y, target=t[0])


def t_flip_if_less(b, x, y, t):
    b.flip_if_less_than(x, y, target=t[0])


def t_flip_if_less_const(b, x, t):
    b.flip_if_less_than(x, 37, target=t[0])


def t_iadd_const(b, x):
    b.iadd(1, target=x)


def t_iadd_const_ctrl(b, x, c):
    b.iadd(1, target=x, control=c[0])


def t_iadd_reg(b, x, y):
    b.iadd(y, target=x)


def t_lookup(b, a, t):
    tab = [(i * 2654435761) % (1 << D) for i in range(1 << len(a))]
    pool = b.alloc_qubits(64)  # del_lookup borrows clean workspace from the free pool
    b.free(pool)
    b.init_lookup(table=tab, address=a, target=t)
    b.del_lookup(table=tab, address=a, target=t)


def t_lookup_init_only(b, a, t):
    tab = [(i * 2654435761) % (1 << D) for i in range(1 << len(a))]
    b.init_lookup(table=tab, address=a, target=t)


def t_cswap_reg(b, x, y, c):
    b.cswap(c[0], x, y)


def t_mcx(b, x, t):
    b.multi_controlled_x(x, t[0])


def t_unary(b, a, t):
    with b.unary_iteration(a) as it:
        for v, mq in it:
            b.cx(mq, t[0])


def t_left_rotate_q(b, x, s):
    b.left_rotate(x, s)


print('--- GF(2^12) arithmetic')
measure('init_gf2_mul + del_gf2_mul', t_init_mul)
measure('init_gf2_mul only', t_init_mul_only)
measure('ixor_gf2_mul (uncontrolled)', t_ixor_mul, 3)
measure('ixor_gf2_mul (qubit control)', t_ixor_mul_ctrl, 4, [D, D, D, 1])
measure('init_gf2_mul+del (qubit control)', t_init_mul_ctrl, 3, [D, D, 1])
measure('init_gf2_inverse (auto-clean)', t_inv, 1)
measure('init+del inverse_with_scaffold', t_inv_scaffold, 1)
measure('init_gf2_div (auto-clean)', t_div)
measure('init+del div_with_scaffold', t_div_scaffold)
measure('ixor_gf2_div', t_ixor_div, 3)
measure('gf2_imul const x2', t_imul_const, 1)
measure('gf2_iadd reg->reg controlled', t_iadd_ctrl, 3, [D, D, 1])
measure('gf2_phase_by_product', t_phase_by_product)
print('--- comparisons / counters')
measure('flip_if_equal(x, 0)', t_flip_if_equal_zero, 2, [D, 1])
measure('flip_if_equal(x, y)', t_flip_if_equal_regs, 3, [D, D, 1])
measure('flip_if_less_than(x, y) 8-bit', t_flip_if_less, 3, [8, 8, 1])
measure('flip_if_less_than(x, 37) 8-bit', t_flip_if_less_const, 2, [8, 1])
measure('iadd(1) 8-bit', t_iadd_const, 1, [8])
measure('iadd(1) 8-bit controlled', t_iadd_const_ctrl, 2, [8, 1])
measure('iadd(reg) 8-bit', t_iadd_reg, 2, [8, 8])
measure('cswap 12-qubit registers', t_cswap_reg, 3, [D, D, 1])
measure('multi_controlled_x 12 controls', t_mcx, 2, [D, 1])
print('--- lookups / iteration')
for k in (4, 6, 7, 8):  # del_lookup needs > len(address) clean qubits above 8 bits and raises
    measure(f'init+del lookup {k}-bit addr -> 12', t_lookup, 2, [k, D])
measure('init lookup 12-bit addr only', t_lookup_init_only, 2, [12, D])
measure('unary_iteration over 7-bit addr', t_unary, 2, [7, 1])
measure('left_rotate 12 by 4-bit quantum shift', t_left_rotate_q, 2, [D, 4])
