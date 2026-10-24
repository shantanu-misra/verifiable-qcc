"""Generator for invsqrt candidate #1: QROM-indexed piecewise polynomial (no Newton step).

Circuit structure (see tasks/invsqrt/notes.md sections 4 and 6, and model_pwquad.py for the
bit-exact classical model this must agree with):

    normalize (reference code)         x := x_norm = x << 2s, unary flags = s
    init_lookup(address = x[22-k:])    c2 | c1 | c0 := table[a]      (one joint QROM)
    step 1  r1 = c1 - T(c2 * delta)    `_add_row` rows, one per delta bit (x[0:22-k])
    del_lookup(c2)                     measurement-based, frees c2's value (ids stay allocated)
    step 2  p  = c0 - T(r1 * delta)    into the low wp+1 bits of y1
    denormalize (reference code)       y1 := p << s
    phase_by_bits (reference code)     Z_POW per bit of y1, F = wp + 11 fractional bits of a turn
    uncompute in reverse               (step 2^-1, init_lookup(c2), step 1^-1, del_lookup(all))
    un-normalize

Registers: exactly one quantum register `x` (22 qubits); everything else is scratch and is
freed at the end. Semantics of every arithmetic row are the reference's `_add_row`, imported
from data/verifiers/invsqrt/nr_reference_circuit.py (read-only), so the circuit computes
exactly `model_pwquad.Config.evaluate`.

Usage (inside Docker):
    python tasks/invsqrt/build_pwquad.py out.kmx [--table tasks/invsqrt/pwquad_tables/deg2_k8_wc22_wp28_d14.json] [--check]
`--check` simulates the compute-only prefix for EVERY x and compares y1 with the model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'data', 'verifiers', 'invsqrt'))
sys.path.insert(0, HERE)
import kickmix as km
import nr_reference_circuit as nr  # reference building blocks, read-only
import model_pwquad as mp

DEFAULT_TABLE = os.path.join(HERE, 'pwquad_tables', 'deg2_k8_wc22_wp28_d14.json')


def load_table(path: str) -> tuple[mp.Config, dict[int, list[int]]]:
    j = json.load(open(path))
    c = j['config']
    cfg = mp.Config(c['deg'], c['k'], c['wc'], c['wp'], c['d'])
    return cfg, {int(a): v for a, v in j['tables'].items()}


def _rows_into(builder, inverse, cfg, x, mult, target, scratch, mult_frac, out_frac):
    """`target -= T(mult * delta)` in `_add_row` semantics: one row per delta bit m = k-1 .. k-2+d.

    delta bit of weight 2^-m is x_norm bit (20 - m) = x[20 - m]. Row exponent e = out_frac - mult_frac - m.
    """
    k = cfg.k
    for j in range(cfg.d):
        m = k - 1 + j
        control = x[mp.WIDTH - 2 - m]
        e = out_frac - mult_frac - m
        terms = [(mult[i], i + e) for i in range(max(0, -e), len(mult))]
        if not terms:
            continue  # mult >> -e == 0 for every value: the model's row is identically zero.
        nr._add_row(builder, inverse, control, terms, target, scratch, subtract=True)


def build(cfg: mp.Config, table: dict[int, list[int]], *, compute_only: bool = False) -> km.Circuit:
    assert cfg.deg == 2, 'degree-3 variant not implemented (see model_pwquad.py)'
    k, wc, wp = cfg.k, cfg.wc, cfg.wp
    F = mp.F_OF(wp)
    builder = km.CircuitBuilder()

    def alloc(n, name):
        return builder.create_quantum_register(n, name=name) if compute_only else builder.alloc_qubits(n)

    x = builder.create_quantum_register(mp.WIDTH, name='x')
    flags = alloc(mp.N_P, 'flags')
    len_c2 = wc
    len_c1 = max(v[1] for v in table.values()).bit_length()  # r1 <= c1, never wraps
    len_c0 = wp + 1
    assert max(v[2] for v in table.values()) < 2**len_c2 and max(v[0] for v in table.values()) < 2**len_c0
    c2 = alloc(len_c2, 'c2')
    c1 = alloc(len_c1, 'c1')
    y1 = alloc(F, 'y1')
    scratch = alloc(max(len_c1, len_c0), 'scratch')
    p = y1[:len_c0]
    address = x[mp.WIDTH - k:]

    full = [0] * 2**k
    tab_c2 = [0] * 2**k
    for a, (c0v, c1v, c2v) in table.items():
        full[a] = c2v | (c1v << len_c2) | (c0v << (len_c2 + len_c1))
        tab_c2[a] = c2v
    joint_target = km.array(list(c2) + list(c1) + list(p))

    with builder.mark('compute_y1'):
        with builder.mark('normalize'):
            nr._normalize(builder, False, x, flags)
        with builder.mark('lookup_all'):
            builder.init_lookup(table=full, address=address, target=joint_target)
        with builder.mark('step1'):
            _rows_into(builder, False, cfg, x, c2, c1, scratch, wc, wc)
        with builder.mark('del_c2'):
            builder.del_lookup(table=tab_c2, address=address, target=c2)
        with builder.mark('step2'):
            _rows_into(builder, False, cfg, x, c1, p, scratch, wc, wp)
        with builder.mark('denormalize'):
            nr._denormalize(builder, False, flags, y1)
    if compute_only:
        return builder.finish_circuit()
    nr._phase_by_bits(builder, y1)
    with builder.mark('uncompute_y1'):
        with builder.mark('denormalize'):
            nr._denormalize(builder, True, flags, y1)
        with builder.mark('step2'):
            _rows_into(builder, True, cfg, x, c1, p, scratch, wc, wp)
        with builder.mark('relookup_c2'):
            builder.init_lookup(table=tab_c2, address=address, target=c2)
        with builder.mark('step1'):
            _rows_into(builder, True, cfg, x, c2, c1, scratch, wc, wc)
        with builder.mark('del_all'):
            builder.del_lookup(table=full, address=address, target=joint_target)
        with builder.mark('normalize'):
            nr._normalize(builder, True, x, flags)
    for reg in (flags, c2, c1, y1, scratch):
        builder.free(reg)
    return builder.finish_circuit()


def check_against_model(cfg: mp.Config, table: dict[int, list[int]], batch: int = 4096) -> None:
    """Simulates the compute-only prefix for every x and compares y1 with model_pwquad."""
    circuit = build(cfg, table, compute_only=True)
    regs = circuit.register_data
    x_reg, y1_reg = km.array(list(regs['x'])), km.array(list(regs['y1']))
    F = mp.F_OF(cfg.wp)
    xs = np.arange(1, mp.MAX_X + 1, dtype=np.int64)
    s = (mp.WIDTH - (np.floor(np.log2(xs)).astype(np.int64) + 1)) // 2
    x_norm = xs << (2 * s)
    addr = x_norm >> (mp.WIDTH - cfg.k)
    p = np.zeros(len(xs), dtype=np.int64)
    for a, coefs in table.items():
        sel = addr == a
        if sel.any():
            p[sel] = cfg.evaluate(mp.delta_bits(x_norm[sel], cfg.k, cfg.d), coefs)
    y1_model = (p.astype(np.uint64) << s.astype(np.uint64)) & np.uint64(2**F - 1)

    n = len(xs)
    pad = (-n) % batch
    xs_pad = np.concatenate([xs, np.ones(pad, dtype=np.int64)]).astype(np.uint64)
    y1_pad = np.concatenate([y1_model, np.repeat(y1_model[:1], pad)])
    sim = km.Simulator(batch_size=batch, seed=1)
    t0 = time.time()
    mism = 0
    for start in range(0, len(xs_pad), batch):
        xb = xs_pad[start:start + batch]
        sim.clear_for_shot()
        sim.write_across_shots(x_reg, (xb[None, :] >> np.arange(mp.WIDTH, dtype=np.uint64)[:, None]) & 1 == 1)
        sim.do(circuit)
        got = (sim.read_across_shots(y1_reg).astype(np.uint64) << np.arange(F, dtype=np.uint64)[:, None]).sum(axis=0)
        bad = np.flatnonzero(got != y1_pad[start:start + batch])
        if len(bad):
            mism += len(bad)
            if mism <= 5:
                i = bad[0]
                print(f'MISMATCH x={int(xb[i])}: circuit y1={int(got[i])} model y1={int(y1_pad[start + i])}')
        # x must still hold x_norm (compute-only prefix leaves it normalised): check the shift too.
        xn = (sim.read_across_shots(x_reg).astype(np.uint64) << np.arange(mp.WIDTH, dtype=np.uint64)[:, None]).sum(axis=0)
        exp = xb << ((mp.WIDTH - (np.floor(np.log2(xb.astype(np.float64))).astype(np.uint64) + 1)) // 2 * 2)
        if (xn != exp).any():
            mism += 1
            print('MISMATCH in x_norm at batch', start)
    print(f'model check: {n} x simulated in {time.time() - t0:.0f}s, mismatches={mism}')
    if mism:
        raise SystemExit('circuit does not reproduce model_pwquad; fix the build before verifying')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('output', nargs='?', help='.kmx or .kmb path to write')
    ap.add_argument('--table', default=DEFAULT_TABLE)
    ap.add_argument('--check', action='store_true', help='simulate every x and compare y1 with the model')
    args = ap.parse_args()
    cfg, table = load_table(args.table)
    circuit = build(cfg, table)
    st = circuit.operation_statistics()
    print(f'{cfg.name()}: num_qubits={circuit.num_qubits} toffoli_like={st.num_toffoli_like} '
          f't_like={st.num_t_like} rotations={st.num_z_pow(min_clifford_hierarchy_level=4)} '
          f'ops={len(circuit)} reaction_depth={circuit.reaction_depth()} (static counts, not a score)')
    if args.check:
        check_against_model(cfg, table)
    if args.output:
        circuit.to_file(args.output, format='kmb' if args.output.endswith('.kmb') else 'kmx')
        print('wrote', args.output)


if __name__ == '__main__':
    main()
