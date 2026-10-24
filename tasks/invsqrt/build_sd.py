"""Generator for the signed-digit Horner invsqrt circuit (classical model: model_sd.py, notes §10).

Our own implementation of ideas published on Kaggle by Sumit Chongder, anazemcev, Rudra Bhaskar,
kirilltym and Alex Chilton (credited in tasks/invsqrt/notes.md §10). Structure:

  normalise        'pairs': reference lane rotations + unary->binary flags (as build_pwquad.py)
                   'binary': five stages (shift 16/8/4/2/1); flag_k = "top 2^k bits are zero",
                             computed by an AND chain; flags = shift count lz in binary, f0 = parity.
  lookup           one 256-entry QROM XORs (c2, R, [coarse], L|Y, [H]) straight into the registers.
  [static high]    with the high/low split, H is shifted by s, phased and erased (del_lookup) first.
  inner rows       R -= sig_i * rd(c2, sh2_i): one Gidney adder per digit, sign by CX-conjugation
                   from digit qubit x[i], rounding carry-in = the operand bit below the cut.
  [merge]          with the derivative split, coarse += resid[RB-1]; derivative = resid[:RB-1] ++ coarse.
  outer rows       L -= sig_i * rd(R, sh1_i), same adder.
  phase            word = L[GY:] ++ 10 clean qubits, shifted by s (controlled shifts 1,2,4,8), one Z_POW
                   per qubit, unshifted; then exact mirror uncompute, del_lookup, un-normalise.
Options: --no-top-carry (ancilla for the last carry), --no-borrow (do not use x[21] as a carry wire),
--no-temp-moves (CSWAP for every shifter move), --no-hmr-flags (recompute flags instead of measuring).

Usage (inside Docker):
    python tasks/invsqrt/build_sd.py out.kmx --table tasks/invsqrt/sd_tables/<cfg>.json [--check]
`--check` simulates the FULL circuit for every x and compares the tracked phase, restored x and clean
scratch with the model (the same test the verifier runs, without its seeds).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'data', 'verifiers', 'invsqrt'))
sys.path.insert(0, HERE)
import kickmix as km
import nr_reference_circuit as nr  # reference normalisation for the 'pairs' variant (read-only)
import model_sd as ms
import verify_submission_invsqrt as verifier  # read-only: half_turns_to_units for --check
from build_pwquad import _flags_to_binary  # thermometer flags -> binary s (our code)


# ---------------------------------------------------------------------------------------------
# Building blocks.

class Adder:
    """Gidney ripple adder: target += operand + cin (mod 2^n), carries measured away.

    One AND per carry position (n - 1 Toffolis for an n-bit target). `top_reuse` computes the
    last carry straight into the target MSB with a CCX instead of an ancilla. `spare` lists known
    |0> qubits used as carry ancillas before allocating fresh ones (returned clean)."""

    def __init__(self, b: km.CircuitBuilder, top_reuse: bool = True, spare=()):
        self.b, self.top_reuse, self.spare = b, top_reuse, list(spare)

    def add(self, operand, target, cin=None, subtract: bool = False) -> None:
        b = self.b
        a, t = list(operand), list(target)
        n, m = len(t), len(a)
        assert m <= n
        if n == 0:
            return
        if subtract:
            b.x(km.array(t))
        if n == 1:
            if m:
                b.cx(a[0], t[0])
            if cin is not None:
                b.cx(cin, t[0])
            if subtract:
                b.x(km.array(t))
            return
        carry: list = [cin] + [None] * (n - 1)   # carry[i] = c_i (None = identically zero)
        fresh: list = []
        spare = list(self.spare)

        def new_carry():
            if spare:
                return spare.pop()
            q = b.alloc_qubits(1)[0]
            fresh.append(q)
            return q

        reuse = self.top_reuse and n >= 2
        for i in range(n - 1):
            ci, ai = carry[i], (a[i] if i < m else None)
            last = i == n - 2
            if ci is None and ai is None:
                continue  # c_{i+1} = 0
            if last and reuse:
                out = t[n - 1]
                and_op = b.ccx
            else:
                out = new_carry()
                carry[i + 1] = out
                and_op = b.init_and
            if ci is None:
                and_op(ai, t[i], out)
            elif ai is not None:
                b.cx(ci, ai)
                b.cx(ci, t[i])
                and_op(ai, t[i], out)
                b.cx(ci, out)
            else:
                and_op(ci, t[i], out)
        if m == n:
            b.cx(a[n - 1], t[n - 1])
        if not reuse and carry[n - 1] is not None:
            b.cx(carry[n - 1], t[n - 1])
        for i in reversed(range(n - 1)):
            ci, ai = carry[i], (a[i] if i < m else None)
            last = i == n - 2
            if ci is None and ai is None:
                continue
            uncompute = not (last and reuse)
            if ci is None:
                if uncompute:
                    b.del_and(ai, t[i], carry[i + 1])
                b.cx(ai, t[i])
            elif ai is not None:
                if uncompute:
                    b.cx(ci, carry[i + 1])
                    b.del_and(ai, t[i], carry[i + 1])
                b.cx(ci, ai)
                b.cx(ai, t[i])
            else:
                if uncompute:
                    b.del_and(ci, t[i], carry[i + 1])
                b.cx(ci, t[i])
        for q in reversed(fresh):
            b.free(q)
        if subtract:
            b.x(km.array(t))


def and_chain_into(b, controls, target) -> None:
    """target ^= AND(controls) with an init_and chain (len-2 ancillas, measured away)."""
    c = list(controls)
    if len(c) == 1:
        b.cx(c[0], target)
        return
    if len(c) == 2:
        b.ccx(c[0], c[1], target)
        return
    anc = b.alloc_qubits(len(c) - 2)
    b.init_and(c[0], c[1], anc[0])
    for j in range(2, len(c) - 1):
        b.init_and(anc[j - 2], c[j], anc[j - 1])
    b.ccx(anc[-1], c[-1], target)
    for j in reversed(range(2, len(c) - 1)):
        b.del_and(anc[j - 2], c[j], anc[j - 1])
    b.del_and(c[0], c[1], anc[0])
    b.free(anc)


def phase_if_all(b, controls) -> None:
    """Phase (-1)^AND(controls): CZ at the end of an AND chain."""
    c = list(controls)
    if len(c) == 1:
        b.z(c[0])
        return
    if len(c) == 2:
        b.cz(c[0], c[1])
        return
    anc = b.alloc_qubits(len(c) - 2)
    b.init_and(c[0], c[1], anc[0])
    for j in range(2, len(c) - 1):
        b.init_and(anc[j - 2], c[j], anc[j - 1])
    b.cz(anc[-1], c[-1])
    for j in reversed(range(2, len(c) - 1)):
        b.del_and(anc[j - 2], c[j], anc[j - 1])
    b.del_and(c[0], c[1], anc[0])
    b.free(anc)


BIN_STAGES = ((4, 16), (3, 8), (2, 4), (1, 2), (0, 1))  # (flag index, shift)


def normalise_binary(b, x, f) -> None:
    """x := x << lz in five stages; f[k] = 1 iff the top 2^k bits were zero at stage k (f = lz in binary)."""
    for k, d in BIN_STAGES:
        top = x[ms.WIDTH - d:]
        if d == 1:
            b.cx(x[ms.WIDTH - 1], f[0])
            b.x(f[0])
        else:
            b.x(top)
            and_chain_into(b, list(top), f[k])
            b.x(top)
        for i in reversed(range(ms.WIDTH - d)):
            b.cswap(f[k], x[i], x[i + d])


def unnormalise_binary(b, x, f, hmr_flags: bool = True) -> None:
    for k, d in reversed(BIN_STAGES):
        for i in range(ms.WIDTH - d):
            b.cswap(f[k], x[i], x[i + d])
        top = x[ms.WIDTH - d:]
        if d == 1:
            b.x(f[0])
            b.cx(x[ms.WIDTH - 1], f[0])
        elif hmr_flags:
            # Measure the flag in the X basis; on outcome 1 the kickback (-1)^flag is undone by
            # recomputing the AND as a phase, only on those shots.
            m = b.alloc_bits(1)[0]
            b.hmr(f[k], m)
            b.push_condition(m)
            b.x(top)
            phase_if_all(b, list(top))
            b.x(top)
            b.pop_condition()
            b.free(m)
        else:
            b.x(top)
            and_chain_into(b, list(top), f[k])
            b.x(top)


def shift_word(b, sbits, word, occupied: int, inverse: bool, temp_moves: bool = True) -> None:
    """word := word << s (s = sum sbits[j] 2^j), as controlled shifts by 1, 2, 4, 8 on the occupied
    prefix; the top len(word) - occupied qubits must be |0>. A move whose destination is still |0>
    for every s is a temp-AND + CNOT (free on the way back) instead of a CSWAP."""
    stages = []
    occ = occupied
    for j, sq in enumerate(sbits):
        d = 2**j
        stages.append((sq, d, occ))
        occ = min(occ + d, len(word))
    for sq, d, occ in (reversed(stages) if inverse else stages):
        idx = range(occ - 1, -1, -1) if not inverse else range(occ)
        for i in idx:
            if i + d >= len(word):
                continue
            src, dst = word[i], word[i + d]
            if temp_moves and i + d >= occ:
                if inverse:
                    b.cx(dst, src)
                    b.del_and(sq, src, dst)
                else:
                    b.init_and(sq, src, dst)
                    b.cx(dst, src)
            else:
                b.cswap(sq, src, dst)


def phase_word(b, cfg: ms.Config, sbits, reg, low_exp: int, temp_moves: bool, ids=None, tag='') -> None:
    """Shift reg (with 10 clean qubits appended) by s, Z_POW each qubit, unshift. Qubit j of reg
    has weight 2^(low_exp + j) turns before the shift."""
    with b.mark('phase' + tag):
        ext = ids.alloc(ms.N_P, 'ext' + tag) if ids else b.alloc_qubits(ms.N_P)
        word = km.array(list(reg) + list(ext))
        shift_word(b, sbits, word, len(reg), False, temp_moves)
        for j in range(len(word)):
            e = low_exp + j
            if e < 0:
                b.z_pow(word[j], Fraction(2) ** (e + 1))  # 2^e turns
        shift_word(b, sbits, word, len(reg), True, temp_moves)
        if ids:
            ids.free('ext' + tag, ext)
        else:
            b.free(ext)


class _Ids:
    """Allocation tracker (asserts live registers never share ids; records peak)."""

    def __init__(self, b):
        self.b, self.live, self.peak, self.history = b, {}, 0, []

    def alloc(self, n, name):
        arr = self.b.alloc_qubits(n)
        ids = [q.id for q in arr]
        for other, oids in self.live.items():
            assert not set(ids) & set(oids), (name, other)
        self.live[name] = ids
        self.history.append((name, ids))
        self.peak = max(self.peak, sum(map(len, self.live.values())))
        return arr

    def free(self, name, arr):
        assert self.live.pop(name) == [q.id for q in arr]
        self.b.free(arr)


class Built:
    def __init__(self, circuit, cfg, tables, lay, ids):
        self.circuit, self.cfg, self.tables, self.lay, self.ids = circuit, cfg, tables, lay, ids


# ---------------------------------------------------------------------------------------------
# The circuit.

def load(path: str):
    j = json.load(open(path))
    c = j['config']
    cfg = ms.Config(c['norm'], c['F2'], c['F1'], c['FY'], c['GY'], tuple(c['drop2']), c['LB'], c['S'], c['RB'])
    tables = {int(k): tuple(v) for k, v in j['tables'].items()}
    lay = {k: ({int(a): b for a, b in v.items()} if isinstance(v, dict) else v) for k, v in j['layout'].items()}
    return cfg, tables, lay


def build(cfg: ms.Config, tables: dict, lay: dict, *, top_carry=True, borrow=True, temp_moves=True,
          hmr_flags=True) -> Built:
    b = km.CircuitBuilder()
    ids = _Ids(b)
    x = b.create_quantum_register(ms.WIDTH, name='x')
    ids.live['x'] = [q.id for q in x]
    w2, w1 = lay['w2'], lay['w1']
    split, dsplit = cfg.LB is not None, cfg.RB is not None
    binary = cfg.norm == 'binary'
    borrow = borrow and binary  # x[21] is 1 after binary normalisation only

    # --- normalise -------------------------------------------------------------------------
    if binary:
        f = ids.alloc(5, 'flags')
        with b.mark('normalise'):
            normalise_binary(b, x, f)
        sbits = [f[1], f[2], f[3], f[4]]          # s = lz >> 1
        address = km.array(list(x[DB_LO:ms.WIDTH - 1]) + [f[0]])
    else:
        f = ids.alloc(ms.N_P, 'flags')
        with b.mark('normalise'):
            nr._normalize(b, False, x, f)
        sbits = _flags_to_binary(b, False, f)
        address = x[DB_LO:]
    digits = x[:ms.DB]

    # --- registers and lookup --------------------------------------------------------------
    c2 = ids.alloc(w2, 'c2')
    if dsplit:
        resid = ids.alloc(cfg.RB, 'resid')
        coarse = ids.alloc(lay['wco'], 'coarse')
        rreg = km.array(list(resid[:cfg.RB - 1]) + list(coarse))   # derivative after the merge
    else:
        rreg = ids.alloc(w1, 'R')
    if split:
        low = ids.alloc(cfg.LB, 'L')
        high = ids.alloc(lay['whi'], 'H')
    else:
        low = ids.alloc(lay['wy'], 'Y')

    fields = [('c2', c2), ('R', resid if dsplit else rreg)] + ([('coarse', coarse)] if dsplit else []) \
        + [('L', low)] + ([('H', high)] if split else [])
    def entry(seg, names):
        K2, K1, K0 = tables[seg]
        vals = dict(c2=K2, R=(lay['resid'][seg] if dsplit else K1), coarse=(lay['coarse'][seg] if dsplit else 0),
                    L=(lay['lo'][seg] if split else K0), H=(lay['hi'][seg] if split else 0))
        v, off = 0, 0
        for name, reg in fields:
            if name in names:
                v |= vals[name] << off
                off += len(reg)
        return v
    def table_for(names):
        tab = [0] * 256
        for seg in tables:
            tab[seg] = entry(seg, names)
        return tab
    def target_for(names):
        return km.array([q for name, reg in fields if name in names for q in reg])
    all_names = [n for n, _ in fields]
    with b.mark('lookup'):
        b.init_lookup(table=table_for(all_names), address=address, target=target_for(all_names))

    spare = []
    if borrow:
        b.x(x[ms.WIDTH - 1])      # always 1 after normalisation: a clean wire until we flip it back
        spare = [x[ms.WIDTH - 1]]
    adder = Adder(b, top_reuse=top_carry, spare=spare)

    # --- static high word: phase and erase before the arithmetic ---------------------------
    if split:
        phase_word(b, cfg, sbits, high, cfg.S - cfg.FY - cfg.SB, temp_moves, ids, '_high')
        with b.mark('erase_high'):
            b.del_lookup(table=table_for(['H']), address=address, target=high)
        ids.free('H', high)

    # --- rows --------------------------------------------------------------------------------
    inner_target = resid if dsplit else rreg

    def inner_rows(inverse: bool):
        with b.mark('inner_rows'):
            for i in (reversed(lay['rows2']) if inverse else lay['rows2']):
                sh = cfg.sh2(i)
                if sh >= 0:
                    operand, target = c2[sh:], inner_target
                    cin = c2[sh - 1] if sh >= 1 else None
                else:
                    operand, target, cin = c2, inner_target[-sh:], None
                b.cx(digits[i], target)
                adder.add(operand, target, cin, subtract=inverse)
                b.cx(digits[i], target)

    def merge(inverse: bool):
        if dsplit:
            with b.mark('merge'):
                adder.add(resid[cfg.RB - 1:], coarse, None, subtract=inverse)

    def outer_rows(inverse: bool):
        with b.mark('outer_rows'):
            order = range(ms.DB) if inverse else reversed(range(ms.DB))
            for i in order:
                sh = cfg.sh1(i)
                assert 0 <= sh < len(rreg), (i, sh)
                b.cx(digits[i], low)
                adder.add(rreg[sh:], low, rreg[sh - 1] if sh >= 1 else None, subtract=inverse)
                b.cx(digits[i], low)

    inner_rows(False)
    merge(False)
    outer_rows(False)
    phase_word(b, cfg, sbits, low[cfg.GY:], cfg.GY - cfg.FY - cfg.SB, temp_moves, ids, '_low')
    outer_rows(True)
    merge(True)
    inner_rows(True)

    if borrow:
        b.x(x[ms.WIDTH - 1])
    rest = [n for n in all_names if n != 'H']
    with b.mark('unlookup'):
        b.del_lookup(table=table_for(rest), address=address, target=target_for(rest))
    alloc_name = dict(c2='c2', R=('resid' if dsplit else 'R'), coarse='coarse', L=('L' if split else 'Y'))
    for name, reg in fields:
        if name != 'H':
            ids.free(alloc_name[name], reg)
    if binary:
        with b.mark('unnormalise'):
            unnormalise_binary(b, x, f, hmr_flags)
    else:
        _flags_to_binary(b, True, f)
        with b.mark('unnormalise'):
            nr._normalize(b, True, x, f)
    ids.free('flags', f)
    assert ids.live.keys() == {'x'}, ids.live
    return Built(b.finish_circuit(), cfg, tables, lay, ids)


DB_LO = ms.DB  # first address bit


# ---------------------------------------------------------------------------------------------
# Full-circuit check against the model (phase, restored x, clean scratch), every x.

def check_against_model(built: Built, batch: int = 4096) -> None:
    cfg, circuit = built.cfg, built.circuit
    regs = circuit.register_data
    x_reg = km.array(list(regs['x']))
    scratch = km.array([km.q(i) for i in range(circuit.num_qubits) if km.q(i) not in set(x_reg)])
    xs = np.arange(1, ms.MAX_X + 1, dtype=np.int64)
    word, _ = ms.phase_words(cfg, built.tables, built.lay, xs)
    expected = word << np.uint64(64 - cfg.F)
    n = len(xs)
    pad = (-n) % batch
    xs_pad = np.concatenate([xs, np.ones(pad, dtype=np.int64)]).astype(np.uint64)
    exp_pad = np.concatenate([expected, np.repeat(expected[:1], pad)])
    sim = km.Simulator(batch_size=batch, seed=7, counting_operations=True)
    t0 = time.time()
    bad_phase = bad_x = bad_scratch = 0
    for start in range(0, len(xs_pad), batch):
        xb = xs_pad[start:start + batch]
        sim.clear_for_shot()
        sim.write_across_shots(x_reg, verifier.to_bits(xb, ms.WIDTH))
        sim.do(circuit)
        got_x = verifier.from_bits(sim.read_across_shots(x_reg))
        bad_x += int((got_x != xb).sum())
        if len(scratch):
            bad_scratch += int(sim.read_across_shots(scratch).any(axis=1).sum())
        theta = np.array([verifier.half_turns_to_units(sim.read_shot_phase(i)) for i in range(batch)], dtype=np.uint64)
        mism = np.flatnonzero(theta != exp_pad[start:start + batch])
        if len(mism) and bad_phase == 0:
            i = mism[0]
            print(f'first phase mismatch x={int(xb[i])}: circuit {int(theta[i])} model {int(exp_pad[start + i])}')
        bad_phase += len(mism)
    stats = sim.operation_statistics()
    print(f'model check: {n} x in {time.time() - t0:.0f}s; phase mismatches={bad_phase}, x not restored={bad_x}, '
          f'dirty scratch={bad_scratch}; mean executed Toffolis/shot={stats.num_toffoli_like / len(xs_pad):.2f}')
    if bad_phase or bad_x or bad_scratch:
        raise SystemExit('circuit disagrees with the model')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('output', nargs='?')
    ap.add_argument('--table', required=True)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--no-top-carry', action='store_true')
    ap.add_argument('--no-borrow', action='store_true')
    ap.add_argument('--no-temp-moves', action='store_true')
    ap.add_argument('--no-hmr-flags', action='store_true')
    args = ap.parse_args()
    cfg, tables, lay = load(args.table)
    built = build(cfg, tables, lay, top_carry=not args.no_top_carry, borrow=not args.no_borrow,
                  temp_moves=not args.no_temp_moves, hmr_flags=not args.no_hmr_flags)
    c = built.circuit
    st = c.operation_statistics()
    print(f'{cfg.name()}: num_qubits={c.num_qubits} toffoli_like(static)={st.num_toffoli_like} t_like={st.num_t_like} '
          f'rotations={st.num_z_pow(min_clifford_hierarchy_level=4)} ops={len(c)} peak_live={built.ids.peak} '
          f'(static counts, not a score)')
    if args.check:
        check_against_model(built)
    if args.output:
        c.to_file(args.output, format='kmb' if args.output.endswith('.kmb') else 'kmx')
        print('wrote', args.output)


if __name__ == '__main__':
    main()
