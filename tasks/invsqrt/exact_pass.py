"""Exact-domain post-pass for kickmix circuits with an enumerable input domain (idea credited to
Artem / anazemcev's "exact compilation" on Kaggle; implementation ours).

Every qubit of a kickmix circuit is, at every point, a deterministic function of the input basis
state (measurements only reset qubits and feed classical condition bits). We evaluate every wire
over ALL legal inputs x = 1..3,133,452 (bit-packed, one uint64 word per 64 inputs) and, at each
Toffoli-like gate, replace it by cheaper gates whenever its action is provably identical on the
whole domain:

  CCX a b t    d = a&b:  d == 0 -> drop | d == 1 -> X t | d == a -> CX a t | d == b -> CX b t
                         d == w for another live wire w -> CX w t   (d == ~w -> X t; CX w t)
  CSWAP c a b  d = c&(a^b): d == 0 -> drop | c == 1 -> SWAP a b | d == c -> CX c a; CX c b
                            d == w -> CX w a; CX w b
  CCZ a b c    any control == 0 -> drop | a == 1 -> CZ b c | ...

Measurement-based uncomputations (HMR + conditioned CZ/CCZ) stay correct: they fix the kickback of
the *function* the wire holds, which the rewrite leaves unchanged. Blocks under PUSH_CONDITION are
evaluated as if they execute; the pass asserts every such block restores all wire values (phase
fixups only), otherwise it aborts. Value-changing gates with a per-op `if` are rejected.

Usage (inside Docker):
    python tasks/invsqrt/exact_pass.py in.kmx out.kmx [--table sd_tables/<cfg>.json]   # --table: run build_sd's all-x check
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kickmix as km

MAX_X = 3133452
TOFFOLI_LIKE = {'CCX', 'CSWAP', 'CCZ'}
VALUE_GATES = {'X', 'CX', 'CCX', 'SWAP', 'CSWAP', 'R', 'HMR'}


class Domain:
    def __init__(self, n: int = MAX_X):
        self.n = n
        self.nw = (n + 63) // 64
        full = np.ones(self.nw * 64, dtype=bool)
        full[n:] = False
        self.mask = self._pack(full)
        self.zero = np.zeros(self.nw, dtype=np.uint64)

    def _pack(self, bits: np.ndarray) -> np.ndarray:
        return np.packbits(bits, bitorder='little').view(np.uint64)

    def input_bit(self, i: int) -> np.ndarray:
        xs = np.arange(1, self.n + 1, dtype=np.int64)
        bits = np.zeros(self.nw * 64, dtype=bool)
        bits[:self.n] = ((xs >> i) & 1).astype(bool)
        return self._pack(bits)


class Pass:
    def __init__(self, lines: list[str], dom: Domain):
        self.dom = dom
        self.lines = lines
        self.w: dict[str, np.ndarray] = {}
        self.h: dict[str, bytes] = {}
        self.byhash: dict[bytes, set] = {}
        self.stats = dict(dropped=0, to_x=0, to_cx=0, to_swap=0, to_cxpair=0, to_cz=0, kept=0, ccx_in=0, cswap_in=0, ccz_in=0)
        self.out: list[str] = []

    # --- wire store -------------------------------------------------------------------------
    def get(self, q: str) -> np.ndarray:
        if q not in self.w:
            self.set(q, self.dom.zero.copy())
        return self.w[q]

    def set(self, q: str, v: np.ndarray) -> None:
        old = self.h.get(q)
        if old is not None:
            self.byhash[old].discard(q)
        self.w[q] = v
        hv = hashlib.blake2b(v.tobytes(), digest_size=16).digest()
        self.h[q] = hv
        self.byhash.setdefault(hv, set()).add(q)

    def find(self, v: np.ndarray, exclude: set) -> str | None:
        hv = hashlib.blake2b(v.tobytes(), digest_size=16).digest()
        for q in self.byhash.get(hv, ()):
            if q not in exclude and np.array_equal(self.w[q], v):
                return q
        return None

    def is_zero(self, v) -> bool:
        return not v.any()

    def is_one(self, v) -> bool:
        return np.array_equal(v, self.dom.mask)

    # --- gates --------------------------------------------------------------------------------
    def run(self) -> None:
        stack = []
        for raw in self.lines:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            op = parts[0]
            cond = None
            if 'if' in parts:
                k = parts.index('if')
                cond = parts[k + 1]
                parts = parts[:k]
            args = parts[1:]
            if op == 'PUSH_CONDITION':
                stack.append(hashlib.blake2b(b''.join(sorted(self.h.values())), digest_size=16).digest())
                self.out.append(line)
                continue
            if op == 'POP_CONDITION':
                snap = stack.pop()
                now = hashlib.blake2b(b''.join(sorted(self.h.values())), digest_size=16).digest()
                assert snap == now, 'conditioned block changes wire values; pass is unsound here'
                self.out.append(line)
                continue
            if op in VALUE_GATES and cond is not None:
                raise SystemExit(f'unsupported: value-changing gate with per-op condition: {line}')
            if op in ('X',):
                for q in args:
                    self.set(q, self.get(q) ^ self.dom.mask)
                self.out.append(line)
            elif op == 'CX':
                c, t = args
                assert c.startswith('q') and t.startswith('q'), line
                self.set(t, self.get(t) ^ self.get(c))
                self.out.append(line)
            elif op == 'SWAP':
                a, b = args
                va, vb = self.get(a), self.get(b)
                self.set(a, vb)
                self.set(b, va)
                self.out.append(line)
            elif op in ('R',):
                for q in args:
                    self.set(q, self.dom.zero.copy())
                self.out.append(line)
            elif op == 'HMR':
                q = args[0]
                self.set(q, self.dom.zero.copy())
                self.out.append(line)
            elif op == 'CCX':
                self.stats['ccx_in'] += 1
                a, b, t = args
                d = self.get(a) & self.get(b)
                rep = self.simplify_and(d, a, b, exclude={t})
                if rep is None:
                    self.out.append(line)
                    self.stats['kept'] += 1
                elif rep == 'drop':
                    self.stats['dropped'] += 1
                elif rep == 'one':
                    self.out.append(f'X {t}')
                    self.stats['to_x'] += 1
                elif rep[0] == 'wire':
                    self.out.append(f'CX {rep[1]} {t}')
                    self.stats['to_cx'] += 1
                elif rep[0] == 'notwire':
                    self.out.append(f'X {t}')
                    self.out.append(f'CX {rep[1]} {t}')
                    self.stats['to_cx'] += 1
                self.set(t, self.get(t) ^ d)
            elif op == 'CSWAP':
                self.stats['cswap_in'] += 1
                c, a, b = args
                vc, va, vb = self.get(c), self.get(a), self.get(b)
                d = vc & (va ^ vb)
                if self.is_zero(d):
                    self.stats['dropped'] += 1
                elif self.is_one(vc):
                    self.out.append(f'SWAP {a} {b}')
                    self.stats['to_swap'] += 1
                elif np.array_equal(d, vc):
                    self.out.append(f'CX {c} {a}')
                    self.out.append(f'CX {c} {b}')
                    self.stats['to_cxpair'] += 1
                else:
                    w = self.find(d, exclude={a, b})
                    if w is not None:
                        self.out.append(f'CX {w} {a}')
                        self.out.append(f'CX {w} {b}')
                        self.stats['to_cxpair'] += 1
                    else:
                        self.out.append(line)
                        self.stats['kept'] += 1
                self.set(a, va ^ d)
                self.set(b, vb ^ d)
            elif op == 'CCZ':
                self.stats['ccz_in'] += 1
                a, b, c = args
                vals = [self.get(q) for q in (a, b, c)]
                suffix = f' if {cond}' if cond else ''
                if any(self.is_zero(v) for v in vals):
                    self.stats['dropped'] += 1
                else:
                    ones = [q for q, v in zip((a, b, c), vals) if self.is_one(v)]
                    if ones:
                        rest = [q for q in (a, b, c) if q != ones[0]]
                        self.out.append(f'CZ {rest[0]} {rest[1]}{suffix}')
                        self.stats['to_cz'] += 1
                    else:
                        self.out.append(line)
                        self.stats['kept'] += 1
            else:
                # Z, CZ, Z_POW, NEG, BIT_*, INFO_TAG, REGISTER, APPEND_TO_REGISTER, DEBUG_PRINT
                self.out.append(line)

    def simplify_and(self, d, a, b, exclude):
        if self.is_zero(d):
            return 'drop'
        if self.is_one(d):
            return 'one'
        if np.array_equal(d, self.w[a]):
            return ('wire', a)
        if np.array_equal(d, self.w[b]):
            return ('wire', b)
        w = self.find(d, exclude | {a, b})
        if w is not None:
            return ('wire', w)
        w = self.find(d ^ self.dom.mask, exclude | {a, b})
        if w is not None:
            return ('notwire', w)
        return None


def load_lines(path: str) -> tuple[list[str], km.Circuit]:
    c = km.Circuit.from_file(path, format='kmb' if path.endswith('.kmb') else 'kmx')
    buf = io.StringIO()
    c.to_file(buf, format='kmx')
    return buf.getvalue().splitlines(), c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('output')
    ap.add_argument('--table', help='sd_tables JSON of the generator config: run build_sd.check_against_model on the result')
    args = ap.parse_args()
    t0 = time.time()
    lines, cin = load_lines(args.input)
    dom = Domain()
    p = Pass(lines, dom)
    # input register: APPEND_TO_REGISTER qN r0 in bit order
    order = [l.split()[1] for l in lines if l.startswith('APPEND_TO_REGISTER')]
    for i, q in enumerate(order):
        p.set(q, dom.input_bit(i))
    p.run()
    text = '\n'.join(p.out) + '\n'
    cout = km.Circuit(text)
    si, so = cin.operation_statistics(), cout.operation_statistics()
    print(f'exact pass: {time.time() - t0:.0f}s  {p.stats}')
    print(f'static toffoli-like {si.num_toffoli_like} -> {so.num_toffoli_like}, qubits {cin.num_qubits} -> {cout.num_qubits}, '
          f'rotations {si.num_z_pow(min_clifford_hierarchy_level=4)} -> {so.num_z_pow(min_clifford_hierarchy_level=4)} (static counts, not a score)')
    cout.to_file(args.output, format='kmb' if args.output.endswith('.kmb') else 'kmx')
    print('wrote', args.output)
    if args.table:
        import build_sd
        cfg, tables, lay = build_sd.load(args.table)
        build_sd.check_against_model(build_sd.Built(cout, cfg, tables, lay, None))


if __name__ == '__main__':
    main()
