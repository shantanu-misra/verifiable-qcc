"""Exact live-qubit profile of the dqi generator: logs CircuitBuilder.num_allocated_qubits at every alloc/free
together with the current mark path and BM step, then reports the peak, the plateau and the per-phase maxima.
(Static: the builder's allocation count is the number of simultaneously live qubit ids.)

Usage (inside Docker): python tasks/dqi/live_profile.py [--n 70 --degree 12 --m 4095]
"""
from __future__ import annotations

import argparse
import collections

import kickmix as km

import build_bm


class LoggingBuilder:
    """Proxy around km.CircuitBuilder that records (live_count, mark_path, step) on every alloc/free."""

    def __init__(self, b: km.CircuitBuilder):
        self._b = b
        self.stack: list[str] = []
        self.log: list[tuple[int, str]] = []
        self.step = -1

    def __getattr__(self, name):
        return getattr(self._b, name)

    def _rec(self):
        self.log.append((self._b.num_allocated_qubits, '/'.join(self.stack), self.step))

    def alloc_qubits(self, n):
        r = self._b.alloc_qubits(n); self._rec(); return r

    def free(self, x):
        self._b.free(x); self._rec()

    def mark(self, name):
        outer = self

        class Mark:  # re-enterable (the sweep reuses one mark object)
            def __enter__(self):
                if name in ('bm_step', 'bm_step_rev', 'forney_step'):
                    outer.step += 1
                outer.stack.append(name)
                self._inner = outer._b.mark(name); self._inner.__enter__()

            def __exit__(self, *exc):
                self._inner.__exit__(*exc); outer.stack.pop()
                return False
        return Mark()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=70); ap.add_argument('--degree', type=int, default=12); ap.add_argument('--m', type=int, default=4095)
    a = ap.parse_args()
    gf = km.GF2Field(a.degree)
    raw = km.CircuitBuilder(); lb = LoggingBuilder(raw)
    S = [raw.create_quantum_register(a.degree, name=f'S{i}') for i in range(a.n)]
    E = [raw.create_classical_register(a.degree, name=f'E{j}') for j in range(a.m)]
    dec = build_bm.BMDecoder(lb, a.n, gf, a.m)
    # the decoder also calls inner primitives (tower/gfmul) through the proxy, so their allocs are logged too
    dec.build(S, E)
    c = raw.finish_circuit()
    log = lb.log
    peak = max(v for v, _, _ in log)
    print(f'num_qubits={c.num_qubits}  peak live (alloc count)={peak}  samples={len(log)}')
    # phase maxima
    phase_max: dict[str, int] = collections.defaultdict(int)
    for v, path, _ in log:
        top = path.split('/')[0] if path else '<top>'
        phase_max[top] = max(phase_max[top], v)
    for k, v in sorted(phase_max.items(), key=lambda kv: -kv[1]):
        print(f'  phase {k:<22} max live {v}')
    # where the peak sits: mark paths and BM steps reaching >= peak - 12 / - 60
    for slack in (0, 12, 60):
        paths = collections.Counter(p for v, p, _ in log if v >= peak - slack)
        steps = sorted(set(s for v, _, s in log if v >= peak - slack and s >= 0))
        print(f'  >= peak-{slack:<3}: {sum(paths.values())} samples, steps {steps[:4]}..{steps[-4:]} ({len(steps)} steps), '
              f'top paths: {paths.most_common(3)}')
    # per-step maximum in bm_forward
    fwd = collections.defaultdict(int)
    for v, p, s in log:
        if p.startswith('bm_forward') and s >= 0:
            fwd[s] = max(fwd[s], v)
    print('  bm_forward per-step max (every 7th):', [fwd[s] for s in range(0, a.n, 7)])


if __name__ == '__main__':
    main()
