"""Per-stage cost breakdown and live-qubit profile of the Benchmark II reference circuit.

Streams a `.kmx` text file (regenerate it in Docker first, 86 M lines:
    python data/verifiers/dqi/dqi_reference_circuit.py build/dqi_ref.kmx --format kmx
) following the `INFO_TAG "@enter:..."/"@exit:..."` stack of `km.CircuitBuilder.mark`, and
attributes every Toffoli-like gate (CCX, CCZ, CSWAP) and every qubit id touched to the enclosing
marks. Also records each qubit id's first/last mention, so the number of *live* qubits at each
top-level stage boundary can be reported (a qubit is live at a boundary if it was touched before
and is touched again after). Static counts: the reference's only classical conditions are the
`is_error` guards around each Forney step (skipped only when M_j = 0), so these numbers are within
0.01 % of what the verifier executes. Not a verifier: scores come only from scripts/verify.sh.

Usage (inside Docker, repo at /work):
    python tasks/dqi/analyze_reference.py build/dqi_ref.kmx [depth=2]
"""
from __future__ import annotations

import collections
import re
import sys

TOFFOLI_LIKE = ('CCX', 'CCZ', 'CSWAP')
QUBIT_RE = re.compile(r'q(\d+)')


def main() -> None:
    path = sys.argv[1]
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    stack: list[str] = []
    counts: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    entries: collections.Counter = collections.Counter()
    qubits: dict[tuple, set] = collections.defaultdict(set)
    order: list[tuple] = []
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    boundaries: list[tuple[int, str]] = []  # (line index, label) at top-level enter/exit
    n = 0
    with open(path) as f:
        for n, line in enumerate(f):
            sp = line.find(' ')
            op = line[:sp] if sp > 0 else line.rstrip()
            if op == 'INFO_TAG':
                tag = line[line.find('"') + 1:line.rfind('"')]
                if tag.startswith('@enter:'):
                    name = tag[7:]
                    stack.append(name)
                    key = tuple(stack[:depth])
                    entries[key] += 1
                    if len(stack) == 1:
                        boundaries.append((n, f'enter {name}'))
                elif tag.startswith('@exit:'):
                    assert stack and stack[-1] == tag[6:], (stack, tag)
                    if len(stack) == 1:
                        boundaries.append((n, f'exit {stack[-1]}'))
                    stack.pop()
                continue
            if op in ('REGISTER', 'APPEND_TO_REGISTER'):
                continue
            ids = [int(m) for m in QUBIT_RE.findall(line)]
            for q in ids:
                if q not in first:
                    first[q] = n
                last[q] = n
            for d in range(0, min(len(stack), depth) + 1):
                key = tuple(stack[:d])
                if key not in counts:
                    order.append(key)
                counts[key][op] += 1
                qubits[key].update(ids)
    boundaries.append((n + 1, 'end'))

    def toff(c):
        return sum(c[k] for k in TOFFOLI_LIKE)

    total = toff(counts[()])
    print(f'{path}: lines={n + 1} toffoli_like={total} distinct_qubits={len(first)} max_id={max(first)}')
    print(f'{"stage":<52} {"calls":>5} {"toff":>9} {"per call":>9} {"CCX":>8} {"CSWAP":>8} {"CX":>9} {"HMR":>8} {"nq":>5}')
    for key in order:
        c, q = counts[key], qubits[key]
        name = '/'.join(key) or '<total>'
        calls = entries[key] if key else 1
        print(f'{"  " * len(key)}{name:<{52 - 2 * len(key)}} {calls:>5} {toff(c):>9} {toff(c) / max(calls, 1):>9.1f} '
              f'{c["CCX"]:>8} {c["CSWAP"]:>8} {c["CX"]:>9} {c["HMR"]:>8} {len(q):>5}')
    print('\nLive qubits at top-level stage boundaries (touched before and again after):')
    firsts = sorted(first.values())
    lasts = sorted(last.values())
    import bisect
    for pos, label in boundaries:
        started = bisect.bisect_left(firsts, pos)        # qubits first touched before pos
        finished = bisect.bisect_left(lasts, pos)        # qubits last touched before pos
        print(f'  line {pos:>9}  {label:<36} live={started - finished:>5}  seen={started:>5}  retired={finished:>5}')
    # Peak live count over the whole circuit (sweep).
    events = sorted([(v, 1) for v in first.values()] + [(v + 1, -1) for v in last.values()])
    live = peak = 0
    for _, d in events:
        live += d
        peak = max(peak, live)
    print(f'Peak simultaneously live qubit ids: {peak}')


if __name__ == '__main__':
    main()
