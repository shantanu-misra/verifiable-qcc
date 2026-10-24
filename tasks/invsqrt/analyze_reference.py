"""Per-stage cost breakdown of the Benchmark I reference circuit.

Parses the `.kmx` text of a circuit, follows the `INFO_TAG "@enter:..."/"@exit:..."`
stack emitted by `km.CircuitBuilder.mark`, and attributes every Toffoli-like gate
(CCX, CCZ, CSWAP), every Z_POW, and every qubit id touched to the enclosing marks.

Static counts only (every gate is unconditional in this circuit except the CZ's inside
`del_and`, which are Clifford), so Toffoli counts here equal what the verifier executes.
Not a verifier: scores come only from scripts/verify.sh.

Usage (inside Docker, repo at /work):
    python tasks/invsqrt/analyze_reference.py data/sample_submission/invsqrt.kmb
"""
from __future__ import annotations

import collections
import io
import re
import sys

import kickmix as km

TOFFOLI_LIKE = {'CCX', 'CCZ', 'CSWAP'}
QUBIT_RE = re.compile(r'\bq(\d+)\b')


def load_text(path: str) -> list[str]:
    c = km.Circuit.from_file(path, format='kmb' if path.endswith('.kmb') else 'kmx')
    buf = io.StringIO()
    c.to_file(buf, format='kmx')
    return buf.getvalue().splitlines(), c


def analyse(lines: list[str], depth: int):
    """Returns {path_tuple: Counter(op -> count)} and {path_tuple: set(qubit ids)}.

    `path_tuple` is the mark stack truncated to `depth` entries. Every op is attributed to
    every prefix of its stack (so parents include children)."""
    stack: list[str] = []
    counts: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    qubits: dict[tuple, set] = collections.defaultdict(set)
    order: list[tuple] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        op = line.split()[0]
        if op == 'INFO_TAG':
            tag = line.split('"', 1)[1].rstrip('"')
            if tag.startswith('@enter:'):
                stack.append(tag[len('@enter:'):])
            elif tag.startswith('@exit:'):
                assert stack and stack[-1] == tag[len('@exit:'):], (stack, tag)
                stack.pop()
            continue
        ids = {int(m) for m in QUBIT_RE.findall(line)}
        for d in range(0, min(len(stack), depth) + 1):
            key = tuple(stack[:d])
            if key not in counts:
                order.append(key)
            counts[key][op] += 1
            qubits[key] |= ids
    return counts, qubits, order


def toff(c: collections.Counter) -> int:
    return sum(c[k] for k in TOFFOLI_LIKE)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/sample_submission/invsqrt.kmb'
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    lines, circuit = load_text(path)
    counts, qubits, order = analyse(lines, depth)
    total = toff(counts[()])
    print(f'{path}: num_qubits={circuit.num_qubits} toffoli_like={total} '
          f'ops={len(circuit)} reaction_depth={circuit.reaction_depth()}')
    print(f'{"stage":<60} {"toff":>6} {"CCX":>5} {"CSWAP":>5} {"ZPOW":>4} {"CX":>6} {"nq":>4}  qubit-id range')
    for key in order:
        c, q = counts[key], qubits[key]
        name = '/'.join(key) or '<total>'
        rng = f'{min(q)}..{max(q)}' if q else ''
        print(f'{"  " * len(key)}{name:<{60 - 2 * len(key)}} {toff(c):>6} {c["CCX"]:>5} {c["CSWAP"]:>5} '
              f'{c["Z_POW"]:>4} {c["CX"]:>6} {len(q):>4}  {rng}')
    # Qubit id ownership: which ids appear only inside which top-level stages.
    all_ids = qubits[()]
    print(f'\nDistinct qubit ids: {len(all_ids)} (max id {max(all_ids)})')
    unmentioned = set(range(circuit.num_qubits)) - all_ids
    if unmentioned:
        print(f'Ids counted by num_qubits but never used by a gate: {sorted(unmentioned)}')


if __name__ == '__main__':
    main()
