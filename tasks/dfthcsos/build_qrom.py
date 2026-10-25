"""Two-bank XOR QROM generator for the dfthcsos ANGLE_k family.

Circuit k performs |a>|z> -> |a>|z XOR delta_k[a]> for the A = 76 + R*B addresses, where
delta_k = batch_k XOR batch_{k-1} of the J-word batches of each row of the angle-word table
(the verifier's contract, data/verifiers/dfthcsos/SCORING_README.md).

Design (our own implementation of the "banked" idea reported on the Kaggle forum):
  * The address is split a = 2 g + a0.  One unary iteration over the 11-bit group index g
    (A/2 = 1988 groups, ~1977 Toffolis) XORs row 2g into the word registers and
    row 2g XOR row 2g+1 into a W = J*brot qubit scratch register S.
  * W Toffolis CCX(a0, S_i, word_i) select the odd row when a0 = 1.
  * S is erased with kickmix's measurement-based `del_lookup` (88 Toffolis for 2^11 addresses).
  Per circuit: ~1977 + W + 88 Toffolis and 12 + 2W + 11 qubits, so W <= 52 fits the 128-qubit cap
  (J = 4 at brot 12, J = 5 at brot 10, J = 6 at brot 8, J = 7 at brot 7).  A plain QROM costs
  ~3974 Toffolis per circuit (the baseline).

Usage (inside Docker; writes angle_k_<k>.kmb + submission.npz into OUT_DIR):
    python tasks/dfthcsos/build_qrom.py SUBMISSION.npz OUT_DIR --batch-size 4 [--brot 12] [--check]
`SUBMISSION.npz` must hold `angle_words` (brot-bit integers, (A, 75)) plus the factorization arrays
and `tree_edges`; `--brot` defaults to the smallest width that holds every word.  `--banks 1` builds plain
QROMs (A - 1 Toffolis, n + J brot + n qubits) for tables whose 2W scratch would not fit.  `--check`
simulates the family with the official verifier's `recover_words_from_circuits` (no salt) and
compares the recovered words with the table.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from math import ceil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
import kickmix as km  # noqa: E402


def batch_deltas(words: np.ndarray, batch_size: int) -> np.ndarray:
    """(K, A, J) tables delta_k = batch_k XOR batch_{k-1}; rows zero padded to K*J words."""
    words = np.asarray(words, dtype=np.int64)
    A, num_angles = words.shape
    K = ceil(num_angles / batch_size)
    padded = np.zeros((A, K * batch_size), dtype=np.int64)
    padded[:, :num_angles] = words
    batches = padded.reshape(A, K, batch_size).transpose(1, 0, 2)
    deltas = batches.copy()
    deltas[1:] ^= batches[:-1]
    return deltas


def pack_rows(delta: np.ndarray, brot: int) -> list[int]:
    """Row a -> one W-bit integer, word j at bits [j*brot, (j+1)*brot)."""
    A, J = delta.shape
    return [sum(int(delta[a, j]) << (j * brot) for j in range(J)) for a in range(A)]


def build_plain_circuit(delta: np.ndarray, brot: int) -> km.Circuit:
    """Plain QROM: one unary iteration over all A addresses, CX loads (~A - 1 Toffolis, n ancillas).
    Used when 2W scratch does not fit the 128-qubit cap (J brot > 52), e.g. the small THC tables with J 10."""
    A, J = delta.shape
    W = J * brot
    n = (A - 1).bit_length()
    rows = pack_rows(delta, brot)
    b = km.CircuitBuilder()
    selection = b.create_quantum_register(n, "selection")
    word_regs = [b.create_quantum_register(brot, f"word_{j}") for j in range(J)]
    target = [word_regs[j][i] for j in range(J) for i in range(brot)]
    with b.unary_iteration(selection, range(A)) as it:
        for a, m in it:
            v = rows[a]
            for i in range(W):
                if (v >> i) & 1:
                    b.cx(m, target[i])
    return b.finish_circuit()


def build_angle_circuit(delta: np.ndarray, brot: int, banks: int = 2) -> km.Circuit:
    if banks == 1:
        return build_plain_circuit(delta, brot)
    A, J = delta.shape
    W = J * brot
    n = (A - 1).bit_length()
    rows = pack_rows(delta, brot)
    G = ceil(A / 2)
    e0 = [rows[2 * g] for g in range(G)]
    e01 = [rows[2 * g] ^ (rows[2 * g + 1] if 2 * g + 1 < A else 0) for g in range(G)]

    b = km.CircuitBuilder()
    selection = b.create_quantum_register(n, "selection")
    word_regs = [b.create_quantum_register(brot, f"word_{j}") for j in range(J)]
    target = [word_regs[j][i] for j in range(J) for i in range(brot)]  # bit w of the packed row
    a0, a_hi = selection[0], selection[1:]
    S = b.alloc_qubits(W)

    with b.unary_iteration(a_hi, range(G)) as it:
        for g, m in it:
            v0, v01 = e0[g], e01[g]
            for i in range(W):
                if (v0 >> i) & 1:
                    b.cx(m, target[i])
                if (v01 >> i) & 1:
                    b.cx(m, S[i])
    for i in range(W):
        b.ccx(a0, S[i], target[i])
    b.del_lookup(table=e01 + [0] * ((1 << (n - 1)) - G), address=a_hi, target=S)
    b.free(S)
    return b.finish_circuit()


def build_family(words: np.ndarray, brot: int, batch_size: int, banks: int = 2) -> list[km.Circuit]:
    return [build_angle_circuit(d, brot, banks) for d in batch_deltas(words, batch_size)]


def write_family(circuits, out_dir: str, npz_arrays: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        if name.startswith("angle_k_") and name.endswith((".kmb", ".kmx")):
            os.remove(os.path.join(out_dir, name))
    for k, c in enumerate(circuits):
        c.to_file(os.path.join(out_dir, f"angle_k_{k}.kmb"), format="kmb")  # NOT text: the metric trusts the suffix
    np.savez(os.path.join(out_dir, "submission.npz"), **npz_arrays)


def check_family(out_dir: str, words: np.ndarray, seed: int = 1) -> dict:
    """Official contract check: the verifier's simulation of the family (no salt)."""
    sys.path.insert(0, os.path.join(REPO, "data", "verifiers"))
    from dfthcsos.scoreinfra.circuits import load_circuits, recover_words_from_circuits  # noqa: E402

    circuits = load_circuits(out_dir)
    A, num_angles = words.shape
    t0 = time.time()
    recovered, layout, toffoli = recover_words_from_circuits(circuits, num_vectors=A, num_angles=num_angles, seed=seed)
    ok = np.array_equal(recovered, words)
    return {"words_match": bool(ok), "angles_toffoli": toffoli, "brot": layout.brot, "J": layout.batch_size,
            "qubits": max(c.num_qubits for c in circuits), "seconds": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission", help="submission.npz with angle_words, factorization arrays and tree_edges")
    ap.add_argument("out_dir")
    ap.add_argument("--batch-size", "-J", type=int, required=True)
    ap.add_argument("--brot", type=int, default=None)
    ap.add_argument("--banks", type=int, default=2, choices=(1, 2), help="2: banked QROM (needs 12 + 2W + 11 qubits); 1: plain")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    with np.load(args.submission) as z:
        arrays = {k: z[k] for k in z.files}
    words = arrays["angle_words"].astype(np.int64)
    brot = args.brot or max(1, int(words.max()).bit_length())
    assert words.min() >= 0 and words.max() < (1 << brot), "words do not fit brot bits"
    t0 = time.time()
    circuits = build_family(words, brot, args.batch_size, args.banks)
    stats = [(c.num_qubits, c.operation_statistics().num_toffoli_like, len(c)) for c in circuits]
    print(f"built {len(circuits)} circuits in {time.time()-t0:.1f}s: qubits {max(s[0] for s in stats)}, "
          f"static Toffolis per circuit {[s[1] for s in stats]}, sum {sum(s[1] for s in stats)} per pass")
    write_family(circuits, args.out_dir, arrays)
    print(f"wrote {args.out_dir}/angle_k_0..{len(circuits)-1}.kmb + submission.npz")
    if args.check:
        res = check_family(args.out_dir, words)
        print("check:", res)
        if not res["words_match"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
