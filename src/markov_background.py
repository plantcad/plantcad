"""Genome-wide k-mer Markov background model for correcting PlantCAD zero-shot scores.

PlantCAD tends to over-estimate conservation in repeats/TEs: highly repeated k-mers are easy to
predict from local sequence alone, so a mutation away from the reference looks "deleterious" even
when it is not under selection. A simple fix is to subtract the log-likelihood ratio predicted by a
genome-wide k-th order Markov model, which captures exactly this "predictable from local k-mer
composition" component:

    markov_llr      = log P(alt | context) - log P(ref | context)
    corrected_score = plantcad_llr - markov_llr

The model is a table of genome-wide (k+1)-mer counts, i.e. counts of the next base given the
preceding k bases. Smoothing is applied at query time, so one table supports several smoothing
schemes.

Build a table (once per reference genome):

    python src/markov_background.py \\
        -input-fasta Zm-B73-REFERENCE-NAM-5.0.fa \\
        -output maize_B73_k13.npz \\
        -k 13 -threads 16

Then pass ``-markov-table maize_B73_k13.npz`` to ``src/zero_shot_score.py``.
"""
import argparse
import logging
import os
import time
from multiprocessing import Pool

import numpy as np

NUCLEOTIDES = "ACGT"
_INVALID = 4

# Byte -> 2-bit code lookup (A=0, C=1, G=2, T=3, anything else = 4). Case-insensitive so
# soft-masked FASTA files are handled.
_LUT = np.full(256, _INVALID, dtype=np.uint8)
for _i, _b in enumerate(NUCLEOTIDES):
    _LUT[ord(_b)] = _i
    _LUT[ord(_b.lower())] = _i


def encode(seq):
    """Encode a str/bytes DNA sequence into uint8 codes (0-3, 4 = invalid)."""
    if isinstance(seq, str):
        seq = seq.encode()
    return _LUT[np.frombuffer(seq, dtype=np.uint8)]


# ----------------------------------------------------------------------------------------------
# Building the count table
# ----------------------------------------------------------------------------------------------

def index_fasta(path):
    """Return a list of (name, seq_start_offset, seq_end_offset) for each record in a FASTA file."""
    records = []
    name, start, offset = None, None, 0
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                if name is not None:
                    records.append((name, start, offset))
                name = line[1:].split()[0].decode()
                start = offset + len(line)
            offset += len(line)
    if name is not None:
        records.append((name, start, offset))
    return records


def _kmer_indices(codes, width):
    """Integer index of every valid (no invalid base) k-mer of the given width, in sequence order."""
    n = len(codes) - width + 1
    if n <= 0:
        return np.empty(0, dtype=np.uint32)
    idx = np.zeros(n, dtype=np.uint32)
    bad = np.zeros(n, dtype=bool)
    for j in range(width):
        window = codes[j:j + n]
        idx = (idx << np.uint32(2)) | (window & 3).astype(np.uint32)
        bad |= window == _INVALID
    return idx[~bad]


def _count_record(task):
    path, name, start, end, width = task
    with open(path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)
    codes = encode(raw.translate(None, b"\r\n \t"))
    seq_len = len(codes)
    idx = _kmer_indices(codes, width)
    del codes
    if idx.size == 0:
        return name, seq_len, np.empty(0, np.uint32), np.empty(0, np.uint64)
    counts = np.bincount(idx, minlength=4 ** width)
    nz = np.flatnonzero(counts).astype(np.uint32)
    return name, seq_len, nz, counts[nz].astype(np.uint64)


def reverse_complement_index(width):
    """Permutation mapping each k-mer index to the index of its reverse complement."""
    w = np.arange(4 ** width, dtype=np.uint32)
    rc = np.zeros_like(w)
    for j in range(width):
        rc = (rc << np.uint32(2)) | (np.uint32(3) - ((w >> np.uint32(2 * j)) & np.uint32(3)))
    return rc


def build_counts(fasta, k=13, threads=8, include=None, exclude=None, both_strands=True):
    """Count genome-wide (k+1)-mers. Returns a uint32 array of shape (4**k, 4)."""
    width = k + 1
    if width > 16:
        raise ValueError("k must be <= 15 (indices are stored as uint32)")
    records = index_fasta(fasta)
    if include:
        records = [r for r in records if r[0] in set(include)]
    if exclude:
        records = [r for r in records if r[0] not in set(exclude)]
    if not records:
        raise ValueError("No FASTA records left after include/exclude filtering")
    # Largest records first so the pool stays busy.
    records.sort(key=lambda r: r[2] - r[1], reverse=True)
    tasks = [(fasta, name, s, e, width) for name, s, e in records]

    total = np.zeros(4 ** width, dtype=np.uint64)
    t0 = time.time()
    with Pool(max(1, min(threads, len(tasks)))) as pool:
        for i, (name, seq_len, nz, cnt) in enumerate(pool.imap_unordered(_count_record, tasks), 1):
            total[nz] += cnt
            if seq_len > 1_000_000 or i == len(tasks):
                logging.info(f"[{i}/{len(tasks)}] counted {name} ({seq_len:,} bp, "
                             f"{int(cnt.sum()):,} valid {width}-mers) - {time.time() - t0:.0f}s")

    if both_strands:
        logging.info("Adding reverse-complement strand counts")
        total = total + total[reverse_complement_index(width)]

    if total.max() >= np.iinfo(np.uint32).max:
        logging.warning("Some counts exceed uint32 range; clipping")
        np.minimum(total, np.iinfo(np.uint32).max, out=total)
    return total.astype(np.uint32).reshape(4 ** k, 4)


def save_table(path, counts, k, both_strands, fasta):
    np.savez(path, counts=counts, k=np.int64(k), both_strands=np.bool_(both_strands),
             source=np.str_(os.path.basename(fasta)))


# ----------------------------------------------------------------------------------------------
# Querying the model
# ----------------------------------------------------------------------------------------------

class MarkovBackground:
    """k-th order Markov model P(base | preceding k bases) from a genome-wide count table.

    Smoothing:
      * ``backoff`` (default): interpolated Dirichlet / hierarchical additive smoothing. The
        order-j estimate is shrunk towards the order-(j-1) estimate with strength ``alpha``:
            P_j(b|c) = (n(c, b) + alpha * P_{j-1}(b|c')) / (n(c) + alpha)
        where c' drops the most distant context base. Rare contexts thus fall back to shorter
        contexts instead of to a uniform distribution.
      * ``add``: additive (Laplace/Lidstone) smoothing, P(b|c) = (n(c,b) + alpha) / (n(c) + 4*alpha).

    Contexts that contain N (or are truncated at a chromosome end) use the longest valid suffix,
    i.e. the bases immediately adjacent to the queried position.

    direction:
      * ``left``: P(base | k bases upstream on the + strand).
      * ``both``: average of the + strand model (upstream context) and the - strand model
        (reverse complement of the downstream context). Symmetric like PlantCAD itself; meaningful
        when the table was built with both strands.
    """

    def __init__(self, path, smoothing="backoff", alpha=None, direction="left"):
        if smoothing not in ("backoff", "add"):
            raise ValueError(f"Unknown smoothing: {smoothing}")
        if direction not in ("left", "both"):
            raise ValueError(f"Unknown direction: {direction}")
        data = np.load(path)
        self.k = int(data["k"])
        self.both_strands = bool(data["both_strands"])
        self.smoothing = smoothing
        self.alpha = float(alpha if alpha is not None else (1.0 if smoothing == "backoff" else 0.5))
        self.direction = direction
        if direction == "both" and not self.both_strands:
            logging.warning("direction='both' with a single-strand table: the downstream model is "
                            "estimated from + strand counts only.")
        top = data["counts"]
        # levels[j]: counts of (context of length j, next base), shape (4**j, 4)
        self.levels = [None] * (self.k + 1)
        self.levels[self.k] = top
        cur = top.astype(np.int64)
        for j in range(self.k - 1, -1, -1):
            cur = cur.reshape(4, 4 ** j, 4).sum(axis=0)
            self.levels[j] = cur
        logging.info(f"Loaded Markov background: k={self.k}, both_strands={self.both_strands}, "
                     f"smoothing={self.smoothing}, alpha={self.alpha}, direction={self.direction}")

    def probs_from_index(self, ctx_idx, order=None):
        """Smoothed P(next base | context) for integer context indices.

        ctx_idx: int array of context indices over the last ``order`` bases (top order = k).
        order:   optional int array (same length) of usable context lengths (0..k).
        Returns float64 array (n, 4).
        """
        ctx_idx = np.asarray(ctx_idx, dtype=np.int64)
        n = len(ctx_idx)
        order = np.full(n, self.k, dtype=np.int64) if order is None else np.asarray(order, np.int64)
        a = self.alpha

        if self.smoothing == "add":
            probs = np.empty((n, 4), dtype=np.float64)
            for j in np.unique(order):
                sel = order == j
                cnt = self.levels[j][ctx_idx[sel] % (4 ** j)].astype(np.float64)
                probs[sel] = (cnt + a) / (cnt.sum(axis=1, keepdims=True) + 4 * a)
            return probs

        base = self.levels[0][0].astype(np.float64)
        probs = np.tile((base + 1.0) / (base.sum() + 4.0), (n, 1))
        for j in range(1, self.k + 1):
            sel = order >= j
            if not sel.any():
                break
            cnt = self.levels[j][ctx_idx[sel] % (4 ** j)].astype(np.float64)
            probs[sel] = (cnt + a * probs[sel]) / (cnt.sum(axis=1, keepdims=True) + a)
        return probs

    def _context_index(self, codes):
        """codes: (n, k) uint8 matrix, last column adjacent to the queried base."""
        k = self.k
        bad = codes == _INVALID
        has_bad = bad.any(axis=1)
        last_bad = k - 1 - np.argmax(bad[:, ::-1], axis=1)
        order = np.where(has_bad, k - 1 - last_bad, k)
        idx = np.zeros(len(codes), dtype=np.int64)
        for j in range(k):
            idx = (idx << 2) | (codes[:, j] & 3)
        return idx, order

    def _encode_contexts(self, contexts):
        k = self.k
        padded = [c[-k:].rjust(k, "N") if len(c) < k else c[-k:] for c in contexts]
        return encode("".join(padded)).reshape(len(padded), k)

    def log_probs(self, left_contexts, right_contexts=None):
        """log P(A/C/G/T at the queried position), shape (n, 4).

        left_contexts:  sequences immediately upstream of the queried base (at least k bp; the last
                        character is adjacent to the base). Shorter or N-containing contexts back off.
        right_contexts: sequences immediately downstream (first character adjacent). Required when
                        direction='both'.
        """
        codes = self._encode_contexts(left_contexts)
        idx, order = self._context_index(codes)
        logp = np.log(self.probs_from_index(idx, order))
        if self.direction == "both":
            if right_contexts is None:
                raise ValueError("right_contexts are required when direction='both'")
            # Minus strand: upstream context is the reverse complement of the downstream sequence,
            # and the predicted base is the complement (A<->T, C<->G = reversed ACGT order).
            rc_codes = np.ascontiguousarray(self._encode_rc(right_contexts))
            ridx, rorder = self._context_index(rc_codes)
            logp_rc = np.log(self.probs_from_index(ridx, rorder))[:, ::-1]
            logp = 0.5 * (logp + logp_rc)
        return logp

    def _encode_rc(self, right_contexts):
        k = self.k
        padded = [c[:k].ljust(k, "N") for c in right_contexts]
        codes = encode("".join(padded)).reshape(len(padded), k)
        rc = np.where(codes == _INVALID, _INVALID, 3 - codes).astype(np.uint8)
        return rc[:, ::-1]

    def llr(self, refs, alts, left_contexts, right_contexts=None):
        """Markov log-likelihood ratio log P(alt) - log P(ref) for each SNV."""
        logp = self.log_probs(left_contexts, right_contexts)
        r = np.array([NUCLEOTIDES.index(x.upper()) for x in refs])
        a = np.array([NUCLEOTIDES.index(x.upper()) for x in alts])
        rows = np.arange(len(r))
        return logp[rows, a] - logp[rows, r]


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a genome-wide (k+1)-mer count table for the PlantCAD Markov background "
                    "correction.")
    p.add_argument("-input-fasta", dest="inputFasta", required=True, help="Reference genome FASTA (uncompressed).")
    p.add_argument("-output", required=True, help="Output .npz table.")
    p.add_argument("-k", type=int, default=13, help="Markov order, i.e. context length (default: 13).")
    p.add_argument("-threads", type=int, default=8, help="Worker processes (default: 8).")
    p.add_argument("-single-strand", dest="singleStrand", action="store_true",
                   help="Count + strand only (default: count both strands).")
    p.add_argument("-include", nargs="+", default=None, help="Only count these FASTA records.")
    p.add_argument("-exclude", nargs="+", default=None,
                   help="Skip these FASTA records (e.g. organelle genomes).")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    args = parse_args()
    if args.inputFasta.endswith(".gz"):
        raise SystemExit("Please provide an uncompressed FASTA (records are read in parallel by offset).")
    both = not args.singleStrand
    counts = build_counts(args.inputFasta, k=args.k, threads=args.threads,
                          include=args.include, exclude=args.exclude, both_strands=both)
    save_table(args.output, counts, args.k, both, args.inputFasta)
    logging.info(f"Saved {args.k}-th order count table ({counts.nbytes / 1e9:.1f} GB) to {args.output}")


if __name__ == "__main__":
    main()
