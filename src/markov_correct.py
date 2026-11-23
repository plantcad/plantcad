"""Add the Markov background correction to zero-shot scores that were already computed.

The correction needs no GPU and no PlantCAD model, only the reference genome and the k-mer table,
so existing results can be corrected in seconds instead of re-running zero_shot_score.py.

Supported inputs (auto-detected):
  * VCF written by zero_shot_score.py (INFO/plantCAD_zero_shot). Adds INFO/plantCAD_markov_llr and
    INFO/plantCAD_zero_shot_corrected; all other content is kept as is.
  * TSV written by zero_shot_score.py in BED mode. Appends corrected_score(s) and markov_llrs columns.
      - `-aggregation all` output (scores column): exact, per alt allele.
      - `score` column from `-aggregation average`: exact (mean of differences = difference of means).
      - `score` column from `-aggregation max`: needs the ref_prob/alt_probs columns
        (`-output-raw-prob`) to recover the per-allele scores; pass `-aggregation max`.

Usage:
    python src/markov_correct.py -input scored.vcf -input-fasta genome.fa \\
        -markov-table genome_k13.npz -output scored.corrected.vcf
"""
import argparse
import logging
import sys

import numpy as np

from markov_background import NUCLEOTIDES, MarkovBackground, index_fasta


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-input", required=True, help="VCF or BED-mode TSV produced by zero_shot_score.py")
    p.add_argument("-input-fasta", dest="inputFasta", required=True,
                   help="Reference genome FASTA (uncompressed; the one used for scoring)")
    p.add_argument("-markov-table", dest="markovTable", required=True,
                   help="k-mer table built by src/markov_background.py")
    p.add_argument("-output", required=True, help="Output file (same format as the input)")
    p.add_argument("-aggregation", choices=["average", "max"], default="average",
                   help="For BED-mode TSVs with a single `score` column: the -aggregation used when "
                        "scoring (default: average). Ignored for VCF and `scores` (all) files.")
    p.add_argument("-markov-smoothing", dest="markovSmoothing", default="add", choices=["add", "backoff"])
    p.add_argument("-markov-alpha", dest="markovAlpha", default=None, type=float)
    p.add_argument("-markov-direction", dest="markovDirection", default="both", choices=["both", "left"])
    args = p.parse_args()
    if args.markovAlpha is None:
        args.markovAlpha = 2.0 if args.markovSmoothing == "add" else 8.0
    return args


class Genome:
    """Lazy per-chromosome FASTA access (keeps one chromosome in memory at a time)."""

    def __init__(self, path):
        if path.endswith(".gz"):
            sys.exit("Please provide an uncompressed FASTA.")
        self.path = path
        self.index = {name: (s, e) for name, s, e in index_fasta(path)}
        self._name, self._seq = None, None

    def resolve(self, chrom):
        """Match VCF/BED chromosome names to FASTA names, tolerating a missing/extra 'chr' prefix."""
        if chrom in self.index:
            return chrom
        for alt in ("chr" + chrom, chrom[3:] if chrom.startswith("chr") else None):
            if alt and alt in self.index:
                return alt
        return None

    def seq(self, name):
        if name != self._name:
            s, e = self.index[name]
            with open(self.path, "rb") as f:
                f.seek(s)
                self._seq = f.read(e - s).translate(None, b"\r\n \t").decode().upper()
            self._name = name
        return self._seq


def contexts_for(genome, chrom, positions0, k):
    seq = genome.seq(chrom)
    left = [seq[max(0, p - k):p] for p in positions0]
    right = [seq[p + 1:p + 1 + k] for p in positions0]
    refs = [seq[p] for p in positions0]
    return left, right, refs


def fmt(x):
    return "." if x is None else repr(float(x))


# ----------------------------------------------------------------------------------------------
# VCF
# ----------------------------------------------------------------------------------------------

def correct_vcf(args, markov, genome):
    header, records = [], []
    with open(args.input) as f:
        for line in f:
            (header if line.startswith("#") else records).append(line.rstrip("\n"))

    fields = [r.split("\t") for r in records]
    # Group records by chromosome so each chromosome is loaded once.
    logp = [None] * len(fields)
    by_chrom = {}
    for i, fs in enumerate(fields):
        by_chrom.setdefault(fs[0], []).append(i)
    n_missing_chrom, n_ref_mismatch = 0, 0
    for chrom, idxs in by_chrom.items():
        name = genome.resolve(chrom)
        if name is None:
            logging.warning(f"Chromosome {chrom} not in FASTA; {len(idxs)} records left uncorrected")
            n_missing_chrom += len(idxs)
            continue
        pos0 = [int(fields[i][1]) - 1 for i in idxs]
        left, right, refs = contexts_for(genome, name, pos0, markov.k)
        lp = markov.log_probs(left, right)
        for j, i in enumerate(idxs):
            if refs[j] != fields[i][3][:1].upper():
                n_ref_mismatch += 1
            logp[i] = lp[j]
    if n_ref_mismatch:
        logging.warning(f"{n_ref_mismatch} records have a REF that does not match the FASTA "
                        f"(wrong reference genome?)")

    new_info = [
        '##INFO=<ID=plantCAD_markov_llr,Number=A,Type=Float,Description="Genome-wide k-mer Markov '
        'background log P(alt) - log P(ref), one per ALT">',
        '##INFO=<ID=plantCAD_zero_shot_corrected,Number=A,Type=Float,Description="Markov-corrected '
        'PlantCAD zero-shot score (plantCAD_zero_shot - plantCAD_markov_llr), one per ALT">',
    ]
    header = [h for h in header if not any(h.startswith(f"##INFO=<ID={x},") for x in
                                           ("plantCAD_markov_llr", "plantCAD_zero_shot_corrected"))]
    n_corrected = 0
    with open(args.output, "w") as out:
        for h in header[:-1]:
            out.write(h + "\n")
        for h in new_info:
            out.write(h + "\n")
        out.write(header[-1] + "\n")
        for fs, lp in zip(fields, logp):
            info = [kv for kv in fs[7].split(";")
                    if kv and kv != "." and not kv.startswith(("plantCAD_markov_llr=", "plantCAD_zero_shot_corrected="))]
            raw = next((kv.split("=", 1)[1] for kv in info if kv.startswith("plantCAD_zero_shot=")), None)
            if raw is not None and lp is not None and fs[3].upper() in NUCLEOTIDES:
                r = NUCLEOTIDES.index(fs[3].upper())
                m_vals, c_vals = [], []
                for alt, s in zip(fs[4].split(","), raw.split(",")):
                    if alt.upper() in NUCLEOTIDES and s not in (".", ""):
                        m = lp[NUCLEOTIDES.index(alt.upper())] - lp[r]
                        m_vals.append(fmt(m))
                        c_vals.append(fmt(float(s) - m))
                    else:
                        m_vals.append(".")
                        c_vals.append(".")
                info += [f"plantCAD_markov_llr={','.join(m_vals)}",
                         f"plantCAD_zero_shot_corrected={','.join(c_vals)}"]
                n_corrected += 1
            fs[7] = ";".join(info) if info else "."
            out.write("\t".join(fs) + "\n")
    logging.info(f"Corrected {n_corrected}/{len(fields)} records -> {args.output}")
    if n_corrected == 0:
        logging.error("No records had INFO/plantCAD_zero_shot; was the VCF scored by zero_shot_score.py?")


# ----------------------------------------------------------------------------------------------
# BED-mode TSV
# ----------------------------------------------------------------------------------------------

def correct_tsv(args, markov, genome):
    with open(args.input) as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    col = {c: i for i, c in enumerate(header)}
    if "corrected_score" in col or "corrected_scores" in col:
        sys.exit("Input already contains corrected scores.")
    mode_all = "scores" in col
    has_probs = "ref_prob" in col and "alt_probs" in col
    if not mode_all and args.aggregation == "max" and not has_probs:
        sys.exit("A `score` column from -aggregation max cannot be corrected without the per-allele "
                 "scores. Re-score with -aggregation all or -output-raw-prob, or run zero_shot_score.py "
                 "with -markov-table directly.")

    logp = [None] * len(rows)
    by_chrom = {}
    for i, r in enumerate(rows):
        by_chrom.setdefault(r[col["chr"]], []).append(i)
    for chrom, idxs in by_chrom.items():
        name = genome.resolve(chrom)
        if name is None:
            sys.exit(f"Chromosome {chrom} not found in FASTA")
        pos0 = [int(rows[i][col["start"]]) for i in idxs]
        left, right, _ = contexts_for(genome, name, pos0, markov.k)
        lp = markov.log_probs(left, right)
        for j, i in enumerate(idxs):
            logp[i] = lp[j]

    new_cols = ["corrected_scores" if mode_all else "corrected_score", "markov_llrs"]
    with open(args.output, "w") as out:
        out.write("\t".join(header + new_cols) + "\n")
        for r, lp in zip(rows, logp):
            ref = r[col["ref"]]
            ri = NUCLEOTIDES.index(ref)
            alts = r[col["alt_alleles"]].split(",") if mode_all else [b for b in NUCLEOTIDES if b != ref]
            m = np.array([lp[NUCLEOTIDES.index(a)] - lp[ri] for a in alts])
            if mode_all:
                s = np.array([float(x) for x in r[col["scores"]].split(",")])
                corrected = ",".join(f"{v:.6f}" for v in s - m)
            elif args.aggregation == "average":
                corrected = f"{float(r[col['score']]) - m.mean():.6f}"
            else:  # max, recovered from raw probabilities
                ref_p = float(r[col["ref_prob"]])
                alt_p = np.array([float(x) for x in r[col["alt_probs"]].split(",")])
                llr = np.log(np.maximum(alt_p, 1e-6) / max(ref_p, 1e-6))
                corrected = f"{(llr - m).max():.6f}"
            out.write("\t".join(r + [corrected, ",".join(f"{v:.6f}" for v in m)]) + "\n")
    logging.info(f"Corrected {len(rows)} positions -> {args.output}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    args = parse_args()
    with open(args.input) as f:
        first = f.readline()
    is_vcf = first.startswith("##fileformat=VCF") or first.startswith("#CHROM")
    if not is_vcf and not first.startswith("chr\tstart\tend\tref"):
        sys.exit("Unrecognised input: expected a VCF or a BED-mode TSV from zero_shot_score.py")

    markov = MarkovBackground(args.markovTable, smoothing=args.markovSmoothing,
                              alpha=args.markovAlpha, direction=args.markovDirection)
    genome = Genome(args.inputFasta)
    (correct_vcf if is_vcf else correct_tsv)(args, markov, genome)


if __name__ == "__main__":
    main()
