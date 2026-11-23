"""Convert VEP tab-delimited output into a minimal VCF and flag variants that overlap repeats.

The output VCF keeps the VEP consequence in INFO/TYPE and adds INFO/TE (a flag) for variants
overlapping any feature in a repeat/TE GFF3 (e.g. Zm-B73-REFERENCE-NAM-5.0.TE.gff3).

Usage:
    python prepare_vcf.py -vep vep_output.txt -te TE.gff3 -output variants.vcf [-chr-prefix chr]
"""
import argparse
import sys
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-vep", required=True, help="VEP tab-delimited output (Uploaded_variation, Location, Allele, ...)")
    p.add_argument("-te", required=True, help="Repeat/TE annotation in GFF3 format")
    p.add_argument("-output", required=True, help="Output VCF")
    p.add_argument("-chr-prefix", default="chr",
                   help="Prefix added to VEP chromosome names so they match the FASTA/GFF (default: chr). "
                        "Not added if the name already starts with it.")
    return p.parse_args()


def fix_chrom(chrom, prefix):
    return chrom if not prefix or chrom.startswith(prefix) else prefix + chrom


def read_vep(path, prefix):
    records = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom, pos = fields[1].split(":")[0], fields[1].split(":")[1].split("-")[0]
            alt = fields[2]
            consequence = fields[6]
            extra = dict(kv.split("=", 1) for kv in fields[13].split(";") if "=" in kv)
            ref = extra.get("REF_ALLELE")
            if ref is None:  # fall back to Uploaded_variation like 8_123_A/T
                ref = fields[0].rsplit("_", 1)[-1].split("/")[0]
            if len(ref) != 1 or len(alt) != 1:
                continue
            records.append((fix_chrom(chrom, prefix), int(pos), fields[0], ref, alt, consequence))
    return records


def te_masks(gff_path, chrom_maxpos):
    """Boolean mask (1-based positions) of TE coverage for each requested chromosome."""
    masks = {c: np.zeros(m + 2, dtype=bool) for c, m in chrom_maxpos.items()}
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.split("\t", 5)
            if len(fields) < 5 or fields[0] not in masks:
                continue
            mask = masks[fields[0]]
            start, end = int(fields[3]), min(int(fields[4]), len(mask) - 1)
            if start <= end:
                mask[start:end + 1] = True
    return masks


def main():
    args = parse_args()
    records = read_vep(args.vep, args.chr_prefix)
    if not records:
        sys.exit("No SNVs parsed from VEP file")

    chrom_maxpos = defaultdict(int)
    for chrom, pos, *_ in records:
        chrom_maxpos[chrom] = max(chrom_maxpos[chrom], pos)
    masks = te_masks(args.te, chrom_maxpos)

    chrom_order = {c: i for i, c in enumerate(sorted(chrom_maxpos, key=lambda c: (len(c), c)))}
    records.sort(key=lambda r: (chrom_order[r[0]], r[1]))

    n_te = defaultdict(int)
    n_all = defaultdict(int)
    with open(args.output, "w") as out:
        out.write("##fileformat=VCFv4.2\n")
        out.write('##INFO=<ID=TYPE,Number=1,Type=String,Description="VEP consequence">\n')
        out.write('##INFO=<ID=TE,Number=0,Type=Flag,Description="Variant overlaps a repeat/TE annotation">\n')
        for chrom in chrom_order:
            out.write(f"##contig=<ID={chrom}>\n")
        out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for chrom, pos, vid, ref, alt, csq in records:
            in_te = bool(masks[chrom][pos])
            n_all[csq] += 1
            n_te[csq] += in_te
            info = f"TYPE={csq}" + (";TE" if in_te else "")
            out.write(f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t.\t.\t{info}\n")

    print(f"Wrote {len(records)} SNVs to {args.output}")
    print(f"{'consequence':<28}{'n':>9}{'in_TE':>9}{'frac':>8}")
    for csq in sorted(n_all, key=lambda c: -n_all[c]):
        print(f"{csq:<28}{n_all[csq]:>9}{n_te[csq]:>9}{n_te[csq] / n_all[csq]:>8.3f}")


if __name__ == "__main__":
    main()
