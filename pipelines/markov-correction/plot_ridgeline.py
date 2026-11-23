"""Ridgeline overlay of raw vs Markov-corrected PlantCAD zero-shot scores per variant consequence.

Input: a VCF scored by src/zero_shot_score.py with -markov-table, carrying INFO/TYPE (consequence),
optionally INFO/TE (repeat overlap flag, see prepare_vcf.py), INFO/plantCAD_zero_shot and
INFO/plantCAD_zero_shot_corrected.

Variants flagged TE whose consequence is in -repeat-from (non-coding classes by default) are
relabelled "repeat", so repeat-derived variants form their own category.

Usage:
    python plot_ridgeline.py -input scored.vcf -output ridgeline.png -title Maize
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DEFAULT_ORDER = [
    "stop_gained", "splice_acceptor", "splice_donor", "start_lost", "missense", "repeat",
    "intergenic", "5_prime_UTR", "upstream_gene", "synonymous", "downstream_gene", "intron",
    "3_prime_UTR",
]
RAW_LINE = "#B8740F"                  # raw scores: step outline
COR_FILL, COR_LINE = "#2F62AE", "#1F4A8A"  # corrected scores: filled bars
INK, INK_2, AXIS = "#1B2230", "#4A5466", "#9AA1AD"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-input", required=True, help="Scored VCF")
    p.add_argument("-output", required=True, help="Output figure (.png/.pdf)")
    p.add_argument("-title", default="", help="Figure title")
    p.add_argument("-repeat-from", nargs="+",
                   default=["intergenic", "intron", "upstream_gene", "downstream_gene"],
                   help="Consequences (short names) relabelled 'repeat' when the variant overlaps a TE.")
    p.add_argument("-categories", nargs="+", default=DEFAULT_ORDER, help="Categories to plot, top to bottom.")
    p.add_argument("-xlim", nargs=2, type=float, default=[-15, 5])
    p.add_argument("-summary", default=None, help="Optional TSV with per-category summary statistics.")
    return p.parse_args()


def short_name(consequence):
    return consequence.replace("_variant", "")


def read_scored_vcf(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            info = {}
            for kv in fields[7].split(";"):
                k, _, v = kv.partition("=")
                info[k] = v if _ else True
            raw = info.get("plantCAD_zero_shot", ".").split(",")[0]
            cor = info.get("plantCAD_zero_shot_corrected", ".").split(",")[0]
            if raw in (".", "") or cor in (".", ""):
                continue
            rows.append((short_name(info.get("TYPE", "unknown")), "TE" in info, float(raw), float(cor)))
    return pd.DataFrame(rows, columns=["type", "te", "raw", "corrected"])


def main():
    args = parse_args()
    df = read_scored_vcf(args.input)
    df["category"] = np.where(df["te"] & df["type"].isin(args.repeat_from), "repeat", df["type"])
    cats = [c for c in args.categories if (df["category"] == c).any()]

    lo, hi = args.xlim
    bins = np.linspace(lo, hi, 61)
    width = bins[1] - bins[0]
    n = len(cats)
    row_h = 0.96  # drawing height of the tallest bar in a row, in row units (rows are 1 apart)

    fig, ax = plt.subplots(figsize=(7.2, 0.55 * n + 1.3))
    summary = []
    for i, cat in enumerate(cats):
        sub = df[df["category"] == cat]
        base = n - 1 - i
        # Density over all values, so out-of-range tails are dropped rather than piled up at the edges.
        dens = {}
        for col in ("raw", "corrected"):
            x = sub[col].to_numpy()
            h, _ = np.histogram(x[(x >= lo) & (x <= hi)], bins=bins)
            dens[col] = h / (len(x) * width)
        scale = row_h / max(d.max() for d in dens.values())
        z = 10 + 2 * i  # upper rows are drawn below lower rows

        ax.bar(bins[:-1], dens["corrected"] * scale, width=width, bottom=base, align="edge",
               color=COR_FILL, alpha=0.5, lw=0, zorder=z)
        ax.stairs(base + dens["raw"] * scale, bins, baseline=base, color=RAW_LINE, lw=1.5, zorder=z + 0.2)
        ax.axhline(base, color=AXIS, lw=0.6, zorder=z + 0.4)

        med_r, med_c = sub["raw"].median(), sub["corrected"].median()
        for m, col in ((med_r, RAW_LINE), (med_c, COR_LINE)):
            ax.plot([m, m], [base, base + 0.13], color=col, lw=1.8, zorder=z + 0.5, solid_capstyle="butt")
        ax.text(lo - 0.3, base + 0.12, cat.replace("_", " "), ha="right", va="bottom", fontsize=10,
                fontweight="bold" if cat == "repeat" else "normal", color=INK)
        summary.append({"category": cat, "n": len(sub), "median_raw": med_r, "median_corrected": med_c,
                        "mean_raw": sub["raw"].mean(), "mean_corrected": sub["corrected"].mean()})

    ax.axvline(0, color=AXIS, lw=0.6, ls=(0, (3, 3)), zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.4, n - 1 + row_h + 0.15)
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=INK_2)
    ax.set_xlabel("Zero-shot score  (log P(alt) − log P(ref))", color=INK)
    handles = [Line2D([], [], color=RAW_LINE, lw=1.5, label="PlantCAD (raw)"),
               Patch(facecolor=COR_FILL, alpha=0.5, label="PlantCAD − Markov")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=9.5)
    if args.title:
        ax.set_title(args.title, fontsize=13, fontweight="bold", loc="left", x=-0.27, y=1.035)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")

    summary = pd.DataFrame(summary)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if args.summary:
        summary.to_csv(args.summary, sep="\t", index=False)


if __name__ == "__main__":
    main()
