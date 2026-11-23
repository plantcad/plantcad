## Markov background correction: reproducing the ridgeline figure

Compares raw PlantCAD zero-shot scores with Markov-corrected scores across variant consequences,
including a separate `repeat` category. Method details: [docs/zero-shot-scoring.md](../../docs/zero-shot-scoring.md#markov-background-correction).

Inputs:
- A VEP-annotated set of simulated SNVs (tab-delimited VEP output), e.g. from the
  [in-silico mutagenesis pipeline](../in-silico-mutagenesis/README.md)
- Reference genome FASTA, e.g. `Zm-B73-REFERENCE-NAM-5.0.fa`
- Repeat/TE annotation GFF3, e.g. `Zm-B73-REFERENCE-NAM-5.0.TE.gff3` (MaizeGDB)

```bash
REF=Zm-B73-REFERENCE-NAM-5.0.fa
TE=Zm-B73-REFERENCE-NAM-5.0.TE.gff3
VEP=maize_extand_annotated_downsampled.vcf   # VEP tab output

# 1. VEP output -> VCF; flag TE-overlapping variants (adds "chr" to chromosome names if missing)
python prepare_vcf.py -vep $VEP -te $TE -output variants.vcf

# 2. Build the genome-wide 13-mer Markov table (once per genome, ~1 min)
python ../../src/markov_background.py -input-fasta $REF -output Zm-B73-NAM5_k13.npz -k 13 -threads 16

# 3. Score with PlantCAD + Markov correction
python ../../src/zero_shot_score.py \
    -input-vcf variants.vcf \
    -input-fasta $REF \
    -output variants.scored.vcf \
    -model kuleshov-group/PlantCaduceus_l32 \
    -contextSize 512 \
    -device cuda:0 \
    -markov-table Zm-B73-NAM5_k13.npz
# (already have a VCF scored without -markov-table? correct it in seconds instead:
#  python ../../src/markov_correct.py -input raw.vcf -input-fasta $REF -markov-table Zm-B73-NAM5_k13.npz -output variants.scored.vcf)

# 4. Plot
python plot_ridgeline.py -input variants.scored.vcf -output ridgeline.png -title Maize -summary summary.tsv
```

TE-overlapping variants whose VEP consequence is intergenic, intron, upstream or downstream are relabelled
`repeat` (change with `-repeat-from`). Coding, UTR and splice variants keep their VEP label even inside TEs.

Runtime note: PlantCaduceus_l32 at 512 bp scores ~65 variants/s on one RTX 6000 Ada, so 1.3M variants take
~5.5 h. For a quicker look, subsample each consequence first (e.g. 20k variants per category, ~1.2 h).
