"""Regression tests for SV allele construction; no model weights or GPU needed.

Run in the PlantCAD environment from the repository root:
    python -m unittest discover -s tests -p test_zero_shot_score_sv.py -v
"""
import importlib.util
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import vcf


SCORER_PATH = Path(__file__).resolve().parents[1] / "src" / "zero_shot_score_sv.py"
spec = importlib.util.spec_from_file_location("zero_shot_score_sv", SCORER_PATH)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)

HEADER = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=2048>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Structural variant type">
##INFO=<ID=END,Number=1,Type=Integer,Description="Last affected reference position">
##INFO=<ID=NOTE,Number=1,Type=String,Description="A field to preserve">
##ALT=<ID=DEL,Description="Deletion">
##ALT=<ID=INS,Description="Insertion">
##ALT=<ID=DUP,Description="Duplication">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
"""


class SVSequenceTests(unittest.TestCase):
    context = 64
    flank = 3
    pos = 801

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        rng = random.Random(20261026)
        self.genome = "".join(rng.choices("ACGT", k=2048))
        self.fasta = self.directory / "reference.fa"
        self.fasta.write_text(">chr1\n" + self.genome + "\n")
        self.input_number = 0

    def write_vcf(self, records, sample=False):
        """records contain (1-based position, REF, ALT, INFO)."""
        self.input_number += 1
        path = self.directory / f"input_{self.input_number}.vcf"
        columns = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"
        if sample:
            columns += "\tFORMAT\tsample1"
        rows = []
        for i, (pos, ref, alt, info) in enumerate(records):
            row = f"chr1\t{pos}\tvariant{i}\t{ref}\t{alt}\t50\tPASS\t{info}"
            rows.append(row + ("\tGT\t0/1" if sample else ""))
        path.write_text(HEADER + columns + "\n" + "\n".join(rows) + "\n")
        return path

    def build(self, ref, alt, info="."):
        path = self.write_vcf([(self.pos, ref, alt, info)])
        args = SimpleNamespace(
            inputVCF=str(path), inputFasta=str(self.fasta),
            contextSize=self.context, flankSize=self.flank,
        )
        # Use actual FASTA and PyVCF parsing, not mocked record objects.
        return scorer.seq_from_vcf_sv(args)

    def assert_allele_and_flanks(self, ref, alt, result):
        refs, mutants, ref_indices, mut_indices, records = result
        self.assertEqual(len(records), 1)
        ref_seq, mutant = refs[0], mutants[0]
        ri, mi = ref_indices[0], mut_indices[0]
        a = self.pos - 1
        n, m = len(ref), len(alt)
        half = self.context // 2

        # Independently apply the VCF replacement to the whole genome.
        haplotype = self.genome[:a] + alt + self.genome[a + n:]
        # The deletion junction is centered; insertions retain the reference
        # window's left boundary, centered immediately after the anchor.
        left_end = a + m if n > m else a + 1
        mutant_start = left_end - half
        expected_mutant = haplotype[mutant_start:mutant_start + self.context]
        self.assertEqual(mutant, expected_mutant)
        self.assertEqual(len(ref_seq), self.context)
        self.assertEqual(len(mutant), self.context)

        # Find the actual reference window in the deterministic random genome,
        # then check that both windows score the same surviving genome bases.
        ref_start = self.genome.index(ref_seq)
        positions = list(range(left_end - self.flank, left_end))
        positions += list(range(a + n, a + n + self.flank))
        self.assertEqual([ref_start + i for i in ri], positions)
        hap_positions = [
            p if p < a + n else p + m - n for p in positions
        ]
        self.assertEqual([mutant_start + i for i in mi], hap_positions)
        expected_bases = "".join(self.genome[p] for p in positions)
        self.assertEqual("".join(ref_seq[i] for i in ri), expected_bases)
        self.assertEqual("".join(mutant[i] for i in mi), expected_bases)

    def test_normalized_insertions_keep_anchor(self):
        anchor = self.genome[self.pos - 1]
        for inserted in ("T", "GCTAG"):
            with self.subTest(inserted=inserted):
                alt = anchor + inserted
                self.assert_allele_and_flanks(
                    anchor, alt, self.build(anchor, alt),
                )

    def test_normalized_deletions_keep_anchor(self):
        a = self.pos - 1
        for deleted_length in (1, 5):
            with self.subTest(deleted_length=deleted_length):
                ref = self.genome[a:a + deleted_length + 1]
                alt = ref[0]
                self.assert_allele_and_flanks(ref, alt, self.build(ref, alt))

    def test_deletion_keeps_the_entire_alt_prefix(self):
        a = self.pos - 1
        ref = self.genome[a:a + 8]
        alt = ref[:3]
        self.assert_allele_and_flanks(ref, alt, self.build(ref, alt))

    def test_symbolic_deletion_matches_explicit_deletion(self):
        a = self.pos - 1
        ref = self.genome[a:a + 6]
        end = self.pos + len(ref) - 1
        explicit = self.build(ref, ref[0])
        symbolic = self.build(ref[0], "<DEL>", f"SVTYPE=DEL;END={end}")
        self.assertEqual(symbolic[:4], explicit[:4])
        self.assert_allele_and_flanks(ref, ref[0], symbolic)

    def test_unsupported_records_are_skipped_without_crashing(self):
        a = self.pos - 1
        anchor = self.genome[a]
        different = next(base for base in "ACGT" if base != self.genome[a + 1])
        cases = [
            (anchor, "<DEL>", "SVTYPE=DEL"),  # Missing END.
            (anchor, "<DEL>", f"END={self.pos + 5}"),  # No SVTYPE.
            (anchor, "<INS>", "SVTYPE=INS"),
            (anchor, "<DUP>", f"SVTYPE=DUP;END={self.pos + 5}"),
            (anchor, anchor + "]chr1:1000]", "."),
            (self.genome[a:a + 4], anchor + different, "."),
        ]
        for ref, alt, info in cases:
            with self.subTest(ref=ref, alt=alt, info=info):
                result = self.build(ref, alt, info)
                self.assertEqual(result, ([], [], [], [], []))

    def test_symbolic_deletion_obeys_reference_span_limit(self):
        a = self.pos - 1
        limit = self.context // 2 - self.flank
        # The current limit includes the anchor: END - POS + 1.
        for span in (limit - 1, limit):
            with self.subTest(span=span):
                ref = self.genome[a:a + span]
                info = f"SVTYPE=DEL;END={self.pos + span - 1}"
                symbolic = self.build(ref[0], "<DEL>", info)
                explicit = self.build(ref, ref[0])
                self.assertEqual(symbolic[:4], explicit[:4])
                if span < limit:
                    self.assert_allele_and_flanks(ref, ref[0], symbolic)
                else:
                    self.assertEqual(symbolic, ([], [], [], [], []))

    def test_main_preserves_unscoreable_records_without_loading_model(self):
        a = self.pos - 1
        anchor = self.genome[a]
        other = next(base for base in "ACGT" if base != anchor)
        records = [
            (self.pos, anchor, other, "NOTE=keep"),
            (self.pos + 10, self.genome[a + 10], "<DUP>",
             f"SVTYPE=DUP;END={self.pos + 20};NOTE=keep"),
        ]
        input_vcf = self.write_vcf(records, sample=True)
        output_vcf = self.directory / "output.vcf"
        args = SimpleNamespace(
            inputVCF=str(input_vcf), inputFasta=str(self.fasta),
            output=str(output_vcf), model="unused", device="cpu",
            batchSize=2, contextSize=self.context, flankSize=self.flank,
        )
        with patch.object(scorer, "parse_args", return_value=args), \
                patch.object(scorer, "load_model_and_tokenizer") as loader:
            scorer.main()
        loader.assert_not_called()
        self.assertTrue(output_vcf.is_file())

        def read_records(path):
            with path.open() as handle:
                return list(vcf.Reader(handle))

        original = read_records(input_vcf)
        written = read_records(output_vcf)
        self.assertEqual(len(written), len(original))
        for before, after in zip(original, written):
            self.assertEqual(
                (after.CHROM, after.POS, after.ID, after.REF,
                 [str(x) for x in after.ALT], after.QUAL, after.FILTER,
                 after.INFO, after.FORMAT, after.samples[0].sample,
                 after.samples[0]["GT"]),
                (before.CHROM, before.POS, before.ID, before.REF,
                 [str(x) for x in before.ALT], before.QUAL, before.FILTER,
                 before.INFO, before.FORMAT, before.samples[0].sample,
                 before.samples[0]["GT"]),
            )
            self.assertNotIn("PlantCAD_SV_Score", after.INFO)


if __name__ == "__main__":
    unittest.main()
