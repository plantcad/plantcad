import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse, sys
from tqdm import tqdm
import logging
import vcf
from Bio import SeqIO
import gzip

from markov_background import MarkovBackground


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score variants using a pre-trained language model. Supports VCF input for specific variants or BED input for genome-wide scoring."
    )
    inputGroup = parser.add_mutually_exclusive_group(required=True)
    inputGroup.add_argument("-input-vcf", dest="inputVCF", type=str, default=None,
                            help="The directory of input VCF file for specific variant scoring.")
    inputGroup.add_argument("-input-bed", dest="inputBed", type=str, default=None,
                            help="The directory of BED file specifying regions for genome-wide scoring.")

    parser.add_argument("-input-fasta", dest="inputFasta", type=str, default=None,
                        help="The directory of reference genome fasta file. Required for both VCF and BED inputs.")
    parser.add_argument("-output", dest="output", required=True, help="The directory of output file.")
    parser.add_argument("-model", dest="model", required=True, help="The directory of pre-trained model.")
    parser.add_argument("-device", dest="device", default="cuda:0", help="The device to run the model (default: cuda:0).")
    parser.add_argument("-batchSize", dest="batchSize", default=128, type=int, help="The batch size for the model (default: 128).")
    parser.add_argument("-contextSize", dest="contextSize", default=2048, type=int,
                        help="The context window size (default: 2048). "
                             "PlantCAD2 models enforce a minimum of 2048; "
                             "PlantCaduceus models are fixed at 512.")
    
    # Arguments for genome-wide (BED) mode
    bed_group = parser.add_argument_group('BED Mode Options', 'These options are only for use with --input-bed')
    bed_group.add_argument("-step-size", dest="stepSize", default=1, type=int,
                        help="Number of positions to extract per window (e.g., 1, 4, etc.). Defaults to 1.")
    bed_group.add_argument("-use-masking", dest="useMasking", action="store_true", default=False,
                        help="Use masking for genome-wide scoring (default: False, use unmasked logits).")
    bed_group.add_argument("-aggregation", dest="aggregation", default="average",
                        choices=["max", "average", "all"],
                        help="How to aggregate alternative allele scores for genome-wide scoring: max, average (default), or all.")
    bed_group.add_argument("-output-raw-prob", dest="outputRawProb", action="store_true", default=False,
                        help="Output raw probabilities in addition to scores for genome-wide scoring.")

    # Markov k-mer background correction (both modes)
    markov_group = parser.add_argument_group(
        'Markov background correction (recommended)',
        'Subtract the LLR of a genome-wide k-mer Markov model from the PlantCAD LLR to remove '
        'over-estimated conservation in repeats/TEs. Build the table with src/markov_background.py.')
    markov_group.add_argument("-markov-table", dest="markovTable", type=str, default=None,
                              help="Path to a k-mer count table (.npz) built by src/markov_background.py. "
                                   "When given, corrected scores are reported alongside the raw scores.")
    markov_group.add_argument("-markov-smoothing", dest="markovSmoothing", default="add",
                              choices=["add", "backoff"],
                              help="Smoothing of the Markov model: additive pseudocounts (default) or "
                                   "interpolated backoff to shorter contexts.")
    markov_group.add_argument("-markov-alpha", dest="markovAlpha", default=None, type=float,
                              help="Pseudocount (add) or backoff strength (backoff). "
                                   "Defaults: 2 for add, 8 for backoff.")
    markov_group.add_argument("-markov-direction", dest="markovDirection", default="both",
                              choices=["both", "left"],
                              help="Context used by the Markov model: 'both' (default) averages the upstream "
                                   "(+ strand) and downstream (- strand) k-mer contexts, matching PlantCAD's "
                                   "bidirectionality; 'left' uses only the k upstream bases.")

    args = parser.parse_args()

    if args.markovAlpha is None:
        args.markovAlpha = 2.0 if args.markovSmoothing == "add" else 8.0

    if (args.inputVCF is not None or args.inputBed is not None) and args.inputFasta is None:
        sys.exit("-input-fasta is required with -input-vcf or -input-bed")

    # Calculate tokenIdx as the middle of the context window
    # For a 512 window: tokenIdx = 255 (255 bases before, variant at 256 （1-based coordinates）, 256 bases after)
    args.tokenIdx = args.contextSize // 2 - 1

    return args


class SequenceDataset(Dataset):
    def __init__(self, sequences, tokenizer, tokenIdx, stepSize=1, useMasking=True):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.tokenIdx = tokenIdx
        self.stepSize = stepSize
        self.useMasking = useMasking

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoding = self.tokenizer.encode_plus(
            sequence,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False
        )
        input_ids = encoding['input_ids']

        if self.useMasking:
            # Mask the center position(s) based on stepSize
            start_offset = -(self.stepSize // 2) + (1 if self.stepSize % 2 == 0 else 0)
            for i in range(self.stepSize):
                mask_pos = self.tokenIdx + start_offset + i
                # Ensure the mask position is within the sequence bounds
                if 0 <= mask_pos < input_ids.shape[1]:
                    input_ids[0, mask_pos] = self.tokenizer.mask_token_id

        return {
            'sequence': sequence,
            'input_ids': input_ids
        }


def load_model_and_tokenizer(model_dir, device):
    logging.info(f"Loading model and tokenizer from {model_dir}")

    # Determine the appropriate dtype based on the GPU capabilities
    def get_optimal_dtype():
        if not torch.cuda.is_available():
            logging.info("Using float32 as no GPU is available.")
            return torch.float32

        device_index = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device_index)

        if capability[0] >= 8:  # sm_80 or higher
            logging.info("Using bfloat16 as the GPU supports sm_80 or higher.")
            return torch.bfloat16
        elif capability[0] >= 6:  # sm_60 or higher
            logging.info("Using float16 as the GPU supports sm_60 or higher.")
            return torch.float16
        else:
            logging.info("Using float32 as the GPU does not support float16 or bfloat16.")
            return torch.float32

    optimal_dtype = get_optimal_dtype()

    # Load the model with the selected dtype
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=optimal_dtype)
        # Note: We set dtype in two places because some transformers versions are inconsistent in
        # honoring torch_dtype during from_pretrained. See:
        # https://github.com/huggingface/transformers/issues/36567
        # Passing torch_dtype here and also calling model.to(dtype) below ensures correctness across
        # versions; when from_pretrained respects torch_dtype, the subsequent .to(dtype) is a no-op.
        model.to(optimal_dtype)
    except Exception as e:
        logging.error(f"Failed to load model with {optimal_dtype}, falling back to float32. Error: {e}")
        model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)
    return model, tokenizer


def create_dataloader(sequences, tokenizer, batch_size, tokenIdx, stepSize=1, useMasking=True):
    logging.info(f"Creating DataLoader with batch size {batch_size}")
    dataset = SequenceDataset(sequences, tokenizer, tokenIdx, stepSize=stepSize, useMasking=useMasking)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def extract_logits(model, dataloader, device, tokenIdx, tokenizer, stepSize=1, window_sizes=None):
    logging.info("Extracting logits")
    nucleotides = list('acgt')
    results = []
    
    window_idx = 0

    for batch in tqdm(dataloader):
        curIDs = batch['input_ids'].to(device)
        curIDs = curIDs.squeeze(1)
        batch_size = curIDs.shape[0]

        with torch.inference_mode():
            outputs = model(input_ids=curIDs)
        all_logits = outputs.logits

        # This logic is for the windowed genome-wide LLR scoring (BED workflow)
        if stepSize > 1 and window_sizes is not None:
            start_offset = -(stepSize // 2) + (1 if stepSize % 2 == 0 else 0)
            for b in range(batch_size):
                actual_window_size = window_sizes[window_idx]
                window_probs = []
                for i in range(actual_window_size):
                    pos = tokenIdx + start_offset + i
                    logits_pos = all_logits[b, pos, [tokenizer.get_vocab()[nc] for nc in nucleotides]]
                    probs = torch.nn.functional.softmax(logits_pos.cpu(), dim=0).numpy()
                    window_probs.append(probs)
                results.extend(window_probs)
                window_idx += 1
        # This is the generic logic for zero-shot scoring (VCF workflow)
        else:
            logits = all_logits[:, tokenIdx, [tokenizer.get_vocab()[nc] for nc in nucleotides]]
            probs = torch.nn.functional.softmax(logits.cpu(), dim=1).numpy()
            results.append(probs)

    if stepSize > 1 and window_sizes is not None:
        return np.array(results)
    else:
        return np.concatenate(results, axis=0)


def load_fasta(fasta_path):
    """Loads a FASTA file, handling .gz compression."""
    logging.info(f"Loading reference genome from {fasta_path}")
    if fasta_path.endswith(".gz"):
        with gzip.open(fasta_path, "rt") as file:
            return SeqIO.to_dict(SeqIO.parse(file, "fasta"))
    else:
        return SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))


def load_bed_file(bed_path):
    """Load BED file and return DataFrame with chr, start, end columns."""
    logging.info(f"Reading BED file from {bed_path}")
    bed_df = pd.read_csv(bed_path, sep='\t', header=None,
                         names=['chr', 'start', 'end'], usecols=[0, 1, 2])
    return bed_df


def extract_sequences_from_bed(args, bed_df, fasta_dict, markov_k=None):
    """Extract sequences using windowed approach for BED regions.

    If markov_k is given, also returns the upstream/downstream k-bp contexts of every position
    (for the Markov background correction).
    """
    logging.info("Extracting sequences from BED regions with windowed approach")

    sequences = []
    position_info = []  # Store chr, pos, ref_allele for each position
    window_sizes = []  # Track actual number of valid positions per window
    markov_ctx = []  # (left, right) context per position, only if markov_k is set

    addIdx = args.contextSize - args.tokenIdx

    logging.info(f"Using step_size={args.stepSize}, masking={args.useMasking}")

    for _, row in tqdm(bed_df.iterrows(), total=len(bed_df)):
        chrom = str(row['chr'])
        start = int(row['start'])
        end = int(row['end'])

        if chrom not in fasta_dict:
            logging.warning(f"Chromosome {chrom} not found in FASTA file, skipping region")
            continue

        for window_start in range(start, end, args.stepSize):
            window_end = min(window_start + args.stepSize, end)
            num_positions = window_end - window_start

            if num_positions == 0:
                continue

            center_pos = window_start + (num_positions - 1) / 2.0
            center_pos_int = int(center_pos)

            try:
                window_refs = []
                window_positions = []

                for pos in range(window_start, window_end):
                    ref_allele = str(fasta_dict[chrom].seq[pos]).upper()
                    if ref_allele not in ['A', 'C', 'G', 'T']:
                        continue
                    window_refs.append(ref_allele)
                    window_positions.append(pos)

                if not window_refs:
                    continue

                seq_start = center_pos_int - args.tokenIdx
                seq_end = center_pos_int + addIdx

                if seq_start < 0:
                    seq = str(fasta_dict[chrom].seq[0:seq_end]).upper().rjust(args.contextSize, "N")
                else:
                    seq = str(fasta_dict[chrom].seq[seq_start:seq_end]).upper().ljust(args.contextSize, "N")

                sequences.append(seq)
                window_sizes.append(len(window_refs))

                for pos, ref in zip(window_positions, window_refs):
                    position_info.append({'chr': chrom, 'pos': pos, 'ref': ref})
                    if markov_k is not None:
                        markov_ctx.append(markov_contexts(seq, args.tokenIdx + (pos - center_pos_int), markov_k))

            except Exception as e:
                logging.warning(f"Error processing window {chrom}:{window_start}-{window_end}, skipping. Error: {e}")
                continue

    logging.info(f"Extracted {len(sequences)} windows covering {len(position_info)} positions")
    if markov_k is not None:
        return sequences, position_info, window_sizes, markov_ctx
    return sequences, position_info, window_sizes


def _aggregate(llrs, aggregation):
    if aggregation == "max":
        return max(llrs)
    if aggregation == "average":
        return np.mean(llrs)
    return llrs


def calculate_genome_wide_llr(position_info, probs_array, aggregation, markov_logp=None):
    """Calculate log likelihood ratios for each position.

    If markov_logp (n, 4) is given, Markov-corrected LLRs (PlantCAD LLR - Markov LLR) are added and
    aggregated the same way as the raw LLRs.
    """
    logging.info("Calculating genome-wide log likelihood ratios")

    nucleotides = ['A', 'C', 'G', 'T']
    results = []

    assert len(probs_array) == len(position_info), f"Mismatch: {len(probs_array)} probs vs {len(position_info)} positions"
    if markov_logp is not None:
        assert len(markov_logp) == len(position_info)

    for i, (pos_info, probs) in enumerate(zip(position_info, probs_array)):
        ref_allele = pos_info['ref']
        ref_idx = nucleotides.index(ref_allele)
        ref_prob = probs[ref_idx]

        alt_indices = [i for i in range(4) if i != ref_idx]
        alt_probs = [probs[i] for i in alt_indices]
        alt_alleles = [nucleotides[i] for i in alt_indices]

        llrs = [np.log(alt_prob / ref_prob) if ref_prob > 0 else np.inf for alt_prob in alt_probs]

        result = {
            'chr': pos_info['chr'],
            'start': pos_info['pos'],
            'end': pos_info['pos'] + 1,
            'ref': ref_allele,
            'ref_prob': ref_prob,
            'alt_alleles': alt_alleles,
            'alt_probs': alt_probs,
            'llrs': llrs
        }

        key = 'scores' if aggregation == "all" else 'score'
        result[key] = _aggregate(llrs, aggregation)

        if markov_logp is not None:
            markov_llrs = [markov_logp[i][j] - markov_logp[i][ref_idx] for j in alt_indices]
            result['markov_llrs'] = markov_llrs
            result['corrected_' + key] = _aggregate([l - m for l, m in zip(llrs, markov_llrs)], aggregation)

        results.append(result)

    return results


def write_llr_output(results, output_path, aggregation, output_raw_prob):
    """Write genome-wide LLR results to output file."""
    logging.info(f"Writing LLR results to {output_path}")

    has_markov = bool(results) and 'markov_llrs' in results[0]

    with open(output_path, 'w') as f:
        if aggregation == "all":
            header = "chr\tstart\tend\tref\talt_alleles\tscores"
            if output_raw_prob: header += "\tref_prob\talt_probs"
            if has_markov: header += "\tcorrected_scores\tmarkov_llrs"
        else:
            header = "chr\tstart\tend\tref\tscore"
            if output_raw_prob: header += "\tref_prob\talt_probs"
            if has_markov: header += "\tcorrected_score\tmarkov_llrs"
        f.write(header + "\n")

        for result in results:
            line = f"{result['chr']}\t{result['start']}\t{result['end']}\t{result['ref']}\t"
            if aggregation == "all":
                scores_str = ','.join([f"{s:.6f}" for s in result['scores']])
                alt_str = ','.join(result['alt_alleles'])
                line += f"{alt_str}\t{scores_str}"
                if output_raw_prob:
                    alt_probs_str = ','.join([f"{p:.6f}" for p in result['alt_probs']])
                    line += f"\t{result['ref_prob']:.6f}\t{alt_probs_str}"
            else:
                line += f"{result['score']:.6f}"
                if output_raw_prob:
                    alt_probs_str = ','.join([f"{p:.6f}" for p in result['alt_probs']])
                    line += f"\t{result['ref_prob']:.6f}\t{alt_probs_str}"
            if has_markov:
                if aggregation == "all":
                    line += "\t" + ','.join([f"{s:.6f}" for s in result['corrected_scores']])
                else:
                    line += f"\t{result['corrected_score']:.6f}"
                line += "\t" + ','.join([f"{m:.6f}" for m in result['markov_llrs']])
            f.write(line + "\n")


def zero_shot_score(snpDF, logits):
    logging.info("Calculating zero-shot scores")
    nucleotides = ['A', 'C', 'G', 'T']
    res = []
    for idx, (snp, probs) in enumerate(zip(snpDF.itertuples(index=False), logits)):
        refAllele = getattr(snp, 'ref')
        altAllele = getattr(snp, 'alt')
        refProb = probs[nucleotides.index(refAllele)]
        altProb = probs[nucleotides.index(altAllele)]
        res.append(np.log(altProb / refProb))
    return res


def load_markov_model(args):
    return MarkovBackground(args.markovTable, smoothing=args.markovSmoothing,
                            alpha=args.markovAlpha, direction=args.markovDirection)


def markov_contexts(seq, idx, k):
    """Upstream and downstream k-bp contexts around position idx of seq."""
    return seq[max(0, idx - k):idx], seq[idx + 1:idx + 1 + k]


def markov_log_probs(markov, sequences, tokenIdx):
    """Markov log P(A/C/G/T) at the centre position of each context sequence, shape (n, 4)."""
    logging.info("Computing Markov background log-probabilities")
    contexts = [markov_contexts(seq, tokenIdx, markov.k) for seq in sequences]
    left, right = zip(*contexts) if contexts else ((), ())
    return markov.log_probs(list(left), list(right))


def _add_info_header(reader, info_id, desc):
    reader.infos[info_id] = vcf.parser._Info(
        info_id, vcf.parser.field_counts["A"], "Float", desc, None, None, vcf.parser._encode_type("Float"))


def zero_shot_score_vcf(args, recordIndices, logits, markovLogProbs=None):
    logging.info("Calculating zero-shot scores")
    nucleotides = ['A', 'C', 'G', 'T']

    reader = vcf.Reader(filename=args.inputVCF)
    _add_info_header(reader, "plantCAD_zero_shot",
                     "PlantCAD zero-shot score, log P(alt) - log P(ref), one per ALT")
    if markovLogProbs is not None:
        _add_info_header(reader, "plantCAD_markov_llr",
                         "Genome-wide k-mer Markov background log P(alt) - log P(ref), one per ALT")
        _add_info_header(reader, "plantCAD_zero_shot_corrected",
                         "Markov-corrected PlantCAD zero-shot score (plantCAD_zero_shot - plantCAD_markov_llr), "
                         "one per ALT")
    writer = vcf.Writer(open(args.output, "w"), reader)

    currentIdx = 0

    rIdx = 0
    for record in reader:
        if rIdx < len(recordIndices) and currentIdx == recordIndices[rIdx]:
            scores, markovScores, corrected = [], [], []

            refAllele = record.REF
            refProb = logits[rIdx][nucleotides.index(refAllele)]

            for alt in record.ALT:
                if alt.type == "SNV":
                    altAllele = alt.sequence
                    altProb = logits[rIdx][nucleotides.index(altAllele)]
                    score = np.log(altProb / refProb)
                    scores.append(score)
                    if markovLogProbs is not None:
                        m = (markovLogProbs[rIdx][nucleotides.index(altAllele)]
                             - markovLogProbs[rIdx][nucleotides.index(refAllele)])
                        markovScores.append(m)
                        corrected.append(score - m)
                else:
                    scores.append(".")
                    markovScores.append(".")
                    corrected.append(".")

            record.INFO["plantCAD_zero_shot"] = ",".join(str(zs) for zs in scores)
            if markovLogProbs is not None:
                record.INFO["plantCAD_markov_llr"] = ",".join(str(zs) for zs in markovScores)
                record.INFO["plantCAD_zero_shot_corrected"] = ",".join(str(zs) for zs in corrected)

            writer.write_record(record)

            rIdx += 1
        currentIdx += 1

    writer.close()


def seq_from_vcf(args):
    logging.info(f"Reading input data from {args.inputVCF}")

    fastaDict = load_fasta(args.inputFasta)

    sequences = []
    recordIndices = []  # keep track of which record a sequence belongs to
    vcfReader = vcf.Reader(filename=args.inputVCF)
    recordIdx = 0
    prevChrom = ""
    addIdx = args.contextSize - args.tokenIdx

    for record in vcfReader:
        if any([alt.type == "SNV" for alt in record.ALT]):
            chrom = record.CHROM
            pos = record.POS - 1  # convert to zero-based indexing

            try:
                if pos - args.tokenIdx < 0:
                    seq = str(fastaDict[chrom][0:pos + addIdx].seq).upper().rjust(args.contextSize, "N")
                else:
                    seq = str(fastaDict[chrom][pos - args.tokenIdx:pos + addIdx].seq).upper().ljust(args.contextSize, "N")

                sequences.append(seq)
                recordIndices.append(recordIdx)

                # RAM saving: remove chromosomes when the corresponding vcf records have been processed
                if prevChrom != chrom:
                    if prevChrom != "":
                        fastaDict.pop(prevChrom)
                    prevChrom = chrom
            except:
                print("Error processing VCF at record " + str(recordIdx))
                print("Check that VCF file is sorted and chromosome names match FASTA file.")
                print(record)
                sys.exit()
        recordIdx += 1
    return sequences, recordIndices


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    args = parse_args()

    # ----- Enforce model-specific context size constraints -----
    model_name_lower = args.model.lower()
    if "plantcad2" in model_name_lower:
        MIN_CTX = 2048
        if args.contextSize < MIN_CTX:
            logging.warning(
                f"PlantCAD2 model detected. Specified contextSize={args.contextSize} is below "
                f"the recommended minimum of {MIN_CTX}. Forcing contextSize={MIN_CTX}."
            )
            args.contextSize = MIN_CTX
            args.tokenIdx = args.contextSize // 2 - 1
    elif "plantcaduceus" in model_name_lower:
        FIXED_CTX = 512
        if args.contextSize != FIXED_CTX:
            logging.warning(
                f"PlantCaduceus (PlantCAD v1) model detected. Forcing contextSize={FIXED_CTX} "
                f"(was {args.contextSize})."
            )
            args.contextSize = FIXED_CTX
            args.tokenIdx = args.contextSize // 2 - 1

    # Determine workflow based on input type
    if args.inputVCF is not None:
        # VCF-based zero-shot scoring workflow
        sequences, recordIndices = seq_from_vcf(args)

        model, tokenizer = load_model_and_tokenizer(args.model, args.device)

        logging.info("Creating data loader")
        loader = create_dataloader(sequences, tokenizer, args.batchSize, args.tokenIdx)

        logits = extract_logits(model, loader, args.device, args.tokenIdx, tokenizer)

        markovLogProbs = None
        if args.markovTable is not None:
            markovLogProbs = markov_log_probs(load_markov_model(args), sequences, args.tokenIdx)

        zero_shot_score_vcf(args, recordIndices, logits, markovLogProbs)
        logging.info(f"Zero-shot scores saved to {args.output}")

    elif args.inputBed is not None:
        # BED-based genome-wide LLR scoring workflow
        logging.info("Starting genome-wide LLR scoring workflow.")
        
        # Load reference genome and BED file
        fasta_dict = load_fasta(args.inputFasta)
        bed_df = load_bed_file(args.inputBed)

        markov = load_markov_model(args) if args.markovTable is not None else None

        # Extract sequences for all positions in BED regions
        if markov is not None:
            sequences, position_info, window_sizes, markov_ctx = extract_sequences_from_bed(
                args, bed_df, fasta_dict, markov_k=markov.k)
        else:
            sequences, position_info, window_sizes = extract_sequences_from_bed(args, bed_df, fasta_dict)

        if not sequences:
            logging.error("No valid sequences extracted. Check BED file and FASTA file.")
            sys.exit(1)

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model, args.device)

        # Create dataloader with BED-specific settings
        loader = create_dataloader(sequences, tokenizer, args.batchSize, args.tokenIdx,
                                   args.stepSize, args.useMasking)

        # Extract logits/probabilities
        probs = extract_logits(model, loader, args.device, args.tokenIdx, tokenizer, 
                               args.stepSize, window_sizes)

        markov_logp = None
        if markov is not None:
            logging.info("Computing Markov background log-probabilities")
            left, right = zip(*markov_ctx)
            markov_logp = markov.log_probs(list(left), list(right))

        # Calculate LLRs
        results = calculate_genome_wide_llr(position_info, probs, args.aggregation, markov_logp)

        # Write output
        write_llr_output(results, args.output, args.aggregation, args.outputRawProb)

        logging.info(f"Genome-wide LLR calculation complete. Results saved to {args.output}")

    else:
        logging.error("No input file specified. Please use -input-vcf or -input-bed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
