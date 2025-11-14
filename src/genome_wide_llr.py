import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import logging
from Bio import SeqIO
import gzip


def parse_args():
    parser = argparse.ArgumentParser(
        description="Development script for testing windowed genome-wide LLR calculation"
    )
    parser.add_argument("-input-fasta", dest="inputFasta", type=str, required=True,
                        help="The directory of reference genome fasta file")
    parser.add_argument("-input-bed", dest="inputBed", type=str, required=True,
                        help="The directory of BED file specifying regions to score")
    parser.add_argument("-output", dest="output", type=str, required=True,
                        help="The directory of output file")
    parser.add_argument("-model", dest="model", type=str, required=True,
                        help="The directory of pre-trained model")
    parser.add_argument("-device", dest="device", default="cuda:0",
                        help="The device to run the model (default: cuda:0)")
    parser.add_argument("-batchSize", dest="batchSize", default=128, type=int,
                        help="The batch size for the model (default: 128)")
    parser.add_argument("-contextSize", dest="contextSize", default=512, type=int,
                        help="The context window size (default: 512)")
    parser.add_argument("-step-size", dest="stepSize", default=1, type=int,
                        help="Number of positions to extract per window (1, 2, 4, 8, 16, 32, 64)")
    parser.add_argument("-use-masking", dest="useMasking", action="store_true", default=False,
                        help="Use masking approach (default: False, use unmasked logits)")
    parser.add_argument("-aggregation", dest="aggregation", default="average",
                        choices=["max", "average", "all"],
                        help="How to aggregate alternative allele scores: max, average (default), or all")
    parser.add_argument("-output-raw-prob", dest="outputRawProb", action="store_true", default=False,
                        help="Output raw probabilities for all four nucleotides")

    args = parser.parse_args()

    # Calculate token indices for center positions
    # For step_size=1: center is at contextSize//2 - 1
    # For step_size=4: centers are at [contextSize//2 - 2, -1, 0, 1] (middle 4 positions)
    args.tokenIdx = args.contextSize // 2 - 1

    return args


class SequenceDataset(Dataset):
    def __init__(self, sequences, tokenizer, tokenIdx, stepSize, useMasking):
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
            # Mask the center position(s)
            # Handle odd/even step sizes: for even, bias to the right
            # step_size=1: tokenIdx
            # step_size=2: tokenIdx, tokenIdx+1
            # step_size=3: tokenIdx-1, tokenIdx, tokenIdx+1
            # step_size=4: tokenIdx-1, tokenIdx, tokenIdx+1, tokenIdx+2
            start_offset = -(self.stepSize // 2) + (1 if self.stepSize % 2 == 0 else 0)
            for i in range(self.stepSize):
                mask_pos = self.tokenIdx + start_offset + i
                input_ids[0, mask_pos] = self.tokenizer.mask_token_id

        return {
            'sequence': sequence,
            'input_ids': input_ids
        }


def load_model_and_tokenizer(model_dir, device):
    logging.info(f"Loading model and tokenizer from {model_dir}")

    def get_optimal_dtype():
        if not torch.cuda.is_available():
            logging.info("Using float32 as no GPU is available.")
            return torch.float32

        device_index = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device_index)

        if capability[0] >= 8:
            logging.info("Using bfloat16 as the GPU supports sm_80 or higher.")
            return torch.bfloat16
        elif capability[0] >= 6:
            logging.info("Using float16 as the GPU supports sm_60 or higher.")
            return torch.float16
        else:
            logging.info("Using float32 as the GPU does not support float16 or bfloat16.")
            return torch.float32

    optimal_dtype = get_optimal_dtype()

    try:
        model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=optimal_dtype)
        model.to(optimal_dtype)
    except Exception as e:
        logging.error(f"Failed to load model with {optimal_dtype}, falling back to float32. Error: {e}")
        model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)
    return model, tokenizer


def create_dataloader(sequences, tokenizer, batch_size, tokenIdx, stepSize, useMasking):
    logging.info(f"Creating DataLoader with batch size {batch_size}")
    dataset = SequenceDataset(sequences, tokenizer, tokenIdx, stepSize, useMasking)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def extract_logits(model, dataloader, device, tokenIdx, tokenizer, stepSize, window_sizes):
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

        # Extract logits for center position(s) based on actual window sizes
        # Handle odd/even step sizes: for even, bias to the right
        # step_size=1: tokenIdx
        # step_size=2: tokenIdx, tokenIdx+1
        # step_size=3: tokenIdx-1, tokenIdx, tokenIdx+1
        # step_size=4: tokenIdx-1, tokenIdx, tokenIdx+1, tokenIdx+2
        start_offset = -(stepSize // 2) + (1 if stepSize % 2 == 0 else 0)

        for b in range(batch_size):
            actual_window_size = window_sizes[window_idx]
            window_probs = []

            for i in range(actual_window_size):
                pos = tokenIdx + start_offset + i
                logits = all_logits[b, pos, [tokenizer.get_vocab()[nc] for nc in nucleotides]]
                probs = torch.nn.functional.softmax(logits.cpu(), dim=0).numpy()
                window_probs.append(probs)

            results.extend(window_probs)
            window_idx += 1

    # Return as array: [total_positions, 4]
    return np.array(results)


def load_bed_file(bed_path):
    """Load BED file and return DataFrame with chr, start, end columns"""
    logging.info(f"Reading BED file from {bed_path}")
    bed_df = pd.read_csv(bed_path, sep='\t', header=None,
                         names=['chr', 'start', 'end'], usecols=[0, 1, 2])
    return bed_df


def extract_sequences_from_bed(args, bed_df, fasta_dict):
    """Extract sequences using windowed approach"""
    logging.info("Extracting sequences from BED regions with windowed approach")

    sequences = []
    position_info = []  # Store chr, pos, ref_allele for each position
    window_sizes = []  # Track actual number of valid positions per window

    addIdx = args.contextSize - args.tokenIdx

    logging.info(f"Using step_size={args.stepSize}, masking={args.useMasking}")

    for _, row in tqdm(bed_df.iterrows(), total=len(bed_df)):
        chrom = str(row['chr'])
        start = int(row['start'])
        end = int(row['end'])

        if chrom not in fasta_dict:
            logging.warning(f"Chromosome {chrom} not found in FASTA file, skipping region")
            continue

        # Slide window by step_size
        # For step_size=4: window centers on positions [start, start+1, start+2, start+3], then [start+4, ...], etc.
        for window_start in range(start, end, args.stepSize):
            window_end = min(window_start + args.stepSize, end)
            num_positions = window_end - window_start

            if num_positions == 0:
                continue

            # Center the window on the middle of these positions
            # For step_size=4: center between position 1 and 2 (positions 0,1,2,3)
            center_pos = window_start + (num_positions - 1) / 2.0
            center_pos_int = int(center_pos)

            try:
                # Extract reference alleles for all positions in this window
                window_refs = []
                window_positions = []

                for pos in range(window_start, window_end):
                    ref_allele = str(fasta_dict[chrom].seq[pos]).upper()
                    if ref_allele not in ['A', 'C', 'G', 'T']:
                        continue
                    window_refs.append(ref_allele)
                    window_positions.append(pos)

                if len(window_refs) == 0:
                    continue

                # Extract sequence context centered on this window
                seq_start = center_pos_int - args.tokenIdx
                seq_end = center_pos_int + addIdx

                if seq_start < 0:
                    seq = str(fasta_dict[chrom].seq[0:seq_end]).upper().rjust(args.contextSize, "N")
                else:
                    seq = str(fasta_dict[chrom].seq[seq_start:seq_end]).upper().ljust(args.contextSize, "N")

                sequences.append(seq)
                window_sizes.append(len(window_refs))  # Track actual window size

                # Store info for each position in this window
                for pos, ref in zip(window_positions, window_refs):
                    position_info.append({
                        'chr': chrom,
                        'pos': pos,
                        'ref': ref
                    })

            except Exception as e:
                logging.warning(f"Error processing window {chrom}:{window_start}-{window_end}, skipping. Error: {e}")
                continue

    logging.info(f"Extracted {len(sequences)} windows covering {len(position_info)} positions")
    return sequences, position_info, window_sizes


def calculate_genome_wide_llr(position_info, probs_array, aggregation):
    """Calculate log likelihood ratios for each position

    probs_array shape: [num_positions, 4]
    position_info: list of dicts with chr, pos, ref
    """
    logging.info("Calculating genome-wide log likelihood ratios")

    nucleotides = ['A', 'C', 'G', 'T']
    results = []

    # Ensure we have the same number of positions
    assert len(probs_array) == len(position_info), f"Mismatch: {len(probs_array)} probs vs {len(position_info)} positions"

    for pos_info, probs in zip(position_info, probs_array):
        ref_allele = pos_info['ref']
        ref_idx = nucleotides.index(ref_allele)
        ref_prob = probs[ref_idx]

        # Get alternative allele indices
        alt_indices = [i for i in range(4) if i != ref_idx]
        alt_probs = [probs[i] for i in alt_indices]
        alt_alleles = [nucleotides[i] for i in alt_indices]

        # Calculate LLR for each alternative
        llrs = [np.log(alt_prob / ref_prob) for alt_prob in alt_probs]

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

        # Aggregate LLRs based on method
        if aggregation == "max":
            result['score'] = max(llrs)
        elif aggregation == "average":
            result['score'] = np.mean(llrs)
        elif aggregation == "all":
            result['scores'] = llrs

        results.append(result)

    return results


def write_output(results, output_path, aggregation, output_raw_prob):
    """Write results to output file"""
    logging.info(f"Writing results to {output_path}")

    with open(output_path, 'w') as f:
        # Write header
        if aggregation == "all":
            if output_raw_prob:
                f.write("chr\tstart\tend\tref\talt_alleles\tscores\tref_prob\talt_probs\n")
            else:
                f.write("chr\tstart\tend\tref\talt_alleles\tscores\n")
        else:
            if output_raw_prob:
                f.write("chr\tstart\tend\tref\tscore\tref_prob\talt_probs\n")
            else:
                f.write("chr\tstart\tend\tref\tscore\n")

        for result in results:
            chr_name = result['chr']
            start = result['start']
            end = result['end']
            ref = result['ref']

            if aggregation == "all":
                # Output three scores, one for each alternative allele
                scores_str = ','.join([f"{score:.6f}" for score in result['scores']])
                alt_str = ','.join(result['alt_alleles'])

                if output_raw_prob:
                    ref_prob_str = f"{result['ref_prob']:.6f}"
                    alt_probs_str = ','.join([f"{p:.6f}" for p in result['alt_probs']])
                    f.write(f"{chr_name}\t{start}\t{end}\t{ref}\t{alt_str}\t{scores_str}\t{ref_prob_str}\t{alt_probs_str}\n")
                else:
                    f.write(f"{chr_name}\t{start}\t{end}\t{ref}\t{alt_str}\t{scores_str}\n")
            else:
                # Output single aggregated score
                score = result['score']

                if output_raw_prob:
                    ref_prob_str = f"{result['ref_prob']:.6f}"
                    alt_probs_str = ','.join([f"{p:.6f}" for p in result['alt_probs']])
                    f.write(f"{chr_name}\t{start}\t{end}\t{ref}\t{score:.6f}\t{ref_prob_str}\t{alt_probs_str}\n")
                else:
                    f.write(f"{chr_name}\t{start}\t{end}\t{ref}\t{score:.6f}\n")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    args = parse_args()

    # Load reference genome
    logging.info(f"Loading reference genome from {args.inputFasta}")
    if args.inputFasta.endswith(".gz"):
        with gzip.open(args.inputFasta, "rt") as file:
            fasta_dict = SeqIO.to_dict(SeqIO.parse(file, "fasta"))
    else:
        fasta_dict = SeqIO.to_dict(SeqIO.parse(args.inputFasta, "fasta"))

    # Load BED file
    bed_df = load_bed_file(args.inputBed)

    # Extract sequences for all positions in BED regions
    sequences, position_info, window_sizes = extract_sequences_from_bed(args, bed_df, fasta_dict)

    if len(sequences) == 0:
        logging.error("No valid sequences extracted. Check BED file and FASTA file.")
        return

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)

    # Create dataloader
    loader = create_dataloader(sequences, tokenizer, args.batchSize, args.tokenIdx,
                               args.stepSize, args.useMasking)

    # Extract logits
    probs = extract_logits(model, loader, args.device, args.tokenIdx, tokenizer, args.stepSize, window_sizes)

    # Calculate LLRs
    results = calculate_genome_wide_llr(position_info, probs, args.aggregation)

    # Write output
    write_output(results, args.output, args.aggregation, args.outputRawProb)

    logging.info(f"Genome-wide LLR calculation complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
