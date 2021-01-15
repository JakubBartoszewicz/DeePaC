import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import math
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer


def frag_genomes(args):
    """Fragment a genome into pseudo-reads."""
    # create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for file in os.listdir(args.genomes_dir):

        if file.endswith(".fasta") or file.endswith(".fna"):
            print("Processing " + file + " ...")
            # load sequences
            instances = []
            with open(args.genomes_dir + "/" + file, "rU") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    instances.append(record)

            fragmented_genomes = []
            # per sequence do overlapping fragmentation
            for instance in instances:

                seq = instance.seq
                seq_len = len(seq)

                num_reads = max(1, math.ceil(float(seq_len - args.read_len + args.shift)/float(args.shift)))
                covered_bases = args.read_len + args.shift*(num_reads - 1)

                # pad sequence (if necessary)
                pad_left = (covered_bases - seq_len) // 2
                pad_right = covered_bases - seq_len - pad_left
                seq_pad = "N"*pad_left + seq + "N"*pad_right

                for i in range(num_reads):
                    # use start and end positions like in bed files (0-indexed, end-position excluded)
                    start = i*args.shift
                    end = start + args.read_len
                    new_record = SeqRecord(seq_pad[start:end],
                                           id=instance.id + ":" + str(start-pad_left) + ".." + str(end-pad_left),
                                           name=instance.id + ":" + str(start-pad_left) + ".." + str(end-pad_left),
                                           description="")
                    fragmented_genomes.append(new_record)

            SeqIO.write(fragmented_genomes, args.out_dir + "/" +
                        os.path.splitext(file)[0] + "_fragmented_genomes.fasta", "fasta")
            # save seqs as .npy arrays
            tokenizer = Tokenizer(char_level=True)
            tokenizer.fit_on_texts('ACGT')
            records = np.array([np.array([tokenizer.texts_to_matrix(record.seq)]) for record in fragmented_genomes])
            # remove unused character
            if not np.count_nonzero(records[:, :, :, 0]):
                records = records[:, :, :, 1:5]
            records = records.squeeze(1)
            np.save(args.out_dir + "/" + os.path.splitext(file)[0] + "_fragmented_genomes.npy", records)
