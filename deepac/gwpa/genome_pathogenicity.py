import re
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from Bio import SeqIO
import pandas as pd
from collections import OrderedDict
from operator import itemgetter


def genome_map(args):
    """Create bedgraph files per genome which show the pathogenicity prediction score over all genomic positions."""
    # create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    mean_scores = []

    # for each fragmented genome do
    for fragments_file in os.listdir(args.dir_fragmented_genomes):

        if fragments_file.endswith(".fasta") or fragments_file.endswith(".fna"):

            genome = os.path.splitext(os.path.basename(fragments_file))[0]
            print("Processing " + genome + " ...")
            # load fragments in fasta format
            tokenizer = Tokenizer(char_level=True)
            tokenizer.fit_on_texts('ACGT')
            fragments = list(SeqIO.parse(args.dir_fragmented_genomes + "/" + fragments_file, "fasta"))
            num_fragments = len(fragments)

            # load predictions per fragment
            preds_file = args.dir_fragmented_genomes_preds + "/" + genome + "_predictions.npy"
            preds = np.load(preds_file)

            assert num_fragments == len(preds), \
                print("Something went wrong! Number fragments in fasta file and predictions differ ...")

            # load genome size
            genome_info_file = args.genomes_dir + "/" + re.split("_fragmented_genomes", genome)[0] + ".genome"
            if not os.path.isfile(genome_info_file):
                print("Skipping " + genome + " since .genome file is missing!")
                continue

            genome_info = pd.read_csv(genome_info_file, sep="\t", index_col=0, header=None)

            # prepare output table
            df = pd.DataFrame(OrderedDict((('seq_name', ""),
                                           ('start', np.zeros(num_fragments, dtype='int32')),
                                           ('end', np.zeros(num_fragments, dtype='int32')),
                                           ('score', np.zeros(num_fragments)))))

            # save pathogenicity score for each nucleotide of all contigs of that genome
            genome_patho_dict = OrderedDict()
            # count by how many reads each nucleotide is covered
            genome_read_counter_dict = OrderedDict()

            # build bed graph file representing pathogenicity over genome
            for fragment_idx in range(num_fragments):

                seq_name, start, end = re.split(":|\.\.", fragments[fragment_idx].id)
                strain_len = int(genome_info.loc[seq_name])
                start = max(0, int(start))
                end = min(int(end), strain_len)
                score = preds[fragment_idx]

                if seq_name not in genome_patho_dict:
                    genome_patho_dict[seq_name] = np.zeros(strain_len)
                    genome_read_counter_dict[seq_name] = np.zeros(strain_len)

                genome_patho_dict[seq_name][start:end] += score
                genome_read_counter_dict[seq_name][start:end] += 1

            c = 0
            for seq_name, genome_read_counter in genome_read_counter_dict.items():

                # compute mean pathogenicity score per nucleotide
                genome_patho_dict[seq_name] /= genome_read_counter
                genome_patho_dict[seq_name] = genome_patho_dict[seq_name] - 0.5

                # convert array of nucelotide pathogenicity scores to intervals (-> bedgraph format)
                strain_len = int(genome_info.loc[seq_name])
                start_interval = 0
                score_interval = 0
                for start, score in enumerate(genome_patho_dict[seq_name]):
                    if start == 0:
                        score_interval = score

                    elif start == strain_len - 1 and score == score_interval:
                        end_interval = start + 1
                        df.loc[c] = [seq_name, start_interval, end_interval, score_interval]
                        c += 1

                    elif start == strain_len - 1 and score != score_interval:
                        end_interval = start
                        df.loc[c] = [seq_name, start_interval, end_interval, score_interval]
                        c += 1
                        start_interval = start
                        score_interval = score
                        end_interval = start + 1
                        df.loc[c] = [seq_name, start_interval, end_interval, score_interval]
                        c += 1

                    # new interval with different scores
                    elif score != score_interval:
                        end_interval = start
                        df.loc[c] = [seq_name, start_interval, end_interval, score_interval]
                        c += 1
                        start_interval = start
                        score_interval = score

            # save results
            out_file = args.out_dir + "/" + genome + "_pathogenicity.bedgraph"
            df[['start', 'end']] = df[['start', 'end']].astype(int)
            df.to_csv(out_file, sep="\t", index=False, header=False)

            mean_score = 0.5 + sum(x * y for x, y in zip(df.score, df.end-df.start)) / sum(df.end-df.start)
            mean_scores.append((genome.replace("_fragmented_genomes", ""), mean_score))

        mean_scores.sort(key=itemgetter(1))
        with open(args.out_dir + "/mean_patho.txt", 'w') as f:
            for name, score in mean_scores:
                f.write("{n}\t{s}\n".format(n=name, s=score))
