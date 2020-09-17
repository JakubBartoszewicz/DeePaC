import re
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import tensorflow as tf
from Bio import SeqIO
import pandas as pd
from collections import OrderedDict
from shap import DeepExplainer, GradientExplainer
from deepac.utils import set_mem_growth
from deepac.explain.filter_contribs import get_reference_seqs


def nt_map(args, allow_eager=False):
    """Create bedgraph files per genome which show the pathogenicity prediction score over all genomic positions."""
    # create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    ref_samples = get_reference_seqs(args, args.read_length)
    if tf.executing_eagerly() and not allow_eager:
        print("Using SHAP. Disabling eager execution...")
        tf.compat.v1.disable_v2_behavior()
    set_mem_growth()
    model = load_model(args.model)
    if args.gradient:
        explainer = GradientExplainer(model, ref_samples)
    else:
        explainer = DeepExplainer(model, ref_samples)
    check_additivity = not args.no_check
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
            records = np.array([tokenizer.texts_to_matrix(record.seq).astype("int32")[:, 1:] for record in fragments])

            chunk_size = args.chunk_size
            i = 0
            scores_nt_chunks = []
            while i < num_fragments:
                if args.gradient:
                    contribs_chunk = explainer.shap_values(records[i:i + chunk_size, :])[0]
                else:
                    contribs_chunk = \
                        explainer.shap_values(records[i:i+chunk_size, :], check_additivity=check_additivity)[0]
                scores_nt_chunk = np.sum(contribs_chunk, axis=-1)
                scores_nt_chunks.append(scores_nt_chunk)
                i = i + chunk_size
                print("Done " + str(min(i, num_fragments)) + " from " + str(num_fragments) + " sequences")
            scores_nt = np.vstack(scores_nt_chunks)

            # load genome size
            genome_info_file = args.genomes_dir + "/" + re.split("_fragmented_genomes", genome)[0] + ".genome"
            if not os.path.isfile(genome_info_file):
                print("Skipping " + genome + " since .genome file is missing!")
                continue

            genome_info = pd.read_csv(genome_info_file, sep="\t", index_col=0, header=None)

            # prepare output table
            df = pd.DataFrame()

            # save pathogenicity score for each nucleotide of all contigs of that genome
            genome_patho_dict = OrderedDict()
            # count by how many reads each nucleotide is covered
            genome_read_counter_dict = OrderedDict()

            # build bed graph file representing pathogenicity over genome
            for fragment_idx in range(num_fragments):

                seq_name, start_f, end_f = re.split(":|\.\.", fragments[fragment_idx].id)
                contig_len = int(genome_info.loc[seq_name])
                start = max(0, int(start_f))
                end = min(int(end_f), contig_len)

                if seq_name not in genome_patho_dict:
                    genome_patho_dict[seq_name] = np.zeros(contig_len)
                    genome_read_counter_dict[seq_name] = np.zeros(contig_len)
                try:
                    genome_patho_dict[seq_name][start:end] += \
                        scores_nt[fragment_idx, start-int(start_f):end-int(start_f)]
                except ValueError as err:
                    print(err)
                    print("Error. Please check if the genome length matches its description in the .genome/.gff3 file.")
                    break
                genome_read_counter_dict[seq_name][start:end] += 1

            for seq_name, genome_read_counter in genome_read_counter_dict.items():

                # compute mean pathogenicity score per nucleotide
                genome_patho_dict[seq_name] /= genome_read_counter

                # convert array of nucelotde pathogenicity scores to intervals (-> bedgraph format)
                scores = genome_patho_dict[seq_name]
                interval_starts = np.arange(scores.shape[0], dtype='int32')
                interval_ends = np.arange(scores.shape[0], dtype='int32') + 1
                df_s = pd.DataFrame(OrderedDict((('seq_name', [seq_name]*scores.shape[0]), ('start', interval_starts),
                                                 ('end', interval_ends),
                                                 ('score', scores))))
                df = df.append(df_s, ignore_index=True)

            # save results
            out_file = args.out_dir + "/" + genome + "_nt_contribs_map.bedgraph"
            df[['start', 'end']] = df[['start', 'end']].astype(int)
            df.to_csv(out_file, sep="\t", index=False, header=False)
