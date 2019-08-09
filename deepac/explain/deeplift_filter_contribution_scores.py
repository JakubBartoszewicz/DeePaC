import argparse
import os
import csv
import numpy as np
import sys
import re

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import keras.backend as K

from Bio import SeqIO

from multiprocessing import Pool
from functools import partial
from shap.explainers.deep import DeepExplainer

sys.path.insert(1, os.path.join(sys.path[0], '..'))


def main():
    """
    Calculates DeepLIFT contribution scores for all neurons in the convolutional layer
    and extract all motifs for which a filter neuron got a non-zero contribution score.
    """

    # parse command line arguments
    args = parse_arguments()
    model = load_model(args.model)
    max_only = args.max_only
    if args.w_norm:
        print("Create model with mean-centered weight matrices ...")
        conv_layer_idx = [idx for idx, layer in enumerate(model.layers) if "Conv1D" in str(layer)][0]
        kernel_normed, bias_normed = normalize_filter_weights(model.get_layer(index=conv_layer_idx).get_weights()[0],
                                                              model.get_layer(index=conv_layer_idx).get_weights()[1])
        model.get_layer(index=conv_layer_idx).set_weights([kernel_normed, bias_normed])
        path = args.model
        if re.search("\.h5$", path) is not None:
            path = re.sub("\.h5$", "", path)
        norm_path = path + "_w_norm.h5"

        model.save(norm_path)
        args.model = norm_path


    # extract some model information
    conv_layer_idx = [idx for idx, layer in enumerate(model.layers) if "Conv1D" in str(layer)][0]
    n_filters = model.get_layer(index=conv_layer_idx).get_weights()[0].shape[-1]
    motif_length = model.get_layer(index=conv_layer_idx).get_weights()[0].shape[0]
    pad_left = (motif_length - 1) // 2
    pad_right = motif_length - 1 - pad_left

    print(model.summary())


    print("Loading test data (.npy) ...")
    test_data_set_name = os.path.splitext(os.path.basename(args.test_data))[0]
    samples = np.load(args.test_data, mmap_mode='r')
    total_num_reads = samples.shape[0]
    len_reads = samples.shape[1]

    print("Loading test data (.fasta) ...")
    nonpatho_reads = list(SeqIO.parse(args.nonpatho_test, "fasta"))
    patho_reads = list(SeqIO.parse(args.patho_test, "fasta"))
    reads = nonpatho_reads + patho_reads
    for idx, r in enumerate(reads):
        r.id = test_data_set_name + "_seq_" + str(idx) + "_" + os.path.basename(r.id)
        r.description = test_data_set_name + "_seq_" + str(idx) + "_" + os.path.basename(r.description)
        r.name = test_data_set_name + "_seq_" + str(idx) + "_" + os.path.basename(r.name)
    print("Padding reads ...")
    reads = ["N" * pad_left + r + "N" * pad_right for r in reads]

    assert len(reads) == total_num_reads, \
        "Test data in .npy-format and fasta files containing different number of reads!"

    # create output directory and subdirectories
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.out_dir + "/filter_scores/"):
        os.makedirs(args.out_dir + "/filter_scores/")
    if not os.path.exists(args.out_dir + "/fasta/"):
        os.makedirs(args.out_dir + "/fasta/")

    # load or create reference sequences
    ref_samples = get_reference_seqs(args, len_reads)
    num_ref_seqs = ref_samples.shape[0]

    print("Running DeepSHAP ...")
    chunk_size = 1000 // num_ref_seqs
    i = 0

    def map2layer(x, layer, out_node):
        feed_dict = dict(zip([model.get_layer(index=0).input], [x]))
        return K.get_session().run(model.get_layer(index=layer).get_output_at(out_node), feed_dict)

    explainer = DeepExplainer(([model.get_layer(index=conv_layer_idx).get_output_at(0),
                                model.get_layer(index=conv_layer_idx).get_output_at(1)],
                               model.layers[-1].output),
                              [map2layer(ref_samples, conv_layer_idx, 0),
                               map2layer(ref_samples, conv_layer_idx, 1)])



    cores = args.n_cpus
    p = Pool(processes=cores)

    while i < total_num_reads:

        print("Done "+str(i)+" from "+str(total_num_reads)+" sequences")
        samples_chunk = samples[i:i+chunk_size, :, :]
        reads_chunk = reads[i:i+chunk_size]

        scores_filter = explainer.shap_values([map2layer(samples_chunk, conv_layer_idx, 0),
                                               map2layer(samples_chunk, conv_layer_idx, 1)])
        scores_fwd, scores_rc = scores_filter[0]
        scores_filter = scores_fwd + scores_rc

        # shape: [num_reads, len_reads, n_filters]

        print("Saving data ...")
        # for each filter do:
        p.map(partial(get_filter_data, scores_filter_avg=scores_filter, input_reads=reads_chunk,
                      out_dir=args.out_dir, data_set_name=test_data_set_name, motif_len=motif_length,
                      max_only=max_only), range(n_filters))

        i += chunk_size


def get_filter_data(filter_id, scores_filter_avg, input_reads, motif_len, out_dir, data_set_name, max_only):
    # determine non-zero contribution scores per read and filter
    # and extract DNA-sequence of corresponding subreads
    num_reads = len(input_reads)
    contribution_scores = []
    motifs = []
    for seq_id in range(num_reads):
        if np.any(scores_filter_avg[seq_id, :, filter_id]):
            if max_only:
                non_zero_neurons = [np.argmax(scores_filter_avg[seq_id, :, filter_id])]
            else:
                non_zero_neurons = np.nonzero(scores_filter_avg[seq_id, :, filter_id])[0]
            scores = scores_filter_avg[seq_id, non_zero_neurons, filter_id]
            # contribution_scores.append((reads_chunk[seq_id].id,
            #                             non_zero_neurons,['%.4g' % n for n in scores]))
            # save scores with less precison to save memory
            contribution_scores.append((input_reads[seq_id].id, non_zero_neurons, scores))
            motifs.append([input_reads[seq_id][non_zero_neuron:(non_zero_neuron + motif_len)]
                           for non_zero_neuron in non_zero_neurons])

    if contribution_scores:

        # save filter contribution scores
        filter_rel_file = \
            out_dir + "/filter_scores/" + data_set_name + "_rel_filter_%d.csv" % filter_id
        with open(filter_rel_file, 'a') as csv_file:
            file_writer = csv.writer(csv_file)
            for dat in contribution_scores:
                file_writer.writerow([">" + dat[0]])
                file_writer.writerow(dat[1])
                file_writer.writerow(dat[2])

        # save subreads which cause non-zero contribution scores
        filter_motifs_file = \
            out_dir + "/fasta/" + data_set_name + "_motifs_filter_%d.fasta" % filter_id
        with open(filter_motifs_file, "a") as output_handle:
            SeqIO.write([subread for motif in motifs for subread in motif], output_handle, "fasta")

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file (.h5)")
    parser.add_argument("-b", "--w_norm", action="store_true",
                        help="Set flag if filter weight matrices should be mean-centered")
    parser.add_argument("-t", "--test_data", required=True, help="Test data (.npy)")
    parser.add_argument("-N", "--nonpatho_test", required=True,
                        help="Nonpathogenic reads of the test data set (.fasta)")
    parser.add_argument("-P", "--patho_test", required=True, help="Pathogenic reads of the test data set (.fasta)")
    parser.add_argument("-o", "--out_dir", default=".", help="Output directory")
    parser.add_argument("-r", "--ref_mode", default="N", choices=['N', 'GC', 'own_ref_file'],
                        help="Modus to calculate reference sequences")
    parser.add_argument("-a", "--train_data",
                        help="Train data (.npy), necessary to calculate reference sequences if ref_mode is 'GC'")
    parser.add_argument("-f", "--ref_seqs",
                        help="User provided reference sequences (.fasta) if ref_mode is 'own_ref_file'")
    parser.add_argument("-n", "--n_cpus", dest="n_cpus", default=8, type=int, help="Number of CPU cores")
    parser.add_argument("-M", "--max_only", dest="max_only", action="store_true",
                        help="Extract only the maximum contribution per filter per read")
    args = parser.parse_args()
    if args.ref_mode == "GC" and args.train_data is None:
        raise ValueError(
            "Training data (--train_data) is required to build reference sequences with the same GC-content!")
    if args.ref_mode == "own_ref_file" and args.ref_seqs is None:
        raise ValueError("File with own reference sequences (--ref_seqs) is missing!")

    return args


def get_reference_seqs(args, len_reads):
    """
    Load or create reference sequences for DeepLIFT.
    """
    # generate reference sequence with N's
    if args.ref_mode == "N":

        print("Generating reference sequence with all N's...")
        num_ref_seqs = 1
        ref_samples = np.zeros((num_ref_seqs, len_reads, 4))

    # create reference sequences with same GC content as the training data set
    elif args.ref_mode == "GC":

        print("Generating reference sequences with same GC-content as training data set...")
        train_samples = np.load(args.train_data, mmap_mode='r')
        num_ref_seqs = 5
        ref_seqs = [0]*num_ref_seqs
        # calculate frequency of each nucleotide (A,C,G,T,N) in the training data set
        probs = np.mean(np.mean(train_samples, axis=1), axis=0).tolist()
        probs.append(1-sum(probs))
        # generate reference seqs
        for i in range(num_ref_seqs):
            ref_seqs[i] = np.random.choice([0, 1, 2, 3, 4], p=probs, size=len_reads, replace=True)
        ref_samples = to_categorical(ref_seqs, num_classes=5)
        # remove channel of N-nucleotide
        ref_samples = ref_samples[:, :, 0:4]
        nc_dict = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
        train_data_set_name = os.path.splitext(os.path.basename(args.train_data))[0]
        # save reference sequences
        with open(args.out_dir + '/' + train_data_set_name + '_references.fasta', 'w') as csv_file:
            file_writer = csv.writer(csv_file)
            for seq_id in range(num_ref_seqs):
                file_writer.writerow([">"+train_data_set_name+"_ref_"+str(seq_id)])
                file_writer.writerow(["".join([nc_dict[base] for base in ref_seqs[seq_id]])])
        del train_samples

    # load own reference sequences (args.ref_mode == "own_ref_file")
    else:

        print("Loading reference sequences...")
        tokenizer = Tokenizer(char_level=True)
        tokenizer.fit_on_texts('ACGT')
        ref_reads = list(SeqIO.parse(args.ref_seqs, "fasta"))
        ref_samples = np.array([np.array([tokenizer.texts_to_matrix(read)]) for read in ref_reads])
        # remove unused character
        if not np.count_nonzero(ref_samples[:, :, :, 0]):
            ref_samples = ref_samples[:, :, :, 1:5]
        ref_samples = ref_samples.squeeze(1)
        # num_ref_seqs = ref_samples.shape[0]

    return ref_samples


def normalize_filter_weights(kernel, bias):
        """
        Performs output-preserving filter weight matrix normalization for 1-constrained inputs as described
        in "Learning Important Features Through Propagating Activation Differences" by Shrikumar et al., 2017
        """
        for filter_index in range(kernel.shape[-1]):
            bias[filter_index] += np.sum(np.mean(kernel[:, :, filter_index], axis=1))
            for pos in range(kernel.shape[0]):
                kernel[pos, :, filter_index] -= np.mean(kernel[pos, :, filter_index])
        return kernel, bias


if __name__ == "__main__":
    main()
