import argparse
import os
import csv
import numpy as np
import sys

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import keras.backend as K

from Bio import SeqIO

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
    if args.w_norm:
        print("Create model with mean-centered weight matrices ...")
        conv_layer_idx = [idx for idx, layer in enumerate(model.layers) if "Conv1D" in str(layer)][0]
        kernel_normed, bias_normed = normalize_filter_weights(model.get_layer(index=conv_layer_idx).get_weights()[0],
                                                              model.get_layer(index=conv_layer_idx).get_weights()[1])
        model.get_layer(index=conv_layer_idx).set_weights([kernel_normed, bias_normed])
        model.save(os.path.dirname(args.model) + "/" + os.path.splitext(os.path.basename(args.model))[0] + "_w_norm.h5")
        args.model = \
            os.path.dirname(args.model) + "/" + os.path.splitext(os.path.basename(args.model))[0] + "_w_norm.h5"


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
    samples = np.concatenate((samples[:50,:,:], samples[-50:,:,:]))
    total_num_reads = samples.shape[0]
    len_reads = samples.shape[1]

    print("Loading test data (.fasta) ...")
    nonpatho_reads = list(SeqIO.parse(args.nonpatho_test, "fasta"))[:50]
    patho_reads = list(SeqIO.parse(args.patho_test, "fasta"))[:50]
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
    chunk_size = 100000 // num_ref_seqs
    i = 0

    def map2layer(x, layer, out_node):
        feed_dict = dict(zip([model.get_layer(index=0).input], [x]))
        return K.get_session().run(model.get_layer(index=layer).get_output_at(out_node), feed_dict)

    la = map2layer(ref_samples, conv_layer_idx, 0)

    # explainer_fwd = DeepExplainer((model.get_layer(index=conv_layer_idx).get_output_at(0), model.layers[-1].output),
    #                             map2layer(ref_samples, conv_layer_idx, 0))

    explainer = DeepExplainer(([model.get_layer(index=conv_layer_idx).get_output_at(0),
                                model.get_layer(index=conv_layer_idx).get_output_at(1)],
                               model.layers[-1].output),
                              [map2layer(ref_samples, conv_layer_idx, 0),
                               map2layer(ref_samples, conv_layer_idx, 1)])


    while i < total_num_reads:

        print("Done "+str(i)+" from "+str(total_num_reads)+" sequences")
        samples_chunk = samples[i:i+chunk_size, :, :]
        reads_chunk = reads[i:i+chunk_size]
        num_reads = samples_chunk.shape[0]

        scores_filter_avg = explainer.shap_values([map2layer(samples_chunk, conv_layer_idx, 0),
                                                   map2layer(samples_chunk, conv_layer_idx, 1)])

        # scores_filter = np.reshape(scores_filter, [num_reads, num_ref_seqs, len_reads, n_filters])

        # average the results per ref sequence
        # scores_filter_avg = np.mean(scores_filter, axis=1)
        print("Saving data ...")
        for filter_index in range(n_filters):
            # determine non-zero contribution scores per read and filter
            # and extract DNA-sequence of corresponding subreads
            contribution_scores = []
            motifs = []
            for seq_id in range(num_reads):
                if np.any(scores_filter_avg[seq_id, :, filter_index]):
                    non_zero_neurons = np.nonzero(scores_filter_avg[seq_id, :, filter_index])[0]
                    scores = scores_filter_avg[seq_id, non_zero_neurons, filter_index]
                    # contribution_scores.append((reads_chunk[seq_id].id,
                    #                             non_zero_neurons,['%.4g' % n for n in scores]))
                    # save scores with less precison to save memory
                    contribution_scores.append((reads_chunk[seq_id].id, non_zero_neurons, scores))
                    motifs.append([reads_chunk[seq_id][non_zero_neuron:(non_zero_neuron+motif_length)]
                                   for non_zero_neuron in non_zero_neurons])

            if contribution_scores:

                # save filter contribution scores
                filter_rel_file = \
                    args.out_dir + "/filter_scores/" + test_data_set_name + "_rel_filter_%d.csv" % filter_index
                with open(filter_rel_file, 'a') as csv_file:
                    file_writer = csv.writer(csv_file)
                    for dat in contribution_scores:
                        file_writer.writerow([">"+dat[0]])
                        file_writer.writerow(dat[1])
                        file_writer.writerow(dat[2])

                # save subreads which cause non-zero contribution scores
                filter_motifs_file = \
                    args.out_dir + "/fasta/" + test_data_set_name + "_motifs_filter_%d.fasta" % filter_index
                with open(filter_motifs_file, "a") as output_handle:
                    SeqIO.write([subread for motif in motifs for subread in motif], output_handle, "fasta")

        i += chunk_size


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
