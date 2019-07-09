import argparse
import os
import csv
import numpy as np
import re
import sys

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from Bio import SeqIO

import deeplift
from deepac.visuals.deeplift_filtering_functions import *
from deeplift.conversion import kerasapi_conversion as conversion
sys.path.insert(1, os.path.join(sys.path[0], '..'))


def main():
    """
    Calculates contribution scores on the nucleotide level for each motif
    recognized by a filter which received a non-zero DeepLIFT contribution score.
    """

    # parse command line arguments
    args = parse_arguments()

    # normalize filter weight matrix + modify bias
    if args.w_norm:
        print("Create model with mean-centered weight matrices ...")
        model = load_model(args.model)
        conv_layer_idx = [idx for idx, layer in enumerate(model.layers) if "Conv1D" in str(layer)][0]
        kernel_normed, bias_normed = normalize_filter_weights(model.get_layer(index=conv_layer_idx).get_weights()[0],
                                                              model.get_layer(index=conv_layer_idx).get_weights()[1])
        model.get_layer(index=conv_layer_idx).set_weights([kernel_normed, bias_normed])
        model.save(os.path.dirname(args.model) + "/" + os.path.splitext(os.path.basename(args.model))[0] + "_w_norm.h5")
        args.model = \
            os.path.dirname(args.model) + "/" + os.path.splitext(os.path.basename(args.model))[0] + "_w_norm.h5"

    # convert keras model to deeplift model
    deeplift_model = conversion.convert_model_from_saved_files(args.model,
                                                               nonlinear_mxts_mode=deeplift.layers.
                                                               NonlinearMxtsMode.DeepLIFT_GenomicsDefault)
    # extract some model information
    conv_layer_idx = [type(layer).__name__ for layer in deeplift_model.get_layers()].index("Conv1D")
    n_filters = deeplift_model.get_layers()[conv_layer_idx].kernel.shape[-1]
    motif_length = deeplift_model.get_layers()[conv_layer_idx].kernel.shape[0]
    pad_left = (motif_length - 1) // 2
    pad_right = motif_length - 1 - pad_left

    kernel = deeplift_model.get_layers()[conv_layer_idx].kernel

    # compile scoring function (find multipliers of convolutional layer
    # w.r.t. the layer preceding the sigmoid output layer)
    deeplift_mxts_func_filter = deeplift_model.get_target_multipliers_func(find_scores_layer_idx=conv_layer_idx,
                                                                           target_layer_idx=-2)

    print("Loading test data (.npy) ...")
    test_data_set_name = os.path.splitext(os.path.basename(args.test_data))[0]
    samples = np.load(args.test_data, mmap_mode='r')
    len_reads = samples.shape[1]

    # load or create reference sequences
    ref_samples = get_reference_seqs(args, len_reads)
    num_ref_seqs = ref_samples.shape[0]

    # create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # define placeholders
    conv_layer = deeplift_model.get_layers()[conv_layer_idx]
    pos_mxts_conv = tf.placeholder(shape=conv_layer._pos_mxts.get_shape(), dtype=tf.float32)
    neg_mxts_conv = tf.placeholder(shape=conv_layer._pos_mxts.get_shape(), dtype=tf.float32)
    diff_from_ref_input = tf.placeholder(shape=conv_layer.inputs.get_activation_vars().get_shape(), dtype=tf.float32)
    filter_mask_pl = tf.placeholder(shape=conv_layer._pos_mxts.get_shape(), dtype=tf.float32)

    # define operations to compute contribution scores of the input layer
    # after performing filtering in the convolutional layer
    contribution_scores_input_filtering = \
        get_contribs_of_inputs_after_filtering_conv_layer(conv_layer, diff_from_ref_input, pos_mxts_conv,
                                                          neg_mxts_conv, filter_mask_pl, kernel)
    for filter_index in range(n_filters):

        # file contains all non-zero contribution scores per read and filter
        file = args.in_dir + "/" + test_data_set_name + "_rel_filter_%d.csv" % filter_index

        # skip filters which did nor receive any non-zero contribution score
        if not os.path.exists(file):
            continue

        print('Processing filter %d' % filter_index)

        # save reads for which the filter got non-zero contribution scores
        read_ids = []
        # save position in the read for which the filter got a non-zero contribution score
        motif_starts = []
        with open(file, 'r') as csvfile:

            reader = csv.reader(csvfile)
            for ind, row in enumerate(reader):
                if ind % 3 == 0:
                    read_id = re.search("seq_[0-9]+", row[0]).group()
                    read_id = int(read_id.replace("seq_", ""))
                    read_ids.append(read_id)
                elif ind % 3 == 1:
                    motif_start = int(row[0])
                    motif_starts.append(motif_start)

        # select all reads for which the filter got a non-zero contribution score
        samples_filter = samples[read_ids, :, :]
        num_reads_filter = len(read_ids)
        chunk_size = 30000
        i = 0
        while i < num_reads_filter:

            print("Done " + str(i) + " from " + str(num_reads_filter))
            samples_chunk = samples_filter[i:i+chunk_size, :, :]
            num_reads = samples_chunk.shape[0]
            input_data_list = np.repeat(samples_chunk, num_ref_seqs, axis=0)
            input_references_list = np.concatenate([ref_samples]*num_reads, axis=0)
            delta = input_data_list-input_references_list

            # compute positive and negative multipliers of the convolutional layer
            # w.r.t. the layer preceding the sigmoid output layer
            mxts = np.array(deeplift_mxts_func_filter(task_idx=0, input_data_list=[input_data_list],
                                                      input_references_list=[input_references_list],
                                                      progress_update=10000,
                                                      batch_size=25))
            pos_mxts = mxts[:, :, :n_filters]
            neg_mxts = mxts[:, :, n_filters:]

            # build filter mask
            # filter_conv = [[motif_start, filter_index] for motif_start in motif_starts[i:i+chunk_size]]
            filter_conv = np.stack((motif_starts[i:i+chunk_size], [filter_index]*num_reads), axis=1)
            filter_conv = np.repeat(filter_conv, num_ref_seqs, 0)
            filter_mask = np.zeros_like(pos_mxts)
            for j in range(num_reads*num_ref_seqs):
                filter_mask[j, filter_conv[j][0], filter_conv[j][1]] = 1

            # compute contribution scores of the input layer
            # w.r.t. the layer preceding the sigmoid output layer after performing filtering in the convolutional layer
            scores_input = np.array(deeplift.util.get_session()
                                    .run(contribution_scores_input_filtering,
                                         feed_dict={diff_from_ref_input: delta,
                                                    pos_mxts_conv: pos_mxts,
                                                    neg_mxts_conv: neg_mxts,
                                                    filter_mask_pl: filter_mask}))
            # average the results per ref sequence
            scores_input = np.reshape(scores_input, [num_reads, num_ref_seqs, len_reads, 4])
            scores_input = np.mean(scores_input, axis=1)

            # sum up scores along the nucleotide axis
            scores_input = np.sum(scores_input, axis=2)

            # add zeros for padding
            scores_input_pad = np.pad(scores_input,
                                      ((0, 0), (pad_left, pad_right)), 'constant', constant_values=(0.0, 0.0))

            # save contribution scores for each nucleotide per filter motif
            with open(args.out_dir + "/" + test_data_set_name + "_rel_filter_%d_nucleotides.csv" % filter_index, 'a') \
                    as csv_file:
                file_writer = csv.writer(csv_file)
                for ind, seq_id in enumerate(read_ids[i:i+chunk_size]):
                    row = ['%.4g' % s
                           for s in scores_input_pad[ind, motif_starts[ind+i]:(motif_starts[ind+i] + motif_length)]]
                    file_writer.writerow([">" + test_data_set_name + "_seq_" + str(seq_id)])
                    file_writer.writerow(row)

            i += chunk_size


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file (.h5)")
    parser.add_argument("-b", "--w_norm", action="store_true",
                        help="Set flag if filter weights should be mean-centered")
    parser.add_argument("-t", "--test_data", required=True, help="Test data (.npy)")
    parser.add_argument("-i", "--in_dir", required=True, help="Directory with the non-zero DeepLIFT scores per filter")
    parser.add_argument("-o", "--out_dir", default=".", help="Output directory")
    parser.add_argument("-r", "--ref_mode", default="N", choices=['N', 'own_ref_file'],
                        help="Modus to calculate reference sequences")
    parser.add_argument("-f", "--ref_seqs",
                        help="User provided reference sequences (.fasta) if ref_mode is 'own_ref_file'")
    args = parser.parse_args()
    if args.ref_mode == "own_ref_file" and args.ref_seqs is None:
        raise ValueError("File with own reference sequences (--ref_seqs) is missing!")
    if args.w_norm:
        print("Mean-centered filter weights will be used during forward and backward pass ...")
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
