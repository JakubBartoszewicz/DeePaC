import os
import re
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf
from Bio import SeqIO
from deepac.utils import set_mem_growth
import pandas as pd
from math import floor, log10


def filter_activations(args):
    """Compute activation values genome-wide."""

    # Creates the model and loads weights
    set_mem_growth()

    model = load_model(args.model)
    conv_layer_ids = [idx for idx, layer in enumerate(model.layers) if "Conv1D" in str(layer)]
    conv_layer_idx = conv_layer_ids[args.inter_layer - 1]
    motif_length = model.get_layer(index=conv_layer_idx).get_weights()[0].shape[0]
    pad_left = (motif_length - 1) // 2
    pad_right = motif_length - 1 - pad_left

    print("Loading test data (.npy) ...")
    test_data_set_name = os.path.splitext(os.path.basename(args.test_data))[0]
    samples = np.load(args.test_data, mmap_mode='r')
    total_num_reads = samples.shape[0]

    print("Loading test data (.fasta) ...")
    reads = list(SeqIO.parse(args.test_fasta, "fasta"))
    assert len(reads) == total_num_reads, \
        "Test data in .npy-format and fasta files containing different number of reads!"

    print("Padding reads ...")
    reads = ["N" * pad_left + r + "N" * pad_right for r in reads]
    # extract genome_id, genomic start and end positions of the reads
    reads_info = []
    for r in reads:
        r_info = re.split(">|:|\.\.", r.id)
        reads_info.append([r_info[0], int(r_info[1]), int(r_info[2])])

    # create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    # Specify input and output of the network

    if tf.executing_eagerly():
        model = tf.keras.Model(model.inputs,
                               (model.get_layer(index=conv_layer_idx).get_output_at(0),
                                model.get_layer(index=conv_layer_idx).get_output_at(1)))
        iterate_fwd = None
        iterate_rc = None
    else:
        # Specify input and output of the network
        input_img = model.layers[0].input

        layer_output_fwd = model.get_layer(index=conv_layer_idx).get_output_at(0)
        iterate_fwd = K.function([input_img, K.learning_phase()],
                                 [layer_output_fwd])

        layer_output_rc = model.get_layer(index=conv_layer_idx).get_output_at(1)
        # index at fwd_output = output size - index at rc_output
        iterate_rc = K.function([input_img, K.learning_phase()],
                                [layer_output_rc])

    print("Computing activations ...")
    chunk_size = args.chunk_size
    n = 0
    all_filter_rows_fwd = None
    all_filter_rows_rc = None
    filter_range = None

    while n < total_num_reads:
        print("Done "+str(n)+" from "+str(total_num_reads)+" sequences")
        samples_chunk = samples[n:n+chunk_size, :, :]
        reads_info_chunk = reads_info[n:n+chunk_size]
        if tf.executing_eagerly():
            activations_fwd, activations_rc = model(samples_chunk, training=False)
            activations_fwd = activations_fwd.numpy()
            activations_rc = activations_rc.numpy()
        else:
            activations_fwd = iterate_fwd([samples_chunk, 0])[0]
            activations_rc = iterate_rc([samples_chunk, 0])[0]

        n_filters = activations_fwd.shape[-1]
        if args.inter_neuron is not None:
            filter_range = args.inter_neuron
        else:
            filter_range = range(n_filters)
        if all_filter_rows_fwd is None:
            all_filter_rows_fwd = [[] for f in range(n_filters)]
            all_filter_rows_rc = [[] for f in range(n_filters)]
        get_activation_data(activations_fwd, filter_range, all_filter_rows_fwd, reads_info_chunk,
                                pad_left, motif_length, rc=False)
        get_activation_data(activations_rc, filter_range, all_filter_rows_rc, reads_info_chunk,
                                pad_left, motif_length, rc=True)

        n += chunk_size

    print("Done " + str(total_num_reads) + " sequences. Saving data...")

    for filter_index in filter_range:
        rows_fwd = pd.concat(all_filter_rows_fwd[filter_index], ignore_index=True)
        rows_rc = pd.concat(all_filter_rows_rc[filter_index], ignore_index=True)
        filter_bed_file = args.out_dir + "/" + test_data_set_name + "_filter_" + str(filter_index) + ".bed"
        # sort by sequence and filter start position
        rows_fwd = rows_fwd.sort_values(['region', 'start', 'end', 'activation'], ascending=[True, True, True, False])
        rows_rc = rows_rc.sort_values(['region', 'start', 'end', 'activation'], ascending=[True, True, True, False])
        # remove duplicates (due to overlapping reads) or take max of two scores at the same genomic position
        # (can occur if filter motif is recognized at the border of one read)
        rows_fwd = rows_fwd.drop_duplicates(['region', 'start', 'end'])
        rows_rc = rows_rc.drop_duplicates(['region', 'start', 'end'])

        all_rows = pd.concat([rows_fwd, rows_rc], ignore_index=True)
        all_rows['activation'] = all_rows['activation'].apply(lambda x: round(x, 3 - int(floor(log10(abs(x))))))
        all_rows.to_csv(filter_bed_file, sep="\t", index=False, header=False)


def get_activation_data(activations, filter_range, all_filter_rows, reads_info_chunk, pad_left, motif_length,
                            rc=False):
    # assumes ReLUs
    pos_indices = np.where(activations[:, :, filter_range] > 0)
    reads = pos_indices[0][:]
    neurons = pos_indices[1][:]
    read_info_name = np.array([reads_info_chunk[read][0] for read in reads])
    read_info_start = np.array([reads_info_chunk[read][1] for read in reads])
    genomic_starts = neurons - pad_left + read_info_start
    genomic_ends = genomic_starts + motif_length

    # if genomic_start <= 0 and genomic_end <= 0: continue
    genomic_starts = genomic_starts[genomic_ends > 0]
    reads = reads[genomic_ends > 0]
    neurons = neurons[genomic_ends > 0]
    region_names = read_info_name[genomic_ends > 0]
    genomic_ends = genomic_ends[genomic_ends > 0]

    for filter_index in filter_range:
        activation_scores = activations[reads, neurons, filter_index]
        if rc:
            row_data = pd.DataFrame({
                'region': region_names,
                'start': np.maximum(0, genomic_starts),
                'end': genomic_ends,
                'filter': np.repeat("filter_" + str(filter_index) + "_rc", region_names.shape[0]),
                'activation': activation_scores.flatten()
                })
        else:
            row_data = pd.DataFrame({
                'region': region_names,
                'start': np.maximum(0, genomic_starts),
                'end': genomic_ends,
                'filter': np.repeat("filter_" + str(filter_index), region_names.shape[0]),
                'activation': activation_scores.flatten()
                })

        all_filter_rows[filter_index].append(row_data)
