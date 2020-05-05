import os
import csv
import re
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf
from Bio import SeqIO
from operator import itemgetter
from itertools import groupby
from deepac.utils import set_mem_growth


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
    while n < total_num_reads:
        print("Done "+str(n)+" from "+str(total_num_reads)+" sequences")
        samples_chunk = samples[n:n+chunk_size, :, :]
        reads_info_chunk = reads_info[n:n+chunk_size]
        if tf.executing_eagerly():
            activations_fwd, activations_rc = model(samples_chunk, training=False)
            activations = activations_fwd.numpy() + activations_rc.numpy()
        else:
            activations_fwd = iterate_fwd([samples_chunk, 0])[0]
            activations_rc = iterate_rc([samples_chunk, 0])[0]
            activations = activations_fwd + activations_rc

        n_filters = activations_fwd.shape[-1]
        for filter_index in range(n_filters):

            filter_bed_file = args.out_dir + "/" + test_data_set_name + "_filter_" + str(filter_index) + ".bed"

            pos_indices = np.where(activations[:, :, filter_index] > 0)
            rows = []
            for i in range(len(pos_indices[0])):
                read = pos_indices[0][i]
                neuron = pos_indices[1][i]
                genomic_start = neuron - pad_left + reads_info_chunk[read][1]
                genomic_end = genomic_start + motif_length
                if genomic_start <= 0 and genomic_end <= 0:
                    continue
                else:
                    activation_score = activations[read, neuron, filter_index]
                    rows.append([reads_info_chunk[read][0], max(0, genomic_start), genomic_end,
                                 "filter_"+str(filter_index), '%.4g' % activation_score])

            # sort by sequence and filter start position
            rows.sort(key=itemgetter(0, 1))
            # remove duplicates (due to overlapping reads) or take max of two scores at the same genomic position
            # (can occur if filter motif is recognized at the border of one read)
            rows = [max(g, key=itemgetter(4)) for k, g in groupby(rows, itemgetter(0, 1))]

            with open(filter_bed_file, 'a') as csv_file:
                file_writer = csv.writer(csv_file, delimiter='\t')
                for r in rows:
                    file_writer.writerow(r)

        n += chunk_size
