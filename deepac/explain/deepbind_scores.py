import numpy as np
import time
import os
import csv
from multiprocessing import get_context
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf
from Bio import SeqIO
from multiprocessing import cpu_count
from functools import partial
from deepac.utils import set_mem_growth
from deepac.explain.rf_sizes import get_rf_size
from tensorflow.keras.utils import get_custom_objects


def get_filter_data(filter_id, activation_list, motif_start_list, reads_chunk, motif_length, test_data_set_name,
                    out_dir, rc=False):
    """Save filter activation scores and activating sequences"""
    filter_activations = activation_list[:, filter_id]
    filter_motif_starts = motif_start_list[:, filter_id]

    pos_act_ids = np.where(filter_activations > 0.0)[0]
    motifs = [reads_chunk[i][filter_motif_starts[i]:filter_motif_starts[i] + motif_length] for i in pos_act_ids]
    activation_scores = [[reads_chunk[i].id, filter_motif_starts[i], filter_activations[i]] for i in pos_act_ids]
    # save filter contribution scores
    filter_act_file = \
        out_dir + "/filter_activations/deepbind_" + test_data_set_name + "_act_filter_%d.csv" % filter_id
    with open(filter_act_file, 'a') as csv_file:
        file_writer = csv.writer(csv_file)
        for dat in activation_scores:
            file_writer.writerow([">" + dat[0]])
            file_writer.writerow([dat[1]])
            file_writer.writerow([dat[2]])

    filename = out_dir + "/fasta/deepbind_" + test_data_set_name + "_motifs_filter_%d.fasta" % filter_id
    with open(filename, "a") as output_handle:
        if rc:
            SeqIO.write([m.reverse_complement(id=m.id + "_rc", description=m.description + "_rc") for m in motifs],
                        output_handle, "fasta")
        else:
            SeqIO.write(motifs, output_handle, "fasta")


def get_max_strand(filter_id, dat_fwd, dat_rc):
    """Get max motif activation data over both strands"""
    for seq_id in range(dat_fwd[0].shape[0]):
        # if abs score on fwd higher than on rc
        if dat_fwd[0][seq_id, filter_id] >= dat_rc[0][seq_id, filter_id]:
            dat_rc[0][seq_id, filter_id] = 0.0
        else:
            dat_fwd[0][seq_id, filter_id] = 0.0


def get_maxact(args):
    """Calculates DeepBind scores for all neurons in the convolutional layer
    and extract all motifs for which a filter neuron got a positive score."""
    set_mem_growth()

    save_activations_npy = args.save_activs_and_maxact or args.save_activs_only
    find_maxact = not args.save_activs_only
    do_rc = args.inter_layer > 0
    merge_activations_npy = args.save_activs_merge if save_activations_npy else None
    reads = None

    # Creates the model and loads weights
    model = load_model(args.model, custom_objects=get_custom_objects())
    print(model.summary())
    do_lstm = args.do_lstm
    if args.inter_layer == 0:
        if find_maxact:
            raise ValueError("Finding max activations not supported for the global pooling layer.")
        # Special case - last layer (after pooling)
        pad_left = 0
        pad_right = 0
        motif_length = None
        conv_layer_ids = [idx for idx, layer in enumerate(model.layers)
                          if "Global" in str(layer)]
        if len(conv_layer_ids) > 1:
            raise ValueError("Only one global pooling layer was assumed. Found more.")
        else:
            conv_layer_idx = conv_layer_ids[0]
    elif args.inter_layer < 0:
        raise ValueError("Negative layer index.")
    elif do_lstm:
        input_layer_id = [idx for idx, layer in enumerate(model.layers) if "Input" in str(layer)][0]
        motif_length = model.get_layer(index=input_layer_id).get_output_at(0).shape[1]
        pad_left = 0
        pad_right = 0
        conv_layer_idx = [idx for idx, layer in enumerate(model.layers)
                          if "Bidirectional" in str(layer)][args.inter_layer - 1]
    else:
        # ignore size-1 Convs in skip connections
        conv_layer_ids = [idx for idx, layer in enumerate(model.layers) if "Conv1D" in str(layer)
                          and layer.kernel_size[0] > 1]
        conv_layer_idx = conv_layer_ids[args.inter_layer - 1]
        motif_length = get_rf_size(model, conv_layer_idx)
        pad_left = (motif_length - 1) // 2
        pad_right = motif_length - 1 - pad_left

    print("Loading test data (.npy) ...")
    test_data_set_name = os.path.splitext(os.path.basename(args.test_data))[0]
    samples = np.load(args.test_data, mmap_mode='r')
    total_num_reads = samples.shape[0]

    # create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if find_maxact:
        print("Loading test data (.fasta) ...")
        if args.nonpatho_test is None or args.patho_test is None:
            raise ValueError("No path to reads (.fasta) supplied.")
        nonpatho_reads = list(SeqIO.parse(args.nonpatho_test, "fasta"))
        patho_reads = list(SeqIO.parse(args.patho_test, "fasta"))
        reads = nonpatho_reads + patho_reads

        assert len(reads) == total_num_reads, "Test data in .npy-format and fasta files containing different" \
                                              " number of reads!"
        print("Padding reads ...")
        reads = ["N" * pad_left + r + "N" * pad_right for r in reads]
        # create output directory
        if not os.path.exists(args.out_dir + "/filter_activations/"):
            os.makedirs(args.out_dir + "/filter_activations/")
        if not os.path.exists(args.out_dir + "/fasta/"):
            os.makedirs(args.out_dir + "/fasta/")

    # Specify input and output of the network

    layer_output_fwd = model.get_layer(index=conv_layer_idx).get_output_at(0)
    layer_output_rc = model.get_layer(index=conv_layer_idx).get_output_at(1) if do_rc else None
    iterate_rc = None
    layer_output_shape = model.get_layer(index=conv_layer_idx).get_output_shape_at(0)
    n_filters = layer_output_shape[-1]
    if tf.executing_eagerly():
        if do_rc:
            model = tf.keras.Model(model.inputs,
                                   (layer_output_fwd, layer_output_rc))
        else:
            model = tf.keras.Model(model.inputs, layer_output_fwd)
        iterate_fwd = None
    else:
        input_img = model.layers[0].input
        if do_lstm:
            iterate_fwd = K.function([input_img, K.learning_phase()],
                                     [layer_output_fwd])
            # index at fwd_output = output size - index at rc_output. returns index at FWD!
            if do_rc:
                iterate_rc = K.function([input_img, K.learning_phase()],
                                        [layer_output_rc])
        else:
            iterate_fwd = K.function([input_img, K.learning_phase()],
                                     [K.max(layer_output_fwd, axis=1), K.argmax(layer_output_fwd, axis=1)])

            # index at fwd_output = output size - index at rc_output. returns index at FWD!
            if do_rc:
                iterate_rc = K.function([input_img, K.learning_phase()],
                                        [K.max(layer_output_rc, axis=1),
                                        layer_output_shape[1] - 1 - K.argmax(layer_output_rc, axis=1)])

    start_time = time.time()

    # number of reads per chunk
    chunk_size = args.chunk_size
    n = 0
    if args.n_cpus is None:
        cores = cpu_count()
    else:
        cores = args.n_cpus

    all_act_fwd = np.zeros((samples.shape[0], n_filters))
    all_act_rc = np.zeros((samples.shape[0], n_filters)) if do_rc else None
    act_fwd, act_rc = None, None
    results_fwd, results_rc = None, None
    reads_chunk = None

    # for each read do:
    while n < total_num_reads:
        print("Done "+str(n)+" from "+str(total_num_reads)+" sequences")
        samples_chunk = samples[n:n+chunk_size, :, :]
        samples_chunk = samples_chunk.astype('float32')
        if find_maxact:
            reads_chunk = reads[n:n+chunk_size]
        if do_lstm:
            if tf.executing_eagerly():
                if do_rc:
                    act_fwd, act_rc = model(samples_chunk, training=False)
                    act_rc = act_rc.numpy()
                else:
                    act_fwd = model(samples_chunk, training=False)
                act_fwd = act_fwd.numpy()
            else:
                act_fwd = iterate_fwd([samples_chunk, 0])[0]
                if do_rc:
                    act_rc = iterate_rc([samples_chunk, 0])[0]
            mot_fwd = np.zeros((chunk_size, n_filters), dtype="int32")
            mot_rc = np.zeros((chunk_size, n_filters), dtype="int32") if do_rc else None
            results_fwd = [act_fwd, mot_fwd]
            results_rc = [act_rc, mot_rc]
        else:
            if tf.executing_eagerly():
                if do_rc:
                    act_fwd, act_rc = model(samples_chunk, training=False)
                else:
                    act_fwd = model(samples_chunk, training=False)
                if save_activations_npy:
                    act_fwd = act_fwd.numpy()
                    act_rc = act_rc.numpy() if do_rc else None
                if find_maxact:
                    results_fwd = [K.max(act_fwd, axis=1).numpy(), K.argmax(act_fwd, axis=1).numpy()]
                    out_shape = model.get_layer(index=conv_layer_idx).get_output_shape_at(1)
                    if do_rc:
                        results_rc = [K.max(act_rc, axis=1).numpy(),
                                      out_shape[1] - 1 - K.argmax(act_rc, axis=1).numpy()]
            else:
                if save_activations_npy:
                    act_fwd = K.function([model.layers[0].input, K.learning_phase()], [layer_output_fwd])
                    if do_rc:
                        act_rc = K.function([model.layers[0].input, K.learning_phase()], [layer_output_rc])
                if find_maxact:
                    results_fwd = iterate_fwd([samples_chunk, 0])
                    if do_rc:
                        results_rc = iterate_rc([samples_chunk, 0])
        if save_activations_npy:
            all_act_fwd[n:n+chunk_size, :] = act_fwd
            if do_rc:
                all_act_rc[n:n+chunk_size, :] = act_rc

        if find_maxact:
            n_filters = results_fwd[0].shape[-1]
            # for each filter do:
            if cores > 1:
                with get_context("spawn").Pool(processes=min(cores, n_filters)) as p:
                    p.map(partial(get_max_strand, dat_fwd=results_fwd, dat_rc=results_rc), range(n_filters))
                    p.map(partial(get_filter_data, activation_list=results_fwd[0], motif_start_list=results_fwd[1],
                                  reads_chunk=reads_chunk, motif_length=motif_length,
                                  test_data_set_name=test_data_set_name, out_dir=args.out_dir),
                          range(n_filters))
                    p.map(partial(get_filter_data, activation_list=results_rc[0], motif_start_list=results_rc[1],
                                  reads_chunk=reads_chunk, motif_length=motif_length,
                                  test_data_set_name=test_data_set_name, out_dir=args.out_dir, rc=True),
                          range(n_filters))
            else:
                list(map(partial(get_max_strand, dat_fwd=results_fwd, dat_rc=results_rc), range(n_filters)))
                list(map(partial(get_filter_data, activation_list=results_fwd[0], motif_start_list=results_fwd[1],
                                 reads_chunk=reads_chunk, motif_length=motif_length,
                                 test_data_set_name=test_data_set_name, out_dir=args.out_dir),
                         range(n_filters)))
                list(map(partial(get_filter_data, activation_list=results_rc[0], motif_start_list=results_rc[1],
                                 reads_chunk=reads_chunk, motif_length=motif_length,
                                 test_data_set_name=test_data_set_name, out_dir=args.out_dir, rc=True),
                         range(n_filters)))

        n += chunk_size

    end_time = time.time()
    print("Done "+str(min(n, total_num_reads))+" from "+str(total_num_reads)+" sequences")
    print("Processed in " + str(end_time - start_time))
    if save_activations_npy:
        print("Saving raw activations...")
        if args.inter_layer == 0:
            all_act_rc = all_act_fwd[:, n_filters//2:]
            all_act_rc = all_act_rc[::, ::-1]
            all_act_fwd = all_act_fwd[:, :n_filters//2]

        if do_rc or args.inter_layer == 0:
            np.save(os.path.join(args.out_dir, "activations_fwd.npy"), all_act_fwd)
            np.save(os.path.join(args.out_dir, "activations_rc.npy"), all_act_rc)

            if merge_activations_npy == "add" or merge_activations_npy == "sum":
                np.save(os.path.join(args.out_dir, "activations.npy"), all_act_fwd + all_act_rc)
            elif merge_activations_npy == "max" or merge_activations_npy == "maximum":
                np.save(os.path.join(args.out_dir, "activations.npy"), np.maximum(all_act_fwd, all_act_rc))
            elif merge_activations_npy == "mul" or merge_activations_npy == "multiply":
                np.save(os.path.join(args.out_dir, "activations.npy"), all_act_fwd * all_act_rc)
            elif merge_activations_npy == "avg" or merge_activations_npy == "average":
                np.save(os.path.join(args.out_dir, "activations.npy"), (all_act_fwd + all_act_rc)/2)
            else:
                raise ValueError(f"Unrecognized merging method: {merge_activations_npy}.")
        else:
            np.save(os.path.join(args.out_dir, "activations.npy"), all_act_fwd)
