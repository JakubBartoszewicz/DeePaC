import os
import csv
import numpy as np
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from deepac.utils import set_mem_growth

from Bio import SeqIO

from shap import DeepExplainer
from deepac.explain.rf_sizes import get_rf_size
from tqdm import tqdm


def get_filter_contribs(args, allow_eager=False):
    """Calculate DeepLIFT contribution scores for all neurons in the convolutional layer
    and extract all motifs for which a filter neuron got a non-zero contribution score."""
    if tf.executing_eagerly() and not allow_eager:
        print("Using SHAP. Disabling eager execution...")
        tf.compat.v1.disable_v2_behavior()
    set_mem_growth()
    model = load_model(args.model)
    max_only = args.partial or args.easy_partial or not args.all_occurrences
    check_additivity = not args.no_check
    if args.w_norm and not args.do_lstm:
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
    if args.do_lstm:
        conv_layer_idx = [idx for idx, layer in enumerate(model.layers)
                          if "Bidirectional" in str(layer)][args.inter_layer - 1]
        n_filters = model.get_layer(index=conv_layer_idx).get_output_at(0).shape[-1]
        input_layer_id = [idx for idx, layer in enumerate(model.layers) if "Input" in str(layer)][0]
        motif_length = model.get_layer(index=input_layer_id).get_output_at(0).shape[1]
        pad_left = 0
        pad_right = 0
    else:
        conv_layer_ids = [idx for idx, layer in enumerate(model.layers) if "Conv1D" in str(layer)]
        conv_layer_idx = conv_layer_ids[args.inter_layer - 1]
        motif_length = get_rf_size(model, conv_layer_idx)
        n_filters = model.get_layer(index=conv_layer_idx).get_weights()[0].shape[-1]
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
    if (args.partial or args.easy_partial) and not os.path.exists(args.out_dir + "/nuc_scores/"):
        os.makedirs(args.out_dir + "/nuc_scores/")

    # load or create reference sequences
    ref_samples = get_reference_seqs(args, len_reads)
    num_ref_seqs = ref_samples.shape[0]

    print("Running DeepSHAP ...")
    chunk_size = args.chunk_size // num_ref_seqs
    i = 0
    if tf.executing_eagerly():
        intermediate_model = tf.keras.Model(model.inputs,
                                            (model.get_layer(index=conv_layer_idx).get_output_at(0),
                                             model.get_layer(index=conv_layer_idx).get_output_at(1)))

        def map2layer(input_samples):
            out = intermediate_model(input_samples, training=False)
            return out[0].numpy(), out[1].numpy()

        intermediate_ref_fwd, intermediate_ref_rc = map2layer(ref_samples)
    else:
        def map2layer(input_samples, layer, out_node):
            feed_dict = dict(zip([model.get_layer(index=0).input], [input_samples]))
            return tf.compat.v1.keras.backend.get_session().run(model.get_layer(index=layer).get_output_at(out_node),
                                                                feed_dict)

        intermediate_ref_fwd = map2layer(ref_samples, conv_layer_idx, 0)
        intermediate_ref_rc = map2layer(ref_samples, conv_layer_idx, 1)
        intermediate_ref_fwd = intermediate_ref_fwd.mean(axis=0, keepdims=True)
        intermediate_ref_rc = intermediate_ref_rc.mean(axis=0, keepdims=True)

    explainer = DeepExplainer(([model.get_layer(index=conv_layer_idx).get_output_at(0),
                                  model.get_layer(index=conv_layer_idx).get_output_at(1)],
                                 model.layers[-1].output),
                                [intermediate_ref_fwd,
                                 intermediate_ref_rc])
    filter_range = range(n_filters)
    if args.inter_neuron is not None:
        filter_range = [None]*n_filters
        for n in args.inter_neuron:
            filter_range[n] = n

    while i < total_num_reads:

        print("Done "+str(i)+" from "+str(total_num_reads)+" sequences")
        samples_chunk = samples[i:i+chunk_size, :, :]
        reads_chunk = reads[i:i+chunk_size]

        if tf.executing_eagerly():
            intermediate_fwd, intermediate_rc = map2layer(ref_samples)
        else:
            intermediate_fwd = map2layer(samples_chunk, conv_layer_idx, 0)
            intermediate_rc = map2layer(samples_chunk, conv_layer_idx, 1)
        inter_diff_fwd = intermediate_fwd - intermediate_ref_fwd
        inter_diff_rc = intermediate_rc - intermediate_ref_rc

        scores_filter = explainer.shap_values([intermediate_fwd,
                                               intermediate_rc], check_additivity=check_additivity)
        scores_fwd, scores_rc = scores_filter[0]

        # shape: [num_reads, len_reads, n_filters]

        print("Getting data ...")
        # for each filter do:
        if args.do_lstm:
            dat_fwd = [get_lstm_data(i, scores_filter_avg=scores_fwd,
                                     input_reads=reads_chunk, motif_len=motif_length) for i in filter_range]
            dat_rc = [get_lstm_data(i, scores_filter_avg=scores_rc,
                                    input_reads=reads_chunk, motif_len=motif_length,
                                    rc=True) for i in filter_range]
        else:
            dat_fwd = [get_filter_data(i, scores_filter_avg=scores_fwd,
                                       input_reads=reads_chunk, motif_len=motif_length,
                                       max_only=max_only) for i in filter_range]
            dat_rc = [get_filter_data(i, scores_filter_avg=scores_rc,
                                      input_reads=reads_chunk, motif_len=motif_length, rc=True,
                                      max_only=max_only) for i in filter_range]

        if max_only:
            dat_max = [get_max_strand(i, dat_fwd=dat_fwd, dat_rc=dat_rc) for i in filter_range]
            contrib_dat_fwd, motif_dat_fwd, contrib_dat_rc, motif_dat_rc = list(zip(*dat_max))
        else:
            contrib_dat_fwd, motif_dat_fwd = list(zip(*dat_fwd))
            contrib_dat_rc, motif_dat_rc = list(zip(*dat_rc))

        print("Saving data ...")
        if contrib_dat_fwd:
            for f in filter_range:
                write_filter_data(f, contribution_data=contrib_dat_fwd, motifs=motif_dat_fwd,
                                  out_dir=args.out_dir, data_set_name=test_data_set_name)
        if contrib_dat_rc:
            for f in filter_range:
                write_filter_data(f, contribution_data=contrib_dat_rc, motifs=motif_dat_rc,
                                  out_dir=args.out_dir, data_set_name=test_data_set_name)

        if args.partial:
            print("Getting partial data ...")
            partials_nt_fwd = [get_partials(i, model=model, conv_layer_idx=conv_layer_idx,
                                            node=0, ref_samples=ref_samples,
                                            contribution_data=contrib_dat_fwd, samples_chunk=samples_chunk,
                                            input_reads=reads_chunk, intermediate_diff=inter_diff_fwd,
                                            pad_left=pad_left, pad_right=pad_right, lstm=args.do_lstm,
                                            check_additivity=check_additivity)
                               for i in filter_range]

            partials_nt_rc = [get_partials(i, model=model, conv_layer_idx=conv_layer_idx,
                                           node=1, ref_samples=ref_samples,
                                           contribution_data=contrib_dat_rc, samples_chunk=samples_chunk,
                                           input_reads=reads_chunk, intermediate_diff=inter_diff_rc,
                                           pad_left=pad_left, pad_right=pad_right, lstm=args.do_lstm,
                                           check_additivity=check_additivity)
                              for i in filter_range]
        elif args.easy_partial:
            print("Getting partial data ...")
            partials_nt_fwd = [get_easy_partials(i, model=model, conv_layer_idx=conv_layer_idx, node=0,
                                                 contribution_data=contrib_dat_fwd, samples_chunk=samples_chunk,
                                                 input_reads=reads_chunk, intermediate_diff=inter_diff_fwd,
                                                 pad_left=pad_left, pad_right=pad_right) for i in filter_range]

            partials_nt_rc = [get_easy_partials(i, model=model, conv_layer_idx=conv_layer_idx, node=1,
                                                contribution_data=contrib_dat_rc, samples_chunk=samples_chunk,
                                                input_reads=reads_chunk, intermediate_diff=inter_diff_rc,
                                                pad_left=pad_left, pad_right=pad_right) for i in filter_range]
        if args.partial or args.easy_partial:
            scores_nt_fwd, read_ids_fwd = list(zip(*partials_nt_fwd))
            scores_nt_rc, read_ids_rc = list(zip(*partials_nt_rc))
            print("Saving partial data ...")
            if scores_nt_fwd:
                for f in filter_range:
                    write_partial_data(f, read_ids=read_ids_fwd, contribution_data=contrib_dat_fwd,
                                       scores_input_pad=scores_nt_fwd, out_dir=args.out_dir,
                                       data_set_name=test_data_set_name, motif_len=motif_length)
            if scores_nt_rc:
                for f in filter_range:
                    write_partial_data(f, read_ids=read_ids_rc, contribution_data=contrib_dat_rc,
                                       scores_input_pad=scores_nt_rc, out_dir=args.out_dir,
                                       data_set_name=test_data_set_name, motif_len=motif_length)
        i += chunk_size
    print("Done "+str(min(i, total_num_reads))+" from "+str(total_num_reads)+" sequences")


def get_max_strand(filter_id, dat_fwd, dat_rc):
    if filter_id is None:
        return [], [], [], []
    i = filter_id
    contrib_dat_fwd = []
    motif_dat_fwd = []
    contrib_dat_rc = []
    motif_dat_rc = []
    for seq_id in range(len(dat_fwd[i][0])):
        record_fwd = dat_fwd[i]
        record_rc = dat_rc[i]
        # if any contributions at all
        if len(record_fwd[0][seq_id][1]) > 0 and len(record_rc[0][seq_id][1]) > 0:
            # if abs score on fwd higher than on rc
            if np.abs(record_fwd[0][seq_id][2]) >= np.abs(record_rc[0][seq_id][2]):
                contrib_dat_fwd.append(record_fwd[0][seq_id])
                motif_dat_fwd.append(record_fwd[1][seq_id])
                contrib_dat_rc.append([])
                motif_dat_rc.append("")
            else:
                contrib_dat_rc.append(record_rc[0][seq_id])
                motif_dat_rc.append(record_rc[1][seq_id])
                contrib_dat_fwd.append([])
                motif_dat_fwd.append("")
        elif len(record_fwd[0][seq_id][1]) > 0 and not (len(record_rc[0][seq_id][1]) > 0):
            contrib_dat_fwd.append(record_fwd[0][seq_id])
            motif_dat_fwd.append(record_fwd[1][seq_id])
            contrib_dat_rc.append([])
            motif_dat_rc.append("")
        elif not (len(record_fwd[0][seq_id][1]) > 0) and len(record_rc[0][seq_id][1]) > 0:
            contrib_dat_rc.append(record_rc[0][seq_id])
            motif_dat_rc.append(record_rc[1][seq_id])
            contrib_dat_fwd.append([])
            motif_dat_fwd.append("")
        else:
            contrib_dat_rc.append([])
            motif_dat_rc.append("")
            contrib_dat_fwd.append([])
            motif_dat_fwd.append("")
    return contrib_dat_fwd, motif_dat_fwd, contrib_dat_rc, motif_dat_rc


def get_filter_data(filter_id, scores_filter_avg, input_reads, motif_len, rc=False, max_only=True):
    # determine non-zero contribution scores per read and filter
    # and extract DNA-sequence of corresponding subreads
    if filter_id is None:
        return [], []
    num_reads = len(input_reads)
    contribution_data = []
    motifs = []
    for seq_id in range(num_reads):
        if np.any(scores_filter_avg[seq_id, :, filter_id]):
            if max_only:
                max_id = np.argmax(np.abs(scores_filter_avg[seq_id, :, filter_id]))
                non_zero_neurons = np.asarray([max_id]) if scores_filter_avg[seq_id, max_id, filter_id] \
                    else np.empty((0,), dtype=int)
            else:
                non_zero_neurons = np.nonzero(scores_filter_avg[seq_id, :, filter_id])[0]

            scores = scores_filter_avg[seq_id, non_zero_neurons, filter_id]
            contribution_data.append((input_reads[seq_id].id, non_zero_neurons, scores))
            if not rc:
                motifs.append([input_reads[seq_id][non_zero_neuron:(non_zero_neuron + motif_len)]
                               for non_zero_neuron in non_zero_neurons])
            else:
                # Assume all reads are the same length
                non_zero_neurons = scores_filter_avg.shape[1] - 1 - non_zero_neurons
                ms = [input_reads[seq_id][non_zero_neuron:(non_zero_neuron + motif_len)]
                      for non_zero_neuron in non_zero_neurons]
                motifs.append([m.reverse_complement(id=m.id + "_rc", description=m.description + "_rc")
                               for m in ms])
        else:
            contribution_data.append((input_reads[seq_id].id, [], []))
            motifs.append("")
    return contribution_data, motifs


def get_lstm_data(filter_id, scores_filter_avg, input_reads, motif_len, rc=False):
    # determine non-zero contribution scores per read and filter
    # and extract DNA-sequence of corresponding subreads
    if filter_id is None:
        return [], []
    num_reads = len(input_reads)
    contribution_data = []
    motifs = []
    for seq_id in range(num_reads):
        if scores_filter_avg[seq_id, filter_id] != 0:
            non_zero_neurons = [0]
            scores = [scores_filter_avg[seq_id, filter_id]]
            contribution_data.append((input_reads[seq_id].id, non_zero_neurons, scores))
            if not rc:
                motifs.append([input_reads[seq_id][non_zero_neuron:(non_zero_neuron + motif_len)]
                               for non_zero_neuron in non_zero_neurons])
            else:
                ms = [input_reads[seq_id][non_zero_neuron:(non_zero_neuron + motif_len)]
                      for non_zero_neuron in non_zero_neurons]
                motifs.append([m.reverse_complement(id=m.id + "_rc", description=m.description + "_rc")
                               for m in ms])
        else:
            contribution_data.append((input_reads[seq_id].id, [], []))
            motifs.append("")
    return contribution_data, motifs


def write_filter_data(filter_id, contribution_data, motifs, data_set_name, out_dir):
    if filter_id is None:
        return
    if contribution_data[filter_id] is not None and motifs[filter_id] is not None and \
            len(contribution_data[filter_id]) > 0 and len(motifs[filter_id]) > 0:
        # save filter contribution scores
        filter_rel_file = \
            out_dir + "/filter_scores/" + data_set_name + "_rel_filter_%d.csv" % filter_id
        with open(filter_rel_file, 'a') as csv_file:
            file_writer = csv.writer(csv_file)
            for dat in contribution_data[filter_id]:
                if len(dat) > 0 and len(dat[1]) > 0:
                    file_writer.writerow([">" + dat[0]])
                    file_writer.writerow(dat[1])
                    file_writer.writerow(dat[2])

        # save subreads which cause non-zero contribution scores
        filter_motifs_file = \
            out_dir + "/fasta/" + data_set_name + "_motifs_filter_%d.fasta" % filter_id
        with open(filter_motifs_file, "a") as output_handle:
            SeqIO.write([subread for motif in motifs[filter_id] for subread in motif], output_handle, "fasta")


def get_partials(filter_id, model, conv_layer_idx, node, ref_samples, contribution_data, samples_chunk,
                 input_reads, intermediate_diff, pad_left, pad_right, lstm=False, check_additivity=False):
    num_reads = len(input_reads)
    if filter_id is None:
        return [], []
    read_ids = []
    scores_pt_all = []
    print("Processing filter: {}".format(filter_id))
    if contribution_data[filter_id] is None or not (len(contribution_data[filter_id]) > 0):
        return [], []
    for seq_id in tqdm(range(num_reads)):
        read_id = re.search("seq_[0-9]+", input_reads[seq_id].id).group()
        read_id = int(read_id.replace("seq_", ""))
        read_ids.append(read_id)

        if contribution_data[filter_id][seq_id] is None or not (len(contribution_data[filter_id][seq_id]) > 0):
            scores_pt_all.append(None)
            continue

        out = model.get_layer(index=conv_layer_idx).get_output_at(node)
        if lstm:
            out = out[:, filter_id:filter_id+1]
        else:
            out = out[:, contribution_data[filter_id][seq_id][1][0], filter_id:filter_id+1]
        explainer_nt = DeepExplainer((model.get_layer(index=0).input, out), ref_samples)

        sample = samples_chunk[seq_id, :, :].reshape((1, ref_samples.shape[1], ref_samples.shape[2]))
        # Get difference in activation of the intermediate neuron
        if lstm:
            diff = intermediate_diff[seq_id, filter_id]
        else:
            diff = intermediate_diff[seq_id, contribution_data[filter_id][seq_id][1][0], filter_id]
        scores_nt = explainer_nt.shap_values(sample, check_additivity=check_additivity)[0]
        partials = np.asarray([phi_i * contribution_data[filter_id][seq_id][2][0] for phi_i in scores_nt]) / diff

        partials = partials.reshape(partials.shape[1], partials.shape[2])
        # Sum along the channel (nt) axis and pad
        scores_pt_pad = np.sum(partials, axis=1)
        scores_pt_pad = np.pad(scores_pt_pad, (pad_left, pad_right), 'constant', constant_values=0.0)
        if node == 1:
            scores_pt_pad = scores_pt_pad[::-1]
        scores_pt_all.append(scores_pt_pad)

    return scores_pt_all, read_ids


def get_easy_partials(filter_id, model, conv_layer_idx, node, contribution_data, samples_chunk,
                      input_reads, intermediate_diff, pad_left, pad_right):
    if filter_id is None:
        return [], []
    num_reads = len(input_reads)
    read_ids = []
    scores_pt_all = []
    if contribution_data[filter_id] is None or not (len(contribution_data[filter_id]) > 0):
        return [], []
    for seq_id in range(num_reads):
        read_id = re.search("seq_[0-9]+", input_reads[seq_id].id).group()
        read_id = int(read_id.replace("seq_", ""))
        read_ids.append(read_id)

        if contribution_data[filter_id][seq_id] is None or not (len(contribution_data[filter_id][seq_id]) > 0):
            scores_pt_all.append(None)
            continue

        motif_length = get_rf_size(model, conv_layer_idx)
        motif_start = contribution_data[filter_id][seq_id][1][0]
        if node == 0:
            sample = samples_chunk[seq_id, ::, ::]
        else:
            sample = samples_chunk[seq_id, ::-1, ::-1]
        sample = np.pad(sample, ((pad_left, pad_right), (0, 0)), 'constant', constant_values=(0.0, 0.0))
        sample = sample.reshape((1, sample.shape[0], sample.shape[1]))
        sample = sample[:, motif_start:motif_start+motif_length, :]
        # Get difference in activation of the intemediate neuron
        diff = intermediate_diff[seq_id, contribution_data[filter_id][seq_id][1][0], filter_id]
        # Assuming: first layer only, all-zero reference, no nonlinearity, one-hot encoded nucleotides
        # Then: contributions to filter output are equal to the weights
        scores_nt = model.get_layer(index=conv_layer_idx).get_weights()[0][:, :, filter_id]
        dilation = model.get_layer(index=conv_layer_idx).get_config()["dilation_rate"][0]
        if dilation > 1:
            scores_nt = np.insert(scores_nt, np.repeat(np.arange(1, scores_nt.shape[0]), dilation-1),
                                  np.zeros(4), axis=0)
        scores_nt = np.multiply(scores_nt, sample)
        partials = np.asarray([phi_i * contribution_data[filter_id][seq_id][2][0] for phi_i in scores_nt]) / diff

        # Sum along the channel (nt) axis and pad
        partials = partials.reshape(partials.shape[1], partials.shape[2])
        scores_pt_pad = np.sum(partials, axis=1)
        pad_right_read = \
            intermediate_diff.shape[1] + pad_right + pad_left - \
            contribution_data[filter_id][seq_id][1][0] - motif_length
        pad_left_read = contribution_data[filter_id][seq_id][1][0]
        scores_pt_pad = np.pad(scores_pt_pad, (pad_left_read, pad_right_read), 'constant', constant_values=0.0)
        scores_pt_all.append(scores_pt_pad)

    return scores_pt_all, read_ids


def write_partial_data(filter_id, read_ids, contribution_data, scores_input_pad, out_dir, data_set_name, motif_len):
    if filter_id is None:
        return
    # save contribution scores for each nucleotide per filter motif
    with open(out_dir + "/nuc_scores/" + data_set_name + "_rel_filter_%d_nucleotides.csv" % filter_id, 'a') \
            as csv_file:
        file_writer = csv.writer(csv_file)
        for ind, seq_id in enumerate(read_ids[filter_id]):
            # contribution_data[filter_id][ind][1][0] is the motif start
            if len(contribution_data[filter_id][ind]) > 0 and len(scores_input_pad[filter_id][ind]) > 0:
                scores = scores_input_pad[filter_id][ind]
                scores = scores.tolist()
                scores = scores[contribution_data[filter_id][ind][1][0]:(contribution_data[filter_id][ind][1][0] +
                                                                         motif_len)]
                row = ['%.4g' % s for s in scores]
                file_writer.writerow([">" + contribution_data[filter_id][ind][0]])
                file_writer.writerow(row)


def get_reference_seqs(args, len_reads):
    """
    Load or create reference sequences for DeepLIFT.
    """
    # generate reference sequence with N's
    if args.ref_mode == "N":

        print("Generating reference sequence with all Ns...")
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
