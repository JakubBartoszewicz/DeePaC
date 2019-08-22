import numpy as np
import time
import os
import csv
import argparse

from keras.models import load_model
from keras import backend as K
from Bio import SeqIO
from multiprocessing import Pool
from functools import partial

"""
Calculates DeepBind scores for all neurons in the convolutional layer 
and extract all motifs for which a filter neuron got a positive score.
"""


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True, help="Model file (.h5)")
parser.add_argument("-t", "--test_data", required=True, help="Test data (.npy)")
parser.add_argument("-N", "--nonpatho_test", required=True, help="Nonpathogenic reads of the test data set (.fasta)")
parser.add_argument("-P", "--patho_test", required=True, help="Pathogenic reads of the test data set (.fasta)")
parser.add_argument("-o", "--out_dir", default=".", help="Output directory")
parser.add_argument("-n", "--n_cpus", dest="n_cpus", default=1, type=int, help="Number of CPU cores")
parser.add_argument("-r", "--recurrent", action="store_true",
                    help="Visualize elements of the LSTM output")

args = parser.parse_args()


# Creates the model and loads weights
model = load_model(args.model)
print(model.summary())
do_lstm = True
if do_lstm:
    conv_layer_idx = [idx for idx, layer in enumerate(model.layers) if "Bidirectional" in str(layer)][0]
    motif_length = 250
    pad_left = 0
    pad_right = 0
else:
    conv_layer_idx = [idx for idx, layer in enumerate(model.layers) if "Conv1D" in str(layer)][0]
    motif_length = model.get_layer(index=conv_layer_idx).get_weights()[0].shape[0]
    pad_left = (motif_length - 1) // 2
    pad_right = motif_length - 1 - pad_left


layer_dict = dict([(layer.name, layer) for layer in model.layers])
if do_lstm:
    output_layer = 'bidirectional_1'
else:
    output_layer = 'conv1d_1'

print("Loading test data (.npy) ...")
test_data_set_name = os.path.splitext(os.path.basename(args.test_data))[0]
samples = np.load(args.test_data, mmap_mode='r')
total_num_reads = samples.shape[0]
len_reads = samples.shape[1]

print("Loading test data (.fasta) ...")
nonpatho_reads = list(SeqIO.parse(args.nonpatho_test, "fasta"))
patho_reads = list(SeqIO.parse(args.patho_test, "fasta"))
reads = nonpatho_reads + patho_reads

assert len(reads) == total_num_reads, "Test data in .npy-format and fasta files containing different number of reads!"

print("Padding reads ...")
reads = ["N" * pad_left + r + "N" * pad_right for r in reads]

# create output directory
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
if not os.path.exists(args.out_dir + "/filter_activations/"):
    os.makedirs(args.out_dir + "/filter_activations/")
if not os.path.exists(args.out_dir + "/fasta/"):
    os.makedirs(args.out_dir + "/fasta/")


# Specify input and output of the network
input_img = model.layers[0].input
layer_output_fwd = layer_dict[output_layer].get_output_at(0)
layer_output_rc = layer_dict[output_layer].get_output_at(1)

if do_lstm:
    iterate_fwd = K.function([input_img, K.learning_phase()],
                             [layer_output_fwd])

    layer_output_shape = layer_dict[output_layer].get_output_shape_at(1)
    # index at fwd_output = output size - index at rc_output. returns index at FWD!
    iterate_rc = K.function([input_img, K.learning_phase()],
                            [layer_output_rc])
else:
    iterate_fwd = K.function([input_img, K.learning_phase()],
                             [K.max(layer_output_fwd, axis=1), K.argmax(layer_output_fwd, axis=1)])

    layer_output_shape = layer_dict[output_layer].get_output_shape_at(1)
    # index at fwd_output = output size - index at rc_output. returns index at FWD!
    iterate_rc = K.function([input_img, K.learning_phase()],
                            [K.max(layer_output_rc, axis=1), layer_output_shape[1] - 1 - K.argmax(layer_output_rc, axis=1)])

start_time = time.time()

# number of reads per chunk
chunk_size = 1000
n = 0
cores = args.n_cpus

def get_filter_data(filter_id, activation_list, motif_start_list, rc=False,):
    filter_activations = activation_list[:, filter_id]
    filter_motif_starts = motif_start_list[:, filter_id]

    pos_act_ids = [i for i, j in enumerate(filter_activations) if j > 0.0]
    motifs = [reads_chunk[i][filter_motif_starts[i]:filter_motif_starts[i] + motif_length] for i in pos_act_ids]
    activation_scores = [[reads_chunk[i].id, filter_motif_starts[i], filter_activations[i]] for i in pos_act_ids]
    # save filter contribution scores
    filter_act_file = \
        args.out_dir + "/filter_activations/deepbind_" + test_data_set_name + "_act_filter_%d.csv" % filter_id
    with open(filter_act_file, 'a') as csv_file:
        file_writer = csv.writer(csv_file)
        for dat in activation_scores:
            file_writer.writerow([">" + dat[0]])
            file_writer.writerow([dat[1]])
            file_writer.writerow([dat[2]])

    filename = args.out_dir + "/fasta/deepbind_" + test_data_set_name + "_motifs_filter_%d.fasta" % filter_id
    with open(filename, "a") as output_handle:
        if rc:
            SeqIO.write([m.reverse_complement(id=m.id + "_rc", description=m.description + "_rc") for m in motifs],
                        output_handle, "fasta")
        else:
            SeqIO.write(motifs, output_handle, "fasta")

def get_max_strand(filter_id, dat_fwd, dat_rc):
    for seq_id in range(dat_fwd[0].shape[0]):
        # if abs score on fwd higher than on rc
        if dat_fwd[0][seq_id, filter_id] >= dat_rc[0][seq_id, filter_id]:
            dat_rc[0][seq_id, filter_id] = 0.0
        else:
            dat_fwd[0][seq_id, filter_id] = 0.0

# for each read do:
while n < total_num_reads:

    print("Done "+str(n)+" from "+str(total_num_reads)+" sequences")
    samples_chunk = samples[n:n+chunk_size, :, :]
    reads_chunk = reads[n:n+chunk_size]
    if do_lstm:
        act_fwd = iterate_fwd([samples_chunk, 0])[0]
        act_rc = iterate_rc([samples_chunk, 0])[0]
        n_filters = act_fwd.shape[-1]
        mot_fwd = np.zeros((chunk_size,n_filters), dtype="int32")
        mot_rc = np.zeros((chunk_size,n_filters), dtype="int32") + len_reads - 1
        results_fwd = [act_fwd, mot_fwd]
        results_rc = [act_rc, mot_rc]
    else:
        results_fwd = iterate_fwd([samples_chunk, 0])
        results_rc = iterate_rc([samples_chunk, 0])
        n_filters = results_fwd[0].shape[-1]
    #activations = np.concatenate((results_fwd[0], results_rc[0]))
    # activations.shape = [total_num_reads, n_filters]
    # results shape: ([total_num_reads, n_filters], [total_num_reads, n_filters])
    #motif_starts = np.concatenate((results_fwd[1], results_rc[1]))

    # for each filter do:
    if cores > 1:
        p = Pool(processes=cores)
        p.map(partial(get_max_strand, dat_fwd=results_fwd, dat_rc=results_rc), range(n_filters))
        p.map(partial(get_filter_data, activation_list=results_fwd[0], motif_start_list=results_fwd[1]), range(n_filters))
        p.map(partial(get_filter_data, activation_list=results_rc[0], motif_start_list=results_rc[1], rc=True), range(n_filters))
    else:
        list(map(partial(get_max_strand, dat_fwd=results_fwd, dat_rc=results_rc), range(n_filters)))
        list(map(partial(get_filter_data, activation_list=results_fwd[0], motif_start_list=results_fwd[1]), range(n_filters)))
        list(map(partial(get_filter_data, activation_list=results_rc[0], motif_start_list=results_rc[1], rc=True), range(n_filters)))

    n += chunk_size

end_time = time.time()
print("Processed in " + str(end_time - start_time))
