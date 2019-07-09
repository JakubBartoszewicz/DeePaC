import sys
import os
import csv
import argparse
import re
import numpy as np
from keras.models import load_model
from keras import backend as K
from Bio import SeqIO
import tensorflow as tf
from rc_layers import *

'''
Compute activation values genome-wide.
'''


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True, help="Model file (.h5)")
parser.add_argument("-t", "--test_data", required=True, help="Test data (.npy)")
parser.add_argument("-f", "--test_fasta", required=True, help="Reads of the test data set (.fasta)")
parser.add_argument("-o", "--out_dir", default = ".", help="Output directory")
args = parser.parse_args()

# Creates the model and loads weights
model = load_model(args.model, custom_objects={'RevCompConv1D': RevCompConv1D, 'RevCompConv1DBatchNorm': RevCompConv1DBatchNorm, 'DenseAfterRevcompWeightedSum': DenseAfterRevcompWeightedSum, 'DenseAfterRevcompConv1D': DenseAfterRevcompConv1D})
conv_layer_idx = [idx for idx, layer in enumerate(model.layers) if "Conv1D" in str(layer)][0]
output_layer='conv1d_1'
motif_length = model.get_layer(index=conv_layer_idx).get_weights()[0].shape[0]
pad_left = (motif_length - 1) // 2
pad_right = motif_length - 1 - pad_left

layer_dict = dict([(layer.name, layer) for layer in model.layers])
RC_architecture = 'rev_comp_conv1d_1' in layer_dict.keys()
if RC_architecture:
	output_layer="rev_comp_conv1d_1"

print("Loading test data (.npy) ...")
test_data_set_name = os.path.splitext(os.path.basename(args.test_data))[0]
samples = np.load(args.test_data, mmap_mode='r')
total_num_reads = samples.shape[0]
len_reads = samples.shape[1]

print("Loading test data (.fasta) ...")
reads = list(SeqIO.parse(args.test_fasta, "fasta"))
assert len(reads) == total_num_reads, "Test data in .npy-format and fasta files containing different number of reads!"

print("Padding reads ...")
reads = ["N" * pad_left + r + "N" * pad_right for r in reads]
#extract genome_id, genomic start and end positions of the reads
reads_info = []
for r in reads:
	r_info = re.split(">|:|\.\.", r.id)
	reads_info.append([r_info[0], int(r_info[1]), int(r_info[2])])

#create output directory
if not os.path.exists(args.out_dir):
	os.makedirs(args.out_dir)
# Specify input and output of the network
input_img = model.layers[0].input
layer_output = layer_dict[output_layer].output
iterate = K.function([input_img, K.learning_phase()], [layer_output])

print("Computing activations ...")
chunk_size = 10000
n = 0
while n < total_num_reads:
	print("Done "+str(n)+" from "+str(total_num_reads)+" sequences")
	samples_chunk = samples[n:n+chunk_size,:,:]
	reads_info_chunk = reads_info[n:n+chunk_size]

	activations = iterate([samples_chunk, 0])[0] #activations.shape = [total_num_reads, len_reads, n_filters]
	n_filters = activations.shape[-1]
	for filter_index in range(n_filters):
	
		print("Processing filter " + str(filter_index) + " ...")
		filter_bed_file = args.out_dir + "/" + test_data_set_name + "_filter_" + str(filter_index) + ".bed"
	
		pos_indices = np.where(activations[:,:,filter_index] > 0)
		with open(filter_bed_file, 'a') as csv_file:
			file_writer = csv.writer(csv_file, delimiter='\t')
			for i  in range(len(pos_indices[0])):
				read = pos_indices[0][i]
				neuron = pos_indices[1][i]
				genomic_start = neuron - pad_left + reads_info_chunk[read][1]
				genomic_end = genomic_start + motif_length
				activation_score = activations[read, neuron, filter_index]
				file_writer.writerow([reads_info_chunk[read][0], max(0, genomic_start), genomic_end, "filter_"+str(filter_index), '%.4g' % activation_score])

	n += chunk_size
