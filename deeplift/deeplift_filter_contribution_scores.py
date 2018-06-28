import argparse
import os
import csv
import random
import numpy as np

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from Bio import SeqIO

import deeplift
from deeplift.conversion import kerasapi_conversion as kc
from deeplift.util import compile_func
import deeplift_with_filtering_conversion as conversion

###Calculates max abs nonzero deeplift contribution score per read and filter

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True, help="Model file (.h5)")
parser.add_argument("-t", "--test_data", required=True, help="Test data (.npy)")
parser.add_argument("-o", "--out_dir", default = ".", help="Output directory")
parser.add_argument("-r", "--ref_mode", default = "N", choices=['N', 'GC', 'own_ref_file'], help="Modus to calculate reference sequences")
parser.add_argument("-a", "--train_data", help="Train data (.npy), necessary to calculate reference sequences if ref_mode is 'GC'")
parser.add_argument("-f", "--ref_seqs", help="User provided reference sequences (.fasta) if ref_mode is 'own_ref_file'")
args = parser.parse_args()

model_file = args.model
test_data_file = args.test_data
out_dir = args.out_dir
ref_mode = args.ref_mode

if args.ref_mode == "GC":
	assert args.train_data is not None, "Training data (--train_data) is required to build reference sequences with the same GC-content!"
	train_data_file = args.train_data

if args.ref_mode == "own_ref_file":
	assert args.ref_seqs is not None, "File with own reference sequences (--ref_seqs) is missing!"
	ref_data_file = args.ref_seqs

#create output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#convert keras model to deeplift model
deeplift_model = conversion.convert_model_from_saved_files(model_file, nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)
conv_layer_idx = [type(layer).__name__ for layer in deeplift_model.get_layers()].index("Conv1DFilter")
n_filters = deeplift_model.get_layers()[conv_layer_idx].kernel.shape[-1]

print("Loading test sequences...")
test_data_set_name = os.path.splitext(os.path.basename(test_data_file))[0]
samples = np.load(test_data_file, mmap_mode='r')
total_num_reads = samples.shape[0]
len_reads = samples.shape[1]


#generate reference sequence with N's
if ref_mode == "N":
	print("Generating reference sequence with all N's...")
	num_ref_seqs = 1
	ref_samples = np.zeros((num_ref_seqs, len_reads, 4))

#create reference sequences with same GC content as the training data set
elif ref_mode == "GC":
	print("Generating reference sequences with same GC-content as training data set...")
	train_samples = np.load(train_data_file, mmap_mode='r')
	num_ref_seqs = 5
	ref_seqs = [0]*num_ref_seqs
	dna = [0,1,2,3] #A,C,G,T
	GC_content = np.sum(np.mean(np.mean(train_samples, axis = 1), axis = 0)[1:3])
	AT_content = 1 - GC_content
	probs = np.array([AT_content, GC_content, GC_content, AT_content])/2.
	for i in range(0, num_ref_seqs):
		ref_seqs[i] = np.random.choice(dna, p = probs, size = len_reads, replace = True)
	ref_samples = to_categorical(ref_seqs)
	#save reference sequences
	dict = {'0':'A', '1':'C', '2':'G', '3':'T'}
	with open(out_dir + '/' + test_data_set_name + '_references.fasta', 'w') as csv_file:
		file_writer = csv.writer(csv_file)
		for seq_id in range(num_ref_seqs):
			file_writer.writerow([">"+test_data_set_name+"_ref_"+str(seq_id)])
			file_writer.writerow(["".join([dict[str(base)] for base in ref_seqs[seq_id]])])

#load own reference sequences
elif ref_mode == "own_ref_file":
	print("Loading reference sequences...")
	tokenizer = Tokenizer(char_level=True)
	tokenizer.fit_on_texts('ACGT')
	ref_reads = list(SeqIO.parse(ref_data_file, "fasta"))
	ref_samples = np.array([np.array([tokenizer.texts_to_matrix(read)]) for read in ref_reads])
	# remove unused character
	if not np.count_nonzero(ref_samples[:,:,:,0]):
		ref_samples = ref_samples[:,:,:,1:5]
	ref_samples = ref_samples.squeeze(1)
	num_ref_seqs = ref_samples.shape[0]
else:
	raise ValueError("Unkown reference mode!")


#find scores of filters (convolutional layer) w.r.t. the layer preceding the sigmoid output layer
#compile scoring function
deeplift_contribs_func_filter = deeplift_model.get_target_contribs_func(find_scores_layer_idx = conv_layer_idx, target_layer_idx = -2, conv_layer_idx = conv_layer_idx)
chunk_size = 50000
i = 0
while i < total_num_reads:

	print("Done "+str(i)+" from "+str(total_num_reads)+str(" sequences"))
	samples_chunk = samples[i:i+chunk_size,:,:]
	num_reads = samples_chunk.shape[0]
	
	input_data_list = np.repeat(samples_chunk, num_ref_seqs, axis = 0)
	input_references_list = np.concatenate([ref_samples]*num_reads, axis = 0)

	scores_filter = np.array(deeplift_contribs_func_filter(task_idx = 0, input_data_list = [input_data_list], input_references_list = [input_references_list], batch_size = 1, progress_update = None))
	#average the results per ref sequence
	scores_filter = np.reshape(scores_filter, [num_reads, num_ref_seqs, len_reads, n_filters])
	scores_filter_avg = np.mean(scores_filter, axis = 1)

	#save contribution scores per sequence and filter (if filter got a nonzero contribution score)
	print("Save contribution scores of the filters ...")
	for filter_index in range(n_filters):
		filter_rel_file = out_dir + "/" + test_data_set_name + "_rel_filter_%d.csv" % filter_index
		with open(filter_rel_file, 'a') as csv_file:
			file_writer = csv.writer(csv_file)
			for seq_id in range(num_reads):
				if np.any(scores_filter_avg[seq_id,:,filter_index] != 0):
					file_writer.writerow([">"+test_data_set_name+"_seq_"+str(seq_id+i)])
					nonzero_neurons = np.nonzero(scores_filter_avg[seq_id, :, filter_index])[0]
					file_writer.writerow(nonzero_neurons)
					file_writer.writerow(scores_filter_avg[seq_id, nonzero_neurons, filter_index])
	i += chunk_size
