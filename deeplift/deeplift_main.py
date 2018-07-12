import argparse
import os
import csv
import random
import numpy as np
import sys

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from Bio import SeqIO

import deeplift
from deeplift.util import compile_func
import deeplift_with_filtering_conversion as conversion


def main():
	'''
	Calculates DeepLIFT contribution scores for all neurons in the convolutional layer 
	and extract all motifs for which a filter neuron got a non-zero contribution score.
	'''

	#parse command line arguments
	args = parse_arguments()
	
	#convert keras model to deeplift model
	print("Building DeepLIFT model ...")
	deeplift_model = conversion.convert_model_from_saved_files(args.model, nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)
	#check whether model uses the RC-architecture and extract some model informations
	RC_architecture = "RevCompConv1DFilter" in [type(layer).__name__ for layer in deeplift_model.get_layers()]
	if RC_architecture:
		conv_layer_idx = [type(layer).__name__ for layer in deeplift_model.get_layers()].index("RevCompConv1DFilter")
		n_filters = 2 * deeplift_model.get_layers()[conv_layer_idx].kernel.shape[-1]
	else:
		conv_layer_idx = [type(layer).__name__ for layer in deeplift_model.get_layers()].index("Conv1DFilter")
		n_filters = deeplift_model.get_layers()[conv_layer_idx].kernel.shape[-1]
	motif_length = deeplift_model.get_layers()[conv_layer_idx].kernel.shape[0]
	pad = int((motif_length - 1)/2)
	#compile scoring function (find scores of filters (convolutional layer) w.r.t. the layer preceding the sigmoid output layer)
	deeplift_contribs_func_filter = deeplift_model.get_target_contribs_func(find_scores_layer_idx = conv_layer_idx, target_layer_idx = -2, conv_layer_idx = conv_layer_idx)

	
	print("Loading test data (.npy) ...")
	test_data_set_name = os.path.splitext(os.path.basename(args.test_data))[0]
	samples = np.load(args.test_data, mmap_mode='r')
	total_num_reads = samples.shape[0]
	len_reads = samples.shape[1]

	print("Loading test data (.fasta) ...")
	nonpatho_reads = list(SeqIO.parse(args.nonpatho_test, "fasta"))
	patho_reads = list(SeqIO.parse(args.patho_test, "fasta"))
	reads = nonpatho_reads + patho_reads
	print("Padding reads ...")
	reads = ["N" * pad + r + "N" * pad for r in reads]
	
	assert len(reads) == total_num_reads, "Test data in .npy-format and fasta files containing different number of reads!"

        #create output directory
	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)
	
	#load or create reference sequences
	ref_samples = get_reference_seqs(args, len_reads)
	num_ref_seqs = ref_samples.shape[0]
	
	print("Running DeepLIFT ...")
	chunk_size = 50000
	i = 0
	while i < total_num_reads:

		print("Done "+str(i)+" from "+str(total_num_reads)+" sequences")
		samples_chunk = samples[i:i+chunk_size,:,:]
		reads_chunk = reads[i:i+chunk_size]
		num_reads = samples_chunk.shape[0]
		
		input_data_list = np.repeat(samples_chunk, num_ref_seqs, axis = 0)
		input_references_list = np.concatenate([ref_samples]*num_reads, axis = 0)

		scores_filter = np.array(deeplift_contribs_func_filter(task_idx = 0, input_data_list = [input_data_list], input_references_list = [input_references_list], batch_size = 1, progress_update = None))
		scores_filter = np.reshape(scores_filter, [num_reads, num_ref_seqs, len_reads, n_filters])
		#average the results per ref sequence
		scores_filter_avg = np.mean(scores_filter, axis = 1)
			
		for filter_index in range(n_filters):

			#determine non-zero contribution scores per read and filter and extract DNA-sequence of corresponding subreads
			contribution_scores = []
			motifs = []
			for seq_id in range(num_reads):
				if np.any(scores_filter_avg[seq_id,:,filter_index]):
					non_zero_neurons = np.nonzero(scores_filter_avg[seq_id,:,filter_index])[0]
					scores = scores_filter_avg[seq_id, non_zero_neurons, filter_index]
					contribution_scores.append((reads_chunk[seq_id].id, non_zero_neurons, scores))
					motifs.append([reads_chunk[seq_id][non_zero_neuron:(non_zero_neuron+motif_length)] for non_zero_neuron in non_zero_neurons])

			if contribution_scores:

				#save filter contribution scores
				filter_rel_file = args.out_dir + "/" + test_data_set_name + "_rel_filter_%d.csv" % filter_index
				with open(filter_rel_file, 'a') as csv_file:
					file_writer = csv.writer(csv_file)
					for dat in contribution_scores:
						file_writer.writerow([">"+str(dat[0])])
						file_writer.writerow(dat[1])
						file_writer.writerow(dat[2])

				#save subreads which cause non-zero contribution scores
				filter_motifs_file = args.out_dir + "/" + test_data_set_name + "_motifs_filter_%d.fasta" % filter_index
				with open(filter_motifs_file, "a") as output_handle:
					SeqIO.write([subread for motif in motifs for subread in motif], output_handle, "fasta")
			
		i += chunk_size
	
def parse_arguments():
	'''
	Parse command line arguments.
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model", required=True, help="Model file (.h5)")
	parser.add_argument("-t", "--test_data", required=True, help="Test data (.npy)")
	parser.add_argument("-n", "--nonpatho_test", required=True, help="Nonpathogenic reads of the test data set (.fasta)")
	parser.add_argument("-p", "--patho_test", required=True, help="Pathogenic reads of the test data set (.fasta)")
	parser.add_argument("-o", "--out_dir", default = ".", help="Output directory")
	parser.add_argument("-r", "--ref_mode", default = "N", choices=['N', 'GC', 'own_ref_file'], help="Modus to calculate reference sequences")
	parser.add_argument("-a", "--train_data", help="Train data (.npy), necessary to calculate reference sequences if ref_mode is 'GC'")
	parser.add_argument("-f", "--ref_seqs", help="User provided reference sequences (.fasta) if ref_mode is 'own_ref_file'")
	args = parser.parse_args()
	if args.ref_mode == "GC" and args.train_data is None:
		raise ValueError("Training data (--train_data) is required to build reference sequences with the same GC-content!")
	if args.ref_mode == "own_ref_file" and args.ref_seqs is None:
		raise ValueError("File with own reference sequences (--ref_seqs) is missing!")
	
	return args
	
def get_reference_seqs(args, len_reads):
	'''
	Load or create reference sequences for DeepLIFT.
	'''
	#generate reference sequence with N's
	if args.ref_mode == "N":
	
		print("Generating reference sequence with all N's...")
		num_ref_seqs = 1
		ref_samples = np.zeros((num_ref_seqs, len_reads, 4))

	#create reference sequences with same GC content as the training data set
	elif args.ref_mode == "GC":
	
		print("Generating reference sequences with same GC-content as training data set...")
		train_samples = np.load(args.train_data, mmap_mode='r')
		num_ref_seqs = 5
		ref_seqs = [0]*num_ref_seqs
		#calculate frequency of each nucleotide (A,C,G,T,N) in the training data set
		probs = np.mean(np.mean(train_samples, axis = 1), axis = 0).tolist()
		probs.append(1-sum(probs))
		#generate reference seqs
		for i in range(num_ref_seqs):
			ref_seqs[i] = np.random.choice([0,1,2,3,4], p = probs, size = len_reads, replace = True)
		ref_samples = to_categorical(ref_seqs, num_classes = 5)
		#remove channel of N-nucleotide
		ref_samples = ref_samples[:,:,0:4]		
		dict = {0:'A', 1:'C', 2:'G', 3:'T', 4:'N'}
		train_data_set_name = os.path.splitext(os.path.basename(args.train_data))[0]
		#save reference sequences
		with open(args.out_dir + '/' + train_data_set_name + '_references.fasta', 'w') as csv_file:
			file_writer = csv.writer(csv_file)
			for seq_id in range(num_ref_seqs):
				file_writer.writerow([">"+train_data_set_name+"_ref_"+str(seq_id)])
				file_writer.writerow(["".join([dict[base] for base in ref_seqs[seq_id]])])
		del train_samples

	#load own reference sequences (args.ref_mode == "own_ref_file")
	else:
	
		print("Loading reference sequences...")
		tokenizer = Tokenizer(char_level=True)
		tokenizer.fit_on_texts('ACGT')
		ref_reads = list(SeqIO.parse(args.ref_seqs, "fasta"))
		ref_samples = np.array([np.array([tokenizer.texts_to_matrix(read)]) for read in ref_reads])
		# remove unused character
		if not np.count_nonzero(ref_samples[:,:,:,0]):
			ref_samples = ref_samples[:,:,:,1:5]
		ref_samples = ref_samples.squeeze(1)
		#num_ref_seqs = ref_samples.shape[0]

	return ref_samples


if __name__ == "__main__":
    main()
