import numpy as np
import time
import os
import h5py
import itertools
import csv


from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras import backend as K
from Bio import SeqIO
import tensorflow as tf

#config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
#                        allow_soft_placement=True, device_count = {'CPU': 1})
#session = tf.Session(config=config)
#K.set_session(session)

# Creates the model and loads weights
model = load_model("cnn-16-e42-d5-f512x15-h256_fold1.h5")
motif_length = 15
pad = (motif_length - 1)/2
n_filters = 512

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts('ACGT')
print("Loading...")
reads = list(SeqIO.parse("SCRATCH_NOBAK/pathogenic_test_1_trimmed.fasta", "fasta"))
print("Preprocessing...")
samples = np.array([np.array([tokenizer.texts_to_matrix(read)]) for read in reads])
#pad reads
reads = ["N" * pad + r + "N" * pad for r in reads]

# Specify input and output of the network
input_img = model.layers[0].input
layer_dict = dict([(layer.name, layer) for layer in model.layers])
layer_name = 'conv1d_1'
layer_output = layer_dict[layer_name].output

# List of best motifs
max_motifs = []
with open('SCRATCH_NOBAK/visfilterd3/patho_max_motifs_fold1_384-512.csv', 'ab') as csv_file:
	file_writer = csv.writer(csv_file)
	file_writer.writerow(("id","seq","ma", "mma"))
for filter_index in range(0,512): 
	print('Processing filter %d' % filter_index)
	start_time = time.time()
	# List of the generated 'images' after evaluation
	kept_images = []

	# The loss is the activation of the neuron for the chosen filter
	loss = K.max(layer_output[:, :, filter_index])
	motif_id = K.argmax(layer_output[:, :, filter_index])

	# this function returns the loss given the input picture
	# also add a flag to disable the learning phase (in our case dropout)
	iterate = K.function([input_img, K.learning_phase()], [loss, motif_id])
	done = 0
	todo = len(samples)
	# we run test activation for all samples
	for s in samples:
		loss_value, motif_start = iterate([s, 0]) # 0 for test phase
		kept_images.append((s, loss_value, motif_start))
		done += 1
		progress = float(done)/float(todo)
		if (done % 100000) == 0:
			print("%f %% done" % progress)

	end_time = time.time()
	print('Filter %d processed in %ds' % (filter_index, end_time - start_time))
	activations = [i[1] for i in kept_images]
	motif_starts = [i[2][0] for i in kept_images]
	max_act = max(activations)
	max_act_ids = [i for i, j in enumerate(activations) if j == max_act]
	max_motif = reads[max_act_ids[0]][motif_starts[max_act_ids[0]]:motif_starts[max_act_ids[0]]+motif_length]
	max_motif.id = "filter %d" % filter_index
	max_motif.name = "filter %d" % filter_index

	pos_act_ids = [i for i, j in enumerate(activations) if j > 0.0]
	mean_max_act = np.average(activations)
	print("%d activations per %d samples" % (len(pos_act_ids), len(samples)))
	print("max. motif: %s - mean max. activation: %f - max. activation: %f" % (str(max_motif.seq), mean_max_act, max_act))
	max_motifs.append((filter_index, max_motif, max_act, mean_max_act))
	row = filter_index, str(max_motif.seq), max_act, mean_max_act
	with open('SCRATCH_NOBAK/visfilterd3/patho_max_motifs_fold1_384-512.csv', 'ab') as csv_file:
		file_writer = csv.writer(csv_file)
		file_writer.writerow(row)
	motifs = [reads[i][motif_starts[i]:motif_starts[i]+motif_length] for i in pos_act_ids]
	filename = "SCRATCH_NOBAK/visfilterd3/patho_motifs_filter_fold1_%d.fasta" % filter_index
	SeqIO.write(motifs, filename, "fasta")
SeqIO.write([m[1] for m in max_motifs], "SCRATCH_NOBAK/visfilterd3/patho_max_motifs_fold1_384-512.fasta", "fasta")

