import math
import argparse
import numpy as np
from Bio.motifs import transfac
import os

'''
Compute information content for each filter motif (.transfac).
'''

#parse command line options
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--in_file", required=True, help="File containing all filter motifs in transfac format")
parser.add_argument("-t", "--train", required=True, help="Training data set (.npy) to normalize for GC-content")
parser.add_argument("-o", "--out_file", default=True, help="Name of the output file")
args = parser.parse_args()

train_samples = np.load(args.train, mmap_mode='r')
probs = np.mean(np.mean(train_samples, axis = 1), axis = 0)
#background
bg = {'A': probs[0], 'C': probs[1], 'G': probs[2], 'T': probs[3]}	

#create output directory
out_dir = os.path.dirname(args.out_file)
if out_dir == "":
    out_dir = "."
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def compute_mean_IC(pwm):
	IC = 0.0
	for column in range(pwm.length):
		for base in pwm.alphabet.letters:
			IC += pwm[base][column] * math.log2(pwm[base][column]/pwm.background[base])
	IC /= float(pwm.length)
	return IC

#load all filter motifs	
with open(args.in_file) as handle:
	records = transfac.read(handle)


#for each motif compute IC:
for m in records:

	pwm = m.counts.normalize(pseudocounts = bg)
	pwm.background = bg
#	pssm = pwm.log_odds(background = bg))
	ic = compute_mean_IC(pwm)
	with open(args.out_file, "a") as file:
		file.write(m.get("ID") + "\t" + str(ic) + "\n")
		



		

