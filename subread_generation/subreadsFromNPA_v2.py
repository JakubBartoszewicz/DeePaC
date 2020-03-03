# task: test the nns trained on 250bp on shorter reads
# 1) create subreads from numpy arrays of length 250bp
#   define length => keep information from first k nt rest is set to 0
# 2) not necessary to do the prediction and then evalute manually since this
# functionality is already part of deepac => use the evaluation functionality

import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description="""Modify numpy arrays. Delete all nts after k.""")

parser.add_argument("-npa",type = str, nargs="+", help="numpy array(s)")

parser.add_argument("-k", nargs="+", type = int,help = "subread length(s)")

def subreads(file,k):
    reads = np.load(file)
    readLength = np.shape(reads)[1]

    # delete nt after k
    reads.resize((reads.shape[0],k,reads.shape[2]))
    # save file
    np.save(file[:-4] + "_subread_" + str(k) + '.npy',reads)

args = parser.parse_args()

for file in args.npa:
    for k in args.k:
        subreads(file,k)
