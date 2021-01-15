# task: test the nns trained on 250bp on shorter reads
# 1) create subreads from numpy arrays of length 250bp
#   define length => keep information from first k nt rest is set to 0
# 2) not necessary to do the prediction and then evalute manually since this
# functionality is already part of deepac => use the evaluation functionality

import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description="""Modify numpy arrays. Set all nts after k (chosen rand between 50 and 250) to [0,0,0,0].""")

parser.add_argument("-npa",type = str, nargs="+", help="numpy array(s)")

def subreads(file):
    reads = np.load(file)
    numbSeq = reads.shape[0]
    lengthSeq = reads.shape[1]
    lengthsRand = np.random.randint(25,250,size=numbSeq)
    startIndicies = lengthsRand + range(0,numbSeq*250,250)
    endIndicies = range(250,numbSeq+1*250,250)
    
    ranges = list(zip(startIndicies,endIndicies))
    indicies = [ list(range(r[0],r[1],1)) for r in ranges ]
    indicies = [val for sublist in indicies for val in sublist]

    reads = reads.reshape((numbSeq*lengthSeq,4))
    reads[indicies] = 0
    reads = reads.reshape((numbSeq,lengthSeq,4))

    # save file
    np.save(file[:-4] + "_random_subread_50-250" + '.npy',reads)

args = parser.parse_args()

for file in args.npa:
        subreads(file)
