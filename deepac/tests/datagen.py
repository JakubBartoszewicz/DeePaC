import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os

def __generate_read(gc=0.5, length =250):
    """Generate a random read."""
    at = 1 - gc
    arr = np.random.choice(['A', 'C', 'G', 'T'], size=length, p=[at/2, gc/2, gc/2, at/2])
    seq = "".join(arr)
    rec = SeqRecord(Seq(seq), "random seq gc {}%".format(0.5 * 100), '', '')
    return rec

def generate_reads(n, filename, gc=0.5, length =250, append = False):
    """Generate random reads to a fasta file."""
    reads = [__generate_read(gc, length) for i in range(0, n)]
    if append:
        with open(filename, "a") as output_handle:
            SeqIO.write(reads, output_handle, "fasta")
    else:
        with open(filename, "w") as output_handle:
            SeqIO.write(reads, output_handle, "fasta")

def generate_sample_data(gc_pos=0.7, gc_neg=0.3, n_train=1024, n_val=1024):
    """Generate a sample random dataset."""
    pos_train = np.ceil(n_train/2).astype(int)
    neg_train = n_train - pos_train
    pos_val = np.ceil(n_val/2).astype(int)
    neg_val = n_val - pos_val

    if not os.path.exists("deepac-tests"):
        os.makedirs("deepac-tests")

    generate_reads(n=neg_train, filename=os.path.join("deepac-tests", "sample-train-neg.fasta"), gc=gc_neg)
    generate_reads(n=pos_train, filename=os.path.join("deepac-tests", "sample-train-pos.fasta"), gc=gc_pos)
    generate_reads(n=neg_val, filename=os.path.join("deepac-tests", "sample-val-neg.fasta"), gc=gc_neg)
    generate_reads(n=pos_val, filename=os.path.join("deepac-tests", "sample-val-pos.fasta"), gc=gc_pos)
