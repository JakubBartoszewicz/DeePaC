import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os


def generate_read(gc=0.5, length=250, header=None, ns=0.01):
    """Generate a random read."""
    if header is None:
        header = "random seq gc {}%".format(gc * 100)
    at_corr = 1 - gc - (ns/2)
    gc_corr = gc - (ns/2)
    arr = np.random.choice(['A', 'C', 'G', 'T', 'N'], size=length, p=[at_corr/2, gc_corr/2, gc_corr/2, at_corr/2, ns])
    seq = "".join(arr)
    rec = SeqRecord(Seq(seq), header, '', '')
    return rec


def generate_reads(n, filename, gc=0.5, length=250, append=False, header=None):
    """Generate random reads to a fasta file."""
    reads = [generate_read(gc, length, header) for i in range(0, n)]
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
    generate_reads(n=neg_val//2, filename=os.path.join("deepac-tests", "sample-test.fasta"), gc=gc_neg)
    generate_reads(n=pos_val//2, filename=os.path.join("deepac-tests", "sample-test.fasta"), gc=gc_pos, append=True)
