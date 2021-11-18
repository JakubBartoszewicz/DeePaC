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


def generate_sample_data(gc_neg=0.3, gc_pos=0.7, n_train=1024, n_val=1024):
    """Generate a sample random dataset."""
    generate_multiclass_sample_data(gc_per_class=(gc_neg, gc_pos), n_train=n_train, n_val=n_val)


def generate_multiclass_sample_data(gc_per_class=(0.3, 0.7, 0.1, 0.9), n_train=1024, n_val=1024):
    """Generate a sample multiclass random dataset."""
    n_classes = len(gc_per_class)

    if not os.path.exists("deepac-tests"):
        os.makedirs("deepac-tests")

    train_left = n_train
    val_left = n_val
    for i in range(n_classes):
        class_train = min(np.ceil(n_train/n_classes).astype(int), train_left)
        train_left = train_left - class_train
        class_val = min(np.ceil(n_val/n_classes).astype(int), val_left)
        val_left = val_left - class_val
        generate_reads(n=class_train, filename=os.path.join("deepac-tests", "sample-train-{}.fasta".format(i)),
                       gc=gc_per_class[i])
        generate_reads(n=class_val, filename=os.path.join("deepac-tests", "sample-val-{}.fasta".format(i)),
                       gc=gc_per_class[i])

        with open(os.path.join("deepac-tests", "sample-val-all.fasta"), 'a') as outfile:
            with open(os.path.join("deepac-tests", "sample-val-{}.fasta".format(i))) as infile:
                for line in infile:
                    outfile.write(line)

        generate_reads(n=class_val//2, filename=os.path.join("deepac-tests", "sample-test.fasta"), gc=gc_per_class[i],
                       append=True)

        generate_species_csv(i, class_val, 2, f"deepac-tests-species.csv")


def generate_species_csv(class_id, n_reads, n_species, filename):
    if n_reads % n_species != 0:
        raise ValueError("Number of generated reads should be divisible by the number of mock species.")
    with open(os.path.join("deepac-tests", filename), 'a') as outfile:
        for s in range(n_species):
            outfile.write(f"{class_id};species_{class_id}_{s};{int(n_reads/n_species)}\n")
