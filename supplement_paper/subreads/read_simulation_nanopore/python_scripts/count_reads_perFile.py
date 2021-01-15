import os
import argparse
from Bio import SeqIO
from statistics import mean
import numpy as np

# this script was written to check the number of reads simulated with DeepSimulator
# was used during the testing of DeepSimulator and might need readjustment to be reused
def get_files(dir,ext):
    walk = os.walk(dir)
    fastas = []
    for root, dirs, files in walk:
       for name in files:
           if ext == name[-len(ext):]:
               fastas.append(os.path.join(root, name))
    return fastas

def get_seq_numb(file, filetype):
    total_seq_numb = 0
    for record in SeqIO.parse(file,filetype):
        total_seq_numb += 1
    return total_seq_numb

def get_seq_lengths(file, filetype):
    seq_lengths = []
    for record in SeqIO.parse(file, filetype):
        seq_lengths.append(len(record.seq))
    return seq_lengths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Count reads per fasta file and contig.""")

    parser.add_argument("-rf",type = str, dest='raw_files', nargs="+", required=True,
                        help="folder(s) containing the assemblies")

    parser.add_argument("-sd",type = str, dest='sim_data', required=True,
                        help="path simulated data")

    args = parser.parse_args()

    for folder in args.raw_files:
        fastas_raw = get_files(folder,"fna")
        fastas_raw = [os.path.basename(file) for file in fastas_raw]

    sim_data = get_files(args.sim_data,"pass.fastq")
    seq_numb_total = 0
    seq_numb_perFile = {}
    seq_length_perFile = {}
    for fasta in fastas_raw:
        sim_data_temp = [file for file in sim_data if fasta in file]
        seq_numb_perFile[fasta] = 0
        seq_length_perFile[fasta] = []
        for file in sim_data_temp:
            numb_seq = get_seq_numb(file,"fastq")
            seq_numb_total += numb_seq
            seq_numb_perFile[fasta] += numb_seq
            if numb_seq > 0:
                seq_length_perFile[fasta] += get_seq_lengths(file,"fastq")

    print("Total number of sequences is %s" % seq_numb_total )
    for fasta, seq_num in seq_numb_perFile.items():
        counts, bins = np.histogram(seq_length_perFile[fasta],bins=10,range=(0,8000),density=False)
        print("Number of sequences for %s is %s." % (fasta,seq_num))
        print(counts)
        print(bins)
