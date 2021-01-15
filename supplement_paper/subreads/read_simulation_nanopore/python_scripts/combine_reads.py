import os
import argparse
from Bio import SeqIO
import pdb

def get_files(dir):
    walk = os.walk(dir)
    fastas = []
    for root, dirs, files in walk:
       for file in files:
           path = os.path.join(root, file)
           if is_fastq(path):
               fastas.append(path)
    return fastas


# https://stackoverflow.com/questions/44293407/how-can-i-check-whether-a-given-file-is-fasta
def is_fastq(filename):
    with open(filename, "r") as handle:
        fastq = SeqIO.parse(handle, "fastq")
        return any(fastq)


def combine_files(files, name_file,filter_length=0,start_pos=0,end_pos=0):
    with open(name_file + ".fasta", "w") as output_handle:
        for file in files:
            for record in SeqIO.parse(file, "fastq"):
                if len(record.seq) >= filter_length:
                    record.letter_annotations = {}
                    if end_pos > 0:
                        record.seq = record.seq[start_pos:end_pos]
                    SeqIO.write(record, output_handle, "fasta")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Combine reads from all fasta files in a dir into one fasta file.
                       Optionally the reads can be filtered by length and 
                       the start and end position of the reads can be specified.""")

    parser.add_argument("-sd",type = str, dest='sim_data', required=True,
                        help="path simulated data")

    parser.add_argument("-n",type = str, dest='file_name', required=True,
                        help="name of the file")

    parser.add_argument("-l",type = int, dest='length', required=False,
                        help="filter length", nargs='?', default=0)

    parser.add_argument("-sp",type = int, dest='start_pos', required=False,
                        help="start position", nargs='?', default=0)

    parser.add_argument("-ep",type = int, dest='end_pos', required=False,
                        help="end position", nargs='?', default=0)
    

    args = parser.parse_args()

    sim_data = get_files(args.sim_data)
    combine_files(sim_data,args.file_name,args.length,args.start_pos,args.end_pos)
