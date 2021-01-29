import os
import argparse
from Bio import SeqIO
import pdb
import shutil
import multiprocessing as mp

# before integrating DeepSimualtor into the already exisitng R scripts
# a separate script for read simulation was written

def simulate_reads(genome, read_numb, read_length_mean, simulator, output_path):
    if simulator == "DeepSimulator":
        command = "./deep_simulator.sh -i %s -n %s -l %s -o %s -c 12" % (genome,read_numb,read_length_mean,output_path)
        os.system(command)

def calc_read_numb(file,total_seq_length,total_read_number):
    genome_size = 0
    for record in SeqIO.parse(file,"fasta"):
        genome_size += len(record.seq)
    read_numb_per_genome = int(genome_size/total_seq_length * total_read_number)
    return read_numb_per_genome

def split_multi_fasta(file,output_path):
    if is_fasta(file):
        for record in SeqIO.parse(file, "fasta"):
            file_name = os.path.basename(file) + "_" + str(record.id)
            SeqIO.write(record, os.path.join(output_path,file_name+".fa"), "fasta")

# https://stackoverflow.com/questions/44293407/how-can-i-check-whether-a-given-file-is-fasta
def is_fasta(filename):
    with open(filename, "r") as handle:
        fasta = SeqIO.parse(handle, "fasta")
        return any(fasta)

def get_total_seq_length(folder):
    total_seq_length = 0
    fasta_files = [os.path.join(folder,file) for file in os.listdir(folder)]
    for file in fasta_files:
        for record in SeqIO.parse(file,"fasta"):
            total_seq_length += len(record.seq)
    return total_seq_length

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Simulate nanopore sequecning data.""")

    parser.add_argument("-i",type = str, dest='input', nargs="+", required=True,
                        help="folder(s) containing the assemblies")

    parser.add_argument("-o",type = str, dest='output', required=True,
                        help="path output")

    parser.add_argument("-s",type = str, dest='simulator', required=True,
                            help="simulator")

    parser.add_argument("-rl", type = int, dest='read_length',
                        help = "mean read length")

    parser.add_argument("-rn", type = int, dest='read_number',
                        help = "total read number")

    args = parser.parse_args()

    for folder in args.input:
        temp_path = os.path.join(folder, ".temp")
        try:
            os.makedirs(temp_path)
        except:
            pass

        files = [ os.path.join(folder,file) for file in os.listdir(folder) if os.path.isfile(os.path.join(folder,file)) ]
        for file in files:
            split_multi_fasta(file,temp_path)


    seq_length_total = get_total_seq_length(temp_path)
    files_temp = [ os.path.join(temp_path,file) for file in os.listdir(temp_path) ]
    for file in files_temp:
        read_numb = calc_read_numb(file,seq_length_total,args.read_number)
        out_folder = args.output + os.path.basename(file)
        simulate_reads(file,read_numb,args.read_length,args.simulator,out_folder)
        
    shutil.rmtree(temp_path)

