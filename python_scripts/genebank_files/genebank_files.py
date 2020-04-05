from Bio import SeqIO
from Bio import Entrez
import os
import time

# script to get the genebank files for a test set seqs
# they were used to investigate the paired read effect
# to quantify avg gene length, seq loss through length cut off (small contigs) ...
Entrez.email = 'uligenske@gmail.com'
dir_fasta = "/home/uli/Desktop/readSimulation/assemblies_raw/test/HP"
dir_genebank = "/home/uli/Documents/genebank_files_test_data/HP"
files = os.listdir(dir_fasta)

for fasta in files[0:1]:
    target_dir = os.path.join(dir_genebank,fasta)
    try:
        os.mkdir(target_dir)
    except:
        pass
    file_path = os.path.join(dir_fasta,fasta)

    for seq_record in SeqIO.parse(file_path, "fasta"):
        time.sleep(1)
        record_id = str(seq_record.id)
        genebank_file = Entrez.efetch(db='nucleotide', id=record_id, rettype='gb', retmode='text')
        filename = os.path.join(target_dir,record_id) + ".gb"
        file = open(filename,"w")
        file.write(genebank_file.read())
        file.close()
