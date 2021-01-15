from Bio import SeqIO
from Bio.SeqIO import parse
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import os
import statistics
import csv

# script to analyze the downloaded genebank files
# needs readjustmenet to be reused

dir = "/home/uli/Documents/RKI/genebank_files_test_data/HP"

meta_info_gb_files = {}

sub_dirs = os.listdir(dir)
for sub_dir in sub_dirs:
    meta_info_gb_files[sub_dir] = {}
    files = os.listdir(os.path.join(dir,sub_dir))

    for file in files:
        meta_info_gb_files[sub_dir][file] = {}
        meta_info_gb_files[sub_dir][file]["gene_lengths"] = []
        meta_info_gb_files[sub_dir][file]["gene_distance"] = []
        meta_info_gb_files[sub_dir][file]["gene_numb"] = 0
        meta_info_gb_files[sub_dir][file]["gene_name"] = []

        meta_info_gb_files[sub_dir][file]["cds_lengths"] = []
        meta_info_gb_files[sub_dir][file]["cds_distance"] = []
        meta_info_gb_files[sub_dir][file]["cds_numb"] = 0

        pos_end_last_gene = None
        pos_end_last_cds = None

        gb_file = os.path.join(dir,sub_dir,file)
        record =  SeqIO.read(open(gb_file,"r"), "genbank")
        meta_info_gb_files[sub_dir][file]["seq_length"] = len(record.seq)
        meta_info_gb_files[sub_dir][file]["organism"] = record.annotations["organism"]

        for feature in record.features:
            if feature.type == "gene":
                meta_info_gb_files[sub_dir][file]["gene_lengths"].append(feature.location.nofuzzy_end - feature.location.nofuzzy_start)
                if pos_end_last_gene:
                    meta_info_gb_files[sub_dir][file]["gene_distance"].append(feature.location.nofuzzy_start - pos_end_last_gene)
                pos_end_last_gene = feature.location.nofuzzy_end
                meta_info_gb_files[sub_dir][file]["gene_numb"] += 1
                if "gene" in feature.qualifiers:
                    meta_info_gb_files[sub_dir][file]["gene_name"].append(",".join(feature.qualifiers["gene"]))
                else:
                    meta_info_gb_files[sub_dir][file]["gene_name"].append("")

            if feature.type == "cds":
                break
                meta_info_gb_files[sub_dir][file]["cds_lengths"].append(feature.location.nofuzzy_end - feature.location.nofuzzy_start)
                if pos_end_last_cds:
                    meta_info_gb_files[sub_dir][file]["cds_distance"].append(feature.location.nofuzzy_start - pos_end_last_cds)
                pos_end_last_cds = feature.location.nofuzzy_end
                meta_info_gb_files[sub_dir][file]["cds_numb"] += 1



with open("HP_metadata.csv","w") as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(["filename","organism","type","contig_id","seq_length",
                     "avg_gene_length","avg_gene_dist","gene_numb","gene_names",
                     "avg_cds_length","avg_cds_dist","cds_numb"])from Bio import SeqIO
from Bio.SeqIO import parse
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import os
import statistics
import csv

dir = "/home/uli/Documents/RKI/genebank_files_test_data/HP"

meta_info_gb_files = {}

sub_dirs = os.listdir(dir)
for sub_dir in sub_dirs:
    meta_info_gb_files[sub_dir] = {}
    files = os.listdir(os.path.join(dir,sub_dir))

    for file in files:
        meta_info_gb_files[sub_dir][file] = {}
        meta_info_gb_files[sub_dir][file]["gene_lengths"] = []
        meta_info_gb_files[sub_dir][file]["gene_distance"] = []
        meta_info_gb_files[sub_dir][file]["gene_numb"] = 0
        meta_info_gb_files[sub_dir][file]["gene_name"] = []

        meta_info_gb_files[sub_dir][file]["cds_lengths"] = []
        meta_info_gb_files[sub_dir][file]["cds_distance"] = []
        meta_info_gb_files[sub_dir][file]["cds_numb"] = 0

        pos_end_last_gene = None
        pos_end_last_cds = None

        gb_file = os.path.join(dir,sub_dir,file)
        record =  SeqIO.read(open(gb_file,"r"), "genbank")
        meta_info_gb_files[sub_dir][file]["seq_length"] = len(record.seq)
        meta_info_gb_files[sub_dir][file]["organism"] = record.annotations["organism"]

        for feature in record.features:
            if feature.type == "gene":
                meta_info_gb_files[sub_dir][file]["gene_lengths"].append(feature.location.nofuzzy_end - feature.location.nofuzzy_start)
                if pos_end_last_gene:
                    meta_info_gb_files[sub_dir][file]["gene_distance"].append(feature.location.nofuzzy_start - pos_end_last_gene)
                pos_end_last_gene = feature.location.nofuzzy_end
                meta_info_gb_files[sub_dir][file]["gene_numb"] += 1
                if "gene" in feature.qualifiers:
                    meta_info_gb_files[sub_dir][file]["gene_name"].append(",".join(feature.qualifiers["gene"]))
                else:
                    meta_info_gb_files[sub_dir][file]["gene_name"].append("")

            if feature.type == "cds":
                break
                meta_info_gb_files[sub_dir][file]["cds_lengths"].append(feature.location.nofuzzy_end - feature.location.nofuzzy_start)
                if pos_end_last_cds:
                    meta_info_gb_files[sub_dir][file]["cds_distance"].append(feature.location.nofuzzy_start - pos_end_last_cds)
                pos_end_last_cds = feature.location.nofuzzy_end
                meta_info_gb_files[sub_dir][file]["cds_numb"] += 1


    for filename in meta_info_gb_files:
        fileinfo = meta_info_gb_files[filename]
        for contigname in meta_info_gb_files[filename]:
            contiginfo = meta_info_gb_files[filename][contigname]
            organism = contiginfo["organism"]
            type = "pathogen"
            seq_length = contiginfo["seq_length"]

            if contiginfo["gene_lengths"]:
                avg_gene_length = statistics.mean(contiginfo["gene_lengths"])
            else:
                avg_gene_length = None

            if contiginfo["gene_distance"]:
                avg_gene_dist = statistics.mean(contiginfo["gene_distance"])
            else:
                avg_gene_dist = None

            gene_numb = contiginfo["gene_numb"]
            gene_names = ",".join(contiginfo["gene_name"])

            if contiginfo["cds_lengths"]:
                avg_cds_length = statistics.mean(contiginfo["cds_lengths"])
            else:
                avg_cds_length = None
            if contiginfo["cds_distance"]:
                avg_cds_dist = statistics.mean(contiginfo["cds_distance"])
            else:
                avg_cds_dist = None
            cds_numb = contiginfo["cds_numb"]

            writer.writerow([filename,organism,type,contigname,seq_length,
                             avg_gene_length, avg_gene_dist, gene_numb, gene_names,
                             avg_cds_length, avg_cds_dist, cds_numb])



contigs_total = [contig for fasta_file in meta_info_gb_files \
                        for contig in meta_info_gb_files[fasta_file] ]

total_seq_length = [meta_info_gb_files[fasta_file][contig]["seq_length"] \
                    for fasta_file in meta_info_gb_files \
                    for contig in meta_info_gb_files[fasta_file] ]


contigs_shorter_20000bp = [contig for fasta_file in meta_info_gb_files \
                                  for contig in meta_info_gb_files[fasta_file] \
                                  if meta_info_gb_files[fasta_file][contig]["seq_length"] < 20000]


contigs_shorter_20000bp_seq_length = [meta_info_gb_files[fasta_file][contig]["seq_length"]\
                            for fasta_file in meta_info_gb_files \
                            for contig in meta_info_gb_files[fasta_file] \
                            if meta_info_gb_files[fasta_file][contig]["seq_length"] < 20000]

species_total_after_cut_off = [fasta_file \
                               for fasta_file in meta_info_gb_files \
                               if len([contig for contig in meta_info_gb_files[fasta_file] \
                                       if meta_info_gb_files[fasta_file][contig]["seq_length"] < 20000]) != 0 ]

len(species_total_after_cut_off)


contigs_shorter_20000bp = [contig for fasta_file in meta_info_gb_files \
                                  for contig in meta_info_gb_files[fasta_file] \
                                  if meta_info_gb_files[fasta_file][contig]["seq_length"] < 20000]


contigs_shorter_20000bp_seq_length = [meta_info_gb_files[fasta_file][contig]["seq_length"]\
                            for fasta_file in meta_info_gb_files \
                            for contig in meta_info_gb_files[fasta_file] \
                            if meta_info_gb_files[fasta_file][contig]["seq_length"] < 20000]

species_affected_by_cutoff = [fasta_file \
                               for fasta_file in meta_info_gb_files \
                               if len([contig for contig in meta_info_gb_files[fasta_file] \
                                       if meta_info_gb_files[fasta_file][contig]["seq_length"] < 20000]) != 0 ]

species_not_affected_by_cutoff = [fasta_file \
                               for fasta_file in meta_info_gb_files \
                               if len([contig for contig in meta_info_gb_files[fasta_file] \
                                       if meta_info_gb_files[fasta_file][contig]["seq_length"] < 20000]) == 0 ]




sum(total_seq_length)

sum(contigs_shorter_20000bp_seq_length)

sum(contigs_shorter_20000bp_seq_length)/sum(total_seq_length)*100
