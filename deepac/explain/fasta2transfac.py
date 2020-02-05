import os
import re
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from Bio.motifs import transfac		
from Bio.motifs.__init__ import Instances
from Bio.motifs import matrix
import numpy as np
import csv

'''
Converts multiple sequence alignment saved in fasta format to transfac format (count matrix).
Either count each sequence once or weight count by DeepLIFT score a filter obtained for it.
'''


def weighted_count(instances_obj, m_weights):
    weighted_counts = {}
    for letter in instances_obj.alphabet.letters:
        weighted_counts[letter] = [0] * instances_obj.length
    for idx, instance in enumerate(instances_obj):
        for position, letter in enumerate(instance):
            weighted_counts[letter][position] += m_weights[idx]
    return weighted_counts


def fa2transfac(args):

    # create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # for each convolutional filter
    for file in os.listdir(args.in_dir):

        if bool(re.search("_motifs_filter_[0-9]+.fasta", file)) and os.stat(args.in_dir + "/" + file).st_size > 0:

            c_filter = re.search("filter_[0-9]+", file).group()
            print("Processing " + c_filter)

            # load sequences from fasta file
            instances = []
            with open(args.in_dir + "/" + file, "rU") as handle:
                for record in SeqIO.parse(handle, "fasta", alphabet=IUPAC.ambiguous_dna):
                    instances.append(record.seq)

            # build motif from sequences
            m = transfac.Motif(instances=Instances(instances, IUPAC.ambiguous_dna), alphabet=IUPAC.ambiguous_dna)
            m["ID"] = c_filter

            # weight sequences according to their DeepLIFT score
            if args.weighting:

                # file contains all non-zero contribution scores per read and filter
                file_weights = [filename for filename in os.listdir(args.weight_dir)
                                if bool(re.search("_(?:rel|act)_"+c_filter+"\.csv", filename))]

                if len(file_weights) != 1 or os.stat(args.weight_dir + "/" + file_weights[0]).st_size == 0:
                    print("File with " + c_filter + " weights is missing or empty!")
                    continue

                motif_weights = np.zeros(len(m.instances), dtype=np.float32)
                c = 0
                with open(args.weight_dir + "/" + file_weights[0], 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        for ind, row in enumerate(reader):
                                if ind % 3 == 2:
                                    motif_weights[c:c+len(row)] = [abs(float(entry)) for entry in row]
                                    c += len(row)

                assert c == len(m.instances), "Number sequences and number of weights differ ... Abort!"

                weighted_counts = weighted_count(m.instances, motif_weights)
                # override count matrix
                m.counts = matrix.FrequencyPositionMatrix(m.instances.alphabet, weighted_counts)

            # save motif in transfac format
            with open(
                    args.out_dir + "/" +
                    file.replace(".fasta", str("_seq_weighting" if args.weighting else "") + ".transfac"), "w")\
                    as out_file:
                out_file.write(m.format("transfac"))
