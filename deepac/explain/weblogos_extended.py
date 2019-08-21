import os
import sys
import argparse
from weblogolib import *
from weblogolib.colorscheme import ColorRule, ColorScheme, SymbolColor, IndexColor
from weblogolib.color import Color
from weblogolib import parse_prior
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

'''
Build extended weblogos per convolutional filter with nucleotide coloring.
'''


#parse command line options
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fasta_dir", required=True, help="Directory containing motifs per filter (.fasta)")
parser.add_argument("-s", "--scores_dir", required=True, help="Directory containing nucleotide scores per filter (.csv)")
parser.add_argument("-l", "--logo_dir", help="Directory containing motifs in weighted transfac format (only required if weighted weblogos should be created)")
parser.add_argument("-t", "--train_data", help="Training data set to compute GC-content")
parser.add_argument("-o", "--out_dir", required=True, help="Output directory")
args = parser.parse_args()

samples = np.load(args.train_data, mmap_mode='r')
GC_content = np.sum(np.mean(np.mean(samples, axis = 1), axis = 0)[1:3])
AT_content = 1 - GC_content
pseudocounts = np.array([AT_content, GC_content, GC_content, AT_content]) / 2.0
motif_len = 15
pseudocounts = np.reshape(np.concatenate([pseudocounts]*motif_len, axis = 0), [motif_len,4])

#create output directory
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)


#define own color rules (color depends on position and nucleotide)
class SymbolIndexColor(ColorRule):
    def __init__(self, symbols, indices, color, description=None):
            self.symbols = symbols.upper()
            self.indices = indices
            self.color = Color.from_string(color)
            self.description = description

    def symbol_color(self, seq_index, symbol, rank):
            if symbol.upper() in self.symbols and seq_index in self.indices:
                return self.color


map = dict({'A':0, 'C':1, 'G':2, 'T':3})
#nucleotide color scheme: blue - grey - red
colormap = plt.cm.coolwarm

#for each convolutional filter
for filter_index in range(512):

    file_fasta = [ filename for filename in os.listdir(args.fasta_dir) if bool(re.search("_motifs_filter_"+str(filter_index)+"\.fasta", filename)) ]
    file_scores = [ filename for filename in os.listdir(args.scores_dir) if bool(re.search("rel_filter_"+str(filter_index)+"_nucleotides\.csv", filename)) ]

    #load transfac files
    if args.logo_dir:
        file_transfac = [ filename for filename in os.listdir(args.logo_dir) if bool(re.search("filter_"+str(filter_index)+"_seq_weighting.transfac", filename)) ]
        if len(file_transfac) == 0:
            continue
    if len(file_fasta) == 1 and len(file_scores) == 1:

        print("Processing filter: " + str(filter_index))

        #load nucleotide contribution scores
        contribution_scores = []
        with open(args.scores_dir + "/" + file_scores[0], 'r') as csvfile:
            reader = csv.reader(csvfile)
            for ind, row in enumerate(reader):
                if ind % 2 == 1:
                    scores = np.array(row, dtype = np.float32)
                    contribution_scores.append(scores)

        #load motifs from fasta file
        fin = open(args.fasta_dir + "/" + file_fasta[0])
        seqs = read_seq_data(fin)

        #load weighted count matrix from transfac file
        if args.logo_dir:
            from corebio.matrix import Motif
            fin = open(args.logo_dir + "/" + file_transfac[0])
            motif = Motif.read_transfac(fin)
            prior = parse_prior(str(GC_content), motif.alphabet)
            data = LogoData.from_counts(motif.alphabet, motif, prior)
            out_file_name = args.out_dir + "/weblogo_extended_" + file_transfac[0].replace(".transfac", ".jpeg")
        else:
            prior = parse_prior(str(GC_content), seqs.alphabet)
            data = LogoData.from_seqs(seqs, prior)
            out_file_name = args.out_dir + "/weblogo_extended_" + file_fasta[0].replace(".fasta", ".jpeg")

        seq_names = [seq.name for seq in seqs]
        seen = set()
        seqs_unique =  [seqs[idx] for idx, seq_name in enumerate(seq_names) if seq_name not in seen and not seen.add(seq_name)]

        assert len(contribution_scores) == len(seqs_unique), "ERROR"

        #compute mean contribution score per nucleotide and logo position
        mean_scores = np.zeros((len(seqs_unique[0]), len(seqs.alphabet)))
        counts = np.zeros_like(data.counts.array)
        for id, read in enumerate(seqs_unique):
            for pos, base in enumerate(read):
                base = str(base)
                if base in map.keys():
                    mean_scores[pos, map[base]] += contribution_scores[id][pos]
                    counts[pos, map[base]] += 1

        mean_scores /= (counts + pseudocounts) #add pseudocount to avoid divion by 0

        #normalize scores to [0, 255] and assign color according the selected color scheme
        min = -(1/(250*512))
        max = 1/(250*512)
        norm_scores = ((mean_scores-min)/(max-min)) * 255
        color_rules = []
        for base in map.keys():
            for pos in range(len(seqs[0])):
                color = matplotlib.colors.rgb2hex(colormap(int(norm_scores[pos,map[base]])))
                color_rules.append(SymbolIndexColor(base, [pos], color))

        #set logo options
        options = LogoOptions()
        options.logo_title = "filter " + str(filter_index)
        options.color_scheme = ColorScheme(color_rules)
        options.stack_width = std_sizes["large"]

        #save filter logo
        format = LogoFormat(data, options)
        jpeg = jpeg_formatter(data, format)
        with open(out_file_name, 'wb') as out_file:
            out_file.write(jpeg)

