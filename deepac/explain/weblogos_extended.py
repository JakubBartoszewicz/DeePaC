from weblogo.matrix import Motif
from weblogo import *
from weblogo.colorscheme import ColorRule, ColorScheme
from weblogo.color import Color
from weblogo import parse_prior
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

"""
Define own color rules (color depends on position and nucleotide)
"""
class SymbolIndexColor(ColorRule):

    def __init__(self, symbols, indices, color, description=None):
        self.symbols = symbols.upper()
        self.indices = indices
        self.color = Color.from_string(color)
        self.description = description

    def symbol_color(self, seq_index, symbol, rank):
        if symbol.upper() in self.symbols and seq_index in self.indices:
            return self.color


def get_weblogos_ext(args):
    """ Build extended weblogos per convolutional filter with nucleotide coloring."""
    samples = np.load(args.train_data, mmap_mode='r')
    gc_content = np.sum(np.mean(np.mean(samples, axis=1), axis=0)[1:3])
    at_content = 1 - gc_content
    pseudocounts = np.array([at_content, gc_content, gc_content, at_content]) / 2.0
    motif_len = 15
    pseudocounts = np.reshape(np.concatenate([pseudocounts]*motif_len, axis=0), [motif_len, 4])

    # create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    letter_dict = dict({'A': 0, 'C': 1, 'G': 2, 'T': 3})
    # nucleotide color scheme: blue - grey - red
    colormap = plt.cm.coolwarm

    # for each convolutional filter
    for filter_index in range(512):

        file_fasta = [filename for filename in os.listdir(args.fasta_dir)
                      if bool(re.search("_motifs_filter_"+str(filter_index)+"\.fasta", filename))]
        file_scores = [filename for filename in os.listdir(args.scores_dir)
                       if bool(re.search("rel_filter_"+str(filter_index)+"_nucleotides\.csv", filename))]

        # load transfac files
        if args.logo_dir:
            file_transfac = [filename for filename in os.listdir(args.logo_dir)
                             if bool(re.search("filter_"+str(filter_index)+"_seq_weighting.transfac", filename))]
            if len(file_transfac) == 0:
                continue
        if len(file_fasta) == 1 and len(file_scores) == 1:

            print("Processing filter: " + str(filter_index))

            # load nucleotide contribution scores
            contribution_scores = []
            with open(args.scores_dir + "/" + file_scores[0], 'r') as csvfile:
                reader = csv.reader(csvfile)
                for ind, row in enumerate(reader):
                    if ind % 2 == 1:
                        scores = np.array(row, dtype=np.float32)
                        contribution_scores.append(scores)

            # load motifs from fasta file
            try:
                fin = open(args.fasta_dir + "/" + file_fasta[0])
                seqs = read_seq_data(fin)
            except IOError:
                print("No data, skipping.")
                continue

            # load weighted count matrix from transfac file
            if args.logo_dir:
                fin = open(args.logo_dir + "/" + file_transfac[0])
                motif = Motif.read_transfac(fin)
                prior = parse_prior(str(gc_content), motif.alphabet)
                data = LogoData.from_counts(motif.alphabet, motif, prior)
                out_file_name = args.out_dir + "/weblogo_extended_" + file_transfac[0].replace(".transfac", ".jpeg")
            else:
                prior = parse_prior(str(gc_content), seqs.alphabet)
                data = LogoData.from_seqs(seqs, prior)
                out_file_name = args.out_dir + "/weblogo_extended_" + file_fasta[0].replace(".fasta", ".jpeg")

            seq_names = [seq.name for seq in seqs]
            seen = set()
            seqs_unique = [seqs[idx] for idx, seq_name in enumerate(seq_names)
                           if seq_name not in seen and not seen.add(seq_name)]

            assert len(contribution_scores) == len(seqs_unique), "Numbers of contribution scores and sequences differ."

            # compute mean contribution score per nucleotide and logo position
            mean_scores = np.zeros((len(seqs_unique[0]), len(seqs.alphabet)))
            counts = np.zeros_like(data.counts.array)
            for r_id, read in enumerate(seqs_unique):
                for pos, base in enumerate(read):
                    base = str(base)
                    if base in letter_dict.keys():
                        mean_scores[pos, letter_dict[base]] += contribution_scores[r_id][pos]
                        counts[pos, letter_dict[base]] += 1

            # add pseudocount to avoid divion by 0
            mean_scores /= (counts + pseudocounts)

            # normalize scores to [0, 255] and assign color according the selected color scheme
            s_min = -(1/(250*512))
            s_max = 1/(250*512)
            norm_scores = ((mean_scores-s_min)/(s_max-s_min)) * 255
            color_rules = []
            for base in letter_dict.keys():
                for pos in range(len(seqs[0])):
                    custom_color = matplotlib.colors.rgb2hex(colormap(int(norm_scores[pos, letter_dict[base]])))
                    color_rules.append(SymbolIndexColor(base, [pos], custom_color))

            # set logo options
            options = LogoOptions()
            options.logo_title = "filter " + str(filter_index)
            options.color_scheme = ColorScheme(color_rules)
            options.stack_width = std_sizes["large"]

            # save filter logo
            l_format = LogoFormat(data, options)
            jpeg = jpeg_formatter(data, l_format)
            with open(out_file_name, 'wb') as out_file:
                out_file.write(jpeg)
