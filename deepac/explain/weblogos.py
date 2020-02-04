import os
from corebio.matrix import Motif
import weblogolib
from weblogolib import *
import numpy as np
import re

'''
Build standard weblogos per convolutional filter.
'''


def get_weblogos(args):

    # create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    samples = np.load(args.train_data, mmap_mode='r')
    gc_content = np.sum(np.mean(np.mean(samples, axis = 1), axis = 0)[1:3])

    # for each convolutional filter
    for file in os.listdir(args.in_dir):

        if bool(re.search("_motifs_filter_[0-9]+.*" + args.file_ext, file)) and os.stat(args.in_dir + "/" + file).st_size > 0:

            c_filter = re.search("filter_[0-9]+", file).group()
            filter_index = c_filter.replace("filter_", "")
            print("Processing filter: " + filter_index)

            fin = open(args.in_dir + "/" + file)

            # load motifs from fasta file
            if args.file_ext == ".fasta":
                seqs = read_seq_data(fin)
                prior = weblogolib.parse_prior(str(gc_content), seqs.alphabet)
                data = LogoData.from_seqs(seqs, prior)

            # load count matrix from transfac file
            elif args.file_ext == ".transfac":

                motif = Motif.read_transfac(fin)
                prior = weblogolib.parse_prior(str(gc_content), motif.alphabet)
                try:
                    data = weblogolib.LogoData.from_counts(motif.alphabet, motif, prior)
                except ValueError as err:
                    print(err)
                    continue

            # set logo options
            options = LogoOptions()
            options.logo_title = "filter " + filter_index
            options.color_scheme = classic
            options.stack_width = std_sizes["large"]

            # save filter logo
            l_format = LogoFormat(data, options)
            jpeg = jpeg_formatter(data, l_format)
            with open(args.out_dir + "/weblogo_" + file.replace(args.file_ext, ".jpeg"), 'wb') as out_file:
                out_file.write(jpeg)