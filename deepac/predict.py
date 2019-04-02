import os
import argparse
from deepac.preproc import tokenize
from multiprocessing import Pool
from functools import partial

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import numpy as np
from Bio.SeqIO.FastaIO import SimpleFastaParser
import itertools


def predict(model, input_fasta, output, token_cores=8):

    p = Pool(processes=token_cores)

    alphabet = "ACGT"
    read_length = 250
    datatype = 'int8'

    ### Load ###
    print("Loading...")
    with open(input_fasta) as in_handle:
        data = [(title, seq) for (title, seq) in SimpleFastaParser(in_handle)]

    ### Preproc ###
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(alphabet)
    print("Preprocessing data...")
    with open(input_fasta) as input_handle:
        # Parse fasta and tokenize in parallel. Partial function takes tokenizer as a fixed argument.
        # Tokenize function is applied to the fasta sequence generator.
        x_data = np.asarray(p.map(partial(tokenize, tokenizer=tokenizer, datatype=datatype,
                                          read_length=read_length), [d[1] for d in data]))
    ### Predict ###
    print("Predicting...")
    y_pred = np.ndarray.flatten(model.predict(x_data))

    np.save(file=output, arr=y_pred)


def filter(input_fasta, predictions, output, threshold=0.5, print_potentials=False, precision=3):
    with open(input_fasta) as in_handle:
        fasta_data = [(title, seq) for (title, seq) in SimpleFastaParser(in_handle)]
    y_pred = np.load(predictions, mmap_mode='r')
    y_pred_class = (y_pred > threshold).astype('int8')
    fasta_filtered = list(itertools.compress(fasta_data, y_pred_class))
    if print_potentials and precision > 0:
        y_pred_filtered = [y for y in y_pred if y > threshold]
        with open(output, "w") as out_handle:
            for ((title, seq), y) in zip(fasta_filtered, y_pred_filtered):
                out_handle.write(
                    ">{}\n{}\n".format(title + " | pp={val:.{precision}f}".format(val=y, precision=precision), seq))
    else:
        with open(output, "w") as out_handle:
            for (title, seq) in fasta_filtered:
                out_handle.write(">{}\n{}\n".format(title, seq))
