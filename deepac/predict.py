"""@package deepac.predict
Predict pathogenic potentials and use them to filter sequences of interest.

"""
from deepac.preproc import read_fasta, tokenize
from multiprocessing import Pool
from functools import partial
import time
import tensorflow as tf
import numpy as np
from Bio.SeqIO.FastaIO import SimpleFastaParser
import itertools


def predict_fasta(model, input_fasta, output, token_cores=8, datatype='int32'):
    """Predict pathogenic potentials from a fasta file."""

    alphabet = "ACGT"
    input_layer_id = [idx for idx, layer in enumerate(model.layers) if "Input" in str(layer)][0]
    read_length = model.get_layer(index=input_layer_id).get_output_at(0).shape[1]

    # Preproc
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(alphabet)

    print("Preprocessing data...")
    start = time.time()
    with open(input_fasta) as input_handle:
        # Parse fasta and tokenize in parallel. Partial function takes tokenizer as a fixed argument.
        # Tokenize function is applied to the fasta sequence generator.
        with Pool(processes=token_cores) as p:
            x_data = np.asarray(p.map(partial(tokenize, tokenizer=tokenizer, datatype=datatype,
                                              read_length=read_length), read_fasta(input_handle)))
    # Predict
    print("Predicting...")
    y_pred = np.ndarray.flatten(model.predict(x_data))
    end = time.time()
    print("Predictions for {} reads done in {} s".format(y_pred.shape[0], end - start))
    np.save(file=output, arr=y_pred)


def predict_npy(model, input_npy, output):
    """Predict pathogenic potentials from a preprocessed numpy array."""
    x_data = np.load(input_npy)
    # Predict
    print("Predicting...")
    start = time.time()
    y_pred = np.ndarray.flatten(model.predict(x_data))
    end = time.time()
    print("Predictions for {} reads done in {} s".format(y_pred.shape[0], end - start))
    np.save(file=output, arr=y_pred)


def filter_fasta(input_fasta, predictions, output, threshold=0.5, print_potentials=False, precision=3):
    """Filter reads in a fasta file by pathogenic potential."""
    filter_paired_fasta(input_fasta, predictions, output, input_fasta_2=None, predictions_2=None,
                        output_neg=None, threshold=threshold, print_potentials=print_potentials, precision=precision)


def ensemble(predictions_list, outpath_npy):
    ys = []
    for p in predictions_list:
        ys.append(np.load(p, mmap_mode='r'))
    ys = np.array(ys)
    y_pred = np.average(ys, 0)
    np.save(outpath_npy, y_pred)


def filter_paired_fasta(input_fasta_1, predictions_1, output_pos, input_fasta_2=None, predictions_2=None,
                        output_neg=None, threshold=0.5, print_potentials=False, precision=3):
    """Filter reads in paired fasta files by pathogenic potential."""
    with open(input_fasta_1) as in_handle:
        fasta_data_1 = [(title, seq) for (title, seq) in SimpleFastaParser(in_handle)]
    y_pred_1 = np.load(predictions_1, mmap_mode='r')
    if input_fasta_2 is not None:
        with open(input_fasta_2) as in_handle:
            fasta_data_2 = [(title, seq) for (title, seq) in SimpleFastaParser(in_handle)]
        y_pred_2 = np.load(predictions_2, mmap_mode='r')
        y_pred = (y_pred_1 + y_pred_2)/2
    else:
        y_pred = y_pred_1
    y_pred_pos = (y_pred > threshold).astype('int8')
    y_pred_neg = (y_pred <= threshold).astype('int8')

    fasta_pos_1 = list(itertools.compress(fasta_data_1, y_pred_pos))
    fasta_neg_1 = list(itertools.compress(fasta_data_1, y_pred_neg))
    if input_fasta_2 is not None:
        fasta_pos_2 = list(itertools.compress(fasta_data_2, y_pred_pos))
        fasta_neg_2 = list(itertools.compress(fasta_data_2, y_pred_neg))
    else:
        fasta_pos_2 = []
        fasta_neg_2 = []
    if print_potentials and precision > 0:
        y_pred_pos = [y for y in y_pred if y > threshold]
        with open(output_pos, "w") as out_handle:
            for ((title, seq), y) in zip(fasta_pos_1, y_pred_pos):
                out_handle.write(
                    ">{}\n{}\n".format(title + " | pp={val:.{precision}f}".format(val=y, precision=precision), seq))
            if input_fasta_2 is not None:
                for ((title, seq), y) in zip(fasta_pos_2, y_pred_pos):
                    out_handle.write(
                        ">{}\n{}\n".format(title + " | pp={val:.{precision}f}".format(val=y,
                                                                                      precision=precision), seq))
        if output_neg is not None:
            y_pred_neg = [y for y in y_pred if y <= threshold]
            with open(output_neg, "w") as out_handle:
                for ((title, seq), y) in zip(fasta_neg_1, y_pred_neg):
                    out_handle.write(
                        ">{}\n{}\n".format(title + " | pp={val:.{precision}f}".format(val=y, precision=precision), seq))
                if input_fasta_2 is not None:
                    for ((title, seq), y) in zip(fasta_neg_2, y_pred_neg):
                        out_handle.write(
                            ">{}\n{}\n".format(title + " | pp={val:.{precision}f}".format(val=y,
                                                                                          precision=precision), seq))
    else:
        with open(output_pos, "w") as out_handle:
            for (title, seq) in fasta_pos_1:
                out_handle.write(">{}\n{}\n".format(title, seq))
            if input_fasta_2 is not None:
                for (title, seq) in fasta_pos_2:
                    out_handle.write(">{}\n{}\n".format(title, seq))
        if output_neg is not None:
            with open(output_neg, "w") as out_handle:
                for (title, seq) in fasta_neg_1:
                    out_handle.write(">{}\n{}\n".format(title, seq))
                if input_fasta_2 is not None:
                    for (title, seq) in fasta_neg_2:
                        out_handle.write(">{}\n{}\n".format(title, seq))
