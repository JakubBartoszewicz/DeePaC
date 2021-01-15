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
from tqdm import tqdm
import os


def predict_fasta(model, input_fasta, output, token_cores=8, datatype='int32', rc=False, replicates=1, batch_size=512):
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
    y_pred, y_std = predict_array(model, x_data, output, rc, replicates, batch_size)
    end = time.time()
    print("Preprocessing & predictions for {} reads done in {} s".format(y_pred.shape[0], end - start))
    return y_pred, y_std


def predict_npy(model, input_npy, output, rc=False, replicates=1, batch_size=512):
    """Predict pathogenic potentials from a preprocessed numpy array."""
    x_data = np.load(input_npy, mmap_mode='r')
    return predict_array(model, x_data, output, rc, replicates, batch_size)


def predict_array(model, x_data, output, rc=False, replicates=1, batch_size=512):
    """Predict pathogenic potentials from a preprocessed numpy array."""
    if rc:
        x_data = x_data[::, ::-1, ::-1]
    # Predict
    print("Predicting...")
    start = time.time()
    if replicates > 1:
        y_preds = []
        for i in tqdm(range(replicates)):
            y_preds.append(np.ndarray.flatten(model.predict(x_data, batch_size=batch_size)))
        y_preds = np.asarray(y_preds)
        y_pred = y_preds.mean(axis=0)
        y_std = y_preds.std(axis=0)
    else:
        y_pred = np.ndarray.flatten(model.predict(x_data, batch_size=batch_size))
        y_std = None

    end = time.time()
    print("Predictions for {} reads done in {} s".format(y_pred.shape[0], end - start))
    np.save(file=output, arr=y_pred)
    if replicates > 1:
        out_std = "{}-std.npy".format(os.path.splitext(output)[0])
        np.save(file=out_std, arr=y_std)
    return y_pred, y_std


def filter_fasta(input_fasta, predictions, output, threshold=0.5, print_potentials=False, precision=3,
                 output_neg=None, confidence_thresh=0.5, output_undef=None, bayes_thresh=None, pred_uncertainty=None):
    """Filter reads in a fasta file by pathogenic potential."""
    filter_paired_fasta(input_fasta, predictions, output, input_fasta_2=None, predictions_2=None,
                        output_neg=output_neg, threshold=threshold, print_potentials=print_potentials,
                        precision=precision, confidence_thresh=confidence_thresh, output_undef=output_undef,
                        bayes_thresh=bayes_thresh, pred_uncertainty=pred_uncertainty)


def ensemble(predictions_list, outpath_npy):
    ys = []
    for p in predictions_list:
        ys.append(np.load(p, mmap_mode='r'))
    ys = np.array(ys)
    y_pred = np.average(ys, 0)
    np.save(outpath_npy, y_pred)


def filter_paired_fasta(input_fasta_1, predictions_1, output_pos, input_fasta_2=None, predictions_2=None,
                        output_neg=None, threshold=0.5, print_potentials=False, precision=3,
                        confidence_thresh=0.5, output_undef=None, bayes_thresh=None, pred_uncertainty=None):
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

    if np.isclose(confidence_thresh, threshold):
        y_pred_class_pos = y_pred > threshold
        y_pred_class_neg = y_pred <= threshold
        y_pred_class_undef = []
    else:
        interval = np.abs(confidence_thresh - threshold)
        y_pred_class_pos = y_pred > (threshold + interval)
        y_pred_class_neg = y_pred < (threshold - interval)
        y_pred_class_undef = np.logical_not(np.any([y_pred_class_pos, y_pred_class_neg], axis=0))

    fasta_pos_1 = list(itertools.compress(fasta_data_1, y_pred_class_pos))
    fasta_neg_1 = list(itertools.compress(fasta_data_1, y_pred_class_neg))
    fasta_undef_1 = list(itertools.compress(fasta_data_1, y_pred_class_undef))
    if input_fasta_2 is not None:
        fasta_pos_2 = list(itertools.compress(fasta_data_2, y_pred_class_pos))
        fasta_neg_2 = list(itertools.compress(fasta_data_2, y_pred_class_neg))
        fasta_undef_2 = list(itertools.compress(fasta_data_2, y_pred_class_undef))
    else:
        fasta_pos_2 = []
        fasta_neg_2 = []
        fasta_undef_2 = []
    if print_potentials and precision > 0:
        y_pred_pos = y_pred[y_pred_class_pos]
        with open(output_pos, "w") as out_handle:
            if pred_uncertainty is not None:
                pred_uncertainty = np.load(pred_uncertainty)
                for ((title, seq), y, ci) in zip(fasta_pos_1, y_pred_pos, pred_uncertainty[y_pred_class_pos]):
                    out_handle.write(
                        ">{}\n{}\n".format(title +
                                           " | pp={val:.{precision}f} "
                                           "+/- {ci:.{precision}f}".format(val=y,
                                                                           precision=precision,
                                                                           ci=ci),
                                           seq))
                if input_fasta_2 is not None:
                    for ((title, seq), y, ci) in zip(fasta_pos_2, y_pred_pos, pred_uncertainty[y_pred_class_pos]):
                        out_handle.write(
                            ">{}\n{}\n".format(title +
                                               " | pp={val:.{precision}f} "
                                               "+/- {ci:.{precision}f}".format(val=y,
                                                                               precision=precision,
                                                                               ci=ci),
                                               seq))
            else:
                for ((title, seq), y) in zip(fasta_pos_1, y_pred_pos):
                    out_handle.write(
                        ">{}\n{}\n".format(title + " | pp={val:.{precision}f}".format(val=y, precision=precision),
                                           seq))
                if input_fasta_2 is not None:
                    for ((title, seq), y) in zip(fasta_pos_2, y_pred_pos):
                        out_handle.write(
                            ">{}\n{}\n".format(title + " | pp={val:.{precision}f}".format(val=y,
                                                                                          precision=precision), seq))
        if output_neg is not None:
            y_pred_neg = y_pred[y_pred_class_neg]
            with open(output_neg, "w") as out_handle:
                if pred_uncertainty is not None:
                    for ((title, seq), y, ci) in zip(fasta_neg_1, y_pred_neg, pred_uncertainty[y_pred_class_neg]):
                        out_handle.write(
                            ">{}\n{}\n".format(title + " | pp={val:.{precision}f} "
                                                       "+/- {ci:.{precision}f}".format(val=y,
                                                                                       precision=precision,
                                                                                       ci=ci),
                                               seq))
                    if input_fasta_2 is not None:
                        for ((title, seq), y, ci) in zip(fasta_neg_2, y_pred_neg, pred_uncertainty[y_pred_class_neg]):
                            out_handle.write(
                                ">{}\n{}\n".format(title + " | pp={val:.{precision}f} "
                                                           "+/- {ci:.{precision}f}".format(val=y,
                                                                                           precision=precision,
                                                                                           ci=ci),
                                                   seq))
                else:
                    for ((title, seq), y) in zip(fasta_neg_1, y_pred_neg):
                        out_handle.write(
                            ">{}\n{}\n".format(title + " | pp={val:.{precision}f}".format(val=y, precision=precision),
                                               seq))
                    if input_fasta_2 is not None:
                        for ((title, seq), y) in zip(fasta_neg_2, y_pred_neg):
                            out_handle.write(
                                ">{}\n{}\n".format(title + " | pp={val:.{precision}f}".format(val=y,
                                                                                              precision=precision),
                                                   seq))
                y_pred_undef = y_pred[y_pred_class_undef]
                if not np.isclose(confidence_thresh, threshold):
                    with open(output_undef, "w") as out_handle:
                        if pred_uncertainty is not None:
                            for ((title, seq), y, ci) in zip(fasta_undef_1, y_pred_undef,
                                                             pred_uncertainty[y_pred_class_undef]):
                                out_handle.write(
                                    ">{}\n{}\n".format(title + " | pp={val:.{precision}f} "
                                                               "+/- {ci:.{precision}f}".format(val=y,
                                                                                               precision=precision,
                                                                                               ci=ci),
                                                       seq))
                            if input_fasta_2 is not None:
                                for ((title, seq), y, ci) in zip(fasta_undef_2, y_pred_undef,
                                                                 pred_uncertainty[y_pred_class_undef]):
                                    out_handle.write(
                                        ">{}\n{}\n".format(title + " | pp={val:.{precision}f} "
                                                                   "+/- {ci:.{precision}f}".format(val=y,
                                                                                                   precision=precision,
                                                                                                   ci=ci),
                                                           seq))
                        else:
                            for ((title, seq), y) in zip(fasta_undef_1, y_pred_undef):
                                out_handle.write(
                                    ">{}\n{}\n".format(
                                        title + " | pp={val:.{precision}f}".format(val=y, precision=precision),
                                        seq))
                            if input_fasta_2 is not None:
                                for ((title, seq), y) in zip(fasta_undef_2, y_pred_undef):
                                    out_handle.write(
                                        ">{}\n{}\n".format(title + " | pp={val:.{precision}f}".format(val=y,
                                                                                                      precision=precision),
                                                           seq))
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
            if not np.isclose(confidence_thresh, threshold):
                with open(output_undef, "w") as out_handle:
                    for (title, seq) in fasta_undef_1:
                        out_handle.write(">{}\n{}\n".format(title, seq))
                    if input_fasta_2 is not None:
                        for (title, seq) in fasta_undef_2:
                            out_handle.write(">{}\n{}\n".format(title, seq))

