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
            y_pred_raw = model.predict(x_data, batch_size=batch_size)
            y_preds.append(y_pred_raw)
        y_preds = np.asarray(y_preds)
        y_pred = y_preds.mean(axis=0)
        y_std = y_preds.std(axis=0)
    else:
        y_pred = model.predict(x_data, batch_size=batch_size)
        y_std = None

    end = time.time()
    print("Predictions for {} reads done in {} s".format(y_pred.shape[0], end - start))
    np.save(file=output, arr=y_pred)
    if replicates > 1:
        out_std = "{}-std.npy".format(os.path.splitext(output)[0])
        np.save(file=out_std, arr=y_std)
    return y_pred, y_std


def filter_fasta(input_fasta, predictions, output, threshold=0.5, print_potentials=False, precision=3,
                 output_neg=None, confidence_thresh=0.5, output_undef=None, pred_uncertainty=None):
    """Filter reads in a fasta file by pathogenic potential."""
    filter_paired_fasta(input_fasta, predictions, output, input_fasta_2=None, predictions_2=None,
                        output_neg=output_neg, threshold=threshold, print_potentials=print_potentials,
                        precision=precision, confidence_thresh=confidence_thresh, output_undef=output_undef,
                        pred_uncertainty=pred_uncertainty)


def ensemble(predictions_list, outpath_npy):
    ys = []
    for p in predictions_list:
        ys.append(np.load(p, mmap_mode='r'))
    ys = np.array(ys)
    y_pred = np.average(ys, 0)
    np.save(outpath_npy, y_pred)


def predict_multiread(array, threshold=0.5, confidence_threshold=0.5):
    if np.isclose(confidence_threshold, threshold):
        pred = np.mean(array)
    else:
        interval = np.abs(confidence_threshold - threshold)
        y_pred_class_pos = array > (threshold + interval)
        y_pred_class_neg = array < (threshold - interval)
        preds = np.concatenate((array[y_pred_class_pos], array[y_pred_class_neg]))
        if preds.size > 0:
            pred = np.mean(preds)
        else:
            pred = np.nan
    return pred


def get_fasta_preds(input_fasta_1, predictions_1, input_fasta_2=None, predictions_2=None):
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
        fasta_data_2 = None
    return fasta_data_1, fasta_data_2, y_pred


def format_record(title, seq, y, ci, precision, print_potentials=False, do_uncert=False):
    record = ">{}\n".format(title)
    if print_potentials and precision > 0:
        record = record + " | pp={val:.{precision}f}".format(val=y, precision=precision)
    if do_uncert is not None and precision > 0:
        record = record + " +/- {ci:.{precision}f}".format(ci=ci, precision=precision)
    record = record + "{}\n".format(seq)
    return record


def write_filtered(output, fasta_1, fasta_2, y_pred_pos, pred_uncertainty, precision,
                   print_potentials=False, do_uncert=False):
    with open(output, "w") as out_handle:
        for ((title, seq), y, ci) in zip(fasta_1, y_pred_pos, pred_uncertainty):
            out_handle.write(format_record(title, seq, y, ci, precision, print_potentials, do_uncert))
        if fasta_2 is not None and len(fasta_2) > 0:
            for ((title, seq), y, ci) in zip(fasta_2, y_pred_pos, pred_uncertainty):
                out_handle.write(format_record(title, seq, y, ci, precision, print_potentials, do_uncert))


def filter_paired_fasta(input_fasta_1, predictions_1, output_pos, input_fasta_2=None, predictions_2=None,
                        output_neg=None, threshold=0.5, print_potentials=False, precision=3,
                        confidence_thresh=0.5, output_undef=None, pred_uncertainty=None, no_classes=2):
    """Filter reads in paired fasta files by pathogenic potential."""
    fasta_data_1, fasta_data_2, y_pred = get_fasta_preds(input_fasta_1, predictions_1, input_fasta_2, predictions_2)

    y_pred_classes = {}

    if no_classes == 2:
        if np.isclose(confidence_thresh, threshold):
            y_pred_classes[0] = y_pred <= threshold
            y_pred_classes[1] = y_pred > threshold
            y_pred_class_undef = []
        else:
            interval = np.abs(confidence_thresh - threshold)
            y_pred_classes[0] = y_pred < (threshold - interval)
            y_pred_classes[1] = y_pred > (threshold + interval)
            y_pred_class_undef = np.logical_not(np.any([y_pred_classes[0], y_pred_classes[1]], axis=0))
    else:
        raise NotImplementedError

    fasta_classes_1 = {}
    fasta_classes_2 = {}

    for i in range(no_classes):
        fasta_classes_1[i] = list(itertools.compress(fasta_data_1, y_pred_classes[i]))
    fasta_undef_1 = list(itertools.compress(fasta_data_1, y_pred_class_undef))
    if fasta_data_2 is not None:
        for i in range(no_classes):
            fasta_classes_2[i] = list(itertools.compress(fasta_data_2, y_pred_classes[i]))
        fasta_undef_2 = list(itertools.compress(fasta_data_2, y_pred_class_undef))
    else:
        for i in range(no_classes):
            fasta_classes_2[i] = []
        fasta_undef_2 = []
    if pred_uncertainty is not None:
        do_uncert = True
        pred_uncertainty = np.load(pred_uncertainty)
    else:
        do_uncert = False
        pred_uncertainty = np.zeros(y_pred.shape)

    if no_classes > 2 or output_neg is not None:
        write_filtered(output_pos, fasta_classes_1[0], fasta_classes_2[0],
                       y_pred[y_pred_classes[0]], pred_uncertainty[y_pred_classes[0]],
                       precision, print_potentials, do_uncert)
    for i in range(1, no_classes):
        write_filtered(output_pos, fasta_classes_1[i], fasta_classes_2[i],
                       y_pred[y_pred_classes[i]], pred_uncertainty[y_pred_classes[i]],
                       precision, print_potentials, do_uncert)
    if not np.isclose(confidence_thresh, threshold) and output_neg is not None:
        write_filtered(output_undef, fasta_undef_1, fasta_undef_2, y_pred[y_pred_class_undef],
                       pred_uncertainty[y_pred_class_undef],
                       precision, print_potentials, do_uncert)
