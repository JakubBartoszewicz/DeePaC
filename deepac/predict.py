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
from scipy.special import softmax, expit
from tensorflow.keras import backend as K


def predict_fasta(model, input_fasta, output, token_cores=8, datatype='int32', rc=False, replicates=1, batch_size=512,
                  get_logits=False):
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
    y_pred, y_std = predict_array(model, x_data, output, rc, replicates, batch_size, get_logits)
    end = time.time()
    print("Preprocessing & predictions for {} reads done in {} s".format(y_pred.shape[0], end - start))
    return y_pred, y_std


def predict_npy(model, input_npy, output, rc=False, replicates=1, batch_size=512, get_logits=False):
    """Predict pathogenic potentials from a preprocessed numpy array."""
    x_data = np.load(input_npy, mmap_mode='r')
    return predict_array(model, x_data, output, rc, replicates, batch_size, get_logits)


def predict_array(model, x_data, output, rc=False, replicates=1, batch_size=512, get_logits=False):
    """Predict pathogenic potentials from a preprocessed numpy array."""
    if rc:
        x_data = x_data[::, ::-1, ::-1]
    n_outputs = model.output.shape[1]
    # Predict
    print("Predicting...")
    start = time.time()
    iterate = None
    if get_logits:
        layer_output = model.get_layer(index=-2).get_output_at(0)
        if tf.executing_eagerly():
            model = tf.keras.Model(model.inputs, layer_output)
        else:
            model_input = model.layers[0].input
            iterate = K.function([model_input, K.learning_phase()], [layer_output])

    def __get_preds(in_data):
        if get_logits and not tf.executing_eagerly():
            raise ValueError("Please turn eager mode on to get the logits efficiently.")
        else:
            out_raw = model.predict(in_data, batch_size=batch_size)
        return out_raw

    if replicates > 1:
        if n_outputs == 1:
            y_preds = np.zeros((x_data.shape[0], replicates))
        else:
            y_preds = np.zeros((x_data.shape[0], n_outputs, replicates))
        for i in tqdm(range(replicates)):
            y_pred_raw = __get_preds(x_data)
            if n_outputs == 1:
                y_preds[:, i] = y_pred_raw.squeeze()
            else:
                y_preds[:, :, i] = y_pred_raw
        y_pred = y_preds.mean(axis=-1)
        y_std = y_preds.std(axis=-1)
    else:
        y_pred = __get_preds(x_data)
        y_std = None

    end = time.time()
    print("Predictions for {} reads done in {} s".format(y_pred.shape[0], end - start))
    np.save(file=output, arr=y_pred)
    if replicates > 1:
        out_std = "{}-std.npy".format(os.path.splitext(output)[0])
        np.save(file=out_std, arr=y_std)
    return y_pred, y_std


def filter_fasta(input_fasta, predictions, output, threshold=0.5, print_potentials=False, precision=3,
                 output_neg=None, confidence_thresh=None, output_undef=None, pred_uncertainty=None, n_classes=2,
                 positive_classes=(1,)):
    """Filter reads in a fasta file by pathogenic potential."""
    filter_paired_fasta(input_fasta, predictions, output, input_fasta_2=None, predictions_2=None,
                        output_neg=output_neg, threshold=threshold, print_potentials=print_potentials,
                        precision=precision, confidence_thresh=confidence_thresh, output_undef=output_undef,
                        pred_uncertainty=pred_uncertainty, n_classes=n_classes, positive_classes=positive_classes)


def ensemble(predictions_list, outpath_npy):
    """Average predictions of multiple classifiers."""
    ys = []
    for p in predictions_list:
        ys.append(np.load(p, mmap_mode='r'))
    ys = np.array(ys)
    y_pred = np.average(ys, 0)
    np.save(outpath_npy, y_pred)


def predict_multiread(array, threshold=0.5, confidence_threshold=None, n_classes=2, add_activ=False):
    """Predict from multiple reads."""
    multiclass = n_classes > 2

    if confidence_threshold is None or \
            (np.isclose(confidence_threshold, threshold) and not multiclass):
        pred = np.mean(array, axis=0)
    else:
        if multiclass:
            above_thresh = np.max(array, axis=-1) > confidence_threshold
            preds = array[above_thresh]
        else:
            interval = np.abs(confidence_threshold - threshold)
            y_pred_class_pos = array > (threshold + interval)
            y_pred_class_neg = array < (threshold - interval)
            preds = np.concatenate((array[y_pred_class_pos], array[y_pred_class_neg]))
        if preds.size > 0:
            pred = np.mean(preds, axis=0)
        else:
            pred = np.nan

    if add_activ:
        if multiclass:
            pred = softmax(pred)  # apply softmax
        else:
            pred = expit(pred)  # apply sigmoid

    return pred


def get_fasta_preds(input_fasta_1, predictions_1, input_fasta_2=None, predictions_2=None):
    """Load predictions for reads or read pairs."""
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
    """Format a filtered sequence."""
    record = ">{}".format(title)
    if print_potentials and precision > 0:
        if isinstance(y, (list, np.ndarray)):
            y_str = "; ".join(["{y_val:.{precision}f}".format(y_val=y_val, precision=precision) for y_val in y])
        else:
            y_str = "{y_val:.{precision}f}".format(y_val=y, precision=precision)
        record = record + " | pp={y_str}".format(y_str=y_str)
    if do_uncert is not None and precision > 0:
        if isinstance(ci, (list, np.ndarray)):
            ci_str = "; ".join(["{ci_val:.{precision}f}".format(ci_val=ci_val, precision=precision) for ci_val in ci])
        else:
            ci_str = "{ci_val:.{precision}f}".format(ci_val=ci, precision=precision)
        record = record + " +/- {ci_str}".format(ci_str=ci_str)
    record = record + "\n{}\n".format(seq)
    return record


def write_filtered(output, fasta_1, fasta_2, y_pred_pos, pred_uncertainty, precision,
                   print_potentials=False, do_uncert=False, mode="a"):
    """Save filtered sequences."""
    with open(output, mode) as out_handle:
        for ((title, seq), y, ci) in zip(fasta_1, y_pred_pos, pred_uncertainty):
            out_handle.write(format_record(title, seq, y, ci, precision, print_potentials, do_uncert))
        if fasta_2 is not None and len(fasta_2) > 0:
            for ((title, seq), y, ci) in zip(fasta_2, y_pred_pos, pred_uncertainty):
                out_handle.write(format_record(title, seq, y, ci, precision, print_potentials, do_uncert))


def filter_paired_fasta(input_fasta_1, predictions_1, output_pos, input_fasta_2=None, predictions_2=None,
                        output_neg=None, threshold=0.5, print_potentials=False, precision=3,
                        confidence_thresh=None, output_undef=None, pred_uncertainty=None, n_classes=2,
                        positive_classes=(1,)):
    """Filter reads in paired fasta files by pathogenic potential."""
    fasta_data_1, fasta_data_2, y_pred = get_fasta_preds(input_fasta_1, predictions_1, input_fasta_2, predictions_2)

    y_pred_classes = {}

    y_pred_class_undef = []
    do_undef = False
    if n_classes == 2:
        if confidence_thresh is None or np.isclose(confidence_thresh, threshold):
            y_pred_classes[0] = y_pred <= threshold
            y_pred_classes[1] = y_pred > threshold
        else:
            do_undef = True
            interval = np.abs(confidence_thresh - threshold)
            y_pred_classes[0] = y_pred < (threshold - interval)
            y_pred_classes[1] = y_pred > (threshold + interval)
            y_pred_class_undef = np.logical_not(np.any([y_pred_classes[0], y_pred_classes[1]], axis=0))
    else:
        y_pred_dense = np.argmax(y_pred, axis=-1)
        if confidence_thresh is None:
            for i in range(n_classes):
                y_pred_classes[i] = y_pred_dense == i
        else:
            do_undef = True
            above_thresh = np.max(y_pred, axis=-1) > confidence_thresh
            y_pred_class_undef = np.logical_not(above_thresh)
            for i in range(n_classes):
                y_pred_classes[i] = np.all([y_pred_dense == i, above_thresh], axis=0)

    fasta_classes_1 = {}
    fasta_classes_2 = {}

    for i in range(n_classes):
        fasta_classes_1[i] = list(itertools.compress(fasta_data_1, y_pred_classes[i]))
    fasta_undef_1 = list(itertools.compress(fasta_data_1, y_pred_class_undef))
    if fasta_data_2 is not None:
        for i in range(n_classes):
            fasta_classes_2[i] = list(itertools.compress(fasta_data_2, y_pred_classes[i]))
        fasta_undef_2 = list(itertools.compress(fasta_data_2, y_pred_class_undef))
    else:
        for i in range(n_classes):
            fasta_classes_2[i] = []
        fasta_undef_2 = []
    if pred_uncertainty is not None:
        do_uncert = True
        pred_uncertainty = np.load(pred_uncertainty)
    else:
        do_uncert = False
        pred_uncertainty = np.zeros(y_pred.shape)

    if os.path.exists(output_pos):
        os.remove(output_pos)
    for i in positive_classes:
        write_filtered(output_pos, fasta_classes_1[i], fasta_classes_2[i],
                       y_pred[y_pred_classes[i]], pred_uncertainty[y_pred_classes[i]],
                       precision, print_potentials, do_uncert)
    if output_neg is not None:
        if os.path.exists(output_neg):
            os.remove(output_neg)
        negative_classes = [c for c in range(n_classes) if c not in positive_classes]
        for i in negative_classes:
            write_filtered(output_neg, fasta_classes_1[i], fasta_classes_2[i],
                           y_pred[y_pred_classes[i]], pred_uncertainty[y_pred_classes[i]],
                           precision, print_potentials, do_uncert)
    if do_undef and output_undef is not None:
        write_filtered(output_undef, fasta_undef_1, fasta_undef_2, y_pred[y_pred_class_undef],
                       pred_uncertainty[y_pred_class_undef],
                       precision, print_potentials, do_uncert, mode="w")
