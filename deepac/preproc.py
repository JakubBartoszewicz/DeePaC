"""@package deepac.preproc
Convert fasta files to numpy arrays for training.

Uses distributed orthographic representation, i.e. every read is coded so that every nucleotide is a one-hot encoded
 vector. Assumes equal length of all the input sequences - no padding!

Requires a config file describing the available devices, input filepaths (two fasta files containing negative and
 positive reads respectively), output filepath (for data and labels) and additional options.

"""
import tensorflow as tf
import numpy as np
from Bio.SeqIO.FastaIO import SimpleFastaParser
from multiprocessing import Pool
from functools import partial
import gzip
import os
import math


def tokenize(seq, tokenizer, datatype='int8', read_length=250):
    """Tokenize and delete the out-of-vocab token (N) column."""
    # Cast to datatype instead of default float64 to save memory
    matrix = tokenizer.texts_to_matrix(seq).astype(datatype)[:, 1:]
    if matrix.shape[0] < read_length:
        # Pad with zeros
        matrix = np.concatenate((matrix, np.zeros((read_length - len(seq), 4), dtype=datatype)))
    if matrix.shape[0] > read_length:
        # Trim
        matrix = matrix[:read_length, :]
    return matrix


def read_fasta(in_handle):
    """Read fasta file with a fast, memory-efficient generator."""
    # Generators save memory compared to lists. SimpleFastaParser is faster than SeqIO.parse.
    for title, seq in SimpleFastaParser(in_handle):
        yield seq


def preproc(config):
    """Preprocess the CNN on Illumina reads using the supplied configuration."""
    # Set the number of cores to use
    max_cores = config['Devices'].getint('N_CPUs')

    # Set input and output paths
    neg_path = config['InputPaths']['Fasta_Class_0']
    pos_path = config['InputPaths']['Fasta_Class_1']
    out_data_path = config['OutputPaths']['OutData']
    out_labels_path = config['OutputPaths']['OutLabels']

    # Set additional options: shuffle, gzip compression, RC augmentation, data type
    do_shuffle = config['Options'].getboolean('Do_shuffle')
    if do_shuffle:
        seed = config['Options'].getint('ShuffleSeed')
        np.random.seed(seed)
    do_gzip = config['Options'].getboolean('Do_gzip')
    do_revc = config['Options'].getboolean('Do_revc')
    datatype = config['Options']['DataType']
    read_length = config['Options'].getint('ReadLength')
    use_tfdata = config['Options'].getboolean('Use_TFData')
    n_files = config['Options'].getint('N_Files')

    # Set alphabet and prepare the tokenizer
    alphabet = "ACGT"
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(alphabet)
    # Preprocess
    if neg_path != "none":
        print("Preprocessing negative data...")
        with open(neg_path) as input_handle:
            # Parse fasta and tokenize in parallel. Partial function takes tokenizer as a fixed argument.
            # Tokenize function is applied to the fasta sequence generator.
            if max_cores > 1:
                with Pool(processes=max_cores) as p:
                    x_train_neg = np.asarray(p.map(partial(tokenize, tokenizer=tokenizer, datatype=datatype,
                                                           read_length=read_length), read_fasta(input_handle)),
                                             dtype=datatype)
            else:
                x_train_neg = np.asarray(list(map(partial(tokenize, tokenizer=tokenizer, datatype=datatype,
                                                          read_length=read_length), read_fasta(input_handle))),
                                         dtype=datatype)
        # Count negative samples
        n_negative = x_train_neg.shape[0]
    else:
        x_train_neg = np.zeros((0, read_length, 4),dtype=np.datatype)
        n_negative = 0

    if pos_path != "none":
        print("Preprocessing positive data...")
        with open(pos_path) as input_handle:
            # Parse fasta, tokenize in parallel & concatenate to negative data
            if max_cores > 1:
                with Pool(processes=max_cores) as p:
                    x_train_pos = np.asarray(p.map(partial(tokenize, tokenizer=tokenizer, datatype=datatype,
                                                           read_length=read_length), read_fasta(input_handle)),
                                             dtype=datatype)
            else:
                x_train_pos = np.asarray(list(map(partial(tokenize, tokenizer=tokenizer, datatype=datatype,
                                                          read_length=read_length), read_fasta(input_handle))),
                                         dtype=datatype)
        # Count positive samples
        n_positive = x_train_pos.shape[0]
    else:
        x_train_pos = np.zeros((0, read_length, 4),dtype=np.datatype)
        n_positive = 0
    # Concatenate
    x_train = np.concatenate((x_train_neg, x_train_pos))
    # Add labels
    y_train = np.concatenate((np.repeat(0, n_negative).astype(datatype), np.repeat(1, n_positive).astype(datatype)))
    # All sequences must have the same length. Then x_train is an array and the view below can be created
    # Note: creating a view instead of reversing element-wise saves a lot of memory

    # RC augmentation: Add reverse-complements by reversing both dimensions of the matrix
    # assumes the following order of columns: "ACGT"
    if do_revc:
        print("Augmenting data...")
        x_train = np.concatenate((x_train, x_train[::, ::-1, ::-1]))
        y_train = np.concatenate((y_train, y_train))

    if do_shuffle:
        indices = np.arange(len(y_train))
        np.random.shuffle(indices)
        x_train = x_train[indices, ::, ::]
        y_train = y_train[indices]

    # Save matrices #
    print("Saving data...")

    # Save output
    if not use_tfdata:
        # Compress output files
        if do_gzip:
            f_data = gzip.GzipFile(out_data_path + ".gz", "w")
            f_labels = gzip.GzipFile(out_labels_path + ".gz", "w")
        else:
            f_data = out_data_path
            f_labels = out_labels_path
        np.save(file=f_data, arr=x_train)
        np.save(file=f_labels, arr=y_train)
    else:
        out_dir = os.path.splitext(out_data_path)[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        n_all = n_negative + n_positive
        slice_size = math.ceil(n_all/n_files)
        for i in range(n_files):
            start = i * slice_size
            end = min((i+1) * slice_size, n_all)
            features_dataset = tf.data.Dataset.from_tensor_slices((x_train[start:end], y_train[start:end]))

            serialized_features_dataset = features_dataset.map(tf_serialize_example)

            filename = os.path.join(out_dir, os.path.splitext(os.path.basename(out_dir))[0]
                                    + '_{}-{}.tfrec'.format(start, end - 1))
            writer = tf.data.experimental.TFRecordWriter(filename)
            if tf.executing_eagerly():
                writer.write(serialized_features_dataset)
            else:
                with tf.compat.v1.Session() as sess:
                    sess.run(writer.write(serialized_features_dataset))

    print("Done!")


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(x_seq, y_label):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'x_seq': _bytes_feature(tf.io.serialize_tensor(x_seq)),
        'y_label': _int64_feature(y_label)
    }
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(x_seq, y_label):
    tf_string = tf.py_function(
        serialize_example,
        (x_seq, y_label),  # pass these args to the above function.
        tf.string)      # the return type is `tf.string`.
    return tf.reshape(tf_string, ())  # The result is a scalar
