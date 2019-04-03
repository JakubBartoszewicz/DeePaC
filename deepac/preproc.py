"""@package deepac.preproc
Convert fasta files to numpy arrays for training.

Uses distributed orthographic representation, i.e. every read is coded so that every nucleotide is a one-hot encoded
 vector. Assumes equal length of all the input sequences - no padding!

Requires a config file describing the available devices, input filepaths (two fasta files containing negative and
 positive reads respectively), output filepath (for data and labels) and additional options.

"""
from keras.preprocessing.text import Tokenizer
import numpy as np
from Bio.SeqIO.FastaIO import SimpleFastaParser
from multiprocessing import Pool
from functools import partial
import gzip


def tokenize(seq, tokenizer, datatype='int8', read_length=250):
    """Tokenize and delete the out-of-vocab token (N) column."""
    # Cast to datatype instead of default float64 to save memory
    matrix = tokenizer.texts_to_matrix(seq).astype(datatype)[:, 1:]
    if matrix.shape[0] < read_length:
        # Pad with zeros
        matrix = np.concatenate((matrix, np.zeros((read_length - len(seq), 4))))
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
    p = Pool(processes=max_cores)
    
    # Set input and output paths
    neg_path = config['InputPaths']['Fasta_Class_0']
    pos_path = config['InputPaths']['Fasta_Class_1']
    out_data_path = config['OutputPaths']['OutData']
    out_labels_path = config['OutputPaths']['OutLabels']
    
    # Set additional options: gzip compression, RC augmentation, data type
    do_gzip = config['Options'].getboolean('Do_gzip')
    do_revc = config['Options'].getboolean('Do_revc')
    datatype = config['Options']['DataType']
    read_length = config['Options'].getint('ReadLength')
    
    # Set alphabet and prepare the tokenizer
    alphabet = "ACGT"  
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(alphabet)
    
    # Preprocess #
    if neg_path != "none":
        print("Preprocessing negative data...")
        with open(neg_path) as input_handle:
            # Parse fasta and tokenize in parallel. Partial function takes tokenizer as a fixed argument.
            # Tokenize function is applied to the fasta sequence generator.
            x_train_neg = np.asarray(p.map(partial(tokenize, tokenizer=tokenizer, datatype=datatype,
                                                   read_length=read_length), read_fasta(input_handle)))
        # Count negative samples
        n_negative = x_train_neg.shape[0]
    else:
        x_train_neg = np.zeros((0, read_length, 4))
        n_negative = 0

    if pos_path != "none":
        print("Preprocessing positive data...")
        with open(pos_path) as input_handle:
            # Parse fasta, tokenize in parallel & concatenate to negative data
            x_train_pos = np.asarray(p.map(partial(tokenize, tokenizer=tokenizer, datatype=datatype,
                                                   read_length=read_length), read_fasta(input_handle)))
        # Count positive samples
        n_positive = x_train_pos.shape[0]
    else:
        x_train_pos = np.zeros((0, read_length, 4))
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
        
    # Save matrices #
    print("Saving data...")
    # Compress output files
    if do_gzip:
        f_data = gzip.GzipFile(out_data_path + ".gz", "w")
        f_labels = gzip.GzipFile(out_labels_path + ".gz", "w")
    else:
        f_data = out_data_path
        f_labels = out_labels_path
    # Save output
    np.save(file=f_data, arr=x_train)
    np.save(file=f_labels, arr=y_train)
    print("Done!")
