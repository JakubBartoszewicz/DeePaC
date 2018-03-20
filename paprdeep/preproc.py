from keras.preprocessing.text import Tokenizer
import numpy as np
from Bio.SeqIO.FastaIO import SimpleFastaParser
from multiprocessing import Pool
from functools import partial
import gzip

# ***ASSUMES EQUAL LENGTH INPUT SEQUENCES - NO PADDING [TODO]***

def tokenize(seq, tokenizer):
    # Tokenize, delete the out-of-vocab token (N) column.
    matrix = tokenizer.texts_to_matrix(seq)[:,1:]
    return matrix

def read_fasta(in_handle):
    # Generators save memory compared to lists. SimpleFastaParser is faster than SeqIO.parse.
    for title, seq in SimpleFastaParser(in_handle):        
        yield seq

def main():
    do_gzip = False
    do_revc = False
    alphabet = "ACGT"  
    neg_path = "SCRATCH_NOBAK/nonpathogenic_test_5_bal.fasta"
    pos_path = "SCRATCH_NOBAK/pathogenic_test_5_bal.fasta"
    out_data_path = "SCRATCH_NOBAK/test_data_5_bal.npy"
    out_labels_path ="SCRATCH_NOBAK/test_labels_5_bal.npy"
    max_cores = 90
    p = Pool(processes = max_cores)
    tokenizer = Tokenizer(char_level = True)
    tokenizer.fit_on_texts(alphabet)
    ### Preproc ###    
    print("Preprocessing negative data...")
    with open(neg_path) as input_handle:
        # Parse fasta and tokenize in parallel. Partial function takes tokenizer as a fixed argument. Tokenize function is applied to the fasta sequence generator.
        x_train = np.asarray(p.map(partial(tokenize, tokenizer = tokenizer), read_fasta(input_handle)))
    # Count negative samples
    n_negative = x_train.shape[0]
    print("Preprocessing positive data...")
    with open(pos_path) as input_handle:
        # Parse fasta, tokenize in parallel & concatenate to negative data
        x_train = np.concatenate((x_train, np.asarray(p.map(partial(tokenize, tokenizer = tokenizer), read_fasta(input_handle)))))
    # Count positive samples
    n_positive = x_train.shape[0] - n_negative
    # Add labels
    y_train = np.concatenate((np.repeat(0,n_negative),np.repeat(1,n_positive)))
    # ** TODO: PADDING (or x_train is an array of arrays, not an array, and it has to be reversed element-wise)
    # All sequences must have the same length. Then x_train is an array and the view below can be created
    # Note: It seems that creating a view instead of reversing element-wise saves a lot of memory (800GB vs 450GB)
    # Add reverse-complements by reversing both dimentions of the matrix - assumes the following order of columns: "ACGT" 
    if do_revc:
        print("Augmenting data...")
        x_train = np.concatenate((x_train, x_train[::,::-1,::-1]))    
        y_train = np.concatenate((y_train, y_train))
    ### Save matrices ###
    print("Saving data...")
    if do_gzip:
        f_data = gzip.GzipFile(out_data_path + ".gz", "w")
        f_labels = gzip.GzipFile(out_labels_path + ".gz", "w")
    else:
        f_data = out_data_path
        f_labels = out_labels_path
    np.save(file = f_data, arr = x_train)
    np.save(file = f_labels, arr = y_train)
    print("Done!")

if __name__ == "__main__":
    main()
