import sys
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Filtering Illumina reads (up to 250 bp) with pathogenic potential predictions.")
    parser.add_argument("-t", "--threshold", help="Threshold [default=0.5]", default=0.5, type=float)    
    parser.add_argument("-z", "--zthreshold", help="Threshold two [default=0.70]", default=0.70, type=float)  
    parser.add_argument("-u", "--uthreshold", help="Threshold three [default=0.90]", default=0.90, type=float)
    parser.add_argument("-p", "--potentials", help="Print pathogenic potential values in .fasta headers", default=False, action="store_true")
    parser.add_argument("-d", "--digits", help="Print pathogenic potential values rounded to this many digits [default=3]", default=3, type=int)
    parser.add_argument("-o", "--output", help="Output file path [.fasta]")
    parser.add_argument("-k", "--koutput", help="Output two file path [.fasta]")
    parser.add_argument("-j", "--joutput", help="Output three file path [.fasta]")
    parser.add_argument("input",  help="Input file path [.fasta]")
    parser.add_argument("model", help="Trained network file path [.h5]")
    
    args = parser.parse_args()

    return args

def main(argv):
    args = parse_args()
    if args.output is None:        
        args.output = os.path.splitext(args.input)[0] + "_filtered_{}.fasta".format(args.threshold)
    if args.koutput is None:        
        args.koutput = os.path.splitext(args.input)[0] + "_filtered_{}.fasta".format(args.zthreshold)
    if args.joutput is None:        
        args.joutput = os.path.splitext(args.input)[0] + "_filtered_{}.fasta".format(args.uthreshold)
        
    from keras.models import Sequential
    from keras.models import load_model
    from keras.preprocessing.text import Tokenizer
    import numpy as np
    from Bio.SeqIO.FastaIO import SimpleFastaParser
    import itertools
    import csv
    
    alphabet = "ACGT"
    seq_dim = len(alphabet) + 1

    ### Load ###
    print("Loading...")
    model = load_model(args.model)
    with open(args.input) as in_handle:
        data = [(title, seq) for (title, seq) in SimpleFastaParser(in_handle)]

    ### Preproc ###
    tokenizer = Tokenizer(char_level = True)
    tokenizer.fit_on_texts(alphabet)
    print("Preprocessing data...")
    x_data = np.array([np.concatenate((tokenizer.texts_to_matrix(x[1]), np.zeros((250 - len(x[1]),5)))) for x in data])

    ### Predict ###
    print("Predicting...")
    y_pred =  np.ndarray.flatten(model.predict_proba(x_data))
    
    ### Filter ###  
    y_pred_class = (y_pred > args.threshold).astype('int32')
    y_pred_filtered = [y for y in y_pred if y > args.threshold]
    data_filtered = list(itertools.compress(data, y_pred_class))  
    if args.potentials and args.digits > 0:
        with open(args.output, "w") as out_handle:
            for ((title, seq), y) in zip(data_filtered , y_pred_filtered):
                out_handle.write(">{}\n{}\n".format(title + " | pp={val:.{digits}f}".format(val=y, digits=args.digits), seq))
    else:
        with open(args.output, "w") as out_handle:
                for (title, seq) in data_filtered :
                    out_handle.write(">{}\n{}\n".format(title, seq))
                    
    ### Filter ###  
    y_pred_class = (y_pred > args.zthreshold).astype('int32')
    y_pred_filtered = [y for y in y_pred if y > args.zthreshold]
    data_filtered = list(itertools.compress(data, y_pred_class))  
    if args.potentials and args.digits > 0:
        with open(args.koutput, "w") as out_handle:
            for ((title, seq), y) in zip(data_filtered , y_pred_filtered):
                out_handle.write(">{}\n{}\n".format(title + " | pp={val:.{digits}f}".format(val=y, digits=args.digits), seq))
    else:
        with open(args.koutput, "w") as out_handle:
                for (title, seq) in data_filtered :
                    out_handle.write(">{}\n{}\n".format(title, seq))
                    
    ### Filter ###  
    y_pred_class = (y_pred > args.uthreshold).astype('int32')
    y_pred_filtered = [y for y in y_pred if y > args.uthreshold]
    data_filtered = list(itertools.compress(data, y_pred_class))  
    if args.potentials and args.digits > 0:
        with open(args.joutput, "w") as out_handle:
            for ((title, seq), y) in zip(data_filtered , y_pred_filtered):
                out_handle.write(">{}\n{}\n".format(title + " | pp={val:.{digits}f}".format(val=y, digits=args.digits), seq))
    else:
        with open(args.joutput, "w") as out_handle:
                for (title, seq) in data_filtered :
                    out_handle.write(">{}\n{}\n".format(title, seq))
    

if __name__=="__main__":
    main(sys.argv)  

