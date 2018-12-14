"""@package eval
Evaluate a NN trained on Illumina reads.

Requires a config file describing the data directory, and dataset and run name, classification threshold and the epoch range.

usage: eval.py [-h] config_file

positional arguments:
  config_file

optional arguments:
  -h, --help   show this help message and exit
  
"""
import sys
import argparse
import configparser
from keras.models import Sequential
from keras.models import load_model
from keras.backend import tensorflow_backend as backend
import sklearn.metrics as mtr
import numpy as np
import csv
from rc_layers import RevCompConv1D, RevCompConv1DBatchNorm, DenseAfterRevcompWeightedSum, DenseAfterRevcompConv1D

def main(argv):
    """Parse the config file and evaluate the NN on Illumina reads."""
    parser = argparse.ArgumentParser(description = "Evaluate a NN trained on Illumina reads.")
    parser.add_argument("config_file")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_file)
    evaluate(config)
   
    
def evaluate(config):
    """Evaluate the NN on Illumina reads using the supplied configuration."""
    # Clear session needed or TypeError can happen in some cases
    backend.clear_session()

    
    # Set the data directory
    dir = config['Data']['DataDir']
    # Set the evaluation dataset name
    set = config['Data']['DataSet']
    # Set the run name
    runname = config['Data']['RunName']
    model_name = "nn-{}".format(runname)
    name_prefix = "nn-{}".format(runname) + "_" + set
    # Set the classification threshold
    thresh = config['Data'].getfloat('Threshold')
    
    # Set the first and last epoch to evaluate
    epoch_start = config['Epochs'].getint('EpochStart')
    epoch_end = config['Epochs'].getint('EpochEnd')

    # Read data to memory
    x_test = np.load("{}/{}_data.npy".format(dir,set))
    y_test = np.load("{}/{}_labels.npy".format(dir,set))
    
    # Write CSV header
    with open("{}-metrics.csv".format(name_prefix), 'a',  newline="") as csv_file:
        file_writer = csv.writer(csv_file)
        file_writer.writerow(("epoch","log_loss","acc","auroc","mcc","f1","precision","recall"))

    # Evaluate for each saved model in epoch range
    for n_epoch in range(epoch_start,epoch_end):
        model = load_model("{p}-e{ne:03d}.h5".format(p=model_name, ne=n_epoch), custom_objects={'RevCompConv1D': RevCompConv1D, 'RevCompConv1DBatchNorm': RevCompConv1DBatchNorm, 'DenseAfterRevcompWeightedSum': DenseAfterRevcompWeightedSum, 'DenseAfterRevcompConv1D': DenseAfterRevcompConv1D})
        
        # Predict class probabilities
        y_pred =  np.ndarray.flatten(model.predict_proba(x_test))
        # Assign classes using the chosen threshold
        y_pred_class = (y_pred > thresh).astype('int32')
        # Backup predicted probabilities for future analyses 
        np.save(file = "{p}-e{ne:03d}-predictions.npy".format(p=name_prefix, ne=n_epoch), arr = y_pred)

        # Calculate performance measures
        log_loss = mtr.log_loss(y_test, y_pred, eps=1e-07)
        acc = mtr.accuracy_score(y_test, y_pred_class)
        auroc = mtr.roc_auc_score(y_test, y_pred)
        mcc = mtr.matthews_corrcoef(y_test, y_pred_class)
        f1 = mtr.f1_score(y_test, y_pred_class)
        precision = mtr.precision_score(y_test, y_pred_class)
        recall = mtr.recall_score(y_test, y_pred_class)

        # Save the results
        with open("{}-metrics.csv".format(name_prefix), 'a',  newline="") as csv_file:
            file_writer = csv.writer(csv_file)
            file_writer.writerow((n_epoch,log_loss,acc,auroc,mcc,f1,precision,recall))

        
if __name__ == "__main__":
    main(sys.argv)
