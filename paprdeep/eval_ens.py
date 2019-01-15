"""@package eval
Evaluate a NN trained on Illumina reads.

Requires a config file describing the data directory, dataset and run name,
classification threshold and the epoch range.

usage: eval.py [-h] config_file

positional arguments:
  config_file

optional arguments:
  -h, --help   show this help message and exit
  
"""
import argparse
import configparser
from keras.models import load_model
from keras.backend import tensorflow_backend as backend
import sklearn.metrics as mtr
import numpy as np
import csv
import matplotlib.pyplot as plt

class EvalConfig():
    """
    Evaluation configuration class

    """

    def __init__(self, config):
        # Set the data directory
        self.dir_path = config['Data']['DataDir']
        # Set the evaluation dataset name
        self.dataset_path = config['Data']['DataSet']
        # Set the paired dataset name
        self.pairedset_path = config['Data']['PairedSet']
        if self.pairedset_path == "none":
            self.pairedset_path = None
            self.combinedset_path = self.dataset_path
        else:
            self.combinedset_path = self.dataset_path + "_" + self.pairedset_path

        # Set the run name        
        self.ensname = config['Data']['EnsembleName']
        self.runnames = [r for r in config['Data']['RunNames'].split(',')]
        self.run_prefixes = ["nn-{}".format(r) for r in self.runnames]        
        self.name_prefix = "nn-{}".format(self.ensname)
        # Set the classification threshold
        self.thresh = config['Data'].getfloat('Threshold')

        # Set the first and last epoch to evaluate
        self.epoch = [int(e) for e in config['Epochs']['Epoch'].split(',')]

        self.do_plots = config['Options'].getboolean('Do_plots')


def main():
    """Parse the config file and evaluate the NN on Illumina reads."""
    parser = argparse.ArgumentParser(description="Evaluate a NN trained on Illumina reads.")
    parser.add_argument("config_file")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_file)
    evaluate(config)



def evaluate(config):
    """Evaluate the NN on Illumina reads using the supplied configuration."""
    # Clear session needed or TypeError can happen in some cases
    backend.clear_session()
    
    evalconfig = EvalConfig(config)

    # Read data to memory
    print("Loading {}_data.npy...".format(evalconfig.dataset_path))
    x_test = np.load("{}/{}_data.npy".format(evalconfig.dir_path, evalconfig.dataset_path))
    y_test = np.load("{}/{}_labels.npy".format(evalconfig.dir_path, evalconfig.dataset_path))
    
    # Write CSV header
    with open("{}-metrics.csv".format(evalconfig.name_prefix), 'a',  newline="") as csv_file:
        file_writer = csv.writer(csv_file)
        file_writer.writerow(("set", "log_loss", "acc", "auroc", "aupr", "precision", "recall", "mcc", "f1"))

    # Evaluate for each saved model in epoch range
    print("Predicting labels for {}_data.npy...".format(evalconfig.dataset_path))
    y_pred_1 = predict(evalconfig, x_test)
    get_performance(evalconfig, y_test, y_pred_1, dataset_name=evalconfig.dataset_path)

    if evalconfig.pairedset_path is not None:
        print("Loading {}_data.npy...".format(evalconfig.pairedset_path))
        x_test = np.load("{}/{}_data.npy".format(evalconfig.dir_path, evalconfig.pairedset_path))
        y_test = np.load("{}/{}_labels.npy".format(evalconfig.dir_path, evalconfig.pairedset_path))

        print("Predicting labels for {}_data.npy...".format(evalconfig.pairedset_path))
        y_pred_2 = predict(evalconfig, x_test, paired=True)
        get_performance(evalconfig, y_test, y_pred_2,  dataset_name=evalconfig.pairedset_path)

        y_pred_combined = np.mean([y_pred_1, y_pred_2], axis=0)
        get_performance(evalconfig, y_test, y_pred_combined, dataset_name=evalconfig.combinedset_path)


def predict(evalconfig, x_test, paired=False):
    """Predict the pathogenic potentials of Illumina reads using the supplied configuration."""
    if paired:
        dataset_path = evalconfig.pairedset_path
    else:
        dataset_path = evalconfig.dataset_path
        
    y_preds = []
    for i in range(0, len(evalconfig.run_prefixes)):        
        model = load_model("{p}-e{ne:03d}.h5".format(p=evalconfig.run_prefixes[i], ne=evalconfig.epoch[i]))
        # Predict class probabilities
        y_preds.append(np.ndarray.flatten(model.predict(x_test)))
    y_pred = np.mean(y_preds, axis=0)
    # Backup predicted probabilities for future analyses
    np.save(file="{p}-predictions-{s}.npy".format(p=evalconfig.name_prefix, s=dataset_path),
            arr=y_pred)
    return y_pred

def get_performance(evalconfig, y_test, y_pred, dataset_name):
    """Get performance measures from predictions using the supplied configuration."""
    # Assign classes using the chosen threshold
    y_pred_class = (y_pred > evalconfig.thresh).astype('int32')
    # Calculate performance measures
    log_loss = mtr.log_loss(y_test, y_pred, eps=1e-07)
    acc = mtr.accuracy_score(y_test, y_pred_class)
    auroc = mtr.roc_auc_score(y_test, y_pred)
    mcc = mtr.matthews_corrcoef(y_test, y_pred_class)
    f1 = mtr.f1_score(y_test, y_pred_class)
    precision = mtr.precision_score(y_test, y_pred_class)
    recall = mtr.recall_score(y_test, y_pred_class)
    aupr = mtr.average_precision_score(y_test, y_pred_class)

    # Save the results
    with open("{}-metrics.csv".format(evalconfig.name_prefix), 'a', newline="") as csv_file:
        file_writer = csv.writer(csv_file)
        file_writer.writerow((dataset_name, log_loss, acc, auroc, aupr, precision, recall, mcc, f1))
    if evalconfig.do_plots:
        fpr, tpr, threshold = mtr.roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auroc)
        plt.title("AUROC: {}".format(dataset_name))
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig("auc_{p}_{s}.png".format(p=evalconfig.name_prefix, s=dataset_name))
        plt.clf()

        precision, recall, thresholds = mtr.precision_recall_curve(y_test, y_pred)
        plt.plot(recall, precision, 'b', label='AUC = %0.2f' % aupr)
        plt.title("AUPR: {}".format(dataset_name))
        plt.legend(loc='lower right')      
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.savefig("aupr_{p}_{s}.png".format(p=evalconfig.name_prefix, s=dataset_name))
        plt.clf()
        
if __name__ == "__main__":
    main()
