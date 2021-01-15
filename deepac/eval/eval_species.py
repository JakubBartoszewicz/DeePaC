"""@package deepac.eval.eval_species
Evaluate a NN trained on Illumina reads, species-wise.

Requires a config file describing the data directory, dataset and run name,
classification threshold and the epoch range.
  
"""
from tensorflow.keras import backend
import numpy as np
import csv
from deepac.eval.eval import get_performance, get_eval_header


class EvalSpecConfig:
    """
    Species-wise evaluation configuration class.

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

        # Set the evaluation dataset pred
        self.datapred_path = config['Data']['DataPredictions']
        # Set the paired dataset pred
        self.pairedpred_path = config['Data']['PairedPredictions']
        if self.pairedpred_path == "none":
            self.pairedpred_path = None

        # Set the evaluation dataset name                                         
        self.poscsv_path = config['Data']['PosCSV']                                     
        # Set the paired dataset name                                       
        self.negcsv_path = config['Data']['NegCSV']
        # Set csv delimiter
        self.delim = config['Data']['Delim']
        # Set Run name
        self.name_prefix = config['Data']['RunName']
        # Set threshold
        self.thresh = config['Data'].getfloat('Threshold')
        self.confidence_thresh = config['Data'].getfloat('ConfidenceThresh')

        # Set plotting
        self.do_plots = config['Options'].getboolean('Do_plots')
        self.ignore_unmatched = config['Options'].getboolean('Ignore_unmatched')


def get_species_preds(y_pred, poscsv_path, negcsv_path, delimiter=';'):
    """Generate species-wise predictions."""
    neglist = []
    with open(negcsv_path, 'r') as negcsv:
        negreader = csv.reader(negcsv, delimiter=delimiter)
        for row in negreader:
            neglist.append((row[0], int(row[1])))
    poslist = []
    with open(poscsv_path, 'r') as poscsv:
        posreader = csv.reader(poscsv, delimiter=delimiter)
        for row in posreader:
            poslist.append((row[0],int(row[1])))
    used = 0
    y_result = []
    for species in neglist:
        pred = np.mean(y_pred[used:used+species[1]])
        y_result.append(pred)
        used = used + species[1]
    for species in poslist:
        pred = np.mean(y_pred[used:used+species[1]])
        y_result.append(pred)
        used = used + species[1]
    return np.asarray(y_result), np.concatenate((np.repeat(0, len(neglist)), np.repeat(1, len(poslist))))


def evaluate_species(config):
    """Evaluate the NN on Illumina reads using the supplied configuration."""
    # Clear session needed or TypeError can happen in some cases
    backend.clear_session()

    evalconfig = EvalSpecConfig(config)

    # Read data to memory
    print("Loading {}...".format(evalconfig.datapred_path))
    y_pred_1_raw = np.load("{}/{}".format(evalconfig.dir_path, evalconfig.datapred_path))

    # Write CSV header
    with open("{}-metrics.csv".format(evalconfig.name_prefix), 'a', newline="") as csv_file:
        file_writer = csv.writer(csv_file)
        file_writer.writerow(get_eval_header())

    # Evaluate for each saved model in epoch range
    print("Predicting labels for {}...".format(evalconfig.datapred_path))
    y_pred_1, y_test = get_species_preds(y_pred_1_raw, evalconfig.poscsv_path, evalconfig.negcsv_path)
    get_performance(evalconfig, y_test, y_pred_1, dataset_name=evalconfig.dataset_path,
                    ignore_unmatched=evalconfig.ignore_unmatched)

    if evalconfig.pairedset_path is not None:
        print("Loading {}...".format(evalconfig.pairedpred_path))      
        y_pred_2_raw = np.load("{}/{}".format(evalconfig.dir_path, evalconfig.pairedpred_path))

        print("Predicting labels for {}...".format(evalconfig.pairedpred_path))
        y_pred_2, y_test = get_species_preds(y_pred_2_raw, evalconfig.poscsv_path, evalconfig.negcsv_path)
        get_performance(evalconfig, y_test, y_pred_2, dataset_name=evalconfig.pairedset_path,
                        ignore_unmatched=evalconfig.ignore_unmatched)

        y_pred_combined_raw = np.mean([y_pred_1_raw, y_pred_2_raw], axis=0)
        y_pred_combined, y_test = get_species_preds(y_pred_combined_raw, evalconfig.poscsv_path, evalconfig.negcsv_path)
        get_performance(evalconfig, y_test, y_pred_combined, dataset_name=evalconfig.combinedset_path,
                        ignore_unmatched=evalconfig.ignore_unmatched)

