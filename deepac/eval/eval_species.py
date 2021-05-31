"""@package deepac.eval.eval_species
Evaluate a NN trained on Illumina reads, species-wise.

Requires a config file describing the data directory, dataset and run name,
classification threshold and the epoch range.
  
"""
from tensorflow.keras import backend
import numpy as np
import csv
from deepac.eval.eval import get_performance, get_eval_header
from deepac.predict import predict_multiread
import os


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
        self.classcsv_path = config['Data']['ClassCSV']
        # Set csv delimiter
        self.delim = config['Data']['Delim']
        # Set Run name
        self.name_prefix = config['Data']['RunName']
        # Set threshold
        self.thresh = config['Data'].getfloat('Threshold')
        self.read_confidence_thresh = config['Data'].get('ReadConfidenceThresh', fallback=None)
        if self.read_confidence_thresh == "none" or self.read_confidence_thresh == "None":
            self.read_confidence_thresh = None
        elif self.read_confidence_thresh is not None:
            self.read_confidence_thresh = float(self.read_confidence_thresh)
        self.confidence_thresh = config['Data'].get('GenomeConfidenceThresh', fallback=None)
        if self.confidence_thresh == "none" or self.confidence_thresh == "None":
            self.confidence_thresh = None
        elif self.confidence_thresh is not None:
            self.confidence_thresh = float(self.confidence_thresh)
        self.n_classes = config['Data'].getint('N_Classes', fallback=2)
        self.target_class = config['Data'].get('TargetClass', fallback="*")
        if self.target_class == "*":
            self.target_class = None
        else:
            self.target_class = config['Data'].getint('TargetClass')

        # Set plotting
        self.do_plots = config['Options'].getboolean('Do_plots')
        self.add_activ = config['Options'].getboolean('Add_activ', fallback="False")


def get_species_preds(y_pred, class_csv_path, delimiter=';',
                      threshold=0.5, confidence_thresh=None, n_classes=2, add_activ=False):
    """Generate species-wise predictions."""
    species_list = []
    with open(class_csv_path, 'r') as class_csv:
        classreader = csv.reader(class_csv, delimiter=delimiter)
        for row in classreader:
            species_list.append((int(row[0]), row[1], int(row[2])))
    used = 0
    y_result = []
    class_dict = {}
    labels = []
    for c in range(n_classes):
        class_dict[c] = []
    for row in species_list:
        labels.append(row[0])
        class_dict[row[0]].append((row[1], row[2]))
    for c in range(n_classes):
        for species in class_dict[c]:
            pred = predict_multiread(y_pred[used:used+species[1]], threshold, confidence_thresh, n_classes,
                                     add_activ=add_activ)
            y_result.append(pred)
            used = used + species[1]
    return np.asarray(y_result), np.asarray(labels)


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
    y_pred_1, y_test = get_species_preds(y_pred_1_raw, evalconfig.classcsv_path,
                                         threshold=evalconfig.thresh,
                                         confidence_thresh=evalconfig.read_confidence_thresh,
                                         n_classes=evalconfig.n_classes,
                                         add_activ=evalconfig.add_activ)
    get_performance(evalconfig, y_test, y_pred_1, dataset_name=evalconfig.dataset_path)

    if evalconfig.pairedset_path is not None:
        print("Loading {}...".format(evalconfig.pairedpred_path))      
        y_pred_2_raw = np.load("{}/{}".format(evalconfig.dir_path, evalconfig.pairedpred_path))

        print("Predicting labels for {}...".format(evalconfig.pairedpred_path))
        y_pred_2, y_test = get_species_preds(y_pred_2_raw, evalconfig.classcsv_path,
                                             threshold=evalconfig.thresh,
                                             confidence_thresh=evalconfig.read_confidence_thresh,
                                             n_classes=evalconfig.n_classes,
                                             add_activ=evalconfig.add_activ)
        get_performance(evalconfig, y_test, y_pred_2, dataset_name=evalconfig.pairedset_path)

        y_pred_combined_raw = np.mean([y_pred_1_raw, y_pred_2_raw], axis=0)
        y_pred_combined, y_test = get_species_preds(y_pred_combined_raw, evalconfig.classcsv_path,
                                                    threshold=evalconfig.thresh,
                                                    confidence_thresh=evalconfig.read_confidence_thresh,
                                                    n_classes=evalconfig.n_classes,
                                                    add_activ=evalconfig.add_activ)
        get_performance(evalconfig, y_test, y_pred_combined, dataset_name=evalconfig.combinedset_path)
        pred_to_save = y_pred_combined
    else:
        pred_to_save = y_pred_1
    if evalconfig.add_activ:
        np.save("{}/{}-activ-genomes.npy".format(evalconfig.dir_path,
                                                 os.path.splitext(evalconfig.datapred_path)[0]), pred_to_save)
    else:
        np.save("{}/{}-genomes.npy".format(evalconfig.dir_path, os.path.splitext(evalconfig.datapred_path)[0]),
                pred_to_save)


