"""@package deepac.eval.eval_ens
Evaluate an ensemble of NNs trained on Illumina reads.

Requires a config file describing the data directory, dataset and run name,
classification threshold and the epoch range.
  
"""
from tensorflow.keras.models import load_model
from tensorflow.keras import backend
import numpy as np
import csv
from deepac.eval.eval import get_performance, get_eval_header
from deepac.predict import predict_array


class EvalEnsConfig:
    """
    Ensemble evaluation configuration class.

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
        self.run_prefixes = self.runnames
        self.name_prefix = self.ensname
        # Set the classification threshold
        self.thresh = config['Data'].getfloat('Threshold')
        self.confidence_thresh = config['Data'].getfloat('ConfidenceThresh')

        # Set the first and last epoch to evaluate
        self.epoch = [int(e) for e in config['Epochs']['Epoch'].split(',')]

        self.do_plots = config['Options'].getboolean('Do_plots')
        self.do_pred = config['Options'].getboolean('Do_Pred')
        self.ignore_unmatched = config['Options'].getboolean('Ignore_unmatched')


def evaluate_ensemble(config):
    """Evaluate the NN on Illumina reads using the supplied configuration."""
    # Clear session needed or TypeError can happen in some cases
    backend.clear_session()
    
    evalconfig = EvalEnsConfig(config)

    # Read data to memory
    print("Loading {}_data.npy...".format(evalconfig.dataset_path))
    y_test = np.load("{}/{}_labels.npy".format(evalconfig.dir_path, evalconfig.dataset_path))
    x_test = []
    if evalconfig.do_pred:
        x_test = np.load("{}/{}_data.npy".format(evalconfig.dir_path, evalconfig.dataset_path))
    
    # Write CSV header
    with open("{}-metrics.csv".format(evalconfig.name_prefix), 'a', newline="") as csv_file:
        file_writer = csv.writer(csv_file)
        file_writer.writerow(get_eval_header())

    # Evaluate for each saved model in epoch range
    print("Predicting labels for {}_data.npy...".format(evalconfig.dataset_path))

    y_pred_1 = predict(evalconfig, x_test, do_pred=evalconfig.do_pred)
    get_performance(evalconfig, y_test, y_pred_1, dataset_name=evalconfig.dataset_path,
                    ignore_unmatched=evalconfig.ignore_unmatched)

    if evalconfig.pairedset_path is not None:
        print("Loading {}_data.npy...".format(evalconfig.pairedset_path))
        y_test = np.load("{}/{}_labels.npy".format(evalconfig.dir_path, evalconfig.pairedset_path))
        if evalconfig.do_pred:
            x_test = np.load("{}/{}_data.npy".format(evalconfig.dir_path, evalconfig.pairedset_path))

        print("Predicting labels for {}_data.npy...".format(evalconfig.pairedset_path))
        y_pred_2 = predict(evalconfig, x_test, paired=True, do_pred=evalconfig.do_pred)
        get_performance(evalconfig, y_test, y_pred_2,  dataset_name=evalconfig.pairedset_path,
                        ignore_unmatched=evalconfig.ignore_unmatched)

        y_pred_combined = np.mean([y_pred_1, y_pred_2], axis=0)
        get_performance(evalconfig, y_test, y_pred_combined, dataset_name=evalconfig.combinedset_path,
                        ignore_unmatched=evalconfig.ignore_unmatched)


def predict(evalconfig, x_test, paired=False, do_pred=True):
    """Predict the pathogenic potentials of Illumina reads using the supplied configuration."""
    if paired:
        dataset_path = evalconfig.pairedset_path
    else:
        dataset_path = evalconfig.dataset_path
        
    y_preds = []
    for i in range(0, len(evalconfig.run_prefixes)):
        filename = "{p}-e{ne:03d}-predictions-{s}.npy".format(p=evalconfig.run_prefixes[i], ne=evalconfig.epoch[i],
                                                              s=dataset_path)
        if do_pred:
            model = load_model("{p}-e{ne:03d}.h5".format(p=evalconfig.run_prefixes[i], ne=evalconfig.epoch[i]))
            # Predict class probabilities
            y_preds.append(predict_array(model, x_test, filename)[0])
        else:
            y_preds.append(np.load(filename))
    y_pred = np.mean(y_preds, axis=0)
    # Backup predicted probabilities for future analyses
    np.save(file="{p}-predictions-{s}.npy".format(p=evalconfig.name_prefix, s=dataset_path),
            arr=y_pred)
    return y_pred
