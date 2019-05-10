"""@package deepac.eval.eval
Evaluate a NN trained on Illumina reads.

Requires a config file describing the data directory, dataset and run name,
classification threshold and the epoch range.
  
"""
from keras.models import load_model
from keras.backend import tensorflow_backend as backend
import sklearn.metrics as mtr
import numpy as np
import csv
import matplotlib.pyplot as plt


class EvalConfig:
    """
    Evaluation configuration class.

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
        self.name_prefix = config['Data']['RunName']
        # Set the classification threshold
        self.thresh = config['Data'].getfloat('Threshold')

        # Set the first and last epoch to evaluate
        self.epoch_start = config['Epochs'].getint('EpochStart')
        self.epoch_end = config['Epochs'].getint('EpochEnd')

        self.do_plots = config['Options'].getboolean('Do_plots')
        self.do_rc = config['Options'].getboolean('Do_RC')
        self.do_pred = config['Options'].getboolean('Do_Pred')


def evaluate_reads(config):
    """Evaluate the NN on Illumina reads using the supplied configuration."""
    # Clear session needed or TypeError can happen in some cases
    backend.clear_session()

    evalconfig = EvalConfig(config)
    x_test = None

    # Read data to memory
    print("Loading {}_data.npy...".format(evalconfig.dataset_path))
    y_test = np.load("{}/{}_labels.npy".format(evalconfig.dir_path, evalconfig.dataset_path))
    if evalconfig.do_pred:
        x_test = np.load("{}/{}_data.npy".format(evalconfig.dir_path, evalconfig.dataset_path))
        if evalconfig.do_rc:
            x_test = x_test[::, ::-1, ::-1]

    # Write CSV header
    with open("{}-metrics.csv".format(evalconfig.name_prefix), 'a', newline="") as csv_file:
        file_writer = csv.writer(csv_file)
        file_writer.writerow(("epoch", "set", "tp", "tn", "fp", "fn", "log_loss", "acc", "auroc", "aupr", "precision",
                              "recall", "spec", "mcc", "f1"))

    # Evaluate for each saved model in epoch range
    for n_epoch in range(evalconfig.epoch_start, evalconfig.epoch_end):
        print("Predicting labels for {}_data.npy...".format(evalconfig.dataset_path))
        if evalconfig.do_pred:
            y_pred_1 = predict(evalconfig, x_test, n_epoch, paired=False, save_as_rc=evalconfig.do_rc)
        else:
            if evalconfig.do_rc:
                filename = "{p}-e{ne:03d}-predictions-{s}-rc.npy".format(p=evalconfig.name_prefix, ne=n_epoch,
                                                                         s=evalconfig.dataset_path)
            else:
                filename = "{p}-e{ne:03d}-predictions-{s}.npy".format(p=evalconfig.name_prefix, ne=n_epoch,
                                                                      s=evalconfig.dataset_path)
            y_pred_1 = np.load(filename)
        get_performance(evalconfig, y_test, y_pred_1, dataset_name=evalconfig.dataset_path, n_epoch=n_epoch)

        if evalconfig.pairedset_path is not None:
            print("Loading {}_data.npy...".format(evalconfig.pairedset_path))
            y_test = np.load("{}/{}_labels.npy".format(evalconfig.dir_path, evalconfig.pairedset_path))
            if evalconfig.do_pred:
                x_test = np.load("{}/{}_data.npy".format(evalconfig.dir_path, evalconfig.pairedset_path))
                if evalconfig.do_rc:
                    x_test = x_test[::, ::-1, ::-1]

            print("Predicting labels for {}_data.npy...".format(evalconfig.pairedset_path))
            if evalconfig.do_pred:
                y_pred_2 = predict(evalconfig, x_test, n_epoch, paired=True, save_as_rc=evalconfig.do_rc)
            else:
                if evalconfig.do_rc:
                    filename = "{p}-e{ne:03d}-predictions-{s}-rc.npy".format(p=evalconfig.name_prefix, ne=n_epoch,
                                                                             s=evalconfig.pairedset_path)
                else:
                    filename = "{p}-e{ne:03d}-predictions-{s}.npy".format(p=evalconfig.name_prefix, ne=n_epoch,
                                                                          s=evalconfig.pairedset_path)
                y_pred_2 = np.load(filename)
            get_performance(evalconfig, y_test, y_pred_2, dataset_name=evalconfig.pairedset_path, n_epoch=n_epoch)

            y_pred_combined = np.mean([y_pred_1, y_pred_2], axis=0)
            get_performance(evalconfig, y_test, y_pred_combined, dataset_name=evalconfig.combinedset_path,
                            n_epoch=n_epoch)


def predict(evalconfig, x_test, n_epoch, paired=False, save_as_rc=False):
    """Predict the pathogenic potentials of Illumina reads using the supplied configuration."""
    if paired:
        dataset_path = evalconfig.pairedset_path
    else:
        dataset_path = evalconfig.dataset_path
    model = load_model("{p}-e{ne:03d}.h5".format(p=evalconfig.name_prefix, ne=n_epoch), )
    # Predict class probabilities
    y_pred = np.ndarray.flatten(model.predict(x_test))
    # Backup predicted probabilities for future analyses
    if save_as_rc:
        filename = "{p}-e{ne:03d}-predictions-{s}-rc.npy".format(p=evalconfig.name_prefix, ne=n_epoch, s=dataset_path)
    else:
        filename = "{p}-e{ne:03d}-predictions-{s}.npy".format(p=evalconfig.name_prefix, ne=n_epoch, s=dataset_path)
    np.save(file=filename, arr=y_pred)
    return y_pred


def get_performance(evalconfig, y_test, y_pred, dataset_name, n_epoch=np.nan):
    """Get performance measures from predictions using the supplied configuration."""
    # Assign classes using the chosen threshold
    y_pred_class = (y_pred > evalconfig.thresh).astype('int8')
    # Calculate performance measures
    try:
        log_loss = mtr.log_loss(y_test, y_pred, eps=1e-07)
    except Exception as err:
        print(err)
        log_loss = np.nan
    try:
        acc = mtr.accuracy_score(y_test, y_pred_class)
    except Exception as err:
        print(err)
        acc = np.nan
    try:
        auroc = mtr.roc_auc_score(y_test, y_pred)
    except Exception as err:
        print(err)
        auroc = np.nan
    try:
        mcc = mtr.matthews_corrcoef(y_test, y_pred_class)
    except Exception as err:
        print(err)
        mcc = np.nan
    try:
        f1 = mtr.f1_score(y_test, y_pred_class)
    except Exception as err:
        print(err)
        f1 = np.nan
    try:
        precision = mtr.precision_score(y_test, y_pred_class)
    except Exception as err:
        print(err)
        precision = np.nan
    try:
        recall = mtr.recall_score(y_test, y_pred_class)
    except Exception as err:
        print(err)
        recall = np.nan
    try:
        tn, fp, fn, tp = mtr.confusion_matrix(y_test, y_pred_class).ravel()
        specificity = tn / (tn + fp)
    except Exception as err:
        print(err)
        recall = np.nan
    try:
        aupr = mtr.average_precision_score(y_test, y_pred)
    except Exception as err:
        print(err)
        aupr = np.nan

    # Save the results
    with open("{}-metrics.csv".format(evalconfig.name_prefix), 'a', newline="") as csv_file:
        file_writer = csv.writer(csv_file)
        file_writer.writerow((n_epoch, dataset_name, tp, tn, fp, fn, log_loss, acc, auroc, aupr, precision, recall,
                              specificity, mcc, f1))
    if evalconfig.do_plots:
        if not np.isnan(auroc):
            fpr, tpr, threshold = mtr.roc_curve(y_test, y_pred)
            plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auroc)
            plt.title("AUROC: {}".format(dataset_name))
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.savefig("{p}_{ne}_{s}_auc.png".format(p=evalconfig.name_prefix, ne=n_epoch, s=dataset_name))
            plt.clf()

        if not np.isnan(aupr):
            precision, recall, thresholds = mtr.precision_recall_curve(y_test, y_pred)
            plt.plot(recall, precision, 'b', label='AUC = %0.2f' % aupr)
            plt.title("AUPR: {}".format(dataset_name))
            plt.legend(loc='lower right')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('Precision')
            plt.xlabel('Recall')
            plt.savefig("{p}_{ne}_{s}_aupr.png".format(p=evalconfig.name_prefix, ne=n_epoch, s=dataset_name))
            plt.clf()
