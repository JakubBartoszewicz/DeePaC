"""@package deepac.eval.eval
Evaluate a NN trained on Illumina reads.

Requires a config file describing the data directory, dataset and run name,
classification threshold and the epoch range.
  
"""
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import backend
import sklearn.metrics as mtr
import numpy as np
import csv
from deepac.predict import predict_array


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
        self.confidence_thresh = config['Data'].getfloat('ConfidenceThresh')

        # Set the first and last epoch to evaluate
        self.epoch_start = config['Epochs'].getint('EpochStart')
        self.epoch_end = config['Epochs'].getint('EpochEnd')

        self.do_plots = config['Options'].getboolean('Do_plots')
        self.do_rc = config['Options'].getboolean('Do_RC')
        self.do_pred = config['Options'].getboolean('Do_Pred')
        self.ignore_unmatched = config['Options'].getboolean('Ignore_unmatched')


def get_eval_header():
    return ("epoch", "set", "tp", "tn", "fp", "fn", "missing", "log_loss", "acc", "auroc",
            "aupr", "precision", "recall", "spec", "mcc", "f1", "pred_rate")


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

    # Write CSV header
    with open("{}-metrics.csv".format(evalconfig.name_prefix), 'a', newline="") as csv_file:
        file_writer = csv.writer(csv_file)
        file_writer.writerow(get_eval_header())

    # Evaluate for each saved model in epoch range
    for n_epoch in range(evalconfig.epoch_start, evalconfig.epoch_end + 1):
        print("Predicting labels for {}_data.npy...".format(evalconfig.dataset_path))
        if evalconfig.do_pred:
            y_pred_1 = predict(evalconfig, x_test, n_epoch, paired=False, rc=evalconfig.do_rc)
        else:
            if evalconfig.do_rc:
                filename = "{p}-e{ne:03d}-predictions-{s}-rc.npy".format(p=evalconfig.name_prefix, ne=n_epoch,
                                                                         s=evalconfig.dataset_path)
            else:
                filename = "{p}-e{ne:03d}-predictions-{s}.npy".format(p=evalconfig.name_prefix, ne=n_epoch,
                                                                      s=evalconfig.dataset_path)
            y_pred_1 = np.load(filename)
        get_performance(evalconfig, y_test, y_pred_1, dataset_name=evalconfig.dataset_path, n_epoch=n_epoch,
                        ignore_unmatched=evalconfig.ignore_unmatched)

        if evalconfig.pairedset_path is not None:
            print("Loading {}_data.npy...".format(evalconfig.pairedset_path))
            y_test = np.load("{}/{}_labels.npy".format(evalconfig.dir_path, evalconfig.pairedset_path))
            if evalconfig.do_pred:
                x_test = np.load("{}/{}_data.npy".format(evalconfig.dir_path, evalconfig.pairedset_path))

            print("Predicting labels for {}_data.npy...".format(evalconfig.pairedset_path))
            if evalconfig.do_pred:
                y_pred_2 = predict(evalconfig, x_test, n_epoch, paired=True, rc=evalconfig.do_rc)
            else:
                if evalconfig.do_rc:
                    filename = "{p}-e{ne:03d}-predictions-{s}-rc.npy".format(p=evalconfig.name_prefix, ne=n_epoch,
                                                                             s=evalconfig.pairedset_path)
                else:
                    filename = "{p}-e{ne:03d}-predictions-{s}.npy".format(p=evalconfig.name_prefix, ne=n_epoch,
                                                                          s=evalconfig.pairedset_path)
                y_pred_2 = np.load(filename)
            get_performance(evalconfig, y_test, y_pred_2, dataset_name=evalconfig.pairedset_path, n_epoch=n_epoch,
                            ignore_unmatched=evalconfig.ignore_unmatched)

            y_pred_combined = np.mean([y_pred_1, y_pred_2], axis=0)
            get_performance(evalconfig, y_test, y_pred_combined, dataset_name=evalconfig.combinedset_path,
                            n_epoch=n_epoch, ignore_unmatched=evalconfig.ignore_unmatched)


def predict(evalconfig, x_test, n_epoch, paired=False, rc=False):
    """Predict the pathogenic potentials of Illumina reads using the supplied configuration."""
    if paired:
        dataset_path = evalconfig.pairedset_path
    else:
        dataset_path = evalconfig.dataset_path
    model = load_model("{p}-e{ne:03d}.h5".format(p=evalconfig.name_prefix, ne=n_epoch), )
    # Backup predicted probabilities for future analyses
    if rc:
        filename = "{p}-e{ne:03d}-predictions-{s}-rc.npy".format(p=evalconfig.name_prefix, ne=n_epoch, s=dataset_path)
    else:
        filename = "{p}-e{ne:03d}-predictions-{s}.npy".format(p=evalconfig.name_prefix, ne=n_epoch, s=dataset_path)
    # Predict class probabilities
    y_pred, _ = predict_array(model, x_test, output=filename, rc=rc)
    return y_pred


def try_metric(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as err:
        print(err)
        return np.nan


def get_performance(evalconfig, y_test, y_pred, dataset_name, n_epoch=np.nan, ignore_unmatched=False):
    """Get performance measures from predictions using the supplied configuration."""
    y_test_main = y_test
    y_test_matched = y_test
    missing = 0
    prediction_rate = 1
    if np.isclose(evalconfig.confidence_thresh, evalconfig.thresh):
        # Assign classes using the chosen threshold
        y_pred_class = (y_pred > evalconfig.thresh).astype('int8')
        y_pred_class_matched = y_pred_class
    else:
        interval = np.abs(evalconfig.confidence_thresh - evalconfig.thresh)
        y_pred_class_pos = y_pred > (evalconfig.thresh + interval)
        y_pred_class_neg = y_pred < (evalconfig.thresh - interval)
        y_pred_class_undef = np.logical_not(np.any([y_pred_class_pos, y_pred_class_neg], axis=0))
        missing = np.sum(y_pred_class_undef.astype('int8'))
        prediction_rate = (y_test.shape[0] - missing) / y_test.shape[0]
        y_pred_class = np.empty(y_pred.shape)
        y_pred_class[:] = np.nan
        y_pred_class[y_pred_class_pos] = np.int8(1)
        y_pred_class[y_pred_class_neg] = np.int8(0)
        y_pred_class_matched = y_pred_class[np.any([y_pred_class_pos, y_pred_class_neg], axis=0)]
        y_test_matched = y_test[np.any([y_pred_class_pos, y_pred_class_neg], axis=0)]
        if ignore_unmatched:
            y_pred_class = y_pred_class_matched
            y_test_main = y_test_matched
        else:
            y_pred_class[np.all([y_pred_class_undef, y_test], axis=0)] = np.int8(0)
            y_pred_class[np.all([y_pred_class_undef, np.logical_not(y_test)], axis=0)] = np.int8(1)
    # Calculate performance measures
    log_loss = try_metric(mtr.log_loss, y_test, y_pred, eps=1e-07)
    auroc = try_metric(mtr.roc_auc_score, y_test, y_pred)
    aupr = try_metric(mtr.average_precision_score, y_test, y_pred)
    acc = try_metric(mtr.accuracy_score, y_test_main, y_pred_class)
    mcc = try_metric(mtr.matthews_corrcoef, y_test_main, y_pred_class)
    f1 = try_metric(mtr.f1_score, y_test_main, y_pred_class)
    precision = try_metric(mtr.precision_score, y_test_main, y_pred_class)
    recall = try_metric(mtr.recall_score, y_test_main, y_pred_class)
    try:
        tn, fp, fn, tp = mtr.confusion_matrix(y_test_matched, y_pred_class_matched).ravel()
    except Exception as err:
        print(err)
        tn, fp, fn, tp = np.nan, np.nan, np.nan, np.nan
    try:
        if ignore_unmatched:
            specificity = tn / (tn + fp)
        else:
            specificity = tn / (tn + fp + missing)
    except Exception as err:
        print(err)
        specificity = np.nan

    # Save the results
    with open("{}-metrics.csv".format(evalconfig.name_prefix), 'a', newline="") as csv_file:
        file_writer = csv.writer(csv_file)
        file_writer.writerow((n_epoch, dataset_name, tp, tn, fp, fn, missing, log_loss, acc, auroc, aupr, precision,
                              recall, specificity, mcc, f1, prediction_rate))
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
            if np.isnan(n_epoch):
                plt.savefig("{p}_{s}_auc.png".format(p=evalconfig.name_prefix, s=dataset_name))
            else:
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
            if np.isnan(n_epoch):
                plt.savefig("{p}_{s}_aupr.png".format(p=evalconfig.name_prefix, s=dataset_name))
            else:
                plt.savefig("{p}_{ne}_{s}_aupr.png".format(p=evalconfig.name_prefix, ne=n_epoch, s=dataset_name))
            plt.clf()
