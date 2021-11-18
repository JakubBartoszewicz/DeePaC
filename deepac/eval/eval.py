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
from tensorflow.keras.utils import get_custom_objects
from termcolor import colored


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
        self.confidence_thresh = config['Data'].get('ConfidenceThresh', fallback=None)
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
        self.replicates = config['Data'].getint('Replicates', fallback=1)
        # Set the first and last epoch to evaluate
        self.epoch_start = config['Epochs'].getint('EpochStart')
        self.epoch_end = config['Epochs'].getint('EpochEnd')

        self.do_plots = config['Options'].getboolean('Do_plots')
        self.do_rc = config['Options'].getboolean('Do_RC')
        self.do_pred = config['Options'].getboolean('Do_Pred')
        self.force_concordance = config['Options'].getboolean('force_concordance')


def get_eval_header():
    return ("epoch", "set", "target_class", "tp", "tn", "fp", "fn", "missing", "log_loss", "matched_acc",
            "matched_recall", "matched_spec", "total_acc", "auroc",
            "aupr", "precision", "total_recall", "total_spec", "mcc", "pred_rate")


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
        get_performance(evalconfig, y_test, y_pred_1, dataset_name=evalconfig.dataset_path, n_epoch=n_epoch)

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
            get_performance(evalconfig, y_test, y_pred_2, dataset_name=evalconfig.pairedset_path, n_epoch=n_epoch)

            y_pred_combined = np.mean([y_pred_1, y_pred_2], axis=0)
            get_performance(evalconfig, y_test, y_pred_combined, dataset_name=evalconfig.combinedset_path,
                            n_epoch=n_epoch)


def predict(evalconfig, x_test, n_epoch, paired=False, rc=False):
    """Predict the pathogenic potentials of Illumina reads using the supplied configuration."""
    if paired:
        dataset_path = evalconfig.pairedset_path
    else:
        dataset_path = evalconfig.dataset_path
    model = load_model("{p}-e{ne:03d}.h5".format(p=evalconfig.name_prefix, ne=n_epoch),
                       custom_objects=get_custom_objects())
    # Backup predicted probabilities for future analyses
    if rc:
        filename = "{p}-e{ne:03d}-predictions-{s}-rc.npy".format(p=evalconfig.name_prefix, ne=n_epoch, s=dataset_path)
    else:
        filename = "{p}-e{ne:03d}-predictions-{s}.npy".format(p=evalconfig.name_prefix, ne=n_epoch, s=dataset_path)
    # Predict class probabilities
    y_pred, _ = predict_array(model, x_test, replicates=evalconfig.replicates, output=filename, rc=rc)
    return y_pred


def try_metric(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as err:
        print(colored(function.__name__ + ": " + str(err), "red"))
        return np.nan


def get_performance(evalconfig, y_test, y_pred, dataset_name, n_epoch=np.nan):
    """Get performance measures from predictions using the supplied configuration."""
    y_test_main = y_test
    missing = 0
    prediction_rate = 1
    multiclass = True if evalconfig.n_classes > 2 else False
    labels = list(range(evalconfig.n_classes))
    average = None
    average_au = None
    target_class = evalconfig.target_class if multiclass else None
    if target_class is None:
        average = "macro" if multiclass else "binary"
        average_au = "macro"
    y_orig_pred_undef = np.isnan(y_pred)
    if len(y_orig_pred_undef.shape) > 1:
        y_orig_pred_undef = np.any(y_orig_pred_undef, axis=-1)
    y_orig_pred_def = np.logical_not(y_orig_pred_undef)

    if evalconfig.confidence_thresh is None or \
            (np.isclose(evalconfig.confidence_thresh, evalconfig.thresh) and not multiclass):
        # Assign classes using the chosen threshold
        if multiclass:
            y_pred_class = np.argmax(y_pred, axis=-1).astype(np.float)
        else:
            y_pred = y_pred.flatten()
            y_pred_class = (y_pred > evalconfig.thresh).astype(np.float)
        y_pred_class_undef = y_orig_pred_undef
        y_pred_class[y_pred_class_undef] = np.nan
        y_pred_class_matched = y_pred_class[np.logical_not(y_pred_class_undef)]
        y_test_matched = y_test[np.logical_not(y_pred_class_undef)]
    else:
        interval = np.abs(evalconfig.confidence_thresh - evalconfig.thresh)
        if multiclass:
            y_pred_class_def = np.all([y_orig_pred_def,
                                       np.max(y_pred, axis=-1) > evalconfig.confidence_thresh], axis=0)
            y_pred_class_undef = np.logical_not(y_pred_class_def)
            y_pred_class = np.argmax(y_pred, axis=-1).astype(np.float)
            y_pred_class[y_pred_class_undef] = np.nan
            y_pred_class_matched = y_pred_class[np.logical_not(y_pred_class_undef)]
            y_test_matched = y_test[y_pred_class_def]
            y_pred_class = y_pred_class_matched
            y_test_main = y_test_matched
        else:
            y_pred = y_pred.flatten()
            y_pred_class_pos = y_pred > (evalconfig.thresh + interval)
            y_pred_class_neg = y_pred < (evalconfig.thresh - interval)
            y_pred_class_def = np.any([y_pred_class_pos, y_pred_class_neg], axis=0)
            y_pred_class_undef = np.logical_not(y_pred_class_def)
            y_pred_class = np.empty(y_pred.shape)
            y_pred_class[:] = np.nan
            y_pred_class[y_pred_class_pos] = 1
            y_pred_class[y_pred_class_neg] = 0
            y_pred_class_matched = y_pred_class[y_pred_class_def]
            y_test_matched = y_test[y_pred_class_def]

    if np.any(y_pred_class_undef):
        if multiclass:
            print(colored("Cannot handle NaN predictions in multiclass setting.", "yellow"))
        else:
            y_pred_class[np.all([y_pred_class_undef, y_test], axis=0)] = 0
            y_pred_class[np.all([y_pred_class_undef, np.logical_not(y_test)], axis=0)] = 1

        missing = np.sum(y_pred_class_undef.astype(np.float))
        prediction_rate = (y_test.shape[0] - missing) / y_test.shape[0]

    # Calculate performance measures
    # missing predictions due to confidence thresh don't affect AUPR, AUC & log loss
    # (but NaNs present in the original predictions must be filtered)
    log_loss = try_metric(mtr.log_loss, y_test[y_orig_pred_def], y_pred[y_orig_pred_def], eps=1e-07, labels=labels)
    if average_au is not None:
        auroc = try_metric(mtr.roc_auc_score, y_test[y_orig_pred_def], y_pred[y_orig_pred_def],
                           multi_class="ovr", average=average_au, labels=labels)
    else:
        # mtr.roc_auc_score doesn't support average=None
        print(colored("AUC not supported for multiclass evaluation with a specific target class", "yellow"))
        auroc = "n/a"
    if multiclass:
        y_test_multi = np.zeros((y_test.shape[0], evalconfig.n_classes))
        for i in range(y_test.shape[0]):
            y_test_multi[i, y_test[i]] = 1
        aupr = try_metric(mtr.average_precision_score, y_test_multi[y_orig_pred_def], y_pred[y_orig_pred_def],
                          average=average_au)
    else:
        aupr = try_metric(mtr.average_precision_score, y_test[y_orig_pred_def], y_pred[y_orig_pred_def],
                          average=average_au)

    # missing predictions don't affect mcc & precision
    mcc = try_metric(mtr.matthews_corrcoef, y_test_matched, y_pred_class_matched)
    precision = try_metric(mtr.precision_score, y_test_matched, y_pred_class_matched, average=average, labels=labels)

    # missing predictions affect acc, recall, specificity & f1
    matched_acc = try_metric(mtr.accuracy_score, y_test_matched, y_pred_class_matched)
    acc = try_metric(mtr.accuracy_score, y_test_main, y_pred_class)
    matched_recall = try_metric(mtr.recall_score, y_test_matched, y_pred_class_matched, average=average, labels=labels)
    recall = try_metric(mtr.recall_score, y_test_main, y_pred_class, average=average, labels=labels)

    if multiclass:
        tn = fp = fn = tp = matched_specificity = specificity = "n/a"
    else:
        matched_specificity = try_metric(mtr.recall_score, y_test_matched, y_pred_class_matched,
                                         average=average, pos_label=0)
        specificity = try_metric(mtr.recall_score, y_test_main, y_pred_class, average=average, pos_label=0)
        try:
            conf_matrix = mtr.confusion_matrix(y_test_matched, y_pred_class_matched).ravel()
            if len(conf_matrix) == 4:
                tn, fp, fn, tp = conf_matrix
            elif len(conf_matrix) == 1:
                tn, fp, fn, tp = 0, 0, 0, 0
                all_y_test = np.all(y_test_matched)
                all_y_pred = np.all(y_pred_class_matched)
                if all_y_test and all_y_pred:
                    tp = conf_matrix[0]
                elif all_y_test and not all_y_pred:
                    fn = conf_matrix[0]
                elif not all_y_test and all_y_pred:
                    fp = conf_matrix[0]
                elif not all_y_test and not all_y_pred:
                    tn = conf_matrix[0]
                else:
                    raise ValueError("Unrecognized confusion matrix structure.")
            else:
                raise ValueError("Unrecognized confusion matrix structure.")
        except Exception as err:
            print(colored("confusion_matrix: " + str(err), "red"))
            tn, fp, fn, tp = np.nan, np.nan, np.nan, np.nan

    if target_class is not None:
        precision = precision[target_class]
        matched_recall = matched_recall[target_class]
        recall = recall[target_class]
        aupr = aupr[target_class]
        if not multiclass:
            matched_specificity = matched_specificity[target_class]
            specificity = specificity[target_class]

    # Save the results
    target_class = "*" if target_class is None else target_class
    with open("{}-metrics.csv".format(evalconfig.name_prefix), 'a', newline="") as csv_file:
        file_writer = csv.writer(csv_file)
        file_writer.writerow((n_epoch, dataset_name, target_class, tp, tn, fp, fn, int(missing), log_loss, matched_acc,
                              matched_recall, matched_specificity, acc, auroc,
                              aupr, precision, recall, specificity, mcc, prediction_rate))
    if evalconfig.do_plots:
        if multiclass:
            print(colored("ROC and PR curve plots possible for binary classification only.", "yellow"))
            return
        if np.any(np.isnan(y_pred)):
            print(colored("ROC and PR curve plots possible for complete data only (no NaNs).", "yellow"))
            return

        if not np.isnan(auroc):
            fpr, tpr, thresholds = mtr.roc_curve(y_test, y_pred)
            scores = tpr-fpr
            best_score = np.max(scores)
            best_thresh = thresholds[scores == best_score]
            print(f"Best balanced acc.: {(best_score+1)/2}. Optimal threshold(s): {best_thresh}.")
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
            bad_thresholds = np.logical_and(np.isclose(0.0, precision), np.isclose(0.0, recall))
            scores = np.zeros(recall.shape[0])
            if np.any(bad_thresholds):
                print('Found thresholds with zero precision and recall. Assuming F1=0.')
                gt = np.logical_not(bad_thresholds)
                scores[gt] = 2 * (precision[gt] * recall[gt]) / (precision[gt] + recall[gt])
            else:
                scores = 2 * (precision * recall) / (precision + recall)
            scores = scores[:thresholds.shape[0]]
            best_score = np.max(scores)
            best_thresh = thresholds[scores == best_score]
            print(f"Best f1: {best_score}. Optimal threshold(s): {best_thresh}.")
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
