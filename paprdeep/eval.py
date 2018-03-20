from keras.models import Sequential
from keras.models import load_model
import sklearn.metrics as mtr
import numpy as np
import csv

n_fold = 1
epoch_start = 0
epoch_end = 12
set = "test"
name_prefix = "cnn-10e7-new"

x_test = np.load("SCRATCH_NOBAK/{}_data_{}_25e5.npy".format(set, n_fold))
y_test = np.load("SCRATCH_NOBAK/{}_labels_{}_25e5.npy".format(set, n_fold))
    
with open("{}-fold{}-metrics-25e5.csv".format(name_prefix, n_fold), 'a',  newline="") as csv_file:
    file_writer = csv.writer(csv_file)
    file_writer.writerow(("epoch","log_loss","acc","auroc","mcc","f1","precision","recall"))

for n_epoch in range(epoch_start,epoch_end):
    model = load_model("{p}-fold{nf}-e{ne:02d}.h5".format(p=name_prefix, nf=n_fold, ne=n_epoch))

    y_pred =  np.ndarray.flatten(model.predict_proba(x_test))      
    y_pred_class = (y_pred > 0.5).astype('int32')
    np.save(file = "{p}-fold{nf}-e{ne:02d}-predictions-25e5.npy".format(p=name_prefix, nf=n_fold, ne=n_epoch), arr = y_pred)

    log_loss = mtr.log_loss(y_test, y_pred, eps=1e-07)
    acc = mtr.accuracy_score(y_test, y_pred_class)
    auroc = mtr.roc_auc_score(y_test, y_pred)
    mcc = mtr.matthews_corrcoef(y_test, y_pred_class)
    f1 = mtr.f1_score(y_test, y_pred_class)
    precision = mtr.precision_score(y_test, y_pred_class)
    recall = mtr.recall_score(y_test, y_pred_class)

    with open("{}-fold{}-metrics-25e5.csv".format(name_prefix, n_fold), 'a',  newline="") as csv_file:
        file_writer = csv.writer(csv_file)
        file_writer.writerow((n_epoch,log_loss,acc,auroc,mcc,f1,precision,recall))
