from keras.models import Sequential
from keras.models import load_model
from sklearn.metrics import roc_auc_score
import numpy as np
import time
import os
import tensorflow as tf
import h5py
import keras.backend as K

#intra_threads = 16
#inter_threads = intra_threads
#num_cpus = intra_threads

#config = tf.ConfigProto(intra_op_parallelism_threads=intra_threads, inter_op_parallelism_threads=inter_threads, \
#                        allow_soft_placement=True, device_count = {'CPU': num_cpus})
#session = tf.Session(config=config)
#K.set_session(session)

model = load_model("nn-fullD20384-e012.h5")

x_test = np.load("SCRATCH_NOBAK/test_1_data.npy")
y_test = np.load("SCRATCH_NOBAK/test_1_labels.npy")

start_time = time.time()
y_pred = model.predict(x_test)
end_time = time.time()
time = end_time - start_time
print(model.summary())
print(time)
print(time/len(y_test))