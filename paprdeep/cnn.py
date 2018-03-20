import numpy as np
np.random.seed(0)
import tensorflow as tf
tf.set_random_seed(0)
import os
os.environ['PYTHONHASHSEED'] = '0'
import random as rn
rn.seed(0)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
import keras.backend as K
from Bio import SeqIO
from paprdeep.io import PaPrSequence, CSVMemoryLogger

def main():
    
    batch_loading_workers = 4
    batch_queue = batch_loading_workers * 10
    intra_threads = 90
    inter_threads = intra_threads
    num_cpus = intra_threads

    config = tf.ConfigProto(intra_op_parallelism_threads=intra_threads, inter_op_parallelism_threads=inter_threads, \
                            allow_soft_placement=True, device_count = {'CPU': num_cpus})
    session = tf.Session(config=config)
    K.set_session(session)

    seq_length = 250
    alphabet = "ACGT"
    seq_dim = len(alphabet)
    n_epochs = 12
    batch_size = 512
    hidden_dims = 256
    filter_size = 15
    n_filters = 512
    drop_out = 0.5
    class_weight = None
    csv_logger = CSVMemoryLogger('training-10e7-new-fold1.csv', append=True)   
    
    ### Load ###
    print("Loading...")
    x_train = np.load("SCRATCH_NOBAK/train_data_1_10e7.npy", mmap_mode='r')
    x_test = np.load("SCRATCH_NOBAK/test_data_1_bal.npy", mmap_mode='r')
    y_train = np.load("SCRATCH_NOBAK/train_labels_1_10e7.npy", mmap_mode='r')
    y_test = np.load("SCRATCH_NOBAK/test_labels_1_bal.npy", mmap_mode='r')
    training_sequence = PaPrSequence(x_train, y_train, batch_size)
    validation_sequence = PaPrSequence(x_test, y_test, batch_size)
    length_train=len(x_train)
    length_test=len(x_test)


    ### Build ###
    print("Building model...")
    model = Sequential()
    model.add(Conv1D(n_filters, filter_size, activation='relu', padding='same', input_shape=(seq_length, seq_dim)))

    model.add(GlobalMaxPooling1D())

    model.add(Dense(hidden_dims))
    model.add(Dropout(drop_out))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
                  
    for i in range(0, n_epochs):
        model.fit_generator(generator = training_sequence,
                        steps_per_epoch = length_train//batch_size,
                        epochs = i+1,
                        callbacks = [csv_logger],
                        validation_data = validation_sequence,
                        validation_steps = length_test//batch_size,                        
                        class_weight = class_weight,
                        max_queue_size = batch_queue,
                        workers = batch_loading_workers,
                        use_multiprocessing = True,
                        initial_epoch = i)
        model.save("cnn-10e7-new-fold1-e{n:02d}.h5".format(n=i))

if __name__ == "__main__":
    main()
