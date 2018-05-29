"""@package cnn_train
Train a CNN on Illumina reads.

Requires a config file describing the available devices, data loading mode, input sequence length, network architecture, paths to input files and how should be the model trained.

usage: cnn_train.py [-h] config_file

positional arguments:
  config_file

optional arguments:
  -h, --help   show this help message and exit
  
"""
# Set seeds at the very beginning for maximum reproducibility
seed = 0
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)
import os
os.environ['PYTHONHASHSEED'] = '0'
import random as rn
rn.seed(seed)

import sys
import argparse
import configparser

from Bio import SeqIO
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
import keras.backend as K
from keras.callbacks import CSVLogger
from keras.utils import multi_gpu_model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_uniform, he_uniform
from keras_contrib.callbacks.dead_relu_detector import DeadReluDetector
from paprdeep import PaPrSequence, CSVMemoryLogger

def main(argv):
    """Parse the config file and train the CNN on Illumina reads."""
    parser = argparse.ArgumentParser(description = "Train a CNN on Illumina reads.")
    parser.add_argument("config_file")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_file)
    train(config)
   
def train(config):
    """Train the CNN on Illumina reads using the supplied configuration."""
    ### Devices Config ###    
    # Get the number of available GPUs
    n_gpus = config['Devices'].getint('N_GPUs')
    multi_gpu = True if n_gpus > 1 else False
    
    # If no GPUs, use CPUs
    if n_gpus == 0:
        n_cpus = config['Devices'].getint('N_CPUs')
        # Use as many intra_threads as the CPUs available
        intra_threads = n_cpus
        # Same for inter_threads
        inter_threads = intra_threads

        tf_config = tf.ConfigProto(intra_op_parallelism_threads=intra_threads, inter_op_parallelism_threads=inter_threads, \
                                allow_soft_placement=True, device_count = {'CPU': n_cpus})
        session = tf.Session(config=tf_config)
        K.set_session(session)
    elif config['Devices'].getboolean('AllowGrowth'):
        # If using GPUs, allow for GPU memory growth, instead of reserving it all
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        session = tf.Session(config=tf_config)
        K.set_session(session)

    ### Data Loading Config ###
    # If using generators to load data batch by batch, set up the number of batch workers and the queue size
    use_generators = config['DataLoad'].getboolean('LoadByBatch')
    if use_generators:
        batch_loading_workers = config['DataLoad'].getint('BatchWorkers')
        batch_queue = config['DataLoad'].getint('BatchQueue')
    
    ### Input Data Config ###
    # Set the sequence length and the alphabet
    seq_length = config['InputData'].getint('SeqLength')
    alphabet = "ACGT"
    seq_dim = len(alphabet)
    
    ### Architecture Config ###
    # Set the seed
    seed = config['Architecture'].getint('Seed') 
    # Advanced activations (e.g PReLUs) are not implemented yet
    adv_activations = config['Architecture'].getboolean('AdvancedActivations')
    if adv_activations:
        raise NotImplementedError('Advanced activations not implemented yet')
    # Set the initializer (choose between He and Glorot uniform)
    init_mode = config['Architecture']['WeightInit']
    if init_mode == 'he_uniform':
        initializer = he_uniform(seed)
    elif init_mode == 'glorot_uniform':
        initializer = glorot_uniform(seed)
    else:
        raise ValueError('Unknown initializer')
    
    # Define the network architecture
    conv_1_units = config['Architecture'].getint('Conv_1_Units')  
    conv_1_filter_size = config['Architecture'].getint('Conv_1_FilterSize')
    conv_1_activation = config['Architecture']['Conv_1_Activation']
    conv_1_bn = config['Architecture']['Conv_1_BN']
    conv_1_pooling = config['Architecture']['Conv_1_Pooling']
    dense_1_units = config['Architecture'].getint('Dense_1_Units')
    dense_1_activation = config['Architecture']['Dense_1_Activation']    
    dense_1_bn = config['Architecture']['Dense_1_BN']
    drop_out = config['Architecture'].getfloat('Dropout')
    
    # If needed, weight classes
    use_weights = config['ClassWeights'].getboolean('UseWeights')
    if use_weights:
        class_weight = {0: config['Architecture'].getfloat('ClassWeight_0'),
                        1: config['Architecture'].getfloat('ClassWeight_1')}
    else:
        class_weight = None 
    
    ### Paths Config ###
    # Set the input data paths
    x_train_path = config['Paths']['TrainingData']
    y_train_path = config['Paths']['TrainingLabels']
    x_val_path = config['Paths']['ValidationData']
    y_val_path = config['Paths']['ValidationLabels']
    #Set the run name
    runname = config['Paths']['RunName']
    
    ### Training Config ###
    # Set the number op epochs, batch size and the optimizer
    n_epochs = config['Training'].getint('N_Epochs')
    batch_size = config['Training'].getint('BatchSize')
    optimizer = config['Training']['Optimizer']
    # If needed, log the memory usage
    if config['DataLoad'].getboolean('MemUsageLog'):
        csv_logger = CSVMemoryLogger("training-{runname}.csv".format(runname=runname), append=True)
    else:
        csv_logger = CSVLogger("training-{runname}.csv".format(runname=runname), append=True)
    
    ### Load ###
    print("Loading...")
    
    if use_generators:
        # Prepare the generators for loading data batch by batch
        x_train = np.load(x_train_path, mmap_mode='r')
        y_train = np.load(y_train_path, mmap_mode='r')
        x_val = np.load(x_val_path, mmap_mode='r')
        y_val = np.load(y_val_path, mmap_mode='r')
        training_sequence = PaPrSequence(x_train, y_train, batch_size)
        validation_sequence = PaPrSequence(x_val, y_val, batch_size)
        length_train = len(x_train)
        length_val = len(x_val)
    else:
        # ... or load all the data to memory
        x_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        x_val = np.load(x_val_path)
        y_val = np.load(y_val_path)
        length_train = x_train.shape
        length_val = x_val.shape
    
    # Prepare the dead relus detector
    dead_relu_callback = DeadReluDetector(x_train, verbose = True)
    ### Build ###
    print("Building model...")
    # Build the model using the CPU
    with tf.device('/cpu:0'):
        model = Sequential()
        
        # Convolutional layer
        model.add(Conv1D(conv_1_units, conv_1_filter_size, padding='same', input_shape=(seq_length, seq_dim), kernel_initializer = initializer))
        if conv_1_bn:
            model.add(BatchNormalization())
        model.add(Activation(conv_1_activation))
        
        # Pooling layer
        if conv_1_pooling == 'max':
            model.add(GlobalMaxPooling1D())
        elif conv_1_pooling == 'average':
            model.add(GlobalAveragePooling1D())
        else:
            raise ValueError('Unknown pooling method')
        
        # Dense layer
        model.add(Dense(dense_1_units, kernel_initializer = initializer))
        if dense_1_bn:
            model.add(BatchNormalization())
        model.add(Activation(dense_1_activation))
        model.add(Dropout(drop_out))
        
        # Output layer for binary classification
        model.add(Dense(1, kernel_initializer = initializer))
        model.add(Activation('sigmoid'))
    
    # If using multiple GPUs, compile a parallel model for data parallelism 
    if multi_gpu:
        parallel_model = multi_gpu_model(model, gpus=n_gpus) 
        parallel_model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])        
    else:
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
                  
    for i in range(0, n_epochs):
        if multi_gpu:
            if use_generators:
                # Fit a parallel model using generators
                parallel_model.fit_generator(generator = training_sequence,
                                             epochs = i+1,
                                             callbacks = [csv_logger],
                                             validation_data = validation_sequence,                       
                                             class_weight = class_weight,
                                             max_queue_size = batch_queue,
                                             workers = batch_loading_workers,
                                             use_multiprocessing = True,
                                             initial_epoch = i)
            else:
                # Fit a parallel model using data in memory
                parallel_model.fit(x = x_train,
                                   y = y_train,
                                   batch_size = batch_size,
                                   epochs = i+1,
                                   callbacks = [csv_logger],
                                   validation_data = (x_val, y_val),
                                   shuffle = True,
                                   class_weight = class_weight,
                                   initial_epoch = i)
        else:
            if use_generators: 
                # Fit a model using generators
                model.fit_generator(generator = training_sequence,
                                    epochs = i+1,
                                    callbacks = [csv_logger],
                                    validation_data = validation_sequence,                       
                                    class_weight = class_weight,
                                    max_queue_size = batch_queue,
                                    workers = batch_loading_workers,
                                    use_multiprocessing = True,
                                    initial_epoch = i)
            else:
                # Fit a model using data in memory
                model.fit(x = x_train,
                          y = y_train,
                          batch_size = batch_size,
                          epochs = i+1,
                          callbacks = [csv_logger],
                          validation_data = (x_val, y_val),
                          shuffle = True,
                          class_weight = class_weight,
                          initial_epoch = i)
        # The parallel model cannot be saved but shares weights with the template    
        model.save("cnn-{runname}-e{n:03d}.h5".format(runname=runname, n=i))
    
if __name__ == "__main__":
    main(sys.argv)
