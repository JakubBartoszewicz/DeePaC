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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random as rn
rn.seed(seed)

import sys
import argparse
import configparser
from contextlib import redirect_stdout
from math import isclose

from Bio import SeqIO
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import CuDNNLSTM, LSTM, Bidirectional
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
from keras.preprocessing.text import Tokenizer
import keras.backend as K
from keras.callbacks import CSVLogger
from keras.utils import multi_gpu_model, plot_model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_uniform, he_uniform, orthogonal
from keras_contrib.callbacks.dead_relu_detector import DeadReluDetector
from paprdeep_utils import PaPrSequence, CSVMemoryLogger

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
    model_build_device = '/cpu:0' if multi_gpu else '/device:GPU:0'

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
        model_build_device = '/cpu:0'
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
    n_conv = config['Architecture'].getint('N_Conv')
    n_recurrent = config['Architecture'].getint('N_Recurrent')
    n_dense = config['Architecture'].getint('N_Dense')
    conv_units = [int(u) for u in config['Architecture']['Conv_Units'].split(',')] 
    conv_filter_size = [int(s) for s in config['Architecture']['Conv_FilterSize'].split(',')]
    conv_activation = config['Architecture']['Conv_Activation']
    conv_bn = config['Architecture'].getboolean('Conv_BN')
    conv_pooling = config['Architecture']['Conv_Pooling']
    conv_drop_out = config['Architecture'].getfloat('Conv_Dropout') 
    recurrent_units = [int(u) for u in config['Architecture']['Recurrent_Units'].split(',')]
    recurrent_bn = config['Architecture'].getboolean('Recurrent_BN')
    recurrent_drop_out = config['Architecture'].getfloat('Recurrent_Dropout')  
    dense_units = [int(u) for u in config['Architecture']['Dense_Units'].split(',')]
    dense_activation = config['Architecture']['Dense_Activation']    
    dense_bn = config['Architecture'].getboolean('Dense_BN')
    dense_drop_out = config['Architecture'].getfloat('Dense_Dropout')
    
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
    with tf.device(model_build_device):
        # Initailize the model
        model = Sequential()
        # Number of added recurrent layers
        current_recurrent = 0
        # The last recurrent layer should return the output for the last unit only. Previous layers must return output for all units
        return_sequences = True if n_recurrent > 1 else False
        # First convolutional/recurrent layer
        if n_conv > 0:
            # Convolutional layers will always be placed before recurrent ones
            model.add(Conv1D(conv_units[0], conv_filter_size[0], padding='same', input_shape=(seq_length, seq_dim), kernel_initializer = initializer))
            if conv_bn:
                # Add batch norm
                model.add(BatchNormalization())
            # Add activation
            model.add(Activation(conv_activation))
        elif n_recurrent > 0:
            # If no convolutional layers, the first layer is recurrent. CuDNNLSTM requires a GPU and tensorflow with cuDNN
            if n_gpus > 0:
                model.add(Bidirectional(CuDNNLSTM(recurrent_units[0], kernel_initializer=initializer, recurrent_initializer=orthogonal(seed), return_sequences = return_sequences), input_shape=(seq_length, seq_dim)))
            else:
                model.add(Bidirectional(LSTM(recurrent_units[0], kernel_initializer=initializer, recurrent_initializer=orthogonal(seed), return_sequences = return_sequences), input_shape=(seq_length, seq_dim)))
            # Add batch norm
            if recurrent_bn:
                model.add(BatchNormalization())
            # Add dropout
            model.add(Dropout(recurrent_drop_out))
            # First recurrent layer already added
            current_recurrent = 1
        else:
            raise ValueError('Input layer should be convolutional or recurrent')
            
        # For next convolutional layers
        for i in range(1,n_conv):
            # Add pooling first
            if conv_pooling == 'max':
                model.add(MaxPooling1D())
            elif conv_pooling == 'average':
                model.add(AveragePooling1D())
            elif not (conv_pooling in ['last_max', 'last_average', 'none']):
                # Skip pooling if it should be applied to the last conv layer or skipped altogether. Throw a ValueError if the pooling method is unrecognized.
                raise ValueError('Unknown pooling method')
            # Add dropout (drops whole features)
            if not isclose(conv_drop_out, 0.0): 
               model.add(Dropout(conv_drop_out))
            # Add layer
            model.add(Conv1D(conv_units[i], conv_filter_size[i], padding='same', kernel_initializer = initializer))
            # Add batch norm
            if conv_bn:
                model.add(BatchNormalization())
            # Add activation
            model.add(Activation(conv_activation))
            
        # Pooling layer
        if n_conv > 0:
            if conv_pooling == 'max' or conv_pooling == 'last_max':
                if n_recurrent == 0:
                    # If no recurrent layers, use global pooling
                    model.add(GlobalMaxPooling1D())
                else:
                    # for recurrent layers, use normal pooling
                    model.add(MaxPooling1D())
            elif conv_pooling == 'average' or conv_pooling == 'last_average':
                if n_recurrent == 0:
                    # if no recurrent layers, use global pooling
                    model.add(GlobalAveragePooling1D())
                else:
                    # for recurrent layers, use normal pooling
                    model.add(AveragePooling1D())
            elif conv_pooling != 'none':
                # Skip pooling if needed or throw a ValueError if the pooling method is unrecognized (should be thrown above)
                raise ValueError('Unknown pooling method')
            # Add dropout (drops whole features)
            if not isclose(conv_drop_out, 0.0):
                model.add(Dropout(conv_drop_out))
        
        # Recurrent layers
        for i in range(current_recurrent, n_recurrent):
            if i == n_recurrent - 1:
                # In the last layer, return output only for the last unit
                return_sequences = False
            # Add a bidirectional recurrent layer. CuDNNLSTM requires a GPU and tensorflow with cuDNN
            if n_gpus > 0:
                model.add(Bidirectional(CuDNNLSTM(recurrent_units[i], kernel_initializer=initializer, recurrent_initializer=orthogonal(seed), return_sequences = return_sequences)))
            else:
                model.add(Bidirectional(LSTM(recurrent_units[i], kernel_initializer=initializer, recurrent_initializer=orthogonal(seed), return_sequences = return_sequences)))
            # Add batch norm
            if recurrent_bn:
                model.add(BatchNormalization())
            # Add dropout
            model.add(Dropout(dense_drop_out))
        
        # Dense layers
        for i in range(0, n_dense):
            model.add(Dense(dense_units[i], kernel_initializer = initializer))
            if dense_bn:
                model.add(BatchNormalization())
            model.add(Activation(dense_activation))
            model.add(Dropout(drop_out))
        
        # Output layer for binary classification
        model.add(Dense(1, kernel_initializer = initializer))
        model.add(Activation('sigmoid'))
    print("Compiling...") 
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
    
    # Print summary and plot model
    with open("summary-{runname}.txt".format(runname=runname), 'w') as f:
        with redirect_stdout(f):
            model.summary()
    plot_model(model, to_file = "plot-{runname}.png".format(runname=runname), show_shapes = True)
    
    print("Training...")              
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
        model.save("nn-{runname}-e{n:03d}.h5".format(runname=runname, n=i))
    
if __name__ == "__main__":
    main(sys.argv)
