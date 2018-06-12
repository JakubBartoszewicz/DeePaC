"""@package nn_train
Train a NN on Illumina reads.

Requires a config file describing the available devices, data loading mode, input sequence length, network architecture, paths to input files and how should be the model trained.

usage: nn_train.py [-h] config_file

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
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import plot_model
from keras import regularizers
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_uniform, he_uniform, orthogonal
from paprdeep_utils import ModelMGPU, PaPrSequence, CSVMemoryLogger

def main(argv):
    """Parse the config file and train the NN on Illumina reads."""
    parser = argparse.ArgumentParser(description = "Train a NN on Illumina reads.")
    parser.add_argument("config_file")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_file)
    paprnet = PaPrNet(config)
    paprnet.train()

class PaPrConfig:
    
    """
    PaPrNet configuration class
    
    """
    
    def __init__(self, config):
        """PaPrConfig constuctor"""
        ### Devices Config ###    
        # Get the number of available GPUs
        self.n_gpus = config['Devices'].getint('N_GPUs')
        self.multi_gpu = True if self.n_gpus > 1 else False
        self.model_build_device = '/cpu:0' if self.multi_gpu else '/device:GPU:0'

        # If no GPUs, use CPUs
        if self.n_gpus == 0:
            self.n_cpus = config['Devices'].getint('N_CPUs')
            # Use as many intra_threads as the CPUs available
            intra_threads = self.n_cpus
            # Same for inter_threads
            inter_threads = intra_threads

            tf_config = tf.ConfigProto(intra_op_parallelism_threads=intra_threads, inter_op_parallelism_threads=inter_threads, \
                                    allow_soft_placement=True, device_count = {'CPU': n_cpus})
            session = tf.Session(config=tf_config)
            K.set_session(session)
            self.model_build_device = '/cpu:0'
        elif config['Devices'].getboolean('AllowGrowth'):
            # If using GPUs, allow for GPU memory growth, instead of reserving it all
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            session = tf.Session(config=tf_config)
            K.set_session(session)
            
        ### Data Loading Config ###
        # If using generators to load data batch by batch, set up the number of batch workers and the queue size
        self.use_generators = config['DataLoad'].getboolean('LoadByBatch')
        if self.use_generators:
            self.batch_loading_workers = config['DataLoad'].getint('BatchWorkers')
            self.batch_queue = config['DataLoad'].getint('BatchQueue')
        
        ### Input Data Config ###
        # Set the sequence length and the alphabet
        self.seq_length = config['InputData'].getint('SeqLength')
        self.alphabet = "ACGT"
        self.seq_dim = len(self.alphabet)
        
        ### Architecture Config ###
        # Set the seed
        self.seed = config['Architecture'].getint('Seed') 
        # Advanced activations (e.g PReLUs) are not implemented yet
        self.adv_activations = config['Architecture'].getboolean('AdvancedActivations')
        if self.adv_activations:
            raise NotImplementedError('Advanced activations not implemented yet')
        # Set the initializer (choose between He and Glorot uniform)
        self.init_mode = config['Architecture']['WeightInit']
        if self.init_mode == 'he_uniform':
            self.initializer = he_uniform(seed)
        elif init_mode == 'glorot_uniform':
            self.initializer = glorot_uniform(seed)
        else:
            raise ValueError('Unknown initializer')
        
        # Define the network architecture
        self.n_conv = config['Architecture'].getint('N_Conv')
        self.n_recurrent = config['Architecture'].getint('N_Recurrent')
        self.n_dense = config['Architecture'].getint('N_Dense')
        self.conv_units = [int(u) for u in config['Architecture']['Conv_Units'].split(',')] 
        self.conv_filter_size = [int(s) for s in config['Architecture']['Conv_FilterSize'].split(',')]
        self.conv_activation = config['Architecture']['Conv_Activation']
        self.conv_bn = config['Architecture'].getboolean('Conv_BN')
        self.conv_pooling = config['Architecture']['Conv_Pooling']
        self.conv_drop_out = config['Architecture'].getfloat('Conv_Dropout') 
        self.recurrent_units = [int(u) for u in config['Architecture']['Recurrent_Units'].split(',')]
        self.recurrent_bn = config['Architecture'].getboolean('Recurrent_BN')
        self.recurrent_drop_out = config['Architecture'].getfloat('Recurrent_Dropout')  
        self.dense_units = [int(u) for u in config['Architecture']['Dense_Units'].split(',')]
        self.dense_activation = config['Architecture']['Dense_Activation']    
        self.dense_bn = config['Architecture'].getboolean('Dense_BN')
        self.dense_drop_out = config['Architecture'].getfloat('Dense_Dropout')
        
        # If needed, weight classes
        self.use_weights = config['ClassWeights'].getboolean('UseWeights')
        if self.use_weights:
            self.class_weight = {0: config['Architecture'].getfloat('ClassWeight_0'),
                            1: config['Architecture'].getfloat('ClassWeight_1')}
        else:
            self.class_weight = None 
        
        ### Paths Config ###
        # Set the input data paths
        self.x_train_path = config['Paths']['TrainingData']
        self.y_train_path = config['Paths']['TrainingLabels']
        self.x_val_path = config['Paths']['ValidationData']
        self.y_val_path = config['Paths']['ValidationLabels']
        #Set the run name
        self.runname = config['Paths']['RunName']
        
        ### Training Config ###
        # Set the number op epochs, batch size and the optimizer
        self.epoch_start = config['Training'].getint('EpochStart')
        self.epoch_end = config['Training'].getint('EpochEnd')
        self.batch_size = config['Training'].getint('BatchSize')

        self.patience = config['Training'].getint('Patience')        
        self.l2 = config['Training'].getfloat('Lambda_L2')  
        self.regularizer = regularizers.l2(self.l2)      
        self.learning_rate = config['Training'].getfloat('LearningRate')
        self.optimization_method = config['Training']['Optimizer']
        if self.optimization_method == "adam":
            self.optimizer = Adam(lr=self.learning_rate)
        else:
            warn("Custom learning rates implemented for Adam only. Using default Keras learning rate.")
            self.optimizer = self.optimization_method
        # If needed, log the memory usage
        self.log_memory = True if config['DataLoad'].getboolean('MemUsageLog') else False
    
class PaPrNet:
    
    """
    Pathogenicity prediction network class.
    
    """
    
    def __init__(self, config):
        """PaPrNet constructor and config parsing"""
        self.config = PaPrConfig(config)
        self.__loadData()
        self.__setCallbacks()
        self.__buildSeqModel()
        self.__compileSeqModel()
        
    def __loadData(self):
        """Load datasets"""
        print("Loading...")
        
        if self.config.use_generators:
            # Prepare the generators for loading data batch by batch
            self.x_train = np.load(self.config.x_train_path, mmap_mode='r')
            self.y_train = np.load(self.config.y_train_path, mmap_mode='r')
            self.x_val = np.load(self.config.x_val_path, mmap_mode='r')
            self.y_val = np.load(self.config.y_val_path, mmap_mode='r')
            self.training_sequence = PaPrSequence(self.x_train, self.y_train, self.config.batch_size)
            self.validation_sequence = PaPrSequence(self.x_val, self.y_val, self.config.batch_size)
            self.length_train = len(self.x_train)
            self.length_val = len(self.x_val)
        else:
            # ... or load all the data to memory
            self.x_train = np.load(self.config.x_train_path)
            self.y_train = np.load(self.config.y_train_path)
            self.x_val = np.load(self.config.x_val_path)
            self.y_val = np.load(self.config.y_val_path)
            self.length_train = self.x_train.shape
            self.length_val = self.x_val.shape
        
    def __buildSeqModel(self):
        """Build the network"""
        print("Building model...")
        # Build the model using the CPU or GPU
        with tf.device(self.config.model_build_device):
            # Initailize the model
            self.model = Sequential()
            # Number of added recurrent layers
            current_recurrent = 0
            # The last recurrent layer should return the output for the last unit only. Previous layers must return output for all units
            return_sequences = True if self.config.n_recurrent > 1 else False
            # First convolutional/recurrent layer
            if self.config.n_conv > 0:
                # Convolutional layers will always be placed before recurrent ones
                self.model.add(Conv1D(self.config.conv_units[0], self.config.conv_filter_size[0], padding='same', kernel_initializer = self.config.initializer, kernel_regularizer=self.config.regularizer, input_shape=(self.config.seq_length, self.config.seq_dim)))
                if self.config.conv_bn:
                    # Add batch norm
                    self.model.add(BatchNormalization())
                # Add activation
                self.model.add(Activation(self.config.conv_activation))
            elif self.config.n_recurrent > 0:
                # If no convolutional layers, the first layer is recurrent. CuDNNLSTM requires a GPU and tensorflow with cuDNN
                if self.config.n_gpus > 0:
                    self.model.add(Bidirectional(CuDNNLSTM(self.config.recurrent_units[0], kernel_initializer=self.config.initializer, recurrent_initializer=orthogonal(self.config.seed), kernel_regularizer=self.config.regularizer, return_sequences = return_sequences), input_shape=(self.config.seq_length, self.config.seq_dim)))
                else:
                    self.model.add(Bidirectional(LSTM(self.config.recurrent_units[0], kernel_initializer=self.config.initializer, recurrent_initializer=orthogonal(self.config.seed), kernel_regularizer=self.config.regularizer, return_sequences = return_sequences), input_shape=(self.config.seq_length, self.config.seq_dim)))
                # Add batch norm
                if self.config.recurrent_bn:
                    self.model.add(BatchNormalization())
                # Add dropout
                self.model.add(Dropout(self.config.recurrent_drop_out))
                # First recurrent layer already added
                current_recurrent = 1
            else:
                raise ValueError('Input layer should be convolutional or recurrent')
                
            # For next convolutional layers
            for i in range(1,self.config.n_conv):
                # Add pooling first
                if self.config.conv_pooling == 'max':
                    self.model.add(MaxPooling1D())
                elif self.config.conv_pooling == 'average':
                    self.model.add(AveragePooling1D())
                elif not (self.config.conv_pooling in ['last_max', 'last_average', 'none']):
                    # Skip pooling if it should be applied to the last conv layer or skipped altogether. Throw a ValueError if the pooling method is unrecognized.
                    raise ValueError('Unknown pooling method')
                # Add dropout (drops whole features)
                if not isclose(self.config.conv_drop_out, 0.0): 
                   self.model.add(Dropout(self.config.conv_drop_out))
                # Add layer
                self.model.add(Conv1D(self.config.conv_units[i], self.config.conv_filter_size[i], padding='same', kernel_initializer = self.config.initializer, kernel_regularizer=self.config.regularizer))
                # Add batch norm
                if self.config.conv_bn:
                    self.model.add(BatchNormalization())
                # Add activation
                self.model.add(Activation(self.config.conv_activation))
                
            # Pooling layer
            if self.config.n_conv > 0:
                if self.config.conv_pooling == 'max' or self.config.conv_pooling == 'last_max':
                    if self.config.n_recurrent == 0:
                        # If no recurrent layers, use global pooling
                        self.model.add(GlobalMaxPooling1D())
                    else:
                        # for recurrent layers, use normal pooling
                        self.model.add(MaxPooling1D())
                elif self.config.conv_pooling == 'average' or self.config.conv_pooling == 'last_average':
                    if self.config.n_recurrent == 0:
                        # if no recurrent layers, use global pooling
                        self.model.add(GlobalAveragePooling1D())
                    else:
                        # for recurrent layers, use normal pooling
                        self.model.add(AveragePooling1D())
                elif self.config.conv_pooling != 'none':
                    # Skip pooling if needed or throw a ValueError if the pooling method is unrecognized (should be thrown above)
                    raise ValueError('Unknown pooling method')
                # Add dropout (drops whole features)
                if not isclose(self.config.conv_drop_out, 0.0):
                    self.model.add(Dropout(self.config.conv_drop_out))
            
            # Recurrent layers
            for i in range(current_recurrent, self.config.n_recurrent):
                if i == self.config.n_recurrent - 1:
                    # In the last layer, return output only for the last unit
                    return_sequences = False
                # Add a bidirectional recurrent layer. CuDNNLSTM requires a GPU and tensorflow with cuDNN
                if self.config.n_gpus > 0:
                    self.model.add(Bidirectional(CuDNNLSTM(self.config.recurrent_units[i], kernel_initializer=self.config.initializer, recurrent_initializer=orthogonal(self.config.seed), kernel_regularizer=self.config.regularizer, return_sequences = return_sequences)))
                else:
                    self.model.add(Bidirectional(LSTM(self.config.recurrent_units[i], kernel_initializer=self.config.initializer, recurrent_initializer=orthogonal(self.config.seed), kernel_regularizer=self.config.regularizer, return_sequences = return_sequences)))
                # Add batch norm
                if self.config.recurrent_bn:
                    self.model.add(BatchNormalization())
                # Add dropout
                self.model.add(Dropout(self.config.recurrent_drop_out))
            
            # Dense layers
            for i in range(0, self.config.n_dense):
                self.model.add(Dense(self.config.dense_units[i], kernel_initializer=self.config.initializer, kernel_regularizer=self.config.regularizer))
                if self.config.dense_bn:
                    self.model.add(BatchNormalization())
                self.model.add(Activation(self.config.dense_activation))
                self.model.add(Dropout(self.config.dense_drop_out))
            
            # Output layer for binary classification
            self.model.add(Dense(1, kernel_initializer=self.config.initializer, kernel_regularizer=self.config.regularizer))
            self.model.add(Activation('sigmoid'))
    
    def __compileSeqModel(self):
        """Compile model and save model summaries"""
        print("Compiling...") 
        # If using multiple GPUs, compile a parallel model for data parallelism. Use a wrapper for the parallell model to use the ModelCheckpoint callback
        if self.config.multi_gpu:
            self.parallel_model = ModelMGPU(model, gpus=n_gpus) 
            self.parallel_model.compile(loss='binary_crossentropy',
                                        optimizer=self.config.optimizer,
                                        metrics=['accuracy'])        
        else:
            self.model.compile(loss='binary_crossentropy',
                               optimizer=self.config.optimizer,
                               metrics=['accuracy'])
        
        # Print summary and plot model
        with open("summary-{runname}.txt".format(runname=self.config.runname), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()
        plot_model(self.model, to_file = "plot-{runname}.png".format(runname=self.config.runname), show_shapes = True)    
    
    def __setCallbacks(self):
        """Set callbacks to use during training"""
        self.callbacks=[]
        # Add CSV callback with or without memory log
        if self.config.log_memory:
            self.callbacks.append(CSVMemoryLogger("training-{runname}.csv".format(runname=sel.self.config.runname), append=True))
        else:
            self.callbacks.append(CSVLogger("training-{runname}.csv".format(runname=self.config.runname), append=True))
        # Save model after every epoch
        checkpoint_name = "nn-{runname}-".format(runname=self.config.runname)
        self.callbacks.append(ModelCheckpoint(filepath = checkpoint_name + "e{epoch:03d}.h5"))
        # Set early stopping
        self.callbacks.append(EarlyStopping(monitor="val_acc", patience = self.config.patience))
        # Set TensorBoard
        self.callbacks.append(TensorBoard(log_dir="./{runname}-logs".format(runname=self.config.runname), histogram_freq=0, batch_size = self.config.batch_size, write_grads=True, write_images=True))
    def train(self):
        """Train the NN on Illumina reads using the supplied configuration."""           
        print("Training...")
        if self.config.multi_gpu:
            if self.config.use_generators:
                # Fit a parallel model using generators
                self.history = self.parallel_model.fit_generator(generator = self.training_sequence,
                                                                 epochs = self.config.epoch_end,
                                                                 callbacks = self.callbacks,
                                                                 validation_data = self.validation_sequence,                       
                                                                 class_weight = self.config.class_weight,
                                                                 max_queue_size = self.config.batch_queue,
                                                                 workers = self.config.batch_loading_workers,
                                                                 use_multiprocessing = True,
                                                                 initial_epoch = self.config.epoch_start)
            else:
                # Fit a parallel model using data in memory
                self.history = self.parallel_model.fit(x = self.x_train,
                                                       y = self.y_train,
                                                       batch_size = self.config.batch_size,
                                                       epochs = self.config.epoch_end,
                                                       callbacks = self.callbacks,
                                                       validation_data = (self.x_val, self.y_val),
                                                       shuffle = True,
                                                       class_weight = self.config.class_weight,
                                                       initial_epoch = self.config.epoch_start)
        else:
            if self.config.use_generators: 
                # Fit a model using generators
                self.history = self.model.fit_generator(generator = self.training_sequence,
                                                        epochs = self.config.epoch_end,
                                                        callbacks = self.callbacks,
                                                        validation_data = self.validation_sequence,                       
                                                        class_weight = self.config.class_weight,
                                                        max_queue_size = self.config.batch_queue,
                                                        workers = self.config.batch_loading_workers,
                                                        use_multiprocessing = True,
                                                        initial_epoch = self.config.epoch_start)
            else:
                # Fit a model using data in memory
                self.history = self.model.fit(x = self.x_train,
                                              y = self.y_train,
                                              batch_size = self.config.batch_size,
                                              epochs = self.config.epoch_end,
                                              callbacks = self.callbacks,
                                              validation_data = (self.x_val, self.y_val),
                                              shuffle = True,
                                              class_weight = self.config.class_weight,
                                              initial_epoch = self.config.epoch_start)
                
if __name__ == "__main__":
    main(sys.argv)