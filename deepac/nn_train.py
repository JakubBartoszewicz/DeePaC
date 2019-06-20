"""@package deepac.nn_train
Train a NN on Illumina reads.

Requires a config file describing the available devices, data loading mode, input sequence length, network architecture,
paths to input files and how should be the model trained.

"""

import numpy as np
import tensorflow as tf
import os

import errno
import warnings
from contextlib import redirect_stdout

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input, Lambda
from keras.layers import concatenate, add, multiply, average, maximum
from keras.layers import CuDNNLSTM, LSTM, Bidirectional
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
import keras.backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import plot_model
from keras import regularizers
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_uniform, he_uniform, orthogonal

from deepac.utils import ModelMGPU, ReadSequence, CSVMemoryLogger


class RCConfig:

    """
    RCNet configuration class.

    """

    def __init__(self, config):
        """RCConfig constructor"""
        # Devices Config #
        # Get the number of available GPUs
        self.n_gpus = config['Devices'].getint('N_GPUs')
        self.n_cpus = config['Devices'].getint('N_CPUs')
        self.multi_gpu = True if self.n_gpus > 1 else False
        self.allow_growth = config['Devices'].getboolean('AllowGrowth')
        self.device_parallel = config['Devices'].getboolean('DeviceParallel') and self.multi_gpu
        if self.device_parallel:
            self.device_fwd = config['Devices']['Device_fwd']
            self.device_rc = config['Devices']['Device_rc']

        self.model_build_device = config['Devices']['Device_build']

        # Data Loading Config #
        # If using generators to load data batch by batch, set up the number of batch workers and the queue size
        self.use_generators_train = config['DataLoad'].getboolean('LoadTrainingByBatch')
        self.use_generators_val = config['DataLoad'].getboolean('LoadValidationByBatch')
        if self.use_generators_train or self.use_generators_val:
            self.batch_loading_workers = config['DataLoad'].getint('BatchWorkers')
            self.batch_queue = config['DataLoad'].getint('BatchQueue')

        # Input Data Config #
        # Set the sequence length and the alphabet
        self.seq_length = config['InputData'].getint('SeqLength')
        self.alphabet = "ACGT"
        self.seq_dim = len(self.alphabet)

        # Architecture Config #
        # Set the seed
        self.seed = config['Architecture'].getint('Seed')
        # Advanced activations (e.g PReLUs) are not implemented yet
        self.adv_activations = config['Architecture'].getboolean('AdvancedActivations')
        if self.adv_activations:
            raise NotImplementedError('Advanced activations not implemented yet')
        # Set the initializer (choose between He and Glorot uniform)
        self.init_mode = config['Architecture']['WeightInit']
        if self.init_mode == 'he_uniform':
            self.initializer = he_uniform(self.seed)
        elif self.init_mode == 'glorot_uniform':
            self.initializer = glorot_uniform(self.seed)
        else:
            raise ValueError('Unknown initializer')

        # Define the network architecture
        self.rc_mode = config['Architecture']['RC_Mode']
        self.n_conv = config['Architecture'].getint('N_Conv')
        self.n_recurrent = config['Architecture'].getint('N_Recurrent')
        self.n_dense = config['Architecture'].getint('N_Dense')
        self.input_dropout = config['Architecture'].getfloat('Input_Dropout')
        self.conv_units = [int(u) for u in config['Architecture']['Conv_Units'].split(',')]
        self.conv_filter_size = [int(s) for s in config['Architecture']['Conv_FilterSize'].split(',')]
        self.conv_activation = config['Architecture']['Conv_Activation']
        self.conv_bn = config['Architecture'].getboolean('Conv_BN')
        self.conv_pooling = config['Architecture']['Conv_Pooling']
        self.conv_dropout = config['Architecture'].getfloat('Conv_Dropout')
        self.recurrent_units = [int(u) for u in config['Architecture']['Recurrent_Units'].split(',')]
        self.recurrent_bn = config['Architecture'].getboolean('Recurrent_BN')
        if self.n_recurrent == 1 and self.recurrent_bn:
            raise ValueError("RC-BN is intended for RC layers with 2D output. Use RC-Conv1D or RC-LSTM returning"
                             " sequences.")
        self.recurrent_dropout = config['Architecture'].getfloat('Recurrent_Dropout')
        merge_dict = {
            # motif on fwd fuzzy OR rc (Goedel t-conorm)
            "maximum": maximum,
            # motif on fwd fuzzy AND rc (product t-norm)
            "multiply": multiply,
            # motif on fwd PLUS/"OR" rc (Shrikumar-style)
            "add": add,
            # motif on fwd PLUS/"OR" rc (Shrikumar-style), rescaled
            "average": average
        }
        if self.rc_mode != "none":
            self.dense_merge = merge_dict.get(config['Architecture']['Dense_Merge'])
            if self.dense_merge is None:
                raise ValueError('Unknown dense merge function')
        self.dense_units = [int(u) for u in config['Architecture']['Dense_Units'].split(',')]
        self.dense_activation = config['Architecture']['Dense_Activation']
        self.dense_bn = config['Architecture'].getboolean('Dense_BN')
        self.dense_dropout = config['Architecture'].getfloat('Dense_Dropout')

        # If needed, weight classes
        self.use_weights = config['ClassWeights'].getboolean('UseWeights')
        if self.use_weights:
            class_count_0 = config['ClassWeights'].getfloat('ClassCount_0')
            class_count_1 = config['ClassWeights'].getfloat('ClassCount_1')
            sum_count = class_count_0 + class_count_1
            self.class_weight = {0: sum_count/(2*class_count_0),
                                 1: sum_count/(2*class_count_1)}
        else:
            self.class_weight = None

        # Paths Config #
        # Set the input data paths
        self.x_train_path = config['Paths']['TrainingData']
        self.y_train_path = config['Paths']['TrainingLabels']
        self.x_val_path = config['Paths']['ValidationData']
        self.y_val_path = config['Paths']['ValidationLabels']
        # Set the run name
        self.runname = config['Paths']['RunName']

        # Training Config #
        # Set the number op epochs, batch size and the optimizer
        self.epoch_start = config['Training'].getint('EpochStart') - 1
        self.epoch_end = config['Training'].getint('EpochEnd') - 1
        self.batch_size = config['Training'].getint('BatchSize')

        self.patience = config['Training'].getint('Patience')
        self.l2 = config['Training'].getfloat('Lambda_L2')
        self.regularizer = regularizers.l2(self.l2)
        self.learning_rate = config['Training'].getfloat('LearningRate')
        self.optimization_method = config['Training']['Optimizer']
        if self.optimization_method == "adam":
            self.optimizer = Adam(lr=self.learning_rate)
        else:
            warnings.warn("Custom learning rates implemented for Adam only. Using default Keras learning rate.")
            self.optimizer = self.optimization_method
        # If needed, log the memory usage
        self.log_memory = config['Training'].getboolean('MemUsageLog')
        self.summaries = config['Training'].getboolean('Summaries')
        self.log_superpath = config['Training']['LogPath']
        self.log_dir = self.log_superpath + "/{runname}-logs".format(runname=self.runname)

        self.use_tb = config['Training'].getboolean('Use_TB')
        if self.use_tb:
            self.tb_hist_freq = config['Training'].getint('TBHistFreq')

    def set_tf_session(self):
        """Set TF session."""
        # If no GPUs, use CPUs
        if self.n_gpus == 0:
            # Use as many intra_threads as the CPUs available
            intra_threads = self.n_cpus
            # Same for inter_threads
            inter_threads = intra_threads

            tf_config = tf.ConfigProto(intra_op_parallelism_threads=intra_threads,
                                       inter_op_parallelism_threads=inter_threads,
                                       allow_soft_placement=True, device_count={'CPU': self.n_cpus})
            session = tf.Session(config=tf_config)
            K.set_session(session)
            self.model_build_device = '/cpu:0'
        elif self.allow_growth:
            # If using GPUs, allow for GPU memory growth, instead of reserving it all
            tf_config = tf.ConfigProto(allow_soft_placement=True)
            tf_config.gpu_options.allow_growth = True
            session = tf.Session(config=tf_config)
            K.set_session(session)


class RCNet:

    """
    Reverse-complement neural network class.

    """

    def __init__(self, config):
        """RCNet constructor and config parsing"""
        self.config = config
        self.history = None

        self.training_sequence = None
        self.x_train = None
        self.y_train = None
        self.length_train = 0
        self.val_indices = None
        self.x_val = None
        self.y_val = None
        self.validation_data = (self.x_val, self.y_val)
        self.length_val = 0
        self.model = None
        self.parallel_model = None

        try:
            os.makedirs(self.config.log_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        self.__set_callbacks()
        if K.backend() == 'tensorflow':
            # Build the model using the CPU or GPU
            with tf.device(self.config.model_build_device):
                if self.config.rc_mode == "full":
                    self.__build_rc_model()
                elif self.config.rc_mode == "siam":
                    self.__build_siam_model()
                elif self.config.rc_mode == "none":
                    self.__build_simple_model()
                else:
                    raise ValueError('Unrecognized RC mode')
        elif K.backend() != 'tensorflow' and self.config.n_gpus > 1:
            raise NotImplementedError('Keras team recommends multi-gpu training with tensorflow')
        else:
            if self.config.use_rc:
                self.__build_rc_model()
            else:
                self.__build_simple_model()

    def load_data(self):
        """Load datasets"""
        print("Loading...")

        if self.config.use_generators_train:
            # Prepare the generators for loading data batch by batch
            self.x_train = np.load(self.config.x_train_path, mmap_mode='r')
            self.y_train = np.load(self.config.y_train_path, mmap_mode='r')
            self.training_sequence = ReadSequence(self.x_train, self.y_train, self.config.batch_size)
            self.length_train = len(self.x_train)
        else:
            # ... or load all the data to memory
            self.x_train = np.load(self.config.x_train_path)
            self.y_train = np.load(self.config.y_train_path)
            self.length_train = self.x_train.shape
        if self.config.use_generators_val:
            # Prepare the generators for loading data batch by batch
            self.x_val = np.load(self.config.x_val_path, mmap_mode='r')
            self.y_val = np.load(self.config.y_val_path, mmap_mode='r')
            self.validation_data = ReadSequence(self.x_val, self.y_val, self.config.batch_size)
            self.length_val = len(self.x_val)
        else:
            # ... or load all the data to memory
            self.x_val = np.load(self.config.x_val_path)
            self.y_val = np.load(self.config.y_val_path)
            self.val_indices = np.arange(len(self.y_val))
            np.random.shuffle(self.val_indices)
            self.x_val = self.x_val[self.val_indices]
            self.y_val = self.y_val[self.val_indices]
            self.validation_data = (self.x_val, self.y_val)
            self.length_val = self.x_val.shape[0]

    def __add_lstm(self, inputs, return_sequences):
        # LSTM with sigmoid activation corresponds to the CuDNNLSTM
        if self.config.n_gpus > 0:
            x = Bidirectional(CuDNNLSTM(self.config.recurrent_units[0], kernel_initializer=self.config.initializer,
                                        recurrent_initializer=orthogonal(self.config.seed),
                                        kernel_regularizer=self.config.regularizer,
                                        return_sequences=return_sequences))(inputs)
        else:
            x = Bidirectional(LSTM(self.config.recurrent_units[0], kernel_initializer=self.config.initializer,
                                   recurrent_initializer=orthogonal(self.config.seed),
                                   kernel_regularizer=self.config.regularizer,
                                   return_sequences=return_sequences,
                                   recurrent_activation='sigmoid'))(inputs)
        return x

    def __add_siam_lstm(self, inputs_fwd, inputs_rc, return_sequences, units):
        # LSTM with sigmoid activation corresponds to the CuDNNLSTM
        if self.config.n_gpus > 0:
            shared_lstm = Bidirectional(CuDNNLSTM(units,
                                                  kernel_initializer=self.config.initializer,
                                                  recurrent_initializer=orthogonal(self.config.seed),
                                                  kernel_regularizer=self.config.regularizer,
                                                  return_sequences=return_sequences))
        else:
            shared_lstm = Bidirectional(LSTM(units, kernel_initializer=self.config.initializer,
                                             recurrent_initializer=orthogonal(self.config.seed),
                                             kernel_regularizer=self.config.regularizer,
                                             return_sequences=return_sequences,
                                             recurrent_activation='sigmoid'))
        if self.config.device_parallel:
            with tf.device(self.config.device_fwd):
                x_fwd = shared_lstm(inputs_fwd)
            # Process the next sequence on another GPU
            with tf.device(self.config.device_rc):
                x_rc = shared_lstm(inputs_rc)
        else:
            x_fwd = shared_lstm(inputs_fwd)
            x_rc = shared_lstm(inputs_rc)
        if return_sequences:
            rev_axes = (1, 2)
        else:
            rev_axes = 1
        revcomp_out = Lambda(lambda x: K.reverse(x, axes=rev_axes), output_shape=shared_lstm.output_shape[1:],
                             name="reverse_lstm_output_{n}".format(n=self.__current_recurrent+1))
        x_rc = revcomp_out(x_rc)
        return x_fwd, x_rc

    def __add_rc_lstm(self, inputs, return_sequences, units):
        revcomp_in = Lambda(lambda x: K.reverse(x, axes=(1, 2)), output_shape=inputs._keras_shape[1:],
                            name="reverse_complement_lstm_input_{n}".format(n=self.__current_recurrent+1))
        inputs_rc = revcomp_in(inputs)
        x_fwd, x_rc = self.__add_siam_lstm(inputs, inputs_rc, return_sequences, units)
        out = concatenate([x_fwd, x_rc], axis=-1)
        return out

    def __add_siam_conv1d(self, inputs_fwd, inputs_rc, units):
        shared_conv = Conv1D(units, self.config.conv_filter_size[0], padding='same',
                             kernel_regularizer=self.config.regularizer)
        if self.config.device_parallel:
            with tf.device(self.config.device_fwd):
                x_fwd = shared_conv(inputs_fwd)
            # Process the next sequence on another GPU
            with tf.device(self.config.device_rc):
                x_rc = shared_conv(inputs_rc)
        else:
            x_fwd = shared_conv(inputs_fwd)
            x_rc = shared_conv(inputs_rc)
        revcomp_out = Lambda(lambda x: K.reverse(x, axes=(1, 2)), output_shape=shared_conv.output_shape[1:],
                             name="reverse_complement_conv1d_output_{n}".format(n=self.__current_conv+1))
        x_rc = revcomp_out(x_rc)
        return x_fwd, x_rc

    def __add_rc_conv1d(self, inputs, units):
        revcomp_in = Lambda(lambda x: K.reverse(x, axes=(1, 2)), output_shape=inputs._keras_shape[1:],
                            name="reverse_complement_conv1d_input_{n}".format(n=self.__current_conv+1))
        inputs_rc = revcomp_in(inputs)
        x_fwd, x_rc = self.__add_siam_conv1d(inputs, inputs_rc, units)
        out = concatenate([x_fwd, x_rc], axis=-1)
        return out

    def __add_siam_batchnorm(self, inputs_fwd, inputs_rc):
        input_shape = inputs_rc._keras_shape
        if len(input_shape) != 3:
            raise ValueError("Intended for RC layers with 2D output. Use RC-Conv1D or RC-LSTM returning sequences." 
                             "Expected dimension: 3, but got: " + str(len(input_shape)))
        rc_in = Lambda(lambda x: K.reverse(x, axes=(1, 2)), output_shape=input_shape[1:],
                       name="reverse_complement_batchnorm_input_{n}".format(n=self.__current_bn+1))
        inputs_rc = rc_in(inputs_rc)
        out = concatenate([inputs_fwd, inputs_rc], axis=1)
        out = BatchNormalization()(out)
        split_shape = out._keras_shape[1] // 2
        new_shape = [split_shape, input_shape[2]]
        fwd_out = Lambda(lambda x: x[:, :split_shape, :], output_shape=new_shape,
                         name="split_batchnorm_fwd_output_{n}".format(n=self.__current_bn+1))
        rc_out = Lambda(lambda x: K.reverse(x[:, split_shape:, :], axes=(1, 2)), output_shape=new_shape,
                        name="split_batchnorm_rc_output_{n}".format(n=self.__current_bn+1))
        if self.config.device_parallel:
            with tf.device(self.config.device_fwd):
                x_fwd = fwd_out(out)
            # Process the next sequence on another GPU
            with tf.device(self.config.device_rc):
                x_rc = rc_out(out)
        else:
            x_fwd = fwd_out(out)
            x_rc = rc_out(out)
        return x_fwd, x_rc

    def __add_rc_batchnorm(self, inputs):
        input_shape = inputs._keras_shape
        if len(input_shape) != 3:
            raise ValueError("Intended for RC layers with 2D output. Use RC-Conv1D or RC-LSTM returning sequences." 
                             "Expected dimension: 3, but got: " + str(len(input_shape)))
        split_shape = inputs._keras_shape[-1] // 2
        new_shape = [input_shape[1], split_shape]
        fwd_in = Lambda(lambda x: x[:, :, :split_shape], output_shape=new_shape,
                        name="split_batchnorm_fwd_input_{n}".format(n=self.__current_bn+1))
        rc_in = Lambda(lambda x: x[:, :, split_shape:], output_shape=new_shape,
                       name="split_batchnorm_rc_input_{n}".format(n=self.__current_bn+1))
        inputs_fwd = fwd_in(inputs)
        inputs_rc = rc_in(inputs)
        x_fwd, x_rc = self.__add_siam_batchnorm(inputs_fwd, inputs_rc)
        out = concatenate([x_fwd, x_rc], axis=-1)
        return out

    def __add_siam_merge_dense(self, inputs_fwd, inputs_rc, units, merge_function=add):
        shared_dense = Dense(units, kernel_regularizer=self.config.regularizer)
        rc_in = Lambda(lambda x: K.reverse(x, axes=1), output_shape=inputs_rc._keras_shape[1:],
                       name="reverse_merging_dense_input_{n}".format(n=1))
        inputs_rc = rc_in(inputs_rc)
        if self.config.device_parallel:
            with tf.device(self.config.device_fwd):
                x_fwd = shared_dense(inputs_fwd)
            # Process the next sequence on another GPU
            with tf.device(self.config.device_rc):
                x_rc = shared_dense(inputs_rc)
        else:
            x_fwd = shared_dense(inputs_fwd)
            x_rc = shared_dense(inputs_rc)
        out = merge_function([x_fwd, x_rc])
        return out

    def __add_rc_merge_dense(self, inputs, units, merge_function=add):
        split_shape = inputs._keras_shape[-1] // 2
        fwd_in = Lambda(lambda x: x[:, :split_shape], output_shape=[split_shape],
                        name="split_merging_dense_input_fwd_{n}".format(n=1))
        rc_in = Lambda(lambda x: x[:, split_shape:], output_shape=[split_shape],
                       name="split_merging_dense_input_rc_{n}".format(n=1))
        x_fwd = fwd_in(inputs)
        x_rc = rc_in(inputs)
        return self.__add_siam_merge_dense(x_fwd, x_rc, units, merge_function)

    def __build_simple_model(self):
        """Build the standard network"""
        print("Building model...")
        # Number of added recurrent layers
        self.__current_recurrent = 0
        # Initialize input
        inputs = Input(shape=(self.config.seq_length, self.config.seq_dim))
        # The last recurrent layer should return the output for the last unit only.
        # Previous layers must return output for all units
        return_sequences = True if self.config.n_recurrent > 1 else False
        # Input dropout
        if not np.isclose(self.config.input_dropout, 0.0):
            x = Dropout(self.config.input_dropout, seed=self.config.seed)(inputs)
        else:
            x = inputs
        # First convolutional/recurrent layer
        if self.config.n_conv > 0:
            # Convolutional layers will always be placed before recurrent ones
            # Standard convolutional layer
            x = Conv1D(self.config.conv_units[0], self.config.conv_filter_size[0], padding='same',
                       kernel_regularizer=self.config.regularizer)(x)
            if self.config.conv_bn:
                # Standard batch normalization layer
                x = BatchNormalization()(x)
            # Add activation
            x = Activation(self.config.conv_activation)(x)
        elif self.config.n_recurrent > 0:
            # If no convolutional layers, the first layer is recurrent.
            # CuDNNLSTM requires a GPU and tensorflow with cuDNN
            x = self.__add_lstm(x, return_sequences)
            if self.config.recurrent_bn and return_sequences:
                # Standard batch normalization layer
                x = BatchNormalization()(x)
            # Add dropout
            x = Dropout(self.config.recurrent_dropout, seed=self.config.seed)(x)
            # First recurrent layer already added
            self.__current_recurrent = 1
        else:
            raise ValueError('First layer should be convolutional or recurrent')

        # For next convolutional layers
        for i in range(1, self.config.n_conv):
            # Add pooling first
            if self.config.conv_pooling == 'max':
                x = MaxPooling1D()(x)
            elif self.config.conv_pooling == 'average':
                x = AveragePooling1D()(x)
            elif not (self.config.conv_pooling in ['last_max', 'last_average', 'none']):
                # Skip pooling if it should be applied to the last conv layer or skipped altogether.
                # Throw a ValueError if the pooling method is unrecognized.
                raise ValueError('Unknown pooling method')
            # Add dropout (drops whole features)
            if not np.isclose(self.config.conv_dropout, 0.0):
                x = Dropout(self.config.conv_dropout, seed=self.config.seed)(x)
            # Add layer
            # Standard convolutional layer
            x = Conv1D(self.config.conv_units[i], self.config.conv_filter_size[i], padding='same',
                       kernel_initializer=self.config.initializer, kernel_regularizer=self.config.regularizer)(x)
            # Add batch norm
            if self.config.conv_bn:
                # Standard batch normalization layer
                x = BatchNormalization()(x)
            # Add activation
            x = Activation(self.config.conv_activation)(x)

        # Pooling layer
        if self.config.n_conv > 0:
            if self.config.conv_pooling == 'max' or self.config.conv_pooling == 'last_max':
                if self.config.n_recurrent == 0:
                    # If no recurrent layers, use global pooling
                    x = GlobalMaxPooling1D()(x)
                else:
                    # for recurrent layers, use normal pooling
                    x = MaxPooling1D()(x)
            elif self.config.conv_pooling == 'average' or self.config.conv_pooling == 'last_average':
                if self.config.n_recurrent == 0:
                    # if no recurrent layers, use global pooling
                    x = GlobalAveragePooling1D()(x)
                else:
                    # for recurrent layers, use normal pooling
                    x = AveragePooling1D()(x)
            elif self.config.conv_pooling != 'none':
                # Skip pooling if needed or throw a ValueError if the pooling method is unrecognized
                # (should be thrown above)
                raise ValueError('Unknown pooling method')
            # Add dropout (drops whole features)
            if not np.isclose(self.config.conv_dropout, 0.0):
                x = Dropout(self.config.conv_dropout, seed=self.config.seed)(x)

        # Recurrent layers
        for i in range(self.__current_recurrent, self.config.n_recurrent):
            if i == self.config.n_recurrent - 1:
                # In the last layer, return output only for the last unit
                return_sequences = False
            # Add a bidirectional recurrent layer. CuDNNLSTM requires a GPU and tensorflow with cuDNN
            x = self.__add_lstm(inputs, return_sequences)
            if self.config.recurrent_bn and return_sequences:
                # Standard batch normalization layer
                x = BatchNormalization()(x)
            # Add dropout
            x = Dropout(self.config.recurrent_dropout, seed=self.config.seed)(x)

        # Dense layers
        for i in range(0, self.config.n_dense):
            x = Dense(self.config.dense_units[i],  kernel_regularizer=self.config.regularizer)(x)
            if self.config.dense_bn:
                # Standard batch normalization layer
                x = BatchNormalization()(x)
            x = Activation(self.config.dense_activation)(x)
            x = Dropout(self.config.dense_dropout, seed=self.config.seed)(x)

        # Output layer for binary classification
        x = Dense(1,  kernel_regularizer=self.config.regularizer)(x)
        x = Activation('sigmoid')(x)

        # Initialize the model
        self.model = Model(inputs, x)

    def __build_rc_model(self):
        """Build the RC network"""
        print("Building RC-model...")
        # Number of added recurrent layers
        self.__current_recurrent = 0
        self.__current_conv = 0
        self.__current_bn = 0
        # Initialize input
        inputs = Input(shape=(self.config.seq_length, self.config.seq_dim))
        # The last recurrent layer should return the output for the last unit only.
        #  Previous layers must return output for all units
        return_sequences = True if self.config.n_recurrent > 1 else False
        # Input dropout
        if not np.isclose(self.config.input_dropout, 0.0):
            x = Dropout(self.config.input_dropout, seed=self.config.seed)(inputs)
        else:
            x = inputs
        # First convolutional/recurrent layer
        if self.config.n_conv > 0:
            # Convolutional layers will always be placed before recurrent ones
            x = self.__add_rc_conv1d(x, self.config.conv_units[0])
            if self.config.conv_bn:
                # Reverse-complemented batch normalization layer
                x = self.__add_rc_batchnorm(x)
                self.__current_bn = self.__current_bn + 1
            x = Activation(self.config.conv_activation)(x)
            self.__current_conv = self.__current_conv + 1
        elif self.config.n_recurrent > 0:
            # If no convolutional layers, the first layer is recurrent.
            # CuDNNLSTM requires a GPU and tensorflow with cuDNN
            x = self.__add_rc_lstm(x, return_sequences, self.config.recurrent_units[0])
            if self.config.recurrent_bn and return_sequences:
                # Reverse-complemented batch normalization layer
                x = self.__add_rc_batchnorm(x)
                self.__current_bn = self.__current_bn + 1
            # Add dropout
            if not np.isclose(self.config.recurrent_dropout, 0.0):
                x = Dropout(self.config.recurrent_dropout, seed=self.config.seed)(x)
            # First recurrent layer already added
            self.__current_recurrent = self.__current_recurrent + 1
        else:
            raise ValueError('First layer should be convolutional or recurrent')

        # For next convolutional layers
        for i in range(1, self.config.n_conv):
            # Add pooling first
            if self.config.conv_pooling == 'max':
                x = MaxPooling1D()(x)
            elif self.config.conv_pooling == 'average':
                x = AveragePooling1D()(x)
            elif not (self.config.conv_pooling in ['last_max', 'last_average', 'none']):
                # Skip pooling if it should be applied to the last conv layer or skipped altogether.
                # Throw a ValueError if the pooling method is unrecognized.
                raise ValueError('Unknown pooling method')
            # Add dropout (drops whole features)
            if not np.isclose(self.config.conv_dropout, 0.0):
                x = Dropout(self.config.conv_dropout, seed=self.config.seed)(x)
            # Add layer
            x = self.__add_rc_conv1d(x, self.config.conv_units[i])
            if self.config.conv_bn:
                # Reverse-complemented batch normalization layer
                x = self.__add_rc_batchnorm(x)
                self.__current_bn = self.__current_bn + 1
            x = Activation(self.config.conv_activation)(x)
            self.__current_conv = self.__current_conv + 1

        # Pooling layer
        if self.config.n_conv > 0:
            if self.config.conv_pooling == 'max' or self.config.conv_pooling == 'last_max':
                if self.config.n_recurrent == 0:
                    # If no recurrent layers, use global pooling
                    x = GlobalMaxPooling1D()(x)
                else:
                    # for recurrent layers, use normal pooling
                    x = MaxPooling1D()(x)
            elif self.config.conv_pooling == 'average' or self.config.conv_pooling == 'last_average':
                if self.config.n_recurrent == 0:
                    # if no recurrent layers, use global pooling
                    x = GlobalAveragePooling1D()(x)
                else:
                    # for recurrent layers, use normal pooling
                    x = AveragePooling1D()(x)
            elif self.config.conv_pooling != 'none':
                # Skip pooling if needed or throw a ValueError if the pooling method is unrecognized
                # (should be thrown above)
                raise ValueError('Unknown pooling method')
            # Add dropout (drops whole features)
            if not np.isclose(self.config.conv_dropout, 0.0):
                x = Dropout(self.config.conv_dropout, seed=self.config.seed)(x)

        # Recurrent layers
        for i in range(self.__current_recurrent, self.config.n_recurrent):
            if i == self.config.n_recurrent - 1:
                # In the last layer, return output only for the last unit
                return_sequences = False
            # Add a bidirectional recurrent layer. CuDNNLSTM requires a GPU and tensorflow with cuDNN
            x = self.__add_rc_lstm(x, return_sequences, self.config.recurrent_units[i])
            if self.config.recurrent_bn and return_sequences:
                # Reverse-complemented batch normalization layer
                x = self.__add_rc_batchnorm(x)
                self.__current_bn = self.__current_bn + 1
            # Add dropout
            if not np.isclose(self.config.recurrent_dropout, 0.0):
                x = Dropout(self.config.recurrent_dropout, seed=self.config.seed)(x)
            self.__current_recurrent = self.__current_recurrent + 1

        # Dense layers
        for i in range(0, self.config.n_dense):
            if i == 0:
                x = self.__add_rc_merge_dense(x, self.config.dense_units[i])
            else:
                x = Dense(self.config.dense_units[i],  kernel_regularizer=self.config.regularizer)(x)
            if self.config.dense_bn:
                x = BatchNormalization()(x)
            x = Activation(self.config.dense_activation)(x)
            if not np.isclose(self.config.dense_dropout, 0.0):
                x = Dropout(self.config.dense_dropout, seed=self.config.seed)(x)

        # Output layer for binary classification
        if self.config.n_dense == 0:
            x = self.__add_rc_merge_dense(x, 1)
        else:
            x = Dense(1,  kernel_regularizer=self.config.regularizer)(x)
        x = Activation('sigmoid')(x)

        # Initialize the model
        self.model = Model(inputs, x)

    def __build_siam_model(self):
        """Build the RC network"""
        print("Building siamese RC-model...")
        # Number of added recurrent layers
        self.__current_recurrent = 0
        self.__current_conv = 0
        self.__current_bn = 0
        # Initialize input
        inputs_fwd = Input(shape=(self.config.seq_length, self.config.seq_dim))
        revcomp_in = Lambda(lambda _x: K.reverse(_x, axes=(1, 2)), output_shape=inputs_fwd._keras_shape[1:],
                            name="reverse_complement_input_{n}".format(n=self.__current_recurrent+1))
        inputs_rc = revcomp_in(inputs_fwd)
        # The last recurrent layer should return the output for the last unit only.
        # Previous layers must return output for all units
        return_sequences = True if self.config.n_recurrent > 1 else False
        # Input dropout
        if not np.isclose(self.config.input_dropout, 0.0):
            x_fwd = Dropout(self.config.input_dropout, seed=self.config.seed)(inputs_fwd)
            x_rc = Dropout(self.config.input_dropout, seed=self.config.seed)(inputs_rc)
        else:
            x_fwd = inputs_fwd
            x_rc = inputs_rc
        # First convolutional/recurrent layer
        if self.config.n_conv > 0:
            # Convolutional layers will always be placed before recurrent ones
            # Reverse-complement convolutional layer
            x_fwd, x_rc = self.__add_siam_conv1d(x_fwd, x_rc, self.config.conv_units[0])
            if self.config.conv_bn:
                # Reverse-complemented batch normalization layer
                x_fwd, x_rc = self.__add_siam_batchnorm(x_fwd, x_rc)
                self.__current_bn = self.__current_bn + 1
            # Add activation
            x_fwd = Activation(self.config.conv_activation)(x_fwd)
            x_rc = Activation(self.config.conv_activation)(x_rc)
            self.__current_conv = self.__current_conv + 1
        elif self.config.n_recurrent > 0:
            # If no convolutional layers, the first layer is recurrent.
            # CuDNNLSTM requires a GPU and tensorflow with cuDNN
            # RevComp input
            x_fwd, x_rc = self.__add_siam_lstm(x_fwd, x_rc, return_sequences, self.config.recurrent_units[0])
            # Add batch norm
            if self.config.recurrent_bn and return_sequences:
                # reverse-complemented batch normalization layer
                x_fwd, x_rc = self.__add_siam_batchnorm(x_fwd, x_rc)
                self.__current_bn = self.__current_bn + 1
            # Add dropout
            if not np.isclose(self.config.recurrent_dropout, 0.0):
                x_fwd = Dropout(self.config.recurrent_dropout, seed=self.config.seed)(x_fwd)
                x_rc = Dropout(self.config.recurrent_dropout, seed=self.config.seed)(x_rc)
            # First recurrent layer already added
            self.__current_recurrent = 1
        else:
            raise ValueError('First layer should be convolutional or recurrent')

        # For next convolutional layers
        for i in range(1, self.config.n_conv):
            # Add pooling first
            if self.config.conv_pooling == 'max':
                x_fwd = MaxPooling1D()(x_fwd)
                x_rc = MaxPooling1D()(x_rc)
            elif self.config.conv_pooling == 'average':
                x_fwd = AveragePooling1D()(x_fwd)
                x_rc = AveragePooling1D()(x_rc)
            elif not (self.config.conv_pooling in ['last_max', 'last_average', 'none']):
                # Skip pooling if it should be applied to the last conv layer or skipped altogether.
                # Throw a ValueError if the pooling method is unrecognized.
                raise ValueError('Unknown pooling method')
            # Add dropout (drops whole features)
            if not np.isclose(self.config.conv_dropout, 0.0):
                x_fwd = Dropout(self.config.conv_dropout, seed=self.config.seed)(x_fwd)
                x_rc = Dropout(self.config.conv_dropout, seed=self.config.seed)(x_rc)
            # Add layer
            # Reverse-complement convolutional layer
            x_fwd, x_rc = self.__add_siam_conv1d(x_fwd, x_rc, self.config.conv_units[i])
            if self.config.conv_bn:
                # Reverse-complemented batch normalization layer
                x_fwd, x_rc = self.__add_siam_batchnorm(x_fwd, x_rc)
                self.__current_bn = self.__current_bn + 1
            # Add activation
            x_fwd = Activation(self.config.conv_activation)(x_fwd)
            x_rc = Activation(self.config.conv_activation)(x_rc)
            self.__current_conv = self.__current_conv + 1

        # Pooling layer
        if self.config.n_conv > 0:
            if self.config.conv_pooling == 'max' or self.config.conv_pooling == 'last_max':
                if self.config.n_recurrent == 0:
                    # If no recurrent layers, use global pooling
                    x_fwd = GlobalMaxPooling1D()(x_fwd)
                    x_rc = GlobalMaxPooling1D()(x_rc)
                else:
                    # for recurrent layers, use normal pooling
                    x_fwd = MaxPooling1D()(x_fwd)
                    x_rc = MaxPooling1D()(x_rc)
            elif self.config.conv_pooling == 'average' or self.config.conv_pooling == 'last_average':
                if self.config.n_recurrent == 0:
                    # if no recurrent layers, use global pooling
                    x_fwd = GlobalAveragePooling1D()(x_fwd)
                    x_rc = GlobalAveragePooling1D()(x_rc)
                else:
                    # for recurrent layers, use normal pooling
                    x_fwd = AveragePooling1D()(x_fwd)
                    x_rc = AveragePooling1D()(x_rc)
            elif self.config.conv_pooling != 'none':
                # Skip pooling if needed or throw a ValueError if the pooling method is unrecognized
                # (should be thrown above)
                raise ValueError('Unknown pooling method')
            # Add dropout (drops whole features)
            if not np.isclose(self.config.conv_dropout, 0.0):
                x_fwd = Dropout(self.config.conv_dropout, seed=self.config.seed)(x_fwd)
                x_rc = Dropout(self.config.conv_dropout, seed=self.config.seed)(x_rc)

        # Recurrent layers
        for i in range(self.__current_recurrent, self.config.n_recurrent):
            if i == self.config.n_recurrent - 1:
                # In the last layer, return output only for the last unit
                return_sequences = False
            # Add a bidirectional recurrent layer. CuDNNLSTM requires a GPU and tensorflow with cuDNN
            x_fwd, x_rc = self.__add_siam_lstm(x_fwd, x_rc, return_sequences, self.config.recurrent_units[i])
            # Add batch norm
            if self.config.recurrent_bn and return_sequences:
                # Reverse-complemented batch normalization layer
                x_fwd, x_rc = self.__add_siam_batchnorm(x_fwd, x_rc)
                self.__current_bn = self.__current_bn + 1
            # Add dropout
            if not np.isclose(self.config.recurrent_dropout, 0.0):
                x_fwd = Dropout(self.config.recurrent_dropout, seed=self.config.seed)(x_fwd)
                x_rc = Dropout(self.config.recurrent_dropout, seed=self.config.seed)(x_rc)

        # Output layer for binary classification
        if self.config.n_dense == 0:
            # Output layer for binary classification
            x = self.__add_siam_merge_dense(x_fwd, x_rc, 1)
        else:
            # Dense layers
            x = self.__add_siam_merge_dense(x_fwd, x_rc, self.config.dense_units[0])
            if self.config.dense_bn:
                # Batch normalization layer
                x = BatchNormalization()(x)
            x = Activation(self.config.dense_activation)(x)
            x = Dropout(self.config.dense_dropout, seed=self.config.seed)(x)
            for i in range(1, self.config.n_dense):
                x = Dense(self.config.dense_units[i], kernel_regularizer=self.config.regularizer)(x)
                if self.config.dense_bn:
                    # Batch normalization layer
                    x = BatchNormalization()(x)
                x = Activation(self.config.dense_activation)(x)
                x = Dropout(self.config.dense_dropout, seed=self.config.seed)(x)
            # Output layer for binary classification
            x = Dense(1,  kernel_regularizer=self.config.regularizer)(x)
        x = Activation('sigmoid')(x)

        # Initialize the model
        self.model = Model(inputs_fwd, x)

    def compile_model(self):
        """Compile model and save model summaries"""
        print("Compiling...")
        # If using multiple GPUs, compile a parallel model for data parallelism.
        # Use a wrapper for the parallel model to use the ModelCheckpoint callback
        if self.config.multi_gpu and not self.config.device_parallel:
            self.parallel_model = ModelMGPU(self.model, gpus=self.config.n_gpus)
            self.parallel_model.compile(loss='binary_crossentropy',
                                        optimizer=self.config.optimizer,
                                        metrics=['accuracy'])
        else:
            self.model.compile(loss='binary_crossentropy',
                               optimizer=self.config.optimizer,
                               metrics=['accuracy'])

        # Print summary and plot model
        if self.config.summaries:
            with open(self.config.log_dir + "/summary-{runname}.txt".format(runname=self.config.runname), 'w') as f:
                with redirect_stdout(f):
                    self.model.summary()
            plot_model(self.model,
                       to_file=self.config.log_dir + "/plot-{runname}.png".format(runname=self.config.runname),
                       show_shapes=False, rankdir='TB')

    def __set_callbacks(self):
        """Set callbacks to use during training"""
        self.callbacks = []

        # Set early stopping
        self.callbacks.append(EarlyStopping(monitor="val_acc", patience=self.config.patience))

        # Add CSV callback with or without memory log
        if self.config.log_memory:
            self.callbacks.append(CSVMemoryLogger(
                self.config.log_dir + "/training-{runname}.csv".format(runname=self.config.runname),
                append=True))
        else:
            self.callbacks.append(CSVLogger(
                self.config.log_dir + "/training-{runname}.csv".format(runname=self.config.runname),
                append=True))
        # Save model after every epoch
        checkpoint_name = self.config.log_dir + "/nn-{runname}-".format(runname=self.config.runname)
        self.callbacks.append(ModelCheckpoint(filepath=checkpoint_name + "e{epoch:03d}.h5"))

        # Set TensorBoard
        if self.config.use_tb:
            self.callbacks.append(TensorBoard(
                log_dir=self.config.log_superpath + "/{runname}-tb".format(runname=self.config.runname),
                histogram_freq=self.config.tb_hist_freq, batch_size=self.config.batch_size,
                write_grads=True, write_images=True))

    def train(self):
        """Train the NN on Illumina reads using the supplied configuration."""
        print("Training...")
        if self.config.multi_gpu and not self.config.device_parallel:
            if self.config.use_generators_train:
                # Fit a parallel model using generators
                self.history = self.parallel_model.fit_generator(generator=self.training_sequence,
                                                                 epochs=self.config.epoch_end,
                                                                 callbacks=self.callbacks,
                                                                 validation_data=self.validation_data,
                                                                 class_weight=self.config.class_weight,
                                                                 max_queue_size=self.config.batch_queue,
                                                                 workers=self.config.batch_loading_workers,
                                                                 use_multiprocessing=True,
                                                                 initial_epoch=self.config.epoch_start)
            else:
                # Fit a parallel model using data in memory
                self.history = self.parallel_model.fit(x=self.x_train,
                                                       y=self.y_train,
                                                       batch_size=self.config.batch_size,
                                                       epochs=self.config.epoch_end,
                                                       callbacks=self.callbacks,
                                                       validation_data=self.validation_data,
                                                       shuffle=True,
                                                       class_weight=self.config.class_weight,
                                                       initial_epoch=self.config.epoch_start)
        else:
            if self.config.use_generators_train:
                # Fit a model using generators
                self.history = self.model.fit_generator(generator=self.training_sequence,
                                                        epochs=self.config.epoch_end,
                                                        callbacks=self.callbacks,
                                                        validation_data=self.validation_data,
                                                        class_weight=self.config.class_weight,
                                                        max_queue_size=self.config.batch_queue,
                                                        workers=self.config.batch_loading_workers,
                                                        use_multiprocessing=True,
                                                        initial_epoch=self.config.epoch_start)
            else:
                # Fit a model using data in memory
                self.history = self.model.fit(x=self.x_train,
                                              y=self.y_train,
                                              batch_size=self.config.batch_size,
                                              epochs=self.config.epoch_end,
                                              callbacks=self.callbacks,
                                              validation_data=self.validation_data,
                                              shuffle=True,
                                              class_weight=self.config.class_weight,
                                              initial_epoch=self.config.epoch_start)
