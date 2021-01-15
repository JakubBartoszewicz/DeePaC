"""@package deepac.nn_train
Train a NN on Illumina reads.

Requires a config file describing the available devices, data loading mode, input sequence length, network architecture,
paths to input files and how should be the model trained.

"""

import numpy as np
import tensorflow as tf
import re
import os
import sys
import errno
import warnings
from contextlib import redirect_stdout
import math

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, Lambda, Masking
from tensorflow.keras.layers import concatenate, add, multiply, average, maximum, Flatten
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import orthogonal
from tensorflow.keras.models import load_model

from deepac.utils import ReadSequence, CSVMemoryLogger, set_mem_growth, DatasetParser


class RCConfig:

    """
    RCNet configuration class.

    """

    def __init__(self, config):
        """RCConfig constructor"""
        try:
            self.strategy_dict = {
                "MirroredStrategy": tf.distribute.MirroredStrategy,
                "OneDeviceStrategy": tf.distribute.OneDeviceStrategy,
                "CentralStorageStrategy": tf.distribute.experimental.CentralStorageStrategy,
                "MultiWorkerMirroredStrategy": tf.distribute.experimental.MultiWorkerMirroredStrategy,
                "TPUStrategy": tf.distribute.experimental.TPUStrategy,
            }

            # Devices Config #
            # Get the number of available GPUs
            try:
                self.strategy = config['Devices']['DistStrategy']
            except KeyError:
                print("Unknown distribution strategy. Using MirroredStrategy.")
                self.strategy = "MirroredStrategy"
            self._n_gpus = 0
            self.tpu_strategy = None

            # for using tf.device instead of strategy
            try:
                self.simple_build = config['Devices'].getboolean('Simple_build') if tf.executing_eagerly() else True
            except KeyError:
                self.simple_build = False if tf.executing_eagerly() else True
            self.base_batch_size = config['Training'].getint('BatchSize')
            self.batch_size = self.base_batch_size

            self.set_n_gpus()
            self.model_build_device = config['Devices']['Device_build']

            # Data Loading Config #
            # If using generators to load data batch by batch, set up the number of batch workers and the queue size
            self.use_generators_keras = config['DataLoad'].getboolean('LoadTrainingByBatch')
            self.use_tf_data = config['DataLoad'].getboolean('Use_TFData')
            self.multiprocessing = config['DataLoad'].getboolean('Multiprocessing')
            self.batch_loading_workers = config['DataLoad'].getint('BatchWorkers')
            self.batch_queue = config['DataLoad'].getint('BatchQueue')

            # Input Data Config #
            # Set the sequence length and the alphabet
            self.seq_length = config['InputData'].getint('SeqLength')
            self.alphabet = "ACGT"
            self.seq_dim = len(self.alphabet)
            try:
                self.mask_zeros = config['InputData'].getboolean('MaskZeros')
            except KeyError:
                self.mask_zeros = False
            # subread settings (subread = first k nucleotides of a read)
            self.use_subreads = config['InputData'].getboolean('UseSubreads')
            self.min_subread_length = config['InputData'].getint('MinSubreadLength')
            self.max_subread_length = config['InputData'].getint('MaxSubreadLength')
            self.dist_subread = config['InputData']['DistSubread']

            # Architecture Config #
            # Set the seed
            if config['Architecture']['Seed'] == "none" or config['Architecture']['Seed'] == "None":
                self.seed = None
            else:
                self.seed = config['Architecture'].getint('Seed')
            # Set the initializer (choose between He and Glorot uniform)
            self.init_mode = config['Architecture']['WeightInit']
            self._initializer_dict = {
                "he_uniform": tf.keras.initializers.he_uniform(self.seed),  # scale=2, mode=fan_in
                "glorot_uniform": tf.keras.initializers.glorot_uniform(self.seed),  # scale=1, mode=fan_avg
            }
            self.initializers = {}
            if self.init_mode == 'custom':
                self.initializers["conv"] = self._initializer_dict[config['Architecture']['WeightInit_Conv']]
                self.initializers["merge"] = self._initializer_dict[config['Architecture']['WeightInit_Merge']]
                self.initializers["lstm"] = self._initializer_dict[config['Architecture']['WeightInit_LSTM']]
                self.initializers["dense"] = self._initializer_dict[config['Architecture']['WeightInit_Dense']]
                self.initializers["out"] = self._initializer_dict[config['Architecture']['WeightInit_Out']]
            else:

                self.initializers["conv"] = self._initializer_dict[config['Architecture']['WeightInit']]
                self.initializers["merge"] = self._initializer_dict[config['Architecture']['WeightInit']]
                self.initializers["lstm"] = self._initializer_dict[config['Architecture']['WeightInit']]
                self.initializers["dense"] = self._initializer_dict[config['Architecture']['WeightInit']]
                self.initializers["out"] = self._initializer_dict[config['Architecture']['WeightInit']]
            self.ortho_gain = config['Architecture'].getfloat('OrthoGain')

            # Define the network architecture
            self.rc_mode = config['Architecture']['RC_Mode']
            self.n_conv = config['Architecture'].getint('N_Conv')
            try:
                self.skip_size = config['Architecture'].getint('Skip_Size')
            except KeyError:
                self.skip_size = 0
            self.n_recurrent = config['Architecture'].getint('N_Recurrent')
            self.n_dense = config['Architecture'].getint('N_Dense')
            self.input_dropout = config['Architecture'].getfloat('Input_Dropout')
            self.conv_units = [int(u) for u in config['Architecture']['Conv_Units'].split(',')]
            self.conv_filter_size = [int(s) for s in config['Architecture']['Conv_FilterSize'].split(',')]
            self.conv_dilation = [int(s) for s in config['Architecture']['Conv_Dilation'].split(',')]
            self.conv_stride = [int(s) for s in config['Architecture']['Conv_Stride'].split(',')]
            self.conv_activation = config['Architecture']['Conv_Activation']
            try:
                self.padding = config['Architecture']['Conv_Padding']
            except KeyError:
                self.padding = "same"
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
            try:
                self.mc_dropout = config['Architecture'].getboolean('MC_Dropout')
                self.dropout_training_mode = None if not self.mc_dropout else True
            except KeyError:
                self.mc_dropout = False
                self.dropout_training_mode = None

            # If needed, weight classes
            self.use_weights = config['ClassWeights'].getboolean('UseWeights')
            if self.use_weights:
                try:
                    counts = [float(x) for x in config['ClassWeights']['ClassCounts'].split(',')]
                except KeyError:
                    counts = [config['ClassWeights'].getfloat('ClassCount_0'),
                              config['ClassWeights'].getfloat('ClassCount_1')]
                sum_count = sum(counts)
                weights = [sum_count/(2*class_count) for class_count in counts]
                classes = range(len(counts))
                self.class_weight = dict(zip(classes, weights))
                self.log_init = False
                if self.log_init:
                    self.output_bias = tf.keras.initializers.Constant(np.log(counts[1]/counts[0]))
                else:
                    self.output_bias = 'zeros'
            else:
                self.class_weight = None
                self.output_bias = 'zeros'

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
            self.epoch_end = config['Training'].getint('EpochEnd')

            self.patience = config['Training'].getint('Patience')
            try:
                self.l1 = config['Training'].getfloat('Lambda_L1')
            except KeyError:
                self.l1 = 0.0
            self.l2 = config['Training'].getfloat('Lambda_L2')
            self.regularizer = regularizers.L1L2(self.l1, self.l2)
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
            self.log_dir = os.path.join(self.log_superpath, "{runname}-logs".format(runname=self.runname))

            self.use_tb = config['Training'].getboolean('Use_TB')
            if self.use_tb:
                self.tb_hist_freq = config['Training'].getint('TBHistFreq')
        except KeyError as ke:
            sys.exit("The config file is not compatible with this version of DeePaC. "
                     "Missing keyword: {}".format(ke))
        except AttributeError as ae:
            sys.exit("The config file is not compatible with this version of DeePaC. "
                     "Error: {}".format(ae))

    def set_tf_session(self):
        """Set TF session."""
        # If no GPUs, use CPUs
        if self._n_gpus == 0:
            self.model_build_device = '/cpu:0'
        set_mem_growth()

    def set_n_gpus(self):
        self._n_gpus = len(tf.config.get_visible_devices('GPU'))
        self.batch_size = self.base_batch_size * self._n_gpus if self._n_gpus > 0 else self.base_batch_size

    def get_n_gpus(self):
        return self._n_gpus

    def set_tpu_resolver(self, tpu_resolver):
        if tpu_resolver is not None:
            self.tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu_resolver)
            self.batch_size = self.base_batch_size * self.tpu_strategy.num_replicas_in_sync


class RCNet:

    """
    Reverse-complement neural network class.

    """

    def __init__(self, config, training_mode=True, verbose_load=False):
        """RCNet constructor and config parsing"""
        self.config = config
        if self.config.use_tf_data and not tf.executing_eagerly():
            warnings.warn("Training with TFRecordDatasets supported only in eager mode. Looking for .npy files...")
            self.config.use_tf_data = False

        self.config.set_tf_session()
        self.history = None
        self.verbose_load = verbose_load

        self._t_sequence = None
        self._v_sequence = None
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

        if training_mode:
            try:
                os.makedirs(self.config.log_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            self._set_callbacks()
            
        # Set strategy
        if self.config.tpu_strategy is not None:
            self.strategy = self.config.tpu_strategy
        elif self.config.simple_build:
            self.strategy = None
        elif self.config.strategy == "OneDeviceStrategy":
            self.strategy = self.config.strategy_dict[self.config.strategy](self.config.model_build_device)
        else:
            self.strategy = self.config.strategy_dict[self.config.strategy]()

        if float(tf.__version__[:3]) > 2.1 and self.config.epoch_start > 0:
            checkpoint_name = self.config.log_dir + "/{runname}-".format(runname=self.config.runname)
            model_file = checkpoint_name + "e{epoch:03d}.h5".format(epoch=self.config.epoch_start)
            print("Loading " + model_file)
            with self.get_device_strategy_scope():
                self.model = load_model(model_file)
        else:
            # Build the model using the CPU or GPU or TPU
            with self.get_device_strategy_scope():
                if self.config.rc_mode == "full":
                    self._build_rc_model()
                elif self.config.rc_mode == "siam":
                    self._build_siam_model()
                elif self.config.rc_mode == "none":
                    self._build_simple_model()
                else:
                    raise ValueError('Unrecognized RC mode')
            if self.config.epoch_start > 0:
                print("WARNING: loading a pre-trained model will reset the optimizer state. Please update to TF>=2.2.")
                checkpoint_name = self.config.log_dir + "/{runname}-".format(runname=self.config.runname)
                model_file = checkpoint_name + "e{epoch:03d}.h5".format(epoch=self.config.epoch_start)
                path = re.sub("\.h5$", "", model_file)
                weights_path = path + "_weights.h5"
                print("Loading " + weights_path)
                self.model.load_weights(weights_path)

    def get_device_strategy_scope(self):
        if self.config.simple_build:
            device_strategy_scope = tf.device(self.config.model_build_device)
        else:
            device_strategy_scope = self.strategy.scope()
        return device_strategy_scope

    def load_data(self):
        """Load datasets"""
        print("Loading...")

        if self.config.use_tf_data:
            prefetch_size = tf.data.experimental.AUTOTUNE

            def count_data_items(filenames):
                n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
                return np.max(n) + 1

            parser = DatasetParser(self.config.seq_length)
            train_filenames = tf.io.gfile.glob(self.config.x_train_path + "/*.tfrec")
            self.length_train = count_data_items(train_filenames)
            self.training_sequence = \
                parser.read_dataset(train_filenames).shuffle(buffer_size=self.config.batch_size*self.config.batch_queue)
            self.training_sequence = \
                self.training_sequence.repeat().batch(self.config.batch_size).prefetch(prefetch_size)

            val_filenames = tf.io.gfile.glob(self.config.x_val_path + "/*.tfrec")
            self.length_val = count_data_items(val_filenames)
            self.validation_data = \
                parser.read_dataset(val_filenames).repeat().batch(self.config.batch_size).prefetch(prefetch_size)
        elif self.config.use_generators_keras:
            # Prepare the generators for loading data batch by batch
            self.x_train = np.load(self.config.x_train_path, mmap_mode='r')
            self.y_train = np.load(self.config.y_train_path, mmap_mode='r')
            self._t_sequence = ReadSequence(self.x_train, self.y_train, self.config.batch_size,
                                            self.config.use_subreads, self.config.min_subread_length,
                                            self.config.max_subread_length, self.config.dist_subread,
                                            verbose_id="TRAIN" if self.verbose_load else None)

            self.training_sequence = self._t_sequence
            self.length_train = len(self.x_train)

            # Prepare the generators for loading data batch by batch
            self.x_val = np.load(self.config.x_val_path, mmap_mode='r')
            self.y_val = np.load(self.config.y_val_path, mmap_mode='r')
            self._v_sequence = ReadSequence(self.x_val, self.y_val, self.config.batch_size,
                                            self.config.use_subreads, self.config.min_subread_length,
                                            self.config.max_subread_length, self.config.dist_subread,
                                            verbose_id="VAL" if self.verbose_load else None)
            self.validation_data = self._v_sequence

            self.length_val = len(self.x_val)
        else:
            # ... or load all the data to memory
            self.x_train = np.load(self.config.x_train_path)
            self.y_train = np.load(self.config.y_train_path)
            self.length_train = self.x_train.shape

            # ... or load all the data to memory
            self.x_val = np.load(self.config.x_val_path)
            self.y_val = np.load(self.config.y_val_path)
            self.val_indices = np.arange(len(self.y_val))
            np.random.shuffle(self.val_indices)
            self.x_val = self.x_val[self.val_indices]
            self.y_val = self.y_val[self.val_indices]
            self.validation_data = (self.x_val, self.y_val)
            self.length_val = self.x_val.shape[0]

    def _add_lstm(self, inputs, return_sequences):
        # LSTM with sigmoid activation corresponds to the CuDNNLSTM
        if not tf.executing_eagerly() and (self.config.get_n_gpus() > 0
                                           and re.match("cpu", self.config.model_build_device, re.IGNORECASE) is None):
            x = Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(self.config.recurrent_units[0],
                                                                  kernel_initializer=self.config.initializers["lstm"],
                                                                  recurrent_initializer=orthogonal(
                                                                      gain=self.config.ortho_gain,
                                                                      seed=self.config.seed),
                                                                  kernel_regularizer=self.config.regularizer,
                                                                  return_sequences=return_sequences))(inputs)
        else:
            x = Bidirectional(LSTM(self.config.recurrent_units[0], kernel_initializer=self.config.initializers["lstm"],
                                   recurrent_initializer=orthogonal(gain=self.config.ortho_gain,
                                                                    seed=self.config.seed),
                                   kernel_regularizer=self.config.regularizer,
                                   return_sequences=return_sequences,
                                   recurrent_activation='sigmoid'))(inputs)
        return x

    def _add_siam_lstm(self, inputs_fwd, inputs_rc, return_sequences, units):
        # LSTM with sigmoid activation corresponds to the CuDNNLSTM
        if not tf.executing_eagerly() and (self.config.get_n_gpus() > 0
                                           and re.match("cpu", self.config.model_build_device, re.IGNORECASE) is None):
            shared_lstm = Bidirectional(
                tf.compat.v1.keras.layers.CuDNNLSTM(units,
                                                    kernel_initializer=self.config.initializers["lstm"],
                                                    recurrent_initializer=orthogonal(
                                                        gain=self.config.ortho_gain,
                                                        seed=self.config.seed),
                                                    kernel_regularizer=self.config.regularizer,
                                                    return_sequences=return_sequences))
        else:
            shared_lstm = Bidirectional(LSTM(units, kernel_initializer=self.config.initializers["lstm"],
                                             recurrent_initializer=orthogonal(gain=self.config.ortho_gain,
                                                                              seed=self.config.seed),
                                             kernel_regularizer=self.config.regularizer,
                                             return_sequences=return_sequences,
                                             recurrent_activation='sigmoid'))

        x_fwd = shared_lstm(inputs_fwd)
        x_rc = shared_lstm(inputs_rc)
        return x_fwd, x_rc

    def _add_rc_lstm(self, inputs, return_sequences, units):
        revcomp_in = Lambda(lambda x: K.reverse(x, axes=(1, 2)), output_shape=inputs.shape[1:],
                            name="reverse_complement_lstm_input_{n}".format(n=self._current_recurrent+1))
        inputs_rc = revcomp_in(inputs)
        x_fwd, x_rc = self._add_siam_lstm(inputs, inputs_rc, return_sequences, units)
        if return_sequences:
            rev_axes = (1, 2)
        else:
            rev_axes = 1
        revcomp_out = Lambda(lambda x: K.reverse(x, axes=rev_axes), output_shape=x_rc.shape[1:],
                             name="reverse_lstm_output_{n}".format(n=self._current_recurrent + 1))
        x_rc = revcomp_out(x_rc)
        out = concatenate([x_fwd, x_rc], axis=-1)
        return out

    def _add_siam_conv1d(self, inputs_fwd, inputs_rc, units, kernel_size, dilation_rate=1, stride=1):
        shared_conv = Conv1D(filters=units, kernel_size=kernel_size, dilation_rate=dilation_rate,
                             padding=self.config.padding,
                             kernel_initializer=self.config.initializers["conv"],
                             kernel_regularizer=self.config.regularizer,
                             strides=stride)
        x_fwd = shared_conv(inputs_fwd)
        x_rc = shared_conv(inputs_rc)
        return x_fwd, x_rc

    def _add_rc_conv1d(self, inputs, units, kernel_size, dilation_rate=1, stride=1):
        revcomp_in = Lambda(lambda x: K.reverse(x, axes=(1, 2)), output_shape=inputs.shape[1:],
                            name="reverse_complement_conv1d_input_{n}".format(n=self._current_conv+1))
        inputs_rc = revcomp_in(inputs)
        x_fwd, x_rc = self._add_siam_conv1d(inputs, inputs_rc, units, kernel_size, dilation_rate, stride)
        revcomp_out = Lambda(lambda x: K.reverse(x, axes=(1, 2)), output_shape=x_rc.shape[1:],
                             name="reverse_complement_conv1d_output_{n}".format(n=self._current_conv + 1))
        x_rc = revcomp_out(x_rc)
        out = concatenate([x_fwd, x_rc], axis=-1)
        return out

    def _add_siam_batchnorm(self, inputs_fwd, inputs_rc):
        input_shape = inputs_rc.shape
        if len(input_shape) != 3:
            raise ValueError("Intended for RC layers with 2D output. Use RC-Conv1D or RC-LSTM returning sequences."
                             "Expected dimension: 3, but got: " + str(len(input_shape)))
        out = concatenate([inputs_fwd, inputs_rc], axis=1)
        out = BatchNormalization()(out)
        split_shape = out.shape[1] // 2
        new_shape = [split_shape, input_shape[2]]
        fwd_out = Lambda(lambda x: x[:, :split_shape, :], output_shape=new_shape,
                         name="split_batchnorm_fwd_output_{n}".format(n=self._current_bn+1))
        rc_out = Lambda(lambda x: x[:, split_shape:, :], output_shape=new_shape,
                        name="split_batchnorm_rc_output1_{n}".format(n=self._current_bn+1))

        x_fwd = fwd_out(out)
        x_rc = rc_out(out)
        return x_fwd, x_rc

    def _add_rc_batchnorm(self, inputs):
        input_shape = inputs.shape
        if len(input_shape) != 3:
            raise ValueError("Intended for RC layers with 2D output. Use RC-Conv1D or RC-LSTM returning sequences."
                             "Expected dimension: 3, but got: " + str(len(input_shape)))
        split_shape = inputs.shape[-1] // 2
        new_shape = [input_shape[1], split_shape]
        fwd_in = Lambda(lambda x: x[:, :, :split_shape], output_shape=new_shape,
                        name="split_batchnorm_fwd_input_{n}".format(n=self._current_bn+1))
        rc_in = Lambda(lambda x: K.reverse(x[:, :, split_shape:], axes=(1, 2)), output_shape=new_shape,
                       name="split_batchnorm_rc_input_{n}".format(n=self._current_bn+1))
        inputs_fwd = fwd_in(inputs)
        inputs_rc = rc_in(inputs)
        x_fwd, x_rc = self._add_siam_batchnorm(inputs_fwd, inputs_rc)
        rc_out = Lambda(lambda x: K.reverse(x, axes=(1, 2)), output_shape=x_rc.shape,
                        name="split_batchnorm_rc_output2_{n}".format(n=self._current_bn+1))
        x_rc = rc_out(x_rc)
        out = concatenate([x_fwd, x_rc], axis=-1)
        return out

    def _add_siam_merge_dense(self, inputs_fwd, inputs_rc, units, merge_function=add):
        shared_dense = Dense(units, kernel_initializer=self.config.initializers["merge"],
                             kernel_regularizer=self.config.regularizer)
        x_fwd = shared_dense(inputs_fwd)
        x_rc = shared_dense(inputs_rc)
        out = merge_function([x_fwd, x_rc])
        return out

    def _add_rc_merge_dense(self, inputs, units, merge_function=add):
        split_shape = inputs.shape[-1] // 2
        fwd_in = Lambda(lambda x: x[:, :split_shape], output_shape=[split_shape],
                        name="split_merging_dense_input_fwd_{n}".format(n=1))
        rc_in = Lambda(lambda x: x[:, split_shape:], output_shape=[split_shape],
                       name="split_merging_dense_input_rc_{n}".format(n=1))
        x_fwd = fwd_in(inputs)
        x_rc = rc_in(inputs)
        rc_rev = Lambda(lambda x: K.reverse(x, axes=1), output_shape=x_rc.shape[1:],
                        name="reverse_merging_dense_input_{n}".format(n=1))
        x_rc = rc_rev(x_rc)
        return self._add_siam_merge_dense(x_fwd, x_rc, units, merge_function)

    def _add_skip(self, source, residual):
        stride = int(round(source.shape[1] / residual.shape[1]))

        if (source.shape[1] != residual.shape[1]) or (source.shape[-1] != residual.shape[-1]):
            source = Conv1D(filters=residual.shape[-1], kernel_size=1, strides=stride, padding=self.config.padding,
                            kernel_initializer=self.config.initializer,
                            kernel_regularizer=self.config.regularizer)(source)

        return add([source, residual])

    def _add_siam_skip(self, source_fwd, source_rc, residual_fwd, residual_rc):
        # Cast Dimension to int for TF 1 compatibility
        stride_fwd = int(round(int(source_fwd.shape[1]) / int(residual_fwd.shape[1])))
        stride_rc = int(round(int(source_rc.shape[1]) / int(residual_rc.shape[1])))
        assert stride_fwd == stride_rc, "Fwd and rc shapes differ."

        fwd_equal = (source_fwd.shape[1] == residual_fwd.shape[1]) and (source_fwd.shape[-1] == residual_fwd.shape[-1])
        rc_equal = (source_rc.shape[1] == residual_rc.shape[1]) and (source_rc.shape[-1] == residual_rc.shape[-1])

        if not (fwd_equal and rc_equal):
            source_fwd, source_rc = self._add_siam_conv1d(source_fwd, source_rc,
                                                          units=residual_fwd.shape[-1],
                                                          kernel_size=1, stride=stride_fwd)

        return add([source_fwd, residual_fwd]), add([source_rc, residual_rc])

    def _add_rc_skip(self, source, residual):
        equal = (source.shape[1] == residual.shape[1]) and (source.shape[-1] == residual.shape[-1])
        if equal:
            return add([source, residual])
        else:
            split_shape_src = source.shape[-1] // 2
            new_shape_src = [source.shape[1], split_shape_src]
            split_shape_res = residual.shape[-1] // 2
            new_shape_res = [residual.shape[1], split_shape_res]
            fwd_src_in = Lambda(lambda x: x[:, :, :split_shape_src],
                                output_shape=new_shape_src,
                                name="forward_skip_src_in_{n}".format(n=self._current_conv + 1))
            rc_src_in = Lambda(lambda x: K.reverse(x[:, :, split_shape_src:], axes=(1, 2)),
                               output_shape=new_shape_src,
                               name="reverse_complement_skip_src_in_{n}".format(n=self._current_conv + 1))
            fwd_res_in = Lambda(lambda x: x[:, :, :split_shape_res],
                                output_shape=new_shape_res,
                                name="forward_skip_res_in_{n}".format(n=self._current_conv + 1))
            rc_res_in = Lambda(lambda x: K.reverse(x[:, :, split_shape_res:], axes=(1, 2)),
                               output_shape=new_shape_res,
                               name="reverse_complement_skip_res_in{n}".format(n=self._current_conv + 1))
            source_fwd = fwd_src_in(source)
            residual_fwd = fwd_res_in(residual)
            source_rc = rc_src_in(source)
            residual_rc = rc_res_in(residual)
            x_fwd, x_rc = self._add_siam_skip(source_fwd, source_rc, residual_fwd, residual_rc)
            revcomp_out = Lambda(lambda x: K.reverse(x, axes=(1, 2)), output_shape=x_rc.shape[1:],
                                 name="reverse_complement_skip_output_{n}".format(n=self._current_conv + 1))
            x_rc = revcomp_out(x_rc)
            out = concatenate([x_fwd, x_rc], axis=-1)
            return out

    def _add_rc_pooling(self, inputs, pooling_layer):
        input_shape = inputs.shape
        if len(input_shape) != 3:
            raise ValueError("Intended for RC layers with 2D output. Use RC-Conv1D or RC-LSTM returning sequences."
                             "Expected dimension: 3, but got: " + str(len(input_shape)))
        split_shape = inputs.shape[-1] // 2
        new_shape = [input_shape[1], split_shape]
        fwd_in = Lambda(lambda x: x[:, :, :split_shape], output_shape=new_shape,
                        name="split_pooling_fwd_input_{n}".format(n=self._current_pool+1))
        rc_in = Lambda(lambda x: K.reverse(x[:, :, split_shape:], axes=(1, 2)), output_shape=new_shape,
                       name="split_pooling_rc_input_{n}".format(n=self._current_pool+1))
        inputs_fwd = fwd_in(inputs)
        inputs_rc = rc_in(inputs)
        x_fwd = pooling_layer(inputs_fwd)
        x_rc = pooling_layer(inputs_rc)
        rc_out = Lambda(lambda x: K.reverse(x, axes=(1, 2)), output_shape=x_rc.shape,
                        name="split_pooling_rc_output_{n}".format(n=self._current_pool+1))
        x_rc = rc_out(x_rc)
        out = concatenate([x_fwd, x_rc], axis=-1)
        return out

    def _build_simple_model(self):
        """Build the standard network"""
        print("Building model...")
        # Number of added recurrent layers
        self._current_recurrent = 0
        # Initialize input
        inputs = Input(shape=(self.config.seq_length, self.config.seq_dim))
        if self.config.mask_zeros:
            x = Masking()(inputs)
        else:
            x = inputs
        # The last recurrent layer should return the output for the last unit only.
        # Previous layers must return output for all units
        return_sequences = True if self.config.n_recurrent > 1 else False
        # Input dropout
        if not np.isclose(self.config.input_dropout, 0.0):
            x = Dropout(self.config.input_dropout, seed=self.config.seed)(x, training=self.config.dropout_training_mode)
        else:
            x = inputs
        # First convolutional/recurrent layer
        if self.config.n_conv > 0:
            # Convolutional layers will always be placed before recurrent ones
            # Standard convolutional layer
            x = Conv1D(filters=self.config.conv_units[0], kernel_size=self.config.conv_filter_size[0],
                       padding=self.config.padding,
                       kernel_initializer=self.config.initializers["conv"],
                       kernel_regularizer=self.config.regularizer,
                       strides=self.config.conv_stride[0])(x)
            if self.config.conv_bn:
                # Standard batch normalization layer
                x = BatchNormalization()(x)
            # Add activation
            x = Activation(self.config.conv_activation)(x)
        elif self.config.n_recurrent > 0:
            # If no convolutional layers, the first layer is recurrent.
            # CuDNNLSTM requires a GPU and tensorflow with cuDNN
            x = self._add_lstm(x, return_sequences)
            if self.config.recurrent_bn and return_sequences:
                # Standard batch normalization layer
                x = BatchNormalization()(x)
            # Add dropout
            x = Dropout(self.config.recurrent_dropout, seed=self.config.seed)(x,
                                                                              training=self.config.dropout_training_mode)
            # First recurrent layer already added
            self._current_recurrent = 1
        else:
            raise ValueError('First layer should be convolutional or recurrent')

        if self.config.skip_size > 0:
            start = x
        else:
            start = None
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
                x = Dropout(self.config.conv_dropout, seed=self.config.seed)(x,
                                                                             training=self.config.dropout_training_mode)
            # Add layer
            # Standard convolutional layer
            x = Conv1D(filters=self.config.conv_units[i], kernel_size=self.config.conv_filter_size[i],
                       padding=self.config.padding,
                       kernel_initializer=self.config.initializers["conv"],
                       kernel_regularizer=self.config.regularizer,
                       strides=self.config.conv_stride[i])(x)
            # Pre-activation skip connections https://arxiv.org/pdf/1603.05027v2.pdf
            if self.config.skip_size > 0:
                if i % self.config.skip_size == 0:
                    end = x
                    x = self._add_skip(start, end)
                    start = x
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
            elif self.config.conv_pooling == 'none':
                if self.config.n_recurrent == 0:
                    x = Flatten()(x)
                else:
                    raise ValueError('No pooling ("none") is not compatible with following LSTM layers.')
            else:
                # Skip pooling if needed or throw a ValueError if the pooling method is unrecognized
                # (should be thrown above)
                raise ValueError('Unknown pooling method')
            # Add dropout (drops whole features)
            if not np.isclose(self.config.conv_dropout, 0.0):
                x = Dropout(self.config.conv_dropout, seed=self.config.seed)(x,
                                                                             training=self.config.dropout_training_mode)

        # Recurrent layers
        for i in range(self._current_recurrent, self.config.n_recurrent):
            if i == self.config.n_recurrent - 1:
                # In the last layer, return output only for the last unit
                return_sequences = False
            # Add a bidirectional recurrent layer. CuDNNLSTM requires a GPU and tensorflow with cuDNN
            x = self._add_lstm(inputs, return_sequences)
            if self.config.recurrent_bn and return_sequences:
                # Standard batch normalization layer
                x = BatchNormalization()(x)
            # Add dropout
            x = Dropout(self.config.recurrent_dropout, seed=self.config.seed)(x,
                                                                              training=self.config.dropout_training_mode)

        # Dense layers
        for i in range(0, self.config.n_dense):
            x = Dense(self.config.dense_units[i],
                      kernel_initializer=self.config.initializers["dense"],
                      kernel_regularizer=self.config.regularizer)(x)
            if self.config.dense_bn:
                # Standard batch normalization layer
                x = BatchNormalization()(x)
            x = Activation(self.config.dense_activation)(x)
            x = Dropout(self.config.dense_dropout, seed=self.config.seed)(x,
                                                                          training=self.config.dropout_training_mode)

        # Output layer for binary classification
        x = Dense(1,
                  kernel_initializer=self.config.initializers["out"],
                  kernel_regularizer=self.config.regularizer, bias_initializer=self.config.output_bias)(x)
        x = Activation('sigmoid')(x)

        # Initialize the model
        self.model = Model(inputs, x)

    def _build_rc_model(self):
        """Build the RC network"""
        print("Building RC-model...")
        # Number of added recurrent layers
        self._current_recurrent = 0
        self._current_conv = 0
        self._current_bn = 0
        self._current_pool = 0
        # Initialize input
        inputs = Input(shape=(self.config.seq_length, self.config.seq_dim))
        if self.config.mask_zeros:
            x = Masking()(inputs)
        else:
            x = inputs
        # The last recurrent layer should return the output for the last unit only.
        #  Previous layers must return output for all units
        return_sequences = True if self.config.n_recurrent > 1 else False
        # Input dropout
        if not np.isclose(self.config.input_dropout, 0.0):
            x = Dropout(self.config.input_dropout, seed=self.config.seed)(x, training=self.config.dropout_training_mode)
        else:
            x = inputs
        # First convolutional/recurrent layer
        if self.config.n_conv > 0:
            # Convolutional layers will always be placed before recurrent ones
            x = self._add_rc_conv1d(x, units=self.config.conv_units[0], kernel_size=self.config.conv_filter_size[0],
                                    dilation_rate=self.config.conv_dilation[0],
                                    stride=self.config.conv_stride[0])
            if self.config.conv_bn:
                # Reverse-complemented batch normalization layer
                x = self._add_rc_batchnorm(x)
                self._current_bn = self._current_bn + 1
            x = Activation(self.config.conv_activation)(x)
            self._current_conv = self._current_conv + 1
        elif self.config.n_recurrent > 0:
            # If no convolutional layers, the first layer is recurrent.
            # CuDNNLSTM requires a GPU and tensorflow with cuDNN
            x = self._add_rc_lstm(x, return_sequences, self.config.recurrent_units[0])
            if self.config.recurrent_bn and return_sequences:
                # Reverse-complemented batch normalization layer
                x = self._add_rc_batchnorm(x)
                self._current_bn = self._current_bn + 1
            # Add dropout
            if not np.isclose(self.config.recurrent_dropout, 0.0):
                x = Dropout(self.config.recurrent_dropout,
                            seed=self.config.seed)(x,
                                                   training=self.config.dropout_training_mode)
            # First recurrent layer already added
            self._current_recurrent = self._current_recurrent + 1
        else:
            raise ValueError('First layer should be convolutional or recurrent')

        if self.config.skip_size > 0:
            start = x
        else:
            start = None
        # For next convolutional layers
        for i in range(1, self.config.n_conv):
            # Add pooling first
            if self.config.conv_pooling == 'max':
                x = self._add_rc_pooling(x, MaxPooling1D())
                self._current_pool = self._current_pool + 1
            elif self.config.conv_pooling == 'average':
                x = self._add_rc_pooling(x, AveragePooling1D())
                self._current_pool = self._current_pool + 1
            elif not (self.config.conv_pooling in ['last_max', 'last_average', 'none']):
                # Skip pooling if it should be applied to the last conv layer or skipped altogether.
                # Throw a ValueError if the pooling method is unrecognized.
                raise ValueError('Unknown pooling method')
            # Add dropout (drops whole features)
            if not np.isclose(self.config.conv_dropout, 0.0):
                x = Dropout(self.config.conv_dropout, seed=self.config.seed)(x,
                                                                             training=self.config.dropout_training_mode)
            # Add layer
            x = self._add_rc_conv1d(x, units=self.config.conv_units[i], kernel_size=self.config.conv_filter_size[i],
                                    dilation_rate=self.config.conv_dilation[i],
                                    stride=self.config.conv_stride[i])
            # Pre-activation skip connections https://arxiv.org/pdf/1603.05027v2.pdf
            if self.config.skip_size > 0:
                if i % self.config.skip_size == 0:
                    end = x
                    x = self._add_rc_skip(start, end)
                    start = x
            if self.config.conv_bn:
                # Reverse-complemented batch normalization layer
                x = self._add_rc_batchnorm(x)
                self._current_bn = self._current_bn + 1
            x = Activation(self.config.conv_activation)(x)
            self._current_conv = self._current_conv + 1

        # Pooling layer
        if self.config.n_conv > 0:
            if self.config.conv_pooling == 'max' or self.config.conv_pooling == 'last_max':
                if self.config.n_recurrent == 0:
                    # If no recurrent layers, use global pooling
                    x = GlobalMaxPooling1D()(x)
                else:
                    # for recurrent layers, use normal pooling
                    x = self._add_rc_pooling(x, MaxPooling1D())
                    self._current_pool = self._current_pool + 1
            elif self.config.conv_pooling == 'average' or self.config.conv_pooling == 'last_average':
                if self.config.n_recurrent == 0:
                    # if no recurrent layers, use global pooling
                    x = GlobalAveragePooling1D()(x)
                else:
                    # for recurrent layers, use normal pooling
                    x = self._add_rc_pooling(x, AveragePooling1D())
                    self._current_pool = self._current_pool + 1
            elif self.config.conv_pooling == 'none':
                if self.config.n_recurrent == 0:
                    x = Flatten()(x)
                else:
                    raise ValueError('No pooling ("none") is not compatible with following LSTM layers.')
            else:
                # Skip pooling if needed or throw a ValueError if the pooling method is unrecognized
                # (should be thrown above)
                raise ValueError('Unknown pooling method')
            # Add dropout (drops whole features)
            if not np.isclose(self.config.conv_dropout, 0.0):
                x = Dropout(self.config.conv_dropout, seed=self.config.seed)(x,
                                                                             training=self.config.dropout_training_mode)

        # Recurrent layers
        for i in range(self._current_recurrent, self.config.n_recurrent):
            if i == self.config.n_recurrent - 1:
                # In the last layer, return output only for the last unit
                return_sequences = False
            # Add a bidirectional recurrent layer. CuDNNLSTM requires a GPU and tensorflow with cuDNN
            x = self._add_rc_lstm(x, return_sequences, self.config.recurrent_units[i])
            if self.config.recurrent_bn and return_sequences:
                # Reverse-complemented batch normalization layer
                x = self._add_rc_batchnorm(x)
                self._current_bn = self._current_bn + 1
            # Add dropout
            if not np.isclose(self.config.recurrent_dropout, 0.0):
                x = Dropout(self.config.recurrent_dropout,
                            seed=self.config.seed)(x,
                                                   training=self.config.dropout_training_mode)
            self._current_recurrent = self._current_recurrent + 1

        # Dense layers
        for i in range(0, self.config.n_dense):
            if i == 0:
                x = self._add_rc_merge_dense(x, self.config.dense_units[i])
            else:
                x = Dense(self.config.dense_units[i],
                          kernel_initializer=self.config.initializers["dense"],
                          kernel_regularizer=self.config.regularizer)(x)
            if self.config.dense_bn:
                x = BatchNormalization()(x)
            x = Activation(self.config.dense_activation)(x)
            if not np.isclose(self.config.dense_dropout, 0.0):
                x = Dropout(self.config.dense_dropout,
                            seed=self.config.seed)(x,
                                                   training=self.config.dropout_training_mode)

        # Output layer for binary classification
        if self.config.n_dense == 0:
            x = self._add_rc_merge_dense(x, 1)
        else:
            x = Dense(1,
                      kernel_initializer=self.config.initializers["out"],
                      kernel_regularizer=self.config.regularizer, bias_initializer=self.config.output_bias)(x)
        x = Activation('sigmoid')(x)

        # Initialize the model
        self.model = Model(inputs, x)

    def _build_siam_model(self):
        """Build the RC network"""
        print("Building siamese RC-model...")
        # Number of added recurrent layers
        self._current_recurrent = 0
        self._current_conv = 0
        self._current_bn = 0
        # Initialize input
        inputs_fwd = Input(shape=(self.config.seq_length, self.config.seq_dim))
        if self.config.mask_zeros:
            x_fwd = Masking()(inputs_fwd)
        else:
            x_fwd = inputs_fwd
        revcomp_in = Lambda(lambda _x: K.reverse(_x, axes=(1, 2)), output_shape=inputs_fwd.shape[1:],
                            name="reverse_complement_input_{n}".format(n=self._current_recurrent+1))
        x_rc = revcomp_in(x_fwd)
        # The last recurrent layer should return the output for the last unit only.
        # Previous layers must return output for all units
        return_sequences = True if self.config.n_recurrent > 1 else False
        # Input dropout
        if not np.isclose(self.config.input_dropout, 0.0):
            x_fwd = Dropout(self.config.input_dropout,
                            seed=self.config.seed)(x_fwd,
                                                   training=self.config.dropout_training_mode)
            x_rc = Dropout(self.config.input_dropout,
                           seed=self.config.seed)(x_rc,
                                                  training=self.config.dropout_training_mode)
        # First convolutional/recurrent layer
        if self.config.n_conv > 0:
            # Convolutional layers will always be placed before recurrent ones
            # Reverse-complement convolutional layer
            x_fwd, x_rc = self._add_siam_conv1d(x_fwd, x_rc, units=self.config.conv_units[0],
                                                kernel_size=self.config.conv_filter_size[0],
                                                dilation_rate=self.config.conv_dilation[0],
                                                stride=self.config.conv_stride[0])
            if self.config.conv_bn:
                # Reverse-complemented batch normalization layer
                x_fwd, x_rc = self._add_siam_batchnorm(x_fwd, x_rc)
                self._current_bn = self._current_bn + 1
            # Add activation
            act = Activation(self.config.conv_activation)
            x_fwd = act(x_fwd)
            x_rc = act(x_rc)
            self._current_conv = self._current_conv + 1
        elif self.config.n_recurrent > 0:
            # If no convolutional layers, the first layer is recurrent.
            # CuDNNLSTM requires a GPU and tensorflow with cuDNN
            # RevComp input
            x_fwd, x_rc = self._add_siam_lstm(x_fwd, x_rc, return_sequences, self.config.recurrent_units[0])
            # Add batch norm
            if self.config.recurrent_bn and return_sequences:
                # reverse-complemented batch normalization layer
                x_fwd, x_rc = self._add_siam_batchnorm(x_fwd, x_rc)
                self._current_bn = self._current_bn + 1
            # Add dropout
            if not np.isclose(self.config.recurrent_dropout, 0.0):
                x_fwd = Dropout(self.config.recurrent_dropout,
                                seed=self.config.seed)(x_fwd,
                                                       training=self.config.dropout_training_mode)
                x_rc = Dropout(self.config.recurrent_dropout,
                               seed=self.config.seed)(x_rc,
                                                      training=self.config.dropout_training_mode)
            # First recurrent layer already added
            self._current_recurrent = 1
        else:
            raise ValueError('First layer should be convolutional or recurrent')

        if self.config.skip_size > 0:
            start_fwd = x_fwd
            start_rc = x_rc
        else:
            start_fwd = None
            start_rc = None
        # For next convolutional layers
        for i in range(1, self.config.n_conv):
            # Add pooling first
            if self.config.conv_pooling == 'max':
                pool = MaxPooling1D()
                x_fwd = pool(x_fwd)
                x_rc = pool(x_rc)
            elif self.config.conv_pooling == 'average':
                pool = AveragePooling1D()
                x_fwd = pool(x_fwd)
                x_rc = pool(x_rc)
            elif not (self.config.conv_pooling in ['last_max', 'last_average', 'none']):
                # Skip pooling if it should be applied to the last conv layer or skipped altogether.
                # Throw a ValueError if the pooling method is unrecognized.
                raise ValueError('Unknown pooling method')
            # Add dropout (drops whole features)
            if not np.isclose(self.config.conv_dropout, 0.0):
                x_fwd = Dropout(self.config.conv_dropout,
                                seed=self.config.seed)(x_fwd,
                                                       training=self.config.dropout_training_mode)
                x_rc = Dropout(self.config.conv_dropout,
                               seed=self.config.seed)(x_rc,
                                                      training=self.config.dropout_training_mode)
            # Add layer
            # Reverse-complement convolutional layer
            x_fwd, x_rc = self._add_siam_conv1d(x_fwd, x_rc, units=self.config.conv_units[i],
                                                kernel_size=self.config.conv_filter_size[i],
                                                dilation_rate=self.config.conv_dilation[i],
                                                stride=self.config.conv_stride[i])
            # Pre-activation skip connections https://arxiv.org/pdf/1603.05027v2.pdf
            if self.config.skip_size > 0:
                if i % self.config.skip_size == 0:
                    end_fwd = x_fwd
                    end_rc = x_rc
                    x_fwd, x_rc = self._add_siam_skip(start_fwd, start_rc, end_fwd, end_rc)
                    start_fwd = x_fwd
                    start_rc = x_rc
            if self.config.conv_bn:
                # Reverse-complemented batch normalization layer
                x_fwd, x_rc = self._add_siam_batchnorm(x_fwd, x_rc)
                self._current_bn = self._current_bn + 1
            # Add activation
            act = Activation(self.config.conv_activation)
            x_fwd = act(x_fwd)
            x_rc = act(x_rc)
            self._current_conv = self._current_conv + 1

        # Pooling layer
        if self.config.n_conv > 0:
            if self.config.conv_pooling == 'max' or self.config.conv_pooling == 'last_max':
                if self.config.n_recurrent == 0:
                    # If no recurrent layers, use global pooling
                    pool = GlobalMaxPooling1D()
                    x_fwd = pool(x_fwd)
                    x_rc = pool(x_rc)
                else:
                    # for recurrent layers, use normal pooling
                    pool = MaxPooling1D()
                    x_fwd = pool(x_fwd)
                    x_rc = pool(x_rc)
            elif self.config.conv_pooling == 'average' or self.config.conv_pooling == 'last_average':
                if self.config.n_recurrent == 0:
                    # if no recurrent layers, use global pooling
                    pool = GlobalAveragePooling1D()
                    x_fwd = pool(x_fwd)
                    x_rc = pool(x_rc)
                else:
                    # for recurrent layers, use normal pooling
                    pool = AveragePooling1D()
                    x_fwd = pool(x_fwd)
                    x_rc = pool(x_rc)
            elif self.config.conv_pooling == 'none':
                if self.config.n_recurrent == 0:
                    pool = Flatten()
                    x_fwd = pool(x_fwd)
                    x_rc = pool(x_rc)
                else:
                    raise ValueError('No pooling ("none") is not compatible with following LSTM layers.')
            else:
                # Skip pooling if needed or throw a ValueError if the pooling method is unrecognized
                # (should be thrown above)
                raise ValueError('Unknown pooling method')
            # Add dropout (drops whole features)
            if not np.isclose(self.config.conv_dropout, 0.0):
                x_fwd = Dropout(self.config.conv_dropout,
                                seed=self.config.seed)(x_fwd, training=self.config.dropout_training_mode)
                x_rc = Dropout(self.config.conv_dropout,
                               seed=self.config.seed)(x_rc, training=self.config.dropout_training_mode)

        # Recurrent layers
        for i in range(self._current_recurrent, self.config.n_recurrent):
            if i == self.config.n_recurrent - 1:
                # In the last layer, return output only for the last unit
                return_sequences = False
            # Add a bidirectional recurrent layer. CuDNNLSTM requires a GPU and tensorflow with cuDNN
            x_fwd, x_rc = self._add_siam_lstm(x_fwd, x_rc, return_sequences, self.config.recurrent_units[i])
            # Add batch norm
            if self.config.recurrent_bn and return_sequences:
                # Reverse-complemented batch normalization layer
                x_fwd, x_rc = self._add_siam_batchnorm(x_fwd, x_rc)
                self._current_bn = self._current_bn + 1
            # Add dropout
            if not np.isclose(self.config.recurrent_dropout, 0.0):
                x_fwd = Dropout(self.config.recurrent_dropout,
                                seed=self.config.seed)(x_fwd, training=self.config.dropout_training_mode)
                x_rc = Dropout(self.config.recurrent_dropout,
                               seed=self.config.seed)(x_rc, training=self.config.dropout_training_mode)

        # Output layer for binary classification
        if self.config.n_dense == 0:
            # Output layer for binary classification
            x = self._add_siam_merge_dense(x_fwd, x_rc, 1)
        else:
            # Dense layers
            x = self._add_siam_merge_dense(x_fwd, x_rc, self.config.dense_units[0])
            if self.config.dense_bn:
                # Batch normalization layer
                x = BatchNormalization()(x)
            x = Activation(self.config.dense_activation)(x)
            x = Dropout(self.config.dense_dropout,
                        seed=self.config.seed)(x, training=self.config.dropout_training_mode)
            for i in range(1, self.config.n_dense):
                x = Dense(self.config.dense_units[i],
                          kernel_initializer=self.config.initializers["dense"],
                          kernel_regularizer=self.config.regularizer)(x)
                if self.config.dense_bn:
                    # Batch normalization layer
                    x = BatchNormalization()(x)
                x = Activation(self.config.dense_activation)(x)
                x = Dropout(self.config.dense_dropout,
                            seed=self.config.seed)(x, training=self.config.dropout_training_mode)
            # Output layer for binary classification
            x = Dense(1,
                      kernel_initializer=self.config.initializers["out"],
                      kernel_regularizer=self.config.regularizer, bias_initializer=self.config.output_bias)(x)
        x = Activation('sigmoid')(x)

        # Initialize the model
        self.model = Model(inputs_fwd, x)

    def compile_model(self):
        """Compile model and save model summaries"""
        if float(tf.__version__[:3]) < 2.2 or self.config.epoch_start == 0:
            print("Compiling...")
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
        else:
            print('Skipping compilation of a pre-trained model...')

    def _set_callbacks(self):
        """Set callbacks to use during training"""
        self.callbacks = []

        # Set early stopping
        self.callbacks.append(EarlyStopping(monitor="val_accuracy", patience=self.config.patience))

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
        checkpoint_name = self.config.log_dir + "/{runname}-".format(runname=self.config.runname)
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
        with self.get_device_strategy_scope():
            if self.config.use_tf_data:
                # Fit a model using tf data
                self.history = self.model.fit(x=self.training_sequence,
                                              epochs=self.config.epoch_end,
                                              callbacks=self.callbacks,
                                              validation_data=self.validation_data,
                                              class_weight=self.config.class_weight,
                                              use_multiprocessing=self.config.multiprocessing,
                                              max_queue_size=self.config.batch_queue,
                                              workers=self.config.batch_loading_workers,
                                              initial_epoch=self.config.epoch_start,
                                              steps_per_epoch=math.ceil(self.length_train/self.config.batch_size),
                                              validation_steps=math.ceil(self.length_val/self.config.batch_size))

            elif self.config.use_generators_keras:
                # Fit a model using generators
                self.history = self.model.fit(x=self.training_sequence,
                                              epochs=self.config.epoch_end,
                                              callbacks=self.callbacks,
                                              validation_data=self.validation_data,
                                              class_weight=self.config.class_weight,
                                              use_multiprocessing=self.config.multiprocessing,
                                              max_queue_size=self.config.batch_queue,
                                              workers=self.config.batch_loading_workers,
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
