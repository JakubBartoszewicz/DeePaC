"""@package deepac.utils
Utility classes for DeePaC.

"""
import numpy as np
import os
import math
import psutil
import tensorflow as tf
import csv
import six
from collections import OrderedDict
from collections import Iterable
import multiprocessing


class ReadSequence(tf.keras.utils.Sequence):

    """
    A Keras sequence for yielding batches from a numpy array loaded in mmap mode.

    """

    def __init__(self, x_set, y_set, batch_size, use_subreads, min_subread_length, max_subread_length, dist_subread,
                 verbose_id=None):
        """ReadSequence constructor"""
        self.X, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(len(self.y))
        self.use_subreads = use_subreads
        self.min_subread_length = min_subread_length
        self.max_subread_length = max_subread_length
        self.dist_subread = dist_subread
        self.verbose_id = verbose_id
        self.epoch = 0
        self.on_epoch_end()

    def __len__(self):
        """Return the number of items of a sequence."""
        return math.floor(len(self.X) / self.batch_size)

    def __getitem__(self, idx):
        """Get a batch at index"""
        batch_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        if self.use_subreads:
            batch_x = np.copy(self.X[batch_indices])
            """Randomly shorten reads"""
            for matrix in batch_x:
                random_length = np.random.randint(self.min_subread_length, self.max_subread_length + 1)
                matrix[random_length:, :] = 0
        else:
            batch_x = self.X[batch_indices]
        batch_y = self.y[batch_indices]

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        """Update indices after each epoch"""
        if self.verbose_id is not None:
            if self.epoch == 0:
                print("::{} sequence INIT".format(self.verbose_id))
            else:
                print("\n::{} sequence epoch {} END".format(self.verbose_id, self.epoch))
            self.epoch = self.epoch + 1
        self.indices = np.arange(len(self.y))
        np.random.shuffle(self.indices)


def get_memory_usage():
    # return the memory usage in MB
    process = psutil.Process()
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


class CSVMemoryLogger(tf.keras.callbacks.CSVLogger):

    """
    A Keras CSV logger with a memory usage field.
    Based on a comment by joelthchao: https://github.com/keras-team/keras/issues/5935#issuecomment-289041967
    """
    def __init__(self, *args, **kwargs):
        self.keys = None
        self.writer = None
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        """Log memory usage and performance after each epoch"""
        logs = logs or {}
        # Get memory usage as maxrss
        mem_usage = get_memory_usage()
        logs["Mem"] = mem_usage

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch'] + self.keys, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()


def set_mem_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def config_cpus(n_cpus):
    if n_cpus is None:
        n_cpus = multiprocessing.cpu_count()
    if n_cpus <= 0:
        raise ValueError("%s is an invalid number of cores" % n_cpus)
    # Use as many intra_threads as the CPUs available
    intra_threads = n_cpus
    # Same for inter_threads
    inter_threads = intra_threads
    tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
    return n_cpus


def config_gpus(gpus):
    physical_devices = tf.config.list_physical_devices('GPU')
    if gpus is None:
        used_devices = tf.config.get_visible_devices('GPU')
    else:
        valid_gpus = [d for d in gpus if d <= len(physical_devices)-1]
        invalid_gpus = [d for d in gpus if d > len(physical_devices)-1]
        invalid_gpus = ["/device:GPU:{}".format(i) for i in invalid_gpus]
        if len(invalid_gpus) > 0:
            print("Devices not found: " + ", ".join(invalid_gpus))
        if len(valid_gpus) == 0:
            return config_gpus(None)
        used_devices = [physical_devices[d] for d in valid_gpus]
        tf.config.set_visible_devices(used_devices, 'GPU')
    print("Physical GPUs: {}".format(", ".join([d.name for d in physical_devices])))
    if len(used_devices) > 0:
        print("Used GPUs: {}".format(", ".join([d.name for d in used_devices])))
    else:
        print("Used GPUs: None")
    n_gpus = len(used_devices)
    return n_gpus


def config_tpus(tpu_name):
    if tpu_name is not None:
        if tpu_name.lower() == "colab":
            try:
                name = 'grpc://' + os.environ['COLAB_TPU_ADDR']
            except KeyError:
                print("TPU not found (COLAB_TPU_ADDR not set).")
                return None
        else:
            name = tpu_name
        print("Setting up TPU: {}".format(name))
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(name)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            return resolver
        except (ValueError, tf.errors.NotFoundError):
            print("TPU not found.")
            return None
    else:
        return None


class DatasetParser:
    """
        A parser for TFRecordDatasets of preprocessed reads.

    """

    def __init__(self, read_length, use_subreads=False, min_subread_length=None, max_subread_length=None,
                 dist_subread=None, dtype=tf.int32):
        """DatasetParser constructor"""
        self.read_length = read_length
        self.use_subreads = use_subreads
        self.min_subread_length = min_subread_length
        self.max_subread_length = max_subread_length
        self.dist_subread = dist_subread
        self.dtype=dtype
        self.feature_description = {
            'x_seq': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'y_label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        }
        self.AUTO = tf.data.experimental.AUTOTUNE

    def _parse_dataset(self, example_proto):

        # Parse the input `tf.Example` proto using the dictionary above.
        example = tf.io.parse_single_example(example_proto, self.feature_description)
        x_seq = tf.reshape(tf.io.parse_tensor(example["x_seq"], out_type=self.dtype), [self.read_length, 4])
        y_label = tf.reshape(tf.cast(example["y_label"], self.dtype), [1])
        return x_seq, y_label

    def read_dataset(self, filenames):
        raw_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=self.AUTO)
        parsed_dataset = raw_dataset.map(self._parse_dataset, num_parallel_calls=self.AUTO)
        return parsed_dataset
