"""@package deepac.utils
Utility classes for DeePaC.
  
"""
import numpy as np

import math
from keras.utils.data_utils import Sequence
import psutil
from keras.callbacks import CSVLogger
from keras import Model
from keras.utils import multi_gpu_model
import csv
import six
from collections import OrderedDict
from collections import Iterable


class ModelMGPU(Model):

    """
    A wrapper for multi_gpu_model allowing saving with the ModelCheckpoint callback
    Based on a comment by avolkov1: https://github.com/keras-team/keras/issues/2436#issuecomment-354882296
    """
    
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        """Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        """
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


class ReadSequence(Sequence):

    """
    A Keras sequence for yielding batches from a numpy array loaded in mmap mode.
    
    """
    
    def __init__(self, x_set, y_set, batch_size):
        """PaPrSequence constructor"""
        self.X, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(len(self.y))
        self.on_epoch_end()

    def __len__(self):
        """Return the number of items of a sequence."""
        return math.floor(len(self.X) / self.batch_size)

    def __getitem__(self, idx):
        """Get a batch at index"""
        batch_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_x = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        
        return np.array(batch_x), np.array(batch_y)
        
    def on_epoch_end(self):
        """Update indices after each epoch"""
        self.indices = np.arange(len(self.y))
        np.random.shuffle(self.indices)


class CSVMemoryLogger(CSVLogger):

    """
    A Keras CSV logger with a memory usage field.
    Based on a comment by joelthchao: https://github.com/keras-team/keras/issues/5935#issuecomment-289041967
    """

    def __get_memory_usage(self):
        # return the memory usage in MB
        process = psutil.Process()
        mem = process.memory_info()[0] / float(2 ** 20)
        return mem
    
    def on_epoch_end(self, epoch, logs=None):
        """Log memory usage and performance after each epoch"""
        logs = logs or {}
        # Get memory usage as maxrss
        mem_usage = self.__get_memory_usage()
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
