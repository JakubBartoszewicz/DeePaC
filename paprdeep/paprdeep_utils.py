"""@package paprdeep_utils
Utility classes for PaPrDeep.
  
"""
# Set seeds at the very beginning for maximum reproducibility
seed = 0
import numpy as np
np.random.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = '0'
import random as rn
rn.seed(seed)

import math
from keras.utils.data_utils import Sequence
import resource
from keras.callbacks import CSVLogger
import csv
import six
from collections import OrderedDict
from collections import Iterable

class PaPrSequence(Sequence):

    """
    A Keras sequence for yielding batches from a numpy array loaded in mmap mode.
    
    """
    
    def __init__(self, x_set, y_set, batch_size):
        """PaPrSequence constructor"""
        self.X,self.y = x_set,y_set
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
         """Return the number of items of a sequence."""
        return math.floor(len(self.X) / self.batch_size)

    def __getitem__(self,idx):
        """Get a batch at index"""
        batch_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_x = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        
        return np.array(batch_x), np.array(batch_y)
        
    def on_epoch_end(self):
        """Update indices after each epoch"""
        self.indices = np.arange(len(self.y))
        np.random.shuffle(self.indices)
            
# based on a comment by joelthchao https://github.com/keras-team/keras/issues/5935#issuecomment-289041967
class CSVMemoryLogger(CSVLogger):

    """
    A Keras CSV logger with a memory usage field.
    
    """
    
    def on_epoch_end(self, epoch, logs=None):
        """Log memory usage and performance after each epoch"""
        logs = logs or {}
        # Get memory usage as maxrss
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
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