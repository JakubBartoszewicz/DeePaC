"""@package deepac.convert_cudnn
Convert a CuDNNLSTM to a CPU-compatible LSTM.

Requires a config file describing the available devices, data loading mode, input sequence length, network architecture,
paths to input files and how should was the model trained, and a file with model weights. The original config file may
be used, as the number of available devices is overridden by this script.

"""

import re

import keras.backend as K
from keras.models import load_model

from deepac.nn_train import RCConfig, RCNet


def convert_cudnn(config, saved_model, no_prep):
    """Convert a CuDNNLSTM to a CPU-compatible LSTM."""

    path = saved_model
    if re.search("\.h5$", path) is not None:
        path = re.sub("\.h5$", "", path)

    if no_prep:
        weights_path = saved_model
    else:
        weights_path = path + "_weights.h5"
        # Prepare weights
        model = load_model(saved_model)
        model.save_weights(weights_path)


    # Load model architecture, device info and weights
    paprconfig = RCConfig(config)

    if K.backend() == 'tensorflow':
        paprconfig.set_tf_session()
    paprnet = RCNet(paprconfig)

    paprnet.model.load_weights(weights_path)

    # Save output
    save_path = path + "_converted.h5"
    paprnet.model.save(save_path)
