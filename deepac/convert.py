"""@package deepac.convert_cudnn
Convert a CuDNNLSTM to a CPU-compatible LSTM.

Requires a config file describing the available devices, data loading mode, input sequence length, network architecture,
paths to input files and how should was the model trained, and a file with model weights. The original config file may
be used, as the number of available devices is overridden by this script.

"""

import re

from tensorflow.keras.models import load_model

from deepac.nn_train import RCConfig, RCNet


def convert_cudnn(config, saved_model, no_prep, no_weights=False):
    """Rebuild the network using a modified configuration."""

    # Load model architecture, device info and weights
    paprconfig = RCConfig(config)

    paprnet = RCNet(paprconfig, training_mode=False)

    path = saved_model

    if not no_weights:
        if re.search("\.h5$", path) is not None:
            path = re.sub("\.h5$", "", path)
        save_path = path + "_converted.h5"
        if no_prep:
            weights_path = saved_model
        else:
            weights_path = path + "_weights.h5"
            # Prepare weights
            model = load_model(saved_model)
            model.save_weights(weights_path)
        paprnet.model.load_weights(weights_path)
    else:
        save_path = path

    paprnet.model.compile(loss='binary_crossentropy',
                          optimizer=paprnet.config.optimizer,
                          metrics=['accuracy'])

    # Save output
    paprnet.model.save(save_path)
    print(paprnet.model.summary())
