"""@package convert_cudnn
Convert a CuDNNLSTM to a CPU-compatible LSTM.

Requires a config file describing the available devices, data loading mode, input sequence length, network architecture,
paths to input files and how should was the model trained, and a file with model weights. The original config file may
be used, as the number of available devices is overridden by this script.

usage: convert_cudnn.py [-h] config_file saved_model

positional arguments:
  config_file
  saved_model

optional arguments:
  -h, --help   show this help message and exit

"""

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

import argparse
import configparser
import re

import keras.backend as K
from keras.models import load_model

from nn_train import RCConfig, RCNet

def main():
    """Parse the config file and train the NN on Illumina reads."""
    parser = argparse.ArgumentParser(description="Convert a CuDNNLSTM to a CPU-compatible LSTM.")
    parser.add_argument("config_file")
    parser.add_argument("saved_model")
    parser.add_argument("--prep_weights", default=True, help="prepare weights based on model file")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_file)

    path = args.saved_model
    if re.search("\.h5$", path) is not None:
        path = re.sub("\.h5$", "", path)
    weights_path = path + "_weights.h5"

    # Prepare weights
    if args.prep_weights:
        model = load_model(args.saved_model)
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



if __name__ == "__main__":
    main()
