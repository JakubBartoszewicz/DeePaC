"""@package convert_cudnn
Convert a CuDNNLSTM to a CPU-compatible LSTM.

Requires a config file describing the available devices, data loading mode, input sequence length, network architecture,
paths to input files and how should was the model trained, and a file with model weights. The original config file may
be used, as the number of available devices is overridden by this script.

usage: convert_cudnn.py [-h] config_file model_weights

positional arguments:
  config_file
  model_weights

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

from nn_train import PaPrConfig, PaPrNet

def main():
    """Parse the config file and train the NN on Illumina reads."""
    parser = argparse.ArgumentParser(description="Convert a CuDNNLSTM to a CPU-compatible LSTM.")
    parser.add_argument("config_file")
    parser.add_argument("model_weights")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_file)
    paprconfig = PaPrConfig(config)

    paprconfig.n_gpus = 0
    paprconfig.multi_gpu = False
    paprconfig.model_build_device = '/cpu:0'

    if K.backend() == 'tensorflow':
        paprconfig.set_tf_session()
    paprnet = PaPrNet(paprconfig)

    paprnet.model.load_weights(args.model_weights)

    save_path = args.model_weights
    if re.search("\.h5$", save_path) is not None:
        save_path = re.sub("\.h5$", "", save_path)
    save_path = save_path + "_cpu.h5"

    paprnet.model.save(save_path)



if __name__ == "__main__":
    main()
