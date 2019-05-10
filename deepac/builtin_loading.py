import keras.backend as K
from deepac import __file__
from deepac.nn_train import RCConfig, RCNet
import configparser
import os


def load_sensitive_model(n_cpus, n_gpus, d_pref="/device:GPU:", device_parallel=False):
    if n_gpus > 1:
        device_parallel = True
    return __load_builtin_model("nn-img-sensitive-lstm", n_cpus, n_gpus, d_pref, device_parallel)


def load_rapid_model(n_cpus, n_gpus, d_pref="/device:GPU:", device_parallel=False, ):
    return __load_builtin_model("nn-img-rapid-cnn", n_cpus, n_gpus, d_pref, device_parallel)


def __load_builtin_model(prefix, n_cpus, n_gpus, d_pref="/device:GPU:", device_parallel=False):
    config_path = os.path.join(os.path.dirname(__file__), "builtin", "config", "{}.ini".format(prefix))
    weights_path = os.path.join(os.path.dirname(__file__), "builtin", "weights", "{}.h5".format(prefix))
    config = configparser.ConfigParser()
    config.read(config_path)
    paprconfig = RCConfig(config)
    paprconfig.n_cpus = n_cpus
    paprconfig.n_gpus = n_gpus
    paprconfig.device_parallel = device_parallel
    if device_parallel:
        paprconfig.device_fwd = d_pref + str(min(0, n_gpus-1))
        paprconfig.device_rc = d_pref + str(min(1, n_gpus-1))
        paprconfig.model_build_device = d_pref + str(min(2, n_gpus-1))

    if K.backend() == 'tensorflow':
        paprconfig.set_tf_session()

    paprnet = RCNet(paprconfig)

    paprnet.model.load_weights(weights_path)

    return paprnet.model


def get_sensitive_training_config(n_cpus, n_gpus, d_pref="/device:GPU:", device_parallel=False):
    if n_gpus > 1:
        device_parallel = True
    return __get_builtin_training_config("nn-img-sensitive-lstm", n_cpus, n_gpus, d_pref, device_parallel)


def get_rapid_training_config(n_cpus, n_gpus, d_pref="/device:GPU:", device_parallel=False):
    return __get_builtin_training_config("nn-img-rapid-cnn", n_cpus, n_gpus, d_pref, device_parallel)


def __get_builtin_training_config(prefix, n_cpus, n_gpus, d_pref="/device:GPU:", device_parallel=False):
    config_path = os.path.join(os.path.dirname(__file__), "builtin", "config", "{}.ini".format(prefix))
    config = configparser.ConfigParser()
    config.read(config_path)
    paprconfig = RCConfig(config)
    paprconfig.n_cpus = n_cpus
    paprconfig.n_gpus = n_gpus
    paprconfig.device_parallel = device_parallel
    if device_parallel:
        paprconfig.device_fwd = d_pref + str(min(0, n_gpus - 1))
        paprconfig.device_rc = d_pref + str(min(1, n_gpus - 1))
        paprconfig.model_build_device = d_pref + str(min(2, n_gpus - 1))

    return paprconfig