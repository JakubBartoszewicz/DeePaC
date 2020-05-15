from deepac.nn_train import RCConfig, RCNet
import configparser
import os
from deepac import __file__
from deepac.utils import config_gpus, config_cpus
import tensorflow as tf


class BuiltinLoader:

    def __init__(self, builtin_configs, builtin_weights):
        if builtin_configs is None:
            modulepath = os.path.dirname(__file__)
            self.builtin_configs = {"rapid": os.path.join(modulepath, "builtin", "config", "nn-img-rapid-cnn.ini"),
                                    "sensitive": os.path.join(modulepath, "builtin", "config",
                                                              "nn-img-sensitive-lstm.ini")}
        else:
            self.builtin_configs = builtin_configs
        if builtin_weights is None:
            modulepath = os.path.dirname(__file__)
            self.builtin_weights = {"rapid": os.path.join(modulepath, "builtin", "weights", "nn-img-rapid-cnn.h5"),
                                    "sensitive": os.path.join(modulepath, "builtin", "weights",
                                                              "nn-img-sensitive-lstm.h5")}
        else:
            self.builtin_weights = builtin_weights

    def load_sensitive_model(self, n_cpus=None, n_gpus=None, log_path="logs", training_mode=True, tpu_resolver=None):
        return self.__load_builtin_model("sensitive", n_cpus, n_gpus, log_path, training_mode, tpu_resolver)

    def load_rapid_model(self, n_cpus=None, n_gpus=None, log_path="logs", training_mode=True, tpu_resolver=None):
        return self.__load_builtin_model("rapid", n_cpus, n_gpus, log_path, training_mode, tpu_resolver)

    def __load_builtin_model(self, modelkey, n_cpus=None, n_gpus=None, log_path="logs", training_mode=True,
                             tpu_resolver=None):
        config_path = self.builtin_configs[modelkey]
        weights_path = self.builtin_weights[modelkey]
        print("Loading {}".format(os.path.basename(weights_path)))
        config = configparser.ConfigParser()
        config.read(config_path)
        paprconfig = RCConfig(config)
        paprconfig.log_superpath = log_path
        paprconfig.log_dir = paprconfig.log_superpath + "/{runname}-logs".format(runname=paprconfig.runname)

        # for backwards compatibility with deepac-live v0.2
        if n_cpus is not None:
            config_cpus(n_cpus)
        if n_gpus is not None:
            if n_gpus == 0:
                tf.config.set_visible_devices([], 'GPU')
            else:
                physical_devices = tf.config.list_physical_devices('GPU')
                n_valid_gpus = min(len(physical_devices), n_gpus)
                valid_gpus = list(range(n_valid_gpus))
                config_gpus(valid_gpus)

        paprconfig.set_tpu_resolver(tpu_resolver)
        paprnet = RCNet(paprconfig, training_mode)

        paprnet.model.load_weights(weights_path)

        return paprnet.model

    def get_sensitive_training_config(self):
        return self.__get_builtin_training_config("sensitive")

    def get_rapid_training_config(self):
        return self.__get_builtin_training_config("rapid")

    def __get_builtin_training_config(self, modelkey):
        config_path = self.builtin_configs[modelkey]
        print("Loading {}".format(os.path.basename(config_path)))
        config = configparser.ConfigParser()
        config.read(config_path)
        paprconfig = RCConfig(config)

        return paprconfig
