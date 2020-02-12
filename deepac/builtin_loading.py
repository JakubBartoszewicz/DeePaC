import tensorflow.compat.v1.keras.backend as K
from deepac.nn_train import RCConfig, RCNet
import configparser
import os


class BuiltinLoader:

    def __init__(self, modulepath):
        self.modulepath = modulepath

    def load_sensitive_model(self, n_cpus, n_gpus, d_pref="/device:GPU:", device_parallel=False, log_path="logs",
                             training_mode=True):
        if n_gpus > 1:
            device_parallel = True
        return self.__load_builtin_model("nn-img-sensitive-lstm", n_cpus, n_gpus, d_pref, device_parallel, log_path,
                                    training_mode)

    def load_rapid_model(self, n_cpus, n_gpus, d_pref="/device:GPU:", device_parallel=False, log_path="logs",
                         training_mode=True):
        return self.__load_builtin_model("nn-img-rapid-cnn", n_cpus, n_gpus, d_pref, device_parallel, log_path,
                                    training_mode)

    def __load_builtin_model(self, prefix, n_cpus, n_gpus, d_pref="/device:GPU:", device_parallel=False, log_path="logs",
                             training_mode=True):
        config_path = os.path.join(self.modulepath, "builtin", "config", "{}.ini".format(prefix))
        weights_path = os.path.join(self.modulepath, "builtin", "weights", "{}.h5".format(prefix))
        config = configparser.ConfigParser()
        config.read(config_path)
        paprconfig = RCConfig(config)
        paprconfig.log_superpath = log_path
        paprconfig.log_dir = paprconfig.log_superpath + "/{runname}-logs".format(runname=paprconfig.runname)
        paprconfig.n_cpus = n_cpus
        paprconfig.n_gpus = n_gpus
        paprconfig.device_parallel = device_parallel
        if device_parallel:
            paprconfig.device_fwd = d_pref + str(min(0, n_gpus-1))
            paprconfig.device_rc = d_pref + str(min(1, n_gpus-1))
            paprconfig.model_build_device = d_pref + str(min(2, n_gpus-1))

        if K.backend() == 'tensorflow':
            paprconfig.set_tf_session()

        paprnet = RCNet(paprconfig, training_mode)

        paprnet.model.load_weights(weights_path)

        return paprnet.model

    def get_sensitive_training_config(self, n_cpus, n_gpus, d_pref="/device:GPU:", device_parallel=False):
        if n_gpus > 1:
            device_parallel = True
        return self.__get_builtin_training_config("nn-img-sensitive-lstm", n_cpus, n_gpus, d_pref, device_parallel)

    def get_rapid_training_config(self, n_cpus, n_gpus, d_pref="/device:GPU:", device_parallel=False):
        return self.__get_builtin_training_config("nn-img-rapid-cnn", n_cpus, n_gpus, d_pref, device_parallel)

    def __get_builtin_training_config(self, prefix, n_cpus, n_gpus, d_pref="/device:GPU:", device_parallel=False):
        config_path = os.path.join(self.modulepath, "builtin", "config", "{}.ini".format(prefix))
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