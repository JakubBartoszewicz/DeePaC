from deepac.nn_train import RCConfig, RCNet
import configparser
import os
from deepac import __file__
from deepac.utils import config_gpus, config_cpus
import tensorflow as tf
from urllib.parse import urlparse
import requests
import json
import wget
import hashlib


class RemoteLoader:
    def __init__(self, remote_repo_url):
        if remote_repo_url is None:
            self.remote_repo_url = "https://zenodo.org/api/records/4456008"
        else:
            self.remote_repo_url = remote_repo_url

    def fetch_models(self, out_dir, do_compile=True, n_cpus=None, n_gpus=None, log_path="logs", training_mode=False,
                     tpu_resolver=None, timeout=15.):
        fetch_dir = os.path.join(out_dir, "latest_weights_configs")
        if not os.path.exists(fetch_dir):
            os.mkdir(fetch_dir)

        r = requests.get(self.remote_repo_url, timeout=timeout)
        model_dict = {}

        if r.ok:
            js = json.loads(r.text)
            files = js['files']
            for f in files:
                link = f['links']['self']
                size = f['size'] / 2 ** 10
                filename = os.path.basename(urlparse(link).path)

                remote_md5 = get_checksum_md5(f['checksum'])
                local_md5 = get_file_md5(os.path.join(fetch_dir, filename))

                if remote_md5 == local_md5:
                    print(f'Found: {filename} size: {get_human_readable_size(size)}')
                else:
                    if os.path.exists(os.path.join(fetch_dir, filename)):
                        print(f'Found: {filename} (incorrect md5 checksum). Deleting...')
                        os.remove(os.path.join(fetch_dir, filename))
                    print(f'Downloading: {filename} size: {get_human_readable_size(size)}')
                    wget.download(link, out=os.path.join(fetch_dir, filename))
                    print()

                if filename.lower().endswith(".h5"):
                    pre, ext = os.path.splitext(filename)
                    model_dict[os.path.join(fetch_dir, filename)] = os.path.join(fetch_dir, pre + ".ini")
            else:
                print('Downloading finished.')
                if do_compile:
                    print('Building downloaded models...')
                    for w in model_dict.keys():
                        model = load_model(model_dict[w], w, n_cpus, n_gpus, log_path, training_mode, tpu_resolver)
                        model.summary()
                        save_path = os.path.basename(w)
                        model.save(os.path.join(out_dir, save_path))
        else:
            print('HTTP error: {}'.format(r.status_code))


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
        return self._load_builtin_model("sensitive", n_cpus, n_gpus, log_path, training_mode, tpu_resolver)

    def load_rapid_model(self, n_cpus=None, n_gpus=None, log_path="logs", training_mode=True, tpu_resolver=None):
        return self._load_builtin_model("rapid", n_cpus, n_gpus, log_path, training_mode, tpu_resolver)

    def _load_builtin_model(self, modelkey, n_cpus=None, n_gpus=None, log_path="logs", training_mode=True,
                            tpu_resolver=None):
        config_path = self.builtin_configs[modelkey]
        weights_path = self.builtin_weights[modelkey]

        return load_model(config_path, weights_path, n_cpus, n_gpus, log_path, training_mode, tpu_resolver)

    def get_sensitive_training_config(self):
        return self._get_builtin_training_config("sensitive")

    def get_rapid_training_config(self):
        return self._get_builtin_training_config("rapid")

    def _get_builtin_training_config(self, modelkey):
        config_path = self.builtin_configs[modelkey]
        print("Loading {}".format(os.path.basename(config_path)))
        config = configparser.ConfigParser()
        config.read(config_path)
        paprconfig = RCConfig(config)

        return paprconfig


def load_model(config_path, weights_path, n_cpus=None, n_gpus=None, log_path="logs", training_mode=True,
               tpu_resolver=None):
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


def get_file_md5(filename):
    if not os.path.exists(filename):
        return "not found"
    with open(filename, 'rb') as f:
        data = f.read()
        return hashlib.md5(data).hexdigest()


def get_checksum_md5(checksum):
    return checksum.split(':')[-1]


def get_human_readable_size(size):
    if size > 1024.0:
        size = size / 2 ** 10
        return f'{size:.1f} MB'
    else:
        return f'{size:.1f} kB'
