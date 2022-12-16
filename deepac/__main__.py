import sklearn # to load libgomp early to solve problems with static TLS on some systems like bioconda mulled tests
import matplotlib.pyplot as plt # also to solve import ordering problems in bioconda mulled tests
import numpy as np
import tensorflow as tf
import random as rn
import os
from deepac.command_line import MainRunner


def main():
    """Run DeePaC CLI."""
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rn.seed(seed)
    modulepath = os.path.dirname(__file__)
    builtin_configs = {"rapid": os.path.join(modulepath, "builtin", "config", "nn-img-rapid-cnn.ini"),
                       "sensitive": os.path.join(modulepath, "builtin", "config", "nn-img-sensitive-lstm.ini")}
    builtin_weights = {"rapid": os.path.join(modulepath, "builtin", "weights", "nn-img-rapid-cnn.h5"),
                       "sensitive": os.path.join(modulepath, "builtin", "weights", "nn-img-sensitive-lstm.h5")}
    remote_repo_urls = ["https://zenodo.org/api/records/4456008", "https://zenodo.org/api/records/5711877"]
    runner = MainRunner(builtin_configs, builtin_weights, remote_repo_urls)
    runner.parse()

    
if __name__ == "__main__":
    main()