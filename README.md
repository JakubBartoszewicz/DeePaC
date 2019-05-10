<!-- {#mainpage} -->

# DeePaC

DeePaC is a python package and a CLI tool for predicting labels (e.g. pathogenic potentials) from short DNA sequences (e.g. Illumina 
reads) with reverse-complement neural networks. For details, see our preprint on bioRxiv: 
<https://www.biorxiv.org/content/10.1101/535286v2>.

Documentation can be found here:
<https://rki_bioinformatics.gitlab.io/DeePaC/>.


## Installation

### Recommended: virtual environment

We recomment setting up an isolated `conda` environment:
```
conda create -n my_env
conda activate my_env
```

or a `virtualenv`:
```
virtualenv --system-site-packages my_env
source my_env/bin/activate
```


### With conda
 [![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](http://bioconda.github.io/recipes/deepac/README.html)
 
You can install DeePaC with `bioconda`. Set up the [bioconda channel](
<https://bioconda.github.io/index.html#set-up-channels>) first, and then:
```
conda install deepac
```

### With pip

You can also install DeePaC with `pip`:
```
pip install deepac
```

### GPU support

To use GPUs, you need to install the GPU version of TensorFlow. In conda, install tensorflow-gpu from the `defaults` channel before deepac:
```
conda remove tensorflow
conda install -c defaults tensorflow-gpu 
conda install deepac
```

If you're using `pip`, you need to install CUDA and CuDNN first (see TensorFlow installation guide for details). Then
you can do the same as above:
```
pip uninstall tensorflow
pip install tensorflow-gpu
```

### Optional: run tests
Optionally, you can run explicit tests of your installation. Note that it may take some time on a CPU.
```
deepac test 
# Test using a GPU
deepac test -g 1
```

### Help

To see help, just use
```
deepac --help
deepac predict --help
deepac train --help
# Etc.
```

## Prediction

You can predict pathogenic potentials with one of the built-in models out of the box:
```
# A rapid CNN (trained on IMG/M data)
deepac predict -r input.fasta
# A sensitive LSTM (trained on IMG/M data)
deepac predict -s input.fasta
# With GPU support
deepac predict -s -g 1 input.fasta
```

The rapid and the sensitive models are trained to predict pathogenic potentials of novel bacterial species.
For details, see <https://www.biorxiv.org/content/10.1101/535286v2>.

To quickly filter your data according to predicted pathogenic potentials, you can use:
```
deepac predict -r input.fasta
deepac filter input.fasta input_predictions.npy -t 0.5
```
Note that after running `predict`, you can use the `input_predictions.npy` to filter your fasta file with different
thresholds. You can also add pathogenic potentials to the fasta headers in the output files:
```
deepac filter input.fasta input_predictions.npy -t 0.75 -p -o output-75.fasta
deepac filter input.fasta input_predictions.npy -t 0.9 -p -o output-90.fasta
```

## Preprocessing

For more complex analyzes, it can be useful to preprocess the fasta files by converting them to binary numpy arrays. Use:
```
deepac preproc preproc_config.ini
```
See the `config_templates` directory of the GitLab repository (https://gitlab.com/rki_bioinformatics/DeePaC/) for a sample configuration file.

## Training
You can use the built-in architectures to train a new model:
```
deepac train -r -g 1 -T train_data.npy -t train_labels.npy -V val_data.npy -v val_labels.npy
deepac train -s -g 1 -T train_data.npy -t train_labels.npy -V val_data.npy -v val_labels.npy

```

To train a new model based on you custom configuration, use
```
deepac train -c nn_train_config.ini
```

If you train an LSTM on a GPU, a CUDNNLSTM implementation will be used. To convert the resulting model to be 
CPU-compatible, use `deepac convert`. You can also use it to save the weights of a model, or recompile a model 
from a set of weights to use it with a different Python binary.

## Evaluation

To evaluate a trained model, use
```
# Read-by-read performance
deepac eval -r eval_config.ini
# Species-by-species performance
deepac eval -s eval_species_config.ini
# Ensemble performance
deepac eval -e eval_ens_config.ini
```
See the configs directory for sample configuration files. Note that `deepac eval -s` requires precomputed predictions 
and a csv file with a number of DNA reads for each species in each of the classes.

## Supplementary data and scripts

In the supplement_paper directory you can find the R scripts and data files used in the paper for dataset preprocessing and benchmarking.