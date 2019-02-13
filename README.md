<!-- {#mainpage} -->

# DeePaC

DeePaC is a python package for predicting labels (e.g. pathogenic potentials) from short DNA sequences (e.g. Illumina reads) with reverse-complement neural networks. For details, see our preprint on bioRxiv: https://www.biorxiv.org/content/10.1101/535286v2

## Installation

You can just use DeePaC as a set of scripts. Executable scripts include `nn_train.py`, `eval.py`, `eval_ens.py`, `eval_species.py`, `preproc.py`, `convert_cudnn.py` and `filter.py`.
You must supply a config file to all of them (except fot `filter.py` -- coming soon). For help, use the `-h` flag. `pip` installation and a more in-depth tutorial will be available shortly. Remember to activate your virtual enviroment (with all the dependencies installed) before using DeePaC:
```
source my_env/bin/activate
```

## Prediction

To quickly filter your data according to predicted pathogenic potentials, you can use:
```
filter.py input.fasta model.h5 -t 0.5 -z 0.75 -u 0.9
```
Note that you can use three different thresholds at once.

## Preprocessing

For more complex analyzes, we rocommend preprocessing the fasta files by converting them to binary numpy arrays. Use:
```
preproc.py preproc_config.ini
```
See the configs directory for a sample configuration file.

## Evaluation

To evaluate a trained model, use
```
eval.py eval_config.ini
eval_ens.py eval_ens_config.ini
eval_species.py eval_species_config.ini
```
See the configs directory for a sample configuration file. Note that `eval_species.py` requires precomputed predictions and a csv file with a number of DNA reads for each species in each of the classes.

## Training
To train a new model, use
```
nn_train.py nn_train_config.ini
```

If you train an LSTM on a GPU, a CUDNNLSTM implementation will be used. To convert the resulting model to be CPU-compatible, use `convert_cudnn.py`. You can also use it to save the weights of a model, or recompile a model from a set of weights to use it with a different Python binary.

## Dependencies
DeePaC requires Tensorflow, Keras, Biopython, Scikit-learn and matplotlib.
