<!-- {#mainpage} -->

# DeePaC

DeePaC is a python package for predicting labels (e.g. pathogenic potentials) from short DNA sequences (e.g. Illumina 
reads) with reverse-complement neural networks. For details, see our preprint on bioRxiv: 
<https://www.biorxiv.org/content/10.1101/535286v2>.

Documentation can be found here:
<https://rki_bioinformatics.gitlab.io/DeePaC/>.


## Installation

You can install DeePaC with `pip`, and use it an a python package, or a CLI tool.
Remember to activate your virtual environment (with the dependencies installed) before using DeePaC:
```
source my_env/bin/activate
pip install deepac
```

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

## Training
To train a new model, use
```
deepac train nn_train_config.ini
```

If you train an LSTM on a GPU, a CUDNNLSTM implementation will be used. To convert the resulting model to be 
CPU-compatible, use `deepac convert`. You can also use it to save the weights of a model, or recompile a model 
from a set of weights to use it with a different Python binary.

## Dependencies
DeePaC requires Tensorflow, Keras, Biopython, Scikit-learn and matplotlib. Python 3.4+ is supported.
