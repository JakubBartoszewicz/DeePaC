<!-- {#mainpage} -->

# DeePaC

DeePaC is a python package and a CLI tool for predicting labels (e.g. pathogenic potentials) from short DNA sequences (e.g. Illumina 
reads) with interpretable reverse-complement neural networks. For details, see our preprint on bioRxiv: 
<https://www.biorxiv.org/content/10.1101/535286v3> and the paper in *Bioinformatics*: <https://doi.org/10.1093/bioinformatics/btz541>.
For details regarding the interpretability functionalities of DeePaC, see the preprint here: <https://www.biorxiv.org/content/10.1101/2020.01.29.925354v2>

Documentation can be found here:
<https://rki_bioinformatics.gitlab.io/DeePaC/>.

## Plug-ins
### DeePaC-strain
Basic version of DeePaC comes with built-in models trained to predict pathogenic potentials of NGS reads originating from
novel *bacteral species*. If you want to predict pathogenicity of novel *strains* of *known* species, try the DeePaC-strain plugin available here:
<https://gitlab.com/JakubBartoszewicz/DeePaC-strain>. 

### DeePaC-vir
If you want to detect novel human viruses, try the DeePaC-vir plugin: <https://gitlab.com/JakubBartoszewicz/DeePaC-vir>. 

### DeePaC-Live
If you want to run the predictions in real-time during an Illumina sequencing run, try DeePaC-Live: <https://gitlab.com/dacs-hpi/deepac-live>. 


## Installation

### Recommended: set up an environment

We recomment setting up an isolated `conda` environment:
```
conda create -n my_env python=3.6
conda activate my_env
```

or, alternatively, a `virtualenv`:
```
virtualenv --system-site-packages my_env
source my_env/bin/activate
```


### With conda (recommended)
 [![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](http://bioconda.github.io/recipes/deepac/README.html)
 
You can install DeePaC with `bioconda`. Set up the [bioconda channel](
<https://bioconda.github.io/user/install.html#set-up-channels>) first, and then:
```
conda install deepac
```

If you want to install the plugins as well, use:

```
conda install deepacvir deepacstrain
```

### With pip

You can also install DeePaC with `pip`:
```
pip install deepac
```
Note: TensorFlow 2.0 is not yet supported.

If you want to install the plugins, use:

```
pip install deepacvir deepacstrain
```

### GPU support

To use GPUs, you need to install the GPU version of TensorFlow. In conda, install tensorflow-gpu before deepac:
```
conda remove tensorflow
conda install tensorflow-gpu
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
# Run standard tests
deepac test
# Run quick tests (eg. on CPUs)
deepac test -q
# Test using a GPU
deepac test -g 1
# Test explainability and gwpa workflows
deepac test -xp
# Full tests
deepac test -a -g 1
# Full quick tests (eg. on GPUs with limited memory)
deepac test -aq -g 1
```

### Help

To see help, just use
```
deepac --help
deepac predict --help
deepac train --help
# Etc.
```

## Basic use: prediction

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
For details, see <https://doi.org/10.1093/bioinformatics/btz541> or <https://www.biorxiv.org/content/10.1101/535286v3>.

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

## Advanced use
### Config templates
To get the config templates in the current working directory, simply use:
```
deepac templates
```
### Preprocessing

For more complex analyzes, it can be useful to preprocess the fasta files by converting them to binary numpy arrays. Use:
```
deepac preproc preproc_config.ini
```
See the `config_templates` directory of the GitLab repository (https://gitlab.com/rki_bioinformatics/DeePaC/) for a sample configuration file.

### Training
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

### Evaluation

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

## Intepretability workflows
### Filter visualization
To find the most relevant filters and visualize them, use the following minimum workflow: 
```
# Calculate filter and nucleotide contibutions (partial Shapley values) for the first convolutional layer
# using mean-centered weight matrices and "easy" calculation mode
deepac explain fcontribs -m model.h5 -eb -t test_data.npy -N test_nonpatho.fasta -P test_patho.fasta -o fcontribs 

# Create filter ranking
deepac explain franking -f fcontribs/filter_scores -y test_labels.npy -p test_predictions.npy -o franking

# Prepare transfac files for filter visualization (weighted by filter contribution)
deepac explain fa2transfac -i fcontribs/fasta -o fcnotribs/transfac -w -d fcontribs/filter_scores

# Visualize nucleotide contribution sequence logos
deepac explain xlogos -f fcontribs/fasta -s fcontribs/filter_scores -l fcnotribs/transfac -t train_data.npy -o xlogos
```
You can browse through other supplementary functionalities and parameters by checking the help:
```
deepac explain -h
deepac explain fcontribs -h
deepac explain xlogos -h
# etc.
```

### Genome-wide phenotype potential analysis (GWPA)
To find interesting regions of a whole genome, use this workflow to generate nucleotide-resolution maps of
predicted phenotype potentials and nucleotide contributions:
```
# Fragment the genomes into pseudoreads
deepac gwpa fragment -g genomes_fasta -o fragmented_genomes

# Predict the pathogenic potential of each pseudoread
deepac predict -r -a fragmented_genomes/sample1_fragmented_genomes.npy -o predictions/sample1_pred.npy

# Create bedgraphs of mean pathogenic potential at each position of the genome
# Can be visualized in IGV
deepac gwpa genomemap -f fragmented_genomes -p predictions -g genomes_genome -o bedgraph

# Rank genes by mean pathogenic potential
deepac gwpa granking -p bedgraph -g genomes_gff -o granking

# Create bedgraphs of mean nuclotide contribution at each position of the genome
# Can be visualized in IGV
deepac gwpa ntcontribs -m model.h5 -f fragmented_genomes -g genomes_genome -o bedgraph_nt
```
You can browse through other supplementary functionalities and parameters by checking the help:
```
deepac gwpa -h
deepac gwpa genomemap -h
deepac gwpa ntcontribs -h
# etc.
```
### Filter enrichment analysis
Finally, you can check for filter enrichment in annotated genes or other genomic features:
```
# Get filter activations, genome-wide
deepac gwpa factiv -m model.h5 -t fragmented_genomes/sample1_fragmented_genomes.npy -f fragmented_genomes/sample1_fragmented_genomes.fasta -o factiv

# Check for enrichment within annotated genomic features
deepac gwpa fenrichment -i factiv -g genomes_gff/sample1.gff -o fenrichment
```
## Supplementary data and scripts
Datasets are available here: <https://doi.org/10.5281/zenodo.3678562> (bacteria) and here: <https://doi.org/10.5281/zenodo.3630803> (viruses).
In the supplement_paper directory you can find the R scripts and data files used in the papers for dataset preprocessing and benchmarking.

## Cite us
If you find DeePaC useful, please cite:

```
@article{10.1093/bioinformatics/btz541,
    author = {Bartoszewicz, Jakub M and Seidel, Anja and Rentzsch, Robert and Renard, Bernhard Y},
    title = "{DeePaC: predicting pathogenic potential of novel DNA with reverse-complement neural networks}",
    journal = {Bioinformatics},
    year = {2019},
    month = {07},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btz541},
    url = {https://doi.org/10.1093/bioinformatics/btz541},
    eprint = {http://oup.prod.sis.lan/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btz541/28971344/btz541.pdf},
}

@article {Bartoszewicz2020.01.29.925354,
    author = {Bartoszewicz, Jakub M. and Seidel, Anja and Renard, Bernhard Y.},
    title = {Interpretable detection of novel human viruses from genome sequencing data},
    elocation-id = {2020.01.29.925354},
    year = {2020},
    doi = {10.1101/2020.01.29.925354},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2020/02/01/2020.01.29.925354},
    eprint = {https://www.biorxiv.org/content/early/2020/02/01/2020.01.29.925354.full.pdf},
    journal = {bioRxiv}
}

```