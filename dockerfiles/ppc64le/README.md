# Installation on IBM Power Systems
Please consult the [IMB PowerAI license](https://github.com/IBM/powerai/tree/master/containers/1.7.0).
## With conda & pip (experimental)
First, add the IBM conda channel to install tensorflow (and cython):
```
conda config --add channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda
conda install tensorflow cython
```

Download [requirements.txt for IBM PowerAI](https://gitlab.com/rki_bioinformatics/DeePaC/-/blob/master/dockerfiles/ppc64le/requirements.txt)
and install with pip (requires libz-dev to be present on your machine:
```
pip install -r requirements.txt
```

Finally, install deepac:
```
pip install deepac --no-deps
```
Note that bedtools and ghostscript are required for the interpretability workflows (ask your admin to install them if this is necessary).
You can train your models and run inference without them.

### With Docker (experimental)

Alternatively, if you are allowed to use the official [IBM PowerAI containers](https://hub.docker.com/r/ibmcom/powerai/) (see [license](https://github.com/IBM/powerai/tree/master/containers/1.7.0)), 
you can use this convenience [Dockerfile](https://gitlab.com/rki_bioinformatics/DeePaC/-/blob/master/dockerfiles/ppc64le/Dockerfile), based on the IBM Tensorflow image to build your own DeePaC container. 

```
docker build -t deepac:0.13.5-ppc64le .
```

Verify that everything worked. You have to accept the [license](https://github.com/IBM/powerai/tree/master/containers/1.7.0) at every run:
```
docker run --rm --env LICENSE=yes deepac:0.13.5-ppc64le
```

To run the container as your user (and bind-mount your working directory), you have to activate the wmlce conda environment:
```
docker run -v $(pwd):/deepac -u $(id -u):$(id -g) --rm --env LICENSE=yes deepac:0.13.5-ppc64le bash -c "source /opt/anaconda/bin/activate && conda activate wmlce && deepac test -q"
```
Modify the last deepac command to run the desired deepac workflow.

Alternatively, you can run in in interactive mode and set everything up yourself:
```
docker run -it -v $(pwd):/deepac -u $(id -u):$(id -g) --rm --env LICENSE=yes deepac:0.13.5-ppc64le bash
```
