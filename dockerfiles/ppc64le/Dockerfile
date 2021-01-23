FROM ibmcom/powerai:1.7.0-tensorflow-ubuntu18.04-py37

COPY requirements.txt .

RUN sudo apt-get update && sudo apt-get install -y \
    bedtools \
    g++ \
    gcc \
    ghostscript \
    libz-dev \
    python3-dev

SHELL ["/bin/bash", "-c"]

RUN source /opt/anaconda/bin/activate && conda activate wmlce && pip install -r requirements.txt
RUN source /opt/anaconda/bin/activate && conda activate wmlce && pip install deepac==0.13.5 deepacstrain==0.2.1 deepacvir==0.2.2 --no-deps

ENV PYTHONUNBUFFERED 1
RUN sudo mkdir /deepac
WORKDIR /deepac

CMD ["bash", "-c", "source /opt/anaconda/bin/activate && conda activate wmlce && deepac"]
