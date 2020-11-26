FROM tensorflow/tensorflow:2.3.1

COPY requirements.txt /tmp/

RUN apt-get update && apt-get install -y \
    bedtools \
    ghostscript

RUN pip install -r /tmp/requirements.txt
RUN pip install deepac deepacvir deepacstrain --no-deps

RUN useradd --create-home deepacuser
WORKDIR /home/deepacuser
USER deepacuser

CMD ["deepac"]
