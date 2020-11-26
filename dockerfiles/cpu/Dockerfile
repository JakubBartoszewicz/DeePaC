FROM tensorflow/tensorflow:2.3.1

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    bedtools \
    ghostscript \
    libz-dev

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install deepac==0.13.3 deepacvir==0.2.2 deepacstrain==0.2.1 --no-deps

RUN useradd --create-home deepacuser
WORKDIR /home/deepacuser
USER deepacuser
ENV PYTHONUNBUFFERED 1

CMD ["deepac"]