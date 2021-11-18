FROM tensorflow/tensorflow:2.4.1-gpu

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    bedtools \
    ghostscript

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install deepac==0.14.0 deepacstrain==0.2.1 deepacvir==0.2.2 --no-deps

RUN mkdir /deepac
WORKDIR /deepac

ENV PYTHONUNBUFFERED 1

CMD ["deepac"]