from tensorflow/tensorflow:latest-gpu

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /code

RUN pip install --upgrade pip
RUN pip install --upgrade tensorflow-hub
RUN pip install tensorflow_text

ADD . /code/