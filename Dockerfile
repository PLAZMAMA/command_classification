from tensorflow/tensorflow:latest-gpu

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /code

RUN pip install --upgrade tensorflow-hub

ADD . /code/