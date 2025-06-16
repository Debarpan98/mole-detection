# syntax = docker/dockerfile:experimental
FROM tensorflow/tensorflow:2.2.2-gpu

# Cloud sql install
RUN apt-get install wget
RUN wget https://dl.google.com/cloudsql/cloud_sql_proxy.linux.amd64 -O cloud_sql_proxy
RUN chmod +x cloud_sql_proxy


COPY ./dist/playground-datascience-289517-4a213c02630a.json .
COPY ./dist/staging-tools-dermago-d3bd8d788ba5.json .


#LIB GTK
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list && apt-get clean && apt-get update
RUN  apt-get update -y
RUN  apt-get install -y git
RUN  apt-get install libgtk2.0-dev -y
RUN  apt-get install -y libheif-dev
RUN  apt-get install -y libffi-dev  libde265-dev

RUN python -m pip install pip==19.3.1

#Requirements
COPY ./requirements.txt /app/requirements.txt

COPY ./dist/resnext101_64x4d-e77a0586.pth /app/models/resnext101_64x4d-e77a0586.pth
COPY ./src/ /app/src/



WORKDIR /app
RUN pip install --upgrade keyrings.alt
RUN --mount=type=cache,target=/root/.cache/pip pip install flask opencv-python-headless
RUN apt-get install -y libgl1-mesa-dev --fix-missing


COPY ./dist/torch-1.10.0+cu102-cp36-cp36m-linux_x86_64.whl ./
RUN --mount=type=cache,target=/root/.cache/pip  pip install torch-1.10.0+cu102-cp36-cp36m-linux_x86_64.whl --default-timeout=100
 
COPY ./dist/torchvision-0.11.0-cp36-cp36m-manylinux1_x86_64.whl ./
RUN --mount=type=cache,target=/root/.cache/pip  pip install torchvision-0.11.0-cp36-cp36m-manylinux1_x86_64.whl --default-timeout=100

RUN --mount=type=cache,target=/root/.cache/pip pip install  -r requirements.txt --default-timeout=300

COPY . /app
WORKDIR /
RUN apt-get clean

RUN  pip uninstall six -y
RUN pip install six==1.13.0


COPY ./train.sh .
RUN  chmod +x ./train.sh

ENTRYPOINT ["/train.sh"]

