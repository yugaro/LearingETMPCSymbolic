FROM nvidia/cuda:11.0-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    vim \
    python3\
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    libgl1-mesa-dev \
    tzdata \
    libx11-dev \
    python3-tk \
    x11-xserver-utils
RUN pip3 install -U pip && pip install \
    numpy \
    opencv-python opencv-contrib-python \
    matplotlib \
    progressbar2 \
    jupyterlab \
    torch torchvision \
    scipy
RUN jupyter serverextension enable --py jupyterlab
WORKDIR /
CMD ["jupyter","lab","--ip=0.0.0.0","--allow-root","--LabApp.token=''"]
