# 构建tensorflow 环境
FROM ubuntu:16.04

MAINTAINER yahengsong <yahengsong@foxmail.com>


RUN apt-get update \
    && apt-get install -y wget \
    && apt-get install -y lrzsz \
    && apt-get install -y unzip \
    && apt-get install -y zip \
    && apt-get install -y vim \
    && apt-get install -y gcc \
    && apt-get install -y g++ \
    && apt-get install -y automake \
    && apt-get install -y autoconf \
    && apt-get install -y libtool \
    && apt-get install -y make \
    && apt-get install -y openssl \
    && apt-get install -y libssl-dev \
    && apt-get install -y zlib-devel \
    && apt-get install -y bzip2-devel \
    && apt-get install -y openssl-devel \
    && apt-get install -y ncurses-devel \
    && apt-get install -y sqlite-devel \
    && apt-get install -y readline-devel \
    && apt-get install -y tk-devel \
    && apt-get install -y gdbm-devel \
    && apt-get install -y libpcap-devel \
    && apt-get install -y xz-devel

RUN wget https://www.python.org/ftp/python/3.6.0/Python-3.6.0.tar.xz \
    && tar -xvf Python-3.6.0.tar.xz \
    && cd Python-3.6.0 \
    && mkdir -p /usr/local/python3 \
    && ./configure --prefix=/usr/local/python3 \
    && make \
    && make install \
    && rm -rf Python-3.6.0* \
    && ln -s /usr/local/python3/bin/python3 /usr/bin/python3 \
    && ln -s /usr/local/python3/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip \
    && pip --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.7.0-cp36-cp36m-linux_x86_64.whl

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

WORKDIR /root
