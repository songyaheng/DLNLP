# 构建tensorflow 环境
FROM ubuntu

MAINTAINER yahengsong <yahengsong@foxmail.com>


RUN apt-get install wget \
    && apt-get install lrzsz \
    && apt-get install unzip \
    && apt-get install zip \
    && apt-get install vim \
    && apt-get install gcc \
    && apt-get install gcc-c++ \
    && apt-get install automake \
    && apt-get install autoconf \
    && apt-get install libtool \
    && apt-get install make \
    && apt-get install openssl-static \
    && apt-get install zlib-devel \
    && apt-get install bzip2-devel \
    && apt-get install openssl-devel \
    && apt-get install ncurses-devel \
    && apt-get install sqlite-devel \
    && apt-get install readline-devel \
    && apt-get install tk-devel \
    && apt-get install gdbm-devel \
    && apt-get install libpcap-devel \
    && apt-get install xz-devel

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
