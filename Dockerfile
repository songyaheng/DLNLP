# 构建tensorflow 环境
FROM ubuntu

MAINTAINER yahengsong <yahengsong@foxmail.com>


RUN sudo apt-get install wget \
    && sudo apt-get install lrzsz \
    && sudo apt-get install unzip \
    && sudo apt-get install zip \
    && sudo apt-get install vim \
    && sudo apt-get install gcc \
    && sudo apt-get install gcc-c++ \
    && sudo apt-get install automake \
    && sudo apt-get install autoconf \
    && sudo apt-get install libtool \
    && sudo apt-get install make \
    && sudo apt-get install openssl-static \
    && sudo apt-get install zlib-devel \
    && sudo apt-get install bzip2-devel \
    && sudo apt-get install openssl-devel \
    && sudo apt-get install ncurses-devel \
    && sudo apt-get install sqlite-devel \
    && sudo apt-get install readline-devel \
    && sudo apt-get install tk-devel \
    && sudo apt-get install gdbm-devel \
    && sudo apt-get install libpcap-devel \
    && sudo apt-get install xz-devel

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
