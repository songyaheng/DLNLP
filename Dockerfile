# 构建tensorflow 环境
FROM centos

MAINTAINER yahengsong <yahengsong@foxmail.com>

ENV PYTHON_VERSION 3.6.0

RUN yum update -y
RUN yum install -y wget
RUN yum install -y lrzsz
RUN yum install -y unzip
RUN yum install -y zip
RUN yum install -y vim

# 安装python 和 pip
RUN wget https://www.python.org/ftp/python/$PYTHON_PIP_VERSION/Python-$PYTHON_PIP_VERSION.tgz
RUN tar -xzvf Python-$PYTHON_PIP_VERSION.tgz
RUN yum install gcc -y
RUN yum install zlib* -y
RUN yum install openssl* -y
RUN ./Python-$PYTHON_PIP_VERSION/configure --prefix=/usr/local \
    && make\
    && make install
RUN yum -y install epel-release \
        && yum install -y python-pip \
        && pip3 install --upgrade pip \
        && pip3 install numpy \
        && pip3 install tensorflow==1.7.0 \
        && ln -s /usr/bin/python3 /usr/bin/python \
        && ln -s /usr/bin/pip3 /usr/bin/pip
# 映射端口
EXPOSE 8888

# 添加文件
ADD vimrc /root/.vimrc