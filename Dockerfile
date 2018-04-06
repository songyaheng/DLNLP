# 构建tensorflow 环境
FROM centos

MAINTAINER yahengsong <yahengsong@foxmail.com>


RUN yum update -y
RUN yum install -y wget
RUN yum install -y lrzsz
RUN yum install -y unzip
RUN yum install -y zip
RUN yum install -y vim

# 安装python 和 pip
RUN wget https://www.python.org/ftp/python/3.6.0/Python-3.6.0.tgz
RUN tar -xzvf Python-3.6.0.tgz
RUN yum install -y gcc
RUN yum install -y zlib*
RUN yum install -y openssl*
RUN ./Python-3.6.0/configure --prefix=/usr/local \
    && make\
    && make install \
    && ln -s /usr/bin/python3 /usr/bin/python
RUN yum -y install epel-release \
        && yum install -y python-pip \
        && pip3 install --upgrade pip \
        && pip3 install -y numpy \
        && pip3 install -y tensorflow==1.7.0 \
        && ln -s /usr/bin/pip3 /usr/bin/pip
# 映射端口
EXPOSE 8888

# 添加文件
ADD vimrc /root/.vimrc