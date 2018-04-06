# 构建tensorflow 环境
FROM centos:7.2.1511

MAINTAINER yahengsong <yahengsong@foxmail.com>

RUN yum update -y
RUN yum -y install wget

# 安装python 和 pip
RUN yum install -y python3-pip \
        && yum install -y python3-dev \
        && yum install -y vim \
        && pip3 install --upgrade pip \
        && pip3 install tensorflow==1.6.0 \
        && ln -s /usr/bin/python3 /usr/bin/python \
        && ln -s /usr/bin/pip3 /usr/bin/pip

# 映射端口
EXPOSE 8888

# 添加文件
ADD vimrc /root/.vimrc