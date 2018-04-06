# 构建tensorflow 环境
FROM centos:7.2.1511

MAINTAINER yahengsong <yahengsong@foxmail.com>

# 安装python 和 pip
RUN apt update \
        && yum install -y python3-pip python3-dev vim \
        && pip3 install --upgrade pip \
        && pip3 install tensorflow==1.7.0 \
        && ln -s /usr/bin/python3 /usr/bin/python \
        && ln -s /usr/bin/pip3 /usr/bin/pip

# 映射端口
EXPOSE 8888

# 添加文件
ADD vimrc /root/.vimrc