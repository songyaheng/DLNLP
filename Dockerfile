# 构建tensorflow 环境
FROM centos

MAINTAINER yahengsong <yahengsong@foxmail.com>


RUN yum update -y
RUN yum install -y wget
RUN yum install -y lrzsz
RUN yum install -y unzip
RUN yum install -y zip
RUN yum install -y vim

RUN yum -y install epel-release \
        && yum install -y python34 \
        && yum install -y python-pip \
        && pip3 install --upgrade pip \
        && pip3 install -y numpy \
        && pip3 install -y tensorflow==1.7.0 \
        && ln -s /usr/bin/pip3 /usr/bin/pip
# 映射端口
EXPOSE 8888

# 添加文件
ADD vimrc /root/.vimrc