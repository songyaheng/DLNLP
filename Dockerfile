# 构建tensorflow 环境
FROM centos:7.2.1511

MAINTAINER yahengsong <yahengsong@foxmail.com>


RUN yum install -y wget \
    && mv /etc/yum.repos.d/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo.backup \
    && curl http://mirrors.aliyun.com/repo/Centos-7.repo -o /etc/yum.repos.d/CentOS-Base.repo \
    && yum update -y \
    && yum install -y lrzsz \
    && yum install -y unzip \
    && yum install -y zip \
    && yum install -y vim \
    && yum install -y gcc \
    && yum install -y gcc-c++ \
    && yum install -y automake \
    && yum install -y autoconf \
    && yum install -y libtool \
    && yum install -y make \
    && yum install -y openssl-static \
    && yum install -y zlib-devel \
    && yum install -y bzip2-devel \
    && yum install -y openssl-devel \
    && yum install -y ncurses-devel \
    && yum install -y sqlite-devel \
    && yum install -y readline-devel \
    && yum install -y tk-devel \
    && yum install -y gdbm-devel \
    && yum install -y libpcap-devel \
    && yum install -y xz-devel \
    && yum clean all
RUN wget https://www.python.org/ftp/python/3.6.0/Python-3.6.0.tar.xz \
    && tar -xvf Python-3.6.0.tar.xz \
    && cd Python-3.6.0 \
    && mkdir -p /usr/local/python3 \
    && ./configure --prefix=/usr/local/python3 \
    && make \
    && make install \
    && rm -rf /usr/bin/python \
    && ln -s /usr/local/python3/bin/python3 /usr/bin/python3 \
    && rm -rf /usr/bin/pip \
    && ln -s /usr/local/python3/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip \
    && pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.7.0-cp36-cp36m-linux_x86_64.whl
# 映射端口
EXPOSE 8888

# 添加文件
ADD vimrc /root/.vimrc