# 使用官方的Anaconda基础镜像
# FROM ubuntu:18.04
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.12-py3.9.12-cuda11.3.1-u20.04
LABEL maintainer="lehang.yu"
LABEL email="1343787541@qq.com"
LABEL description="Container used for magic sim2real"
# RUN apt-get update && apt-get install -y wget && apt-get install -y bash
SHELL [ "/bin/bash", "-c" ]
ENV SHELL=/bin/bash

RUN apt-get update && apt-get install -y wget
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
RUN rm -rf Miniconda3-latest-Linux-x86_64.sh
RUN /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
RUN /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

# 将Conda添加到PATH环境变量中

# COPY /usr/local/cuda-11.1 /usr/local/cuda-11.1
ENV CUDA_HOME=/usr/local/cuda/
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

ENV PATH=/opt/conda/bin:$PATH
RUN source ~/.bashrc

# 创建Conda环境
RUN conda init bash
RUN source ~/.bashrc
RUN conda create -n openpcdet python=3.8 -y

# 将项目代码复制到镜像根目录下
COPY . /OpenPCDet
WORKDIR /OpenPCDet

#降级pip
RUN conda run -n openpcdet pip install pip==24.1.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN conda run -n openpcdet pip install numpy==1.23.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN conda run -n openpcdet pip install torch-1.8.1+cu111-cp38-cp38-linux_x86_64.whl
RUN conda run -n openpcdet pip install torchvision-0.9.1+cu111-cp38-cp38-linux_x86_64.whl
RUN conda run -n openpcdet pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN conda run -n openpcdet pip install spconv-cu111 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN conda run -n openpcdet python setup.py develop
RUN conda run -n openpcdet pip install torch_scatter-2.0.7-cp38-cp38-linux_x86_64.whl
RUN conda run -n openpcdet pip install pillow==10.0.0
RUN conda run -n openpcdet pip install av2 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN conda run -n openpcdet pip install numba==0.57.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# COPY /data2/yulehang/anaconda3/envs/openpcdet2 /opt/conda/envs/

# 激活环境
RUN echo "source activate openpcdet" > ~/.bashrc
ENV PATH /opt/conda/envs/openpcdet/bin:$PATH
RUN source ~/.bashrc

WORKDIR /OpenPCDet
CMD ["bash"]
