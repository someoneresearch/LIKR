FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /workspace


RUN pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
RUN pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html

RUN pip install torch_geometric
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    gnupg2 \
    libx11-6 \
    libx11-dev \
    tk-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pyyaml \
    tqdm \
    tensorboardX \
    scikit-learn \
    ogb \
    wandb \
    easydict

RUN pip install gensim scikit-learn
RUN pip install recbole==1.1.1
RUN pip install openai==0.27.2
RUN pip install pylcs
RUN pip install replicate==0.0.1a10

COPY . /workspace
