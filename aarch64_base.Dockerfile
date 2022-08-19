# syntax=docker/dockerfile:experimental
FROM toolboc/jetson-nano-l4t-cuda-cudnn:latest

# System library
RUN apt-get update && \
    apt-get install -y python3.6-dev python3-pip wget rsync git build-essential \
                    libopenblas-base libopenmpi-dev libomp-dev \
                    libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev

# Replace pip3 with pip
RUN pip3 install --upgrade pip
RUN rm /usr/bin/pip3 /usr/local/bin/pip3
RUN ln -s $(which pip) /usr/bin/pip3

# Install requirements
RUN pip install --upgrade setuptools
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Install pytorch 1.1.0 for aarch64
RUN wget https://nvidia.box.com/shared/static/mmu3xb3sp4o8qg9tji90kkxl1eijjfc6.whl -O torch-1.1.0-cp36-cp36m-linux_aarch64.whl && \
    pip install numpy torch-1.1.0-cp36-cp36m-linux_aarch64.whl

# Install torchvision that is compatible with pytorch
ENV BUILD_VERSION 0.3.0
RUN git clone --branch v0.3.0 https://github.com/pytorch/vision torchvision && \
    cd torchvision && \
    python3 setup.py install --user

# Install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI && git checkout aca78bcd6b4345d25405a64fdba1120dfa5da1ab \
 && python3 setup.py build_ext install

# Install apex
RUN git clone https://github.com/NVIDIA/apex.git \
 && cd apex && git checkout 4ff153cd50e4533b21dc1fd97c0ed609e19c4042 \
 && python3 setup.py install --cuda_ext --cpp_ext

# Install PyTorch Detection
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
RUN git clone https://github.com/facebookresearch/maskrcnn-benchmark.git \
 && cd maskrcnn-benchmark && git checkout a44d65dcdb9c9098a25dd6b23ca3c36f1b887aba\
 && python3 setup.py build develop

# Copy weight in a separated layer
COPY data /app/data

# Fix sklearn can't allocate memory in static TLS block
ENV LD_PRELOAD /usr/lib/aarch64-linux-gnu/libgomp.so.1

# Copy the rest of the code
RUN --mount=target=/ctx rsync -r --exclude='data'\
                                 /ctx/ /app/

WORKDIR /app