FROM nvcr.io/nvidia/tritonserver:21.06-py3
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ADD model.ckpt model.ckpt
RUN apt-get update 
RUN apt-get -y install libsparsehash-dev
RUN apt-get -y install python3-dev
RUN pip3 install torch
RUN pip3 install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
RUN pip3 install numpy pandas sklearn pytorch-lightning hydra
RUN git clone https://github.com/mit-han-lab/calo-cluster.git
RUN pip install -e calo-cluster