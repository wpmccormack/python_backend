FROM nvcr.io/nvidia/tritonserver:21.05-py3
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV PYTHONPATH "${PYTHONPATH}:/code/spvnas-dev"
# Install base utilities
RUN apt-get update && \
    apt-get install -y wget
RUN apt-get -y install libsparsehash-dev
RUN apt-get -y install python3-dev 
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install scikit-learn
RUN pip3 install torch==1.12 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=7.5 pip3 install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
RUN pip3 install torchpack
RUN pip3 install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
RUN pip3 install pandas