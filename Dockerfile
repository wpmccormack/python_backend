FROM nvcr.io/nvidia/tritonserver:22.05-py3
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
RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=7.5 python3 -c "$(curl -fsSL https://raw.githubusercontent.com/mit-han-lab/torchsparse/master/install.py)"
RUN pip3 install torchpack rootpath
RUN pip3 install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
RUN pip3 install pandas