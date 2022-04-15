FROM nvcr.io/nvidia/tritonserver:21.05-py3
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ADD . /code
ENV PYTHONPATH "${PYTHONPATH}:/code/spvnas-dev"
# ADD model.ckpt model.ckpt
RUN apt-get update 
RUN apt-get -y install libsparsehash-dev
RUN apt-get -y install python3-dev 
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install scikit-learn
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=7.5 pip3 install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
RUN pip3 install torchpack
