FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER Yannis Assael & Brendan Shillingford

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install python
RUN apt-get update -y &&  apt-get -y install python3 python3-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler && \
    curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
    rm -rf /var/lib/apt/lists/*

# install hdf5
RUN cd /usr/lib/x86_64-linux-gnu && \
    ln -s libhdf5_serial.so.8.0.2 libhdf5.so && \
    ln -s libhdf5_serial_hl.so.8.0.2 libhdf5_hl.so
RUN pip3 install h5py

# Clone torch (and package) repos:
RUN mkdir -p /opt && git clone https://github.com/torch/distro.git /opt/torch --recursive

# Run installation script
RUN cd /opt/torch && ./install.sh -b

# Export environment variables manually
ENV TORCH_DIR /opt/torch/pkg/torch/build/cmake-exports/
ENV LUA_PATH '/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/opt/torch/install/share/lua/5.1/?.lua;/opt/torch/install/share/lua/5.1/?/init.lua;./?.lua;/opt/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
ENV LUA_CPATH '/root/.luarocks/lib/lua/5.1/?.so;/opt/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH /opt/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH /opt/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH /opt/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH '/opt/torch/install/lib/?.so;'$LUA_CPATH

# Install torch packages
RUN luarocks install totem && \
    luarocks install https://raw.githubusercontent.com/deepmind/torch-hdf5/master/hdf5-0-0.rockspec && \
    luarocks install unsup && \
    luarocks install csvigo && \
    luarocks install loadcaffe && \
    luarocks install classic && \
    luarocks install pprint && \
    luarocks install class && \
    luarocks install image && \
    luarocks install nninit && \
    luarocks install optnet && \
    luarocks install https://raw.githubusercontent.com/deepmind/torch-distributions/master/distributions-0-0.rockspec


# install original Baidu warp-ctc, since nnx needs that (conflicts with ours in nnob)
# RUN luarocks install https://raw.githubusercontent.com/baidu-research/warp-ctc/master/torch_binding/rocks/warp-ctc-scm-1.rockspec
RUN luarocks install https://raw.githubusercontent.com/iassael/warp-ctc/master/torch_binding/rocks/warp-ctc-scm-1.rockspec
# install nnx for nn.CTCCriterion
RUN luarocks install https://raw.githubusercontent.com/clementfarabet/lua---nnx/master/nnx-0.1-1.rockspec
# fb debugger, with deps removed:
RUN luarocks install https://raw.githubusercontent.com/bshillingford/fbdebugger-minimal/master/fbdebugger-standalone-1.rockspec


WORKDIR /project
ADD . /project/
