FROM tensorflow/tensorflow:latest-gpu
# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libtcmalloc-minimal4 \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py
RUN pip install \
        h5py \
        pyyaml \
        requests \
        Pillow \
        tensorflow-gpu \
        tensorflow-probability \
        tf_agents \
        scipy \
        numpy \
        pandas \
        matplotlib \
        keras \
        tqdm \
# --- DO NOT EDIT OR DELETE BETWEEN THE LINES --- #
# These lines will be edited automatically by parameterized_docker_build.sh. #
# COPY _PIP_FILE_ /
# RUN pip --no-cache-dir install /_PIP_FILE_
# RUN rm -f /_PIP_FILE_
# Install TensorFlow GPU version.
# --- ~ DO NOT EDIT OR DELETE BETWEEN THE LINES --- #
# RUN ln -s /usr/bin/python3 /usr/bin/python#
# Set up our notebook config.s
# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.