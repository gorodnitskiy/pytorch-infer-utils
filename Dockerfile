ARG BUILD_VERSION="21.05"
FROM nvcr.io/nvidia/tensorrt:${BUILD_VERSION}-py3

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

ARG WORKDIR_PATH="/workspace/"
WORKDIR ${WORKDIR_PATH}

#################################################

RUN mkdir pytorch_infer_utils
COPY setup.cfg setup.py ./pytorch_infer_utils/
COPY pytorch_infer_utils/ ./pytorch_infer_utils/pytorch_infer_utils/
RUN python3 -m pip install --no-cache-dir pip==21.3.1 && \
    cd pytorch_infer_utils/ && \
    python3 -m pip install . && \
    cd ../ && \
    rm -rf pytorch_infer_utils/

#################################################

# Use git clone for onnx-tensorrt installation
# due to issue with import onnx_tensorrt in onnx_tensorrt setup.py
# https://github.com/onnx/onnx-tensorrt/blob/e9456d57605c883cdf985e634ab483e2c1500bb1/setup.py#L5
RUN git clone https://github.com/onnx/onnx-tensorrt.git && \
    cd onnx-tensorrt/ && \
    cp -r onnx_tensorrt/ /usr/local/lib/python3.8/dist-packages/ && \
    cd ../ && \
    rm -rf onnx-tensorrt/