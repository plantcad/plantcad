FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/workspace/.hf_home

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    vim \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir \ 
    pandas==2.3.1 \
    scipy==1.15.2 \
    biopython==1.85 \
    scikit-learn==1.7.1 \
    matplotlib==3.10.7 \
    transformers==4.55.0 \
    xgboost-cpu==3.1.1 \
    git+https://github.com/dridk/PyVCF3.git@1.0.4

# Install from github because build from source for pypi package is
# broken and there are no wheels for torch2.7.1 + cuda12.8 + mamba2.2.4;
# e.g. errors like this are very consistent across systems:
# ninja: error: '/tmp/pip-install-8pggvch2/mamba-ssm_c9225c3c1a984f9a9499f8b40b42391a/csrc/selective_scan/selective_scan.cpp', needed by '/tmp/pip-install-8pggvch2/mamba-ssm_c9225c3c1a984f9a9499f8b40b42391a/build/temp.linux-x86_64-cpython-311/csrc/selective_scan/selective_scan.o', missing and no known rule to make it
RUN pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/state-spaces/mamba@v2.2.4 \
    git+https://github.com/Dao-AILab/causal-conv1d@v1.5.0.post8

CMD ["bash"]