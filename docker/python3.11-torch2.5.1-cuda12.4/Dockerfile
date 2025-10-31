FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

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
    pandas==2.2.3 \
    scipy==1.15.2 \
    biopython==1.85 \
    scikit-learn==1.6.1 \
    matplotlib==3.10.1 \
    transformers==4.49.0 \
    xgboost-cpu==2.1.4 \
    git+https://github.com/dridk/PyVCF3.git@1.0.4

RUN pip install --no-cache-dir --no-build-isolation \
    causal-conv1d==1.5.0.post8 \
    mamba-ssm==2.2.4

CMD ["bash"]

