# CUDA + cuDNN, совместимо с PyTorch
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# --- system deps ---
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# python / pip defaults
RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# upgrade tooling
RUN pip install --upgrade pip setuptools wheel

# --- app ---
WORKDIR /app

# deps first (for docker cache)
COPY requirements.txt .

# PyTorch с CUDA (НЕ класть в requirements.txt)
RUN pip install \
    torch==2.1.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# остальное
RUN pip install --no-cache-dir -r requirements.txt

# код проекта
COPY . .

# Datasphere сам задаёт команду, оставляем shell
CMD ["/bin/bash"]
