FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
  curl \
  git \
  unzip \
  libgl1 \
  libglib2.0-0 \
  build-essential \
  libgoogle-perftools-dev \
  wget

# We need the latest pip
RUN pip install --upgrade --no-cache-dir pip

# Install dependencies
COPY requirements.txt .
ENV TORCH_CUDA_ARCH_LIST=All
ENV MAX_JOBS=4
ENV LD_PRELOAD=libtcmalloc.so
RUN pip install --upgrade --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install -v -U git+https://github.com/chengzeyi/stable-fast.git@v0.0.15#egg=stable-fast

COPY ./app ./

ENV HOST='[::]'
ENV PORT=1234

CMD ["python", "main.py"]