FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

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
  wget \
  jq

# We need the latest pip
RUN pip install --upgrade --no-cache-dir pip

# Install dependencies
COPY requirements.txt .
ENV TORCH_CUDA_ARCH_LIST=All
ENV MAX_JOBS=4
ENV LD_PRELOAD=libtcmalloc.so
RUN pip install --upgrade --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
RUN WITH_CUDA=0 pip install -v -U git+https://github.com/chengzeyi/stable-fast.git@v1.0.4#egg=stable-fast

COPY ./app ./

ENV HOST='*'
ENV PORT=1234

RUN wget https://raw.githubusercontent.com/SaladTechnologies/stable-diffusion-configurator/main/configure -O configure && chmod +x configure && echo "Config Utility Installed."

COPY entrypoint.sh /entrypoint.sh

CMD ["/entrypoint.sh"]