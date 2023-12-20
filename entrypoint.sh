#! /usr/bin/env bash

set -eou pipefail

# wget https://raw.githubusercontent.com/SaladTechnologies/stable-diffusion-configurator/main/configure -O configure
# chmod +x configure

data_dir=${DATA_DIR:-/data}
model_dir=$data_dir/models
ckpt_dir=$model_dir/checkpoints
lora_dir=$model_dir/lora
vae_dir=$model_dir/vae/sd1-5/
controlnet_dir=$model_dir/controlnet
export HF_HOME=${HF_HOME:-$data_dir/huggingface}

mkdir -p $ckpt_dir
mkdir -p $lora_dir
mkdir -p $vae_dir
mkdir -p $controlnet_dir

manifest=$(./configure --ckpt-path $ckpt_dir --lora-path $lora_dir --vae-path $vae_dir --controlnet-model-path $controlnet_dir)

load_only=${LOAD_ONLY:-0}

if [[ "$load_only" == "1" ]]; then
  echo "Exiting after loading models"
  exit 0
fi

echo "Starting inference server"
python server.py