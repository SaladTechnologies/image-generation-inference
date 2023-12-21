import os
from __version__ import VERSION
from torch import __version__ as torch_version
from torch.version import cuda as cuda_version
from xformers import __version__ as xformers_version
from triton import __version__ as triton_version
from diffusers import __version__ as diffusers_version
from transformers import __version__ as transformers_version
from sfast import __version__ as sfast_version

version = VERSION

package_versions = {
    "torch": torch_version,
    "cuda": cuda_version,
    "xformers": xformers_version,
    "triton": triton_version,
    "diffusers": diffusers_version,
    "transformers": transformers_version,
    "stable_fast": sfast_version,
}

data_dir = os.getenv("DATA_DIR", "/data")
model_dir = os.path.join(data_dir, "models")
checkpoint_dir = os.path.join(model_dir, "checkpoints")
vae_dir = os.path.join(model_dir, "vae")
lora_dir = os.path.join(model_dir, "lora")
controlnet_dir = os.path.join(model_dir, "controlnet")
image_dir = os.getenv("IMAGE_DIR", os.path.join(data_dir, "images"))

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(vae_dir, exist_ok=True)
os.makedirs(lora_dir, exist_ok=True)
os.makedirs(controlnet_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

cuda_graph = os.getenv("CUDA_GRAPH", "false").lower() == "true"

load_safety_checker = os.getenv("LOAD_SAFETY_CHECKER", "false").lower() == "true"

host = os.getenv("HOST", "*")
port = int(os.getenv("PORT", "1234"))

launch_ckpt = os.getenv("LAUNCH_CHECKPOINT", None)
launch_vae = os.getenv("LAUNCH_VAE", None)

image_storage_strategy = os.getenv("IMAGE_STORAGE_STRATEGY", "disk")

webhooks = {
    "model.loaded": os.getenv("WEBHOOK_MODEL_LOADED", None),
    "model.unloaded": os.getenv("WEBHOOK_MODEL_UNLOADED", None),
    "image.generated": os.getenv("WEBHOOK_IMAGE_GENERATED", None),
    "image.stored": os.getenv("WEBHOOK_IMAGE_STORED", None),
}


identity = {
    "salad_machine_id": os.getenv("SALAD_MACHINE_ID", None),
    "salad_container_group_id": os.getenv("SALAD_CONTAINER_GROUP_ID", None),
}


max_data_dir_size_gb = int(os.getenv("MAX_DATA_DIR_SIZE_GB", "0"))
