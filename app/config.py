import os


data_dir = os.getenv("DATA_DIR", "/data")
model_dir = os.path.join(data_dir, "models")
checkpoint_dir = os.path.join(model_dir, "checkpoints")
vae_dir = os.path.join(model_dir, "vae")
lora_dir = os.path.join(model_dir, "lora")
controlnet_dir = os.path.join(model_dir, "controlnet")
image_dir = os.path.join(data_dir, "images")

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(vae_dir, exist_ok=True)
os.makedirs(lora_dir, exist_ok=True)
os.makedirs(controlnet_dir, exist_ok=True)

cuda_graph = os.getenv("CUDA_GRAPH", "false").lower() == "true"


host = os.getenv("HOST", "*")
port = int(os.getenv("PORT", "1234"))

launch_ckpt = os.getenv("LAUNCH_CHECKPOINT", None)
