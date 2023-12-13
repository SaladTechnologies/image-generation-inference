import torch
import os
import time
import xformers
import triton
from transformers import __version__ as transformers_version
from diffusers import __version__ as diffusers_version, DiffusionPipeline
from sfast.compilers.stable_diffusion_pipeline_compiler import (
    compile,
    CompilationConfig,
)

torch.backends.cuda.matmul.allow_tf32 = True

print("Torch version:", torch.__version__, flush=True)
print("XFormers version:", xformers.__version__, flush=True)
print("Triton version:", triton.__version__, flush=True)
print("Diffusers version:", diffusers_version, flush=True)
print("Transformers version:", transformers_version, flush=True)
print("CUDA Available:", torch.cuda.is_available(), flush=True)

data_dir = os.getenv("DATA_DIR", "/data")
model_dir = os.path.join(data_dir, "models")
checkpoint_dir = os.path.join(model_dir, "checkpoints")
vae_dir = os.path.join(model_dir, "vae")
lora_dir = os.path.join(model_dir, "lora")
controlnet_dir = os.path.join(model_dir, "controlnet")
diffusers_cache_dir = os.path.join(model_dir, "diffusers_cache")

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(vae_dir, exist_ok=True)
os.makedirs(lora_dir, exist_ok=True)
os.makedirs(controlnet_dir, exist_ok=True)

cuda_graph = os.getenv("CUDA_GRAPH", "false").lower() == "true"

compile_config = CompilationConfig.Default()
compile_config.enable_xformers = True
compile_config.enable_triton = True
compile_config.enable_cuda_graph = cuda_graph
compile_config.memory_format = torch.channels_last

loaded_checkpoints = {}
loaded_vae = {}
loaded_lora = {}
loaded_controlnet = {}


def load_checkpoint(model_name: str) -> DiffusionPipeline:
    """_summary_

    Args:
        model_name (str): Either a HuggingFace model name (e.g. stabilityai/stable-diffusion-xl-base-1.0) or a local checkpoint filename (e.g. dreamshaper_8.safetensors)

    Returns:
        DiffusionPipeline: A DiffusionPipeline that is approriate for the model type
    """
    model_kwargs = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "cache_dir": diffusers_cache_dir,
        "extract_ema": True,
        "device_map": "auto",
        "load_safety_checker": False,
        "use_safetensors": True,
    }

    # if the model_name looks like org/repo, then we assume it's a HuggingFace model
    # otherwise, we assume it's a local model
    start = time.perf_counter()
    if "/" in model_name:
        model = DiffusionPipeline.from_pretrained(
            model_name,
            **model_kwargs,
        )
    else:
        model = DiffusionPipeline.from_single_file(
            os.path.join(checkpoint_dir, model_name),
            **model_kwargs,
        )
    end = time.perf_counter()
    model_class = model.__class__.__name__
    print(f"Loaded {model_class} with {model_name} in {end - start:.2f}s", flush=True)
    return model


def get_checkpoint(model_name: str) -> DiffusionPipeline:
    """_summary_

    Args:
        model_name (str): Either a HuggingFace model name (e.g. stabilityai/stable-diffusion-xl-base-1.0) or a local checkpoint filename (e.g. dreamshaper_8.safetensors)

    Returns:
        DiffusionPipeline: A DiffusionPipeline that is approriate for the model type
    """
    if model_name not in loaded_checkpoints:
        pipe = load_checkpoint(model_name)
        print(f"Compiling {model_name}", flush=True)
        start = time.perf_counter()
        pipe.to("cuda")
        pipe.text_encoder.eval()
        pipe.unet.eval()
        pipe.vae.eval()
        pipe = compile(pipe, compile_config)
        end = time.perf_counter()
        print(f"Compiled {model_name} in {end - start:.2f}s", flush=True)
        print(f"Warming up {model_name}", flush=True)
        pipe(prompt="Leafy Green Salad", num_inference_steps=15)
        loaded_checkpoints[model_name] = pipe

    return loaded_checkpoints[model_name]
