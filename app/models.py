import torch
import os
import time
import xformers
import triton
from transformers import __version__ as transformers_version
import diffusers
from sfast.compilers.stable_diffusion_pipeline_compiler import (
    compile,
    CompilationConfig,
)
import huggingface_hub
from safetensors import safe_open

torch.backends.cuda.matmul.allow_tf32 = True

print("Torch version:", torch.__version__, flush=True)
print("XFormers version:", xformers.__version__, flush=True)
print("Triton version:", triton.__version__, flush=True)
print("Diffusers version:", diffusers.__version__, flush=True)
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


def load_checkpoint(model_name: str):
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
    if "/" in model_name and not model_name.endswith(".safetensors"):
        model_info = huggingface_hub.model_info(model_name)
        if "config" in model_info and "diffusers" in model_info["config"]:
            default_pipeline = model_info["config"]["diffusers"]["class_name"]
        else:
            raise Exception(f"Unable to find config for {model_name}")
        PipeClass = getattr(diffusers, default_pipeline)
        pipe = PipeClass.from_pretrained(model_name, **model_kwargs)
    elif model_name.endswith(".safetensors"):
        model_path = os.path.join(checkpoint_dir, model_name)
        default_pipeline = "StableDiffusionPipeline"
        # if the model is > 4GB, then we assume it's a SDXL checkpoint
        if os.path.getsize(model_path) > 4 * 1024 * 1024 * 1024:
            default_pipeline = "StableDiffusionXLPipeline"
        PipeClass = getattr(diffusers, default_pipeline)
        pipe = PipeClass.from_single_file(model_path, **model_kwargs)
    end = time.perf_counter()
    model_class = pipe.__class__.__name__
    print(f"Loaded {model_class} with {model_name} in {end - start:.2f}s", flush=True)
    return pipe


class ModelManager:
    __pipes__ = {}

    def __init__(self, model_name):
        self.model_name = model_name
        pipe = load_checkpoint(model_name)
        pipe_type = pipe.__class__.__name__
        print(f"Compiling {model_name} ({pipe_type})", flush=True)
        start = time.perf_counter()
        pipe.to("cuda")
        pipe.text_encoder.eval()
        pipe.unet.eval()
        pipe.vae.eval()
        if pipe_type == "StableDiffusionXLPipeline":
            pipe.text_encoder_2.eval()
        pipe = compile(pipe, compile_config)
        end = time.perf_counter()
        print(f"Compiled {model_name} in {end - start:.2f}s", flush=True)
        print(f"Warming up {model_name}", flush=True)
        start = time.perf_counter()
        pipe(prompt="Leafy Green Salad", num_inference_steps=2)
        end = time.perf_counter()
        print(f"Warmed up {model_name} in {end - start:.2f}s", flush=True)
        self.__pipes__[pipe_type] = pipe

    def get_pipeline(self, pipeline_type):
        if pipeline_type not in self.__pipes__:
            try:
                PipeClass = getattr(diffusers, pipeline_type)
            except AttributeError:
                raise Exception(f"Unknown pipeline type {pipeline_type}")
            if "StableDiffusionPipeline" in self.__pipes__:
                pipe = PipeClass(**self.__pipes__["StableDiffusionPipeline"].components)
            elif "StableDiffusionXLPipeline" in self.__pipes__:
                pipe = PipeClass(
                    **self.__pipes__["StableDiffusionXLPipeline"].components
                )
            else:
                raise Exception("Unable to find a valid base pipeline")
            self.__pipes__[pipeline_type] = pipe
        return self.__pipes__[pipeline_type]


def get_checkpoint(model_name: str):
    """_summary_

    Args:
        model_name (str): Either a HuggingFace model name (e.g. stabilityai/stable-diffusion-xl-base-1.0) or a local checkpoint filename (e.g. dreamshaper_8.safetensors)

    Returns:
        DiffusionPipeline: A DiffusionPipeline that is approriate for the model type
    """
    if model_name not in loaded_checkpoints:
        loaded_checkpoints[model_name] = ModelManager(model_name)

    return loaded_checkpoints[model_name]
