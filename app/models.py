import torch
import os
import time
import diffusers
from sfast.compilers.diffusion_pipeline_compiler import (
    compile,
    compile_vae,
    CompilationConfig,
)
import huggingface_hub
import gc
import config

torch.backends.cuda.matmul.allow_tf32 = True

print("Torch version:", config.package_versions["torch"], flush=True)
print("XFormers version:", config.package_versions["xformers"], flush=True)
print("Triton version:", config.package_versions["triton"], flush=True)
print("Diffusers version:", config.package_versions["diffusers"], flush=True)
print("Transformers version:", config.package_versions["transformers"], flush=True)
print("CUDA Version:", config.package_versions["cuda"], flush=True)
print("Stable Fast version:", config.package_versions["stable_fast"], flush=True)


compile_config = CompilationConfig.Default()
compile_config.enable_xformers = True
compile_config.enable_triton = True
compile_config.enable_cuda_graph = config.cuda_graph
compile_config.memory_format = torch.channels_last

loaded_checkpoints = {}
loaded_vae = {}
loaded_lora = {}
loaded_controlnet = {}


def load_checkpoint(model_name: str, vae: str = None):
    """_summary_

    Args:
        model_name (str): Either a HuggingFace model name (e.g. stabilityai/stable-diffusion-xl-base-1.0) or a local checkpoint filename (e.g. dreamshaper_8.safetensors)

    Returns:
        DiffusionPipeline: A DiffusionPipeline that is approriate for the model type
    """
    model_kwargs = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "extract_ema": True,
        "device_map": "auto",
        "load_safety_checker": config.load_safety_checker,
        "use_safetensors": True,
        "vae": None,
    }
    if vae is not None and vae not in loaded_vae:
        vae_model = diffusers.AutoencoderKL.from_pretrained(
            os.path.join(config.vae_dir, vae),
            use_safetensors=True,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        model_kwargs["vae"] = vae_model
        print(f"Loaded VAE {vae} for {model_name}", flush=True)
    elif vae is not None and vae in loaded_vae:
        model_kwargs["vae"] = loaded_vae[vae]
        print(f"Using loaded VAE {vae} for {model_name}", flush=True)

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
        model_path = os.path.join(config.checkpoint_dir, model_name)
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
    __schedulers__ = {}
    __safety_checker__ = None
    __feature_extractor__ = None
    __vae_name__ = None

    def __init__(self, model_name: str, vae: str = None):
        self.model_name = model_name
        pipe = load_checkpoint(model_name, vae=vae)
        pipe_type = pipe.__class__.__name__
        self.default_pipeline = pipe_type
        self.__vae_name__ = vae
        if hasattr(pipe, "safety_checker"):
            self.__safety_checker__ = pipe.safety_checker
            self.__feature_extractor__ = pipe.feature_extractor
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
        if vae is None and "default_vae" not in loaded_vae:
            loaded_vae["default_vae"] = pipe.vae
        elif vae is not None and vae not in loaded_vae:
            loaded_vae[vae] = pipe.vae
        print(f"Compiled {model_name} in {end - start:.2f}s", flush=True)
        print(f"Warming up {model_name}", flush=True)
        start = time.perf_counter()
        for _ in range(2):
            pipe(prompt="Leafy Green Salad", num_inference_steps=1)
        end = time.perf_counter()
        print(f"Warmed up {model_name} in {end - start:.2f}s", flush=True)
        self.__pipes__[pipe_type] = pipe

    def get_safety_checker(self):
        return self.__safety_checker__

    def get_feature_extractor(self):
        return self.__feature_extractor__

    def get_pipeline(self, pipeline_type: str = None):
        if pipeline_type is None:
            pipeline_type = self.default_pipeline
        if pipeline_type not in self.__pipes__:
            try:
                PipeClass = getattr(diffusers, pipeline_type)
            except AttributeError:
                raise Exception(f"Unknown pipeline type {pipeline_type}")
            pipe = PipeClass(**self.__pipes__[self.default_pipeline].components)
            self.__pipes__[pipeline_type] = pipe
        return self.__pipes__[pipeline_type]

    def get_scheduler(
        self, scheduler: str = None, scheduler_config: dict = {}, alias: str = None
    ):
        default_scheduler = self.__pipes__[self.default_pipeline].scheduler
        if scheduler is None:
            return default_scheduler
        if alias is None:
            alias = scheduler
        if alias not in self.__schedulers__:
            SchedulerClass = getattr(diffusers, scheduler)
            scheduler = SchedulerClass.from_config(
                default_scheduler.config, **scheduler_config
            )
            self.__schedulers__[alias] = scheduler
        return self.__schedulers__[alias]

    def get_compatible_schedulers(self, pipeline_type: str):
        pipe = self.get_pipeline(pipeline_type)
        # if the scheduler is tuple, print it
        if isinstance(pipe.scheduler, tuple):
            # print the type and value of all members of the tuple
            print([(scheduler, type(scheduler)) for scheduler in pipe.scheduler])
        return pipe.scheduler._compatibles

    def get_a1111_scheduler(self, a1111_alias: str):
        mapping = {
            "DPM++ 2M": {"class": "DPMSolverMultistepScheduler", "config": {}},
            "DPM++ 2M Karras": {
                "class": "DPMSolverMultistepScheduler",
                "config": {"use_karras_sigmas": True},
            },
            "DPM++ 2M SDE": {
                "class": "DPMSolverMultistepScheduler",
                "config": {"algorithm_type": "sde-dpmsolver++"},
            },
            "DPM++ 2M SDE Karras": {
                "class": "DPMSolverMultistepScheduler",
                "config": {
                    "algorithm_type": "sde-dpmsolver++",
                    "use_karras_sigmas": True,
                },
            },
            "DPM++ SDE": {"class": "	DPMSolverSinglestepScheduler", "config": {}},
            "DPM++ SDE Karras": {
                "class": "DPMSolverSinglestepScheduler",
                "config": {"use_karras_sigmas": True},
            },
            "DPM2": {
                "class": "KDPM2DiscreteScheduler",
                "config": {},
            },
            "DPM2 Karras": {
                "class": "KDPM2DiscreteScheduler",
                "config": {"use_karras_sigmas": True},
            },
            "DPM2 a": {
                "class": "KDPM2AncestralDiscreteScheduler",
                "config": {},
            },
            "DPM2 a Karras": {
                "class": "KDPM2AncestralDiscreteScheduler",
                "config": {"use_karras_sigmas": True},
            },
            "Euler": {
                "class": "EulerDiscreteScheduler",
                "config": {},
            },
            "Euler a": {
                "class": "EulerAncestralDiscreteScheduler",
                "config": {},
            },
            "Heun": {
                "class": "HeunDiscreteScheduler",
                "config": {},
            },
            "LMS": {
                "class": "LMSDiscreteScheduler",
                "config": {},
            },
            "LMS Karras": {
                "class": "LMSDiscreteScheduler",
                "config": {"use_karras_sigmas": True},
            },
        }

        if a1111_alias not in mapping:
            raise Exception(
                f"Unknown A1111 alias {a1111_alias}. Use one of {list(mapping.keys())}"
            )

        scheduler = self.get_scheduler(
            mapping[a1111_alias]["class"], mapping[a1111_alias]["config"], a1111_alias
        )
        return scheduler

    def set_vae(self, vae: str):
        if vae == self.__vae_name__:
            return
        if vae not in loaded_vae:
            loaded_vae[vae] = diffusers.AutoencoderKL.from_pretrained(
                os.path.join(config.vae_dir, vae),
                use_safetensors=True,
                torch_dtype=torch.float16,
                variant="fp16",
            ).to("cuda")
            loaded_vae[vae].eval()
            loaded_vae[vae] = compile_vae(loaded_vae[vae], compile_config)

        for pipe in self.__pipes__.values():
            pipe.vae = loaded_vae[vae]
        self.__vae_name__ = vae


def get_checkpoint(model_name: str, vae: str = None) -> ModelManager:
    """_summary_

    Args:
        model_name (str): Either a HuggingFace model name (e.g. stabilityai/stable-diffusion-xl-base-1.0) or a local checkpoint filename (e.g. dreamshaper_8.safetensors)

    Returns:
        DiffusionPipeline: A DiffusionPipeline that is approriate for the model type
    """
    if model_name not in loaded_checkpoints:
        loaded_checkpoints[model_name] = ModelManager(model_name, vae=vae)

    model = loaded_checkpoints[model_name]
    if vae is not None:
        model.set_vae(vae)
    return model


def list_local_checkpoints():
    return os.listdir(config.checkpoint_dir)


def list_local_vae():
    return os.listdir(config.vae_dir)


def list_local_lora():
    return os.listdir(config.lora_dir)


def list_local_controlnet():
    return os.listdir(config.controlnet_dir)


def list_loaded_checkpoints():
    return list(loaded_checkpoints.keys())
