import torch
import os
import time
import diffusers
from sfast.compilers.diffusion_pipeline_compiler import (
    compile,
    compile_vae,
    compile_unet,
    CompilationConfig,
)
import huggingface_hub
import config
from compel import Compel, ReturnedEmbeddingsType
import logging
import webhooks
import asyncio
import threading
import inspect
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True

logging.info("Torch version: %s", config.package_versions["torch"])
logging.info("XFormers version: %s", config.package_versions["xformers"])
logging.info("Triton version: %s", config.package_versions["triton"])
logging.info("Diffusers version: %s", config.package_versions["diffusers"])
logging.info("Transformers version: %s", config.package_versions["transformers"])
logging.info("CUDA Version: %s", config.package_versions["cuda"])
logging.info("Stable Fast version: %s", config.package_versions["stable_fast"])


compile_config = CompilationConfig.Default()
compile_config.enable_xformers = True
compile_config.enable_triton = True
compile_config.enable_cuda_graph = config.cuda_graph
compile_config.memory_format = torch.channels_last

loaded_checkpoints = {}
loaded_vae = {}
loaded_lora = {}
loaded_controlnet = {}

warmup_image_path = os.path.join(os.path.dirname(__file__), "cat.png")
with open(warmup_image_path, "rb") as f:
    warmup_image = Image.open(f).convert("RGB")


def run_asyncio_coroutine(coroutine):
    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(coroutine)
        loop.close()

    thread = threading.Thread(target=run)
    thread.start()


def load_checkpoint(
    model_name: str, vae: str = None, is_refiner: bool = False
) -> diffusers.DiffusionPipeline:
    """_summary_

    Args:
        model_name (str): Either a HuggingFace model name (e.g. stabilityai/stable-diffusion-xl-base-1.0) or a local checkpoint filename (e.g. dreamshaper_8.safetensors)

    Returns:
        DiffusionPipeline: A DiffusionPipeline that is approriate for the model type
    """
    logging.info("Loading Checkpoint: %s", model_name)
    model_kwargs = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "extract_ema": True,
        "load_safety_checker": config.load_safety_checker,
        "use_safetensors": True,
        # "vae": None,
    }
    if torch.cuda.is_available():
        model_kwargs["device"] = "cuda"
    if vae is not None and vae not in loaded_vae:
        vae_model = diffusers.AutoencoderKL.from_pretrained(
            os.path.join(config.vae_dir, vae),
            use_safetensors=True,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        model_kwargs["vae"] = vae_model
        logging.info("Loaded VAE %s for %s", vae, model_name)
    elif vae is not None and vae in loaded_vae:
        model_kwargs["vae"] = loaded_vae[vae]
        logging.info("Using loaded VAE %s for %s", vae, model_name)

    if is_refiner:
        model_kwargs["variant"] = "fp16"

    # if the model_name looks like org/repo, then we assume it's a HuggingFace model
    # otherwise, we assume it's a local model
    start = time.perf_counter()
    if "/" in model_name and not model_name.endswith(".safetensors"):
        model_info = huggingface_hub.model_info(model_name)
        if "diffusers" in model_info.config:
            default_pipeline = model_info.config["diffusers"]["class_name"]
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
    logging.info("Loaded %s with %s in %.2fs", model_class, model_name, end - start)
    return pipe


def load_controlnet(model_name: str) -> diffusers.ControlNetModel:
    """
    Load a ControlNet model.

    Args:
        model_name (str): The name of the model to load. Either a huggingface model ID (e.g. diffusers/controlnet-canny-sdxl-1.0) or a local safetensors filename (e.g. qrCodeMonster_v20.safetensors)

    Returns:
        ControlNetModel: The loaded ControlNet model.

    """
    logging.info("Loading ControlNet: %s", model_name)
    model_kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
    start = time.perf_counter()
    if "/" in model_name and not model_name.endswith(".safetensors"):
        controlnet = diffusers.ControlNetModel.from_pretrained(
            model_name, **model_kwargs
        )
    elif model_name.endswith(".safetensors"):
        model_path = os.path.join(config.controlnet_dir, model_name)
        controlnet = diffusers.ControlNetModel.from_single_file(
            model_path, **model_kwargs
        )
    end = time.perf_counter()
    logging.info("Loaded ControlNet %s in %.2fs", model_name, end - start)
    return controlnet


class ModelManager:
    __pipes__ = {}
    __schedulers__ = {}
    __safety_checker__ = None
    __feature_extractor__ = None
    __vae_name__ = None

    def __init__(self, model_name: str, vae: str = None, is_refiner: bool = False):
        self.model_name = model_name
        pipe = load_checkpoint(model_name, vae=vae, is_refiner=is_refiner)
        self.is_refiner = is_refiner
        pipe_type = pipe.__class__.__name__
        self.default_pipeline = pipe_type
        self.__vae_name__ = vae
        if hasattr(pipe, "safety_checker"):
            self.__safety_checker__ = pipe.safety_checker
            self.__feature_extractor__ = pipe.feature_extractor
        logging.info("Moving %s (%s) to GPU", model_name, pipe_type)
        start = time.perf_counter()
        pipe.to("cuda")
        end = time.perf_counter()
        logging.info("Moved %s to GPU in %.2fs", model_name, end - start)
        tokenizers = []
        text_encoders = []
        if hasattr(pipe, "tokenizer") and pipe.tokenizer is not None:
            tokenizers.append(pipe.tokenizer)
        if hasattr(pipe, "tokenizer_2") and pipe.tokenizer_2 is not None:
            tokenizers.append(pipe.tokenizer_2)
        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            text_encoders.append(pipe.text_encoder)
        if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
            text_encoders.append(pipe.text_encoder_2)
        if vae is None and "default_vae" not in loaded_vae:
            loaded_vae["default_vae"] = pipe.vae
        elif vae is not None and vae not in loaded_vae:
            loaded_vae[vae] = pipe.vae
        if config.compile_model:
            logging.info("Compiling %s", model_name)
            start = time.perf_counter()
            if hasattr(pipe, "unet") and pipe.unet is not None:
                pipe.unet.eval()
            if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
                pipe.text_encoder.eval()
            if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
                pipe.text_encoder_2.eval()
            if hasattr(pipe, "vae") and pipe.vae is not None:
                pipe.vae.eval()
            pipe = compile(pipe, compile_config)
            end = time.perf_counter()
            logging.info("Compiled %s in %.2fs", model_name, end - start)
        compel_kwargs = {}
        if len(tokenizers) == 1:
            compel_kwargs["tokenizer"] = tokenizers[0]
        else:
            compel_kwargs["tokenizer"] = tokenizers
        if len(text_encoders) == 1:
            compel_kwargs["text_encoder"] = text_encoders[0]
        else:
            compel_kwargs["text_encoder"] = text_encoders
            compel_kwargs[
                "returned_embeddings_type"
            ] = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
            compel_kwargs["requires_pooled"] = [False, True]
        self.compel = Compel(**compel_kwargs)
        logging.info("Warming up %s", model_name)
        start = time.perf_counter()
        warmup_params = {
            "prompt": "Leafy Green Salad",
            "num_inference_steps": 1,
        }
        expected_kwargs = inspect.signature(pipe.__class__.__call__).parameters.keys()
        if "image" in expected_kwargs:
            warmup_params["image"] = warmup_image
            warmup_params["num_inference_steps"] = 5
            logging.info("Warming up with image")
        for i in range(2):
            logging.info("Warmup #%d", i)
            pipe(**warmup_params)
        end = time.perf_counter()
        logging.info("Warmed up %s in %.2fs", model_name, end - start)
        self.__pipes__[pipe_type] = pipe
        run_asyncio_coroutine(webhooks.model_loaded({"checkpoint": model_name}))

    def get_safety_checker(self):
        return self.__safety_checker__

    def get_feature_extractor(self):
        return self.__feature_extractor__

    def get_pipeline(self, pipeline_type: str = None, control_model: str = None):
        if pipeline_type is None:
            pipeline_type = self.default_pipeline
        if pipeline_type not in self.__pipes__:
            try:
                PipeClass = getattr(diffusers, pipeline_type)
            except AttributeError:
                raise Exception(f"Unknown pipeline type {pipeline_type}")
            pipe_kwargs = {**self.__pipes__[self.default_pipeline].components}
            if control_model is not None:
                pipe_kwargs["controlnet"] = get_controlnet(control_model)
            expected_kwargs = inspect.signature(PipeClass.__init__).parameters.keys()
            pipe_kwargs = {k: v for k, v in pipe_kwargs.items() if k in expected_kwargs}
            pipe = PipeClass(**pipe_kwargs)
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
            logging.debug(
                [(scheduler, type(scheduler)) for scheduler in pipe.scheduler]
            )
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


def get_checkpoint(
    model_name: str, vae: str = None, is_refiner: bool = False
) -> ModelManager:
    """_summary_

    Args:
        model_name (str): Either a HuggingFace model name (e.g. stabilityai/stable-diffusion-xl-base-1.0) or a local checkpoint filename (e.g. dreamshaper_8.safetensors)

    Returns:
        DiffusionPipeline: A DiffusionPipeline that is approriate for the model type
    """
    if model_name not in loaded_checkpoints:
        loaded_checkpoints[model_name] = ModelManager(
            model_name, vae=vae, is_refiner=is_refiner
        )

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


def get_controlnet(model_name: str) -> diffusers.ControlNetModel:
    """_summary_

    Args:
        model_name (str): Either a HuggingFace model name (e.g. diffusers/controlnet-canny-sdxl-1.0) or a local safetensors filename (e.g. qrCodeMonster_v20.safetensors)

    Returns:
        ControlNetModel: A ControlNetModel that is approriate for the model type
    """
    if model_name not in loaded_controlnet:
        controlnet = load_controlnet(model_name)
        logging.info("Moving %s to GPU", model_name)
        start = time.perf_counter()
        controlnet.to("cuda")
        end = time.perf_counter()
        logging.info("Moved %s to GPU in %.2fs", model_name, end - start)
        if config.compile_model:
            logging.info("Compiling %s", model_name)
            start = time.perf_counter()
            controlnet = compile_unet(controlnet, compile_config)
            end = time.perf_counter()
            logging.info("Compiled %s in %.2fs", model_name, end - start)
        loaded_controlnet[model_name] = controlnet
        run_asyncio_coroutine(webhooks.model_loaded({"controlnet": model_name}))

    return loaded_controlnet[model_name]
