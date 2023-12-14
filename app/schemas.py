from pydantic import BaseModel, Extra
from typing import Optional, Union, List, Tuple
from enum import Enum
import config


class PipelineOptions(Enum):
    StableDiffusionPipeline = "StableDiffusionPipeline"
    StableDiffusionImg2ImgPipeline = "StableDiffusionImg2ImgPipeline"
    StableDiffusionInpaintPipeline = "StableDiffusionInpaintPipeline"

    StableDiffusionXLPipeline = "StableDiffusionXLPipeline"
    StableDiffusionXLImg2ImgPipeline = "StableDiffusionXLImg2ImgPipeline"


class StableDiffusionPipelineParams(BaseModel):
    prompt: Union[str, List[str]]
    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: Optional[int] = 15
    guidance_scale: Optional[float] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    num_images_per_prompt: Optional[int] = None
    eta: Optional[float] = None
    seed: Optional[Union[int, List[int]]] = None
    clip_skip: Optional[int] = None
    guidance_rescale: Optional[float] = None
    timesteps: Optional[List[int]] = None

    class Config:
        extra = Extra.forbid


class StableDiffusionImg2ImgPipelineParams(BaseModel):
    prompt: Union[str, List[str]]
    image: Union[str, List[str]]
    num_inference_steps: Optional[int] = 15
    strength: Optional[float] = None
    guidance_scale: Optional[float] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    num_images_per_prompt: Optional[int] = None
    eta: Optional[float] = None
    seed: Optional[Union[int, List[int]]] = None
    clip_skip: Optional[int] = None
    timesteps: Optional[List[int]] = None

    class Config:
        extra = Extra.forbid


class StableDiffusionInpaintPipelineParams(BaseModel):
    prompt: Union[str, List[str]]
    image: Union[str, List[str]]
    mask_image: Union[str, List[str]]
    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: Optional[int] = 15
    strength: Optional[float] = None
    guidance_scale: Optional[float] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    num_images_per_prompt: Optional[int] = None
    eta: Optional[float] = None
    seed: Optional[Union[int, List[int]]] = None
    clip_skip: Optional[int] = None
    timesteps: Optional[List[int]] = None

    class Config:
        extra = Extra.forbid


class StableDiffusionXLPipelineParams(BaseModel):
    prompt: Union[str, List[str]]
    prompt_2: Optional[Union[str, List[str]]] = None
    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: Optional[int] = 15
    denoising_end: Optional[float] = None
    guidance_scale: Optional[float] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    negative_prompt_2: Optional[Union[str, List[str]]] = None
    num_images_per_prompt: Optional[int] = None
    eta: Optional[float] = None
    seed: Optional[Union[int, List[int]]] = None
    original_size: Optional[Tuple[int, int]] = None
    target_size: Optional[Tuple[int, int]] = None
    crops_coords_top_left: Optional[Tuple[int, int]] = None
    negative_original_size: Optional[Tuple[int, int]] = None
    negative_crops_coords_top_left: Optional[Tuple[int, int]] = None
    negative_target_size: Optional[Tuple[int, int]] = None
    guidance_rescale: Optional[float] = None
    timesteps: Optional[List[int]] = None

    class Config:
        extra = Extra.forbid


class StableDiffusionXLImg2ImgPipelineParams(BaseModel):
    prompt: Union[str, List[str]]
    image: Union[str, List[str]]
    num_inference_steps: Optional[int] = 15
    strength: Optional[float] = None
    prompt_2: Optional[Union[str, List[str]]] = None
    denoising_start: Optional[float] = None
    denoising_end: Optional[float] = None
    guidance_scale: Optional[float] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    negative_prompt_2: Optional[Union[str, List[str]]] = None
    num_images_per_prompt: Optional[int] = None
    eta: Optional[float] = None
    seed: Optional[Union[int, List[int]]] = None
    original_size: Optional[Tuple[int, int]] = None
    target_size: Optional[Tuple[int, int]] = None
    crops_coords_top_left: Optional[Tuple[int, int]] = None
    negative_original_size: Optional[Tuple[int, int]] = None
    negative_crops_coords_top_left: Optional[Tuple[int, int]] = None
    negative_target_size: Optional[Tuple[int, int]] = None
    aesthetic_score: Optional[float] = None
    negative_aesthetic_score: Optional[float] = None
    clip_skip: Optional[int] = None
    guidance_rescale: Optional[float] = None
    timesteps: Optional[List[int]] = None

    class Config:
        extra = Extra.forbid


class GenerateParams(BaseModel):
    checkpoint: str
    pipeline: Optional[PipelineOptions] = PipelineOptions.StableDiffusionPipeline
    scheduler: Optional[str] = None
    a1111_scheduler: Optional[str] = None
    safety_checker: Optional[bool] = config.load_safety_checker
    parameters: Union[
        StableDiffusionPipelineParams,
        StableDiffusionImg2ImgPipelineParams,
        StableDiffusionInpaintPipelineParams,
        StableDiffusionXLPipelineParams,
        StableDiffusionXLImg2ImgPipelineParams,
    ]
    return_images: Optional[bool] = True

    class Config:
        extra = Extra.forbid

    class Config:
        extra = Extra.forbid


class ModelListFilters(BaseModel):
    loaded: Optional[bool] = False


class LoadOrUnloadCheckpointParams(BaseModel):
    checkpoint: str


class CPUFrequency(BaseModel):
    current: float
    min: float
    max: float


class CPUPerformance(BaseModel):
    utilization: List[float]
    frequency: List[CPUFrequency]


class MemoryPerformance(BaseModel):
    total: float
    available: float
    percent: float
    used: float
    free: float


class StoragePerformance(BaseModel):
    used: float
    free: float


class GPUPerformance(BaseModel):
    id: int
    name: str
    load: float
    free_memory: float
    used_memory: float
    total_memory: float
    temperature: float


class PackageVersions(BaseModel):
    torch: str
    cuda: str
    xformers: str
    triton: str
    diffusers: str
    transformers: str
    stable_fast: str


class SystemPerformance(BaseModel):
    cpu: CPUPerformance
    memory: MemoryPerformance
    storage: StoragePerformance
    gpu: List[GPUPerformance]
    packages: PackageVersions


class ModelLoadedEvent(BaseModel):
    checkpoint: str
    time_utc: float
    worker_identity: dict
    stats: SystemPerformance
