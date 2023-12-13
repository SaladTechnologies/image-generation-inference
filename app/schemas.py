from pydantic import BaseModel
from typing import Optional, Union, List, Tuple
from enum import Enum


class PipelineOptions(Enum):
    StableDiffusionPipeline = "StableDiffusionPipeline"
    StableDiffusionXLPipeline = "StableDiffusionXLPipeline"


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


class GenerateParams(BaseModel):
    checkpoint: str
    pipeline: Optional[PipelineOptions] = PipelineOptions.StableDiffusionPipeline
    scheduler: Optional[str] = None
    a1111_scheduler: Optional[str] = None
    parameters: Union[StableDiffusionPipelineParams, StableDiffusionXLPipelineParams]
    return_images: Optional[bool] = True


class ModelListFilters(BaseModel):
    loaded: Optional[bool] = False


class UnloadCheckpointParams(BaseModel):
    checkpoint: str
