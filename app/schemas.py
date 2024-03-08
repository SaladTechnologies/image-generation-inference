from pydantic import BaseModel, Extra, Field, create_model
from typing import Optional, Union
from enum import Enum
import config


class PipelineOptions(Enum):
    StableDiffusionPipeline = "StableDiffusionPipeline"
    StableDiffusionImg2ImgPipeline = "StableDiffusionImg2ImgPipeline"
    StableDiffusionInpaintPipeline = "StableDiffusionInpaintPipeline"
    StableDiffusionControlNetPipeline = "StableDiffusionControlNetPipeline"
    StableDiffusionControlNetImg2ImgPipeline = (
        "StableDiffusionControlNetImg2ImgPipeline"
    )
    StableDiffusionControlNetInpaintPipeline = (
        "StableDiffusionControlNetInpaintPipeline"
    )

    StableDiffusionXLPipeline = "StableDiffusionXLPipeline"
    StableDiffusionXLImg2ImgPipeline = "StableDiffusionXLImg2ImgPipeline"
    StableDiffusionXLInpaintPipeline = "StableDiffusionXLInpaintPipeline"
    StableDiffusionXLControlNetPipeline = "StableDiffusionXLControlNetPipeline"
    StableDiffusionXLControlNetImg2ImgPipeline = (
        "StableDiffusionXLControlNetImg2ImgPipeline"
    )
    StableDiffusionXLControlNetInpaintPipeline = (
        "StableDiffusionXLControlNetInpaintPipeline"
    )


prompt_field = Field(
    ...,
    description="The prompt or prompts to guide image generation.",
)
height_field = Field(
    None,
    description="The height in pixels of the generated image. defaults to self.unet.config.sample_size * self.vae_scale_factor",
)
width_field = Field(
    None,
    description="The width in pixels of the generated image. defaults to self.unet.config.sample_size * self.vae_scale_factor",
)
num_inference_steps_field = Field(
    50,
    description="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference",
)
timesteps_field = Field(
    None,
    description="Custom timesteps to use for the denoising process with schedulers which support a timesteps argument in their set_timesteps method. If not defined, the default behavior when num_inference_steps is passed will be used. Must be in descending order.",
)
guidance_scale_field = Field(
    7.5,
    description="A higher guidance scale value encourages the model to generate images closely linked to the text prompt at the expense of lower image quality. Guidance scale is enabled when guidance_scale > 1.",
)
negative_prompt_field = Field(
    None,
    description="The prompt or prompts to guide what to not include in image generation. Ignored when not using guidance (guidance_scale < 1).",
)
num_images_per_prompt_field = Field(
    1,
    description="The number of images to generate per prompt.",
)
eta_field = Field(
    0.0,
    description="Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies to the [DDIMScheduler](https://huggingface.co/docs/diffusers/v0.24.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), and is ignored in other schedulers.",
)
seed_field = Field(
    None,
    description="A random seed or list of seeds to use for image generation. Makes generation deterministic.",
)
guidance_rescale_field = Field(
    0.0,
    description="Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when using zero terminal SNR.",
)
clip_skip_field = Field(
    None,
    description="Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that the output of the pre-final layer will be used for computing the prompt embeddings.",
)
image_field = Field(
    ...,
    description="Base64-encoded image or list of images to be used as the starting point. Images are converted to RGB and resized for dimensions to be a multiple of 8.",
)
strength_field = Field(
    0.8,
    description="Indicates extent to transform the reference image. Must be between 0 and 1. image is used as a starting point and more noise is added the higher the strength. The number of denoising steps depends on the amount of noise initially added. When strength is 1, added noise is maximum and the denoising process runs for the full number of iterations specified in num_inference_steps. A value of 1 essentially ignores image.",
)
mask_image_field = Field(
    ...,
    description="Base64-encoded image or list of images to be used as the mask. White pixels in the mask are repainted while black pixels are preserved. Images are converted to RGB and resized for dimensions to be a multiple of 8.",
)
control_image_field = Field(
    ...,
    description="The ControlNet input condition to provide guidance to the unet for generation. The dimensions of the output image defaults to image’s dimensions. If height and/or width are passed, image is resized accordingly. If multiple ControlNets are specified in init, images must be passed as a list such that each element of the list can be correctly batched for input to a single ControlNet. The image or images should be base64-encoded.",
)
controlnet_conditioning_scale_field = Field(
    1.0,
    description="The outputs of the ControlNet are multiplied by controlnet_conditioning_scale before they are added to the residual in the original unet. If multiple ControlNets are specified in init, you can set the corresponding scale as a list.",
)
guess_mode_field = Field(
    False,
    description="The ControlNet encoder tries to recognize the content of the input image even if you remove all prompts. A guidance_scale value between 3.0 and 5.0 is recommended.",
)
control_guidance_start_field = Field(
    0.0,
    description="The percentage of total steps at which the ControlNet starts applying.",
)
control_guidance_end_field = Field(
    1.0,
    description="The percentage of total steps at which the ControlNet stops applying.",
)
prompt_2_field = Field(
    None,
    description="The prompt or prompts to be sent to the tokenizer_2 and text_encoder_2. If not defined, prompt is used in both text-encoders",
)
denoising_end_field = Field(
    None,
    description=" When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be completed before it is intentionally prematurely terminated. As a result, the returned sample will still retain a substantial amount of noise as determined by the discrete timesteps selected by the scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a “Mixture of Denoisers” multi-pipeline setup, as elaborated in [Refining the Image Output](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)",
)
negative_prompt_2_field = Field(
    None,
    description="The prompt or prompts not to guide the image generation to be sent to tokenizer_2 and text_encoder_2. If not defined, negative_prompt is used in both text-encoders.",
)
original_size_field = Field(
    None,
    description="If original_size is not the same as target_size the image will appear to be down- or upsampled. original_size defaults to (height, width) if not specified. Part of SDXL’s micro-conditioning as explained in section 2.2 of https://huggingface.co/papers/2307.01952.",
)
crops_coords_top_left_field = Field(
    (0, 0),
    description="crops_coords_top_left can be used to generate an image that appears to be “cropped” from the position crops_coords_top_left downwards. Favorable, well-centered images are usually achieved by setting crops_coords_top_left to (0, 0). Part of SDXL’s micro-conditioning as explained in section 2.2 of https://huggingface.co/papers/2307.01952.",
)
target_size_field = Field(
    None,
    description="For most cases, target_size should be set to the desired height and width of the generated image. If not specified it will default to (height, width). Part of SDXL’s micro-conditioning as explained in section 2.2 of https://huggingface.co/papers/2307.01952.",
)
negative_original_size_field = Field(
    (1024, 1024),
    description="To negatively condition the generation process based on a specific image resolution. Part of SDXL’s micro-conditioning as explained in section 2.2 of https://huggingface.co/papers/2307.01952. For more information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.",
)
negative_crops_coords_top_left_field = Field(
    (0, 0),
    description="To negatively condition the generation process based on a specific crop coordinates. Part of SDXL’s micro-conditioning as explained in section 2.2 of https://huggingface.co/papers/2307.01952. For more information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.",
)
negative_target_size_field = Field(
    (1024, 1024),
    description="To negatively condition the generation process based on a target image resolution. It should be as same as the target_size for most cases. Part of SDXL’s micro-conditioning as explained in section 2.2 of https://huggingface.co/papers/2307.01952. For more information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.",
)
denoising_start_field = Field(
    None,
    description="When specified, indicates the fraction (between 0.0 and 1.0) of the total denoising process to be bypassed before it is initiated. Consequently, the initial part of the denoising process is skipped and it is assumed that the passed image is a partly denoised image. Note that when this is specified, strength will be ignored. The denoising_start parameter is particularly beneficial when this pipeline is integrated into a “Mixture of Denoisers” multi-pipeline setup, as detailed in Refine Image Quality.",
)
aeshtetic_score_field = Field(
    6.0,
    description="Used to simulate an aesthetic score of the generated image by influencing the positive text condition. Part of SDXL’s micro-conditioning as explained in section 2.2 of https://huggingface.co/papers/2307.01952.",
)
negative_aesthetic_score_field = Field(
    2.5,
    description="Part of SDXL’s micro-conditioning as explained in section 2.2 of https://huggingface.co/papers/2307.01952. Can be used to simulate an aesthetic score of the generated image by influencing the negative text condition.",
)
checkpoint_field = Field(
    ...,
    description="Either a HuggingFace model name (e.g. stabilityai/stable-diffusion-xl-base-1.0) or a local checkpoint filename (e.g. dreamshaper_8.safetensors)",
)
batch_id_field = Field(
    None,
    description="A unique identifier for the batch of images being generated. If not specified, a random UUID will be generated. Images stored to disk will be named {batch_id}-{image_index}.jpg",
)
scheduler_field = Field(
    None,
    description="See [the diffusers documentation for more details]https://huggingface.co/docs/diffusers/api/schedulers/overview)",
)
a1111_scheduler_field = Field(
    None,
    description="Scheduler names you might be familiar with from Automatic1111 or k-diffusion are also accepted. See [the diffusers documentation for more details](https://huggingface.co/docs/diffusers/api/schedulers/overview)",
)
pipeline_field = Field(
    PipelineOptions.StableDiffusionPipeline,
    description="The pipeline class to use for image generation.",
)
vae_field = Field(
    None,
    description="Either a HuggingFace model name (e.g. stabilityai/stable-diffusion-xl-base-1.0) or a local vae safetensors filename (e.g. some-vae.safetensors)",
)
return_images_field = Field(
    True,
    description="Whether to return images or not. If false, and images are instead stored, only the image paths will be returned. If true, images will be returned as base64-encoded strings.",
)
store_images_field = Field(
    False,
    description="Whether to store images or not. If true, images will be stored according to the image_storage_strategy. If false, images will not be stored.",
)


class BaseStableDiffusionParams(BaseModel):
    prompt: Union[str, list[str]] = prompt_field
    num_inference_steps: Optional[int] = num_inference_steps_field
    timesteps: Optional[list[int]] = timesteps_field
    guidance_scale: Optional[float] = guidance_scale_field
    negative_prompt: Optional[Union[str, list[str]]] = negative_prompt_field
    num_images_per_prompt: Optional[int] = num_images_per_prompt_field
    eta: Optional[float] = eta_field
    seed: Optional[Union[int, list[int]]] = seed_field

    class Config:
        extra = Extra.forbid


class InputDimensions(BaseModel):
    height: Optional[int] = height_field
    width: Optional[int] = width_field


class ImageInputs(BaseModel):
    image: Union[str, list[str]] = image_field
    strength: Optional[float] = strength_field


class ControlNetInputs(BaseModel):
    controlnet_conditioning_scale: Optional[
        Union[float, list[float]]
    ] = controlnet_conditioning_scale_field
    guess_mode: Optional[bool] = guess_mode_field
    control_guidance_start: Optional[
        Union[float, list[float]]
    ] = control_guidance_start_field
    control_guidance_end: Optional[
        Union[float, list[float]]
    ] = control_guidance_end_field


class BaseSDXLParams(BaseStableDiffusionParams):
    prompt_2: Optional[Union[str, list[str]]] = prompt_2_field
    negative_prompt_2: Optional[Union[str,
                                      list[str]]] = negative_prompt_2_field
    denoising_end: Optional[float] = denoising_end_field
    original_size: Optional[tuple[int, int]] = original_size_field
    crops_coords_top_left: Optional[tuple[int,
                                          int]] = crops_coords_top_left_field
    target_size: Optional[tuple[int, int]] = target_size_field
    negative_original_size: Optional[tuple[int,
                                           int]] = negative_original_size_field
    negative_crops_coords_top_left: Optional[
        tuple[int, int]
    ] = negative_crops_coords_top_left_field
    negative_target_size: Optional[tuple[int, int]
                                   ] = negative_target_size_field
    guidance_rescale: Optional[float] = guidance_rescale_field


class StableDiffusionPipelineParams(BaseStableDiffusionParams, InputDimensions):
    """
    Parameters for the Stable Diffusion pipeline. [See the diffusers documentation for more details](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
    """

    guidance_rescale: Optional[float] = guidance_rescale_field
    clip_skip: Optional[int] = clip_skip_field


class StableDiffusionImg2ImgPipelineParams(BaseStableDiffusionParams, ImageInputs):
    """
    Parameters for the Stable Diffusion Image to Image pipeline. [See the diffusers documentation for more details](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img#diffusers.StableDiffusionImg2ImgPipeline)
    """

    clip_skip: Optional[int] = clip_skip_field


class StableDiffusionInpaintPipelineParams(
    BaseStableDiffusionParams, ImageInputs, InputDimensions
):
    """
    Parameters for the Stable Diffusion Inpainting pipeline. [See the diffusers documentation for more details](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline)
    """

    mask_image: Union[str, list[str]] = mask_image_field
    clip_skip: Optional[int] = clip_skip_field


class StableDiffusionControlNetPipelineParams(
    BaseStableDiffusionParams, InputDimensions, ControlNetInputs
):
    """
    Parameters for the Stable Diffusion ControlNet pipeline. [See the diffusers documentation for more details](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/controlnet#diffusers.StableDiffusionControlNetPipeline)
    """

    image: Union[str, list[str]] = control_image_field
    clip_skip: Optional[int] = clip_skip_field


class StableDiffusionControlNetImg2ImgPipelineParams(
    BaseStableDiffusionParams, InputDimensions, ControlNetInputs
):
    """
    Parameters for the Stable Diffusion ControlNet Image to Image pipeline. [See the diffusers documentation for more details](https://huggingface.co/docs/diffusers/api/pipelines/controlnet#diffusers.StableDiffusionControlNetImg2ImgPipeline)
    """

    image: Union[str, list[str]] = image_field
    control_image: Union[str, list[str]] = control_image_field
    clip_skip: Optional[int] = clip_skip_field


class StableDiffusionControlNetInpaintPipelineParams(
    BaseStableDiffusionParams, ImageInputs, InputDimensions, ControlNetInputs
):
    """
    Parameters for the Stable Diffusion ControlNet Inpainting pipeline. [See the diffusers documentation for more details](https://huggingface.co/docs/diffusers/api/pipelines/controlnet#diffusers.StableDiffusionControlNetInpaintPipeline)
    """

    mask_image: Union[str, list[str]] = mask_image_field
    control_image: Union[str, list[str]] = control_image_field
    clip_skip: Optional[int] = clip_skip_field


class StableDiffusionXLPipelineParams(BaseSDXLParams, InputDimensions):
    """
    Parameters for the Stable Diffusion XL pipeline. [See the diffusers documentation for more details](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline)
    """


class StableDiffusionXLImg2ImgPipelineParams(BaseSDXLParams, ImageInputs):
    """
    Parameters for the Stable Diffusion XL Image to Image pipeline. [See the diffusers documentation for more details]https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLImg2ImgPipeline)
    """

    denoising_start: Optional[float] = denoising_start_field
    aesthetic_score: Optional[float] = aeshtetic_score_field
    negative_aesthetic_score: Optional[float] = negative_aesthetic_score_field
    clip_skip: Optional[int] = clip_skip_field


OptionalStableDiffusionXLImg2ImgPipelineParams = create_model(
    "OptionalStableDiffusionXLImg2ImgPipelineParams",
    **{
        name: (Optional[type_], None)
        for name, type_ in StableDiffusionXLImg2ImgPipelineParams.__annotations__.items()
    }
)


class StableDiffusionXLInpaintPipelineParams(
    BaseSDXLParams, ImageInputs, InputDimensions
):
    """
    Parameters for the Stable Diffusion XL Inpainting pipeline. [See the diffusers documentation for more details](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline)
    """

    mask_image: Union[str, list[str]] = mask_image_field
    denoising_start: Optional[float] = denoising_start_field
    aesthetic_score: Optional[float] = aeshtetic_score_field
    negative_aesthetic_score: Optional[float] = negative_aesthetic_score_field
    clip_skip: Optional[int] = clip_skip_field


class StableDiffusionXLControlNetPipelineParams(
    BaseSDXLParams, InputDimensions, ControlNetInputs
):
    """
    Parameters for the Stable Diffusion XL ControlNet pipeline. [See the diffusers documentation for more details](https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sdxl#diffusers.StableDiffusionXLControlNetPipeline)
    """

    image: Union[str, list[str]] = control_image_field
    clip_skip: Optional[int] = clip_skip_field


class StableDiffusionXLControlNetImg2ImgPipelineParams(
    BaseSDXLParams, InputDimensions, ControlNetInputs, ImageInputs
):
    """
    Parameters for the Stable Diffusion XL ControlNet Image to Image pipeline. [See the diffusers documentation for more details](https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sdxl#diffusers.StableDiffusionXLControlNetImg2ImgPipeline)
    """

    control_image: Union[str, list[str]] = control_image_field
    aesthetic_score: Optional[float] = aeshtetic_score_field
    negative_aesthetic_score: Optional[float] = negative_aesthetic_score_field
    clip_skip: Optional[int] = clip_skip_field


class StableDiffusionXLControlNetInpaintPipelineParams(
    BaseSDXLParams, InputDimensions, ImageInputs
):
    """
    Parameters for the Stable Diffusion XL ControlNet Inpainting pipeline. [See the diffusers documentation for more details](https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sdxl#diffusers.StableDiffusionXLControlNetInpaintPipeline)
    """

    mask_image: Union[str, list[str]] = mask_image_field
    aesthetic_score: Optional[float] = aeshtetic_score_field
    negative_aesthetic_score: Optional[float] = negative_aesthetic_score_field
    clip_skip: Optional[int] = clip_skip_field


AllParameters = Union[
    StableDiffusionPipelineParams,
    StableDiffusionImg2ImgPipelineParams,
    StableDiffusionInpaintPipelineParams,
    StableDiffusionControlNetPipelineParams,
    StableDiffusionControlNetImg2ImgPipelineParams,
    StableDiffusionControlNetInpaintPipelineParams,
    StableDiffusionXLPipelineParams,
    StableDiffusionXLImg2ImgPipelineParams,
    StableDiffusionXLInpaintPipelineParams,
    StableDiffusionXLControlNetPipelineParams,
    StableDiffusionControlNetImg2ImgPipelineParams,
    StableDiffusionXLControlNetInpaintPipelineParams,
]


class BaseGenerateRequest(BaseModel):
    batch_id: Optional[str] = batch_id_field
    checkpoint: str = checkpoint_field
    scheduler: Optional[str] = scheduler_field
    a1111_scheduler: Optional[str] = a1111_scheduler_field
    vae: Optional[str] = vae_field
    return_images: Optional[bool] = return_images_field
    store_images: Optional[bool] = store_images_field

    class Config:
        extra = Extra.forbid


class GenerateRequest(BaseGenerateRequest):
    pipeline: Optional[PipelineOptions] = pipeline_field
    checkpoint: str = checkpoint_field
    refiner_model: Optional[str] = None
    control_model: Optional[Union[str, list[str]]] = None
    safety_checker: Optional[bool] = config.load_safety_checker
    parameters: AllParameters
    refiner_parameters: Optional[StableDiffusionXLImg2ImgPipelineParams] = None


class GenerateStableDiffusionRequest(BaseGenerateRequest):
    safety_checker: Optional[bool] = config.load_safety_checker
    parameters: StableDiffusionPipelineParams


class GenerateStableDiffusionImg2ImgRequest(BaseGenerateRequest):
    safety_checker: Optional[bool] = config.load_safety_checker
    parameters: StableDiffusionImg2ImgPipelineParams


class GenerateStableDiffusionInpaintRequest(BaseGenerateRequest):
    safety_checker: Optional[bool] = config.load_safety_checker
    parameters: StableDiffusionInpaintPipelineParams


class GenerateStableDiffusionControlNetRequest(BaseGenerateRequest):
    control_model: Optional[Union[str, list[str]]] = None
    safety_checker: Optional[bool] = config.load_safety_checker
    parameters: StableDiffusionControlNetPipelineParams


class GenerateStableDiffusionControlNetImg2ImgRequest(BaseGenerateRequest):
    control_model: Optional[Union[str, list[str]]] = None
    safety_checker: Optional[bool] = config.load_safety_checker
    parameters: StableDiffusionControlNetImg2ImgPipelineParams


class GenerateStableDiffusionControlNetInpaintRequest(BaseGenerateRequest):
    control_model: Optional[Union[str, list[str]]] = None
    safety_checker: Optional[bool] = config.load_safety_checker
    parameters: StableDiffusionControlNetInpaintPipelineParams


class GenerateStableDiffusionXLRequest(BaseGenerateRequest):
    parameters: StableDiffusionXLPipelineParams
    refiner_parameters: Optional[OptionalStableDiffusionXLImg2ImgPipelineParams] = None
    refiner_model: Optional[str] = None


class GenerateStableDiffusionXLImg2ImgRequest(BaseGenerateRequest):
    parameters: StableDiffusionXLImg2ImgPipelineParams
    refiner_parameters: Optional[OptionalStableDiffusionXLImg2ImgPipelineParams] = None
    refiner_model: Optional[str] = None


class GenerateStableDiffusionXLInpaintRequest(BaseGenerateRequest):
    parameters: StableDiffusionXLInpaintPipelineParams
    refiner_parameters: Optional[OptionalStableDiffusionXLImg2ImgPipelineParams] = None
    refiner_model: Optional[str] = None


class GenerateStableDiffusionXLControlNetRequest(BaseGenerateRequest):
    parameters: StableDiffusionXLControlNetPipelineParams
    refiner_parameters: Optional[OptionalStableDiffusionXLImg2ImgPipelineParams] = None
    refiner_model: Optional[str] = None
    control_model: Optional[Union[str, list[str]]] = None


class GenerateStableDiffusionXLControlNetImg2ImgRequest(BaseGenerateRequest):
    parameters: StableDiffusionXLControlNetImg2ImgPipelineParams
    refiner_parameters: Optional[OptionalStableDiffusionXLImg2ImgPipelineParams] = None
    refiner_model: Optional[str] = None
    control_model: Optional[Union[str, list[str]]] = None


class GenerateStableDiffusionXLControlNetInpaintRequest(BaseGenerateRequest):
    parameters: StableDiffusionXLControlNetInpaintPipelineParams
    refiner_parameters: Optional[OptionalStableDiffusionXLImg2ImgPipelineParams] = None
    refiner_model: Optional[str] = None
    control_model: Optional[Union[str, list[str]]] = None


class GenerateMetadata(BaseModel):
    model_load_time: float
    generation_time: float
    b64_encoding_time: float
    total_time: float

    class Config:
        extra = Extra.forbid
        protected_namespaces = ()


class GenerateResponse(BaseModel):
    images: list[str]
    inputs: Union[
        GenerateRequest,
        GenerateStableDiffusionRequest,
        GenerateStableDiffusionImg2ImgRequest,
        GenerateStableDiffusionInpaintRequest,
        GenerateStableDiffusionControlNetRequest,
        GenerateStableDiffusionControlNetImg2ImgRequest,
        GenerateStableDiffusionControlNetInpaintRequest,
        GenerateStableDiffusionXLRequest,
        GenerateStableDiffusionXLImg2ImgRequest,
        GenerateStableDiffusionXLInpaintRequest,
        GenerateStableDiffusionXLControlNetRequest,
        GenerateStableDiffusionXLControlNetImg2ImgRequest,
        GenerateStableDiffusionXLControlNetInpaintRequest,
    ]
    meta: GenerateMetadata

    class Config:
        extra = Extra.forbid


class ModelListFilters(BaseModel):
    loaded: Optional[bool] = False


class LoadCheckpointParams(BaseModel):
    checkpoint: str
    vae: Optional[str] = None


class CPUFrequency(BaseModel):
    current: float
    min: float
    max: float


class CPUPerformance(BaseModel):
    utilization: list[float]
    frequency: list[CPUFrequency]


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
    gpu: list[GPUPerformance]
    packages: PackageVersions


class ModelLoadedEvent(BaseModel):
    checkpoint: str
    time_utc: float
    worker_identity: dict
    stats: SystemPerformance
