import logging
from typing import Optional
from schemas import (
    PipelineOptions,
    AllParameters,
    StableDiffusionXLImg2ImgPipelineParams,
)
from models import get_checkpoint
from fastapi import HTTPException, BackgroundTasks
import json
from PIL import Image
import torch
import time
from image_utils import pil_to_b64, prepare_image_field, store_image
import asyncio
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline


def get_pipes(
    checkpoint: str,
    pipeline: PipelineOptions,
    vae: str = None,
    control_model: str = None,
    refiner: str = None,
    scheduler: str = None,
    a1111_scheduler: str = None,
    safety_checker: bool = False,
) -> tuple[DiffusionPipeline, Optional[StableDiffusionXLImg2ImgPipeline]]:
    model = get_checkpoint(checkpoint, vae=vae)

    refiner_pipe = None
    try:
        pipe = model.get_pipeline(pipeline.value, control_model=control_model)
    except Exception as e:
        logging.exception(e)
        raise HTTPException(
            detail=json.dumps({"error": str(e)}),
            status_code=500,
            headers={"Content-Type": "application/json"},
        )
    if refiner is not None:
        try:
            refiner = get_checkpoint(refiner, refiner_for=checkpoint)
            refiner_pipe = refiner.get_pipeline()
        except Exception as e:
            logging.exception(e)
            raise HTTPException(
                detail=json.dumps({"error": str(e)}),
                status_code=500,
                headers={"Content-Type": "application/json"},
            )

    # Check if the scheduler is compatible with the pipeline
    compatible_schedulers = model.get_compatible_schedulers(pipeline.value)
    if scheduler is not None and scheduler not in compatible_schedulers:
        raise HTTPException(
            detail=json.dumps(
                {
                    "error": f"Scheduler {scheduler} is not compatible with pipeline {pipeline.value}. Compatible schedulers are {compatible_schedulers}"
                }
            ),
            status_code=400,
            headers={"Content-Type": "application/json"},
        )
    elif scheduler is not None:
        pipe.scheduler = model.get_scheduler(scheduler)
    elif a1111_scheduler is not None:
        pipe.scheduler = model.get_a1111_scheduler(a1111_scheduler)

    # Manage safety checker preferences
    if safety_checker and hasattr(pipe, "safety_checker"):
        pipe.safety_checker = model.get_safety_checker()
        pipe.feature_extractor = model.get_feature_extractor()
    elif hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
        pipe.feature_extractor = None

    return pipe, refiner_pipe


def prepare_parameters(
    gen_params: AllParameters,
    refiner_params: StableDiffusionXLImg2ImgPipelineParams = None,
) -> tuple[dict, dict, float]:
    gen_params = {k: v for k, v in gen_params.model_dump().items() if v is not None}

    # We need to convert seed to torch generators
    if "seed" in gen_params:
        if not isinstance(gen_params["seed"], list):
            generator = torch.Generator(device="cuda").manual_seed(gen_params["seed"])
        else:
            generator = [
                torch.Generator(device="cuda").manual_seed(s)
                for s in gen_params["seed"]
            ]
        gen_params["generator"] = generator
        del gen_params["seed"]

    # We need to convert base64 images to PIL images
    img_decode_time = 0
    if "image" in gen_params:
        b64_decode = time.perf_counter()
        gen_params["image"] = prepare_image_field(gen_params["image"])
        img_decode_time += time.perf_counter() - b64_decode
    if "mask_image" in gen_params:
        b64_decode = time.perf_counter()
        gen_params["mask_image"] = prepare_image_field(gen_params["mask_image"])
        img_decode_time += time.perf_counter() - b64_decode
    if "control_image" in gen_params:
        b64_decode = time.perf_counter()
        gen_params["control_image"] = prepare_image_field(gen_params["control_image"])
        img_decode_time += time.perf_counter() - b64_decode

    if (
        refiner_params is not None
        and "denoising_end" in gen_params
        and gen_params["denoising_end"] is not None
        and gen_params["denoising_end"] < 1
    ):
        gen_params["output_type"] = "latent"

    # Refiner values should default to the original generation parameters
    if refiner_params is not None:
        refiner_params = refiner_params.model_dump()
        for k, v in refiner_params.items():
            if v is None and k in gen_params:
                refiner_params[k] = gen_params[k]

        refiner_params = {k: v for k, v in refiner_params.items() if v is not None}

    return gen_params, refiner_params, img_decode_time


def handle_images(
    batch_id: str,
    images: list[Image.Image],
    background_tasks: BackgroundTasks,
    store_images: bool = False,
    return_images: bool = True,
) -> list[str]:
    image_paths = []
    if store_images:
        for idx, img in enumerate(images):
            image_name = f"{batch_id}-{idx}"
            image_paths.append(f"{image_name}.jpg")
            background_tasks.add_task(run_async, store_image, img, image_name)
    # if we're returning images, then we need to encode them as base64
    if return_images:
        images = [pil_to_b64(img) for img in images]
    else:
        images = image_paths

    return images


def run_async(func, *args):
    asyncio.run(func(*args))
