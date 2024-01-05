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
import uuid


def get_pipes(
    checkpoint: str,
    pipeline: PipelineOptions,
    vae: str = None,
    control_model: str = None,
    refiner_model: str = None,
    scheduler: str = None,
    a1111_scheduler: str = None,
    safety_checker: bool = False,
    **kwargs,
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
    if refiner_model is not None:
        try:
            refiner_model = get_checkpoint(refiner_model, is_refiner=True)
            refiner_pipe = refiner_model.get_pipeline()
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
    if "image" in gen_params and "control_image" in gen_params:
        gen_params["control_image"] = gen_params["control_image"].resize(
            gen_params["image"].size
        )

    if (
        refiner_params is not None
        and "denoising_end" in gen_params
        and gen_params["denoising_end"] is not None
        and gen_params["denoising_end"] < 1
    ):
        gen_params["output_type"] = "latent"

    # Refiner values should default to the original generation parameters
    if refiner_params is not None:
        model_fields = [
            k
            for k in list(StableDiffusionXLImg2ImgPipelineParams.model_fields.keys())
            if k != "denoising_end"
        ]
        refiner_params = refiner_params.model_dump()
        # Iterate through all field names on the pydantic class
        for field in model_fields:
            if (
                field not in refiner_params or refiner_params[field] is None
            ) and field in gen_params:
                refiner_params[field] = gen_params[field]

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


async def generate_images_common(
    params,
    background_tasks: BackgroundTasks,
    pipeline_option: PipelineOptions,
):
    if params.batch_id is None:
        params.batch_id = str(uuid.uuid4())
    start = time.perf_counter()
    get_pipes_params = params.model_dump()
    get_pipes_params["pipeline"] = pipeline_option
    pipe, refiner_pipe = get_pipes(**get_pipes_params)
    pipe_loaded = time.perf_counter()

    gen_params, refiner_params, img_decode_time = prepare_parameters(
        params.parameters,
        params.refiner_parameters if hasattr(params, "refiner_parameters") else None,
    )

    try:
        gen_start = time.perf_counter()
        images = pipe(**gen_params).images
        if refiner_pipe is not None:
            refiner_params["image"] = images
            images = refiner_pipe(**refiner_params).images
        stop = time.perf_counter()
        logging.info("Generated %d images in %s seconds", len(images), stop - gen_start)
        images = handle_images(
            params.batch_id,
            images,
            background_tasks,
            store_images=params.store_images,
            return_images=params.return_images,
        )
        b64_stop = time.perf_counter()
        return {
            "images": images,
            "inputs": params.model_dump(),
            "meta": {
                "model_load_time": pipe_loaded - start,
                "generation_time": stop - gen_start,
                "b64_encoding_time": b64_stop - stop + img_decode_time,
                "total_time": b64_stop - start,
            },
        }
    except Exception as e:
        logging.exception(e)
        # Return a 500 if something goes wrong
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
