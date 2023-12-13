import time
import os
import torch
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from typing import Optional, Union, List, Tuple
import json
import uvicorn
import io
from models import get_checkpoint
from enum import Enum
import base64
from PIL import Image
from __version__ import VERSION

host = os.getenv("HOST", "*")
port = int(os.getenv("PORT", "1234"))

launch_ckpt = os.getenv("LAUNCH_CHECKPOINT", None)
if launch_ckpt is not None:
    print(f"Preloading checkpoint {launch_ckpt}")
    get_checkpoint(launch_ckpt)

app = FastAPI()


@app.get("/hc")
async def health_check():
    return {"status": "ok", "version": VERSION}


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
    parameters: Union[StableDiffusionPipelineParams, StableDiffusionXLPipelineParams]
    return_images: Optional[bool] = True


def pil_to_b64(image: Image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")


@app.post("/generate")
async def generate(params: GenerateParams):
    start = time.perf_counter()
    model = get_checkpoint(params.checkpoint)
    try:
        pipe = model.get_pipeline(params.pipeline.value)
    except Exception as e:
        return Response(
            json.dumps({"error": str(e)}),
            status_code=500,
            media_type="application/json",
        )
    pipe_loaded = time.perf_counter()

    # Check if the scheduler is compatible with the pipeline
    compatible_schedulers = model.get_compatible_schedulers(params.pipeline.value)
    if params.scheduler is not None and params.scheduler not in compatible_schedulers:
        return Response(
            json.dumps(
                {
                    "error": f"Scheduler {params.scheduler} is not compatible with pipeline {params.pipeline}. Compatible schedulers are {compatible_schedulers}"
                }
            ),
            status_code=400,
            media_type="application/json",
        )
    elif params.scheduler is not None:
        pipe.scheduler = model.get_scheduler(params.scheduler)

    gen_params = {
        k: v for k, v in params.parameters.model_dump().items() if v is not None
    }

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

    try:
        images = pipe(**gen_params).images
        stop = time.perf_counter()
        print(f"Generated {len(images)} images in {stop - pipe_loaded} seconds")
        # if we're returning images, then we need to encode them as base64
        if params.return_images:
            images = [pil_to_b64(img) for img in images]
        b64_stop = time.perf_counter()
        return {
            "images": images,
            "inputs": params.model_dump(),
            "meta": {
                "model_load_time": pipe_loaded - start,
                "generation_time": stop - pipe_loaded,
                "b64_encoding_time": b64_stop - stop,
                "total_time": b64_stop - start,
            },
        }
    except Exception as e:
        # Return a 500 if something goes wrong
        return Response(
            json.dumps({"error": str(e)}),
            status_code=500,
            media_type="application/json",
        )


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
