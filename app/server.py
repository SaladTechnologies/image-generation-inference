import time
import os
import sys
import torch
from fastapi import FastAPI, Response, Depends, BackgroundTasks
import json
import uvicorn
from models import (
    get_checkpoint,
    list_local_checkpoints,
    list_local_controlnet,
    list_local_lora,
    list_local_vae,
    list_loaded_checkpoints,
)
from schemas import (
    GenerateParams,
    ModelListFilters,
    LoadCheckpointParams,
    SystemPerformance,
)
from image_utils import pil_to_b64, b64_to_pil
import config
from monitoring import get_detailed_system_performance
import logging


logging.basicConfig(level=logging.INFO)


if config.launch_ckpt is not None or config.launch_vae is not None:
    print(f"Preloading checkpoint {config.launch_ckpt}", flush=True)
    get_checkpoint(config.launch_ckpt, config.launch_vae)

app = FastAPI(
    title="🥗 Image Generation Inference",
    version=config.version,
)


@app.get("/hc")
async def health_check():
    return {"status": "ok", "version": config.version}


@app.get("/stats", response_model=SystemPerformance)
async def system_stats():
    return get_detailed_system_performance()


@app.post("/generate")
async def generate(params: GenerateParams):
    start = time.perf_counter()
    model = get_checkpoint(params.checkpoint, params.vae)
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
    elif params.a1111_scheduler is not None:
        pipe.scheduler = model.get_a1111_scheduler(params.a1111_scheduler)

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

    # We need to convert base64 images to PIL images
    img_decode_time = 0
    if "image" in gen_params:
        b64_decode = time.perf_counter()
        if isinstance(gen_params["image"], str):
            gen_params["image"] = b64_to_pil(gen_params["image"])
        elif isinstance(gen_params["image"], list):
            gen_params["image"] = [b64_to_pil(img) for img in gen_params["image"]]
        img_decode_time += time.perf_counter() - b64_decode
    if "mask_image" in gen_params:
        b64_decode = time.perf_counter()
        if isinstance(gen_params["mask_image"], str):
            gen_params["mask_image"] = b64_to_pil(gen_params["mask_image"])
        elif isinstance(gen_params["mask_image"], list):
            gen_params["mask_image"] = [
                b64_to_pil(img) for img in gen_params["mask_image"]
            ]
        img_decode_time += time.perf_counter() - b64_decode

    # Manage safety checker preferences
    if params.safety_checker and hasattr(pipe, "safety_checker"):
        pipe.safety_checker = model.get_safety_checker()
        pipe.feature_extractor = model.get_feature_extractor()
    elif hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
        pipe.feature_extractor = None

    try:
        gen_start = time.perf_counter()
        images = pipe(**gen_params).images
        stop = time.perf_counter()
        print(f"Generated {len(images)} images in {stop - gen_start} seconds")
        # if we're returning images, then we need to encode them as base64
        if params.return_images:
            images = [pil_to_b64(img) for img in images]
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
        return Response(
            json.dumps({"error": str(e)}),
            status_code=500,
            media_type="application/json",
        )


@app.post("/load/checkpoint", response_model=list[str])
async def load(params: LoadCheckpointParams):
    get_checkpoint(params.checkpoint, params.vae)
    return list_loaded_checkpoints()


@app.get("/checkpoints", response_model=list[str])
def list_checkpoints(params: ModelListFilters = Depends()):
    if params.loaded:
        return list_loaded_checkpoints()
    return list_local_checkpoints()


@app.get("/controlnet", response_model=list[str])
def list_controlnet():
    return list_local_controlnet()


@app.get("/lora", response_model=list[str])
def list_lora():
    return list_local_lora()


@app.get("/vae", response_model=list[str])
def list_vae():
    return list_local_vae()


def restart_server():
    try:
        # Note: sys.executable is the path to the Python interpreter
        #       sys.argv is the list of command line arguments passed to the Python script
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        # Handle exceptions if any
        print(f"Error during program restart: {e}")


@app.post("/restart")
def restart(background_tasks: BackgroundTasks):
    # Return a 202 to indicate that the request has been accepted for processing,
    # but the processing has not been completed.
    background_tasks.add_task(restart_server)
    return Response(status_code=202)


if __name__ == "__main__":
    uvicorn.run(app, host=config.host, port=config.port)
