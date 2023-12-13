import time
import os
import torch
from fastapi import FastAPI, Response, Depends
import json
import uvicorn
import io
from models import (
    get_checkpoint,
    list_local_checkpoints,
    list_local_controlnet,
    list_local_lora,
    list_local_vae,
    list_loaded_checkpoints,
    unload_checkpoint,
)
from schemas import GenerateParams, ModelListFilters, UnloadCheckpointParams

import base64
from PIL import Image
from __version__ import VERSION

host = os.getenv("HOST", "*")
port = int(os.getenv("PORT", "1234"))

launch_ckpt = os.getenv("LAUNCH_CHECKPOINT", None)
if launch_ckpt is not None:
    print(f"Preloading checkpoint {launch_ckpt}", flush=True)
    get_checkpoint(launch_ckpt)

app = FastAPI()


@app.get("/hc")
async def health_check():
    return {"status": "ok", "version": VERSION}


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


@app.post("/unload")
async def unload(params: UnloadCheckpointParams):
    unload_checkpoint(params.checkpoint)
    return {"status": "ok"}


@app.get("/checkpoints")
def list_checkpoints(params: ModelListFilters = Depends()):
    if params.loaded:
        return list_loaded_checkpoints()
    return list_local_checkpoints()


@app.get("/controlnet")
def list_controlnet():
    return list_local_controlnet()


@app.get("/lora")
def list_lora():
    return list_local_lora()


@app.get("/vae")
def list_vae():
    return list_local_vae()


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
