import logging
import os

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level)

logging.basicConfig(level=log_level)

import time
import sys
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

import config
from monitoring import get_detailed_system_performance
import uuid
import webhooks
from generate import get_pipes, prepare_parameters, handle_images, run_async

if config.launch_ckpt is not None or config.launch_vae is not None:
    logging.info("Preloading checkpoint %s", config.launch_ckpt)
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
async def generate(params: GenerateParams, background_tasks: BackgroundTasks):
    if params.batch_id is None:
        params.batch_id = str(uuid.uuid4())
    start = time.perf_counter()
    pipe, refiner_pipe = get_pipes(
        checkpoint=params.checkpoint,
        pipeline=params.pipeline,
        vae=params.vae,
        control_model=params.control_model,
        refiner=params.refiner,
        scheduler=params.scheduler,
        a1111_scheduler=params.a1111_scheduler,
        safety_checker=params.safety_checker,
    )
    pipe_loaded = time.perf_counter()

    gen_params, refiner_params, img_decode_time = prepare_parameters(
        params.parameters, params.refiner_parameters
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
        logging.exception(e)


@app.post("/restart")
def restart(background_tasks: BackgroundTasks):
    # Return a 202 to indicate that the request has been accepted for processing,
    # but the processing has not been completed.
    for checkpoint in list_loaded_checkpoints():
        background_tasks.add_task(
            run_async, webhooks.model_unloaded, {"checkpoint": checkpoint}
        )
    for vae in list_local_vae():
        background_tasks.add_task(run_async, webhooks.model_unloaded, {"vae": vae})
    for controlnet in list_local_controlnet():
        background_tasks.add_task(
            run_async, webhooks.model_unloaded, {"controlnet": controlnet}
        )
    for lora in list_local_lora():
        background_tasks.add_task(run_async, webhooks.model_unloaded, {"lora": lora})
    background_tasks.add_task(restart_server)
    return Response(status_code=202)


if __name__ == "__main__":
    uvicorn.run(app, host=config.host, port=config.port)
