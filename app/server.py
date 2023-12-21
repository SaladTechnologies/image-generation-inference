import logging
import os

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level)

logging.basicConfig(level=log_level)

import sys
from fastapi import FastAPI, Response, Depends, BackgroundTasks
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
    GenerateRequest,
    ModelListFilters,
    LoadCheckpointParams,
    SystemPerformance,
    GenerateResponse,
    PipelineOptions,
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
)

import config
from monitoring import get_detailed_system_performance
import webhooks
from generate import (
    run_async,
    generate_images_common,
)

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


@app.post("/generate", response_model=GenerateResponse)
async def generate(params: GenerateRequest, background_tasks: BackgroundTasks):
    return await generate_images_common(params, background_tasks, params.pipeline)


@app.post("/generate/StableDiffusionPipeline", response_model=GenerateResponse)
async def generate_with_stable_diffusion_pipeline(
    params: GenerateStableDiffusionRequest, background_tasks: BackgroundTasks
):
    return await generate_images_common(
        params, background_tasks, PipelineOptions.StableDiffusionPipeline
    )


@app.post("/generate/StableDiffusionImg2ImgPipeline", response_model=GenerateResponse)
async def generate_with_stable_diffusion_img2img_pipeline(
    params: GenerateStableDiffusionImg2ImgRequest, background_tasks: BackgroundTasks
):
    return await generate_images_common(
        params, background_tasks, PipelineOptions.StableDiffusionImg2ImgPipeline
    )


@app.post("/generate/StableDiffusionInpaintPipeline", response_model=GenerateResponse)
async def generate_with_stable_diffusion_inpaint_pipeline(
    params: GenerateStableDiffusionInpaintRequest, background_tasks: BackgroundTasks
):
    return await generate_images_common(
        params, background_tasks, PipelineOptions.StableDiffusionInpaintPipeline
    )


@app.post(
    "/generate/StableDiffusionControlNetPipeline", response_model=GenerateResponse
)
async def generate_with_stable_diffusion_controlnet_pipeline(
    params: GenerateStableDiffusionControlNetRequest, background_tasks: BackgroundTasks
):
    return await generate_images_common(
        params, background_tasks, PipelineOptions.StableDiffusionControlNetPipeline
    )


@app.post(
    "/generate/StableDiffusionControlNetImg2ImgPipeline",
    response_model=GenerateResponse,
)
async def generate_with_stable_diffusion_controlnet_img2img_pipeline(
    params: GenerateStableDiffusionControlNetImg2ImgRequest,
    background_tasks: BackgroundTasks,
):
    return await generate_images_common(
        params,
        background_tasks,
        PipelineOptions.StableDiffusionControlNetImg2ImgPipeline,
    )


@app.post(
    "/generate/StableDiffusionControlNetInpaintPipeline",
    response_model=GenerateResponse,
)
async def generate_with_stable_diffusion_controlnet_inpaint_pipeline(
    params: GenerateStableDiffusionControlNetInpaintRequest,
    background_tasks: BackgroundTasks,
):
    return await generate_images_common(
        params,
        background_tasks,
        PipelineOptions.StableDiffusionControlNetInpaintPipeline,
    )


@app.post("/generate/StableDiffusionXLPipeline", response_model=GenerateResponse)
async def generate_with_stable_diffusion_xl_pipeline(
    params: GenerateStableDiffusionXLRequest, background_tasks: BackgroundTasks
):
    return await generate_images_common(
        params, background_tasks, PipelineOptions.StableDiffusionXLPipeline
    )


@app.post("/generate/StableDiffusionXLImg2ImgPipeline", response_model=GenerateResponse)
async def generate_with_stable_diffusion_xl_img2img_pipeline(
    params: GenerateStableDiffusionXLImg2ImgRequest, background_tasks: BackgroundTasks
):
    return await generate_images_common(
        params, background_tasks, PipelineOptions.StableDiffusionXLImg2ImgPipeline
    )


@app.post("/generate/StableDiffusionXLInpaintPipeline", response_model=GenerateResponse)
async def generate_with_stable_diffusion_xl_inpaint_pipeline(
    params: GenerateStableDiffusionXLInpaintRequest, background_tasks: BackgroundTasks
):
    return await generate_images_common(
        params, background_tasks, PipelineOptions.StableDiffusionXLInpaintPipeline
    )


@app.post(
    "/generate/StableDiffusionXLControlNetPipeline", response_model=GenerateResponse
)
async def generate_with_stable_diffusion_xl_controlnet_pipeline(
    params: GenerateStableDiffusionXLControlNetRequest,
    background_tasks: BackgroundTasks,
):
    return await generate_images_common(
        params, background_tasks, PipelineOptions.StableDiffusionXLControlNetPipeline
    )


@app.post(
    "/generate/StableDiffusionXLControlNetImg2ImgPipeline",
    response_model=GenerateResponse,
)
async def generate_with_stable_diffusion_xl_controlnet_img2img_pipeline(
    params: GenerateStableDiffusionXLControlNetImg2ImgRequest,
    background_tasks: BackgroundTasks,
):
    return await generate_images_common(
        params,
        background_tasks,
        PipelineOptions.StableDiffusionXLControlNetImg2ImgPipeline,
    )


@app.post(
    "/generate/StableDiffusionXLControlNetInpaintPipeline",
    response_model=GenerateResponse,
)
async def generate_with_stable_diffusion_xl_controlnet_inpaint_pipeline(
    params: GenerateStableDiffusionXLControlNetInpaintRequest,
    background_tasks: BackgroundTasks,
):
    return await generate_images_common(
        params,
        background_tasks,
        PipelineOptions.StableDiffusionXLControlNetInpaintPipeline,
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
