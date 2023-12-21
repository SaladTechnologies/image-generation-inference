# 🥗 Image Generation Inference

🚧 This is under active development. Don't use it in production! 🚧

This is a minimalist image generation inference server wrapping Diffusers and stable-fast.

API docs at `/docs`

## Goals

Everything you need and nothing you don't. Server-only, no UI.

The goal here is to support all image generation models and pipelines supported by Diffusers.
Additionally:
- Model swapping
- Lora-per-request
- Controlnet-per-request
- vae swapping

## Supported Pipelines

### StableDiffusionPipeline

### StableDiffusionImg2ImgPipeline

### StableDiffusionInpaintPipeline

### StableDiffusionControlNetPipeline

### StableDiffusionXLPipeline

### StableDiffusionXLImg2ImgPipeline

### StableDiffusionXLInpaintPipeline

## Supported Model Types

- SD1.5
- SD2.1
- SDXL 1.0
- SDXL Turbo

## Environment Variables

| Environment Variable       | Default Value        | Description                                                           |
| -------------------------- | -------------------- | --------------------------------------------------------------------- |
| `DATA_DIR`                 | `/data`              | Base directory for data storage.                                      |
| `IMAGE_DIR`                | `${DATA_DIR}/images` | Directory for storing images, if you use the "disk" storage strategy. |
| `CUDA_GRAPH`               | `false`              | Flag to enable or disable CUDA graph.                                 |
| `LOAD_SAFETY_CHECKER`      | `false`              | Flag to load the safety checker.                                      |
| `HOST`                     | `*`                  | The host address for the service.                                     |
| `PORT`                     | `1234`               | The port number for the service.                                      |
| `LAUNCH_CHECKPOINT`        | `None`               | Path to a specific checkpoint to launch.                              |
| `LAUNCH_VAE`               | `None`               | Path to a specific VAE to launch.                                     |
| `IMAGE_STORAGE_STRATEGY`   | `disk`               | Strategy for storing images (e.g., 'disk', 'post').                   |
| `WEBHOOK_MODEL_LOADED`     | `None`               | Webhook URL for model loaded event.                                   |
| `WEBHOOK_MODEL_UNLOADED`   | `None`               | Webhook URL for model unloaded event.                                 |
| `WEBHOOK_IMAGE_GENERATED`  | `None`               | Webhook URL for image generated event.                                |
| `WEBHOOK_IMAGE_STORED`     | `None`               | Webhook URL for image stored event.                                   |
| `SALAD_MACHINE_ID`         | `None`               | Identifier for the Salad machine.                                     |
| `SALAD_CONTAINER_GROUP_ID` | `None`               | Identifier for the Salad container group.                             |
| `MAX_DATA_DIR_SIZE_GB`     | `0`                  | Maximum size for the data directory, in gigabytes.                    |
| `LOG_LEVEL`                | `info`               | Log level for the service.                                            |
