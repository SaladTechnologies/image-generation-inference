# 🥗 Image Generation Inference

🚧 This is under active development. Don't use it in production! 🚧

This is a minimalist image generation inference server wrapping Diffusers and stable-fast.

API docs at `/docs`

## Goals

Everything you need and nothing you don't. Server-only, no UI.

The goal here is to support all image generation models supported by Diffusers.
Additionally:
- Model swapping
- Lora-per-request
- Controlnet-per-request
- vae swapping

## Supported Pipelines

### StableDiffusionPipeline

### StableDiffusionImg2ImgPipeline

### StableDiffusionInpaintPipeline

### StableDiffusionXLPipeline

### StableDiffusionXLImg2ImgPipeline

## Supported Model Types

- SD1.5
- SD2.1
- SDXL Turbo