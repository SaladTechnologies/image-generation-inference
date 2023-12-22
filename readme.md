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

## Tests

```bash
python -m unittest discover -s test
```

## Support

- ✅ = Supported, with tests
- 🤷 = Expected to work, but not tested yet

| Pipeline                                   | SD1.5 | SD2.1 | SDXL  | SDXL Turbo |
| ------------------------------------------ | :---: | :---: | :---: | :--------: |
| StableDiffusionPipeline                    |   ✅   |   ✅   |       |            |
| StableDiffusionImg2ImgPipeline             |   ✅   |   ✅   |       |            |
| StableDiffusionInpaintPipeline             |   ✅   |   ✅   |       |            |
| StableDiffusionControlNetPipeline          |   ✅   |       |       |            |
| StableDiffusionControlNetImg2ImgPipeline   |   🤷   |       |       |            |
| StableDiffusionControlNetInpaintPipeline   |   🤷   |       |       |            |
| StableDiffusionXLPipeline                  |       |       |   ✅   |     ✅      |
| StableDiffusionXLImg2ImgPipeline           |       |       |   ✅   |     ✅      |
| StableDiffusionXLInpaintPipeline           |       |       |   ✅   |     ✅      |
| StableDiffusionXLControlNetPipeline        |       |       |   🤷   |     🤷      |
| StableDiffusionXLControlNetImg2ImgPipeline |       |       |   🤷   |     🤷      |
| StableDiffusionXLControlNetInpaintPipeline |       |       |   🤷   |     🤷      |




## Webhooks

IGI emits webhooks for various events. The webhook URLs can be configured via environment variables. All webhooks include the following fields:

```json
{
  "event": "event.name",
  "node_info": {
    "identity": {
      "salad_machine_id": "string",
      "salad_container_group_id": "string"
    },
    "system_stats": {
      "cpu": {
        "utilization": [
          0
        ],
        "frequency": [
          {
            "current": 0,
            "min": 0,
            "max": 0
          }
        ]
      },
      "memory": {
        "total": 0,
        "available": 0,
        "percent": 0,
        "used": 0,
        "free": 0
      },
      "storage": {
        "used": 0,
        "free": 0
      },
      "gpu": [
        {
          "id": 0,
          "name": "string",
          "load": 0,
          "free_memory": 0,
          "used_memory": 0,
          "total_memory": 0,
          "temperature": 0
        }
      ],
      "packages": {
        "torch": "string",
        "cuda": "string",
        "xformers": "string",
        "triton": "string",
        "diffusers": "string",
        "transformers": "string",
        "stable_fast": "string"
      }
    }
  }
}
```

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

### Storage Strategies

#### Disk

The disk storage strategy stores images on disk. This is the default strategy. Images will only be stored when a generation request includes `{"store_images": true}`. The images will be stored in the directory specified by the `IMAGE_DIR` environment variable. The `image.stored` event will be sent to the `WEBHOOK_IMAGE_STORED` webhook URL via POST, with a payload of the form:

```
{
  "image": "/data/images/sdfasdfasdfasdf.jpg",
  "event": "image.stored",
  ...
}
```

#### Post

The post storage strategy sends images to the `WEBHOOK_IMAGE_STORED` webhook URL via POST. Images will only be stored when a generation request includes `{"store_images": true}`. The `image.stored` event will be sent to the `WEBHOOK_IMAGE_STORED` webhook URL via POST, with a payload of the form:

```
{
  "image": "base64-encoded-image-string"
  "event": "image.stored",
  ...
}
```