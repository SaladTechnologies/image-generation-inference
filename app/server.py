import time
import os
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from typing import Optional
import json
import uvicorn
import io
from models import get_checkpoint
from __version__ import VERSION

host = os.getenv("HOST", "*")
port = int(os.getenv("PORT", "1234"))

app = FastAPI()


@app.get("/hc")
async def health_check():
    return {"status": "ok", "version": VERSION}


class GenerateParams(BaseModel):
    prompt: str
    checkpoint: str


@app.post("/generate")
async def generate(params: GenerateParams):
    start = time.perf_counter()
    checkpoint = get_checkpoint(params.checkpoint)
    try:
        images = checkpoint(params.prompt)
        stop = time.perf_counter()
        print(f"Generated {len(images)} images in {stop - start} seconds")
        print(images)
        return {"images": images}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
