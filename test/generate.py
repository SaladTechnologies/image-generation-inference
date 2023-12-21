import os
import sys
import json
import base64
import requests
from pathlib import Path
import time

last_model = None


def do_job(file_path, fixture_dir, outputs_dir):
    with open(file_path) as f:
        body = json.load(f)
    global last_model
    pipeline = os.path.basename(fixture_dir)

    name = Path(file_path).stem

    print(f"Generating {file_path}")

    if "parameters" in body:
        if "image" in body["parameters"]:
            with open(
                os.path.join(fixture_dir, body["parameters"]["image"]), "rb"
            ) as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode()
            body["parameters"]["image"] = encoded_image

        if "mask_image" in body["parameters"]:
            with open(
                os.path.join(fixture_dir, body["parameters"]["mask_image"]), "rb"
            ) as mask_file:
                encoded_mask_image = base64.b64encode(mask_file.read()).decode()
            body["parameters"]["mask_image"] = encoded_mask_image

    current_model = body["checkpoint"] if "checkpoint" in body else None
    if last_model and last_model != current_model:
        unload_model()

    result = requests.post(
        "http://localhost:1234/generate",
        headers={"Content-Type": "application/json"},
        json=body,
    )
    body = result.json()
    if result.status_code != 200:
        print(
            f"----------------------\nFailed to generate {name}.json\n----------------------"
        )
        print(json.dumps(body, indent=2))
        return

    last_model = current_model

    meta = body["meta"]
    print(json.dumps(meta, indent=2))

    if body["inputs"]["return_images"]:
        for idx, image in enumerate(result.json().get("images", [])):
            filename = os.path.join(outputs_dir, f"{pipeline}-{name}-{idx + 1}.jpg")
            with open(filename, "wb") as img_file:
                img_file.write(base64.b64decode(image))
    elif body["inputs"]["store_images"]:
        print(json.dumps(body, indent=2))


def unload_model():
    start = time.perf_counter()
    try:
        requests.post(
            "http://localhost:1234/restart",
            headers={"Content-Type": "application/json"},
        )
    except requests.HTTPError:
        pass

    # Poll /hc every .1 seconds until it returns 200
    while True:
        try:
            requests.get("http://localhost:1234/hc")
        except:
            time.sleep(0.1)
            continue
        break
    print(f"Restarted server in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    pipeline = sys.argv[1] if len(sys.argv) > 1 else "StableDiffusionPipeline"
    job_id = sys.argv[2] if len(sys.argv) > 2 else None
    fixture_dir = os.environ.get("FIXTURE_DIR", f"test/pipelines/{pipeline}")
    outputs_dir = os.environ.get("OUTPUT_DIR", "test/outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    if job_id:
        do_job(os.path.join(fixture_dir, f"{job_id}.json"), fixture_dir, outputs_dir)
    else:
        for file in sorted(Path(fixture_dir).glob("*.json")):
            do_job(file, fixture_dir, outputs_dir)
    # unload_model()
