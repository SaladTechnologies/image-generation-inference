import unittest
import requests
from PIL import Image
import io
import base64
import time


def b64_to_pil(image: str) -> Image.Image:
    img = Image.open(io.BytesIO(base64.b64decode(image)))

    # Convert image to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Coerce image dimmensions to be a multiple of 8
    if img.size[0] % 8 != 0:
        img = img.resize((img.size[0] - img.size[0] % 8, img.size[1]))
    if img.size[1] % 8 != 0:
        img = img.resize((img.size[0], img.size[1] - img.size[1] % 8))

    return img


def restart_server():
    start = time.perf_counter()
    try:
        requests.post(
            "http://localhost:1234/restart",
            headers={"Content-Type": "application/json"},
        )
    except:
        pass
    time.sleep(0.2)

    # Poll /hc every .1 seconds until it returns 200
    while True:
        try:
            response = requests.get("http://localhost:1234/hc")
            response.raise_for_status()
        except:
            time.sleep(0.1)
            continue
        break
    print(f"Restarted server in {time.perf_counter() - start:.2f}s")


class IGITest(unittest.TestCase):
    api_url = "http://localhost:1234"
    checkpoint = None
    pipeline = None

    @classmethod
    def setUpClass(cls):
        if cls.checkpoint is None or cls.pipeline is None:
            print("Skipping test")
            return
        loaded_checkpoints = requests.get(
            f"{cls.api_url}/checkpoints?loaded=true"
        ).json()
        if (
            len(loaded_checkpoints) > 0 and cls.checkpoint not in loaded_checkpoints
        ) or len(loaded_checkpoints) > 1:
            print(f"Unloading checkpoints {loaded_checkpoints}")
            restart_server()
        elif len(loaded_checkpoints) == 1 and loaded_checkpoints[0] == cls.checkpoint:
            print(f"Checkpoint {cls.checkpoint} already loaded")
            return

        print(f"Loading checkpoint {cls.checkpoint}")
        requests.post(
            f"{cls.api_url}/load/checkpoint",
            json={"checkpoint": cls.checkpoint},
        )

    def assertIsImage(self, image: str, size: tuple[int, int] = None):
        """
        Asserts that the given string is a valid base64 encoded image, and that
        the image has the given size (if provided). Finally, asserts that the
        image is not empty.
        """
        img = b64_to_pil(image)
        if size is not None:
            self.assertEqual(img.size, size)
        self.assertEqual(img.mode, "RGB")
        self.assertTrue(img.getbbox())
