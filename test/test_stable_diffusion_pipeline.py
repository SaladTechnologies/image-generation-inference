import unittest
import requests
from PIL import Image
import io
import base64
import json


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


api_url = "http://localhost:1234"


class StableDiffusion15Test(unittest.TestCase):
    pipeline = "StableDiffusionPipeline"

    def test_batch_1_a1111_safe(self):
        """
        Test stable diffusion pipeline with a 1.5 finetune, batch size 1,
        a1111 scheduler alias, safety checker
        """

        url = f"{api_url}/generate/{self.pipeline}"
        payload = {
            "checkpoint": "dreamshaper_8.safetensors",
            "a1111_scheduler": "Euler a",
            "safety_checker": True,
            "parameters": {"prompt": "cat", "num_inference_steps": 15},
            "return_images": True,
        }
        response = requests.post(url, json=payload)
        self.assertEqual(
            response.status_code, 200, json.dumps(response.json(), indent=2)
        )

        response_body = response.json()

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)

        # body.images[0] should be a base64 encoded image
        image = b64_to_pil(response_body["images"][0])
        self.assertEqual(image.size, (512, 512))

    def test_batch_1_a1111_unsafe(self):
        """
        Test stable diffusion pipeline with a 1.5 finetune, batch size 1,
        a1111 scheduler alias, no safety checker
        """

        url = f"{api_url}/generate/{self.pipeline}"
        payload = {
            "checkpoint": "dreamshaper_8.safetensors",
            "a1111_scheduler": "Euler a",
            "safety_checker": False,
            "parameters": {"prompt": "cat", "num_inference_steps": 15},
            "return_images": True,
        }
        response = requests.post(url, json=payload)
        self.assertEqual(
            response.status_code, 200, json.dumps(response.json(), indent=2)
        )

        response_body = response.json()

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)

        # body.images[0] should be a base64 encoded image
        image = b64_to_pil(response_body["images"][0])
        self.assertEqual(image.size, (512, 512))

        # image is not all black
        self.assertTrue(image.getbbox())

    def test_batch_1_scheduler(self):
        """
        Test stable diffusion pipeline with a 1.5 finetune, batch size 1,
        euler scheduler, no a1111 alias
        """

        url = f"{api_url}/generate/{self.pipeline}"
        payload = {
            "checkpoint": "dreamshaper_8.safetensors",
            "scheduler": "EulerDiscreteScheduler",
            "parameters": {"prompt": "cat", "num_inference_steps": 15},
            "return_images": True,
        }
        response = requests.post(url, json=payload)
        self.assertEqual(
            response.status_code, 200, json.dumps(response.json(), indent=2)
        )

        response_body = response.json()

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)

        # body.images[0] should be a base64 encoded image
        image = b64_to_pil(response_body["images"][0])
        self.assertEqual(image.size, (512, 512))

        # image is not all black
        self.assertTrue(image.getbbox())

    def test_batch_2_scheduler(self):
        """
        Test stable diffusion pipeline with a 1.5 finetune, batch size 2,
        euler scheduler, no a1111 alias
        """

        url = f"{api_url}/generate/{self.pipeline}"
        payload = {
            "checkpoint": "dreamshaper_8.safetensors",
            "scheduler": "EulerDiscreteScheduler",
            "parameters": {
                "prompt": "cat",
                "num_inference_steps": 15,
                "num_images_per_prompt": 2,
            },
            "return_images": True,
        }
        response = requests.post(url, json=payload)
        self.assertEqual(
            response.status_code, 200, json.dumps(response.json(), indent=2)
        )

        response_body = response.json()

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 2)

        # body.images[0] should be a base64 encoded image
        image1 = b64_to_pil(response_body["images"][0])
        image2 = b64_to_pil(response_body["images"][1])
        self.assertEqual(image1.size, (512, 512))
        self.assertEqual(image2.size, (512, 512))

        # image is not all black
        self.assertTrue(image1.getbbox())
        self.assertTrue(image2.getbbox())

    def test_diff_dimensions(self):
        """
        Test stable diffusion pipeline with a 1.5 finetune, batch size 1,
        euler scheduler, no a1111 alias, different dimensions
        """

        url = f"{api_url}/generate/{self.pipeline}"
        payload = {
            "checkpoint": "dreamshaper_8.safetensors",
            "scheduler": "EulerDiscreteScheduler",
            "parameters": {
                "prompt": "cat",
                "num_inference_steps": 15,
                "height": 768,
                "width": 512,
            },
            "return_images": True,
        }
        response = requests.post(url, json=payload)
        self.assertEqual(
            response.status_code, 200, json.dumps(response.json(), indent=2)
        )

        response_body = response.json()

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)

        # body.images[0] should be a base64 encoded image
        image = b64_to_pil(response_body["images"][0])
        self.assertEqual(image.size, (512, 768))

        # image is not all black
        self.assertTrue(image.getbbox())


if __name__ == "__main__":
    unittest.main()
