import unittest
import requests
import json
from test_utils import b64_to_pil, IGITest


class StableDiffusion15Test(IGITest):
    pipeline = "StableDiffusionPipeline"
    checkpoint = "dreamshaper_8.safetensors"

    def test_batch_1_a1111_safe(self):
        """
        Test stable diffusion pipeline with a 1.5 finetune, batch size 1,
        a1111 scheduler alias, safety checker
        """

        url = f"{self.api_url}/generate/{self.pipeline}"
        payload = {
            "checkpoint": self.checkpoint,
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
        self.assertIsImage(response_body["images"][0], (512, 512))

    def test_batch_1_a1111_unsafe(self):
        """
        Test stable diffusion pipeline with a 1.5 finetune, batch size 1,
        a1111 scheduler alias, no safety checker
        """

        url = f"{self.api_url}/generate/{self.pipeline}"
        payload = {
            "checkpoint": self.checkpoint,
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
        self.assertIsImage(response_body["images"][0], (512, 512))

    def test_batch_1_scheduler(self):
        """
        Test stable diffusion pipeline with a 1.5 finetune, batch size 1,
        euler scheduler, no a1111 alias
        """

        url = f"{self.api_url}/generate/{self.pipeline}"
        payload = {
            "checkpoint": self.checkpoint,
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
        self.assertIsImage(response_body["images"][0], (512, 512))

    def test_batch_2_scheduler(self):
        """
        Test stable diffusion pipeline with a 1.5 finetune, batch size 2,
        euler scheduler, no a1111 alias
        """

        url = f"{self.api_url}/generate/{self.pipeline}"
        payload = {
            "checkpoint": self.checkpoint,
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
        self.assertIsImage(response_body["images"][0], (512, 512))
        self.assertIsImage(response_body["images"][1], (512, 512))

    def test_diff_dimensions(self):
        """
        Test stable diffusion pipeline with a 1.5 finetune, batch size 1,
        euler scheduler, no a1111 alias, different dimensions
        """

        url = f"{self.api_url}/generate/{self.pipeline}"
        payload = {
            "checkpoint": self.checkpoint,
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
        self.assertIsImage(response_body["images"][0], (512, 768))
