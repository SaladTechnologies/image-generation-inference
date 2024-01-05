from test_utils import IGITest
import base64
import os

current_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(current_path, "cat.png"), "rb") as f:
    cat = base64.b64encode(f.read()).decode()

with open(os.path.join(current_path, "mask.jpg"), "rb") as f:
    mask = base64.b64encode(f.read()).decode()

with open(os.path.join(current_path, "qr.png"), "rb") as f:
    qr = base64.b64encode(f.read()).decode()


class StableDiffusionPipelineTest(IGITest):
    pipeline = "StableDiffusionPipeline"
    checkpoint = "dreamshaper_8.safetensors"
    url = f"{IGITest.api_url}/generate/{pipeline}"

    def test_batch_1_a1111_safe(self):
        """
        Test stable diffusion pipeline with a 1.5 finetune, batch size 1,
        a1111 scheduler alias, safety checker
        """

        payload = {
            "checkpoint": self.checkpoint,
            "a1111_scheduler": "Euler a",
            "safety_checker": True,
            "parameters": {"prompt": "cat", "num_inference_steps": 15},
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)
        self.assertIsImage(response_body["images"][0], (512, 512))

    def test_batch_1_a1111_unsafe(self):
        """
        Test stable diffusion pipeline with a 1.5 finetune, batch size 1,
        a1111 scheduler alias, no safety checker
        """

        payload = {
            "checkpoint": self.checkpoint,
            "a1111_scheduler": "Euler a",
            "safety_checker": False,
            "parameters": {"prompt": "cat", "num_inference_steps": 15},
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)

        # body.images[0] should be a base64 encoded image
        self.assertIsImage(response_body["images"][0], (512, 512))

    def test_batch_1_scheduler(self):
        """
        Test stable diffusion pipeline with a 1.5 finetune, batch size 1,
        euler scheduler, no a1111 alias
        """

        payload = {
            "checkpoint": self.checkpoint,
            "scheduler": "EulerDiscreteScheduler",
            "parameters": {"prompt": "cat", "num_inference_steps": 15},
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)

        # body.images[0] should be a base64 encoded image
        self.assertIsImage(response_body["images"][0], (512, 512))

    def test_batch_2_scheduler(self):
        """
        Test stable diffusion pipeline with a 1.5 finetune, batch size 2,
        euler scheduler, no a1111 alias
        """

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
        response_body = self.assertPostSuccessful(payload)

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
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)

        # body.images[0] should be a base64 encoded image
        self.assertIsImage(response_body["images"][0], (512, 768))


class StableDiffusionImg2ImgPipelineTest(IGITest):
    pipeline = "StableDiffusionImg2ImgPipeline"
    checkpoint = "dreamshaper_8.safetensors"
    url = f"{IGITest.api_url}/generate/{pipeline}"

    def test_batch_1_a1111_safe(self):
        """
        Test stable diffusion img2img pipeline with a 1.5 finetune, batch size 1,
        a1111 scheduler alias, safety checker
        """
        payload = {
            "checkpoint": self.checkpoint,
            "a1111_scheduler": "Euler a",
            "safety_checker": True,
            "parameters": {
                "prompt": "stained glass cat",
                "num_inference_steps": 15,
                "image": cat,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)
        self.assertIsImage(response_body["images"][0], (512, 512))

    def test_batch_1_a1111_unsafe(self):
        """
        Test stable diffusion img2img pipeline with a 1.5 finetune, batch size 1,
        a1111 scheduler alias, no safety checker
        """

        payload = {
            "checkpoint": self.checkpoint,
            "a1111_scheduler": "Euler a",
            "safety_checker": False,
            "parameters": {
                "prompt": "stained glass cat",
                "num_inference_steps": 15,
                "image": cat,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)

        # body.images[0] should be a base64 encoded image
        self.assertIsImage(response_body["images"][0], (512, 512))

    def test_batch_1_scheduler(self):
        """
        Test stable diffusion img2img pipeline with a 1.5 finetune, batch size 1,
        euler scheduler, no a1111 alias
        """

        payload = {
            "checkpoint": self.checkpoint,
            "scheduler": "EulerDiscreteScheduler",
            "parameters": {
                "prompt": "stained glass cat",
                "num_inference_steps": 15,
                "image": cat,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)

        # body.images[0] should be a base64 encoded image
        self.assertIsImage(response_body["images"][0], (512, 512))

    def test_batch_2_scheduler(self):
        """
        Test stable diffusion img2img pipeline with a 1.5 finetune, batch size 2,
        euler scheduler, no a1111 alias
        """

        payload = {
            "checkpoint": self.checkpoint,
            "scheduler": "EulerDiscreteScheduler",
            "parameters": {
                "prompt": "stained glass cat",
                "num_inference_steps": 15,
                "num_images_per_prompt": 2,
                "image": cat,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 2)

        # body.images[0] should be a base64 encoded image
        self.assertIsImage(response_body["images"][0], (512, 512))
        self.assertIsImage(response_body["images"][1], (512, 512))


class StableDiffusionInpaintPipelineTest(IGITest):
    pipeline = "StableDiffusionInpaintPipeline"
    checkpoint = "dreamshaper_8.safetensors"
    url = f"{IGITest.api_url}/generate/{pipeline}"

    def test_batch_1_a1111_safe(self):
        """
        Test stable diffusion inpaint pipeline with a 1.5 finetune, batch size 1,
        a1111 scheduler alias, safety checker
        """
        payload = {
            "checkpoint": self.checkpoint,
            "a1111_scheduler": "Euler a",
            "safety_checker": True,
            "parameters": {
                "prompt": "burning inferno",
                "num_inference_steps": 15,
                "image": cat,
                "mask_image": mask,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)
        self.assertIsImage(response_body["images"][0], (512, 512))

    def test_batch_1_a1111_unsafe(self):
        """
        Test stable diffusion inpaint pipeline with a 1.5 finetune, batch size 1,
        a1111 scheduler alias, no safety checker
        """

        payload = {
            "checkpoint": self.checkpoint,
            "a1111_scheduler": "Euler a",
            "safety_checker": False,
            "parameters": {
                "prompt": "burning inferno",
                "num_inference_steps": 15,
                "image": cat,
                "mask_image": mask,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)

        # body.images[0] should be a base64 encoded image
        self.assertIsImage(response_body["images"][0], (512, 512))

    def test_batch_1_scheduler(self):
        """
        Test stable diffusion inpaint pipeline with a 1.5 finetune, batch size 1,
        euler scheduler, no a1111 alias
        """

        payload = {
            "checkpoint": self.checkpoint,
            "scheduler": "EulerDiscreteScheduler",
            "parameters": {
                "prompt": "burning inferno",
                "num_inference_steps": 15,
                "image": cat,
                "mask_image": mask,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)

        # body.images[0] should be a base64 encoded image
        self.assertIsImage(response_body["images"][0], (512, 512))

    def test_batch_2_scheduler(self):
        """
        Test stable diffusion inpaint pipeline with a 1.5 finetune, batch size 2,
        euler scheduler, no a1111 alias
        """

        payload = {
            "checkpoint": self.checkpoint,
            "scheduler": "EulerDiscreteScheduler",
            "parameters": {
                "prompt": "burning inferno",
                "num_inference_steps": 15,
                "num_images_per_prompt": 2,
                "image": cat,
                "mask_image": mask,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 2)

        # body.images[0] should be a base64 encoded image
        self.assertIsImage(response_body["images"][0], (512, 512))
        self.assertIsImage(response_body["images"][1], (512, 512))


class StableDiffusionControlNetPipelineTest(IGITest):
    checkpoint = "dreamshaper_8.safetensors"
    pipeline = "StableDiffusionControlNetPipeline"
    control_model = "qrCodeMonster_v20.safetensors"
    url = f"{IGITest.api_url}/generate/{pipeline}"

    def test_batch_1_a1111_safe(self):
        """
        Test stable diffusion controlnet pipeline with a 1.5 finetune, batch size 1,
        a1111 scheduler alias, safety checker
        """

        payload = {
            "checkpoint": self.checkpoint,
            "control_model": self.control_model,
            "a1111_scheduler": "Euler a",
            "safety_checker": True,
            "parameters": {
                "prompt": "delicate lace sticker, vector art",
                "num_inference_steps": 20,
                "image": qr,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)

        # body.images[0] should be a base64 encoded image
        self.assertIsImage(response_body["images"][0], (1024, 1024))

    def test_batch_1_a1111_unsafe(self):
        """
        Test stable diffusion controlnet pipeline with a 1.5 finetune, batch size 1,
        a1111 scheduler alias, no safety checker
        """

        payload = {
            "checkpoint": self.checkpoint,
            "control_model": self.control_model,
            "a1111_scheduler": "Euler a",
            "safety_checker": False,
            "parameters": {
                "prompt": "delicate lace sticker, vector art",
                "num_inference_steps": 20,
                "image": qr,
                "width": 512,
                "height": 512,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)

        # body.images[0] should be a base64 encoded image
        self.assertIsImage(response_body["images"][0], (512, 512))

    def test_batch_1_scheduler(self):
        """
        Test stable diffusion controlnet pipeline with a 1.5 finetune, batch size 1,
        euler scheduler, no a1111 alias
        """

        payload = {
            "checkpoint": self.checkpoint,
            "control_model": self.control_model,
            "scheduler": "EulerDiscreteScheduler",
            "parameters": {
                "prompt": "delicate lace sticker, vector art",
                "num_inference_steps": 20,
                "image": qr,
                "width": 512,
                "height": 512,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)

        # body.images[0] should be a base64 encoded image
        self.assertIsImage(response_body["images"][0], (512, 512))

    def test_batch_2_scheduler(self):
        """
        Test stable diffusion controlnet pipeline with a 1.5 finetune, batch size 2,
        euler scheduler, no a1111 alias
        """

        payload = {
            "checkpoint": self.checkpoint,
            "control_model": self.control_model,
            "scheduler": "EulerDiscreteScheduler",
            "parameters": {
                "prompt": "delicate lace sticker, vector art",
                "num_inference_steps": 20,
                "num_images_per_prompt": 2,
                "image": qr,
                "width": 512,
                "height": 512,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 2)

        # body.images[0] should be a base64 encoded image
        self.assertIsImage(response_body["images"][0], (512, 512))
        self.assertIsImage(response_body["images"][1], (512, 512))


class StableDiffusionControlNetImg2ImgPipelineTest(IGITest):
    checkpoint = "dreamshaper_8.safetensors"
    pipeline = "StableDiffusionControlNetImg2ImgPipeline"
    control_model = "qrCodeMonster_v20.safetensors"
    url = f"{IGITest.api_url}/generate/{pipeline}"

    def test_batch_1_a1111_safe(self):
        """
        Test stable diffusion controlnet img2img pipeline with a 1.5 finetune, batch size 1,
        a1111 scheduler alias, safety checker
        """

        payload = {
            "checkpoint": self.checkpoint,
            "control_model": self.control_model,
            "a1111_scheduler": "Euler a",
            "safety_checker": True,
            "parameters": {
                "prompt": "delicate lace sticker, vector art",
                "num_inference_steps": 20,
                "image": cat,
                "control_image": qr,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)

        # body.images[0] should be a base64 encoded image
        self.assertIsImage(response_body["images"][0], (512, 512))

    def test_batch_1_a1111_unsafe(self):
        """
        Test stable diffusion controlnet img2img pipeline with a 1.5 finetune, batch size 1,
        a1111 scheduler alias, no safety checker
        """

        payload = {
            "checkpoint": self.checkpoint,
            "control_model": self.control_model,
            "a1111_scheduler": "Euler a",
            "safety_checker": False,
            "parameters": {
                "prompt": "delicate lace sticker, vector art",
                "num_inference_steps": 20,
                "image": cat,
                "control_image": qr,
                "width": 512,
                "height": 512,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)

        # body.images[0] should be a base64 encoded image
        self.assertIsImage(response_body["images"][0], (512, 512))

    def test_batch_1_scheduler(self):
        """
        Test stable diffusion controlnet img2img pipeline with a 1.5 finetune, batch size 1,
        euler scheduler, no a1111 alias
        """

        payload = {
            "checkpoint": self.checkpoint,
            "control_model": self.control_model,
            "scheduler": "EulerDiscreteScheduler",
            "parameters": {
                "prompt": "delicate lace sticker, vector art",
                "num_inference_steps": 20,
                "image": cat,
                "control_image": qr,
                "width": 512,
                "height": 512,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)

        # body.images[0] should be a base64 encoded image
        self.assertIsImage(response_body["images"][0], (512, 512))

    def test_batch_2_scheduler(self):
        """
        Test stable diffusion controlnet img2img pipeline with a 1.5 finetune, batch size 2,
        euler scheduler, no a1111 alias
        """

        payload = {
            "checkpoint": self.checkpoint,
            "control_model": self.control_model,
            "scheduler": "EulerDiscreteScheduler",
            "parameters": {
                "prompt": "delicate lace sticker, vector art",
                "num_inference_steps": 20,
                "num_images_per_prompt": 2,
                "image": cat,
                "control_image": qr,
                "width": 512,
                "height": 512,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 2)

        # body.images[0] should be a base64 encoded image
        self.assertIsImage(response_body["images"][0], (512, 512))
        self.assertIsImage(response_body["images"][1], (512, 512))
