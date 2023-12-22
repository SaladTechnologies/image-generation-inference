from test_utils import IGITest
import base64


with open("test/cat.png", "rb") as f:
    cat = base64.b64encode(f.read()).decode()

with open("test/mask.jpg", "rb") as f:
    mask = base64.b64encode(f.read()).decode()


class StableDiffusionPipelineTest(IGITest):
    checkpoint = "stickerArt_sticker.safetensors"
    pipeline = "StableDiffusionPipeline"
    url = f"{IGITest.api_url}/generate/{pipeline}"

    def test_batch_1_a1111_safe(self):
        """
        Test stable diffusion pipeline with a 2.1 finetune, batch size 1,
        a1111 scheduler alias, safety checker
        """

        payload = {
            "checkpoint": self.checkpoint,
            "a1111_scheduler": "Euler a",
            "safety_checker": True,
            "parameters": {
                "prompt": "midcentury octopus sticker, by frank lloyd wright",
                "num_inference_steps": 20,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)
        self.assertIsImage(response_body["images"][0], (768, 768))

    def test_batch_1_a1111_unsafe(self):
        """
        Test stable diffusion pipeline with a 2.1 finetune, batch size 1,
        a1111 scheduler alias, no safety checker
        """

        payload = {
            "checkpoint": self.checkpoint,
            "a1111_scheduler": "Euler a",
            "safety_checker": False,
            "parameters": {
                "prompt": "midcentury octopus sticker, by frank lloyd wright",
                "num_inference_steps": 20,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)
        self.assertIsImage(response_body["images"][0], (768, 768))

    def test_batch_1_scheduler(self):
        """
        Test stable diffusion pipeline with a 2.1 finetune, batch size 1,
        euler scheduler, no a1111 alias
        """

        payload = {
            "checkpoint": self.checkpoint,
            "scheduler": "EulerDiscreteScheduler",
            "parameters": {
                "prompt": "midcentury octopus sticker, by frank lloyd wright",
                "num_inference_steps": 20,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)
        self.assertIsImage(response_body["images"][0], (768, 768))

    def test_batch_2_scheduler(self):
        """
        Test stable diffusion pipeline with a 2.1 finetune, batch size 2,
        euler scheduler, no a1111 alias
        """

        payload = {
            "checkpoint": self.checkpoint,
            "scheduler": "EulerDiscreteScheduler",
            "parameters": {
                "prompt": "midcentury octopus sticker, by frank lloyd wright",
                "num_inference_steps": 20,
                "num_images_per_prompt": 2,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 2
        self.assertEqual(len(response_body["images"]), 2)
        self.assertIsImage(response_body["images"][0], (768, 768))
        self.assertIsImage(response_body["images"][1], (768, 768))

    def test_diff_dimensions(self):
        """
        Test stable diffusion pipeline with a 2.1 finetune, batch size 1,
        euler scheduler, no a1111 alias, different dimensions
        """

        payload = {
            "checkpoint": self.checkpoint,
            "scheduler": "EulerDiscreteScheduler",
            "parameters": {
                "prompt": "midcentury octopus sticker, by frank lloyd wright",
                "num_inference_steps": 20,
                "height": 512,
                "width": 768,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)
        self.assertIsImage(response_body["images"][0], (768, 512))


class StableDiffusionImg2ImgPipelineTest(IGITest):
    checkpoint = "stickerArt_sticker.safetensors"
    pipeline = "StableDiffusionImg2ImgPipeline"
    url = f"{IGITest.api_url}/generate/{pipeline}"

    def test_batch_1_a1111_safe(self):
        """
        Test stable diffusion img2img pipeline with a 2.1 finetune, batch size 1,
        a1111 scheduler alias, safety checker
        """

        payload = {
            "checkpoint": self.checkpoint,
            "a1111_scheduler": "Euler a",
            "safety_checker": True,
            "parameters": {
                "prompt": "midcentury cat sticker, by frank lloyd wright",
                "num_inference_steps": 20,
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
        Test stable diffusion img2img pipeline with a 2.1 finetune, batch size 1,
        a1111 scheduler alias, no safety checker
        """

        payload = {
            "checkpoint": self.checkpoint,
            "a1111_scheduler": "Euler a",
            "safety_checker": False,
            "parameters": {
                "prompt": "midcentury cat sticker, by frank lloyd wright",
                "num_inference_steps": 20,
                "image": cat,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)
        self.assertIsImage(response_body["images"][0], (512, 512))

    def test_batch_1_scheduler(self):
        """
        Test stable diffusion img2img pipeline with a 2.1 finetune, batch size 1,
        euler scheduler, no a1111 alias
        """

        payload = {
            "checkpoint": self.checkpoint,
            "scheduler": "EulerDiscreteScheduler",
            "parameters": {
                "prompt": "midcentury cat sticker, by frank lloyd wright",
                "num_inference_steps": 20,
                "image": cat,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)
        self.assertIsImage(response_body["images"][0], (512, 512))

    def test_batch_2_scheduler(self):
        """
        Test stable diffusion img2img pipeline with a 2.1 finetune, batch size 2,
        euler scheduler, no a1111 alias
        """

        payload = {
            "checkpoint": self.checkpoint,
            "scheduler": "EulerDiscreteScheduler",
            "parameters": {
                "prompt": "midcentury cat sticker, by frank lloyd wright",
                "num_inference_steps": 20,
                "num_images_per_prompt": 2,
                "image": cat,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 2
        self.assertEqual(len(response_body["images"]), 2)
        self.assertIsImage(response_body["images"][0], (512, 512))
        self.assertIsImage(response_body["images"][1], (512, 512))


class StableDiffusionInpaintPipelineTest(IGITest):
    pipeline = "StableDiffusionInpaintPipeline"
    checkpoint = "stickerArt_sticker.safetensors"
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
        self.assertIsImage(response_body["images"][0], (768, 768))

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
        self.assertIsImage(response_body["images"][0], (768, 768))

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
        self.assertIsImage(response_body["images"][0], (768, 768))

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
        self.assertIsImage(response_body["images"][0], (768, 768))
        self.assertIsImage(response_body["images"][1], (768, 768))
