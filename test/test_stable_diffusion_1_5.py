from test_utils import IGITest
import base64

with open("test/cat.png", "rb") as f:
    cat = base64.b64encode(f.read()).decode()


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
