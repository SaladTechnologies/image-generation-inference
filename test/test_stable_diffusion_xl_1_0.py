from test_utils import IGITest
import base64


class StableDiffusionXLPipelineTest(IGITest):
    pipeline = "StableDiffusionXLPipeline"
    checkpoint = "rundiffusionXL_beta.safetensors"
    url = f"{IGITest.api_url}/generate/{pipeline}"

    def test_batch_1_a1111(self):
        """
        Test Stable Diffusion XL Pipeline with a SDXL1.0 finetune, batch size 1,
        a1111 scheduler alias, no refiner
        """

        payload = {
            "checkpoint": self.checkpoint,
            "a1111_scheduler": "DPM++ SDE Karras",
            "parameters": {
                "prompt": "a housecat, believing itself to be a tiger in the jungle",
                "num_inference_steps": 35,
                "guidance_scale": 7.0,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)
        self.assertIsImage(response_body["images"][0], (1024, 1024))

    def test_batch_1_scheduler(self):
        """
        Test Stable Diffusion XL Pipeline with a SDXL1.0 finetune, batch size 1,
        custom scheduler, no refiner
        """

        payload = {
            "checkpoint": self.checkpoint,
            "scheduler": "DPMSolverSinglestepScheduler",
            "parameters": {
                "prompt": "a housecat, believing itself to be a tiger in the jungle",
                "num_inference_steps": 35,
                "guidance_scale": 7.0,
            },
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)
        self.assertIsImage(response_body["images"][0], (1024, 1024))

    def test_batch_1_refiner(self):
        """
        Test Stable Diffusion XL Pipeline with a SDXL1.0 finetune, batch size 1,
        a1111 scheduler alias, with refiner
        """

        payload = {
            "checkpoint": self.checkpoint,
            "a1111_scheduler": "DPM++ SDE Karras",
            "refiner_model": "stabilityai/stable-diffusion-xl-refiner-1.0",
            "parameters": {
                "prompt": "a housecat, believing itself to be a tiger in the jungle",
                "num_inference_steps": 35,
                "guidance_scale": 7.0,
                "denoising_end": 0.7,
            },
            "refiner_parameters": {"denoising_start": 0.7},
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)
        self.assertIsImage(response_body["images"][0], (1024, 1024))

    def test_batch_2_refiner(self):
        """
        Test Stable Diffusion XL Pipeline with a SDXL1.0 finetune, batch size 2,
        a1111 scheduler alias, with refiner
        """

        payload = {
            "checkpoint": self.checkpoint,
            "a1111_scheduler": "DPM++ SDE Karras",
            "refiner_model": "stabilityai/stable-diffusion-xl-refiner-1.0",
            "parameters": {
                "prompt": "a housecat, believing itself to be a tiger in the jungle",
                "num_inference_steps": 35,
                "guidance_scale": 7.0,
                "denoising_end": 0.7,
                "num_images_per_prompt": 2,
            },
            "refiner_parameters": {"denoising_start": 0.7},
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 2
        self.assertEqual(len(response_body["images"]), 2)
        self.assertIsImage(response_body["images"][0], (1024, 1024))
        self.assertIsImage(response_body["images"][1], (1024, 1024))

    def test_batch_1_refiner_diff_dimensions(self):
        """
        Test Stable Diffusion XL Pipeline with a SDXL1.0 finetune, batch size 1,
        a1111 scheduler alias, with refiner, with different dimensions
        """

        payload = {
            "checkpoint": self.checkpoint,
            "a1111_scheduler": "DPM++ SDE Karras",
            "refiner_model": "stabilityai/stable-diffusion-xl-refiner-1.0",
            "parameters": {
                "prompt": "a housecat, believing itself to be a tiger in the jungle",
                "num_inference_steps": 35,
                "guidance_scale": 7.0,
                "denoising_end": 0.7,
                "width": 512,
                "height": 768,
            },
            "refiner_parameters": {"denoising_start": 0.7},
            "return_images": True,
        }
        response_body = self.assertPostSuccessful(payload)

        # body.images should have length 1
        self.assertEqual(len(response_body["images"]), 1)
        self.assertIsImage(response_body["images"][0], (512, 768))
