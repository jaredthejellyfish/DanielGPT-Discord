import aiohttp
import asyncio
import numpy as np
import logging
from urllib.parse import quote
import uuid
import io


class JellyHostML():
    def __init__(self, base_url, ):
        self.base_url = base_url
        self.possible_upscalers = {
            "espcn": [2, 4],
            "edsr": [2, 4],
            "lapsrn": [2, 4, 8]
        }

    async def _get(self, url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return io.BytesIO(await response.read())

    async def _post(self, url, data):
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                if response.status == 200:
                    return io.BytesIO(await response.read())

    async def generate(self, prompt: str, inference_steps: int = 50, guideance_scale: float = 7.5, negative_prompt: str = None, height: int = 568, width: int = 568, seed: int = None):
        prompt, inference_steps, guideance_scale, negative_prompt, height, width, seed = self._make_input_safe_generate(
            prompt, inference_steps, guideance_scale, negative_prompt, height, width, seed)

        url = self._make_url_generate(
            prompt, inference_steps, guideance_scale, negative_prompt, height, width, seed)

        return await self._get(url), self._f_name()

    async def upscale(self, image, upscaler="espcn", scale=4):
        if self._check_upscaler(upscaler, scale):
            url = f"{self.base_url}/upscale?upscaler={upscaler}&scale={scale}"

            return await self._post(url, data={"image": image.read()}), self._f_name()

    async def img2img(self, image, prompt: str, strength: float = 0.8, num_inference_steps: int = 50, guidance_scale: float = 7.5, negative_prompt: str = None):
        prompt, strength, num_inference_steps, guidance_scale, negative_prompt = self._make_input_safe_img2img(prompt, strength, num_inference_steps, guidance_scale, negative_prompt)
        url = self._make_url_img2img(prompt, strength, num_inference_steps, guidance_scale, negative_prompt)

        return await self._post(url, data={"image": image.read()}), self._f_name()

    def _make_input_safe_generate(self, prompt, inference_steps, guideance_scale, negative_prompt, height, width, seed):
        if inference_steps > 100:
            logging.warn("InputRectifier: Inference steps maxed at 100")
            inference_steps = 100
        elif inference_steps < 1:
            logging.warn("InputRectifier: Inference steps minned at 1")
            inference_steps = 1

        if guideance_scale > 25:
            logging.warn("InputRectifier: Guideance scale maxed at 10.0")
            guideance_scale = 25.0
        elif guideance_scale < 1:
            logging.warn("InputRectifier: Guideance scale minned at 1.0")
            guideance_scale = 1.0

        if height > 568:
            logging.warn("InputRectifier: Height maxed at 568")
            height = 568
        elif height < 1:
            logging.warn("InputRectifier: Height minned at 1")
            height = 1

        if width > 568:
            logging.warn("InputRectifier: Width maxed at 568")
            width = 568
        elif width < 1:
            logging.warn("InputRectifier: Width minned at 1")
            width = 1

        if seed == None:
            seed = np.random.randint(10000000000, 1000000000000)

        if negative_prompt is not None:
            negative_prompt = quote(negative_prompt)

        prompt = quote(prompt)

        return prompt, inference_steps, guideance_scale, negative_prompt, height, width, seed

    def _make_url_generate(self, prompt, inference_steps, guideance_scale, negative_prompt, height, width, seed):
        base_url = self.base_url
        base_url += f"/generate?prompt={quote(prompt)}"
        base_url += f"&inference_steps={inference_steps}"
        base_url += f"&guideance_scale={guideance_scale}"
        base_url += f"&height={height}"
        base_url += f"&width={width}"

        if negative_prompt is not None:
            base_url += f"&negative_prompt={quote(negative_prompt)}"
        if seed is not None:
            base_url += f"&seed={seed}"

        logging.debug(f"Query url is '{base_url}'")

        return base_url

    def _check_upscaler(self, upscaler, scale):
        if upscaler in self.possible_upscalers.keys():
            if scale in self.possible_upscalers[upscaler]:
                return True
            else:
                return False

    def _make_input_safe_img2img(self, prompt, strength, num_inference_steps, guidance_scale, negative_prompt):
        if strength > 1:
            logging.warn("InputRectifier: Strength maxed at 1.0")
            strength = 1.0
        elif strength < 0:
            logging.warn("InputRectifier: Strength minned at 0.0")
            strength = 0.0

        if num_inference_steps > 100:
            logging.warn("InputRectifier: Inference steps maxed at 100")
            num_inference_steps = 100
        elif num_inference_steps < 1:
            logging.warn("InputRectifier: Inference steps minned at 1")
            num_inference_steps = 1

        if guidance_scale > 25:
            logging.warn("InputRectifier: Guideance scale maxed at 10.0")
            guidance_scale = 25.0
        elif guidance_scale < 1:
            logging.warn("InputRectifier: Guideance scale minned at 1.0")
            guidance_scale = 1.0

        if negative_prompt is not None:
            negative_prompt = quote(negative_prompt)

        prompt = quote(prompt)

        return prompt, strength, num_inference_steps, guidance_scale, negative_prompt

    def _make_url_img2img(self, prompt, strength, num_inference_steps, guidance_scale, negative_prompt):
        base_url = self.base_url
        base_url += f"/img2img?prompt={quote(prompt)}"
        base_url += f"&strength={strength}"
        base_url += f"&num_inference_steps={num_inference_steps}"
        base_url += f"&guideance_scale={guidance_scale}"

        if negative_prompt is not None:
            base_url += f"&negative_prompt={quote(negative_prompt)}"

        return base_url

    def _f_name(self):
        return f"{uuid.uuid4()}.png"
