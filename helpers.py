import io
import logging
import os
import uuid
from urllib.parse import quote

import aiohttp
import discord
import numpy as np

logging.basicConfig(level=logging.INFO)

IMG_SERVER_URL = os.getenv("IMG_SERVER_URL")
UPSCALER = os.getenv("UPSCALER")


possible_upscalers = {
    "espcn": [2, 4],
    "edsr": [2, 4],
    "lapsrn": [2, 4, 8]
}

def check_upscaler(upscaler, scale):
    if upscaler in possible_upscalers.keys():
        if scale in possible_upscalers[upscaler]:
            return True
        else:
            return False

def make_input_safe(width, height, inference_steps, guideance_scale):

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

    return width, height, inference_steps, guideance_scale

def make_url(prompt, negative_prompt, inference_steps, guideance_scale, height, width, seed):

    base_url = IMG_SERVER_URL
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

def make_embed(prompt, width, height, inference_steps, guideance_scale, negative_prompt, seed):

    embed = discord.Embed(title="DanielGPT - StableDiffusionAPI",
                          description=f"{prompt}")
    embed.add_field(name="Width", value=width, inline=True)
    embed.add_field(name="Height", value=height, inline=True)
    embed.add_field(name="\u200b", value="\u200b", inline=True)
    embed.add_field(name="Inference Steps", value=inference_steps, inline=True)
    embed.add_field(name="Guideance Scale", value=guideance_scale, inline=True)
    embed.add_field(name="\u200b", value="\u200b", inline=True)
    embed.add_field(name="Negative Prompt", value=negative_prompt, inline=True)
    embed.add_field(name="Seed", value=seed, inline=False)

    return embed

class ImageButtons(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None) # timeout of the view must be set to None

    @discord.ui.button(label="Upscale", custom_id="upscale", style=discord.ButtonStyle.primary) # the button has a custom_id set
    async def upscale_button_callback(self, button, interaction):
        await interaction.response.defer()
        await interaction.edit_original_response(file=discord.File("./temporary_regen_image.gif"))

        prompt = interaction.message.embeds[0].description
        negative_prompt = interaction.message.embeds[0].fields[6].value
        inference_steps = int(interaction.message.embeds[0].fields[3].value)
        guideance_scale = float(interaction.message.embeds[0].fields[4].value)
        height = int(interaction.message.embeds[0].fields[1].value) * 4
        width = int(interaction.message.embeds[0].fields[0].value) * 4
        seed = np.random.randint(10000000000, 1000000000000)

        attachment_url = interaction.message.attachments[0].url
        f_name = f"{uuid.uuid4()}.png"
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment_url) as resp:
                if resp.status == 200:
                    image_buffer = io.BytesIO(await resp.read())
                else:
                    raise Exception("Failed to fetch image.")

            async with session.post(f"{IMG_SERVER_URL}/upscale?upscaler=espcn&scale=4", data={"image": image_buffer.read()}) as resp:
                if resp.status == 200:
                    pic = discord.File(io.BytesIO(await resp.read()), filename=f_name)
                    logging.info(f"imagine: Upscaled image.")
                    await interaction.edit_original_response(file=pic, embed=make_embed(prompt, width, height, inference_steps, guideance_scale, negative_prompt, seed))
                else:
                    raise Exception("Failed to upscale image.")

    @discord.ui.button(label="Regenerate", custom_id="regenerate", style=discord.ButtonStyle.green) # the button has a custom_id set
    async def regenerate_button_callback(self, button, interaction):
        await interaction.response.defer()
        await interaction.edit_original_response(file=discord.File("./temporary_regen_image.gif"))

        prompt = interaction.message.embeds[0].description
        width = int(interaction.message.embeds[0].fields[0].value)
        height = int(interaction.message.embeds[0].fields[1].value)
        inference_steps = int(interaction.message.embeds[0].fields[3].value)
        guideance_scale = float(interaction.message.embeds[0].fields[4].value)
        negative_prompt = interaction.message.embeds[0].fields[6].value
        seed = int(interaction.message.embeds[0].fields[7].value)

        width, height, inference_steps, guideance_scale = make_input_safe(
            width, height, inference_steps, guideance_scale)

        logging.info(f"imagine: Regenerating image...")

        f_name = f"{uuid.uuid4()}.png"
        async with aiohttp.ClientSession() as session:
            logging.info(f"imagine: Fetching image...")
            async with session.get(make_url(prompt, negative_prompt, inference_steps, guideance_scale, height, width, seed)) as response:
                if response.status == 200:
                    logging.info(f"imagine: Image received.")
                    pic = discord.File(io.BytesIO(await response.read()), filename=f_name)
                    logging.info(f"imagine: Converted image to discord.File.")
                    await interaction.edit_original_response(file=pic, embed=make_embed(prompt, width, height, inference_steps, guideance_scale, negative_prompt, seed))
                    logging.info(f"imagine: Image sent to client.")
                else:
                    raise Exception(f"imagine: Error: {response.status}")


    @discord.ui.button(label="Freeze", custom_id="freeze", style=discord.ButtonStyle.red) # the button has a custom_id set
    async def freeze_button_callback(self, button, interaction):
        await interaction.response.defer()
        for button in self.children:
            button.disabled = True
        await interaction.edit_original_response(view=self)
