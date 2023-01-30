import io
import os
import json
import logging
import discord
import aiohttp

from jellyhost_ml import JellyHostML
from jellyhost_ml.danielgpt import make_embed, make_embed_im2

logging.basicConfig(level=logging.INFO)

IMG_SERVER_URL = os.getenv("IMG_SERVER_URL")
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")
UPSCALER = os.getenv("UPSCALER")
jhml = JellyHostML(IMG_SERVER_URL)


class ImageButtons(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)  # timeout of the view must be set to None

    # the button has a custom_id set
    @discord.ui.button(label="Upscale", custom_id="upscale", style=discord.ButtonStyle.primary)
    async def upscale_button_callback(self, button, interaction):
        await interaction.response.defer()
        await interaction.edit_original_response(file=discord.File("./temporary_regen_image.gif"))

        prompt = interaction.message.embeds[0].description
        negative_prompt = interaction.message.embeds[0].fields[6].value
        inference_steps = int(interaction.message.embeds[0].fields[3].value)
        guideance_scale = float(interaction.message.embeds[0].fields[4].value)
        height = int(interaction.message.embeds[0].fields[1].value) * 4
        width = int(interaction.message.embeds[0].fields[0].value) * 4

        try:
            seed = int(interaction.message.embeds[0].fields[7].value)
        except ValueError:
            seed = None

        attachment_url = interaction.message.attachments[0].url

        async with aiohttp.ClientSession() as session:
            async with session.get(attachment_url) as resp:
                if resp.status == 200:
                    image_bytesio = io.BytesIO(await resp.read())
                else:
                    raise Exception("Failed to fetch image from Discord.")

                image, f_name = await jhml.upscale(image_bytesio, 'espcn', 4)
                pic = discord.File(image, filename=f_name)
                await interaction.edit_original_response(file=pic, embed=make_embed(prompt, width, height, inference_steps, guideance_scale, negative_prompt, seed))

    # the button has a custom_id set
    @discord.ui.button(label="Regenerate", custom_id="regenerate", style=discord.ButtonStyle.green)
    async def regenerate_button_callback(self, button, interaction):
        await interaction.response.defer()
        await interaction.edit_original_response(file=discord.File("./temporary_regen_image.gif"))

        prompt = interaction.message.embeds[0].description
        width = int(interaction.message.embeds[0].fields[0].value)
        height = int(interaction.message.embeds[0].fields[1].value)
        inference_steps = int(interaction.message.embeds[0].fields[3].value)
        guideance_scale = float(interaction.message.embeds[0].fields[4].value)
        negative_prompt = interaction.message.embeds[0].fields[6].value
        seed = None

        image, f_name = await jhml.generate(prompt, inference_steps, guideance_scale, negative_prompt, height, width, seed)
        pic = discord.File(image, filename=f_name)
        await interaction.edit_original_response(file=pic, embed=make_embed(prompt, width, height, inference_steps, guideance_scale, negative_prompt, seed))

    # the button has a custom_id set
    @discord.ui.button(label="Freeze", custom_id="freeze", style=discord.ButtonStyle.red)
    async def freeze_button_callback(self, button, interaction):
        await interaction.response.defer()
        for button in self.children:
            button.disabled = True
        await interaction.edit_original_response(view=self)
