from urllib.parse import quote
import logging
import os
import discord
from discord.ui import Button, View
import io
import aiohttp
import uuid

logging.basicConfig(level=logging.INFO)

IMG_SERVER_URL = os.getenv("IMG_SERVER_URL")
UPSCALER = os.getenv("UPSCALER")


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


def make_url(prompt, negative_prompt, inference_steps, guideance_scale, height, width):

    base_url = IMG_SERVER_URL

    base_url += f"/generate?prompt={quote(prompt)}"

    base_url += f"&inference_steps={inference_steps}"

    if negative_prompt is not None:
        base_url += f"&negative_prompt={quote(negative_prompt)}"

    base_url += f"&guideance_scale={guideance_scale}"

    base_url += f"&height={height}"

    base_url += f"&width={width}"

    logging.debug(f"Query url is '{base_url}'")

    return base_url


def make_embed(prompt, width, height, inference_steps, guideance_scale, negative_prompt):

    embed = discord.Embed(title="DanielGPT - StableDiffusionAPI",
                          description=f"{prompt}")
    embed.add_field(name="Width", value=width, inline=True)
    embed.add_field(name="Height", value=height, inline=True)
    embed.add_field(name="\u200b", value="\u200b", inline=True)
    embed.add_field(name="Inference Steps", value=inference_steps, inline=True)
    embed.add_field(name="Guideance Scale", value=guideance_scale, inline=True)
    embed.add_field(name="\u200b", value="\u200b", inline=True)
    embed.add_field(name="Negative Prompt", value=negative_prompt, inline=True)

    return embed


async def upscale_button_callback(interaction):
    await interaction.response.defer()
    await interaction.edit_original_response(file=discord.File("./temporary_regen_image.gif"))

    prompt = interaction.message.embeds[0].description
    negative_prompt = interaction.message.embeds[0].fields[6].value
    inference_steps = int(interaction.message.embeds[0].fields[3].value)
    guideance_scale = float(interaction.message.embeds[0].fields[4].value)
    height = int(interaction.message.embeds[0].fields[1].value) * 4
    width = int(interaction.message.embeds[0].fields[0].value) * 4

    attachment_url = interaction.message.attachments[0].url
    f_name = f"{uuid.uuid4()}.png"
    async with aiohttp.ClientSession() as session:
        async with session.get(attachment_url) as resp:
            if resp.status == 200:
                image_buffer = io.BytesIO(await resp.read())
            else:
                raise Exception("Failed to fetch image.")

        async with session.post("http://10.10.20.10:9568/upscale?upscaler=espcn&scale=4", data={"image": image_buffer.read()}) as resp:
            if resp.status == 200:
                pic = discord.File(io.BytesIO(await resp.read()), filename=f_name)
                logging.info(f"imageine: Upscaled image.")
                await interaction.edit_original_response(file=pic, embed=make_embed(prompt, width, height, inference_steps, guideance_scale, negative_prompt))
            else:
                raise Exception("Failed to upscale image.")


async def regenerate_button_callback(interaction):
    await interaction.response.defer()
    await interaction.edit_original_response(file=discord.File("./temporary_regen_image.gif"))

    prompt = interaction.message.embeds[0].description
    negative_prompt = interaction.message.embeds[0].fields[6].value
    inference_steps = int(interaction.message.embeds[0].fields[3].value)
    guideance_scale = float(interaction.message.embeds[0].fields[4].value)
    height = int(interaction.message.embeds[0].fields[1].value)
    width = int(interaction.message.embeds[0].fields[0].value)

    width, height, inference_steps, guideance_scale = make_input_safe(
        width, height, inference_steps, guideance_scale)

    logging.info(f"imageine: Regenerating image image...")

    f_name = f"{uuid.uuid4()}.png"
    async with aiohttp.ClientSession() as session:
        logging.info(f"imageine: Fetching image...")
        async with session.get(make_url(prompt, negative_prompt, inference_steps, guideance_scale, height, width)) as response:
            if response.status == 200:
                logging.info(f"imageine: Image received.")
                pic = discord.File(io.BytesIO(await response.read()), filename=f_name)
                logging.info(f"imageine: Converted image to discord.File.")
                await interaction.edit_original_response(file=pic, embed=make_embed(prompt, width, height, inference_steps, guideance_scale, negative_prompt))
                logging.info(f"imageine: Image sent to client.")
            else:
                raise Exception(f"imageine: Error: {response.status}")


async def freeze_button_callback(interaction):
    await interaction.response.defer()

    upscale_button = Button(
        label="Upscale", style=discord.ButtonStyle.primary, disabled=True)
    regenerate_button = Button(
        label="Regenerate", style=discord.ButtonStyle.green, disabled=True)
    freeze_button = Button(
        label="Freeze", style=discord.ButtonStyle.red, disabled=True)

    view = View()
    view.add_item(upscale_button)
    view.add_item(regenerate_button)
    view.add_item(freeze_button)

    await interaction.edit_original_response(view=view)


def make_buttons():

    upscale_button = Button(label="Upscale", style=discord.ButtonStyle.primary)
    regenerate_button = Button(
        label="Regenerate", style=discord.ButtonStyle.green)
    freeze_button = Button(label="Freeze", style=discord.ButtonStyle.red)

    upscale_button.callback = upscale_button_callback
    regenerate_button.callback = regenerate_button_callback
    freeze_button.callback = freeze_button_callback

    view = View()
    view.add_item(upscale_button)
    view.add_item(regenerate_button)
    view.add_item(freeze_button)

    return view
