import asyncio
import io
import json
import logging
import os
import textwrap
import uuid

import aiohttp
import discord
import numpy as np
import openai
from DanielGPT import DanielGPT
from discord import option
from discord.ext import commands
from helpers import (ImageButtons, check_upscaler, make_embed, make_input_safe,
                     make_url, make_input_safe_im2, make_url_im2, make_embed_im2)

logging.basicConfig(level=logging.INFO)

intents = discord.Intents.all()
bot = discord.Bot(intents=intents)

chat_engine = DanielGPT()

CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")

DC_API_KEY = os.getenv("DC_API_KEY")
IMG_SERVER_URL = os.getenv("IMG_SERVER_URL")
memory_depth = int(os.getenv("MEMORY_DEPTH"))
cap_tokens = int(os.getenv("MAX_TOKENS"))
openai.api_key = os.getenv("OPENAI_API_KEY")

commands = {
    "ping": "Pings the bot",
    "upscale": "Upscales an image.",
    "imagine": "Generates an image from a prompt.",
    "transform": "Transforms an image into another image.",
    "write": "Writes a long text from a prompt.",
    "code": "Writes code from a prompt.",
    "help": "Shows this message."
}


@bot.slash_command(name="ping", description="Pings the bot")
async def ping(ctx):
    """This function is called when the '/ping' command is used. It returns
    an embed message with the bot latency."""

    embed = discord.Embed(color=0xF44336)
    embed.add_field(
        name="🏓 Pong! ",
        value=f"The bot latency is {round(bot.latency * 1000)}ms",
        inline=False,
    )
    await ctx.respond(embed=embed)
    logging.info(f"Tested latency is {round(bot.latency * 1000)}ms")


@bot.slash_command(name="upscale", description="Upscales an image")
@option(
    "attachment",
    discord.Attachment,
    description="An image to upscale",
    # The default value will be None if the user doesn't provide a file.
    required=True,
)
async def upscale(ctx, attachment: discord.Attachment, upscaler: str = "espcn", scale: int = 2):
    await ctx.defer()

    if check_upscaler(upscaler, scale):
        if attachment.content_type == "image/png" or attachment.content_type == "image/jpeg":

            logging.info(
                f"upscale: Upscaling image with {upscaler} and scale {scale}.")

            attachment_url = attachment.url
            url = f"{IMG_SERVER_URL}/upscale?upscaler={upscaler}&scale={scale}"
            f_name = f"{uuid.uuid4()}.png"

            async with aiohttp.ClientSession() as session:
                async with session.get(attachment_url) as resp:
                    if resp.status == 200:
                        image_buffer = io.BytesIO(await resp.read())
                    else:
                        raise Exception("Failed to fetch image from Discord.")

                async with session.post(url, data={"image": image_buffer.read()}) as resp:
                    if resp.status == 200:
                        # image = io.BytesIO(await resp.read())
                        headers = {
                            'Authorization': f'Client-ID {CLIENT_ID}'
                        }

                        r = await resp.read()

                        payload = {
                            'image': r,
                        }

                        async with aiohttp.request('post', f"https://api.imgur.com/3/image", headers=headers, data=payload) as resp:
                            resp.raise_for_status
                            imgur_response = json.loads(await resp.text())
                            await ctx.respond(content=imgur_response['data']['link'])
                    else:
                        raise Exception("Failed to upscale image.")
        else:
            logging.error("upscale: Invalid image type")
            raise Exception("Invalid image type")
    else:
        logging.error("upscale: Invalid upscaler or scale")
        raise Exception("Invalid upscaler or scale")


@bot.slash_command(name="imagine", description="Imagines a prompt")
async def imagine(ctx, prompt: str,
                  negative_prompt: str = None,
                  inference_steps: int = 50,
                  guideance_scale: float = 7.5,
                  height: int = 512,
                  width: int = 512,
                  seed: int = None):

    await ctx.defer()

    if not seed:
        seed = np.random.randint(10000000000, 1000000000000)

    width, height, inference_steps, guideance_scale = make_input_safe(width,
                                                                      height,
                                                                      inference_steps,
                                                                      guideance_scale)

    embed = make_embed(prompt, width, height, inference_steps,
                       guideance_scale, negative_prompt, seed)

    url = make_url(prompt, negative_prompt, inference_steps,
                   guideance_scale, height, width, seed)

    message = await ctx.respond(embed=embed)
    logging.info(f'imageine: Generating image with prompt: "{prompt}".')

    f_name = f"{uuid.uuid4()}.png"
    async with aiohttp.ClientSession() as session:
        logging.info(f"imageine: Fetching image...")
        async with session.get(url) as response:
            if response.status == 200:
                logging.info(f"imageine: Image received.")
                pic = discord.File(io.BytesIO(await response.read()), filename=f_name)
                logging.info(f"imageine: Converted image to discord.File.")
                await message.edit(view=ImageButtons(), embed=embed, file=pic)
                logging.info(f"imageine: Image sent to client.")
            else:
                raise Exception(f"imageine: Error: {response.status}")


@bot.slash_command(name="transform", description="Transforms an image based on a prompt")
@option(
    "attachment",
    discord.Attachment,
    description="An image to transform",
    # The default value will be None if the user doesn't provide a file.
    required=True,
)
async def transform(ctx, prompt: str, attachment: discord.Attachment, strength: float = 0.8, num_inference_steps: int = 50, guidance_scale: float = 7.5, negative_prompt: str = None):
    await ctx.defer()

    strength, num_inference_steps, guidance_scale = make_input_safe_im2(strength,
                                                                        num_inference_steps,
                                                                        guidance_scale)
    transform_url = make_url_im2(
        prompt, negative_prompt, strength, num_inference_steps, guidance_scale)
    attachment_url = attachment.url

    f_name = f"{uuid.uuid4()}.png"

    async with aiohttp.ClientSession() as session:
        async with session.get(attachment_url) as resp:
            if resp.status == 200:
                image_buffer = io.BytesIO(await resp.read())
                async with session.post(transform_url, data={"image": image_buffer.read()}) as resp:
                    if resp.status == 200:
                        pic = discord.File(io.BytesIO(await resp.read()), filename=f_name)
                        logging.info(f"imagine: Upscaled image.")
                        await ctx.respond(file=pic, embed=make_embed_im2(prompt, strength, num_inference_steps, guidance_scale, negative_prompt))
                    else:
                        raise Exception("Failed to transform image.")
            else:
                raise Exception("Failed to fetch image.")


@bot.slash_command(name="write", description="Write a prompt")
async def write(ctx, prompt: str, cap_tokens: int = 2000):
    await ctx.defer()

    prompt = f"Please write: {prompt}"

    logging.info(f"write:prompt={prompt}")
    response = await chat_engine.contextless_runner(prompt=prompt, cap_tokens=cap_tokens)
    if len(response) < 2000:
        await ctx.respond(response)
    else:
        chunks = textwrap.wrap(
            response, width=1900, break_long_words=False, replace_whitespace=False)

        logging.info(
            f"write:error: Message too long split into ({len(chunks)}) chunks.")
        await ctx.respond(chunks[0])
        for chunk in chunks[1:]:
            await asyncio.sleep(0.1)
            await ctx.send(chunk)


@bot.slash_command(name="code", description="Code a prompt")
async def code(ctx, prompt: str, cap_tokens: int = 2000, language: str = "python"):
    await ctx.defer()

    prompt = f"language: {language}\n{prompt}"

    logging.info(f"code:prompt={prompt}")
    response = await chat_engine.contextless_runner(prompt=prompt, cap_tokens=cap_tokens)
    if len(response) < 2000:
        response = f"```{response}```"
        await ctx.respond(response)
    else:
        chunks = textwrap.wrap(
            response, width=1900, break_long_words=False, replace_whitespace=False)

        logging.info(
            f"code:error: Message too long split into ({len(chunks)}) chunks.")
        await ctx.respond(chunks[0])
        for chunk in chunks[1:]:
            chunk = f"```{chunk}```"
            await asyncio.sleep(0.1)
            await ctx.send(chunk)


@bot.event
async def on_ready():
    bot.add_view(ImageButtons())
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.playing, name="Human Simulator"))
    logging.info("Conntected to Discord servers as DanielGPT#0832")


@bot.event
async def on_message(ctx: discord.ApplicationContext):
    """This function is called when a new message is sent in a channel the bot can see."""

    if ctx.author == bot.user:
        return

    mention = f"<@{bot.user.id}>"

    try:
        if mention in ctx.content or isinstance(ctx.channel, discord.channel.DMChannel):
            response = await chat_engine.run(ctx, bot, 12, 950)
            logging.info(f"message='{response}'")
            await ctx.channel.send(response)

    except Exception as e:
        logging.error(e)
        await ctx.channel.send(f"Error: {e}")


@bot.event
async def on_application_command_error(ctx, error):
    if isinstance(error, commands.errors.CommandOnCooldown):
        await ctx.respond(
            f"Please wait **{error.retry_after:.0f} seconds** before using this command again."
        )
    else:
        await ctx.respond(f"Error: {error}")


@bot.command(name="help", description="Displays a list of all commands and their descriptions")
async def help_command(ctx):
    embed = discord.Embed(color=0xF44336)
    embed.title = "Commands:"

    for command, description in commands.items():
        embed.add_field(
            name=f"/{command}",
            value=description,
            inline=False,
        )
    await ctx.respond(embed=embed)


bot.run(DC_API_KEY)
