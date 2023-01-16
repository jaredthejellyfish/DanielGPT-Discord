import os
import uuid
import openai
import asyncio
import logging
import discord
from discord.ext import commands
from discord.ui import Button, View
import aiohttp
import textwrap
import multiprocessing as mp
from DanielGPT import DanielGPT
import io

from helpers import make_url, make_embed, make_buttons, make_input_safe

logging.basicConfig(level=logging.INFO)

intents = discord.Intents.all()
bot = discord.Bot(intents=intents)

chat_engine = DanielGPT()


DC_API_KEY = os.getenv("DC_API_KEY")
memory_depth = int(os.getenv("MEMORY_DEPTH"))
cap_tokens = int(os.getenv("MAX_TOKENS"))
openai.api_key = os.getenv("OPENAI_API_KEY")


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


@bot.slash_command(name="imagine", description="imagines a prompt")
async def imagine(ctx, prompt: str,
                  negative_prompt: str = None,
                  inference_steps: int = 50,
                  guideance_scale: float = 7.5,
                  height: int = 512,
                  width: int = 512):

    await ctx.defer()

    width, height, inference_steps, guideance_scale = make_input_safe(width,
                                                                      height,
                                                                      inference_steps,
                                                                      guideance_scale)

    embed = make_embed(prompt, width, height, inference_steps,
                       guideance_scale, negative_prompt)

    url = make_url(prompt, negative_prompt, inference_steps,
                   guideance_scale, height, width)

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
                await message.edit(view=make_buttons(), embed=embed, file=pic)
                logging.info(f"imageine: Image sent to client.")
            else:
                raise Exception(f"imageine: Error: {response.status}")


@bot.slash_command(name="write", description="Write a prompt")
async def write(ctx, prompt: str, cap_tokens: int = 2000):
    await ctx.defer()

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


@bot.event
async def on_ready():
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

bot.run(DC_API_KEY)
