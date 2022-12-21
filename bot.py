import discord
import os
import openai

intents = discord.Intents.all()

bot = discord.Bot(intents=intents)

from helpers import *

openai.api_key = os.getenv("OPENAI_API_KEY")
DC_API_KEY = os.getenv("DC_API_KEY")
memory_depth = int(os.getenv("MEMORY_DEPTH"))


@bot.event
async def on_ready():
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.playing, name="Human Simulator"
        )
    )
    print(f"Connected as {bot.user}")


@bot.slash_command(name="ping", description="Pings the bot")
async def ping(ctx):
    print(memory_depth)
    embed = discord.Embed(color=0xF44336)
    embed.add_field(
        name="🏓 Pong! ",
        value=f"The bot latency is {round(bot.latency * 1000)}ms",
        inline=False,
    )
    await ctx.respond(embed=embed)


@bot.slash_command(name="help", description="Shows the help menu")
async def help(ctx):
    embed = discord.Embed(color=0xF44336)
    embed.add_field(
        name="🤖 Bot Commands",
        value="`/ping` - Pings the bot\n`/memd` - Sets the memory depth of the bot\n`/info` - Gives info about the bot's execution environment\n`/imagine` - Imagines an image from a prompt\n`/help` - Shows the help menu\n",
        inline=False,
    )
    await ctx.respond(embed=embed)


@bot.slash_command(name="memd", description="Sets the memory depth of the bot")
async def memd(ctx, depth):
    global memory_depth

    os.environ["MEMORY_DEPTH"] = str(depth)
    memory_depth = int(depth)

    embed = discord.Embed(
        title="Memory depth is now:", description=f"{memory_depth}", color=0xF44336
    )
    await ctx.respond(embed=embed)


@bot.slash_command(
    name="info", description="Gives info about the bot's execution environment"
)
async def memd(ctx):
    await ctx.defer()
    cpu_percent, memory_usage, available_memory, load_avg = await get_machine_stats(ctx)
    embed = discord.Embed(
        title="DanielGPT", description="I'm still alive, thanks for checking :)", color=0xF44336
    )
    embed.add_field(name="Load (avg)", value=f"{load_avg}%", inline=False)
    embed.add_field(name="Ping", value=f"{round(bot.latency * 1000)}ms", inline=True)
    embed.add_field(name="Memdepth", value=f"{memory_depth} messages", inline=True)
    embed.add_field(name=chr(173), value=chr(173))
    embed.add_field(name="CPU %", value=f"{cpu_percent}%", inline=True)
    embed.add_field(name="Memory %", value=f"{memory_usage}%", inline=True)
    embed.add_field(name="Memory (avail)", value=f"{available_memory}%", inline=True)

    await ctx.respond(embed=embed)


@bot.slash_command(name="imagine", description="Imagines an image from a prompt")
async def imagine(ctx, prompt: str):
    await ctx.defer()
    try:
        image_url = await image_generator(prompt, bot)
        embed = discord.Embed()
        embed.title = prompt.capitalize()
        embed.set_image(url=image_url)
        embed.set_footer(text="Powered by OpenAI.")
        await ctx.respond(embed=embed)
    except Exception as e:
        await ctx.respond(f"Error: {e}".replace(" our ", " my "))


@bot.event
async def on_message(ctx: discord.ApplicationContext):
    if ctx.author == bot.user:
        return

    mention = f"<@{bot.user.id}>"

    try:
        if mention in ctx.content or isinstance(ctx.channel, discord.channel.DMChannel):
            response = await message_handler(ctx, bot, memory_depth)
            await ctx.channel.send(response)
            return
    except Exception as e:
        await ctx.channel.send(e)


bot.run(DC_API_KEY)
