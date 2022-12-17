import discord
import os
import openai
import re

intents = discord.Intents.all()

bot = discord.Bot(intents=intents)

openai.api_key = os.getenv("OPENAI_API_KEY")
DC_API_KEY = os.getenv('DC_API_KEY')


@bot.event
async def on_ready():
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.playing, name="Human Simulator"))
    print(f"Connected as {bot.user}")

@bot.slash_command(name="ping", description="Pings the bot")
async def ping(ctx):
    embed = discord.Embed(color=0xf44336)
    embed.add_field(
        name="🏓 Pong! ", value=f"The bot latency is {round(bot.latency * 1000)}ms", inline=False)
    await ctx.respond(embed=embed)

async def generate_response(prompt):
    return openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

async def context_generator(ctx: discord.ApplicationContext, limit=5):
    messages = []
    async for message in ctx.channel.history(limit=limit):
        messages.append(f"{message.author.name}:{message.content}")

    context = "\n".join(messages[::-1])

    pattern = r"<(?:@|@&)(\d)*>"

    matches = re.finditer(pattern, context)

    for match in matches:
        context = context.replace(match.group(), userify(match.group()))

    return context + f"\n{bot.user.name}: "

def userify(user_id):
    clean_user_id = int(user_id.replace("<@", "").replace(">", "").replace("&", ""))
    if bot.get_user(clean_user_id):
        return f"@{bot.get_user(clean_user_id).name}"
    else:
        return f"@{clean_user_id}"

async def extract_response(conversation):
    return conversation.choices[0].text.split(f"\n{bot.user.name}: ")[-1]

async def message_handler(ctx: discord.ApplicationContext):
    context = await context_generator(ctx, limit=12)
    conversation = await generate_response(context)
    response = await extract_response(conversation)

    return response

@bot.event
async def on_message(ctx: discord.ApplicationContext):
    if ctx.author == bot.user:
        return

    mention = f'<@{bot.user.id}>'

    if mention in ctx.content:
        response = await message_handler(ctx)
        await ctx.channel.send(response, reference=ctx, mention_author=False)
        return

    if isinstance(ctx.channel, discord.channel.DMChannel):
        response = await message_handler(ctx)
        await ctx.channel.send(response)
        return

bot.run(DC_API_KEY)
