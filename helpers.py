import re
import discord
import openai
import psutil


async def generate_response(prompt):
    """This function takes in a prompt, and uses the OpenAI API to generate a response."""
    return openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )


async def context_generator(ctx: discord.ApplicationContext, bot, limit=5):
    """This function takes in a discord context and a bot, and creates a context string of up to 'limit' previous messages in the channel."""
    messages = []
    print(limit)
    async for message in ctx.channel.history(limit=limit):
        messages.append(f"{message.author.name}:{message.content}")

    context = "\n".join(messages[::-1])

    pattern = r"<(?:@|@&)(\d)*>"

    matches = re.finditer(pattern, context)

    for match in matches:
        context = context.replace(match.group(), userify(match.group(), bot))

    return context + f"\n{bot.user.name}: "


def userify(user_id, bot):
    """This function takes in a user_id and a bot, and returns a clean veresion of the user_id."""
    clean_user_id = int(user_id.replace("<@", "").replace(">", "").replace("&", ""))
    if bot.get_user(clean_user_id):
        return f"@{bot.get_user(clean_user_id).name}"
    else:
        return f"@{clean_user_id}"


async def extract_response(conversation, bot):
    """This function takes in a conversation and a bot, and extracts the response from the conversation generated by the OpenAI API."""
    return conversation.choices[0].text.split(f"\n{bot.user.name}: ")[-1]


async def message_handler(ctx: discord.ApplicationContext, bot, memory_depth):
    """This function takes in a discord context, a bot, and a memory_depth limit,
    and generates a response using the OpenAI API. It creates a context string of
    up to the specified number of previous messages in the channel, and then generates
    a response based on this context. It then returns the response generated by the OpenAI API."""

    context = await context_generator(ctx, bot, limit=memory_depth)
    conversation = await generate_response(context)
    response = await extract_response(conversation, bot)

    return response


async def image_generator(prompt, bot):
    """This function takes in a prompt and a bot, and generates an image using the OpenAI API."""
    response = openai.Image.create(prompt=prompt, n=1, size="1024x1024")
    image_url = response["data"][0]["url"]
    return image_url


async def get_machine_stats(ctx):
    """This function takes in a discord context and returns the current CPU percentage, memory usage, available memory, and load average for a machine."""
    cpu_percent = psutil.cpu_percent()
    memory_usage = round(psutil.virtual_memory().percent, 2)
    available_memory = round(
        psutil.virtual_memory().available * 100 / psutil.virtual_memory().total, 2
    )
    load_avg = "% | ".join(
        [str(round(x / psutil.cpu_count() * 100, 1)) for x in psutil.getloadavg()]
    )
    return [cpu_percent, memory_usage, available_memory, load_avg]
