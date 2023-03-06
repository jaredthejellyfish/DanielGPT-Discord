import re
from helpers import num_tokens_from_string
import discord
import openai


class OldDanielGPT:
    async def run(self, ctx, bot, memory_depth, cap_tokens):
        """
        This method takes in a discord context, a bot, a memory depth, and a token cap.
        It creates a context string of previous messages in the channel, generates a response
        using the OpenAI API, and returns the response.
        """
        self.memory_depth = memory_depth
        self.cap_tokens = cap_tokens
        self.bot = bot
        context = await self.context_generator(ctx)
        conversation = await self.generate_response(context)
        response = conversation.choices[0].text.split(
            f"\n{bot.user.name}: ")[-1]

        return response

    async def contextless_runner(self, prompt, cap_tokens):
        """
        This method takes in a prompt and a token cap, and uses the OpenAI API to generate a response.
        """
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=cap_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].text

    async def code_completion(self, prompt, cap_tokens):
        """
        This method takes in a prompt and a token cap, and uses the OpenAI API to generate a response.
        """

        response = openai.Completion.create(
            model="code-davinci-002",
            prompt=prompt,
            temperature=0.7,
            max_tokens=cap_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].text

    async def context_generator(self, ctx: discord.ApplicationContext):
        """This method takes in a discord context and a bot, and creates a context string
        of previous messages in the channel and optimizes its length to not surpass a certain
        limit of tokens
        """

        messages = []
        async for message in ctx.channel.history(limit=self.memory_depth):
            messages.append(f"{message.author.name}: {message.content}")

        count = 0
        length_adjusted_messages = []
        for message in messages:
            length_adjusted_messages.append(message)
            count += num_tokens_from_string(message, "text-davinci-003")
            if count > self.cap_tokens:
                break

        context = "\n".join(length_adjusted_messages[::-1])
        pattern = r"<(?:@|@&)(\d)*>"
        matches = re.finditer(pattern, context)

        for match in matches:
            context = context.replace(
                match.group(), self.userify(match.group(), self.bot))

        return context + f"\n{self.bot.user.name}: "

    async def generate_response(self, prompt):
        """
        This method takes in a prompt, and uses the OpenAI API to generate a response.
        """

        return openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=self.cap_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

    def userify(self, user_id, bot):
        """
        This method takes in a user_id and a bot, and returns a clean veresion of the user_id.
        """

        clean_user_id = int(user_id.replace(
            "<@", "").replace(">", "").replace("&", ""))
        if bot.get_user(clean_user_id):
            return f"@{bot.get_user(clean_user_id).name}"
        else:
            return f"@{clean_user_id}"
