from .message import Message
from .memory_manager import MemoryManager
import openai
from datetime import datetime


class DanielGPT:
    def __init__(self, mongodb_uri: str, openai_api_key: str) -> None:
        self.mm = MemoryManager(mongodb_uri)
        openai.api_key = openai_api_key

    def __get_context(self, message: Message, conversation_id: str) -> list:
        messages = self.mm.get_messages(conversation_id)
        embedded_messages = self.mm.messages_from_embedding(
            message.embedding, conversation_id, 5
        )

        live_context = sorted(
            messages, key=lambda x: datetime.fromtimestamp(x.time), reverse=True
        )[:5][::-1]

        embedded_context = embedded_messages[:3][::-1]

        context = [
            {"role": x.author, "content": x.message}
            for x in embedded_context + live_context
        ]

        return context

    def __get_response(self, message: Message, context: list) -> str:
        formatted_message = [{"role": message.author, "content": message.message}]
        current_date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        knowledge_cutoff = "September 2021"

        initial_context = [
            {
                "role": "user",
                "content": f"You are DanielGPT, a large language model discord bot. Answer as concisely as possible. Format your responses in github markdown. Knowledge cutoff: {knowledge_cutoff} Current date: {current_date}",
            }
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=initial_context + context + formatted_message,
        )

        return response["choices"][0]["message"]["content"]

    def __post_message(self, message: Message, conversation_id: str) -> None:
        self.mm.post_message(conversation_id, message)

    async def message(self, message: str, author: str, conversation_id: str) -> str:
        message = Message(message, author)

        context = self.__get_context(message, conversation_id)
        response = self.__get_response(message, context)

        self.__post_message(message, conversation_id)
        self.__post_message(Message(response, "assistant"), conversation_id)
        return response
