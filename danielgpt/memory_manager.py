import numpy as np
import pandas as pd
from .message import Message
from pymongo import MongoClient
from openai.embeddings_utils import cosine_similarity


class MemoryManager:
    def __init__(self, mongodb_uri: str):
        cluster = MongoClient(mongodb_uri)
        self.db = cluster["conversations"]

    def get_conversation(self, conversation_id: str) -> dict:
        return self.db[conversation_id]

    def post_message(self, conversation_id: str, message: Message) -> None:
        collection = self.db[conversation_id]
        collection.insert_one(message.__dict__)

    def messages_from_embedding(self, embedding: list, conversation_id: str, n: int=10) -> list:
        conversation = self.get_conversation(conversation_id)
        messages = conversation.find({})

        messages_list = [x for _, x in enumerate(messages)]
        df = pd.DataFrame(messages_list)

        if df.empty:
            return []

        df["embedding"] = df.embedding.apply(
            lambda x: x if isinstance(x, list) else eval(x)
        ).apply(np.array)

        df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, embedding))

        results = df.sort_values("similarity", ascending=False).head(n)

        messages = results.to_dict("records")

        messages = [
            x for x in messages if x["similarity"] > 0.82 and x["similarity"] < 0.98
        ]

        messages = sorted(messages, key=lambda x: x["similarity"], reverse=True)

        messages_list = [
            Message(x["message"], x["author"], x["time"], x["embedding"])
            for x in messages
        ]

        return messages_list

    def get_messages(self, conversation_id: str) -> list:
        conversation = self.get_conversation(conversation_id)
        messages = conversation.find({})
        messages_list = [
            Message(x["message"], x["author"], x["time"], x["embedding"])
            for _, x in enumerate(messages)
        ]
        return messages_list
