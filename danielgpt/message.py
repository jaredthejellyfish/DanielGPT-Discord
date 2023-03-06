import time
from openai.embeddings_utils import get_embedding


class Message:
    def __init__(
        self, message: str, author: str, sent_time: float = None, embedding: list = None
    ) -> None:
        self.message = message
        self.author = author

        if sent_time is None:
            self.time = time.time()
        else:
            self.time = sent_time

        if embedding is None:
            self.embedding = get_embedding(
                f"{self.author}: {self.message}", engine="text-embedding-ada-002"
            )
        else:
            self.embedding = embedding

    def __str__(self) -> str:
        return f"{self.author}: {self.message}"

    def __repr__(self) -> str:
        return f"{self.author}: {self.message} at {self.time}"

    def toDict(self) -> dict:
        return {
            "message": self.message,
            "author": self.author,
            "embedding": self.embedding,
            "time": self.time,
        }
