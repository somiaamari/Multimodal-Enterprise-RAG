import asyncio
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

load_dotenv()


class HuggingFaceInferenceEmbedding(BaseEmbedding):
    _client: InferenceClient = PrivateAttr()

    def __init__(self, model_name: str, token: str | None = None, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self._client = InferenceClient(
            model=model_name,
            token=token,
            provider="hf-inference",
            timeout=60,
        )

    def _embed(self, text: str) -> list[float]:
        vector = self._client.feature_extraction(text, normalize=True)
        return vector.tolist()

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._embed, query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._embed, text)

embed_model = HuggingFaceInferenceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    token=os.getenv("HF_TOKEN"),
)

Settings.embed_model = embed_model
