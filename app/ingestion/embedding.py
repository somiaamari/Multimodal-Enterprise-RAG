import os

from llama_index.core import Settings
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding

embed_model = HuggingFaceInferenceAPIEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    token=os.getenv("HF_TOKEN"),
)


Settings.embed_model = embed_model
