"""FastEmbed embeddings — fully synchronous, no async HTTP, no API key needed."""

from llama_index.core import Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding

embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
