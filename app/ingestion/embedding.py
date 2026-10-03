import os

from llama_index.embeddings.huggingface_api import \
    HuggingFaceInferenceAPIEmbedding

from app.utils.config import Settings as AppSettings

# Hugging Face Inference API embedding
# Uses the free inference endpoint; requires HF_TOKEN for higher limits.
embed_model = HuggingFaceInferenceAPIEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    token=os.getenv("HF_TOKEN"),
)
