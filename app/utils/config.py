import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()


class Settings:
    # Qdrant configuration
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
    QDRANT_USE_HTTPS = os.getenv("QDRANT_USE_HTTPS", "false").lower() == "true"

    # Groq API
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

    # Local embedding model
    EMBED_MODEL = "all-MiniLM-L6-v2"

    # Collection name for our vectors
    COLLECTION_NAME = "enterprise_docs"


def get_qdrant_client() -> QdrantClient:
    """
    Build a QdrantClient that works both locally and on Qdrant Cloud.

    Handles common env-var quirks:
    - Strips surrounding whitespace
    - Strips accidental quotes (from dashboard copy-paste)
    - Auto-detects Qdrant Cloud by domain
    """
    # Sanitize the host value
    host = (Settings.QDRANT_HOST or "").strip().strip('"').strip("'")

    print(f"🔍 Qdrant client init: host={host!r}")

    # Auto-detect Qdrant Cloud
    is_cloud = (
        host.startswith("http://")
        or host.startswith("https://")
        or host.endswith(".qdrant.io")
        or host.endswith(".cloud.qdrant.io")
    )

    if is_cloud:
        # Ensure protocol is present
        if not host.startswith("http://") and not host.startswith("https://"):
            host = "https://" + host

        print(f"🔍 Using url param (cloud): {host}")
        return QdrantClient(
            url=host,
            api_key=Settings.QDRANT_API_KEY,
        )

    # Local Docker
    print(f"🔍 Using host param (local): {host}:{Settings.QDRANT_PORT}")
    return QdrantClient(
        host=host,
        port=Settings.QDRANT_PORT,
        api_key=Settings.QDRANT_API_KEY,
        https=Settings.QDRANT_USE_HTTPS,
    )
