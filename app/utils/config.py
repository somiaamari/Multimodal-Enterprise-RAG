import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Qdrant configuration
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    
    
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
    QDRANT_USE_HTTPS = os.getenv("QDRANT_USE_HTTPS", "false").lower() == "true"
    
    EMBED_MODEL = "all-MiniLM-L6-v2"
    
    COLLECTION_NAME = "enterprise_docs"