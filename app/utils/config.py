import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Qdrant configuration
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    
    # Groq API (free - get from console.groq.com)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    # Local embedding model (runs completely offline, no API key needed)
    EMBED_MODEL = "all-MiniLM-L6-v2"
    
    # Collection name for our vectors
    COLLECTION_NAME = "enterprise_docs"