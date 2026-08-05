from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage import StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from qdrant_client import QdrantClient
from app.utils.config import Settings as AppSettings
from app.ingestion.embedding import embed_model

# Set up LLM
Settings.llm = OpenAI(
    api_key=AppSettings.GROQ_API_KEY,
    api_base="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile", 
    temperature=0.1,
    context_window=128000,
    max_tokens=4096,
)

# Load index
client = QdrantClient(host=AppSettings.QDRANT_HOST, port=AppSettings.QDRANT_PORT)
vector_store = QdrantVectorStore(client=client, collection_name=AppSettings.COLLECTION_NAME)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# Query
question = "What is the main business of Tesla?"
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query(question)

print(f"❓ Question: {question}")
print(f"📝 Answer: {response}")
print("\n📚 Sources:")
for i, node in enumerate(response.source_nodes, 1):
    print(f"\n[{i}] Score: {node.score:.4f}")
    print(f"    {node.text[:200]}...")