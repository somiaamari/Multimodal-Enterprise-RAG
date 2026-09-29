import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_parse import LlamaParse
from app.utils.config import Settings
from app.ingestion.embedding import embed_model
from app.ingestion.captioner import ImageCaptioner

logger = logging.getLogger(__name__)

def create_collection(client, collection_name):
    """Create a Qdrant collection with the correct vector size"""
    collections = client.get_collections().collections
    if collection_name in [c.name for c in collections]:
        print(f"⚠️  Collection '{collection_name}' exists. Recreating...")
        client.delete_collection(collection_name)
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=384,  # all-MiniLM-L6-v2 dimension
            distance=models.Distance.COSINE
        )
    )
    print(f"✅ Collection '{collection_name}' created")

def ingest_document(file_path, collection_name=None):
    """
    Ingest a PDF document into Qdrant.
    Now with MULTIMODAL support: extracts and captions images!
    """
    if collection_name is None:
        collection_name = Settings.COLLECTION_NAME
    
    print(f"📄 Ingesting: {file_path}")
    
    # 1. Connect to Qdrant
    client = QdrantClient(host=Settings.QDRANT_HOST, port=Settings.QDRANT_PORT)
    
    # 2. Create/Recreate the collection
    create_collection(client, collection_name)
    
    # 3. Parse PDF with LlamaParse (extract images too!)
    print("📖 Parsing PDF with LlamaParse (including images)...")
    
    # Create a directory for extracted images
    image_dir = os.path.join("data", "extracted_images")
    os.makedirs(image_dir, exist_ok=True)
    
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_PARSE_API_KEY"),
        result_type="markdown",
        verbose=True,
        images_extract_path=image_dir,  # Extract images to this folder!
    )
    
    documents = parser.load_data(file_path)
    print(f"📖 Loaded {len(documents)} pages")
    
    # 4. Check for extracted images
    extracted_images = []
    if os.path.exists(image_dir):
        extracted_images = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        print(f"🖼️  Found {len(extracted_images)} extracted images")
    
    # 5. Caption images (Multimodal step!)
    if extracted_images:
        print("\n🔄 Captioning images...")
        captioner = ImageCaptioner()
        image_captions = captioner.caption_images_batch(extracted_images)
        
        # Create Document objects for each image caption
        image_docs = []
        for img_path, caption in image_captions.items():
            # Add image info to the text
            metadata = {
                "source": file_path,
                "type": "image",
                "image_path": img_path,
                "page": os.path.basename(img_path).split('_')[0] if '_' in os.path.basename(img_path) else "unknown",
            }
            # Store both caption and the fact that it's an image
            doc_text = f"[Image] {caption} (Image file: {os.path.basename(img_path)})"
            doc = Document(text=doc_text, metadata=metadata)
            image_docs.append(doc)
        
        print(f"✅ Created {len(image_docs)} image caption documents")
    else:
        image_docs = []
    
    # 6. Chunk text nodes
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    text_nodes = splitter.get_nodes_from_documents(documents)
    print(f"✂️  Created {len(text_nodes)} text chunks")
    
    # 7. Combine text nodes and image caption nodes
    all_nodes = text_nodes + image_docs
    print(f"📦 Total nodes: {len(all_nodes)} (text: {len(text_nodes)}, images: {len(image_docs)})")
    
    # 8. Set up Qdrant vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 9. Build the index
    index = VectorStoreIndex(
        nodes=all_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    
    print(f"✅ Index complete! {len(all_nodes)} nodes stored in Qdrant.")
    return index, len(all_nodes)

def test_connection():
    """Quick test to verify Qdrant is reachable"""
    try:
        client = QdrantClient(
            host=AppSettings.QDRANT_HOST,
            port=AppSettings.QDRANT_PORT,
            api_key=AppSettings.QDRANT_API_KEY,
            https=AppSettings.QDRANT_USE_HTTPS,
)   
        collections = client.get_collections()
        print(f"✅ Qdrant connected. Collections: {collections}")
        return client
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return None