from app.ingestion.embedding import test_indexing

if __name__ == "__main__":
    print("🚀 Starting LlamaIndex integration test...")
    try:
        index = test_indexing()
        print("\n🎉 LlamaIndex + Qdrant integration is fully working!")
        print("   Your RAG pipeline foundation is solid.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure Qdrant is running (docker-compose up -d)")