from app.ingestion.pipeline import test_connection, create_collection_if_not_exists

if __name__ == "__main__":
    print("Testing environment...")
    client = test_connection()
    
    if client:
        create_collection_if_not_exists(client)
        print("\n Environment is fully ready!")
    else:
        print("\n❌ Environment check failed. Make sure Docker is running and Qdrant is up.")