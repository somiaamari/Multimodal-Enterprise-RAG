from app.retrieval.retriever import DocumentRetriever

def ask_question(question, top_k=3):
    """Ask a question and get an answer with sources"""
    print(f"\n❓ Question: {question}")
    print("-" * 60)
    
    # Initialize retriever
    retriever = DocumentRetriever()
    
    # Get answer
    result = retriever.query(question, similarity_top_k=top_k)
    
    # Print answer
    print(f"\n📝 Answer:\n{result['answer']}")
    
    # Print sources
    print(f"\n📚 Sources ({result['source_count']}):")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n[{i}] Score: {source['score']:.4f}")
        print(f"    Preview: {source['text'][:200]}...")
        if 'file_name' in source['metadata']:
            print(f"    File: {source['metadata']['file_name']}")
        if 'page_label' in source['metadata']:
            print(f"    Page: {source['metadata']['page_label']}")
    
    return result

if __name__ == "__main__":
    # Example questions - replace with your own
    questions = [
        "What is the main business of Tesla?",
        "What were Tesla's revenues in 2023?",
        "What are the main risks mentioned in the report?",
    ]
    
    for q in questions:
        ask_question(q, top_k=3)
        print("\n" + "=" * 60)