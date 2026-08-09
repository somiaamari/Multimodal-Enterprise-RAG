from app.retrieval.retriever import DocumentRetriever

def ask_question(question, top_k=10, use_hybrid=False, use_reranker=True):
    """Ask a question with hybrid search and reranking"""
    print(f"\n❓ Question: {question}")
    print("-" * 60)
    print(f"⚙️  Hybrid: {use_hybrid}, Reranker: {use_reranker}")
    print("-" * 60)
    
    # Initialize retriever with new features
    retriever = DocumentRetriever(
        use_hybrid=use_hybrid,
        use_reranker=use_reranker
    )
    
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
    # Test questions
    questions = [
        "What were Tesla's revenues in 2023?",
        "What is the main business of Tesla?",
        "What are the main risks mentioned in the report?",
        "TSLA stock",  # Keyword-specific test!
    ]
    
    for q in questions:
        ask_question(q, top_k=10, use_hybrid=True, use_reranker=True)
        print("\n" + "=" * 60)