from app.agents.self_rag import SelfRAGEngine

def run_demo():
    """Run a Self-RAG demo with a set of test questions."""
    
    # Initialize Self-RAG
    print("🚀 Initializing Self-RAG Engine...")
    rag = SelfRAGEngine(max_retries=2, relevance_threshold=-1.0)
    
    # Test questions (mix of good and difficult ones)
    questions = [
        # Good questions (should work)
        "What were Tesla's revenues in 2023?",
        "What is the main business of Tesla?",
        "What are the main risks mentioned in the report?",
        "TSLA stock",
        
        # Tricky questions (might require rewriting or rejection)
        "What color is the CEO's car?",
        "How many employees does Tesla have in China?",
    ]
    
    print("\n" + "=" * 70)
    print("SELF-RAG DEMO")
    print("=" * 70)
    
    for q in questions:
        result = rag.query(q, top_k=5)
        
        print("\n" + "=" * 60)
        print("📝 FINAL RESULT:")
        print("=" * 60)
        
        if result.get("status") == "rejected":
            print(f"❌ {result['answer']}")
        else:
            print(f"✅ Answer: {result['answer']}")
            print(f"\n📚 Sources: {result.get('source_count', 0)}")
            print(f"   Retries: {result.get('retry_count', 0)}")
            print(f"   Retrieval Score: {result.get('retrieval_score', 0):.2f}")
        
        print("\n" + "=" * 70)
        
        # Pause between questions
        input("\nPress Enter to continue to next question...")

if __name__ == "__main__":
    run_demo()