import logging
from typing import Optional, Dict, Any
from app.retrieval.retriever import DocumentRetriever
from app.agents.evaluator import RelevanceEvaluator
from llama_index.llms.openai_like import OpenAILike
from app.utils.config import Settings as AppSettings

logger = logging.getLogger(__name__)

class SelfRAGEngine:
    """
    Self-RAG (Self-Reflective Retrieval-Augmented Generation)
    Evaluates retrieval quality and decides whether to answer, retry, or reject.
    """
    
    def __init__(
        self,
        retriever: Optional[DocumentRetriever] = None,
        evaluator: Optional[RelevanceEvaluator] = None,
        max_retries: int = 2,
        relevance_threshold: float = -1.0,
    ):
        """
        Args:
            retriever: DocumentRetriever instance. If None, creates one.
            evaluator: RelevanceEvaluator instance. If None, creates one.
            max_retries: Maximum number of retrieval attempts (including query rewriting).
            relevance_threshold: Minimum average score to consider retrieval relevant.
        """
        self.retriever = retriever or DocumentRetriever(
            use_hybrid=False,  # Use vector only for simplicity
            use_reranker=True   # Keep the reranker for better scores
        )
        self.evaluator = evaluator or RelevanceEvaluator()
        self.max_retries = max_retries
        self.relevance_threshold = relevance_threshold
        
        # Initialize LLM for query rewriting
        self.llm = OpenAILike(
            api_key=AppSettings.GROQ_API_KEY,
            api_base="https://api.groq.com/openai/v1",
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            context_window=128000,
            max_tokens=4096,
            is_chat_model=True,
        )
    
    def _rewrite_query(self, original_query: str) -> str:
        """
        Rewrite the query to be more specific or better phrased
        for retrieval from a financial document.
        """
        prompt = f"""
You are a query rewriter for a document retrieval system. The user asked:

Original question: "{original_query}"

This question will be used to search a financial document (Tesla 10-K annual report).
Rewrite it to be more specific and likely to retrieve relevant information.

IMPORTANT:
- Keep the same meaning but improve search phrasing
- For stock tickers (like TSLA), keep the ticker and add context like "trading" or "market"
- For numbers, keep them specific
- Return ONLY the rewritten question, nothing else.
"""
        try:
            response = self.llm.complete(prompt)
            rewritten = str(response).strip()
            print(f"   ✏️  Rewritten: {rewritten}")
            return rewritten
        except Exception as e:
            print(f"   ⚠️  Query rewriting failed: {e}")
            return original_query
    
    def _reject_response(self, question: str) -> Dict[str, Any]:
        """Generate a rejection response when retrieval fails."""
        return {
            "answer": f"I couldn't find relevant information in the document to answer: \"{question}\". Please try rephrasing or ask something else.",
            "sources": [],
            "source_count": 0,
            "retry_count": self.max_retries,
            "status": "rejected"
        }
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Main entry point for Self-RAG.
        Returns answer with sources, or rejection message.
        """
        print(f"\n❓ Question: {question}")
        print("-" * 60)
        
        current_query = question
        retrieved_nodes = None
        
        for attempt in range(self.max_retries):
            print(f"\n🔍 Attempt {attempt + 1}/{self.max_retries}")
            
            # 1. Retrieve
            if attempt > 0:
                # Rewrite the query for subsequent attempts
                current_query = self._rewrite_query(question)
            
            print(f"   Query: {current_query}")
            
            # Use the retriever's internal retrieval (without generating answer)
            # We need to get nodes first
            if self.retriever.index is None:
                self.retriever.load_index()
            
            # Get raw retriever from the query engine
            if self.retriever.use_hybrid:
                base_retriever = self.retriever._get_hybrid_retriever(top_k)
            else:
                base_retriever = self.retriever._get_vector_retriever(top_k)
            
            # Apply reranker if enabled
            if self.retriever.use_reranker and self.retriever.reranker:
                try:
                    base_retriever = self.retriever.reranker.wrap_retriever(
                        base_retriever, top_k=3
                    )
                except Exception:
                    pass
            
            # Retrieve nodes
            from llama_index.core.schema import QueryBundle
            query_bundle = QueryBundle(query_str=current_query)
            retrieved_nodes = base_retriever.retrieve(query_bundle)
            
            # 2. Evaluate retrieval quality
            avg_score, is_relevant, reason = self.evaluator.evaluate_retrieval(
                current_query, retrieved_nodes
            )
            
            print(f"   📊 Score: {avg_score:.2f}")
            print(f"   📋 Status: {'✅ RELEVANT' if is_relevant else '❌ NOT RELEVANT'}")
            print(f"   💬 Reason: {reason}")
            
            # 3. Decide
            if is_relevant or avg_score > self.relevance_threshold:
                # Good retrieval! Generate answer
                print("\n💡 Generating answer...")
                
                # Create query engine and run
                self.retriever.create_query_engine(similarity_top_k=top_k)
                response = self.retriever.query(question)
                
                # Add retry info
                response["retry_count"] = attempt + 1
                response["status"] = "answered"
                response["retrieval_score"] = avg_score
                
                return response
            
            elif attempt == self.max_retries - 1:
                # Last attempt failed
                print(f"\n❌ Retrieval failed after {self.max_retries} attempts.")
                return self._reject_response(question)
            
            else:
                # Will retry
                print("   🔄 Retrying with rewritten query...")
        
        # Should never reach here
        return self._reject_response(question)