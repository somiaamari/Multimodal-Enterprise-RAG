from typing import List
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    """
    Uses a Cross-Encoder model to re-rank retrieved nodes.
    Cross-Encoders are more accurate than bi-encoders (like our embedding model)
    but are slower, so we only use them on the top-k candidates.
    """
    
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # This model is small, fast, and free (local)
        print(f"🔄 Loading reranker model: {model_name}...")
        self.model = CrossEncoder(model_name)
        print("✅ Reranker model loaded!")
    
    def rerank(self, query: str, nodes: List[NodeWithScore], top_k: int = 3) -> List[NodeWithScore]:
        """
        Rerank nodes based on the query using a cross-encoder.
        Returns the top-k nodes with updated scores.
        """
        if not nodes:
            return nodes
        
        # Prepare pairs for cross-encoder: (query, passage)
        pairs = [(query, node.text) for node in nodes]
        
        # Get similarity scores from cross-encoder (0 to 1)
        scores = self.model.predict(pairs)
        
        # Assign scores back to nodes
        for node, score in zip(nodes, scores):
            node.score = float(score)  # Overwrite with cross-encoder score
        
        # Sort by new score (descending)
        sorted_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
        
        # Return top-k
        return sorted_nodes[:top_k]
    
    def wrap_retriever(self, retriever: BaseRetriever, top_k: int = 3):
        """Wraps a retriever to automatically rerank results"""
        
        class RerankedRetriever(BaseRetriever):
            def __init__(self, retriever, reranker, top_k):
                self._retriever = retriever
                self._reranker = reranker
                self._top_k = top_k
                super().__init__()
            
            def _retrieve(self, query_bundle):
                # 1. Get initial results from base retriever (fetch more than top_k)
                initial_nodes = self._retriever.retrieve(query_bundle)
                
                # 2. Rerank them
                reranked_nodes = self._reranker.rerank(
                    query_bundle.query_str, 
                    initial_nodes, 
                    self._top_k
                )
                
                return reranked_nodes
        
        return RerankedRetriever(retriever, self, top_k)