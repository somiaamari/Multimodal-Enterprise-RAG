from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Tuple

class RelevanceEvaluator:
    """
    Evaluates relevance between a question and retrieved text chunks
    using a Cross-Encoder model.
    """
    
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"🔄 Loading evaluator model: {model_name}...")
        self.model = CrossEncoder(model_name)
        print("✅ Evaluator model loaded!")
    
    def score(self, query: str, contexts: List[str]) -> List[float]:
        """
        Score each context against the query.
        Returns a list of scores (higher = more relevant).
        """
        if not contexts:
            return []
        
        # Prepare pairs: (query, context)
        pairs = [(query, context) for context in contexts]
        
        # Get scores (cross-encoder outputs similarity)
        scores = self.model.predict(pairs)
        
        # Convert to list if it's a numpy array
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        
        return scores
    
    def evaluate_retrieval(self, query: str, nodes: List) -> Tuple[float, bool, str]:
        """
        Evaluate retrieved nodes for relevance to the query.
        Returns:
            - average_score: float
            - is_relevant: bool (True if avg_score > threshold)
            - reason: str
        """
        if not nodes:
            return 0.0, False, "No chunks retrieved"
        
        # Extract text from nodes
        contexts = [node.text for node in nodes]
        
        # Score each context
        scores = self.score(query, contexts)
        
        # Calculate average score
        avg_score = sum(scores) / len(scores)
        
        # Determine if relevant (threshold tuned for MiniLM cross-encoder)
        # Scores from cross-encoder typically range from -10 to +10
        # For this model, > 2.0 is generally relevant
        threshold = 2.0
        is_relevant = avg_score > threshold
        
        # Generate reason
        if is_relevant:
            reason = f"Retrieved chunks are relevant (avg score: {avg_score:.2f})"
        else:
            reason = f"Retrieved chunks are not relevant enough (avg score: {avg_score:.2f})"
        
        return avg_score, is_relevant, reason