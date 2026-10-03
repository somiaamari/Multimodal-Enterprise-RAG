from typing import List

import numpy as np
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore


class Reranker:
    """
    Uses a Cross-Encoder model to re-rank retrieved nodes.
    Model is loaded lazily to reduce memory footprint at startup.
    """

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            print(f"🔄 Loading reranker model: {self.model_name}...")
            self._model = CrossEncoder(self.model_name)
            print("✅ Reranker model loaded!")
        return self._model

    def rerank(
        self, query: str, nodes: List[NodeWithScore], top_k: int = 3
    ) -> List[NodeWithScore]:
        if not nodes:
            return nodes

        pairs = [(query, node.text) for node in nodes]
        scores = self.model.predict(pairs)

        for node, score in zip(nodes, scores):
            node.score = float(score)

        sorted_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
        return sorted_nodes[:top_k]

    def wrap_retriever(self, retriever: BaseRetriever, top_k: int = 3):
        class RerankedRetriever(BaseRetriever):
            def __init__(self, retriever, reranker, top_k):
                self._retriever = retriever
                self._reranker = reranker
                self._top_k = top_k
                super().__init__()

            def _retrieve(self, query_bundle):
                initial_nodes = self._retriever.retrieve(query_bundle)
                return self._reranker.rerank(
                    query_bundle.query_str,
                    initial_nodes,
                    self._top_k,
                )

        return RerankedRetriever(retriever, self, top_k)
