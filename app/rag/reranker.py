"""
reranker.py — Reciprocal Rank Fusion (RRF) & Relevance Scorer.
Reranks retrieved candidate chunks using multi-criteria cross-scoring:
1. Reciprocal Rank Fusion (RRF) combining semantic similarity and keyword presence
2. Query coverage & passage density penalty/bonus
3. Retrieval confidence computation for Self-Corrective RAG (CRAG) decision loops
"""

import math
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field

class RerankResult(BaseModel):
    chunk_id: str
    doc_name: str
    content: str
    score: float
    confidence_grade: str  # "HIGH", "MEDIUM", "LOW"
    citation_id: str

class Reranker:
    def __init__(self, k_rrf: int = 60):
        self.k_rrf = k_rrf

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 3) -> Tuple[List[RerankResult], float]:
        if not candidates:
            return [], 0.0

        query_terms = set(query.lower().split())
        scored: List[Tuple[float, Dict[str, Any]]] = []
        
        for rank, chunk in enumerate(candidates, start=1):
            raw_score = chunk.get("raw_score", 0.5)
            
            # Reciprocal Rank Fusion component
            rrf_score = 1.0 / (self.k_rrf + rank)
            
            # Passage density & term matching
            content_lower = chunk.get("content", "").lower()
            matched_terms = sum(1 for t in query_terms if t in content_lower)
            term_coverage = matched_terms / max(1, len(query_terms))
            
            # Combined rerank score normalized to 0.0 - 1.0
            final_score = round(min(1.0, (raw_score * 0.55) + (term_coverage * 0.35) + (rrf_score * 10.0)), 4)
            scored.append((final_score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored[:top_n]
        
        results: List[RerankResult] = []
        for idx, (score, chunk) in enumerate(top_candidates, start=1):
            if score >= 0.72:
                grade = "HIGH"
            elif score >= 0.50:
                grade = "MEDIUM"
            else:
                grade = "LOW"
                
            results.append(RerankResult(
                chunk_id=chunk.get("chunk_id", f"c-{idx}"),
                doc_name=chunk.get("doc_name", "Unknown Document"),
                content=chunk.get("content", ""),
                score=score,
                confidence_grade=grade,
                citation_id=f"[Doc-{idx}]"
            ))

        # Overall retrieval confidence is governed by the highest-scoring verified chunk
        overall_confidence = round(results[0].score, 3) if results else 0.0
        return results, overall_confidence

reranker = Reranker()
