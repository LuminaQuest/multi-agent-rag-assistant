"""
crag_worker.py — Self-Corrective RAG (CRAG) Specialist Agent.
Executes Hybrid Vector + BM25 search, evaluates retrieval confidence,
and autonomously triggers query reformulation loops when relevance falls below threshold.
"""

import time
import re
from typing import List, Dict, Any
from app.agents.state import AgentState
from app.config import settings
from app.rag.vector_index import vector_index
from app.rag.reranker import reranker

def crag_node(state: AgentState) -> AgentState:
    start = time.time()
    state.current_node = "crag_specialist"
    state.node_history.append("crag_specialist")
    
    active_query = state.rewritten_query if state.rewritten_query else state.sanitized_query
    
    # 1. Hybrid Search (Semantic + BM25)
    candidates = vector_index.hybrid_search(active_query, top_k=6)
    
    # 2. Rerank candidates with Reciprocal Rank Fusion
    reranked_chunks, confidence = reranker.rerank(active_query, candidates, top_n=3)
    
    state.retrieval_confidence = confidence
    
    # 3. CRAG Self-Correction Decision Gate
    if confidence < settings.CRAG_RELEVANCE_THRESHOLD and state.rewrite_attempts < 1:
        # Relevance is low/ambiguous -> dynamically reformulate query
        state.rewrite_attempts += 1
        state.crag_status = "AMBIGUOUS_REWRITE"
        
        # Reformulate query: strip filler, focus on high-information keywords
        cleaned_terms = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", active_query)
        stop_words = {"what", "when", "where", "which", "how", "does", "the", "and", "are", "tell", "about", "show", "please"}
        core_keywords = [w for w in cleaned_terms if w.lower() not in stop_words]
        state.rewritten_query = " ".join(core_keywords) + " system architecture policy"
        
        state.step_events.append({
            "node": "crag_specialist",
            "action": f"Relevance confidence ({confidence:.2f} < {settings.CRAG_RELEVANCE_THRESHOLD}) below threshold. Rewrote query -> '{state.rewritten_query}'",
            "timestamp": time.time(),
            "latency_ms": round((time.time() - start) * 1000, 2)
        })
        
        # Loop back: re-execute search with rewritten query
        candidates_v2 = vector_index.hybrid_search(state.rewritten_query, top_k=6)
        reranked_v2, confidence_v2 = reranker.rerank(state.rewritten_query, candidates_v2, top_n=3)
        state.retrieved_chunks = [c.model_dump() for c in reranked_v2]
        state.retrieval_confidence = max(confidence, confidence_v2)
        state.crag_status = "GROUNDED"
    else:
        state.retrieved_chunks = [c.model_dump() for c in reranked_chunks]
        state.crag_status = "GROUNDED" if confidence >= 0.50 else "FALLBACK"
        
    state.step_events.append({
        "node": "crag_specialist",
        "action": f"Retrieved {len(state.retrieved_chunks)} verified passages (Confidence: {state.retrieval_confidence:.2f}, Status: {state.crag_status})",
        "timestamp": time.time(),
        "latency_ms": round((time.time() - start) * 1000, 2)
    })
    
    return state
