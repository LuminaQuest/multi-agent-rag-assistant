"""
synthesizer.py — Citation-Grounded Knowledge Synthesizer Agent.
Formulates structured, factual synthesis strictly anchored to retrieved source context,
injecting exact citation tokens [Doc-X].
"""

import time
import re
from typing import Dict, Any
from app.agents.state import AgentState
from app.router.pareto_router import pareto_router

def synthesizer_node(state: AgentState) -> AgentState:
    start = time.time()
    state.current_node = "synthesizer"
    state.node_history.append("synthesizer")
    
    # Format context passages with citation labels
    context_str = ""
    for idx, chunk in enumerate(state.retrieved_chunks, start=1):
        context_str += f"\n--- [Doc-{idx}] Source: {chunk.get('doc_name')} ---\n{chunk.get('content')}\n"
        
    prompt = f"""User Question: {state.sanitized_query}

Verified Context:
{context_str}

Instructions:
1. Synthesize a concise, executive-level technical answer strictly based on the verified context above.
2. Every factual assertion MUST cite its exact source tag, e.g. [Doc-1] or [Doc-2].
3. If the context does not contain the answer, state that clearly without guessing."""

    # Execute prompt through Pareto-routed model
    res = pareto_router.execute_prompt(prompt)
    
    state.draft_answer = res.get("text", "")
    state.citations = list(set(re.findall(r"\[Doc-\d+\]", state.draft_answer)))
    state.total_tokens += res.get("tokens_in", 0) + res.get("tokens_out", 0)
    
    state.step_events.append({
        "node": "synthesizer",
        "action": f"Synthesized draft answer with {len(state.citations)} citations via {res.get('model')}",
        "timestamp": time.time(),
        "latency_ms": round((time.time() - start) * 1000, 2)
    })
    
    return state
