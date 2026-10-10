"""
supervisor.py — Autonomous Multi-Agent Supervisor & Planner.
Directs tasks, establishes sub-goals, evaluates recursion bounds, and manages
the agent convergence loop.
"""

import time
from typing import Dict, Any
from app.agents.state import AgentState
from app.config import settings

def supervisor_node(state: AgentState) -> AgentState:
    start = time.time()
    state.current_node = "supervisor"
    state.node_history.append("supervisor")
    
    # Check iteration limits to guarantee convergence
    if state.iteration_count >= state.max_iterations:
        state.workflow_status = "completed"
        state.final_answer = (
            state.draft_answer or 
            "Workflow reached maximum convergence limit (4 iterations). Result synthesized safely."
        )
        return state
        
    state.iteration_count += 1
    
    # Analyze query goal
    query_lower = state.sanitized_query.lower()
    needs_rag = True
    if any(g in query_lower for g in ["hello", "hi", "who are you", "what can you do", "help"]):
        needs_rag = False
        
    state.step_events.append({
        "node": "supervisor",
        "action": f"Cycle {state.iteration_count}: Delegating to {'CRAG Specialist' if needs_rag else 'Direct Synthesizer'}",
        "timestamp": time.time(),
        "latency_ms": round((time.time() - start) * 1000, 2)
    })
    
    return state
