"""
auditor.py — Adversarial Red-Team Fact-Checker & Governance Auditor.
Evaluates:
1. Hallucination Risk Index & Grounding fidelity
2. Critical Policy / Human-in-the-Loop (HITL) triggers
"""

import time
import re
from typing import Dict, Any
from app.agents.state import AgentState
from app.guardrails.output_guards import output_guard
from app.config import settings

def auditor_node(state: AgentState) -> AgentState:
    start = time.time()
    state.current_node = "auditor"
    state.node_history.append("auditor")
    
    # 1. Audit hallucination & groundedness
    audit_res = output_guard.audit_response(state.draft_answer, state.retrieved_chunks)
    state.grounding_score = audit_res.grounding_score
    state.hallucination_risk = audit_res.hallucination_risk_index
    state.citations = audit_res.citations_detected
    
    # 2. Check for High-Impact Human-in-the-Loop (HITL) Triggers
    hitl_triggers = [
        r"\b(transfer|fund|delete|drop\s+table|mutate|terminate|deploy\s+production|override\s+policy)\b",
        r"\b(approve\s+budget|execute\s+transaction|critical\s+action)\b"
    ]
    
    query_hitl = any(re.search(pat, state.user_query.lower()) for pat in hitl_triggers)
    confidence_low = state.retrieval_confidence < settings.HITL_CONFIDENCE_THRESHOLD and len(state.retrieved_chunks) > 0
    
    if query_hitl:
        state.requires_hitl = True
        state.workflow_status = "paused_hitl"
        state.hitl_reason = "High-impact autonomous mutation detected (financial, database, or policy override). Executive human approval required."
        state.auditor_verdict = "PAUSED_FOR_HITL"
    elif confidence_low:
        state.requires_hitl = True
        state.workflow_status = "paused_hitl"
        state.hitl_reason = f"Low retrieval confidence ({state.retrieval_confidence:.2f} < {settings.HITL_CONFIDENCE_THRESHOLD}). Manual confirmation required before releasing output."
        state.auditor_verdict = "PAUSED_FOR_HITL"
    else:
        state.requires_hitl = False
        state.workflow_status = "completed"
        state.auditor_verdict = "APPROVED"
        state.final_answer = audit_res.sanitized_response
        
    state.step_events.append({
        "node": "auditor",
        "action": f"Audit complete. Grounding: {state.grounding_score*100:.0f}%, Hallucination Risk: {state.hallucination_risk:.2f}, Verdict: {state.auditor_verdict}",
        "timestamp": time.time(),
        "latency_ms": round((time.time() - start) * 1000, 2)
    })
    
    return state
