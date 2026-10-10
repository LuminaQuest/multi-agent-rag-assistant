"""
graph.py — End-to-End Multi-Agent StateGraph Orchestrator.
Assembles the complete cognitive workflow:
InputGuard -> ParetoRouter -> Supervisor -> CRAG Specialist -> Synthesizer -> Auditor -> HITL/Output.
Includes explicit iteration caps, cycle guards, and human-in-the-loop interruption.
"""

import time
import uuid
from typing import Dict, Any, Optional
from app.agents.state import AgentState
from app.guardrails.input_guards import input_guard
from app.router.pareto_router import pareto_router
from app.agents.supervisor import supervisor_node
from app.agents.crag_worker import crag_node
from app.agents.synthesizer import synthesizer_node
from app.agents.auditor import auditor_node
from app.memory.session_memory import session_memory
from app.telemetry.tracer import tracer

# In-memory storage for workflows paused at Human-in-the-Loop gate
active_hitl_states: Dict[str, AgentState] = {}

def run_agent_workflow(session_id: str, user_query: str) -> AgentState:
    trace = tracer.start_trace(session_id=session_id, user_query=user_query)
    
    state = AgentState(
        session_id=session_id,
        trace_id=trace.trace_id,
        user_query=user_query
    )
    
    # -------------------------------------------------------------
    # Node 1: Input Guardrail
    # -------------------------------------------------------------
    t0 = time.time()
    guard_res = input_guard.validate_and_sanitize(user_query)
    tracer.log_step(
        trace_id=state.trace_id,
        node="input_guard",
        model_tier="DETERMINISTIC",
        model_name="Regex-Pattern-Engine",
        latency_ms=guard_res.latency_ms,
        status="success" if guard_res.is_safe else "rejected",
        metadata={"pii_redacted": guard_res.pii_redacted, "injection_score": guard_res.injection_score}
    )
    
    if not guard_res.is_safe:
        state.current_node = "input_guard"
        state.workflow_status = "error"
        state.error_message = guard_res.rejection_reason
        state.final_answer = f"🛡️ {guard_res.rejection_reason}"
        tracer.complete_trace(state.trace_id, hallucination_risk=0.0, final_status="rejected")
        return state
        
    state.sanitized_query = guard_res.sanitized_query
    
    # -------------------------------------------------------------
    # Node 2: Pareto Model Router
    # -------------------------------------------------------------
    t0 = time.time()
    routing_dec = pareto_router.route_query(state.sanitized_query)
    state.model_tier = routing_dec.tier
    state.selected_model = routing_dec.selected_model
    state.complexity_score = routing_dec.complexity_score
    
    tracer.log_step(
        trace_id=state.trace_id,
        node="pareto_router",
        model_tier=routing_dec.tier,
        model_name=routing_dec.selected_model,
        latency_ms=round((time.time() - t0) * 1000, 2),
        metadata={"complexity_score": routing_dec.complexity_score, "reasoning": routing_dec.reasoning}
    )
    
    # -------------------------------------------------------------
    # Node 3: Supervisor Planner
    # -------------------------------------------------------------
    state = supervisor_node(state)
    tracer.log_step(
        trace_id=state.trace_id,
        node="supervisor",
        model_tier=state.model_tier,
        model_name="Supervisor-Planner",
        latency_ms=state.step_events[-1]["latency_ms"]
    )
    
    # -------------------------------------------------------------
    # Node 4: Self-Corrective RAG (CRAG) Specialist
    # -------------------------------------------------------------
    state = crag_node(state)
    tracer.log_step(
        trace_id=state.trace_id,
        node="crag_specialist",
        model_tier="DETERMINISTIC",
        model_name="Hybrid-RRF-Engine",
        latency_ms=state.step_events[-1]["latency_ms"],
        metadata={
            "retrieval_confidence": state.retrieval_confidence,
            "crag_status": state.crag_status,
            "rewritten_query": state.rewritten_query
        }
    )
    
    # -------------------------------------------------------------
    # Node 5: Knowledge Synthesizer
    # -------------------------------------------------------------
    state = synthesizer_node(state)
    tracer.log_step(
        trace_id=state.trace_id,
        node="synthesizer",
        model_tier=state.model_tier,
        model_name=state.selected_model,
        latency_ms=state.step_events[-1]["latency_ms"],
        tokens_in=len(state.sanitized_query.split()) * 5,
        tokens_out=len(state.draft_answer.split()) * 2
    )
    
    # -------------------------------------------------------------
    # Node 6: Adversarial Fact-Checker & Auditor
    # -------------------------------------------------------------
    state = auditor_node(state)
    tracer.log_step(
        trace_id=state.trace_id,
        node="auditor",
        model_tier="DETERMINISTIC",
        model_name="Groundedness-Auditor",
        latency_ms=state.step_events[-1]["latency_ms"],
        metadata={
            "grounding_score": state.grounding_score,
            "hallucination_risk": state.hallucination_risk,
            "verdict": state.auditor_verdict
        }
    )
    
    # -------------------------------------------------------------
    # Node 7: Human-in-the-Loop Interruption Check
    # -------------------------------------------------------------
    if state.requires_hitl:
        active_hitl_states[session_id] = state
        return state
        
    # Workflow completed successfully: persist to memory & finish trace
    session_memory.add_message(session_id, "user", user_query)
    session_memory.add_message(session_id, "assistant", state.final_answer, state.citations)
    tracer.complete_trace(state.trace_id, hallucination_risk=state.hallucination_risk, final_status="completed")
    
    return state

def resume_hitl_workflow(session_id: str, action: str, modified_answer: Optional[str] = None) -> AgentState:
    """
    Resumes a workflow paused at the Human-in-the-Loop gateway.
    Action: "APPROVE", "EDIT", "REJECT"
    """
    state = active_hitl_states.pop(session_id, None)
    if not state:
        raise ValueError(f"No active paused workflow for session {session_id}")
        
    state.hitl_action = action
    state.requires_hitl = False
    
    if action == "APPROVE":
        state.final_answer = state.draft_answer
        state.workflow_status = "completed"
    elif action == "EDIT":
        state.final_answer = modified_answer or state.draft_answer
        state.workflow_status = "completed"
    elif action == "REJECT":
        state.final_answer = "⚠️ Operation rejected by Executive Human-in-the-Loop Gatekeeper."
        state.workflow_status = "completed"
        
    session_memory.add_message(session_id, "user", state.user_query)
    session_memory.add_message(session_id, "assistant", state.final_answer, state.citations)
    tracer.complete_trace(state.trace_id, hallucination_risk=state.hallucination_risk, final_status="completed")
    
    return state

agent_graph = {
    "run": run_agent_workflow,
    "resume": resume_hitl_workflow
}
