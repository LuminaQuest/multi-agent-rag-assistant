"""
state.py — Cognitive State Schema for Multi-Agent Workflow.
Defines typed state passed across LangGraph nodes, tracking plans, retrieval confidence,
adversarial audits, token metrics, and Human-in-the-Loop decision gates.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class AgentNodeStatus(BaseModel):
    node_name: str
    status: str  # "idle", "active", "completed", "retrying", "paused"
    message: str
    timestamp: float

class AgentState(BaseModel):
    model_config = {"protected_namespaces": ()}
    # Session & Identity
    session_id: str
    trace_id: str
    user_query: str
    sanitized_query: str = ""
    
    # Model Routing
    model_tier: str = "SLM"
    selected_model: str = "Local-Cognitive"
    complexity_score: float = 0.0
    
    # Control Loop & Caps
    iteration_count: int = 0
    max_iterations: int = 4
    current_node: str = "input_guard"
    workflow_status: str = "running"  # "running", "paused_hitl", "completed", "error"
    error_message: Optional[str] = None
    
    # RAG & CRAG Loop
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_confidence: float = 0.0
    crag_status: str = "PENDING"  # "GROUNDED", "AMBIGUOUS_REWRITE", "FALLBACK"
    rewritten_query: Optional[str] = None
    rewrite_attempts: int = 0
    
    # Synthesis & Audit
    draft_answer: str = ""
    final_answer: str = ""
    citations: List[str] = Field(default_factory=list)
    grounding_score: float = 1.0
    hallucination_risk: float = 0.0
    auditor_verdict: str = "PENDING"  # "APPROVED", "FLAGGED", "REVISE"
    
    # Human-in-the-Loop (HITL)
    requires_hitl: bool = False
    hitl_reason: str = ""
    hitl_action: Optional[str] = None  # "APPROVED", "EDITED", "REJECTED"
    hitl_modified_answer: Optional[str] = None
    
    # Real-time Telemetry
    node_history: List[str] = Field(default_factory=list)
    step_events: List[Dict[str, Any]] = Field(default_factory=list)
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    cost_saved_usd: float = 0.0
