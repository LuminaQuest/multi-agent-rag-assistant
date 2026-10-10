"""
main.py — Production FastAPI Service for NexusCore Multi-Agent RAG Assistant.
Serves:
1. RESTful agent endpoints (/api/chat, /api/hitl/action, /api/documents, /api/evals/run, /api/telemetry)
2. Static Executive Command Center UI
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from app.agents.graph import run_agent_workflow, resume_hitl_workflow, active_hitl_states
from app.rag.vector_index import vector_index
from app.telemetry.tracer import tracer
from evals.run_evals import execute_evaluation_suite

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Enterprise-grade Autonomous Multi-Agent Cognitive Assistant with Self-Corrective RAG, Pareto Model Routing & Guardrails."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Request & Response Models
# -------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str = Field(default="exec-session-default")
    query: str = Field(..., min_length=1, max_length=4000)

class HitlActionRequest(BaseModel):
    session_id: str
    action: str = Field(..., pattern="^(APPROVE|EDIT|REJECT)$")
    modified_answer: Optional[str] = None

class AddDocumentRequest(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=10)
    category: str = "General"

# -------------------------------------------------------------
# Core API Endpoints
# -------------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "documents_indexed": len(vector_index.documents),
        "chunks_indexed": len(vector_index.chunks)
    }

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    try:
        state = run_agent_workflow(session_id=req.session_id, user_query=req.query)
        return {
            "session_id": state.session_id,
            "trace_id": state.trace_id,
            "status": state.workflow_status,
            "final_answer": state.final_answer,
            "draft_answer": state.draft_answer,
            "citations": state.citations,
            "grounding_score": state.grounding_score,
            "hallucination_risk": state.hallucination_risk,
            "model_tier": state.model_tier,
            "selected_model": state.selected_model,
            "retrieval_confidence": state.retrieval_confidence,
            "crag_status": state.crag_status,
            "requires_hitl": state.requires_hitl,
            "hitl_reason": state.hitl_reason,
            "step_events": state.step_events,
            "retrieved_chunks": state.retrieved_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/hitl/action")
def hitl_action_endpoint(req: HitlActionRequest):
    try:
        if req.session_id not in active_hitl_states:
            raise HTTPException(status_code=404, detail="No active workflow paused at Human-in-the-Loop gateway for this session.")
            
        state = resume_hitl_workflow(
            session_id=req.session_id,
            action=req.action,
            modified_answer=req.modified_answer
        )
        return {
            "session_id": state.session_id,
            "status": state.workflow_status,
            "hitl_action": state.hitl_action,
            "final_answer": state.final_answer,
            "citations": state.citations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/hitl/pending")
def hitl_pending_endpoint():
    pending = []
    for sid, state in active_hitl_states.items():
        pending.append({
            "session_id": sid,
            "user_query": state.user_query,
            "draft_answer": state.draft_answer,
            "hitl_reason": state.hitl_reason,
            "retrieval_confidence": state.retrieval_confidence,
            "hallucination_risk": state.hallucination_risk
        })
    return {"pending": pending}

@app.get("/api/documents")
def list_documents():
    return {
        "documents": list(vector_index.documents.values()),
        "total_chunks": len(vector_index.chunks)
    }

@app.post("/api/documents")
def add_document(req: AddDocumentRequest):
    chunk_ids = vector_index.add_document(
        title=req.title,
        text=req.content,
        metadata={"category": req.category}
    )
    return {
        "title": req.title,
        "chunks_created": len(chunk_ids),
        "chunk_ids": chunk_ids
    }

@app.get("/api/evals/run")
def trigger_evals():
    summary = execute_evaluation_suite()
    return summary

@app.get("/api/telemetry")
def get_telemetry():
    return {
        "aggregate": tracer.get_aggregate_metrics(),
        "recent_traces": tracer.get_recent_traces(limit=10)
    }

# -------------------------------------------------------------
# Static UI Mounting
# -------------------------------------------------------------
static_dir = ROOT_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def serve_dashboard():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "NexusCore API Online. Visit /docs for OpenAPI documentation."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
