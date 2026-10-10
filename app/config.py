"""
config.py — Central Configuration & Model Provider Settings.
Supports dynamic multi-provider routing (Gemini, Groq, OpenAI, Ollama, and Local-Cognitive).
Zero-friction setup: defaults gracefully to Local-Cognitive engine if external API keys are not supplied.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

class Settings(BaseModel):
    # App Information
    APP_NAME: str = "NexusCore: Autonomous Multi-Agent Cognitive Assistant"
    VERSION: str = "2.1.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    
    # Storage Paths
    DB_PATH: Path = DATA_DIR / "session_memory.db"
    VECTOR_STORAGE_PATH: Path = DATA_DIR / "vector_index.json"
    LOGS_PATH: Path = DATA_DIR / "telemetry_traces.jsonl"
    
    # Model Provider API Keys (optional for zero-friction demo)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Multi-Agent Workflow Controls & Guardrails
    MAX_ITERATIONS: int = 4
    CRAG_RELEVANCE_THRESHOLD: float = 0.60
    ROUTING_COMPLEXITY_THRESHOLD: float = 0.65
    HITL_CONFIDENCE_THRESHOLD: float = 0.35
    
    # Token Economics (Cost approximations per 1K tokens for telemetry)
    COST_FRONTIER_INPUT_1K: float = 0.0050
    COST_FRONTIER_OUTPUT_1K: float = 0.0150
    COST_SLM_INPUT_1K: float = 0.0001
    COST_SLM_OUTPUT_1K: float = 0.0003

settings = Settings()
