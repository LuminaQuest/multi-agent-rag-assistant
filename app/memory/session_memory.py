"""
session_memory.py — Cross-Session Persistent Agent Memory.
Persists multi-turn conversations, state snapshots, and cognitive summaries using SQLite.
Enables agents to maintain context across sessions without context-window overflow.
"""

import json
import sqlite3
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.config import settings

class SessionMessage(BaseModel):
    message_id: str
    session_id: str
    role: str  # "user", "assistant", "system"
    content: str
    citations: List[str] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)

class SessionMemory:
    def __init__(self, db_path=settings.DB_PATH):
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_metadata (
                    session_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_active REAL NOT NULL
                )
            """)
            conn.commit()

    def add_message(self, session_id: str, role: str, content: str, citations: List[str] = None) -> str:
        import uuid
        msg_id = str(uuid.uuid4())[:8]
        citations_json = json.dumps(citations or [])
        now = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages (message_id, session_id, role, content, citations, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (msg_id, session_id, role, content, citations_json, now))
            
            cursor.execute("""
                INSERT INTO session_metadata (session_id, summary, created_at, last_active)
                VALUES (?, '', ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET last_active = ?
            """, (session_id, now, now, now))
            conn.commit()
            
        return msg_id

    def get_history(self, session_id: str, limit: int = 10) -> List[SessionMessage]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT message_id, session_id, role, content, citations, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
            """, (session_id, limit))
            rows = cursor.fetchall()
            
        messages = []
        for r in rows:
            messages.append(SessionMessage(
                message_id=r[0],
                session_id=r[1],
                role=r[2],
                content=r[3],
                citations=json.loads(r[4]),
                timestamp=r[5]
            ))
        return messages

    def clear_session(self, session_id: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM session_metadata WHERE session_id = ?", (session_id,))
            conn.commit()

session_memory = SessionMemory()
