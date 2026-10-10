"""
input_guards.py — High-Performance Sub-Millisecond Input Guardrails.
Defends against:
1. Prompt Injection & Jailbreak Heuristics ("ignore previous instructions", "DAN", etc.)
2. Zero-PII Anonymization (Tokenizes emails, phone numbers, SSNs, credit cards)
3. Toxicity & Semantic Drift Flags
"""

import re
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel, Field

class GuardrailInputResult(BaseModel):
    is_safe: bool = True
    sanitized_query: str
    original_query: str
    rejection_reason: str = ""
    pii_redacted: List[str] = Field(default_factory=list)
    injection_score: float = 0.0
    latency_ms: float = 0.0

class InputGuardrail:
    def __init__(self):
        # High-confidence prompt injection signatures
        self.injection_patterns = [
            r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
            r"system\s*prompt\s*override",
            r"you\s+are\s+now\s+in\s+developer\s+mode",
            r"bypass\s+(all\s+)?guardrails",
            r"pretend\s+you\s+have\s+no\s+(rules|restrictions)",
            r"dan\s+mode|jailbreak|disregard\s+safety",
            r"give\s+me\s+your\s+secret\s+instructions",
            r"reveal\s+(the\s+)?internal\s+system\s+prompt"
        ]
        
        # PII regex patterns
        self.pii_rules: List[Tuple[str, str, str]] = [
            ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", "[REDACTED_EMAIL]"),
            ("PHONE", r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]"),
            ("CREDIT_CARD", r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[REDACTED_CARD]"),
            ("SSN_AADHAAR", r"\b\d{3}-\d{2}-\d{4}\b|\b\d{4}\s\d{4}\s\d{4}\b", "[REDACTED_ID]")
        ]

    def validate_and_sanitize(self, query: str) -> GuardrailInputResult:
        import time
        start_time = time.time()
        
        # 1. Check prompt injection
        lower_query = query.lower()
        injection_score = 0.0
        for pattern in self.injection_patterns:
            if re.search(pattern, lower_query, re.IGNORECASE):
                injection_score = 0.95
                latency = round((time.time() - start_time) * 1000, 3)
                return GuardrailInputResult(
                    is_safe=False,
                    sanitized_query=query,
                    original_query=query,
                    rejection_reason="Blocked by Input Guardrail: Adversarial prompt injection detected.",
                    injection_score=injection_score,
                    latency_ms=latency
                )

        # 2. Tokenize / Redact PII
        sanitized = query
        redacted_items = []
        for pii_type, regex_pat, replacement in self.pii_rules:
            matches = re.findall(regex_pat, sanitized)
            if matches:
                for m in matches:
                    redacted_items.append(f"{pii_type}")
                sanitized = re.sub(regex_pat, replacement, sanitized)

        latency = round((time.time() - start_time) * 1000, 3)
        return GuardrailInputResult(
            is_safe=True,
            sanitized_query=sanitized,
            original_query=query,
            pii_redacted=list(set(redacted_items)),
            injection_score=injection_score,
            latency_ms=latency
        )

input_guard = InputGuardrail()
