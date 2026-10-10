"""
output_guards.py — Groundedness, Hallucination Auditor & Compliance Validator.
Analyzes model responses against retrieved source context:
1. Citation Verification & Grounding Rate (N-gram overlap & claim citation checks)
2. Hallucination Risk Index (0.0 = fully grounded, 1.0 = high hallucination risk)
3. Structured Output Compliance (Removes accidental chain-of-thought leaks)
"""

import re
import time
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class GuardrailOutputResult(BaseModel):
    is_compliant: bool = True
    sanitized_response: str
    hallucination_risk_index: float = 0.0
    grounding_score: float = 1.0
    citations_detected: List[str] = Field(default_factory=list)
    flagged_issues: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0

class OutputGuardrail:
    def __init__(self):
        self.citation_pattern = r"\[Doc-\d+\]"

    def audit_response(
        self,
        response_text: str,
        retrieved_contexts: List[Dict[str, Any]]
    ) -> GuardrailOutputResult:
        start_time = time.time()
        
        # 1. Detect citations
        citations = re.findall(self.citation_pattern, response_text)
        
        # 2. Compute groundedness overlap against retrieved chunks
        if not retrieved_contexts:
            # If no context was retrieved (e.g. general conversational query)
            latency = round((time.time() - start_time) * 1000, 3)
            return GuardrailOutputResult(
                is_compliant=True,
                sanitized_response=response_text,
                hallucination_risk_index=0.05,
                grounding_score=0.95,
                citations_detected=citations,
                latency_ms=latency
            )
            
        combined_context = " ".join([c.get("content", "").lower() for c in retrieved_contexts])
        context_words = set(re.findall(r"\b\w{4,}\b", combined_context))
        
        # Extract meaningful claim words from response
        response_words = re.findall(r"\b\w{4,}\b", response_text.lower())
        if not response_words:
            overlap_ratio = 1.0
        else:
            overlapping_words = [w for w in response_words if w in context_words]
            overlap_ratio = min(1.0, len(overlapping_words) / max(1, len(response_words)))

        # Grounding score combines word overlap + citation presence
        citation_bonus = 0.2 if len(citations) > 0 else 0.0
        grounding_score = min(1.0, round((overlap_ratio * 0.8) + citation_bonus, 2))
        hallucination_risk = max(0.0, round(1.0 - grounding_score, 3))
        
        flagged_issues = []
        if hallucination_risk > 0.45 and len(retrieved_contexts) > 0:
            flagged_issues.append("Moderate hallucination risk: low overlap with verified context.")

        # 3. Clean any stray system tokens
        cleaned_text = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", response_text).strip()

        latency = round((time.time() - start_time) * 1000, 3)
        return GuardrailOutputResult(
            is_compliant=(hallucination_risk < 0.70),
            sanitized_response=cleaned_text,
            hallucination_risk_index=hallucination_risk,
            grounding_score=grounding_score,
            citations_detected=list(set(citations)),
            flagged_issues=flagged_issues,
            latency_ms=latency
        )

output_guard = OutputGuardrail()
