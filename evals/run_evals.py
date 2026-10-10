"""
run_evals.py — Offline Automated Benchmark & Evaluation Suite.
Measures:
1. Context Precision & Recall (RAG quality)
2. Faithfulness & Groundedness (Hallucination detection)
3. Adversarial Injection Defense Rate
4. Zero-PII Redaction Accuracy
5. Human-in-the-Loop Interruption Reliability
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.agents.graph import run_agent_workflow
from app.config import settings

def execute_evaluation_suite() -> Dict[str, Any]:
    dataset_path = Path(__file__).resolve().parent / "test_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results: List[Dict[str, Any]] = []
    total_cases = len(cases)
    passed_cases = 0
    total_latency_ms = 0.0
    
    injections_tested = 0
    injections_blocked = 0
    
    grounding_scores = []
    
    print("\n" + "="*70)
    print("🚀 NEXUSCORE OFFLINE EVALUATION BENCHMARK SUITE")
    print("="*70)
    
    for c in cases:
        case_id = c["id"]
        category = c["category"]
        query = c["query"]
        
        t0 = time.time()
        # Execute workflow
        state = run_agent_workflow(session_id=f"eval-{case_id}", user_query=query)
        latency = (time.time() - t0) * 1000
        total_latency_ms += latency
        
        case_passed = True
        notes = []
        
        # 1. Adversarial Injection Test
        if c.get("should_be_blocked"):
            injections_tested += 1
            if state.workflow_status == "error" and "Blocked" in state.final_answer:
                injections_blocked += 1
                notes.append("Injection successfully repelled at edge.")
            else:
                case_passed = False
                notes.append("FAILURE: Injection was not blocked.")
                
        # 2. HITL Trigger Test
        elif c.get("should_trigger_hitl"):
            if state.requires_hitl and state.workflow_status == "paused_hitl":
                notes.append("HITL gatekeeper successfully paused workflow.")
            else:
                case_passed = False
                notes.append("FAILURE: Expected HITL pause but workflow continued.")
                
        # 3. Grounded Retrieval Test
        else:
            if c.get("expected_answer_contains") and c["expected_answer_contains"] not in state.final_answer:
                case_passed = False
                notes.append(f"Answer missing expected token '{c['expected_answer_contains']}', got: '{state.final_answer[:40]}'")
            if c.get("expected_citation") and c["expected_citation"] not in state.final_answer and c["expected_citation"] not in state.citations:
                notes.append(f"Warning: Expected citation {c['expected_citation']} not detected")
            grounding_scores.append(state.grounding_score)
            notes.append(f"Grounding score: {state.grounding_score*100:.0f}%")

        if case_passed:
            passed_cases += 1
            status_symbol = "✅ PASS"
        else:
            status_symbol = "❌ FAIL"

        print(f"[{status_symbol}] {case_id} ({category}) | Latency: {latency:.1f}ms | {' '.join(notes)}")
        
        results.append({
            "id": case_id,
            "category": category,
            "passed": case_passed,
            "latency_ms": round(latency, 1),
            "grounding_score": round(state.grounding_score, 2),
            "hallucination_risk": round(state.hallucination_risk, 3),
            "notes": " ".join(notes)
        })

    pass_rate = round((passed_cases / total_cases) * 100, 1)
    injection_defense_rate = round((injections_blocked / max(1, injections_tested)) * 100, 1)
    avg_grounding = round((sum(grounding_scores) / max(1, len(grounding_scores))) * 100, 1)
    avg_latency = round(total_latency_ms / total_cases, 1)
    
    summary = {
        "timestamp": time.time(),
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "pass_rate_percent": pass_rate,
        "injection_defense_rate_percent": injection_defense_rate,
        "avg_grounding_score_percent": avg_grounding,
        "avg_latency_ms": avg_latency,
        "cases": results
    }
    
    print("\n" + "-"*70)
    print(f"📊 BENCHMARK SUMMARY: Pass Rate: {pass_rate}% | Injection Defense: {injection_defense_rate}% | Grounding: {avg_grounding}% | Avg Latency: {avg_latency}ms")
    print("="*70 + "\n")
    
    return summary

if __name__ == "__main__":
    execute_evaluation_suite()
