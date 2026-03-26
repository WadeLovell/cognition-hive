"""
Archivist Agent: Writes structured memory for each request lifecycle.
Uses Haiku for cost efficiency on summarization tasks.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from agents.base import BaseAgent
from agents.client import call_claude, parse_json_response

logger = logging.getLogger("cognition-hive.archivist")

MEMORY_DIR = Path("memory")

SYSTEM_PROMPT = """You are the Archivist agent in CognitionHive. You create concise, structured memory entries that capture the full lifecycle of a request.

Given the request, evidence, verification, and result, produce a JSON memory entry:
{
  "summary": "One-sentence summary of what happened",
  "key_findings": ["Most important verified facts"],
  "decisions_made": ["What was decided and why"],
  "open_questions": ["What remains unresolved"],
  "reusable_artifacts": ["Citations, contacts, specs worth keeping"]
}

Be concise. This memory will be retrieved later to inform future requests."""


class ArchivistAgent(BaseAgent):

    def __init__(self, config: dict, thresholds: dict):
        super().__init__(config, thresholds)
        MEMORY_DIR.mkdir(exist_ok=True)

    def record(
        self,
        request_summary: str,
        category: str,
        evidence: dict,
        verification: dict,
        result: dict,
        session_id: str,
    ) -> dict:
        user_message = f"""Record this request lifecycle:

REQUEST: {request_summary}
CATEGORY: {category}
CLAIMS FOUND: {len(evidence.get('claims', []))}
VERIFICATION: {verification.get('recommendation', 'unknown')} (confidence: {verification.get('overall_confidence', 0):.2f})
RESULT STATUS: {result.get('status', 'unknown')}
OUTPUT PREVIEW: {str(result.get('output', ''))[:500]}
OPEN QUESTIONS: {json.dumps(verification.get('open_questions', []))}

Produce the memory entry as JSON."""

        response = call_claude(
            agent_name="archivist",
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=1024,
        )

        try:
            memory_content = parse_json_response(response)
        except (json.JSONDecodeError, ValueError):
            memory_content = {"summary": request_summary, "raw_response": response}

        entry = {
            "entry_id": f"mem_{session_id}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_summary": request_summary,
            "category": category,
            "status": result.get("status", "unknown"),
            "verification_confidence": verification.get("overall_confidence", 0),
            "verification_recommendation": verification.get("recommendation", "unknown"),
            "memory": memory_content,
        }

        memory_path = MEMORY_DIR / f"{session_id}.json"
        with open(memory_path, "w") as f:
            json.dump(entry, f, indent=2)

        logger.info(f"[{session_id}] Archivist recorded: {memory_path}")
        return {"entry_id": entry["entry_id"], "path": str(memory_path)}
