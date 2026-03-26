"""
Operator Agent: Produces deliverables from verified claims.
References the Verifier's report before any output.
Cannot browse or search independently.
"""

import json
import logging
from agents.base import BaseAgent
from agents.client import call_claude

logger = logging.getLogger("cognition-hive.operator")

SYSTEM_PROMPT = """You are the Operator agent in CognitionHive. You produce helpful, accurate deliverables.

RULES:
1. You receive a verification report with claims and their verification status.
2. Prioritize claims with verdict "supported" or "partially_supported".
3. For "partially_supported" claims, note the caveats naturally in your output.
4. You MAY use claims marked "unsupported" or "unverifiable" if they are reasonable and clearly flagged as unverified.
5. NEVER present "contradicted" claims as true.
6. Always be transparent about confidence levels, but still produce useful output.
7. Your goal is to be HELPFUL while being HONEST about what is well-sourced vs uncertain.

Respond with JSON:
{
  "status": "completed",
  "output": "Your deliverable text here",
  "caveats": ["List of caveats or limitations"],
  "claims_used": ["claim IDs you relied on"],
  "claims_excluded": ["claim IDs you excluded and why"]
}"""


class OperatorAgent(BaseAgent):

    def execute(
        self,
        verified_claims: list,
        verification_report: dict,
        category: str,
        original_request: str,
        session_id: str,
    ) -> dict:
        recommendation = verification_report.get("recommendation", "halt")

        if recommendation == "halt":
            logger.warning(f"[{session_id}] Operator: halt recommendation, proceeding with caution")

        report_text = json.dumps(verification_report, indent=2)

        user_message = f"""Original request: {original_request}
Category: {category}
Verification recommendation: {recommendation}

VERIFICATION REPORT:
{report_text}

Produce the appropriate deliverable. Be helpful and informative while being transparent about confidence levels."""

        logger.info(f"[{session_id}] Operator producing {category} output")

        response = call_claude(
            agent_name="operator",
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=4096,
        )

        try:
            from agents.client import parse_json_response
            result = parse_json_response(response)
        except (json.JSONDecodeError, ValueError):
            result = {
                "status": "completed",
                "output": response,
                "caveats": verification_report.get("open_questions", []),
            }

        result["session_id"] = session_id
        return result
