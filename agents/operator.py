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

SYSTEM_PROMPT = """You are the Operator agent in CognitionHive. You produce deliverables based on VERIFIED claims only.

RULES:
1. You receive a verification report. ONLY use claims with verdict "supported" or "partially_supported".
2. For "partially_supported" claims, include the Verifier's caveats.
3. NEVER include claims marked "unsupported" or "contradicted" without explicitly flagging them.
4. If the recommendation is "proceed_with_caveats", include a caveats section in your output.
5. If the recommendation is "halt", refuse to produce output.

Your output should be clear, actionable, and honest about what is well-supported vs uncertain.

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
            logger.warning(f"[{session_id}] Operator: halt recommendation, refusing")
            return {
                "session_id": session_id,
                "status": "halted",
                "reason": "Verification confidence too low to proceed",
                "output": None,
            }

        report_text = json.dumps(verification_report, indent=2)

        user_message = f"""Original request: {original_request}
Category: {category}
Verification recommendation: {recommendation}

VERIFICATION REPORT:
{report_text}

Produce the appropriate deliverable based on the verified claims. Follow your rules strictly."""

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
            # If the Operator returned plain text, wrap it
            result = {
                "status": "completed",
                "output": response,
                "caveats": verification_report.get("open_questions", []),
            }

        result["session_id"] = session_id
        return result
