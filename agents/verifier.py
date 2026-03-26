"""
Verifier Agent: Independent verification and confidence scoring.

The architectural differentiator of CognitionHive.

Design decisions:
1. Does NOT see the original request — only claims and sources (reduces confirmation bias)
2. Scores each claim independently with explicit evidence mapping
3. Can force re-retrieval but cannot retrieve on its own
4. Output (verification report) is required input for the Operator
"""

import json
import logging
from agents.base import BaseAgent
from agents.client import call_claude, parse_json_response

logger = logging.getLogger("cognition-hive.verifier")

SYSTEM_PROMPT = """You are the Verifier agent in CognitionHive. You independently assess claims for accuracy and source support.

CRITICAL: You do NOT know what was originally asked. You only see claims and their sources. This is by design to reduce confirmation bias. Assess each claim solely on whether the provided sources support it.

For each claim, determine:
1. VERDICT: supported | partially_supported | unsupported | contradicted | unverifiable
2. CONFIDENCE: 0.0 to 1.0 (would you bet money on this claim at these odds?)
3. FLAGS: zero or more from this list:
   - no_primary_source
   - single_source_only
   - source_outdated
   - claim_overstated
   - hedging_removed
   - causal_claim_from_correlation
   - numeric_unverified
   - attribution_missing
   - scope_broader_than_evidence
4. SUGGESTED_REVISION: if partially_supported or unsupported, rewrite the claim to match the evidence

After assessing all claims, determine an overall recommendation:
- proceed: overall confidence >= 0.7
- proceed_with_caveats: overall confidence 0.5 to 0.7
- re_retrieve: overall confidence 0.3 to 0.5 (include re_retrieval_queries)
- halt: overall confidence < 0.3

Respond with ONLY valid JSON:
{
  "claims": [
    {
      "claim_id": "c1",
      "claim_text": "the claim as stated",
      "verdict": "supported",
      "confidence": 0.85,
      "flags": [],
      "evidence_summary": "Source X directly states...",
      "suggested_revision": null
    }
  ],
  "overall_confidence": 0.82,
  "recommendation": "proceed",
  "open_questions": ["Any unresolved questions"],
  "re_retrieval_queries": []
}

Be ruthlessly honest. A confidently wrong answer is far more dangerous than an honest 'I cannot verify this.'"""


class VerifierAgent(BaseAgent):

    def __init__(self, config: dict, thresholds: dict):
        super().__init__(config, thresholds)
        verification = thresholds.get("verification", {})
        self.min_proceed = verification.get("minimum_confidence_to_proceed", 0.7)
        self.min_caveats = verification.get("minimum_confidence_with_caveats", 0.5)
        self.min_re_retrieve = verification.get("force_re_retrieval_below", 0.3)

    def verify(self, claims: list, sources: list, session_id: str) -> dict:
        # Build the input for the Verifier — deliberately excludes original query
        claims_text = json.dumps(claims, indent=2)
        sources_text = json.dumps(sources, indent=2)

        user_message = f"""Assess the following claims against the provided sources.

CLAIMS:
{claims_text}

SOURCES:
{sources_text}

Verify each claim independently. Return your assessment as JSON."""

        logger.info(
            f"[{session_id}] Verifier assessing {len(claims)} claims "
            f"against {len(sources)} sources"
        )

        response = call_claude(
            agent_name="verifier",
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=4096,
        )

        try:
            result = parse_json_response(response)
        except (json.JSONDecodeError, ValueError):
            logger.error(f"[{session_id}] Verifier returned unparseable response")
            result = {
                "claims": [],
                "overall_confidence": 0.0,
                "recommendation": "halt",
                "open_questions": ["Verifier failed to produce structured output"],
                "re_retrieval_queries": [],
            }

        # Add metadata
        result["report_id"] = f"vr_{session_id}"
        result["scout_session_id"] = session_id
        result["original_query_visible"] = False

        logger.info(
            f"[{session_id}] Verification: {result['recommendation']} "
            f"(confidence: {result.get('overall_confidence', 0):.2f})"
        )

        return result
