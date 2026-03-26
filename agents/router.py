"""
Router Agent: Classifies incoming requests and routes to the correct pipeline.
Uses Haiku for speed and cost efficiency on a simple classification task.
"""

import logging
from agents.base import BaseAgent
from agents.client import call_claude, parse_json_response

logger = logging.getLogger("cognition-hive.router")

SYSTEM_PROMPT = """You are the Router agent in CognitionHive. Your only job is to classify an incoming request and produce a routing decision.

Classify into exactly ONE category:
- research: finding information, comparing options, literature review
- writing: drafting documents, emails, reports, manuscripts
- monitoring: watching for changes, alerts, recurring checks
- execution: taking action, sending messages, creating deliverables
- scheduling: time-based planning, reminders, calendar
- procurement: evaluating purchases, comparing vendors/products
- follow_up: continuing a previous request, checking status

Respond with ONLY valid JSON, no other text:
{
  "category": "one_category",
  "refined_query": "a clear, specific version of the request optimized for retrieval",
  "priority": "normal"
}"""


class RouterAgent(BaseAgent):

    def classify(self, request: str, session_id: str) -> dict:
        response = call_claude(
            agent_name="router",
            system_prompt=SYSTEM_PROMPT,
            user_message=request,
            max_tokens=256,
        )
        result = parse_json_response(response)
        result["session_id"] = session_id
        logger.info(f"[{session_id}] Routed as: {result['category']}")
        return result
