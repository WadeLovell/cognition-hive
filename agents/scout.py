"""
Scout Agent: Retrieves source material using web search and extracts
discrete claims with source references for the Verifier.

Uses Sonnet for strong retrieval and structured extraction.
Has web_search tool access. Cannot execute or modify anything.
"""

import json
import logging
from agents.base import BaseAgent
from agents.client import client, MODELS, AGENT_MODEL_MAP

logger = logging.getLogger("cognition-hive.scout")

SYSTEM_PROMPT = """You are the Scout agent in CognitionHive. Your job is to retrieve source material and extract discrete, verifiable claims.

RULES:
1. Use the web_search tool to find relevant sources.
2. After searching, extract DISCRETE CLAIMS from what you found. Each claim should be a single factual assertion.
3. Link every claim to its source URL.
4. Do NOT editorialize, synthesize, or add your own analysis. Just extract what the sources say.
5. Do NOT assess whether claims are true. That is the Verifier's job.

After you have gathered sources and extracted claims, respond with JSON in this exact format:
{
  "claims": [
    {
      "claim_id": "c1",
      "claim_text": "The specific factual claim",
      "source_url": "https://...",
      "source_title": "Title of the source"
    }
  ],
  "sources": [
    {
      "url": "https://...",
      "title": "Source title",
      "retrieved_snippet": "Brief relevant excerpt"
    }
  ],
  "queries_executed": ["list of search queries you ran"]
}"""


class ScoutAgent(BaseAgent):

    def retrieve(self, query: str, category: str, session_id: str, is_retry: bool = False) -> dict:
        queries = query if isinstance(query, list) else [query]
        user_message = f"Find information about: {'; '.join(queries)}"
        if is_retry:
            user_message = f"RETRY with refined queries. Previous results were insufficient. Search for: {'; '.join(queries)}"

        logger.info(f"[{session_id}] Scout searching: {queries}")

        model_tier = AGENT_MODEL_MAP.get("scout", "balanced")
        model = MODELS[model_tier]

        # First call: let Claude search
        messages = [{"role": "user", "content": user_message}]
        tools = [{"type": "web_search_20250305", "name": "web_search"}]

        # Agentic loop: keep going until Claude stops using tools
        while True:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=tools,
                temperature=0.0,
            )

            logger.info(
                f"[{session_id}] Scout: {response.usage.input_tokens} in, "
                f"{response.usage.output_tokens} out, stop={response.stop_reason}"
            )

            # If Claude is done (no more tool use), extract the final text
            if response.stop_reason == "end_turn":
                text_blocks = [
                    block.text for block in response.content
                    if block.type == "text"
                ]
                full_text = "\n".join(text_blocks)

                try:
                    # Try to parse structured JSON from the response
                    result = self._parse_scout_output(full_text)
                except (json.JSONDecodeError, ValueError):
                    # If Claude didn't return structured JSON, wrap the text
                    logger.warning(f"[{session_id}] Scout returned unstructured text, wrapping")
                    result = {
                        "claims": [{"claim_id": "c1", "claim_text": full_text, "source_url": "", "source_title": ""}],
                        "sources": [],
                        "queries_executed": queries,
                    }

                result["session_id"] = session_id
                result["is_retry"] = is_retry
                logger.info(f"[{session_id}] Scout extracted {len(result['claims'])} claims")
                return result

            # If Claude wants to use tools, add the response and continue
            messages.append({"role": "assistant", "content": response.content})

            # Process tool results
            tool_results = []
            for block in response.content:
                if block.type == "server_tool_use":
                    # Server-side tools (like web_search) are handled automatically
                    # by the API - results come back in the next response
                    pass

            # For server-side tools, just continue the loop
            # The API handles web_search results internally
            if response.stop_reason == "tool_use":
                # For server-side tools, results are injected automatically
                # We just need to continue the conversation
                continue
            else:
                break

        # Fallback
        return {
            "claims": [],
            "sources": [],
            "queries_executed": queries,
            "session_id": session_id,
            "is_retry": is_retry,
        }

    def _parse_scout_output(self, text: str) -> dict:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        # Find JSON object in the text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in Scout output")

        return json.loads(cleaned[start:end])
