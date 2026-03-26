"""
Shared Anthropic API client for all CognitionHive agents.
Centralizes model selection, error handling, and token tracking.
"""

import json
import time
import logging
import anthropic

logger = logging.getLogger("cognition-hive.client")

MODELS = {
    "fast": "claude-haiku-4-5-20251001",
    "balanced": "claude-sonnet-4-6",
}

AGENT_MODEL_MAP = {
    "router": "fast",
    "scout": "balanced",
    "verifier": "balanced",
    "operator": "balanced",
    "archivist": "fast",
    "warden": "fast",
}

client = anthropic.Anthropic()


def call_claude(
    agent_name: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    model_override: str = None,
    tools: list = None,
) -> str:
    """
    Send a message to Claude and return the response text.
    For tool-use responses, returns the full content blocks as JSON.
    """
    model_tier = AGENT_MODEL_MAP.get(agent_name, "balanced")
    model = model_override or MODELS[model_tier]

    logger.info(f"[{agent_name}] Calling {model}")

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools

    try:
        time.sleep(65)  # Rate limit buffer - wait for token window to reset
        message = client.messages.create(**kwargs)

        logger.info(
            f"[{agent_name}] {message.usage.input_tokens} in, "
            f"{message.usage.output_tokens} out"
        )

        # If the response contains tool use, return all content blocks
        if any(block.type == "tool_use" for block in message.content):
            return message

        # Otherwise return the text
        text_blocks = [
            block.text for block in message.content if block.type == "text"
        ]
        return "\n".join(text_blocks)

    except anthropic.APIError as e:
        logger.error(f"[{agent_name}] API error: {e}")
        raise


def parse_json_response(text: str) -> dict:
    """
    Parse JSON from Claude's response, stripping markdown fences if present.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(cleaned)
