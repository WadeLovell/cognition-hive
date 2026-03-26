"""
CognitionHive: An Epistemic-First Multi-Agent Architecture

Main entry point. Orchestrates the full pipeline:
Router -> Scout -> Verifier -> Operator -> Archivist -> Warden
"""

import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime, timezone

from agents import (
    RouterAgent, ScoutAgent, VerifierAgent,
    OperatorAgent, ArchivistAgent, WardenAgent,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("cognition-hive")


def load_config(config_dir: Path) -> dict:
    policies_path = config_dir / "tool_policies.yaml"
    if not policies_path.exists():
        logger.warning(f"No config at {policies_path}, using defaults")
        return {"agents": {}, "thresholds": {
            "verification": {
                "minimum_confidence_to_proceed": 0.7,
                "minimum_confidence_with_caveats": 0.5,
                "force_re_retrieval_below": 0.3,
            },
            "warden": {"max_tool_violations_before_halt": 3},
        }}
    with open(policies_path) as f:
        return yaml.safe_load(f)


def init_agents(config: dict, include_warden: bool = False) -> dict:
    agent_configs = config.get("agents", {})
    thresholds = config.get("thresholds", {})

    agents = {
        "router": RouterAgent(agent_configs.get("router", {}), thresholds),
        "scout": ScoutAgent(agent_configs.get("scout", {}), thresholds),
        "verifier": VerifierAgent(agent_configs.get("verifier", {}), thresholds),
        "operator": OperatorAgent(agent_configs.get("operator", {}), thresholds),
        "archivist": ArchivistAgent(agent_configs.get("archivist", {}), thresholds),
    }

    if include_warden:
        agents["warden"] = WardenAgent(
            agent_configs.get("warden", {}),
            thresholds,
            monitored_agents=agents,
        )
        logger.info("Warden agent active")

    return agents


def process_request(request: str, agents: dict) -> dict:
    """
    Full pipeline: Router -> Scout -> Verifier -> Operator -> Archivist -> Warden

    Returns a dict with the result and all pipeline metadata.
    """
    session_id = f"s_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"[{session_id}] Processing: {request[:80]}...")

    # 1. Route
    route = agents["router"].classify(request, session_id)

    # 2. Retrieve
    scout_output = agents["scout"].retrieve(
        query=route["refined_query"],
        category=route["category"],
        session_id=session_id,
    )

    # 3. Verify (does NOT see the original request)
    verification = agents["verifier"].verify(
        claims=scout_output.get("claims", []),
        sources=scout_output.get("sources", []),
        session_id=session_id,
    )

    # 4. Re-retrieve if needed
    if verification["recommendation"] == "re_retrieve":
        re_queries = verification.get("re_retrieval_queries", [route["refined_query"]])
        logger.info(f"[{session_id}] Re-retrieving with: {re_queries}")
        scout_output = agents["scout"].retrieve(
            query=re_queries,
            category=route["category"],
            session_id=session_id,
            is_retry=True,
        )
        verification = agents["verifier"].verify(
            claims=scout_output.get("claims", []),
            sources=scout_output.get("sources", []),
            session_id=session_id,
        )

    # 5. Execute
    result = agents["operator"].execute(
        verified_claims=verification.get("claims", []),
        verification_report=verification,
        category=route["category"],
        original_request=request,
        session_id=session_id,
    )

    # 6. Archive
    agents["archivist"].record(
        request_summary=request,
        category=route["category"],
        evidence=scout_output,
        verification=verification,
        result=result,
        session_id=session_id,
    )

    # 7. Warden review (if active)
    if "warden" in agents:
        agents["warden"].review_session(
            session_id=session_id,
            verification_report=verification,
            result=result,
        )

    # Attach pipeline metadata to result for API consumers
    claims_found = len(scout_output.get("claims", []))
    overall_confidence = verification.get("overall_confidence", 0)
    recommendation = verification.get("recommendation", "unknown")

    result["_category"] = route["category"]
    result["_claims_found"] = claims_found
    result["_verification_confidence"] = overall_confidence
    result["_verification_recommendation"] = recommendation

    # Print summary (for CLI usage)
    print(f"\n{'='*60}")
    print(f"Session: {session_id}")
    print(f"Category: {route['category']}")
    print(f"Claims found: {claims_found}")
    print(f"Verification: {recommendation} "
          f"(confidence: {overall_confidence:.2f})")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"{'='*60}")

    if result.get("output"):
        print(f"\nOutput:\n{result['output'][:2000]}")

    if result.get("caveats"):
        print(f"\nCaveats: {result['caveats']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="CognitionHive")
    parser.add_argument(
        "--agents", choices=["core", "all"], default="core",
        help="'core' = 5 agents; 'all' = adds Warden",
    )
    parser.add_argument(
        "--config-dir", type=Path, default=Path("config"),
    )
    parser.add_argument(
        "--request", type=str, required=True,
        help="Request to process",
    )
    args = parser.parse_args()

    config = load_config(args.config_dir)
    agents = init_agents(config, include_warden=(args.agents == "all"))

    logger.info(f"Agents: {', '.join(agents.keys())}")
    process_request(args.request, agents)


if __name__ == "__main__":
    main()
