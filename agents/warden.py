"""
Warden Agent: Policy enforcement and audit.
Pure observer. Can read everything, modify nothing.
Add only after the core five agents are stable.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from agents.base import BaseAgent

logger = logging.getLogger("cognition-hive.warden")

LOG_DIR = Path("logs")


class WardenAgent(BaseAgent):

    def __init__(self, config: dict, thresholds: dict, monitored_agents: dict = None):
        super().__init__(config, thresholds)
        self.monitored_agents = monitored_agents or {}
        self.max_violations = thresholds.get("warden", {}).get(
            "max_tool_violations_before_halt", 3
        )
        self.violation_count = 0
        LOG_DIR.mkdir(exist_ok=True)
        self.log_path = LOG_DIR / "warden_audit.jsonl"

    def review_session(
        self,
        session_id: str,
        verification_report: dict,
        result: dict,
    ) -> dict:
        violations = []

        # Check 1: Did verification happen?
        if not verification_report.get("report_id"):
            violations.append({
                "type": "verification_bypassed",
                "severity": "critical",
                "detail": "No verification report found for this session",
            })

        # Check 2: Did Operator act despite halt recommendation?
        if (
            verification_report.get("recommendation") == "halt"
            and result.get("status") == "completed"
        ):
            violations.append({
                "type": "execution_without_verification",
                "severity": "critical",
                "detail": "Operator produced output despite halt recommendation",
            })

        # Check 3: Were unsupported claims used?
        unsupported = [
            c for c in verification_report.get("claims", [])
            if c.get("verdict") in ("unsupported", "contradicted")
        ]
        output_text = str(result.get("output", ""))
        for claim in unsupported:
            claim_text = claim.get("claim_text", "")
            # Simple check: see if claim text appears in output
            if claim_text and claim_text.lower() in output_text.lower():
                violations.append({
                    "type": "unsupported_claim_in_output",
                    "severity": "violation",
                    "detail": f"Claim '{claim.get('claim_id')}' is {claim['verdict']} but may appear in output",
                })

        # Check 4: Low confidence proceeding
        confidence = verification_report.get("overall_confidence", 0)
        recommendation = verification_report.get("recommendation", "halt")
        if confidence < 0.5 and recommendation in ("proceed", "proceed_with_caveats"):
            violations.append({
                "type": "confidence_below_threshold",
                "severity": "warning",
                "detail": f"Proceeded with confidence {confidence:.2f}",
            })

        entry = {
            "log_id": f"audit_{session_id}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "violations_found": len(violations),
            "violations": violations,
            "session_compliant": len(violations) == 0,
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        if violations:
            self.violation_count += len(violations)
            for v in violations:
                logger.warning(f"[{session_id}] Warden: {v['type']} ({v['severity']})")
            if self.violation_count >= self.max_violations:
                logger.critical(
                    f"Warden: {self.violation_count} cumulative violations, "
                    f"threshold {self.max_violations} exceeded"
                )
        else:
            logger.info(f"[{session_id}] Warden: session compliant")

        return entry
