"""Structured audit logging for all agent decisions."""
import json
import os
from datetime import datetime
from typing import Any, Dict


class DecisionLogger:
    """Writes a machine-readable audit trail in JSONL format.

    Every agent action, rule evaluation, LLM call, and debate round
    is logged as a separate JSON line with a timestamp, making the
    entire decision process fully traceable and replayable.
    """

    def __init__(self, log_dir: str = "./audit_logs"):
        """Initialise the logger.

        Args:
            log_dir: Directory where log files will be written.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"underwriting_{self.session_id}.jsonl")

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a single event.

        Args:
            event_type: Category of event (e.g., "rule_evaluation", "agent_assessment",
                        "debate_round", "llm_call", "final_decision").
            data: Dictionary of event data to log.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "event_type": event_type,
            **data,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_rule_evaluation(
        self, agent_name: str, rule_id: str, matched: bool, details: str = ""
    ) -> None:
        """Log a single rule evaluation."""
        self.log_event("rule_evaluation", {
            "agent": agent_name,
            "rule_id": rule_id,
            "matched": matched,
            "details": details,
        })

    def log_agent_assessment(self, assessment: Any) -> None:
        """Log a complete agent assessment."""
        self.log_event("agent_assessment", assessment.dict() if hasattr(assessment, "dict") else assessment)

    def log_debate_round(self, round_data: Dict) -> None:
        """Log a debate round."""
        self.log_event("debate_round", round_data)

    def log_final_decision(self, decision: Dict) -> None:
        """Log the final underwriting decision."""
        self.log_event("final_decision", decision)

    def log_llm_call(self, agent_name: str, prompt: str, response: str) -> None:
        """Log an LLM interaction for audit purposes."""
        self.log_event("llm_call", {
            "agent": agent_name,
            "prompt_summary": prompt[:200],
            "response_summary": response[:200],
        })
