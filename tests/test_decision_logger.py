"""Tests for DecisionLogger — structured JSONL audit logging."""

import json
import os
import tempfile
from typing import Any, Dict

import pytest

from underwriting.audit.logger import DecisionLogger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_log_dir() -> str:
    """Create and return a temporary directory for audit logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture()
def logger(temp_log_dir: str) -> DecisionLogger:
    """Return a DecisionLogger writing to a temporary directory."""
    return DecisionLogger(log_dir=temp_log_dir)


# ---------------------------------------------------------------------------
# Test: Directory and file creation
# ---------------------------------------------------------------------------


class TestLoggerCreation:
    """Tests for DecisionLogger initialisation."""

    def test_creates_log_directory(self, temp_log_dir: str) -> None:
        """DecisionLogger creates the log directory if it does not exist."""
        sub_dir = os.path.join(temp_log_dir, "nested", "logs")
        assert not os.path.exists(sub_dir)

        DecisionLogger(log_dir=sub_dir)
        assert os.path.isdir(sub_dir)

    def test_creates_log_file(self, temp_log_dir: str) -> None:
        """DecisionLogger creates the JSONL file on first write."""
        log = DecisionLogger(log_dir=temp_log_dir)
        log.log_event("test", {"key": "value"})

        assert os.path.isfile(log.log_path)
        assert log.log_path.endswith(".jsonl")

    def test_session_id_format(self, temp_log_dir: str) -> None:
        """Session ID follows YYYYMMDD_HHMMSS format."""
        log = DecisionLogger(log_dir=temp_log_dir)
        session_id = log.session_id
        assert len(session_id) == 15  # 8 + 1 + 6
        assert session_id[8] == "_"

    def test_log_path_contains_session_id(self, temp_log_dir: str) -> None:
        """Log file name embeds the session ID."""
        log = DecisionLogger(log_dir=temp_log_dir)
        assert log.session_id in log.log_path


# ---------------------------------------------------------------------------
# Test: log_event
# ---------------------------------------------------------------------------


class TestLogEvent:
    """Tests for the generic log_event method."""

    def test_writes_valid_jsonl_line(self, logger: DecisionLogger) -> None:
        """log_event writes a line that parses as valid JSON."""
        logger.log_event("test_event", {"field": "value"})
        assert os.path.isfile(logger.log_path)

        with open(logger.log_path) as f:
            lines = f.readlines()
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["event_type"] == "test_event"
        assert entry["field"] == "value"

    def test_includes_timestamp(self, logger: DecisionLogger) -> None:
        """Each entry contains an ISO-format timestamp."""
        logger.log_event("test_event", {"field": "value"})

        with open(logger.log_path) as f:
            entry = json.loads(f.readline())

        assert "timestamp" in entry
        assert "T" in entry["timestamp"]  # ISO format check

    def test_includes_session_id(self, logger: DecisionLogger) -> None:
        """Each entry contains the logger's session ID."""
        logger.log_event("test_event", {"field": "value"})

        with open(logger.log_path) as f:
            entry = json.loads(f.readline())

        assert entry["session_id"] == logger.session_id


# ---------------------------------------------------------------------------
# Test: log_agent_assessment
# ---------------------------------------------------------------------------


class TestLogAgentAssessment:
    """Tests for agent assessment logging."""

    def test_serializes_pydantic_assessment(self, logger: DecisionLogger) -> None:
        """Assessments with a .dict() method are serialised via dict()."""
        assessment = {
            "agent_name": "medical_agent",
            "risk_tier": "loading",
            "confidence_score": 0.85,
        }

        class MockAssessment:
            """Minimal object with a .dict() method."""
            def dict(self) -> Dict[str, Any]:
                return assessment

        logger.log_agent_assessment(MockAssessment())

        with open(logger.log_path) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "agent_assessment"
        assert entry["agent_name"] == "medical_agent"
        assert entry["risk_tier"] == "loading"

    def test_serializes_plain_dict(self, logger: DecisionLogger) -> None:
        """Plain dicts are logged directly."""
        data: Dict[str, Any] = {"agent_name": "finance_agent", "score": 42}
        logger.log_agent_assessment(data)

        with open(logger.log_path) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "agent_assessment"
        assert entry["agent_name"] == "finance_agent"
        assert entry["score"] == 42


# ---------------------------------------------------------------------------
# Test: log_final_decision
# ---------------------------------------------------------------------------


class TestLogFinalDecision:
    """Tests for final decision logging."""

    def test_writes_complete_decision(self, logger: DecisionLogger) -> None:
        """log_final_decision writes the full decision dict."""
        decision = {
            "applicant": "Alex Standard",
            "verdict": "approve",
            "premium_multiplier": 1.25,
            "reasoning": "Standard risk profile",
        }
        logger.log_final_decision(decision)

        with open(logger.log_path) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "final_decision"
        assert entry["applicant"] == "Alex Standard"
        assert entry["verdict"] == "approve"
        assert entry["premium_multiplier"] == 1.25

    def test_all_lines_valid_json(self, logger: DecisionLogger) -> None:
        """Every line in the log file parses as valid JSON."""
        decisions = [
            {"verdict": "approve"},
            {"verdict": "decline"},
            {"verdict": "load"},
        ]
        for d in decisions:
            logger.log_final_decision(d)

        with open(logger.log_path) as f:
            for line in f:
                json.loads(line)  # Raises if invalid


# ---------------------------------------------------------------------------
# Test: log_llm_call
# ---------------------------------------------------------------------------


class TestLogLlmCall:
    """Tests for LLM call logging."""

    def test_logs_prompt_and_response_summaries(self, logger: DecisionLogger) -> None:
        """log_llm_call stores truncated prompt and response summaries."""
        long_prompt = "A" * 500
        long_response = "B" * 500
        logger.log_llm_call("medical_agent", long_prompt, long_response)

        with open(logger.log_path) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "llm_call"
        assert entry["agent"] == "medical_agent"
        assert entry["prompt_summary"] == "A" * 200
        assert entry["response_summary"] == "B" * 200

    def test_short_strings_preserved(self, logger: DecisionLogger) -> None:
        """Short prompts/responses are kept intact."""
        logger.log_llm_call("finance_agent", "short prompt", "short response")

        with open(logger.log_path) as f:
            entry = json.loads(f.readline())

        assert entry["prompt_summary"] == "short prompt"
        assert entry["response_summary"] == "short response"


# ---------------------------------------------------------------------------
# Test: Multiple events
# ---------------------------------------------------------------------------


class TestMultipleEvents:
    """Tests for multi-event logging scenarios."""

    def test_multiple_events_create_multiple_lines(self, logger: DecisionLogger) -> None:
        """Each log call appends a new line."""
        logger.log_event("event_1", {"n": 1})
        logger.log_event("event_2", {"n": 2})
        logger.log_event("event_3", {"n": 3})

        with open(logger.log_path) as f:
            lines = f.readlines()

        assert len(lines) == 3
        assert json.loads(lines[0])["n"] == 1
        assert json.loads(lines[1])["n"] == 2
        assert json.loads(lines[2])["n"] == 3

    def test_all_lines_parse_as_valid_json(self, logger: DecisionLogger) -> None:
        """Every line in the log file is valid JSON."""
        logger.log_event("a", {"x": 1})
        logger.log_rule_evaluation("agent", "R-001", True)
        logger.log_final_decision({"verdict": "approve"})
        logger.log_llm_call("agent", "prompt", "response")

        with open(logger.log_path) as f:
            for line in f:
                json.loads(line)  # Should not raise

    def test_different_event_types(self, logger: DecisionLogger) -> None:
        """Different log methods produce correct event_type values."""
        logger.log_rule_evaluation("med_agent", "MED-001", True)
        logger.log_agent_assessment({"agent_name": "med"})
        logger.log_debate_round({"round": 1})
        logger.log_final_decision({"verdict": "approve"})
        logger.log_llm_call("agent", "p", "r")

        with open(logger.log_path) as f:
            entries = [json.loads(line) for line in f]

        event_types = [e["event_type"] for e in entries]
        assert event_types == [
            "rule_evaluation",
            "agent_assessment",
            "debate_round",
            "final_decision",
            "llm_call",
        ]
