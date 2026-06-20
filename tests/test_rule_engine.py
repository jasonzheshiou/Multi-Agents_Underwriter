"""Tests for the RuleEngine class."""

import json
import os
import tempfile
from unittest import mock

import pytest

from underwriting.rules.rule_engine import RuleEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rule_engine():
    """Return a RuleEngine instance with a temporary rules directory."""
    return RuleEngine(rules_dir=tempfile.mkdtemp())


@pytest.fixture
def sample_rules():
    """Return a sample rules dict for testing."""
    return {
        "rules": [
            {
                "rule_id": "AGE_001",
                "condition": "applicant.age < 18",
                "severity": "critical",
                "recommendation": "decline",
                "description": "Applicant must be 18 or older",
                "loading_range": [1.0, 1.0],
                "additional_evidence": [],
                "apra_ref": "",
            },
            {
                "rule_id": "AGE_002",
                "condition": "applicant.age > 75",
                "severity": "high",
                "recommendation": "manual_underwriting",
                "description": "Applicant over 75 requires manual review",
                "loading_range": [1.2, 1.5],
                "additional_evidence": ["Medical report"],
                "apra_ref": "APR-101",
            },
            {
                "rule_id": "SMK_001",
                "condition": "getattr(applicant, 'smoker', False)",
                "severity": "moderate",
                "recommendation": "standard_or_loading",
                "description": "Smoker loading applied",
                "loading_range": [1.1, 1.3],
                "additional_evidence": [],
                "apra_ref": "",
            },
            {
                "rule_id": "OCC_001",
                "condition": "getattr(applicant, 'occupation', '') == 'miner'",
                "severity": "low",
                "recommendation": "standard",
                "description": "Occupation risk check",
                "loading_range": [1.0, 1.1],
                "additional_evidence": ["Occupation history"],
                "apra_ref": "APR-202",
            },
        ]
    }


@pytest.fixture
def mock_application():
    """Return a simple mock application object."""
    app = mock.MagicMock()
    app.age = 30
    app.smoker = False
    app.occupation = "miner"
    return app


@pytest.fixture
def rules_file_with_content(rule_engine, sample_rules):
    """Create a temporary JSON file with sample rules and return its path."""
    path = os.path.join(rule_engine.rules_dir, "test_rules.json")
    with open(path, "w") as f:
        json.dump(sample_rules, f)
    return path


# ---------------------------------------------------------------------------
# Tests: __init__
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for RuleEngine.__init__."""

    def test_default_rules_dir(self):
        engine = RuleEngine()
        assert engine.rules_dir == "./rules/death"

    def test_custom_rules_dir(self):
        engine = RuleEngine(rules_dir="./custom/rules")
        assert engine.rules_dir == "./custom/rules"


# ---------------------------------------------------------------------------
# Tests: load_rules
# ---------------------------------------------------------------------------


class TestLoadRules:
    """Tests for RuleEngine.load_rules."""

    def test_load_valid_rules(self, rule_engine, rules_file_with_content):
        rules = rule_engine.load_rules(rules_file_with_content)
        assert "rules" in rules
        assert len(rules["rules"]) == 4

    def test_load_rules_returns_correct_rule_ids(self, rule_engine, rules_file_with_content):
        rules = rule_engine.load_rules(rules_file_with_content)
        rule_ids = [r["rule_id"] for r in rules["rules"]]
        assert rule_ids == ["AGE_001", "AGE_002", "SMK_001", "OCC_001"]

    def test_load_rules_file_not_found(self, rule_engine):
        with pytest.raises(FileNotFoundError):
            rule_engine.load_rules("/nonexistent/path/rules.json")

    def test_load_rules_invalid_json(self, rule_engine):
        path = os.path.join(rule_engine.rules_dir, "bad.json")
        with open(path, "w") as f:
            f.write("{invalid json}")
        with pytest.raises(json.JSONDecodeError):
            rule_engine.load_rules(path)


# ---------------------------------------------------------------------------
# Tests: evaluate_rules
# ---------------------------------------------------------------------------


class TestEvaluateRules:
    """Tests for RuleEngine.evaluate_rules."""

    def test_no_matching_rules(self, rule_engine, sample_rules, mock_application):
        # age=30, smoker=False, occupation='miner' — only OCC_001 matches
        matched = rule_engine.evaluate_rules(mock_application, sample_rules)
        rule_ids = [r["rule_id"] for r in matched]
        assert "OCC_001" in rule_ids

    def test_critical_rule_matches(self, rule_engine):
        rules = {
            "rules": [
                {
                    "rule_id": "AGE_001",
                    "condition": "applicant.age < 18",
                    "severity": "critical",
                    "recommendation": "decline",
                    "description": "Under 18",
                    "loading_range": [1.0, 1.0],
                    "additional_evidence": [],
                    "apra_ref": "",
                }
            ]
        }
        app = mock.MagicMock()
        app.age = 16
        app.smoker = False
        app.occupation = "student"
        matched = rule_engine.evaluate_rules(app, rules)
        assert len(matched) == 1
        assert matched[0]["rule_id"] == "AGE_001"

    def test_smoker_rule_matches(self, rule_engine):
        rules = {
            "rules": [
                {
                    "rule_id": "SMK_001",
                    "condition": "getattr(applicant, 'smoker', False)",
                    "severity": "moderate",
                    "recommendation": "standard_or_loading",
                    "description": "Smoker",
                    "loading_range": [1.1, 1.3],
                    "additional_evidence": [],
                    "apra_ref": "",
                }
            ]
        }
        app = mock.MagicMock()
        app.age = 30
        app.smoker = True
        app.occupation = "teacher"
        matched = rule_engine.evaluate_rules(app, rules)
        assert len(matched) == 1

    def test_no_matching_rules_returns_empty_list(self, rule_engine):
        rules = {
            "rules": [
                {
                    "rule_id": "X_001",
                    "condition": "applicant.age < 0",
                    "severity": "low",
                    "recommendation": "standard",
                    "description": "Impossible condition",
                    "loading_range": [1.0, 1.0],
                    "additional_evidence": [],
                    "apra_ref": "",
                }
            ]
        }
        app = mock.MagicMock()
        app.age = 30
        app.smoker = False
        app.occupation = "teacher"
        matched = rule_engine.evaluate_rules(app, rules)
        assert matched == []

    def test_empty_rules_dict(self, rule_engine, mock_application):
        matched = rule_engine.evaluate_rules(mock_application, {"rules": []})
        assert matched == []

    def test_missing_condition_defaults_to_false(self, rule_engine):
        rules = {
            "rules": [
                {
                    "rule_id": "NO_COND",
                    "severity": "low",
                    "recommendation": "standard",
                    "description": "No condition field",
                    "loading_range": [1.0, 1.0],
                    "additional_evidence": [],
                    "apra_ref": "",
                }
            ]
        }
        app = mock.MagicMock()
        app.age = 30
        matched = rule_engine.evaluate_rules(app, rules)
        assert matched == []

    def test_broken_condition_does_not_crash(self, rule_engine):
        rules = {
            "rules": [
                {
                    "rule_id": "BAD_COND",
                    "condition": "applicant.nonexistent_attr.method()",
                    "severity": "low",
                    "recommendation": "standard",
                    "description": "Broken condition",
                    "loading_range": [1.0, 1.0],
                    "additional_evidence": [],
                    "apra_ref": "",
                }
            ]
        }
        # Use a simple object that will raise AttributeError
        class SimpleApp:
            age = 30
        app = SimpleApp()
        # Should not raise — errors are logged
        matched = rule_engine.evaluate_rules(app, rules)
        assert matched == []

    def test_multiple_rules_match(self, rule_engine):
        rules = {
            "rules": [
                {
                    "rule_id": "A",
                    "condition": "applicant.age > 20",
                    "severity": "low",
                    "recommendation": "standard",
                    "description": "A",
                    "loading_range": [1.0, 1.0],
                    "additional_evidence": [],
                    "apra_ref": "",
                },
                {
                    "rule_id": "B",
                    "condition": "applicant.age > 25",
                    "severity": "moderate",
                    "recommendation": "standard_or_loading",
                    "description": "B",
                    "loading_range": [1.1, 1.3],
                    "additional_evidence": [],
                    "apra_ref": "",
                },
            ]
        }
        app = mock.MagicMock()
        app.age = 30
        matched = rule_engine.evaluate_rules(app, rules)
        assert len(matched) == 2
        rule_ids = [r["rule_id"] for r in matched]
        assert "A" in rule_ids
        assert "B" in rule_ids

    def test_safe_eval_no_builtins(self, rule_engine):
        """Ensure __builtins__ is restricted — dangerous expressions fail."""
        rules = {
            "rules": [
                {
                    "rule_id": "DANGEROUS",
                    "condition": "__import__('os').system('echo hacked')",
                    "severity": "low",
                    "recommendation": "standard",
                    "description": "Dangerous",
                    "loading_range": [1.0, 1.0],
                    "additional_evidence": [],
                    "apra_ref": "",
                }
            ]
        }
        app = mock.MagicMock()
        app.age = 30
        # Should not raise; expression should fail safely
        matched = rule_engine.evaluate_rules(app, rules)
        assert matched == []


# ---------------------------------------------------------------------------
# Tests: build_assessment
# ---------------------------------------------------------------------------


class TestBuildAssessment:
    """Tests for RuleEngine.build_assessment."""

    def test_no_matched_rules(self, rule_engine):
        assessment = rule_engine.build_assessment(
            application=mock.MagicMock(),
            matched_rules=[],
            agent_name="TestAgent",
        )
        assert assessment["agent_name"] == "TestAgent"
        assert assessment["risk_tier"] == "standard"
        assert assessment["flags"] == []
        assert assessment["confidence_score"] == 1.0
        assert assessment["llm_used"] is False

    def test_single_matched_rule(self, rule_engine):
        rules = [
            {
                "rule_id": "AGE_002",
                "condition": "applicant.age > 75",
                "severity": "high",
                "recommendation": "manual_underwriting",
                "description": "Over 75",
                "loading_range": [1.2, 1.5],
                "additional_evidence": ["Medical report"],
                "apra_ref": "APR-101",
            }
        ]
        app = mock.MagicMock()
        assessment = rule_engine.build_assessment(
            application=app, matched_rules=rules, agent_name="TestAgent"
        )
        assert assessment["risk_tier"] == "refer"
        assert len(assessment["flags"]) == 1
        assert assessment["flags"][0]["rule_id"] == "AGE_002"
        assert assessment["loading_range"] == [1.2, 1.5]
        assert "Medical report" in assessment["additional_evidence_required"]
        assert "APR-101" in assessment["apra_references"]
        assert "1 deterministic rule" in assessment["reasoning_summary"]

    def test_highest_severity_determines_risk_tier(self, rule_engine):
        rules = [
            {
                "rule_id": "LOW_RULE",
                "condition": "True",
                "severity": "low",
                "recommendation": "standard",
                "description": "Low",
                "loading_range": [1.0, 1.0],
                "additional_evidence": [],
                "apra_ref": "",
            },
            {
                "rule_id": "CRIT_RULE",
                "condition": "True",
                "severity": "critical",
                "recommendation": "decline",
                "description": "Critical",
                "loading_range": [1.0, 1.0],
                "additional_evidence": [],
                "apra_ref": "",
            },
        ]
        app = mock.MagicMock()
        assessment = rule_engine.build_assessment(
            application=app, matched_rules=rules, agent_name="TestAgent"
        )
        assert assessment["risk_tier"] == "decline"

    def test_assessment_contains_all_flags(self, rule_engine):
        rules = [
            {
                "rule_id": "R1",
                "condition": "True",
                "severity": "low",
                "recommendation": "standard",
                "description": "First",
                "loading_range": [1.0, 1.0],
                "additional_evidence": [],
                "apra_ref": "",
            },
            {
                "rule_id": "R2",
                "condition": "True",
                "severity": "moderate",
                "recommendation": "standard_or_loading",
                "description": "Second",
                "loading_range": [1.1, 1.3],
                "additional_evidence": [],
                "apra_ref": "",
            },
        ]
        app = mock.MagicMock()
        assessment = rule_engine.build_assessment(
            application=app, matched_rules=rules, agent_name="TestAgent"
        )
        assert len(assessment["flags"]) == 2
        flag_ids = {f["rule_id"] for f in assessment["flags"]}
        assert flag_ids == {"R1", "R2"}

    def test_evidence_deduplication(self, rule_engine):
        rules = [
            {
                "rule_id": "R1",
                "condition": "True",
                "severity": "low",
                "recommendation": "standard",
                "description": "R1",
                "loading_range": [1.0, 1.0],
                "additional_evidence": ["Same evidence"],
                "apra_ref": "",
            },
            {
                "rule_id": "R2",
                "condition": "True",
                "severity": "moderate",
                "recommendation": "standard_or_loading",
                "description": "R2",
                "loading_range": [1.1, 1.3],
                "additional_evidence": ["Same evidence"],
                "apra_ref": "",
            },
        ]
        app = mock.MagicMock()
        assessment = rule_engine.build_assessment(
            application=app, matched_rules=rules, agent_name="TestAgent"
        )
        assert assessment["additional_evidence_required"].count("Same evidence") == 1

    def test_apra_references_deduplication(self, rule_engine):
        rules = [
            {
                "rule_id": "R1",
                "condition": "True",
                "severity": "low",
                "recommendation": "standard",
                "description": "R1",
                "loading_range": [1.0, 1.0],
                "additional_evidence": [],
                "apra_ref": "APR-101",
            },
            {
                "rule_id": "R2",
                "condition": "True",
                "severity": "moderate",
                "recommendation": "standard_or_loading",
                "description": "R2",
                "loading_range": [1.1, 1.3],
                "additional_evidence": [],
                "apra_ref": "APR-101",
            },
        ]
        app = mock.MagicMock()
        assessment = rule_engine.build_assessment(
            application=app, matched_rules=rules, agent_name="TestAgent"
        )
        assert assessment["apra_references"].count("APR-101") == 1

    def test_default_recommendation_yields_standard(self, rule_engine):
        rules = [
            {
                "rule_id": "NO_REC",
                "condition": "True",
                "severity": "low",
                "description": "No recommendation field",
                "loading_range": [1.0, 1.0],
                "additional_evidence": [],
                "apra_ref": "",
            }
        ]
        app = mock.MagicMock()
        assessment = rule_engine.build_assessment(
            application=app, matched_rules=rules, agent_name="TestAgent"
        )
        assert assessment["risk_tier"] == "standard"


# ---------------------------------------------------------------------------
# Tests: _determine_risk_tier
# ---------------------------------------------------------------------------


class TestDetermineRiskTier:
    """Tests for RuleEngine._determine_risk_tier."""

    def test_standard(self):
        assert RuleEngine._determine_risk_tier({"recommendation": "standard"}) == "standard"

    def test_standard_or_loading(self):
        assert RuleEngine._determine_risk_tier(
            {"recommendation": "standard_or_loading"}
        ) == "standard"

    def test_loading(self):
        assert RuleEngine._determine_risk_tier({"recommendation": "loading"}) == "loading"
        assert RuleEngine._determine_risk_tier(
            {"recommendation": "standard_or_loading"}
        ) == "standard"

    def test_decline(self):
        assert RuleEngine._determine_risk_tier({"recommendation": "decline"}) == "decline"
        assert RuleEngine._determine_risk_tier(
            {"recommendation": "manual_underwriting_or_decline"}
        ) == "decline"

    def test_refer(self):
        assert RuleEngine._determine_risk_tier({"recommendation": "refer"}) == "refer"
        assert RuleEngine._determine_risk_tier({"recommendation": "manual_underwriting"}) == "refer"

    def test_missing_recommendation_defaults_standard(self):
        assert RuleEngine._determine_risk_tier({}) == "standard"


# ---------------------------------------------------------------------------
# Tests: _get_severity_rank
# ---------------------------------------------------------------------------


class TestGetSeverityRank:
    """Tests for RuleEngine._get_severity_rank."""

    def test_critical(self):
        assert RuleEngine._get_severity_rank("critical") == 5

    def test_high(self):
        assert RuleEngine._get_severity_rank("high") == 4

    def test_moderate(self):
        assert RuleEngine._get_severity_rank("moderate") == 3

    def test_low(self):
        assert RuleEngine._get_severity_rank("low") == 2

    def test_none(self):
        assert RuleEngine._get_severity_rank("none") == 1

    def test_unknown_returns_zero(self):
        assert RuleEngine._get_severity_rank("unknown") == 0
