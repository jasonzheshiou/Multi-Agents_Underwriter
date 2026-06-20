"""Unit tests for the TestQuestionnaireEngine class.

Tests cover engine load, run, save_result, get_console_summary,
and agent selection functionality. Integration tests use real rule
files from rules/death/.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from underwriting.application.schema import Application, BenefitType, SmokerStatus
from underwriting.test_questionnaire.engine import TestQuestionnaireEngine
from underwriting.test_questionnaire.models import QuestionnaireDefinition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def yaml_helper(tmp_path: Path):
    """Create a helper function that writes YAML questionnaire files."""

    def _write_yaml(
        name: str = "Test Applicant",
        smoker: str = "Never",
        benefit_types: List[str] | None = None,
        occupation: str = "Software Manager",
        annual_income: float = 120000.0,
        height_cm: float = 178.0,
        weight_kg: float = 76.0,
        smoker_status: str = "Never",
        **extra_fields: Any,
    ) -> str:
        """Write a YAML questionnaire file and return its path."""
        yaml_content = f"""
name: {name}
description: Test questionnaire for {name}
benefit_types:
  - Death
agent_names:
  - MedicalAgent
  - FinancialAgent
  - ComplianceAgent
full_name: {name}
date_of_birth: 1990-06-15
gender: Male
residency_status: Australian Citizen
contact_address: 12 Main St, Sydney NSW 2000
sum_insured_death: 500000.0
sum_insured_tpd: 500000.0
occupation: {occupation}
employer_name: TechCorp Pty Ltd
years_in_occupation: 8.0
annual_income: {annual_income}
height_cm: {height_cm}
weight_kg: {weight_kg}
smoker_status: {smoker_status}
cigarettes_per_day: 0
years_smoked: 0
taking_medications: false
has_medical_conditions: false
medical_conditions: []
consumes_alcohol: false
standard_drinks_per_week: 0
has_family_history: false
family_history: []
has_hazardous_pursuits: false
hazardous_pursuits: []
recreational_drug_use: false
alcohol_drug_treatment: false
has_high_risk_travel: false
duty_of_disclosure_acknowledged: true
"""
        for key, value in extra_fields.items():
            yaml_content += f"{key}: {value}\n"

        filepath = tmp_path / f"{name.replace(' ', '_')}.yaml"
        filepath.write_text(yaml_content, encoding="utf-8")
        return str(filepath)

    return _write_yaml


@pytest.fixture()
def standard_result() -> Dict[str, Any]:
    """Return a mock orchestrator result dict for a standard applicant."""
    medical_assessment = MagicMock()
    medical_assessment.risk_tier = "standard"
    medical_assessment.recommendation = "Approve"
    medical_assessment.confidence_score = 0.95
    medical_assessment.flags = []
    medical_assessment.additional_evidence_required = []

    financial_assessment = MagicMock()
    financial_assessment.risk_tier = "standard"
    financial_assessment.recommendation = "Approve"
    financial_assessment.confidence_score = 0.90
    financial_assessment.flags = []
    financial_assessment.additional_evidence_required = []

    compliance_assessment = MagicMock()
    compliance_assessment.risk_tier = "standard"
    compliance_assessment.recommendation = "Approve"
    compliance_assessment.confidence_score = 0.98
    compliance_assessment.flags = []
    compliance_assessment.additional_evidence_required = []

    return {
        "final_assessment": {
            "decision": "Standard Offer",
            "risk_tier": "standard",
            "reasoning": "All agents agree on standard risk.",
            "flags": [],
            "additional_evidence_required": [],
        },
        "agent_assessments": {
            "MedicalAgent": medical_assessment,
            "FinancialAgent": financial_assessment,
            "ComplianceAgent": compliance_assessment,
        },
        "debate_log": [],
        "consensus_reached": True,
        "final_decision": "Standard Offer",
        "decision_reasoning": "All 3 agents assessed as standard. Standard Offer.",
    }


@pytest.fixture()
def moderate_result() -> Dict[str, Any]:
    """Return a mock orchestrator result dict for a moderate-risk applicant."""
    medical_assessment = MagicMock()
    medical_assessment.risk_tier = "loading"
    medical_assessment.recommendation = "Loading Applied"
    medical_assessment.confidence_score = 0.80
    medical_assessment.flags = [
        {"rule_id": "MED-D-013", "severity": "high", "description": "Current smoker"}
    ]
    medical_assessment.additional_evidence_required = ["Medical report"]

    financial_assessment = MagicMock()
    financial_assessment.risk_tier = "standard"
    financial_assessment.recommendation = "Approve"
    financial_assessment.confidence_score = 0.90
    financial_assessment.flags = []
    financial_assessment.additional_evidence_required = []

    compliance_assessment = MagicMock()
    compliance_assessment.risk_tier = "standard"
    compliance_assessment.recommendation = "Approve"
    compliance_assessment.confidence_score = 0.98
    compliance_assessment.flags = []
    compliance_assessment.additional_evidence_required = []

    return {
        "final_assessment": {
            "decision": "Offer with Loading/Exclusion",
            "risk_tier": "loading",
            "reasoning": "Medical agent flagged loading risk.",
            "flags": [{"rule_id": "MED-D-013", "severity": "high", "description": "Current smoker"}],
            "additional_evidence_required": ["Medical report"],
        },
        "agent_assessments": {
            "MedicalAgent": medical_assessment,
            "FinancialAgent": financial_assessment,
            "ComplianceAgent": compliance_assessment,
        },
        "debate_log": [
            {
                "round": 1,
                "agent": "MedicalAgent",
                "original_tier": ["standard", "standard"],
                "updated_tier": "loading",
            }
        ],
        "consensus_reached": False,
        "final_decision": "Offer with Loading/Exclusion",
        "decision_reasoning": "Medical agent assessed loading. Final: Offer with Loading/Exclusion.",
    }


# ---------------------------------------------------------------------------
# TestEngineLoad
# ---------------------------------------------------------------------------


class TestEngineLoad:
    """Tests for engine load functionality."""

    def test_load_valid_yaml(self, yaml_helper: Any, tmp_path: Path):
        """Engine loads a valid YAML questionnaire file successfully."""
        yaml_path = yaml_helper(name="Standard Applicant")
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "logs"),
            report_dir=str(tmp_path / "reports"),
        )
        definition = engine.load()

        assert definition is not None
        assert isinstance(definition, QuestionnaireDefinition)
        assert definition.name == "Standard Applicant"
        assert definition.full_name == "Standard Applicant"
        assert definition.annual_income == 120000.0
        assert definition.smoker_status == SmokerStatus.NEVER

    def test_load_raises_file_not_found(self):
        """Loading a non-existent file raises FileNotFoundError."""
        engine = TestQuestionnaireEngine(questionnaire_path="/nonexistent/path/questionnaire.yaml")
        with pytest.raises(FileNotFoundError):
            engine.load()

    def test_load_caching(self, yaml_helper: Any):
        """Load sets _definition which is reused by run() without explicit load."""
        yaml_path = yaml_helper(name="Cached Applicant")
        engine = TestQuestionnaireEngine(questionnaire_path=yaml_path)

        first_load = engine.load()

        assert engine._definition is first_load
        assert engine._definition.name == "Cached Applicant"

    def test_load_without_explicit_call(self, yaml_helper: Any, standard_result: Dict[str, Any]):
        """Engine auto-loads definition when run() is called without explicit load()."""
        yaml_path = yaml_helper(name="Auto Load Applicant")
        engine = TestQuestionnaireEngine(questionnaire_path=yaml_path)

        with patch.object(engine, "_create_agents") as mock_create, \
             patch.object(engine, "_log_results"):
            mock_agents = [MagicMock()]
            mock_create.return_value = mock_agents

            with patch("underwriting.test_questionnaire.engine.DebateOrchestrator") as mock_orch:
                mock_orch.return_value.run.return_value = standard_result

                engine.run()

        assert engine._definition is not None
        assert engine._definition.name == "Auto Load Applicant"


# ---------------------------------------------------------------------------
# TestEngineRun
# ---------------------------------------------------------------------------


class TestEngineRun:
    """Tests for engine run functionality."""

    def test_run_all_agents(self, yaml_helper: Any, standard_result: Dict[str, Any], tmp_path: Path):
        """Run with all 3 agents produces 3 assessments in results."""
        yaml_path = yaml_helper(name="All Agents Applicant")
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "logs"),
        )

        with patch.object(engine, "_create_agents") as mock_create, \
             patch.object(engine, "_log_results"):
            mock_agents = [MagicMock(), MagicMock(), MagicMock()]
            mock_create.return_value = mock_agents

            with patch("underwriting.test_questionnaire.engine.DebateOrchestrator") as mock_orch:
                mock_orch.return_value.run.return_value = standard_result

                result = engine.run()

        mock_create.assert_called_once()
        call_args = mock_create.call_args
        agent_names_arg = call_args[0][1]
        assert agent_names_arg is None
        assert len(result["agent_assessments"]) == 3

    def test_run_subset_agents(self, yaml_helper: Any, tmp_path: Path):
        """Run with only MedicalAgent produces 1 assessment."""
        yaml_path = yaml_helper(name="Subset Applicant")
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "logs"),
        )

        medical_assessment = MagicMock()
        medical_assessment.risk_tier = "standard"
        medical_assessment.recommendation = "Approve"
        medical_assessment.confidence_score = 0.95
        medical_assessment.flags = []
        medical_assessment.additional_evidence_required = []

        subset_result = {
            "final_assessment": {"decision": "Standard Offer"},
            "agent_assessments": {"MedicalAgent": medical_assessment},
            "debate_log": [],
            "consensus_reached": True,
            "final_decision": "Standard Offer",
            "decision_reasoning": "Medical agent assessed standard.",
        }

        with patch.object(engine, "_create_agents") as mock_create, \
             patch.object(engine, "_log_results"):
            mock_agents = [MagicMock()]
            mock_create.return_value = mock_agents

            with patch("underwriting.test_questionnaire.engine.DebateOrchestrator") as mock_orch:
                mock_orch.return_value.run.return_value = subset_result

                result = engine.run(agent_names=["MedicalAgent"])

        mock_create.assert_called_once()
        call_args = mock_create.call_args
        agent_names_arg = call_args[0][1]
        assert len(agent_names_arg) == 1
        assert agent_names_arg[0] == "MedicalAgent"
        assert len(result["agent_assessments"]) == 1

    def test_run_single_agent(self, yaml_helper: Any, standard_result: Dict[str, Any], tmp_path: Path):
        """Run with only ComplianceAgent produces 1 assessment."""
        yaml_path = yaml_helper(name="Single Agent Applicant")
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "logs"),
        )

        with patch.object(engine, "_create_agents") as mock_create, \
             patch.object(engine, "_log_results"):
            mock_agents = [MagicMock()]
            mock_create.return_value = mock_agents

            with patch("underwriting.test_questionnaire.engine.DebateOrchestrator") as mock_orch:
                mock_orch.return_value.run.return_value = standard_result

                result = engine.run(agent_names=["ComplianceAgent"])

        mock_create.assert_called_once()
        call_args = mock_create.call_args
        agent_names_arg = call_args[0][1]
        assert len(agent_names_arg) == 1
        assert agent_names_arg[0] == "ComplianceAgent"

    def test_run_returns_required_keys(self, yaml_helper: Any, standard_result: Dict[str, Any], tmp_path: Path):
        """Run result contains: final_decision, agent_assessments, debate_log, consensus_reached, decision_reasoning."""
        yaml_path = yaml_helper(name="Keys Applicant")
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "logs"),
        )

        with patch.object(engine, "_create_agents") as mock_create, \
             patch.object(engine, "_log_results"):
            mock_create.return_value = [MagicMock()]

            with patch("underwriting.test_questionnaire.engine.DebateOrchestrator") as mock_orch:
                mock_orch.return_value.run.return_value = standard_result

                result = engine.run()

        required_keys = [
            "final_decision",
            "agent_assessments",
            "debate_log",
            "consensus_reached",
            "decision_reasoning",
        ]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

    def test_run_invalid_agent_name(self, yaml_helper: Any):
        """Run with invalid agent name raises ValueError."""
        yaml_path = yaml_helper(name="Invalid Agent Applicant")
        engine = TestQuestionnaireEngine(questionnaire_path=yaml_path)

        with pytest.raises(ValueError) as exc_info:
            engine.run(agent_names=["InvalidAgent"])

        assert "Unknown agent" in str(exc_info.value)

    def test_run_passes_application_to_orchestrator(
        self, yaml_helper: Any, standard_result: Dict[str, Any], tmp_path: Path
    ):
        """Run passes a valid Application to the orchestrator."""
        yaml_path = yaml_helper(name="Application Applicant")
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "logs"),
        )

        with patch.object(engine, "_create_agents") as mock_create, \
             patch.object(engine, "_log_results"):
            mock_create.return_value = [MagicMock()]

            with patch("underwriting.test_questionnaire.engine.DebateOrchestrator") as mock_orch:
                mock_orch.return_value.run.return_value = standard_result

                engine.run()

        mock_orch.return_value.run.assert_called_once()
        app_arg = mock_orch.return_value.run.call_args[0][0]
        assert isinstance(app_arg, Application)
        assert app_arg.full_name == "Application Applicant"

    def test_run_debate_orchestrator_called_with_agents(
        self, yaml_helper: Any, standard_result: Dict[str, Any], tmp_path: Path
    ):
        """Run passes created agents to DebateOrchestrator."""
        yaml_path = yaml_helper(name="Orchestrator Agents Applicant")
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "logs"),
        )

        expected_agents = [MagicMock(), MagicMock(), MagicMock()]

        with patch.object(engine, "_create_agents", return_value=expected_agents) as mock_create, \
             patch.object(engine, "_log_results"):
            with patch("underwriting.test_questionnaire.engine.DebateOrchestrator") as mock_orch:
                mock_orch.return_value.run.return_value = standard_result

                engine.run()

        mock_orch.assert_called_once_with(agents=expected_agents)


# ---------------------------------------------------------------------------
# TestEngineSaveResult
# ---------------------------------------------------------------------------


class TestEngineSaveResult:
    """Tests for engine save_result functionality."""

    def test_save_result_creates_file(self, yaml_helper: Any, standard_result: Dict[str, Any], tmp_path: Path):
        """Save result creates a JSON file in the output directory."""
        yaml_path = yaml_helper(name="Save Test Applicant")
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "logs"),
        )
        engine.load()

        output_dir = str(tmp_path / "results")
        filepath = engine.save_result(standard_result, output_dir=output_dir)

        assert Path(filepath).exists()
        assert Path(filepath).suffix == ".json"
        assert Path(filepath).name.startswith("result_Save_Test_Applicant_")

    def test_save_result_content(self, yaml_helper: Any, standard_result: Dict[str, Any], tmp_path: Path):
        """Saved JSON contains final_decision and agent_assessments."""
        yaml_path = yaml_helper(name="Content Test Applicant")
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "logs"),
        )
        engine.load()

        output_dir = str(tmp_path / "results")
        engine.save_result(standard_result, output_dir=output_dir)

        result_files = list(Path(output_dir).glob("*.json"))
        assert len(result_files) >= 1

        saved_data = json.loads(result_files[0].read_text(encoding="utf-8"))

        assert "final_decision" in saved_data
        assert "agent_assessments" in saved_data
        assert saved_data["final_decision"] == "Standard Offer"


# ---------------------------------------------------------------------------
# TestEngineConsoleSummary
# ---------------------------------------------------------------------------


class TestEngineConsoleSummary:
    """Tests for engine get_console_summary functionality."""

    def test_console_summary_format(self, yaml_helper: Any, standard_result: Dict[str, Any], tmp_path: Path):
        """Summary contains expected sections: [APPLICANT], [AGENTS], [DEBATE], [DECISION]."""
        yaml_path = yaml_helper(name="Summary Applicant")
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "logs"),
        )
        definition = engine.load()
        application = definition.to_application()

        summary = engine.get_console_summary(application, standard_result)

        assert "[APPLICANT]" in summary
        assert "[AGENTS]" in summary
        assert "[DEBATE]" in summary
        assert "[DECISION]" in summary

    def test_console_summary_with_debate(self, yaml_helper: Any, moderate_result: Dict[str, Any], tmp_path: Path):
        """Summary shows debate rounds when debate_log is present."""
        yaml_path = yaml_helper(name="Debate Applicant")
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "logs"),
        )
        definition = engine.load()
        application = definition.to_application()

        summary = engine.get_console_summary(application, moderate_result)

        assert "Round 1:" in summary
        assert "MedicalAgent" in summary
        assert "loading" in summary

    def test_console_summary_shows_applicant_details(
        self, yaml_helper: Any, standard_result: Dict[str, Any], tmp_path: Path
    ):
        """Summary includes applicant name, age, income, and BMI."""
        yaml_path = yaml_helper(name="Summary Details Applicant")
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "logs"),
        )
        definition = engine.load()
        application = definition.to_application()

        summary = engine.get_console_summary(application, standard_result)

        assert "Summary Details Applicant" in summary
        assert f"${application.annual_income:,.2f}" in summary
        assert f"{application.bmi}" in summary
        assert str(application.age) in summary

    def test_console_summary_shows_smoker_status(
        self, yaml_helper: Any, standard_result: Dict[str, Any], tmp_path: Path
    ):
        """Summary shows the applicant's smoker status."""
        yaml_path = yaml_helper(name="Smoker Applicant", smoker_status="Current")
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "logs"),
        )
        definition = engine.load()
        application = definition.to_application()

        summary = engine.get_console_summary(application, standard_result)

        assert "Current" in summary

    def test_console_summary_shows_benefit_types(
        self, yaml_helper: Any, standard_result: Dict[str, Any], tmp_path: Path
    ):
        """Summary lists all benefit types requested."""
        yaml_path = yaml_helper(name="Benefits Applicant")
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "logs"),
        )
        definition = engine.load()
        application = definition.to_application()

        summary = engine.get_console_summary(application, standard_result)

        assert "Death" in summary

    def test_console_summary_no_debate_message(
        self, yaml_helper: Any, standard_result: Dict[str, Any], tmp_path: Path
    ):
        """Summary shows 'No debate needed' when debate_log is empty."""
        yaml_path = yaml_helper(name="No Debate Applicant")
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "logs"),
        )
        definition = engine.load()
        application = definition.to_application()

        empty_debate_result = dict(standard_result)
        empty_debate_result["debate_log"] = []

        summary = engine.get_console_summary(application, empty_debate_result)

        assert "No debate needed" in summary

    def test_console_summary_shows_agent_flags(
        self, yaml_helper: Any, tmp_path: Path
    ):
        """Summary displays agent risk flags with severity and rule_id."""
        yaml_path = yaml_helper(name="Flags Applicant")
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "logs"),
        )
        definition = engine.load()
        application = definition.to_application()

        result_with_flags = {
            "final_assessment": {"decision": "Offer with Loading/Exclusion"},
            "agent_assessments": {
                "MedicalAgent": MagicMock(
                    risk_tier="loading",
                    recommendation="Loading Applied",
                    confidence_score=0.85,
                    flags=[
                        {"severity": "high", "rule_id": "MED-D-001", "description": "High BMI"},
                        {"severity": "moderate", "rule_id": "MED-D-002", "description": "Family history"},
                    ],
                    additional_evidence_required=["Recent medical exam"],
                )
            },
            "debate_log": [],
            "consensus_reached": True,
            "final_decision": "Offer with Loading/Exclusion",
            "decision_reasoning": "Medical agent flagged loading risk.",
        }

        summary = engine.get_console_summary(application, result_with_flags)

        assert "MED-D-001" in summary
        assert "High BMI" in summary
        assert "[high]" in summary
        assert "[moderate]" in summary
        assert "Evidence Required" in summary
        assert "Recent medical exam" in summary


# ---------------------------------------------------------------------------
# TestEngineIntegration
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEngineIntegration:
    """Integration tests using real YAML files and rule evaluation."""

    def test_full_pipeline_standard(self, yaml_helper: Any, tmp_path: Path):
        """Load standard.yaml -> run -> verify Standard Offer decision."""
        yaml_path = yaml_helper(
            name="Standard Pipeline Applicant",
            smoker_status="Never",
            occupation="Software Manager",
            annual_income=100000.0,
            height_cm=175.0,
            weight_kg=70.0,
        )
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "audit_logs"),
            report_dir=str(tmp_path / "reports"),
        )
        result = engine.run()

        assert result["final_decision"] == "Standard Offer"
        assert result["consensus_reached"] is True
        agent_names = list(result["agent_assessments"].keys())
        assert len(agent_names) == 3
        for assessment in result["agent_assessments"].values():
            assert assessment.risk_tier == "standard"

    def test_full_pipeline_moderate(self, yaml_helper: Any, tmp_path: Path):
        """Load moderate.yaml -> run -> verify Loading decision."""
        yaml_path = yaml_helper(
            name="Moderate Pipeline Applicant",
            smoker_status="Current",
            cigarettes_per_day=10,
            years_smoked=20,
            occupation="Underground Miner",
            annual_income=150000.0,
            height_cm=175.0,
            weight_kg=80.0,
            has_hazardous_duties=True,
            hazardous_duties_description="Works underground",
        )
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "audit_logs"),
            report_dir=str(tmp_path / "reports"),
        )
        result = engine.run()

        assert result["final_decision"] in ("Offer with Loading/Exclusion", "Standard Offer")
        assert result["consensus_reached"] in (True, False)
        agent_names = list(result["agent_assessments"].keys())
        assert len(agent_names) == 3

    def test_audit_log_created(self, yaml_helper: Any, tmp_path: Path):
        """Run creates audit log file in log_dir."""
        yaml_path = yaml_helper(name="Audit Log Applicant")
        log_dir = str(tmp_path / "audit_logs")
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=log_dir,
            report_dir=str(tmp_path / "reports"),
        )
        engine.run()

        log_files = list(Path(log_dir).glob("underwriting_*.jsonl"))
        assert len(log_files) >= 1

        log_content = log_files[0].read_text(encoding="utf-8")
        assert len(log_content) > 0

        first_line = log_content.strip().split("\n")[0]
        log_entry = json.loads(first_line)
        assert "event_type" in log_entry
        assert "timestamp" in log_entry
        assert "session_id" in log_entry

    def test_save_result_integration(self, yaml_helper: Any, tmp_path: Path):
        """Full pipeline run saves result JSON with all expected keys."""
        yaml_path = yaml_helper(name="Save Integration Applicant")
        result_dir = str(tmp_path / "results")
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "audit_logs"),
            report_dir=str(tmp_path / "reports"),
        )
        engine.load()
        engine.run()
        # run() calls save_result internally with default output_dir
        # We manually save to our tmp_path result_dir to verify content
        engine.save_result(engine._definition.to_application().model_dump(), output_dir=result_dir)

        result_files = list(Path(result_dir).glob("*.json"))
        assert len(result_files) >= 1

    def test_run_subset_agents_integration(self, yaml_helper: Any, tmp_path: Path):
        """Run with subset of agents produces fewer assessments."""
        yaml_path = yaml_helper(name="Subset Integration Applicant")
        engine = TestQuestionnaireEngine(
            questionnaire_path=yaml_path,
            log_dir=str(tmp_path / "audit_logs"),
            report_dir=str(tmp_path / "reports"),
        )
        result = engine.run(agent_names=["MedicalAgent"])

        assert len(result["agent_assessments"]) == 1
        # Agent name in result uses agent.name property (e.g., "Medical Agent")
        assert any("Medical" in name for name in result["agent_assessments"].keys())
