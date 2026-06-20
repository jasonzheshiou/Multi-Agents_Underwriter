"""Test Questionnaire Engine - orchestrates the full test questionnaire pipeline."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from underwriting.agents.base_agent import BaseAgent
from underwriting.agents.compliance_agent import ComplianceAgent
from underwriting.agents.debate_orchestrator import DebateOrchestrator
from underwriting.agents.financial_agent import FinancialAgent
from underwriting.agents.medical_agent import MedicalAgent
from underwriting.application.schema import Application
from underwriting.audit.logger import DecisionLogger
from underwriting.llm.llm_client import LLMClient
from underwriting.test_questionnaire.models import QuestionnaireDefinition


class TestQuestionnaireEngine:
    """Orchestrates the full test questionnaire pipeline.

    Loads a YAML questionnaire definition, converts it to an Application,
    runs the multi-agent underwriting pipeline via DebateOrchestrator,
    logs results to the audit trail, and saves results to disk.
    """

    def __init__(self, questionnaire_path: str, log_dir: str = "./audit_logs", report_dir: str = "./audit_reports"):
        """Initialize the engine.

        Args:
            questionnaire_path: Path to the YAML questionnaire file.
            log_dir: Directory for JSONL audit logs.
            report_dir: Directory for markdown/JSON reports.
        """
        self.questionnaire_path = questionnaire_path
        self.log_dir = log_dir
        self.report_dir = report_dir
        self._definition: Optional[QuestionnaireDefinition] = None

    def load(self) -> QuestionnaireDefinition:
        """Load and return the questionnaire definition."""
        self._definition = QuestionnaireDefinition.from_yaml(self.questionnaire_path)
        return self._definition

    def run(self, agent_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run the full pipeline and return results.

        Args:
            agent_names: Optional list of agent class names to run.
                If None, runs all agents (MedicalAgent, FinancialAgent, ComplianceAgent).
                Valid values: "MedicalAgent", "FinancialAgent", "ComplianceAgent"

        Returns:
            Dictionary containing orchestrator results.
        """
        if self._definition is None:
            self.load()

        application = self._definition.to_application()

        agents = self._create_agents(application, agent_names)

        orchestrator = DebateOrchestrator(agents=agents)
        results = orchestrator.run(application)

        self._log_results(application, results)

        self.save_result(results)

        return results

    def save_result(self, results: Dict[str, Any], output_dir: str = "data/test_questionnaires/results") -> str:
        """Save results to a JSON file. Returns the file path."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        name = self._definition.name.replace(" ", "_") if self._definition else "test"
        filename = f"result_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        return filepath

    def get_console_summary(self, application: Application, results: Dict[str, Any]) -> str:
        """Return a formatted console summary string.

        Matches the exact format from demo/pipeline.py print_summary().
        """
        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("  MULTI-AGENT UNDERWRITING RULES ENGINE - PIPELINE SUMMARY")
        lines.append("=" * 70)

        lines.append("")
        lines.append("  [APPLICANT] APPLICANT")
        lines.append(f"     Name:          {application.full_name}")
        lines.append(f"     Date of Birth: {application.date_of_birth}")
        lines.append(f"     Age:           {application.age}")
        lines.append(f"     Gender:        {application.gender}")
        lines.append(f"     Occupation:    {application.occupation}")
        lines.append(f"     Income:        ${application.annual_income:,.2f}")
        lines.append(f"     BMI:           {application.bmi}")
        lines.append(f"     Smoker:        {application.smoker_status.value}")
        lines.append(f"     Benefits:      {', '.join(bt.value for bt in application.benefit_types)}")

        lines.append("")
        lines.append("  [AGENTS] AGENT ASSESSMENTS")
        agent_assessments = results.get("agent_assessments", {})
        for agent_name, assessment in agent_assessments.items():
            lines.append("")
            lines.append(f"     +-- {agent_name}")
            if "Compliance" in agent_name:
                # Compliance is observer only — show compliance risk level, not underwriting tier
                severity_to_risk = {"critical": "HIGH", "high": "HIGH",
                                    "moderate": "MEDIUM", "low": "LOW",
                                    "none": "NONE"}
                max_sev = "none"
                if assessment.flags:
                    sev_order = {"critical": 4, "high": 3, "moderate": 2, "low": 1, "none": 0}
                    max_sev = max(assessment.flags, key=lambda f: sev_order.get(f.get("severity", "none"), 0)).get("severity", "none")
                lines.append(f"     | Compliance Risk: {severity_to_risk.get(max_sev, 'NONE')}")
                lines.append(f"     | Role:            Observer/Informer (does NOT affect underwriting decision)")
                lines.append(f"     | Observations:    {len(assessment.flags)}")
            else:
                lines.append(f"     | Risk Tier:       {assessment.risk_tier}")
                lines.append(f"     | Recommendation:  {assessment.recommendation}")
                lines.append(f"     | Confidence:      {assessment.confidence_score:.0%}")
                lines.append(f"     | Flags:           {len(assessment.flags)}")
            if assessment.flags:
                for flag in assessment.flags:
                    severity = flag.get("severity", "unknown")
                    rule_id = flag.get("rule_id", "N/A")
                    desc = flag.get("description", "")
                    lines.append(f"     |   [{severity}] {rule_id}: {desc}")
            if assessment.additional_evidence_required:
                lines.append("     | Evidence Required:")
                for ev in assessment.additional_evidence_required:
                    lines.append(f"     |   - {ev}")
            lines.append("     +--------------------------------")

        lines.append("")
        lines.append("  [DEBATE] DEBATE")
        debate_log = results.get("debate_log", [])
        if debate_log:
            for entry in debate_log:
                round_num = entry.get("round", "?")
                agent = entry.get("agent", "?")
                original = entry.get("original_tier", [])
                updated = entry.get("updated_tier", "?")
                lines.append(f"     Round {round_num}: {agent} -- tiers {original} -> {updated}")
        else:
            lines.append("     No debate needed -- consensus reached.")

        lines.append("")
        lines.append("  [DECISION] FINAL DECISION")
        final_decision = results.get("final_decision", "Unknown")
        consensus = results.get("consensus_reached", False)
        reasoning = results.get("decision_reasoning", "")
        lines.append(f"     Decision:       {final_decision}")
        lines.append(f"     Consensus:      {'Yes' if consensus else 'No'}")
        lines.append(f"     Reasoning:      {reasoning}")
        lines.append("=" * 70)
        lines.append("")

        return "\n".join(lines)

    def _create_agents(self, application: Application, agent_names: Optional[List[str]] = None) -> List[BaseAgent]:
        """Create agents based on the specified names.

        Args:
            application: The Application model.
            agent_names: List of agent class names. None = all agents.

        Returns:
            List of BaseAgent instances.
        """
        benefit_types = [bt.value for bt in application.benefit_types]
        if "Death" in benefit_types:
            rules_dir = "rules/death"
        elif "TPD" in benefit_types:
            rules_dir = "rules/tpd"
        elif "Trauma/CI" in benefit_types:
            rules_dir = "rules/trauma"
        elif "Income Protection" in benefit_types:
            rules_dir = "rules/ip"
        else:
            rules_dir = "rules/death"

        agent_map = {
            "MedicalAgent": ("MedicalAgent", MedicalAgent, f"{rules_dir}/medical_rules.json"),
            "FinancialAgent": ("FinancialAgent", FinancialAgent, f"{rules_dir}/financial_rules.json"),
            "ComplianceAgent": ("ComplianceAgent", ComplianceAgent, f"{rules_dir}/compliance_rules.json"),
        }

        if agent_names is None:
            agent_names = ["MedicalAgent", "FinancialAgent", "ComplianceAgent"]

        try:
            llm_client = LLMClient(config_path="./config.yaml")
        except FileNotFoundError:
            llm_client = None

        agents = []
        for name in agent_names:
            if name not in agent_map:
                raise ValueError(f"Unknown agent: {name}. Valid agents: {list(agent_map.keys())}")
            display_name, agent_class, rules_path = agent_map[name]
            agents.append(agent_class(rules_path=rules_path, llm_client=llm_client))

        return agents

    def _log_results(self, application: Application, results: Dict[str, Any]) -> None:
        """Log results to the audit trail.

        Args:
            application: The evaluated Application model.
            results: Orchestrator results dict.
        """
        logger_ = DecisionLogger(log_dir=self.log_dir)

        agent_assessments = results.get("agent_assessments", {})
        for agent_name, assessment in agent_assessments.items():
            logger_.log_agent_assessment(assessment)
            logger_.log_event("agent_evaluation_complete", {
                "agent": agent_name,
                "applicant": application.full_name,
                "risk_tier": assessment.risk_tier,
                "recommendation": assessment.recommendation,
                "flags_count": len(assessment.flags),
            })

        debate_log = results.get("debate_log", [])
        if debate_log:
            for round_entry in debate_log:
                logger_.log_debate_round(round_entry)
        else:
            logger_.log_event("debate_status", {
                "status": "no_debate_needed",
                "reason": "All agents reached consensus",
            })

        decision_data = {
            "final_decision": results.get("final_decision", "Unknown"),
            "consensus_reached": results.get("consensus_reached", False),
            "decision_reasoning": results.get("decision_reasoning", ""),
            "debate_rounds": len(results.get("debate_log", [])),
        }
        logger_.log_final_decision(decision_data)
