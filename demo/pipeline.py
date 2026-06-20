#!/usr/bin/env python3
"""End-to-end demo pipeline for the Multi-Agents Underwriting Rules Engine.

Loads a synthetic applicant profile, runs the full multi-agent underwriting
pipeline (Medical → Financial → Compliance → Debate), logs all decisions
to JSONL, generates a markdown audit report, and prints a console summary.

Usage:
    python demo/pipeline.py --applicant data/synthetic_applicants/standard.json
    python demo/pipeline.py --applicant data/synthetic_applicants/moderate.json
    python demo/pipeline.py --help
"""

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

# Ensure the src directory is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from underwriting.agents.base_agent import BaseAgent
from underwriting.agents.compliance_agent import ComplianceAgent
from underwriting.agents.debate_orchestrator import DebateOrchestrator
from underwriting.agents.financial_agent import FinancialAgent
from underwriting.agents.medical_agent import MedicalAgent
from underwriting.application.schema import (
    Application,
    BenefitType,
    FamilyHistoryCondition,
    HazardousPursuit,
    MedicalCondition,
    SmokerStatus,
)
from underwriting.audit.logger import DecisionLogger
from underwriting.audit.report_generator import ReportGenerator

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("pipeline")

# Suppress noisy agent-level debug logs during demo
logging.getLogger("underwriting.agents").setLevel(logging.WARNING)
logging.getLogger("underwriting.rules").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Synthetic applicant loader
# ---------------------------------------------------------------------------

def _benefit_str_to_enum(benefit_str: str) -> BenefitType:
    """Convert a benefit type string from JSON to a BenefitType enum.

    Args:
        benefit_str: String value from the synthetic applicant JSON
            (e.g. ``"Death"``, ``"TPD"``).

    Returns:
        Corresponding :class:`BenefitType` enum member.
    """
    mapping: Dict[str, BenefitType] = {
        "Death": BenefitType.DEATH,
        "TPD": BenefitType.TPD,
        "Trauma/CI": BenefitType.TRAUMA,
        "Income Protection": BenefitType.IP,
    }
    return mapping[benefit_str]


def load_applicant_data(filepath: str) -> Dict[str, Any]:
    """Load synthetic applicant data from a JSON file.

    Args:
        filepath: Path to the synthetic applicant JSON file.

    Returns:
        Dictionary with all applicant fields.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Applicant file not found: {filepath}")

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_application(data: Dict[str, Any]) -> Application:
    """Construct an :class:`Application` Pydantic model from raw applicant data.

    Maps JSON keys to the Application schema, handling nested models
    (medical conditions, family history, hazardous pursuits) and
    enum conversions.

    Args:
        data: Dictionary of applicant fields loaded from JSON.

    Returns:
        A validated :class:`Application` instance.
    """
    # Parse nested medical conditions
    medical_conditions: List[MedicalCondition] = []
    for cond in data.get("medical_conditions") or []:
        medical_conditions.append(
            MedicalCondition(
                condition_name=cond["condition_name"],
                diagnosis_date=date.fromisoformat(cond["diagnosis_date"]),
                treating_doctor_name=cond["treating_doctor_name"],
                treating_doctor_contact=cond["treating_doctor_contact"],
                diagnostic_tests=cond.get("diagnostic_tests"),
                treatment_start_date=(
                    date.fromisoformat(cond["treatment_start_date"])
                    if cond.get("treatment_start_date")
                    else None
                ),
                treatment_description=cond.get("treatment_description"),
                symptoms=cond.get("symptoms"),
                symptom_frequency=cond.get("symptom_frequency"),
                last_symptom_date=(
                    date.fromisoformat(cond["last_symptom_date"])
                    if cond.get("last_symptom_date")
                    else None
                ),
                hospitalisations=cond.get("hospitalisations"),
                time_off_work=cond.get("time_off_work"),
                lifestyle_affected=cond.get("lifestyle_affected"),
            )
        )

    # Parse nested family history
    family_history: List[FamilyHistoryCondition] = []
    for fh in data.get("family_history") or []:
        family_history.append(
            FamilyHistoryCondition(
                relationship=fh["relationship"],
                condition=fh["condition"],
                age_at_diagnosis=fh["age_at_diagnosis"],
            )
        )

    # Parse hazardous pursuits
    hazardous_pursuits: List[HazardousPursuit] = []
    for hp in data.get("hazardous_pursuits") or []:
        hazardous_pursuits.append(
            HazardousPursuit(
                activity=hp["activity"],
                frequency=hp["frequency"],
                level=hp["level"],
            )
        )

    # Convert benefit type strings to enums
    benefit_types: List[BenefitType] = [
        _benefit_str_to_enum(bt)
        for bt in data.get("benefit_types") or []
    ]

    return Application(
        # Section A: Personal & Demographic
        full_name=data["full_name"],
        date_of_birth=date.fromisoformat(data["date_of_birth"]),
        gender=data["gender"],
        residency_status=data["residency_status"],
        contact_address=data["contact_address"],
        # Section B: Cover Requested
        benefit_types=benefit_types,
        sum_insured_death=data.get("sum_insured_death"),
        sum_insured_tpd=data.get("sum_insured_tpd"),
        sum_insured_trauma=data.get("sum_insured_trauma"),
        ip_monthly_benefit=data.get("ip_monthly_benefit"),
        ip_benefit_period=data.get("ip_benefit_period"),
        ip_agreed_value=data.get("ip_agreed_value"),
        has_other_policies=data.get("has_other_policies", False),
        total_existing_policies=data.get("total_existing_policies", 0),
        other_policy_details=data.get("other_policy_details"),
        previous_declination=data.get("previous_declination", False),
        # Section C: Occupation & Income
        occupation=data["occupation"],
        employer_name=data["employer_name"],
        years_in_occupation=data["years_in_occupation"],
        annual_income=data["annual_income"],
        has_hazardous_duties=data.get("has_hazardous_duties", False),
        hazardous_duties_description=data.get("hazardous_duties_description"),
        # Section D: Health — General
        height_cm=data["height_cm"],
        weight_kg=data["weight_kg"],
        smoker_status=SmokerStatus(data["smoker_status"]),
        cigarettes_per_day=data.get("cigarettes_per_day"),
        years_smoked=data.get("years_smoked"),
        years_since_quit=data.get("years_since_quit"),
        taking_medications=data.get("taking_medications", False),
        medication_details=data.get("medication_details"),
        has_medical_conditions=data.get("has_medical_conditions", False),
        medical_conditions=medical_conditions,
        consumes_alcohol=data.get("consumes_alcohol", False),
        standard_drinks_per_week=data.get("standard_drinks_per_week"),
        # Section F: Family History
        has_family_history=data.get("has_family_history", False),
        family_history=family_history,
        # Section G: Lifestyle
        has_hazardous_pursuits=data.get("has_hazardous_pursuits", False),
        hazardous_pursuits=hazardous_pursuits,
        recreational_drug_use=data.get("recreational_drug_use", False),
        drug_use_details=data.get("drug_use_details"),
        alcohol_drug_treatment=data.get("alcohol_drug_treatment", False),
        has_high_risk_travel=data.get("has_high_risk_travel", False),
        high_risk_travel_details=data.get("high_risk_travel_details"),
        # Section H: Financial
        total_net_worth=data.get("total_net_worth"),
        financial_obligations=data.get("financial_obligations"),
        obligation_end_dates=data.get("obligation_end_dates"),
        bankruptcy_status=data.get("bankruptcy_status", "None"),
        previous_bankruptcy=data.get("previous_bankruptcy", False),
        criminal_convictions=data.get("criminal_convictions", False),
        # Compliance
        duty_of_disclosure_acknowledged=data.get(
            "duty_of_disclosure_acknowledged", False
        ),
    )


# ---------------------------------------------------------------------------
# Agent instantiation & pipeline execution
# ---------------------------------------------------------------------------

def create_agents(application: Application) -> List[BaseAgent]:
    """Create the three underwriting agents for the given application.

    Uses the benefit types requested by the applicant to select the
    appropriate rule files from ``rules/<benefit_type>/``.

    Args:
        application: The validated Application model.

    Returns:
        List containing the MedicalAgent, FinancialAgent, and
        ComplianceAgent instances (in execution order).
    """
    # Determine which rules directory to use based on the first requested benefit
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
        rules_dir = "rules/death"  # fallback

    medical_agent = MedicalAgent(
        rules_path=os.path.join(rules_dir, "medical_rules.json"),
    )
    financial_agent = FinancialAgent(
        rules_path=os.path.join(rules_dir, "financial_rules.json"),
    )
    compliance_agent = ComplianceAgent(
        rules_path=os.path.join(rules_dir, "compliance_rules.json"),
    )

    return [medical_agent, financial_agent, compliance_agent]


def run_pipeline(
    application: Application,
    logger_: DecisionLogger,
) -> Dict[str, Any]:
    """Execute the full multi-agent underwriting pipeline.

    Runs Medical → Financial → Compliance agents, then the Debate
    Orchestrator, logs every step, and returns the final results.

    Args:
        application: The validated Application model.
        logger_: The :class:`DecisionLogger` instance for audit trail.

    Returns:
        Dictionary containing the orchestrator results:
        ``final_assessment``, ``agent_assessments``, ``debate_log``,
        ``consensus_reached``, ``final_decision``, ``decision_reasoning``.
    """
    # Phase 1: Create agents
    agents = create_agents(application)

    # Phase 2: Run debate orchestrator
    orchestrator = DebateOrchestrator(agents=agents)
    results = orchestrator.run(application)

    # Phase 3: Log everything
    _log_agent_assessments(logger_, application, results)
    _log_debate_rounds(logger_, results)
    _log_final_decision(logger_, results)

    return results


def _log_agent_assessments(
    logger_: DecisionLogger,
    application: Application,
    results: Dict[str, Any],
) -> None:
    """Log each agent's assessment to the JSONL audit trail.

    Args:
        logger_: The :class:`DecisionLogger` instance.
        application: The evaluated Application model.
        results: Orchestrator results dict.
    """
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


def _log_debate_rounds(
    logger_: DecisionLogger,
    results: Dict[str, Any],
) -> None:
    """Log debate rounds to the JSONL audit trail.

    Args:
        logger_: The :class:`DecisionLogger` instance.
        results: Orchestrator results dict.
    """
    debate_log = results.get("debate_log", [])
    if debate_log:
        for round_entry in debate_log:
            logger_.log_debate_round(round_entry)
    else:
        logger_.log_event("debate_status", {
            "status": "no_debate_needed",
            "reason": "All agents reached consensus",
        })


def _log_final_decision(
    logger_: DecisionLogger,
    results: Dict[str, Any],
) -> None:
    """Log the final underwriting decision to the JSONL audit trail.

    Args:
        logger_: The :class:`DecisionLogger` instance.
        results: Orchestrator results dict.
    """
    decision_data = {
        "final_decision": results.get("final_decision", "Unknown"),
        "consensus_reached": results.get("consensus_reached", False),
        "decision_reasoning": results.get("decision_reasoning", ""),
        "debate_rounds": len(results.get("debate_log", [])),
    }
    logger_.log_final_decision(decision_data)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_reports(
    results: Dict[str, Any],
    application: Application,
    output_dir: str = "./audit_reports",
) -> str:
    """Generate markdown and JSON audit reports.

    Saves a markdown report and a JSON summary to *output_dir*.

    Args:
        results: Orchestrator results dict.
        application: The evaluated Application model.
        output_dir: Directory to save the reports.

    Returns:
        Path to the generated markdown report file.
    """
    generator = ReportGenerator(output_dir=output_dir)

    # Build decision data for the report generator
    decision_data: Dict[str, Any] = {
        "applicant": {
            "full_name": application.full_name,
            "date_of_birth": str(application.date_of_birth),
            "age": application.age,
            "gender": application.gender,
            "residency_status": application.residency_status,
            "occupation": application.occupation,
            "annual_income": application.annual_income,
            "bmi": application.bmi,
            "smoker_status": application.smoker_status.value,
            "benefit_types": [bt.value for bt in application.benefit_types],
            "sum_insured_death": application.sum_insured_death or 0,
        },
        "agent_assessments": {
            name: assessment.model_dump()
            for name, assessment in results.get("agent_assessments", {}).items()
        },
        "debate_log": results.get("debate_log", []),
        "final_decision": results.get("final_decision", "Unknown"),
        "decision_reasoning": results.get("decision_reasoning", ""),
        "consensus_reached": results.get("consensus_reached", False),
        "flags": results.get("flags", []),
        "additional_evidence_required": results.get(
            "additional_evidence_required", []
        ),
    }

    # Generate and save markdown report
    report_filename = f"audit_report_{application.full_name.replace(' ', '_')}.md"
    markdown_path = generator.save_report(decision_data, report_filename)

    # Also save a JSON summary alongside
    json_dir = os.path.join(output_dir, "json")
    os.makedirs(json_dir, exist_ok=True)
    json_filename = f"decision_{application.full_name.replace(' ', '_')}.json"
    json_path = os.path.join(json_dir, json_filename)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(decision_data, fh, indent=2, default=str)

    logger.info("Reports saved: %s, %s", markdown_path, json_path)
    return markdown_path


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(
    application: Application,
    results: Dict[str, Any],
) -> None:
    """Print a human-readable summary to the console.

    Displays applicant info, agent assessments, debate outcome,
    and the final decision in a formatted block.

    Args:
        application: The evaluated Application model.
        results: Orchestrator results dict.
    """
    print(f"\n{'=' * 70}")
    print("  MULTI-AGENT UNDERWRITING RULES ENGINE - PIPELINE SUMMARY")
    print(f"{'=' * 70}")

    # Applicant info
    print("\n  [APPLICANT] APPLICANT")
    print(f"     Name:          {application.full_name}")
    print(f"     Date of Birth: {application.date_of_birth}")
    print(f"     Age:           {application.age}")
    print(f"     Gender:        {application.gender}")
    print(f"     Occupation:    {application.occupation}")
    print(f"     Income:        ${application.annual_income:,.2f}")
    print(f"     BMI:           {application.bmi}")
    print(f"     Smoker:        {application.smoker_status.value}")
    print(f"     Benefits:      {', '.join(bt.value for bt in application.benefit_types)}")

    # Agent assessments
    print("\n  [AGENTS] AGENT ASSESSMENTS")
    agent_assessments = results.get("agent_assessments", {})
    for agent_name, assessment in agent_assessments.items():
        print(f"\n     +-- {agent_name}")
        print(f"     | Risk Tier:       {assessment.risk_tier}")
        print(f"     | Recommendation:  {assessment.recommendation}")
        print(f"     | Confidence:      {assessment.confidence_score:.0%}")
        print(f"     | Flags:           {len(assessment.flags)}")
        if assessment.flags:
            for flag in assessment.flags:
                severity = flag.get("severity", "unknown")
                rule_id = flag.get("rule_id", "N/A")
                desc = flag.get("description", "")
                print(f"     |   [{severity}] {rule_id}: {desc}")
        if assessment.additional_evidence_required:
            print(f"     | Evidence Required:")
            for ev in assessment.additional_evidence_required:
                print(f"     |   - {ev}")
        print(f"     +--------------------------------")

    # Debate
    debate_log = results.get("debate_log", [])
    print("\n  [DEBATE] DEBATE")
    if debate_log:
        for entry in debate_log:
            round_num = entry.get("round", "?")
            agent = entry.get("agent", "?")
            original = entry.get("original_tier", [])
            updated = entry.get("updated_tier", "?")
            print(f"     Round {round_num}: {agent} -- tiers {original} -> {updated}")
    else:
        print("     No debate needed -- consensus reached.")

    # Final decision
    print("\n  [DECISION] FINAL DECISION")
    final_decision = results.get("final_decision", "Unknown")
    consensus = results.get("consensus_reached", False)
    reasoning = results.get("decision_reasoning", "")
    print(f"     Decision:       {final_decision}")
    print(f"     Consensus:      {'Yes' if consensus else 'No'}")
    print(f"     Reasoning:      {reasoning}")
    print(f"{'=' * 70}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description=(
            "Run the full multi-agent underwriting pipeline on a "
            "synthetic applicant profile."
        ),
        epilog=(
            "Examples:\n"
            "  python demo/pipeline.py --applicant data/synthetic_applicants/standard.json\n"
            "  python demo/pipeline.py --applicant data/synthetic_applicants/moderate.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--applicant",
        type=str,
        required=True,
        help=(
            "Path to a synthetic applicant JSON file. "
            "Example: data/synthetic_applicants/standard.json"
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./audit_logs",
        help="Directory for JSONL audit logs (default: ./audit_logs)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="./audit_reports",
        help="Directory for audit reports (default: ./audit_reports)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Entry point for the underwriting pipeline demo.

    Parses CLI arguments, loads the applicant, runs the full pipeline,
    generates reports, and prints a console summary.

    Args:
        argv: Optional argument list for testing.
    """
    args = parse_args(argv)

    print(f"\n[LOAD] Loading applicant from: {args.applicant}")

    # Load applicant data
    applicant_data = load_applicant_data(args.applicant)
    application = build_application(applicant_data)
    print(f"[OK] Applicant loaded: {application.full_name} (age {application.age})")

    # Initialise audit logger
    logger_ = DecisionLogger(log_dir=args.log_dir)
    print(f"[LOG] Audit log: {logger_.log_path}")

    # Run pipeline
    print("\n[RUN] Running underwriting pipeline...")
    results = run_pipeline(application, logger_)
    print("[OK] Pipeline complete.")

    # Generate reports
    report_path = generate_reports(results, application, args.report_dir)
    print(f"[RPT] Report saved: {report_path}")

    # Print summary
    print_summary(application, results)


if __name__ == "__main__":
    main()
