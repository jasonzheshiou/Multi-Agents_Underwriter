# Shared test fixtures and configuration for the underwriting rules engine.

import tempfile
from datetime import date
from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from underwriting.application.schema import (
    Application,
    BenefitType,
    FamilyHistoryCondition,
    HazardousPursuit,
    MedicalCondition,
    SmokerStatus,
)
from underwriting.config import load_config

# ---------------------------------------------------------------------------
# Applicant fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_applicant() -> Application:
    """Return a basic standard-risk applicant.

    BMI is approximately 24, non-smoker, no medical conditions, and no
    hazardous pursuits — the profile that should receive a standard offer.
    """
    return Application(
        # Section A: Personal & Demographic
        full_name="Alex Standard",
        date_of_birth=date(1990, 6, 15),
        gender="Male",
        residency_status="Australian Citizen",
        contact_address="12 Main St, Sydney NSW 2000",
        # Section B: Cover Requested
        benefit_types=[BenefitType.DEATH],
        sum_insured_death=500_000.0,
        sum_insured_tpd=500_000.0,
        has_other_policies=False,
        previous_declination=False,
        # Section C: Occupation & Income
        occupation="Software Manager",
        employer_name="TechCorp Pty Ltd",
        years_in_occupation=8.0,
        annual_income=120_000.0,
        has_hazardous_duties=False,
        # Section D: Health — General
        height_cm=178.0,
        weight_kg=76.0,
        smoker_status=SmokerStatus.NEVER,
        taking_medications=False,
        has_medical_conditions=False,
        medical_conditions=[],
        consumes_alcohol=False,
        standard_drinks_per_week=0,
        # Section F: Family History
        has_family_history=False,
        family_history=[],
        # Section G: Lifestyle
        has_hazardous_pursuits=False,
        hazardous_pursuits=[],
        recreational_drug_use=False,
        alcohol_drug_treatment=False,
        has_high_risk_travel=False,
        # Section H: Financial
        total_net_worth=350_000.0,
        financial_obligations="Mortgage $300k",
        obligation_end_dates="2035-12-31",
        previous_bankruptcy=False,
        criminal_convictions=False,
        # Compliance
        duty_of_disclosure_acknowledged=True,
    )


@pytest.fixture()
def complex_applicant() -> Application:
    """Return an applicant with multiple moderate risk factors.

    Profile: construction manager (heavy manual), former smoker,
    hypertension, family history of heart disease.
    """
    return Application(
        full_name="Jordan Builder",
        date_of_birth=date(1985, 3, 22),
        gender="Male",
        residency_status="Australian Citizen",
        contact_address="45 Hill Rd, Melbourne VIC 3000",
        benefit_types=[BenefitType.DEATH, BenefitType.TPD],
        sum_insured_death=600_000.0,
        sum_insured_tpd=600_000.0,
        has_other_policies=False,
        previous_declination=False,
        occupation="Construction Manager",
        employer_name="BuildRight Constructions",
        years_in_occupation=12.0,
        annual_income=145_000.0,
        has_hazardous_duties=False,
        height_cm=180.0,
        weight_kg=85.0,
        smoker_status=SmokerStatus.FORMER,
        cigarettes_per_day=10,
        years_smoked=15,
        years_since_quit=3.0,
        taking_medications=True,
        medication_details="Amlodipine 5 mg daily",
        has_medical_conditions=True,
        medical_conditions=[
            MedicalCondition(
                condition_name="Hypertension",
                diagnosis_date=date(2020, 7, 10),
                treating_doctor_name="Dr Sarah Lee",
                treating_doctor_contact="02 9999 1234",
                treatment_start_date=date(2020, 7, 15),
                treatment_description="Amlodipine 5 mg daily",
                symptoms="Occasional headaches",
                symptom_frequency="Monthly",
                last_symptom_date=date(2024, 11, 5),
                lifestyle_affected=False,
            ),
        ],
        consumes_alcohol=True,
        standard_drinks_per_week=8,
        has_family_history=True,
        family_history=[
            FamilyHistoryCondition(
                relationship="father",
                condition="Heart disease",
                age_at_diagnosis=55,
            ),
        ],
        has_hazardous_pursuits=False,
        hazardous_pursuits=[],
        recreational_drug_use=False,
        alcohol_drug_treatment=False,
        has_high_risk_travel=False,
        total_net_worth=400_000.0,
        financial_obligations="Mortgage $450k",
        obligation_end_dates="2040-06-30",
        previous_bankruptcy=False,
        criminal_convictions=False,
        duty_of_disclosure_acknowledged=True,
    )


@pytest.fixture()
def high_risk_applicant() -> Application:
    """Return an applicant with high-risk factors likely to trigger a decline.

    Profile: underground miner (hazardous), current smoker, Type 1
    diabetes, and hazardous pursuits (rock climbing, scuba diving).
    """
    return Application(
        full_name="Casey Miner",
        date_of_birth=date(1978, 11, 8),
        gender="Female",
        residency_status="Permanent Resident",
        contact_address="7 Mine Rd, Perth WA 6000",
        benefit_types=[BenefitType.DEATH, BenefitType.TPD, BenefitType.TRAUMA],
        sum_insured_death=800_000.0,
        sum_insured_tpd=800_000.0,
        sum_insured_trauma=100_000.0,
        has_other_policies=True,
        other_policy_details="Life policy with InsureCo $200k",
        previous_declination=False,
        occupation="Underground Miner",
        employer_name="DeepEarth Mining",
        years_in_occupation=15.0,
        annual_income=160_000.0,
        has_hazardous_duties=True,
        hazardous_duties_description="Works 120 m below surface, exposure to heavy machinery",
        height_cm=165.0,
        weight_kg=62.0,
        smoker_status=SmokerStatus.CURRENT,
        cigarettes_per_day=15,
        years_smoked=25,
        taking_medications=True,
        medication_details="Insulin glargine 20 units nightly",
        has_medical_conditions=True,
        medical_conditions=[
            MedicalCondition(
                condition_name="Type 1 Diabetes",
                diagnosis_date=date(1995, 4, 1),
                treating_doctor_name="Dr Michael Chen",
                treating_doctor_contact="08 9444 5678",
                treatment_start_date=date(1995, 4, 5),
                treatment_description="Insulin therapy",
                symptoms="Managed",
                symptom_frequency="Ongoing",
                lifestyle_affected=True,
            ),
        ],
        consumes_alcohol=False,
        standard_drinks_per_week=0,
        has_family_history=True,
        family_history=[
            FamilyHistoryCondition(
                relationship="mother",
                condition="Heart disease",
                age_at_diagnosis=50,
            ),
            FamilyHistoryCondition(
                relationship="brother",
                condition="Heart attack",
                age_at_diagnosis=48,
            ),
        ],
        has_hazardous_pursuits=True,
        hazardous_pursuits=[
            HazardousPursuit(
                activity="Rock climbing",
                frequency="Monthly",
                level="amateur",
            ),
            HazardousPursuit(
                activity="Scuba diving",
                frequency="Quarterly",
                level="amateur",
            ),
        ],
        recreational_drug_use=False,
        alcohol_drug_treatment=False,
        has_high_risk_travel=False,
        total_net_worth=250_000.0,
        financial_obligations="Mortgage $350k",
        obligation_end_dates="2038-09-30",
        previous_bankruptcy=False,
        criminal_convictions=False,
        duty_of_disclosure_acknowledged=True,
    )


# ---------------------------------------------------------------------------
# LLM mock
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_llm() -> MagicMock:
    """Return a pre-configured MagicMock acting as an LLM client.

    The mock returns a predictable JSON-like dict when called, allowing
    tests to verify LLM interaction without a real API connection.
    """
    client = MagicMock()
    client.chat_completion.return_value = {
        "model": "mock-model",
        "choices": [
            {
                "message": {
                    "content": (
                        '{"risk_tier": "standard", "reasoning_summary": '
                        '"Mock assessment for testing", "flags": []}'
                    ),
                },
            },
        ],
    }
    return client


# ---------------------------------------------------------------------------
# Temporary rules directory
# ---------------------------------------------------------------------------

@pytest.fixture()
def temp_rules_dir() -> str:
    """Create and return a temporary directory for test rule files.

    The directory is automatically cleaned up at the end of the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ---------------------------------------------------------------------------
# AgentAssessment factory
# ---------------------------------------------------------------------------

def _default_assessment(
    agent_name: str = "test_agent",
    risk_tier: str = "standard",
    flags: List[Dict[str, str]] | None = None,
    recommendation: str = "Approve",
    loading_range: List[float] | None = None,
    additional_evidence_required: List[str] | None = None,
    confidence_score: float = 0.95,
    reasoning_summary: str = "Test reasoning",
    apra_references: List[str] | None = None,
    llm_used: bool = False,
) -> BaseModel:
    """Create a minimal AgentAssessment-compatible Pydantic model.

    Because the actual ``AgentAssessment`` class lives in ``plan.md``
    (reference schema), this factory builds a compatible model so tests
    can assert on assessment fields without importing the production class.
    """
    from pydantic import create_model

    return create_model(
        "AgentAssessment",
        agent_name=(str, agent_name),
        risk_tier=(str, risk_tier),
        flags=(list, flags or []),
        recommendation=(str, recommendation),
        loading_range=(list, loading_range or [1.0, 1.0]),
        additional_evidence_required=(list, additional_evidence_required or []),
        confidence_score=(float, confidence_score),
        reasoning_summary=(str, reasoning_summary),
        apra_references=(list, apra_references or []),
        llm_used=(bool, llm_used),
    )


@pytest.fixture()
def agent_assessment_factory() -> Callable[..., BaseModel]:
    """Return a factory function that creates AgentAssessment models.

    Example::

        assessment = agent_assessment_factory(
            agent_name="medical",
            risk_tier="loading",
            flags=[{"rule_id": "MED-001", "severity": "medium", "description": "BMI > 30"}],
        )
    """
    return _default_assessment


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def config() -> Dict[str, Any]:
    """Load the project configuration from *config.yaml*.

    Uses the default path ``./config.yaml`` relative to the working
    directory.  Environment variable overrides are applied automatically
    by ``load_config``.
    """
    return load_config()
