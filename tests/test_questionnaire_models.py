"""Tests for QuestionnaireDefinition model."""
from datetime import date

import pytest
import yaml
from pydantic import ValidationError

from underwriting.application.schema import (
    Application,
    BenefitType,
    FamilyHistoryCondition,
    HazardousPursuit,
    MedicalCondition,
    SmokerStatus,
)
from underwriting.test_questionnaire.models import QuestionnaireDefinition

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_QD = {
    "name": "Standard Life Cover",
    "description": "Standard life insurance questionnaire",
    "benefit_types": [BenefitType.DEATH, BenefitType.TPD],
    "agent_names": ["MedicalAgent", "FinancialAgent"],
    "full_name": "Jane Doe",
    "date_of_birth": date(1990, 6, 15),
    "gender": "Female",
    "residency_status": "Australian Citizen",
    "contact_address": "123 Main St, Sydney NSW 2000",
    "sum_insured_death": 500_000,
    "sum_insured_tpd": 500_000,
    "occupation": "Software Engineer",
    "employer_name": "Tech Corp",
    "years_in_occupation": 5.0,
    "annual_income": 120_000,
    "height_cm": 165.0,
    "weight_kg": 68.0,
    "smoker_status": SmokerStatus.NEVER,
    "taking_medications": False,
    "consumes_alcohol": False,
    "has_medical_conditions": False,
    "has_family_history": False,
    "has_hazardous_pursuits": False,
    "recreational_drug_use": False,
    "alcohol_drug_treatment": False,
    "has_high_risk_travel": False,
    "bankruptcy_status": "None",
    "previous_bankruptcy": False,
    "criminal_convictions": False,
    "duty_of_disclosure_acknowledged": True,
}


FULL_QD = {
    **MINIMAL_QD,
    "sum_insured_trauma": 100_000,
    "ip_monthly_benefit": 5_000,
    "ip_benefit_period": 240,
    "ip_agreed_value": True,
    "has_other_policies": True,
    "total_existing_policies": 2,
    "other_policy_details": "Policy A with InsCo X, Policy B with InsCo Y",
    "previous_declination": False,
    "has_hazardous_duties": True,
    "hazardous_duties_description": "Working at heights on construction sites",
    "cigarettes_per_day": 10,
    "years_smoked": 15,
    "years_since_quit": None,
    "smoker_status": SmokerStatus.CURRENT,
    "taking_medications": True,
    "medication_details": "Lisinopril 10mg daily for hypertension",
    "has_medical_conditions": True,
    "medical_conditions": [
        MedicalCondition(
            condition_name="Hypertension",
            diagnosis_date=date(2018, 3, 1),
            treating_doctor_name="Dr Smith",
            treating_doctor_contact="0400 000 000",
            symptoms="Occasional headaches",
            symptom_frequency="Weekly",
        ),
    ],
    "consumes_alcohol": True,
    "standard_drinks_per_week": 8,
    "has_family_history": True,
    "family_history": [
        FamilyHistoryCondition(
            relationship="father",
            condition="heart disease",
            age_at_diagnosis=58,
        ),
        FamilyHistoryCondition(
            relationship="mother",
            condition="type 2 diabetes",
            age_at_diagnosis=62,
        ),
    ],
    "has_hazardous_pursuits": True,
    "hazardous_pursuits": [
        HazardousPursuit(activity="Scuba diving", frequency="Monthly", level="amateur"),
    ],
    "recreational_drug_use": False,
    "alcohol_drug_treatment": False,
    "has_high_risk_travel": True,
    "high_risk_travel_details": "Planning travel to remote areas in Papua New Guinea",
    "total_net_worth": 1_500_000,
    "financial_obligations": "Mortgage $800k, car loan $30k",
    "obligation_end_dates": "Mortgage 2035, car loan 2028",
    "bankruptcy_status": "None",
    "previous_bankruptcy": False,
    "criminal_convictions": False,
    "duty_of_disclosure_acknowledged": True,
}


SAMPLE_YAML = """\
name: Standard Life Cover
description: Standard life insurance questionnaire
benefit_types:
- Death
- TPD
agent_names:
- MedicalAgent
- FinancialAgent
full_name: Jane Doe
date_of_birth: '1990-06-15'
gender: Female
residency_status: Australian Citizen
contact_address: 123 Main St, Sydney NSW 2000
sum_insured_death: 500000.0
sum_insured_tpd: 500000.0
occupation: Software Engineer
employer_name: Tech Corp
years_in_occupation: 5.0
annual_income: 120000.0
height_cm: 165.0
weight_kg: 68.0
smoker_status: Never
taking_medications: false
consumes_alcohol: false
has_medical_conditions: false
has_family_history: false
has_hazardous_pursuits: false
recreational_drug_use: false
alcohol_drug_treatment: false
has_high_risk_travel: false
bankruptcy_status: None
previous_bankruptcy: false
criminal_convictions: false
duty_of_disclosure_acknowledged: true
"""


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

class TestQuestionnaireDefinitionConstruction:
    def test_minimal_construction(self):
        qd = QuestionnaireDefinition(**MINIMAL_QD)
        assert qd.name == "Standard Life Cover"
        assert qd.full_name == "Jane Doe"
        assert qd.sum_insured_death == 500_000

    def test_full_construction(self):
        qd = QuestionnaireDefinition(**FULL_QD)
        assert len(qd.medical_conditions) == 1
        assert len(qd.family_history) == 2
        assert len(qd.hazardous_pursuits) == 1
        assert qd.smoker_status == SmokerStatus.CURRENT

    def test_default_values(self):
        qd = QuestionnaireDefinition(**MINIMAL_QD)
        assert qd.has_other_policies is False
        assert qd.total_existing_policies == 0
        assert qd.previous_declination is False
        assert qd.has_hazardous_duties is False
        assert qd.taking_medications is False
        assert qd.medication_details is None
        assert qd.medical_conditions == []
        assert qd.has_family_history is False
        assert qd.family_history == []
        assert qd.has_hazardous_pursuits is False
        assert qd.hazardous_pursuits == []
        assert qd.recreational_drug_use is False
        assert qd.drug_use_details is None
        assert qd.alcohol_drug_treatment is False
        assert qd.has_high_risk_travel is False
        assert qd.high_risk_travel_details is None
        assert qd.total_net_worth is None
        assert qd.financial_obligations is None
        assert qd.obligation_end_dates is None
        assert qd.previous_bankruptcy is False
        assert qd.criminal_convictions is False

    def test_benefit_types_enum(self):
        qd = QuestionnaireDefinition(**MINIMAL_QD)
        assert BenefitType.DEATH in qd.benefit_types
        assert BenefitType.TPD in qd.benefit_types

    def test_gender_validation(self):
        with pytest.raises(ValidationError):
            QuestionnaireDefinition(**{**MINIMAL_QD, "gender": "Other"})  # type: ignore

    def test_residency_validation(self):
        with pytest.raises(ValidationError):
            QuestionnaireDefinition(**{**MINIMAL_QD, "residency_status": "Visitor"})  # type: ignore

    def test_smoker_status_validation(self):
        with pytest.raises(ValidationError):
            QuestionnaireDefinition(**{**MINIMAL_QD, "smoker_status": "Unknown"})  # type: ignore

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            QuestionnaireDefinition(**{**MINIMAL_QD, "unknown_field": "value"})  # type: ignore

    def test_agent_names_optional(self):
        qd = QuestionnaireDefinition(**{**MINIMAL_QD, "agent_names": None})
        assert qd.agent_names is None

        qd2 = QuestionnaireDefinition(**{**MINIMAL_QD, "agent_names": ["MedicalAgent"]})
        assert qd2.agent_names == ["MedicalAgent"]


# ---------------------------------------------------------------------------
# YAML load / save
# ---------------------------------------------------------------------------

class TestYamlLoadSave:
    def test_save_yaml(self, tmp_path):
        qd = QuestionnaireDefinition(**MINIMAL_QD)
        yaml_path = tmp_path / "test_qd.yaml"
        qd.to_yaml(str(yaml_path))
        assert yaml_path.exists()

    def test_save_yaml_creates_directories(self, tmp_path):
        qd = QuestionnaireDefinition(**MINIMAL_QD)
        yaml_path = tmp_path / "nested" / "dir" / "test_qd.yaml"
        qd.to_yaml(str(yaml_path))
        assert yaml_path.exists()

    def test_save_yaml_content(self, tmp_path):
        qd = QuestionnaireDefinition(**MINIMAL_QD)
        yaml_path = tmp_path / "test_qd.yaml"
        qd.to_yaml(str(yaml_path))
        with open(yaml_path, "r") as f:
            saved = yaml.safe_load(f)
        assert saved["name"] == "Standard Life Cover"
        assert saved["full_name"] == "Jane Doe"
        assert saved["sum_insured_death"] == 500_000.0

    def test_load_yaml(self, tmp_path):
        yaml_path = tmp_path / "test_qd.yaml"
        with open(yaml_path, "w") as f:
            f.write(SAMPLE_YAML)
        qd = QuestionnaireDefinition.from_yaml(str(yaml_path))
        assert qd.name == "Standard Life Cover"
        assert qd.full_name == "Jane Doe"
        assert qd.date_of_birth == date(1990, 6, 15)
        assert qd.smoker_status == SmokerStatus.NEVER
        assert BenefitType.DEATH in qd.benefit_types

    def test_round_trip(self, tmp_path):
        qd = QuestionnaireDefinition(**FULL_QD)
        yaml_path = tmp_path / "roundtrip.yaml"
        qd.to_yaml(str(yaml_path))
        loaded = QuestionnaireDefinition.from_yaml(str(yaml_path))
        assert loaded.name == qd.name
        assert loaded.full_name == qd.full_name
        assert loaded.sum_insured_death == qd.sum_insured_death
        assert len(loaded.medical_conditions) == len(qd.medical_conditions)
        assert len(loaded.family_history) == len(qd.family_history)
        assert len(loaded.hazardous_pursuits) == len(qd.hazardous_pursuits)
        assert loaded.smoker_status == qd.smoker_status

    def test_load_yaml_with_nested_models(self, tmp_path):
        yaml_with_nested = """\
name: Full Questionnaire
description: Full test questionnaire
benefit_types:
- Death
- TPD
- Trauma/CI
full_name: John Smith
date_of_birth: '1985-03-20'
gender: Male
residency_status: Permanent Resident
contact_address: 456 Oak Rd, Melbourne VIC 3000
sum_insured_death: 750000
sum_insured_tpd: 750000
sum_insured_trauma: 150000
occupation: Doctor
employer_name: City Hospital
years_in_occupation: 10.0
annual_income: 250000
height_cm: 178.0
weight_kg: 75.0
smoker_status: Former
taking_medications: false
consumes_alcohol: true
standard_drinks_per_week: 5
has_medical_conditions: true
medical_conditions:
- condition_name: Asthma
  diagnosis_date: '2010-05-15'
  treating_doctor_name: Dr Jones
  treating_doctor_contact: '0411 000 000'
  symptoms: Exercise-induced
has_family_history: true
family_history:
- relationship: mother
  condition: breast cancer
  age_at_diagnosis: 52
has_hazardous_pursuits: true
hazardous_pursuits:
- activity: Rock climbing
  frequency: Weekly
  level: amateur
recreational_drug_use: false
alcohol_drug_treatment: false
has_high_risk_travel: false
bankruptcy_status: None
previous_bankruptcy: false
criminal_convictions: false
duty_of_disclosure_acknowledged: true
"""
        yaml_path = tmp_path / "nested.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_with_nested)
        qd = QuestionnaireDefinition.from_yaml(str(yaml_path))
        assert len(qd.medical_conditions) == 1
        assert qd.medical_conditions[0].condition_name == "Asthma"
        assert len(qd.family_history) == 1
        assert qd.family_history[0].relationship == "mother"
        assert len(qd.hazardous_pursuits) == 1
        assert qd.hazardous_pursuits[0].activity == "Rock climbing"


# ---------------------------------------------------------------------------
# Conversion to Application
# ---------------------------------------------------------------------------

class TestToApplication:
    def test_minimal_conversion(self):
        qd = QuestionnaireDefinition(**MINIMAL_QD)
        app = qd.to_application()
        assert isinstance(app, Application)
        assert app.full_name == "Jane Doe"
        assert app.date_of_birth == date(1990, 6, 15)
        assert app.gender == "Female"
        assert app.residency_status == "Australian Citizen"
        assert app.contact_address == "123 Main St, Sydney NSW 2000"
        assert BenefitType.DEATH in app.benefit_types
        assert BenefitType.TPD in app.benefit_types
        assert app.sum_insured_death == 500_000
        assert app.sum_insured_tpd == 500_000
        assert app.occupation == "Software Engineer"
        assert app.employer_name == "Tech Corp"
        assert app.years_in_occupation == 5.0
        assert app.annual_income == 120_000
        assert app.height_cm == 165.0
        assert app.weight_kg == 68.0
        assert app.smoker_status == SmokerStatus.NEVER

    def test_full_conversion(self):
        qd = QuestionnaireDefinition(**FULL_QD)
        app = qd.to_application()
        assert app.sum_insured_trauma == 100_000
        assert app.ip_monthly_benefit == 5_000
        assert app.ip_agreed_value is True
        assert app.has_other_policies is True
        assert app.total_existing_policies == 2
        assert app.other_policy_details == "Policy A with InsCo X, Policy B with InsCo Y"
        assert app.has_hazardous_duties is True
        assert app.hazardous_duties_description == "Working at heights on construction sites"
        assert app.cigarettes_per_day == 10
        assert app.years_smoked == 15
        assert app.taking_medications is True
        assert app.medication_details == "Lisinopril 10mg daily for hypertension"
        assert app.has_medical_conditions is True
        assert len(app.medical_conditions) == 1
        assert app.consumes_alcohol is True
        assert app.standard_drinks_per_week == 8
        assert app.has_family_history is True
        assert len(app.family_history) == 2
        assert app.has_hazardous_pursuits is True
        assert len(app.hazardous_pursuits) == 1
        assert app.has_high_risk_travel is True
        assert app.high_risk_travel_details == "Planning travel to remote areas in Papua New Guinea"
        assert app.total_net_worth == 1_500_000
        assert app.financial_obligations == "Mortgage $800k, car loan $30k"
        assert app.obligation_end_dates == "Mortgage 2035, car loan 2028"
        assert app.duty_of_disclosure_acknowledged is True

    def test_ip_benefit_period_conversion(self):
        qd = QuestionnaireDefinition(**{
            **MINIMAL_QD,
            "ip_benefit_period": 240,
        })
        app = qd.to_application()
        assert app.ip_benefit_period == "240"

    def test_ip_benefit_period_none(self):
        qd = QuestionnaireDefinition(**MINIMAL_QD)
        app = qd.to_application()
        assert app.ip_benefit_period is None

    def test_years_since_quit_conversion(self):
        qd = QuestionnaireDefinition(**{
            **MINIMAL_QD,
            "smoker_status": SmokerStatus.FORMER,
            "years_since_quit": 5,
        })
        app = qd.to_application()
        assert app.years_since_quit == 5.0

    def test_years_since_quit_none(self):
        qd = QuestionnaireDefinition(**MINIMAL_QD)
        app = qd.to_application()
        assert app.years_since_quit is None

    def test_conversion_preserves_defaults(self):
        qd = QuestionnaireDefinition(**MINIMAL_QD)
        app = qd.to_application()
        assert app.has_other_policies is False
        assert app.previous_declination is False
        assert app.previous_bankruptcy is False
        assert app.criminal_convictions is False

    def test_round_trip_via_yaml(self, tmp_path):
        qd = QuestionnaireDefinition(**FULL_QD)
        yaml_path = tmp_path / "roundtrip.yaml"
        qd.to_yaml(str(yaml_path))
        loaded = QuestionnaireDefinition.from_yaml(str(yaml_path))
        app = loaded.to_application()
        assert app.full_name == qd.full_name
        assert app.sum_insured_death == qd.sum_insured_death
        assert len(app.medical_conditions) == len(qd.medical_conditions)
        assert app.smoker_status == qd.smoker_status

    def test_computed_fields_on_converted_application(self):
        qd = QuestionnaireDefinition(**MINIMAL_QD)
        app = qd.to_application()
        assert isinstance(app.age, int)
        assert isinstance(app.bmi, float)
        assert app.bmi == round(68.0 / ((165.0 / 100) ** 2), 1)
