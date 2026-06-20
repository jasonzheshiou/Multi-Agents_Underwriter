"""Questionnaire definition model for test questionnaire feature."""
from datetime import date
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field

from underwriting.application.schema import (
    Application,
    BenefitType,
    FamilyHistoryCondition,
    HazardousPursuit,
    MedicalCondition,
    SmokerStatus,
)


class QuestionnaireDefinition(BaseModel):
    """A questionnaire definition that can be converted into an Application.

    This model captures all fields needed to populate an
    :class:`Application` instance, with optional fields for
    benefits, health, occupation, lifestyle, and financial data.
    """

    name: str
    description: str
    benefit_types: List[BenefitType]
    agent_names: Optional[List[str]] = None
    full_name: str
    date_of_birth: date
    gender: Literal["Male", "Female", "Non-binary"]
    residency_status: Literal["Australian Citizen", "Permanent Resident", "Temporary Visa"]
    contact_address: str
    sum_insured_death: Optional[float] = None
    sum_insured_tpd: Optional[float] = None
    sum_insured_trauma: Optional[float] = None
    ip_monthly_benefit: Optional[float] = None
    ip_benefit_period: Optional[int] = None
    ip_agreed_value: Optional[bool] = None
    has_other_policies: bool = False
    total_existing_policies: int = 0
    other_policy_details: Optional[str] = None
    previous_declination: bool = False
    occupation: str
    employer_name: str
    years_in_occupation: float
    annual_income: float
    has_hazardous_duties: bool = False
    hazardous_duties_description: Optional[str] = None
    height_cm: float
    weight_kg: float
    smoker_status: SmokerStatus
    cigarettes_per_day: Optional[int] = None
    years_smoked: Optional[int] = None
    years_since_quit: Optional[int] = None
    taking_medications: bool = False
    medication_details: Optional[str] = None
    has_medical_conditions: bool = False
    medical_conditions: List[MedicalCondition] = Field(default_factory=list)
    consumes_alcohol: bool = False
    standard_drinks_per_week: Optional[int] = None
    has_family_history: bool = False
    family_history: List[FamilyHistoryCondition] = Field(default_factory=list)
    has_hazardous_pursuits: bool = False
    hazardous_pursuits: List[HazardousPursuit] = Field(default_factory=list)
    recreational_drug_use: bool = False
    drug_use_details: Optional[str] = None
    alcohol_drug_treatment: bool = False
    has_high_risk_travel: bool = False
    high_risk_travel_details: Optional[str] = None
    total_net_worth: Optional[float] = None
    financial_obligations: Optional[str] = None
    obligation_end_dates: Optional[str] = None
    bankruptcy_status: str = "None"
    previous_bankruptcy: bool = False
    criminal_convictions: bool = False
    duty_of_disclosure_acknowledged: bool = False

    model_config = {"extra": "forbid"}

    def to_application(self) -> Application:
        """Convert this questionnaire definition into an Application."""
        return Application(
            # Section A: Personal & Demographic
            full_name=self.full_name,
            date_of_birth=self.date_of_birth,
            gender=self.gender,
            residency_status=self.residency_status,
            contact_address=self.contact_address,
            # Section B: Cover Requested
            benefit_types=self.benefit_types,
            sum_insured_death=self.sum_insured_death,
            sum_insured_tpd=self.sum_insured_tpd,
            sum_insured_trauma=self.sum_insured_trauma,
            ip_monthly_benefit=self.ip_monthly_benefit,
            ip_benefit_period=str(self.ip_benefit_period) if self.ip_benefit_period is not None else None,
            ip_agreed_value=self.ip_agreed_value,
            has_other_policies=self.has_other_policies,
            total_existing_policies=self.total_existing_policies,
            other_policy_details=self.other_policy_details,
            previous_declination=self.previous_declination,
            # Section C: Occupation & Income
            occupation=self.occupation,
            employer_name=self.employer_name,
            years_in_occupation=self.years_in_occupation,
            annual_income=self.annual_income,
            has_hazardous_duties=self.has_hazardous_duties,
            hazardous_duties_description=self.hazardous_duties_description,
            # Section D: Health - General
            height_cm=self.height_cm,
            weight_kg=self.weight_kg,
            smoker_status=self.smoker_status,
            cigarettes_per_day=self.cigarettes_per_day,
            years_smoked=self.years_smoked,
            years_since_quit=float(self.years_since_quit) if self.years_since_quit is not None else None,
            taking_medications=self.taking_medications,
            medication_details=self.medication_details,
            has_medical_conditions=self.has_medical_conditions,
            medical_conditions=self.medical_conditions,
            consumes_alcohol=self.consumes_alcohol,
            standard_drinks_per_week=self.standard_drinks_per_week,
            # Section F: Family History
            has_family_history=self.has_family_history,
            family_history=self.family_history,
            # Section G: Lifestyle
            has_hazardous_pursuits=self.has_hazardous_pursuits,
            hazardous_pursuits=self.hazardous_pursuits,
            recreational_drug_use=self.recreational_drug_use,
            drug_use_details=self.drug_use_details,
            alcohol_drug_treatment=self.alcohol_drug_treatment,
            has_high_risk_travel=self.has_high_risk_travel,
            high_risk_travel_details=self.high_risk_travel_details,
            # Section H: Financial
            total_net_worth=self.total_net_worth,
            financial_obligations=self.financial_obligations,
            obligation_end_dates=self.obligation_end_dates,
            bankruptcy_status=self.bankruptcy_status,
            previous_bankruptcy=self.previous_bankruptcy,
            criminal_convictions=self.criminal_convictions,
            # Compliance
            duty_of_disclosure_acknowledged=self.duty_of_disclosure_acknowledged,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "QuestionnaireDefinition":
        """Load a QuestionnaireDefinition from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            A validated QuestionnaireDefinition instance.
        """
        yaml_path = Path(path)
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save this QuestionnaireDefinition to a YAML file.

        Args:
            path: Path to write the YAML file to.
        """
        yaml_path = Path(path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode='json', exclude_none=False)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
