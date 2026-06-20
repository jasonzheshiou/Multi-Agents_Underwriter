"""Pydantic v2 models for the underwriting application."""
from datetime import date
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, computed_field


class SmokerStatus(str, Enum):
    NEVER = "Never"
    FORMER = "Former"
    CURRENT = "Current"


class OccupationClass(str, Enum):
    PROFESSIONAL = "Professional"
    WHITE_COLLAR = "White Collar"
    LIGHT_MANUAL = "Light Manual"
    BLUE_COLLAR = "Blue Collar"
    HEAVY_MANUAL = "Heavy Manual"
    HAZARDOUS = "Hazardous"


class BenefitType(str, Enum):
    DEATH = "Death"
    TPD = "TPD"
    TRAUMA = "Trauma/CI"
    IP = "Income Protection"


class TPDDefinition(str, Enum):
    """TPD definition type."""

    ANY_OCCUPATION = "Any Occupation"
    OWN_OCCUPATION = "Own Occupation"


TRAUMA_CONDITION_OPTIONS = [
    "Cancer (any type)",
    "Heart attack",
    "Stroke",
    "Coronary artery bypass surgery",
    "Major organ failure",
    "Kidney failure",
    "Liver failure",
    "Loss of limbs",
    "Loss of vision",
    "Loss of hearing",
    "Paralysis",
    "Multiple sclerosis",
    "Parkinson's disease",
    "Coma",
    "Burns (third degree)",
]


class MedicalCondition(BaseModel):
    """A single pre-existing medical condition."""
    condition_name: str
    diagnosis_date: date
    treating_doctor_name: str
    treating_doctor_contact: str
    diagnostic_tests: Optional[str] = None
    treatment_start_date: Optional[date] = None
    treatment_description: Optional[str] = None
    symptoms: Optional[str] = None
    symptom_frequency: Optional[str] = None
    last_symptom_date: Optional[date] = None
    hospitalisations: Optional[str] = None
    time_off_work: Optional[str] = None
    lifestyle_affected: Optional[bool] = None


class FamilyHistoryCondition(BaseModel):
    """A single family history entry."""
    relationship: str  # "father", "mother", "brother", "sister"
    condition: str
    age_at_diagnosis: int


class HazardousPursuit(BaseModel):
    """A single hazardous activity."""
    activity: str
    frequency: str
    level: Literal["amateur", "professional"]


class Application(BaseModel):
    """Complete life insurance underwriting application."""

    # Section A: Personal & Demographic
    full_name: str
    date_of_birth: date
    gender: Literal["Male", "Female", "Non-binary"]
    residency_status: Literal["Australian Citizen", "Permanent Resident", "Temporary Visa"]
    contact_address: str

    # Section B: Cover Requested
    benefit_types: List[BenefitType]
    sum_insured_death: Optional[float] = None
    sum_insured_tpd: Optional[float] = None
    tpd_definition: Optional[str] = None
    sum_insured_trauma: Optional[float] = None
    trauma_condition_list: Optional[List[str]] = None
    ip_monthly_benefit: Optional[float] = None
    ip_benefit_period: Optional[str] = None
    ip_agreed_value: Optional[bool] = None
    has_other_policies: bool = False
    total_existing_policies: int = 0
    other_policy_details: Optional[str] = None
    previous_declination: bool = False

    # Section C: Occupation & Income
    occupation: str
    employer_name: str
    years_in_occupation: float
    annual_income: float
    has_hazardous_duties: bool = False
    hazardous_duties_description: Optional[str] = None

    # Section D: Health — General
    height_cm: float
    weight_kg: float
    smoker_status: SmokerStatus
    cigarettes_per_day: Optional[int] = None
    years_smoked: Optional[int] = None
    years_since_quit: Optional[float] = None
    taking_medications: bool = False
    medication_details: Optional[str] = None
    has_medical_conditions: bool = False
    medical_conditions: List[MedicalCondition] = Field(default_factory=list)
    consumes_alcohol: bool = False
    standard_drinks_per_week: Optional[int] = None

    # Section F: Family History
    has_family_history: bool = False
    family_history: List[FamilyHistoryCondition] = Field(default_factory=list)

    # Section G: Lifestyle
    has_hazardous_pursuits: bool = False
    hazardous_pursuits: List[HazardousPursuit] = Field(default_factory=list)
    recreational_drug_use: bool = False
    drug_use_details: Optional[str] = None
    alcohol_drug_treatment: bool = False
    has_high_risk_travel: bool = False
    high_risk_travel_details: Optional[str] = None

    # Section H: Financial
    total_net_worth: Optional[float] = None
    financial_obligations: Optional[str] = None
    obligation_end_dates: Optional[str] = None
    bankruptcy_status: str = "None"  # "None", "Current", "Recent", "Discharged"
    previous_bankruptcy: bool = False
    criminal_convictions: bool = False

    # Compliance
    duty_of_disclosure_acknowledged: bool = False
    mental_health_assessed: bool = False
    vulnerability_response_applied: bool = False

    @computed_field
    def age(self) -> int:
        """Calculate age from date of birth."""
        today = date.today()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )

    @computed_field
    def bmi(self) -> float:
        """Calculate BMI: weight (kg) / height (m)^2."""
        return round(self.weight_kg / ((self.height_cm / 100) ** 2), 1)

    @computed_field
    def occupation_class(self) -> OccupationClass:
        """Derive occupation class from duties and description.

        Simplified; in production this would use a classification table.
        """
        desc = (self.occupation or "").lower()
        if any(w in desc for w in ["miner", "offshore", "explosive", "underwater"]):
            return OccupationClass.HAZARDOUS
        if any(w in desc for w in ["construction", "rigger", "scaffolder", "roofer"]):
            return OccupationClass.HEAVY_MANUAL
        if any(w in desc for w in ["mechanic", "electrician", "plumber", "welder"]):
            return OccupationClass.BLUE_COLLAR
        if any(w in desc for w in ["warehouse", "driver", "machine operator", "cleaner"]):
            return OccupationClass.LIGHT_MANUAL
        if any(w in desc for w in ["manager", "administrator", "sales", "retail", "clerk"]):
            return OccupationClass.WHITE_COLLAR
        if any(
            w in desc
            for w in ["doctor", "lawyer", "accountant", "engineer",
                      "architect", "consultant", "analyst"]
        ):
            return OccupationClass.PROFESSIONAL
        return OccupationClass.WHITE_COLLAR  # default

    def has_condition(self, condition_category: str) -> bool:
        """Check if applicant has a condition in a given category."""
        categories = {
            "cardiovascular_disease": [
                "heart disease", "heart attack", "stroke",
                "hypertension", "high blood pressure",
                "angina", "arrhythmia", "cardiomyopathy",
            ],
            "diabetes_type_1": ["type 1 diabetes", "insulin-dependent diabetes"],
            "diabetes_type_2": ["type 2 diabetes", "non-insulin-dependent diabetes"],
            "mental_health": ["anxiety", "depression", "bipolar", "schizophrenia", "ptsd", "ocd"],
            "severe_mental_illness": ["schizophrenia", "bipolar disorder", "psychosis"],
        }
        keywords = categories.get(condition_category, [condition_category])
        return any(
            any(kw in (c.condition_name or "").lower() for kw in keywords)
            for c in self.medical_conditions
        )

    def check_family_history(self, condition_category: str) -> bool:
        """Check family history for a condition category."""
        keywords_map = {
            "cardiovascular_disease": ["heart disease", "heart attack", "stroke"],
            "cancer": ["cancer", "carcinoma", "melanoma", "leukemia", "lymphoma"],
        }
        keywords = keywords_map.get(condition_category, [condition_category])
        return any(
            any(kw in (fh.condition or "").lower() for kw in keywords)
            for fh in self.family_history
        )

    @property
    def family_age_at_diagnosis(self) -> Optional[int]:
        """Return the youngest age at diagnosis in family history."""
        if not self.family_history:
            return None
        return min(fh.age_at_diagnosis for fh in self.family_history)

    @property
    def has_multiple_policies(self) -> bool:
        return self.has_other_policies

    @property
    def total_sum_insured_across_all_policies(self) -> float:
        """Estimate total sum insured. Simplified for demo."""
        return sum(filter(None, [
            self.sum_insured_death or 0,
            self.sum_insured_tpd or 0,
            self.sum_insured_trauma or 0
        ]))

    @property
    def sum_insured(self) -> float:
        """Return the primary sum insured (death benefit).

        This property bridges the Application model with financial rules
        that reference ``applicant.sum_insured`` directly.
        """
        return self.sum_insured_death or 0.0

    def has_hazardous_pursuit(self, pursuit_name: str) -> bool:
        """Check if applicant participates in a specific hazardous pursuit."""
        pursuit_lower = pursuit_name.lower().strip()
        pursuit_words = pursuit_lower.split()
        for hp in self.hazardous_pursuits:
            activity_lower = hp.activity.lower().strip()
            # Direct substring check
            if pursuit_lower in activity_lower:
                return True
            # Activity in pursuit (reverse check)
            if activity_lower in pursuit_lower:
                return True
            # Word-level: any pursuit word in any activity word
            activity_words = activity_lower.split()
            for pw in pursuit_words:
                for aw in activity_words:
                    if pw in aw or aw in pw:
                        return True
                    # Stemming-like check: allow for a difference at the last
                    # character (e.g. "skydive" vs "skydiving")
                    min_len = min(len(pw), len(aw))
                    if min_len >= 3 and pw[:min_len - 1] == aw[:min_len - 1]:
                        return True
        return False
