"""Synthetic data generator for diverse underwriting applicant profiles.

Generates standardized, moderate-risk, high-risk, and edge-case applicant
profiles that exercise every branch of the underwriting rules engine.

Usage:
    python generate_synthetic.py --count 10
    python generate_synthetic.py --count 5 --output data/synthetic_applicants/
"""

import argparse
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path so src packages are importable.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.underwriting.application.schema import (
    Application,
    BenefitType,
    FamilyHistoryCondition,
    HazardousPursuit,
    MedicalCondition,
    OccupationClass,
    SmokerStatus,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FIRST_NAMES: List[str] = [
    "James", "Emma", "Li", "Maria", "Ahmed", "Sophie", "David",
    "Yuki", "Carlos", "Olga", "Raj", "Chloe", "Michael", "Aisha",
    "Thomas", "Priya", "Daniel", "Hannah", "Wei", "Fatima",
]
_LAST_NAMES: List[str] = [
    "Smith", "Chen", "Garcia", "Mueller", "Tanaka", "Patel", "Kim",
    "Johnson", "Williams", "Brown", "Jones", "Miller", "Wilson",
    "Moore", "Taylor", "Anderson", "Lee", "Wang", "Singh", "Martin",
]

_STREETS: List[str] = [
    "12 Main Street", "45 Oak Avenue", "78 River Road", "123 Park Lane",
    "56 Beach Boulevard", "90 Hillcrest Drive", "34 Forest Way",
    "67 Lakeview Terrace", "21 Church Road", "89 Garden Crescent",
]

_CITIES: List[str] = [
    "Sydney NSW 2000", "Melbourne VIC 3000", "Brisbane QLD 4000",
    "Perth WA 6000", "Adelaide SA 5000", "Canberra ACT 2600",
]

_PROFESSIONS: List[str] = [
    "Software Engineer", "Doctor", "Lawyer", "Accountant",
    "Architect", "Consultant", "Analyst", "Manager",
    "Administrator", "Sales Representative", "Retail Manager",
    "Clerk", "Mechanic", "Electrician", "Plumber",
    "Warehouse Operator", "Driver", "Construction Worker",
    "Rigger", "Roofer", "Miner", "Offshore Worker",
]

_EMPLOYERS: List[str] = [
    "TechCorp Pty Ltd", "HealthPlus Clinic", "Law Associates",
    "FinanceHub Pty Ltd", "BuildRight Construction", "LogiTrans Solutions",
    "RetailGroup Australia", "MediCare Hospital", "EngineeringWorks",
    "DataDrive Analytics",
]

_CONDITION_NAMES: List[str] = [
    "Hypertension", "Type 2 Diabetes", "Asthma", "Back Pain",
    "Anxiety Disorder", "Depression", "High Cholesterol",
    "GERD (Gastric Reflux)", "Migraine", "Knee Osteoarthritis",
    "Heart Disease", "Type 1 Diabetes", "Bipolar Disorder",
    "Schizophrenia", "COPD", "Kidney Disease",
]

_FAMILY_RELATIONSHIPS: List[str] = ["father", "mother", "brother", "sister"]

_FAMILY_CONDITIONS: List[str] = [
    "Heart Disease", "Stroke", "Heart Attack",
    "Cancer", "Breast Cancer", "Colorectal Cancer",
    "Diabetes", "Alzheimer's Disease",
]

_HAZARDOUS_ACTIVITIES: List[str] = [
    "Scuba Diving", "Rock Climbing", "Skydiving",
    "Motorcycle Racing", "Wingsuit Flying", "White Water Rafting",
]

_TRAVEL_COUNTRIES: List[str] = [
    "Somalia", "Yemen", "Afghanistan", "Syria", "Iraq",
    "Venezuela", "Haiti", "Myanmar",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_dob(age_at: int) -> date:
    """Return a random date-of-birth that makes the person approximately *age_at*."""
    base = date(2026, 1, 1) - timedelta(days=int(365.25 * age_at))
    offset = random.randint(-180, 180)
    return base + timedelta(days=offset)


def _random_medical(
    condition_name: str,
    years_before: float = 3.0,
) -> MedicalCondition:
    """Build a single MedicalCondition with realistic defaults."""
    diag_years_ago = random.uniform(0.5, years_before)
    diagnosis_date = date(2026, 1, 1) - timedelta(days=int(365.25 * diag_years_ago))
    treatment_start = diagnosis_date + timedelta(days=random.randint(0, 30))
    last_symptom = date(2026, 1, 1) - timedelta(days=random.randint(1, 30))
    return MedicalCondition(
        condition_name=condition_name,
        diagnosis_date=diagnosis_date,
        treating_doctor_name=f"Dr {''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=1))}. {''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))}",
        treating_doctor_contact=f"+61 4{random.randint(10000000, 99999999)}",
        diagnostic_tests=random.choice(["Blood panel", "ECG", "X-ray", "MRI", "CT scan", "Blood panel and ECG"]),
        treatment_start_date=treatment_start,
        treatment_description=f"Ongoing management of {condition_name.lower()}",
        symptoms=f"Mild to moderate symptoms of {condition_name.lower()}",
        symptom_frequency=random.choice(["Daily", "Weekly", "Monthly"]),
        last_symptom_date=last_symptom,
        lifestyle_affected=True,
    )


def _random_family_history(
    min_age: int = 40,
    max_age: int = 70,
) -> FamilyHistoryCondition:
    """Build a single FamilyHistoryCondition entry."""
    return FamilyHistoryCondition(
        relationship=random.choice(_FAMILY_RELATIONSHIPS),
        condition=random.choice(_FAMILY_CONDITIONS),
        age_at_diagnosis=random.randint(min_age, max_age),
    )


def _random_hazardous_pursuit() -> HazardousPursuit:
    """Build a single HazardousPursuit entry."""
    return HazardousPursuit(
        activity=random.choice(_HAZARDOUS_ACTIVITIES),
        frequency=random.choice(["Monthly", "Quarterly", "Bi-annually", "Annually"]),
        level=random.choice(["amateur", "professional"]),
    )


def _make_base_applicant(name_prefix: str = "") -> Dict[str, Any]:
    """Return a minimal dict of shared fields for any applicant."""
    first = random.choice(_FIRST_NAMES)
    last = random.choice(_LAST_NAMES)
    if name_prefix:
        first = name_prefix + first
    return {
        "full_name": f"{first} {last}",
        "gender": random.choice(["Male", "Female", "Non-binary"]),
        "residency_status": random.choice([
            "Australian Citizen", "Permanent Resident", "Temporary Visa",
        ]),
        "contact_address": f"{random.choice(_STREETS)}, {random.choice(_CITIES)}",
        "employer_name": random.choice(_EMPLOYERS),
        "benefit_types": random.sample(list(BenefitType), k=random.randint(2, 4)),
        "has_other_policies": random.choice([True, False]),
        "total_existing_policies": random.randint(0, 5),
        "taking_medications": random.choice([True, False]),
        "consumes_alcohol": random.choice([True, False]),
        "standard_drinks_per_week": random.randint(0, 20) if random.choice([True, False]) else 0,
        "has_hazardous_pursuits": False,
        "hazardous_pursuits": [],
        "recreational_drug_use": False,
        "alcohol_drug_treatment": False,
        "has_high_risk_travel": False,
        "duty_of_disclosure_acknowledged": True,
    }


def _build_application(data: Dict[str, Any]) -> Application:
    """Convert a dict into a validated Application model instance."""
    return Application(**data)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Profile generators
# ---------------------------------------------------------------------------

def generate_standard_profile(seed: Optional[int] = None) -> Application:
    """Generate a **standard risk** applicant profile.

    Health: good, no conditions.
    Income: moderate-to-high.
    Occupation: professional (white-collar).
    """
    if seed is not None:
        random.seed(seed)

    age = random.randint(28, 45)
    dob = _random_dob(age)
    bmi_target = random.uniform(21.0, 24.5)
    height = random.uniform(165, 185)
    weight = round(bmi_target * (height / 100) ** 2, 1)

    data: Dict[str, Any] = _make_base_applicant()
    data.update({
        "date_of_birth": dob,
        "occupation": random.choice([
            "Software Engineer", "Doctor", "Lawyer", "Accountant",
            "Architect", "Consultant", "Analyst",
        ]),
        "years_in_occupation": random.uniform(3, 20),
        "annual_income": random.uniform(80_000, 180_000),
        "height_cm": height,
        "weight_kg": weight,
        "smoker_status": SmokerStatus.NEVER,
        "medication_details": None,
        "has_medical_conditions": False,
        "medical_conditions": [],
        "has_family_history": False,
        "family_history": [],
        "sum_insured_death": random.uniform(500_000, 1_500_000),
        "sum_insured_tpd": random.uniform(300_000, 800_000),
        "sum_insured_trauma": random.uniform(100_000, 300_000),
        "ip_monthly_benefit": random.choice([None, random.uniform(5_000, 10_000)]),
        "ip_benefit_period": random.choice([None, "To age 65", "2 years"]),
        "ip_agreed_value": random.choice([True, False, None]),
        "other_policy_details": None,
        "previous_declination": False,
        "has_hazardous_duties": False,
        "hazardous_duties_description": None,
        "total_net_worth": random.uniform(200_000, 1_000_000),
        "financial_obligations": random.choice([
            "Mortgage", "Car loan", "Student debt", "None",
        ]),
        "obligation_end_dates": random.choice([
            "2040", "2035", "2030", "None",
        ]),
        "bankruptcy_status": "None",
        "previous_bankruptcy": False,
        "criminal_convictions": False,
        "high_risk_travel_details": None,
        "drug_use_details": None,
    })
    return _build_application(data)


def generate_moderate_profile(seed: Optional[int] = None) -> Application:
    """Generate a **moderate risk** applicant profile.

    Health: one or two manageable conditions.
    Income: mid-range.
    Occupation: white-collar or light manual.
    """
    if seed is not None:
        random.seed(seed)

    age = random.randint(35, 55)
    dob = _random_dob(age)
    bmi_target = random.uniform(25.0, 29.9)
    height = random.uniform(160, 185)
    weight = round(bmi_target * (height / 100) ** 2, 1)

    conditions_count = random.randint(1, 2)
    conditions = [
        _random_medical(random.choice([
            "Hypertension", "High Cholesterol", "GERD (Gastric Reflux)",
            "Migraine", "Knee Osteoarthritis", "Asthma",
        ]), years_before=5.0)
        for _ in range(conditions_count)
    ]

    data: Dict[str, Any] = _make_base_applicant()
    data.update({
        "date_of_birth": dob,
        "occupation": random.choice([
            "Manager", "Administrator", "Sales Representative",
            "Retail Manager", "Clerk", "Warehouse Operator",
        ]),
        "years_in_occupation": random.uniform(2, 15),
        "annual_income": random.uniform(50_000, 100_000),
        "height_cm": height,
        "weight_kg": weight,
        "smoker_status": random.choice([SmokerStatus.NEVER, SmokerStatus.FORMER]),
        "cigarettes_per_day": None,
        "years_smoked": None,
        "years_since_quit": random.uniform(5, 20) if random.choice([True, False]) else None,
        "medication_details": "Metformin 500mg",
        "has_medical_conditions": True,
        "medical_conditions": conditions,
        "has_family_history": random.choice([True, False]),
        "family_history": [
            _random_family_history(min_age=50, max_age=70)
            for _ in range(random.randint(1, 2))
        ] if random.choice([True, False]) else [],
        "sum_insured_death": random.uniform(300_000, 800_000),
        "sum_insured_tpd": random.uniform(200_000, 500_000),
        "sum_insured_trauma": random.uniform(50_000, 200_000),
        "ip_monthly_benefit": random.choice([None, random.uniform(4_000, 8_000)]),
        "ip_benefit_period": random.choice([None, "To age 65", "2 years"]),
        "ip_agreed_value": random.choice([True, False, None]),
        "other_policy_details": None,
        "previous_declination": False,
        "has_hazardous_duties": random.choice([True, False]),
        "hazardous_duties_description": "Occasional lifting of heavy equipment" if random.choice([True, False]) else None,
        "total_net_worth": random.uniform(100_000, 500_000),
        "financial_obligations": random.choice(["Mortgage", "Car loan", "Student debt"]),
        "obligation_end_dates": random.choice(["2040", "2035", "2030"]),
        "bankruptcy_status": "None",
        "previous_bankruptcy": False,
        "criminal_convictions": False,
        "high_risk_travel_details": None,
        "drug_use_details": None,
    })
    return _build_application(data)


def generate_high_risk_profile(seed: Optional[int] = None) -> Application:
    """Generate a **high risk** applicant profile.

    Health: multiple conditions, former/current smoker, hazardous occupation.
    Income: low-to-mid.
    """
    if seed is not None:
        random.seed(seed)

    age = random.randint(40, 62)
    dob = _random_dob(age)
    bmi_target = random.uniform(30.0, 38.0)
    height = random.uniform(160, 180)
    weight = round(bmi_target * (height / 100) ** 2, 1)

    conditions_count = random.randint(2, 4)
    conditions = [
        _random_medical(random.choice([
            "Heart Disease", "Type 2 Diabetes", "Type 1 Diabetes",
            "COPD", "Kidney Disease", "Anxiety Disorder", "Depression",
            "Hypertension", "High Cholesterol",
        ]), years_before=8.0)
        for _ in range(conditions_count)
    ]

    smoker_status = random.choice([SmokerStatus.FORMER, SmokerStatus.CURRENT])
    years_smoked = random.randint(10, 35)

    data: Dict[str, Any] = _make_base_applicant()
    data.update({
        "date_of_birth": dob,
        "occupation": random.choice([
            "Miner", "Offshore Worker", "Construction Worker",
            "Rigger", "Roofer", "Mechanic", "Electrician", "Plumber",
        ]),
        "years_in_occupation": random.uniform(5, 30),
        "annual_income": random.uniform(35_000, 70_000),
        "height_cm": height,
        "weight_kg": weight,
        "smoker_status": smoker_status,
        "cigarettes_per_day": random.randint(5, 20) if smoker_status == SmokerStatus.CURRENT else None,
        "years_smoked": years_smoked,
        "years_since_quit": random.uniform(1, 5) if smoker_status == SmokerStatus.FORMER else None,
        "medication_details": "Multiple — see medical conditions",
        "has_medical_conditions": True,
        "medical_conditions": conditions,
        "has_family_history": True,
        "family_history": [
            _random_family_history(min_age=40, max_age=60)
            for _ in range(random.randint(1, 3))
        ],
        "sum_insured_death": random.uniform(200_000, 500_000),
        "sum_insured_tpd": random.uniform(100_000, 300_000),
        "sum_insured_trauma": random.uniform(50_000, 150_000),
        "ip_monthly_benefit": random.choice([None, random.uniform(3_000, 6_000)]),
        "ip_benefit_period": random.choice([None, "To age 65", "2 years"]),
        "ip_agreed_value": random.choice([True, False, None]),
        "other_policy_details": None,
        "previous_declination": random.choice([True, False]),
        "has_hazardous_duties": True,
        "hazardous_duties_description": "Working at heights / heavy machinery / explosives",
        "total_net_worth": random.uniform(50_000, 200_000),
        "financial_obligations": "Mortgage",
        "obligation_end_dates": "2040",
        "bankruptcy_status": random.choice(["None", "Discharged"]),
        "previous_bankruptcy": random.choice([True, False]),
        "criminal_convictions": False,
        "high_risk_travel_details": None,
        "drug_use_details": None,
    })
    return _build_application(data)


def generate_edge_case_profiles() -> List[Application]:
    """Generate **edge-case** applicant profiles that hit exact boundaries.

    Covers:
    - BMI exactly 30.0, 35.0, 40.0
    - Age boundaries: 30, 40, 50, 60, 65, 70
    - Former smoker with varying years_since_quit (1, 2, 5, 10, 15, 20)
    - Multiple pre-existing conditions (1, 2, 3+)
    - Family history with varying ages at diagnosis
    """
    profiles: List[Application] = []

    # --- BMI edge cases ---
    for target_bmi in [30.0, 35.0, 40.0]:
        height = 175.0
        weight = round(target_bmi * (height / 100) ** 2, 1)
        data: Dict[str, Any] = _make_base_applicant(name_prefix=f"BMI-{int(target_bmi)}-")
        data.update({
            "date_of_birth": _random_dob(35),
            "occupation": "Software Engineer",
            "years_in_occupation": 5.0,
            "annual_income": 90_000.0,
            "height_cm": height,
            "weight_kg": weight,
            "smoker_status": SmokerStatus.NEVER,
            "medication_details": None,
            "has_medical_conditions": False,
            "medical_conditions": [],
            "has_family_history": False,
            "family_history": [],
            "sum_insured_death": 500_000.0,
            "sum_insured_tpd": 300_000.0,
            "sum_insured_trauma": 100_000.0,
            "ip_monthly_benefit": None,
            "ip_benefit_period": None,
            "ip_agreed_value": None,
            "other_policy_details": None,
            "previous_declination": False,
            "has_hazardous_duties": False,
            "hazardous_duties_description": None,
            "total_net_worth": 200_000.0,
            "financial_obligations": "Mortgage",
            "obligation_end_dates": "2040",
            "bankruptcy_status": "None",
            "previous_bankruptcy": False,
            "criminal_convictions": False,
            "high_risk_travel_details": None,
            "drug_use_details": None,
        })
        profiles.append(_build_application(data))

    # --- Age boundary cases ---
    for target_age in [30, 40, 50, 60, 65, 70]:
        dob = _random_dob(target_age)
        data = _make_base_applicant(name_prefix=f"Age-{target_age}-")
        data.update({
            "date_of_birth": dob,
            "occupation": "Manager",
            "years_in_occupation": float(target_age - 25),
            "annual_income": 75_000.0 + (target_age * 500),
            "height_cm": 170.0,
            "weight_kg": 75.0,
            "smoker_status": SmokerStatus.NEVER,
            "medication_details": None,
            "has_medical_conditions": False,
            "medical_conditions": [],
            "has_family_history": False,
            "family_history": [],
            "sum_insured_death": 500_000.0,
            "sum_insured_tpd": 300_000.0,
            "sum_insured_trauma": 100_000.0,
            "ip_monthly_benefit": None,
            "ip_benefit_period": None,
            "ip_agreed_value": None,
            "other_policy_details": None,
            "previous_declination": False,
            "has_hazardous_duties": False,
            "hazardous_duties_description": None,
            "total_net_worth": 300_000.0,
            "financial_obligations": "Mortgage",
            "obligation_end_dates": "2045",
            "bankruptcy_status": "None",
            "previous_bankruptcy": False,
            "criminal_convictions": False,
            "high_risk_travel_details": None,
            "drug_use_details": None,
        })
        profiles.append(_build_application(data))

    # --- Former smoker with varying years_since_quit ---
    for years_quit in [1, 2, 5, 10, 15, 20]:
        data = _make_base_applicant(name_prefix=f"Quit-{years_quit}y-")
        data.update({
            "date_of_birth": _random_dob(45),
            "occupation": "Administrator",
            "years_in_occupation": 10.0,
            "annual_income": 65_000.0,
            "height_cm": 168.0,
            "weight_kg": 72.0,
            "smoker_status": SmokerStatus.FORMER,
            "cigarettes_per_day": None,
            "years_smoked": random.randint(10, 25),
            "years_since_quit": float(years_quit),
            "medication_details": None,
            "has_medical_conditions": False,
            "medical_conditions": [],
            "has_family_history": False,
            "family_history": [],
            "sum_insured_death": 400_000.0,
            "sum_insured_tpd": 250_000.0,
            "sum_insured_trauma": 80_000.0,
            "ip_monthly_benefit": None,
            "ip_benefit_period": None,
            "ip_agreed_value": None,
            "other_policy_details": None,
            "previous_declination": False,
            "has_hazardous_duties": False,
            "hazardous_duties_description": None,
            "total_net_worth": 150_000.0,
            "financial_obligations": "Car loan",
            "obligation_end_dates": "2030",
            "bankruptcy_status": "None",
            "previous_bankruptcy": False,
            "criminal_convictions": False,
            "high_risk_travel_details": None,
            "drug_use_details": None,
        })
        profiles.append(_build_application(data))

    # --- Multiple pre-existing conditions (1, 2, 3+) ---
    for count in [1, 2, 3, 4, 5]:
        conditions = [
            _random_medical(random.choice(_CONDITION_NAMES), years_before=5.0)
            for _ in range(count)
        ]
        data = _make_base_applicant(name_prefix=f"Conditions-{count}-")
        data.update({
            "date_of_birth": _random_dob(42),
            "occupation": "Clerk",
            "years_in_occupation": 8.0,
            "annual_income": 55_000.0,
            "height_cm": 172.0,
            "weight_kg": 78.0,
            "smoker_status": SmokerStatus.NEVER,
            "medication_details": "See conditions",
            "has_medical_conditions": True,
            "medical_conditions": conditions,
            "has_family_history": False,
            "family_history": [],
            "sum_insured_death": 300_000.0,
            "sum_insured_tpd": 200_000.0,
            "sum_insured_trauma": 60_000.0,
            "ip_monthly_benefit": None,
            "ip_benefit_period": None,
            "ip_agreed_value": None,
            "other_policy_details": None,
            "previous_declination": False,
            "has_hazardous_duties": False,
            "hazardous_duties_description": None,
            "total_net_worth": 100_000.0,
            "financial_obligations": "None",
            "obligation_end_dates": None,
            "bankruptcy_status": "None",
            "previous_bankruptcy": False,
            "criminal_convictions": False,
            "high_risk_travel_details": None,
            "drug_use_details": None,
        })
        profiles.append(_build_application(data))

    # --- Family history with varying ages at diagnosis ---
    for age_diag in [40, 50, 60, 70, 80]:
        fh = FamilyHistoryCondition(
            relationship=random.choice(_FAMILY_RELATIONSHIPS),
            condition=random.choice(_FAMILY_CONDITIONS),
            age_at_diagnosis=age_diag,
        )
        data = _make_base_applicant(name_prefix=f"FamHx-{age_diag}-")
        data.update({
            "date_of_birth": _random_dob(35),
            "occupation": "Teacher",
            "years_in_occupation": 10.0,
            "annual_income": 70_000.0,
            "height_cm": 165.0,
            "weight_kg": 65.0,
            "smoker_status": SmokerStatus.NEVER,
            "medication_details": None,
            "has_medical_conditions": False,
            "medical_conditions": [],
            "has_family_history": True,
            "family_history": [fh],
            "sum_insured_death": 450_000.0,
            "sum_insured_tpd": 280_000.0,
            "sum_insured_trauma": 90_000.0,
            "ip_monthly_benefit": None,
            "ip_benefit_period": None,
            "ip_agreed_value": None,
            "other_policy_details": None,
            "previous_declination": False,
            "has_hazardous_duties": False,
            "hazardous_duties_description": None,
            "total_net_worth": 250_000.0,
            "financial_obligations": "Mortgage",
            "obligation_end_dates": "2042",
            "bankruptcy_status": "None",
            "previous_bankruptcy": False,
            "criminal_convictions": False,
            "high_risk_travel_details": None,
            "drug_use_details": None,
        })
        profiles.append(_build_application(data))

    return profiles


# ---------------------------------------------------------------------------
# Profile pool & selection
# ---------------------------------------------------------------------------

PROFILE_GENERATORS: List[tuple[str, callable]] = [
    ("standard", generate_standard_profile),
    ("moderate", generate_moderate_profile),
    ("high_risk", generate_high_risk_profile),
]

# Edge-case profiles are always included regardless of --count.
EDGE_CASES = generate_edge_case_profiles()


def _generate_profiles(count: int) -> List[Application]:
    """Generate *count* profiles, mixing risk categories evenly."""
    profiles: List[Application] = []

    if count <= 0:
        return profiles

    # Distribute count across categories.
    per_category = max(1, count // len(PROFILE_GENERATORS))
    remainder = count - per_category * len(PROFILE_GENERATORS)

    for idx, (name, generator) in enumerate(PROFILE_GENERATORS):
        n = per_category + (1 if idx < remainder else 0)
        for i in range(n):
            profiles.append(generator(seed=i))

    # Always append edge-case profiles.
    profiles.extend(EDGE_CASES)

    return profiles


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _application_to_dict(app: Application) -> Dict[str, Any]:
    """Serialise an Application to a plain dict (JSON-serialisable)."""
    result: Dict[str, Any] = {}
    for field_name in app.model_fields:
        value = getattr(app, field_name)
        if isinstance(value, date):
            value = value.isoformat()
        elif isinstance(value, (SmokerStatus, OccupationClass, BenefitType)):
            value = value.value
        elif isinstance(value, (MedicalCondition, FamilyHistoryCondition, HazardousPursuit)):
            value = value.model_dump()
        elif isinstance(value, list):
            value = [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in value
            ]
        result[field_name] = value
    return result


def save_profiles(
    profiles: List[Application],
    output_dir: Path,
) -> List[Path]:
    """Save profiles as individual JSON files into *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for idx, app in enumerate(profiles):
        file_path = output_dir / f"applicant_{idx + 1:04d}.json"
        data = _application_to_dict(app)
        file_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        saved.append(file_path)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic underwriting applicant profiles.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of random profiles to generate (default: 10). Edge-case profiles are always included.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic_applicants/",
        help="Output directory for JSON files (default: data/synthetic_applicants/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point."""
    args = parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)

    output_dir = Path(args.output)
    profiles = _generate_profiles(args.count)
    saved = save_profiles(profiles, output_dir)

    print(f"Generated {len(profiles)} applicant profiles ({len(saved)} files).")
    print(f"Output directory: {output_dir.resolve()}")

    # Print summary statistics.
    risk_counts: Dict[str, int] = {}
    for app in profiles:
        age = app.age
        bmi = app.bmi
        smoker = app.smoker_status.value
        occ_class = app.occupation_class.value
        conditions = len(app.medical_conditions)

        # Classify risk tier.
        if bmi >= 35 or conditions >= 3 or app.smoker_status == SmokerStatus.CURRENT or app.has_medical_conditions:
            tier = "high"
        elif conditions >= 1 or app.has_family_history or app.smoker_status == SmokerStatus.FORMER:
            tier = "moderate"
        else:
            tier = "standard"

        risk_counts[tier] = risk_counts.get(tier, 0) + 1

    print(f"Risk distribution: {risk_counts}")


if __name__ == "__main__":
    main()
