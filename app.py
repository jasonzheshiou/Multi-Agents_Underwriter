"""Streamlit application for the Multi-Agents Underwriting Rules Engine.

Provides a web-based questionnaire for life insurance applications,
results display, and debate log viewer.
"""

import os
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

import streamlit as st
import yaml
from pydantic import ValidationError

# Ensure the src directory is on the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

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
from underwriting.llm.llm_client import LLMClient

# ---------------------------------------------------------------------------
# Agent styles for chat UI
# ---------------------------------------------------------------------------
AGENT_STYLES = {
    "Medical Agent": {"color": "#2E86AB", "emoji": "\U0001F7BA", "initials": "MA"},
    "Financial Agent": {"color": "#A23B72", "emoji": "\U0001F4B0", "initials": "FA"},
    "Compliance Agent": {"color": "#F18F01", "emoji": "\u2696\uFE0F", "initials": "CA"},
    "user": {"color": "#2B2D42", "emoji": "\U0001F464", "initials": "ME"},
    "system": {"color": "#6C757D", "emoji": "\U0001F916", "initials": "SYS"},
}

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Underwriting Rules Engine",
    layout="wide",
    page_icon="\U0001F4C1",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config_yaml(path: str = "config.yaml") -> dict:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Dictionary with configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_benefit_options() -> list:
    """Return available benefit type options for multi-select.

    Returns:
        List of BenefitType enum values.
    """
    return [
        BenefitType.DEATH,
        BenefitType.TPD,
        BenefitType.TRAUMA,
        BenefitType.IP,
    ]


def _build_application_from_data(data: dict) -> Application:
    """Construct an Application model from a data dictionary.

    Used by the "View in Chat" path to recreate an Application
    from the questionnaire editable data.

    Args:
        data: Dictionary with questionnaire field values.

    Returns:
        A validated Application instance.
    """
    def safe_str(val, default=""):
        return val if val is not None else default

    def safe_int(val, default=0):
        return int(val) if val is not None else default

    def safe_float(val, default=0.0):
        return float(val) if val is not None else default

    def safe_bool(val, default=False):
        return bool(val) if val is not None else default

    def safe_index(options, val, default):
        v = val if val is not None else default
        try:
            return options.index(v)
        except ValueError:
            return 0

    smoker_map = {"Never": SmokerStatus.NEVER, "Former": SmokerStatus.FORMER, "Current": SmokerStatus.CURRENT}
    smoker = smoker_map.get(safe_str(data.get("smoker_status"), "Never"), SmokerStatus.NEVER)

    benefit_map = {
        "Death": BenefitType.DEATH,
        "TPD": BenefitType.TPD,
        "Trauma/CI": BenefitType.TRAUMA,
        "Income Protection": BenefitType.IP,
    }
    benefit_types = [benefit_map[b] for b in data.get("benefit_types", ["Death"]) if b in benefit_map]

    return Application(
        full_name=safe_str(data.get("full_name")),
        date_of_birth=date.fromisoformat(safe_str(data.get("date_of_birth"), "1990-01-01")),
        gender=safe_str(data.get("gender")),
        residency_status=safe_str(data.get("residency_status")),
        contact_address=safe_str(data.get("contact_address")),
        benefit_types=benefit_types,
        sum_insured_death=safe_int(data.get("sum_insured_death"), 500000),
        sum_insured_tpd=safe_int(data.get("sum_insured_tpd"), 500000),
        sum_insured_trauma=safe_int(data.get("sum_insured_trauma"), 0),
        occupation=safe_str(data.get("occupation")),
        employer_name=safe_str(data.get("employer_name")),
        years_in_occupation=safe_float(data.get("years_in_occupation"), 5.0),
        annual_income=safe_float(data.get("annual_income"), 100000.0),
        height_cm=safe_float(data.get("height_cm"), 170.0),
        weight_kg=safe_float(data.get("weight_kg"), 70.0),
        smoker_status=smoker,
        taking_medications=safe_bool(data.get("taking_medications")),
        consumes_alcohol=safe_bool(data.get("consumes_alcohol")),
    )


def build_application_from_form() -> Application:
    """Construct an Application model from the Streamlit form state.

    Reads all form inputs from ``st.session_state`` and returns a validated
    :class:`Application` Pydantic model.

    Returns:
        A validated Application instance.

    Raises:
        ValidationError: If any field fails Pydantic validation.
    """
    benefit_types = st.session_state.get("_benefit_types", [])
    has_other = st.session_state.get("_has_other_policies", False)
    previous_decl = st.session_state.get("_previous_declination", False)
    smoker = st.session_state.get("_smoker_status", "Never")
    takes_meds = st.session_state.get("_taking_medications", False)
    has_conditions = st.session_state.get("_has_medical_conditions", False)
    consumes = st.session_state.get("_consumes_alcohol", False)
    has_family = st.session_state.get("_has_family_history", False)
    has_hazardous = st.session_state.get("_has_hazardous_pursuits", False)
    drug_use = st.session_state.get("_recreational_drug_use", False)
    drug_treatment = st.session_state.get("_alcohol_drug_treatment", False)
    has_travel = st.session_state.get("_has_high_risk_travel", False)
    has_bankruptcy = st.session_state.get("_has_bankruptcy", False)
    has_convictions = st.session_state.get("_has_convictions", False)
    acknowledged = st.session_state.get("_acknowledged", False)

    # Build medical conditions list
    medical_conditions: list = []
    if has_conditions:
        num_conditions = st.session_state.get("_num_conditions", 0)
        for i in range(num_conditions):
            prefix = f"_cond_{i}_"
            diag_date_raw = st.session_state.get(f"{prefix}diag_date", "")
            treat_start_raw = st.session_state.get(f"{prefix}treat_start", "")
            last_symptom_raw = st.session_state.get(f"{prefix}last_symptom", "")

            try:
                diag_date = date.fromisoformat(diag_date_raw) if diag_date_raw else None
            except ValueError:
                diag_date = None
            try:
                treat_start = date.fromisoformat(treat_start_raw) if treat_start_raw else None
            except ValueError:
                treat_start = None
            try:
                last_symptom = date.fromisoformat(last_symptom_raw) if last_symptom_raw else None
            except ValueError:
                last_symptom = None

            medical_conditions.append(
                MedicalCondition(
                    condition_name=st.session_state.get(f"{prefix}name", ""),
                    diagnosis_date=diag_date,
                    treating_doctor_name=st.session_state.get(f"{prefix}doctor_name", ""),
                    treating_doctor_contact=st.session_state.get(f"{prefix}doctor_contact", ""),
                    diagnostic_tests=st.session_state.get(f"{prefix}tests", None),
                    treatment_start_date=treat_start,
                    treatment_description=st.session_state.get(f"{prefix}treatment", None),
                    symptoms=st.session_state.get(f"{prefix}symptoms", None),
                    symptom_frequency=st.session_state.get(f"{prefix}symptom_freq", None),
                    last_symptom_date=last_symptom,
                    hospitalisations=st.session_state.get(f"{prefix}hospital", None),
                    time_off_work=st.session_state.get(f"{prefix}time_off", None),
                    lifestyle_affected=st.session_state.get(f"{prefix}lifestyle_affected", None),
                )
            )

    # Build family history list
    family_history: list = []
    if has_family:
        num_family = st.session_state.get("_num_family", 0)
        for i in range(num_family):
            prefix = f"_fam_{i}_"
            family_history.append(
                FamilyHistoryCondition(
                    relationship=st.session_state.get(f"{prefix}relationship", ""),
                    condition=st.session_state.get(f"{prefix}condition", ""),
                    age_at_diagnosis=int(st.session_state.get(f"{prefix}age", 0)),
                )
            )

    # Build hazardous pursuits list
    hazardous_pursuits: list = []
    if has_hazardous:
        num_pursuits = st.session_state.get("_num_pursuits", 0)
        for i in range(num_pursuits):
            prefix = f"_pursuit_{i}_"
            hazardous_pursuits.append(
                HazardousPursuit(
                    activity=st.session_state.get(f"{prefix}activity", ""),
                    frequency=st.session_state.get(f"{prefix}frequency", ""),
                    level=st.session_state.get(f"{prefix}level", "amateur"),
                )
            )

    # Smoker-derived fields
    cigarettes_per_day = None
    years_smoked = None
    years_since_quit = None
    if smoker in ("Current", "Former"):
        cigarettes_per_day = st.session_state.get("_cigarettes_per_day", None)
        years_smoked = st.session_state.get("_years_smoked", None)
    if smoker == "Former":
        years_since_quit = st.session_state.get("_years_since_quit", None)

    return Application(
        # Section A
        full_name=st.session_state.get("_full_name", ""),
        date_of_birth=st.session_state.get("_date_of_birth", date.today()),
        gender=st.session_state.get("_gender", "Male"),
        residency_status=st.session_state.get("_residency_status", "Australian Citizen"),
        contact_address=st.session_state.get("_contact_address", ""),
        # Section B
        benefit_types=benefit_types,
        sum_insured_death=st.session_state.get("_sum_insured_death", None),
        sum_insured_tpd=st.session_state.get("_sum_insured_tpd", None),
        sum_insured_trauma=st.session_state.get("_sum_insured_trauma", None),
        ip_monthly_benefit=st.session_state.get("_ip_monthly_benefit", None),
        ip_benefit_period=st.session_state.get("_ip_benefit_period", None),
        ip_agreed_value=st.session_state.get("_ip_agreed_value", None),
        has_other_policies=has_other,
        total_existing_policies=int(st.session_state.get("_total_existing_policies", 0)),
        other_policy_details=st.session_state.get("_other_policy_details", None) if has_other else None,
        previous_declination=previous_decl,
        # Section C
        occupation=st.session_state.get("_occupation", ""),
        employer_name=st.session_state.get("_employer_name", ""),
        years_in_occupation=float(st.session_state.get("_years_in_occupation", 0)),
        annual_income=float(st.session_state.get("_annual_income", 0)),
        has_hazardous_duties=st.session_state.get("_has_hazardous_duties", False),
        hazardous_duties_description=st.session_state.get("_hazardous_duties_desc", None)
        if st.session_state.get("_has_hazardous_duties", False)
        else None,
        # Section D
        height_cm=float(st.session_state.get("_height_cm", 0)),
        weight_kg=float(st.session_state.get("_weight_kg", 0)),
        smoker_status=SmokerStatus(smoker),
        cigarettes_per_day=cigarettes_per_day,
        years_smoked=years_smoked,
        years_since_quit=years_since_quit,
        taking_medications=takes_meds,
        medication_details=st.session_state.get("_medication_details", None) if takes_meds else None,
        has_medical_conditions=has_conditions,
        medical_conditions=medical_conditions,
        consumes_alcohol=consumes,
        standard_drinks_per_week=int(st.session_state.get("_standard_drinks_per_week", 0))
        if consumes
        else None,
        # Section F
        has_family_history=has_family,
        family_history=family_history,
        # Section G
        has_hazardous_pursuits=has_hazardous,
        hazardous_pursuits=hazardous_pursuits,
        recreational_drug_use=drug_use,
        drug_use_details=st.session_state.get("_drug_use_details", None) if drug_use else None,
        alcohol_drug_treatment=drug_treatment,
        has_high_risk_travel=has_travel,
        high_risk_travel_details=st.session_state.get("_high_risk_travel_details", None)
        if has_travel
        else None,
        # Section H
        total_net_worth=float(st.session_state.get("_total_net_worth", 0))
        if st.session_state.get("_total_net_worth", 0)
        else None,
        financial_obligations=st.session_state.get("_financial_obligations", None),
        obligation_end_dates=st.session_state.get("_obligation_end_dates", None),
        bankruptcy_status="Current"
        if has_bankruptcy and st.session_state.get("_bankruptcy_status", "None") == "Current"
        else "Recent"
        if has_bankruptcy and st.session_state.get("_bankruptcy_status", "None") == "Recent"
        else "Discharged"
        if has_bankruptcy and st.session_state.get("_bankruptcy_status", "None") == "Discharged"
        else "None",
        previous_bankruptcy=has_bankruptcy,
        criminal_convictions=has_convictions,
        # Compliance
        duty_of_disclosure_acknowledged=acknowledged,
    )


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialise all Streamlit session state keys used by the application."""
    defaults = {
        "_page": "Questionnaire",
        "_submitted": False,
        "_application": None,
        "_results": None,
        "_errors": None,
        # Section A
        "_full_name": "",
        "_date_of_birth": date.today(),
        "_gender": "Male",
        "_residency_status": "Australian Citizen",
        "_contact_address": "",
        # Section B
        "_benefit_types": [],
        "_sum_insured_death": None,
        "_sum_insured_tpd": None,
        "_sum_insured_trauma": None,
        "_ip_monthly_benefit": None,
        "_ip_benefit_period": "2yr",
        "_ip_agreed_value": True,
        "_has_other_policies": False,
        "_total_existing_policies": 0,
        "_other_policy_details": "",
        "_previous_declination": False,
        # Section C
        "_occupation": "",
        "_employer_name": "",
        "_years_in_occupation": 0,
        "_annual_income": 0,
        "_has_hazardous_duties": False,
        "_hazardous_duties_desc": "",
        # Section D
        "_height_cm": 0,
        "_weight_kg": 0,
        "_smoker_status": "Never",
        "_cigarettes_per_day": 10,
        "_years_smoked": 0,
        "_years_since_quit": 0,
        "_taking_medications": False,
        "_medication_details": "",
        "_has_medical_conditions": False,
        "_num_conditions": 0,
        "_consumes_alcohol": False,
        "_standard_drinks_per_week": 0,
        # Section F
        "_has_family_history": False,
        "_num_family": 0,
        # Section G
        "_has_hazardous_pursuits": False,
        "_num_pursuits": 0,
        "_recreational_drug_use": False,
        "_drug_use_details": "",
        "_alcohol_drug_treatment": False,
        "_has_high_risk_travel": False,
        "_high_risk_travel_details": "",
        # Section H
        "_total_net_worth": 0,
        "_financial_obligations": "",
        "_obligation_end_dates": "",
        "_has_bankruptcy": False,
        "_bankruptcy_status": "None",
        "_has_convictions": False,
        # Compliance
        "_acknowledged": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Page 1: Interactive Questionnaire
# ---------------------------------------------------------------------------


def render_questionnaire() -> None:
    """Render the full underwriting questionnaire (Sections A-H).

    Displays all form fields with conditional sections, collects data
    into :class:`Application`, and handles submission / validation.
    """
    init_session_state()

    st.title("\U0001F4C1 Underwriting Questionnaire")
    st.caption("Complete all sections to submit your insurance application.")

    load_config_yaml()  # noqa: F841

    # ------------------------------------------------------------------
    # Section A: Personal & Demographic
    # ------------------------------------------------------------------
    st.header("Section A: Personal & Demographic")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.text_input(
                "Full Name *",
                key="_full_name",
                help="Your full legal name",
            )
            st.date_input(
                "Date of Birth *",
                key="_date_of_birth",
                help="YYYY-MM-DD format",
                min_value=date(1900, 1, 1),
                max_value=date.today(),
            )
        with col2:
            st.selectbox(
                "Gender *",
                options=["Male", "Female", "Non-binary"],
                key="_gender",
            )
            st.selectbox(
                "Residency / Citizenship Status *",
                options=["Australian Citizen", "Permanent Resident", "Temporary Visa"],
                key="_residency_status",
            )
    st.text_area(
        "Contact Address *",
        key="_contact_address",
        help="Full postal address",
        height=60,
    )

    # ------------------------------------------------------------------
    # Section B: Cover Requested
    # ------------------------------------------------------------------
    st.header("Section B: Cover Requested")
    with st.container():
        benefit_options = get_benefit_options()
        selected = st.multiselect(
            "Which benefit types are you applying for? *",
            options=benefit_options,
            default=[],
            key="_benefit_types",
        )

        if BenefitType.DEATH in selected:
            st.number_input(
                "Sum Insured - Death (AUD) *",
                key="_sum_insured_death",
                min_value=0,
                step=10000,
            )
        else:
            st.session_state["_sum_insured_death"] = None

        if BenefitType.TPD in selected:
            st.number_input(
                "Sum Insured - TPD (AUD) *",
                key="_sum_insured_tpd",
                min_value=0,
                step=10000,
            )
        else:
            st.session_state["_sum_insured_tpd"] = None

        if BenefitType.TRAUMA in selected:
            st.number_input(
                "Sum Insured - Trauma/CI (AUD) *",
                key="_sum_insured_trauma",
                min_value=0,
                step=10000,
            )
        else:
            st.session_state["_sum_insured_trauma"] = None

        if BenefitType.IP in selected:
            c1, c2 = st.columns(2)
            with c1:
                st.number_input(
                    "Monthly Benefit (AUD) *",
                    key="_ip_monthly_benefit",
                    min_value=0,
                    step=500,
                )
            with c2:
                st.selectbox(
                    "Benefit Period *",
                    options=["2yr", "5yr", "to age 65", "to age 70"],
                    key="_ip_benefit_period",
                )
            st.radio(
                "Agreed Value or Indemnity?",
                options=[True, False],
                format_func=lambda x: "Agreed Value" if x else "Indemnity",
                key="_ip_agreed_value",
            )

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.checkbox(
                "Do you currently hold any other life insurance policies? *",
                key="_has_other_policies",
            )
        with col_b:
            st.checkbox(
                "Have you ever had an application declined, deferred, or offered with special terms? *",
                key="_previous_declination",
            )

        if st.session_state.get("_has_other_policies", False):
            st.number_input(
                "Total existing policies count",
                key="_total_existing_policies",
                min_value=0,
                step=1,
            )
            st.text_area(
                "Policy details (insurer(s) and amounts)",
                key="_other_policy_details",
                height=60,
                help="Required if you have other policies",
            )

    # ------------------------------------------------------------------
    # Section C: Occupation & Income
    # ------------------------------------------------------------------
    st.header("Section C: Occupation & Income")
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            st.text_input(
                "Current Occupation (full description of duties) *",
                key="_occupation",
                help="e.g. 'Software Engineer - writes code, attends meetings'",
            )
            st.text_input(
                "Employer Name *",
                key="_employer_name",
            )
        with c2:
            st.number_input(
                "Years in this Occupation *",
                key="_years_in_occupation",
                min_value=0.0,
                step=0.5,
            )
            st.number_input(
                "Annual Earned Income Before Tax (AUD) *",
                key="_annual_income",
                min_value=0,
                step=1000,
            )

    st.checkbox(
        "Does your occupation involve hazardous duties?",
        key="_has_hazardous_duties",
        help="Working at heights, underground, with heavy machinery, offshore, or with hazardous chemicals",
    )
    if st.session_state.get("_has_hazardous_duties", False):
        st.text_area(
            "Describe the hazardous duties and frequency",
            key="_hazardous_duties_desc",
            height=60,
        )

    # ------------------------------------------------------------------
    # Section D: Health — General
    # ------------------------------------------------------------------
    st.header("Section D: Health — General")
    with st.container():
        d1, d2, d3 = st.columns(3)
        with d1:
            st.number_input(
                "Height (cm) *",
                key="_height_cm",
                min_value=0.0,
                step=0.5,
            )
        with d2:
            st.number_input(
                "Weight (kg) *",
                key="_weight_kg",
                min_value=0.0,
                step=0.5,
            )
        with d3:
            st.selectbox(
                "Smoker Status *",
                options=["Never", "Former", "Current"],
                key="_smoker_status",
            )

    smoker = st.session_state.get("_smoker_status", "Never")
    if smoker in ("Current", "Former"):
        with st.container():
            s1, s2 = st.columns(2)
            with s1:
                st.number_input(
                    "Cigarettes per Day",
                    key="_cigarettes_per_day",
                    min_value=0,
                    step=1,
                )
            with s2:
                st.number_input(
                    "Years Smoked",
                    key="_years_smoked",
                    min_value=0,
                    step=1,
                )
    if smoker == "Former":
        st.number_input(
            "Years Since Quitting",
            key="_years_since_quit",
            min_value=0.0,
            step=0.5,
        )

    st.divider()

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.checkbox(
            "Are you currently taking any medications? *",
            key="_taking_medications",
        )
    with col_m2:
        st.checkbox(
            "Do you consume alcohol? *",
            key="_consumes_alcohol",
        )

    if st.session_state.get("_taking_medications", False):
        st.text_area(
            "List all medications, dosage, and the condition being treated",
            key="_medication_details",
            height=80,
        )

    if st.session_state.get("_consumes_alcohol", False):
        st.number_input(
            "Standard Drinks per Week",
            key="_standard_drinks_per_week",
            min_value=0,
            step=1,
        )

    st.divider()

    st.checkbox(
        "Have you ever been diagnosed with or treated for any medical condition? *",
        key="_has_medical_conditions",
    )

    # ------------------------------------------------------------------
    # Section E: Pre-existing Condition Follow-up (conditional)
    # ------------------------------------------------------------------
    if st.session_state.get("_has_medical_conditions", False):
        st.header("Section E: Health — Pre-existing Condition Follow-up")
        st.caption("Provide details for each condition. Add more as needed.")

        num = st.session_state.get("_num_conditions", 0)
        add_cond = st.button("\U00002795 Add Condition")
        if add_cond:
            num += 1
            st.session_state["_num_conditions"] = num

        for i in range(num):
            st.subheader(f"Condition {i + 1}")
            prefix = f"_cond_{i}_"
            with st.container():
                e1, e2 = st.columns(2)
                with e1:
                    st.text_input(
                        "Condition Name *",
                        key=f"{prefix}name",
                        help="e.g. Type 2 Diabetes",
                    )
                    st.date_input(
                        "Date of First Diagnosis",
                        key=f"{prefix}diag_date",
                    )
                    st.text_input(
                        "Treating Doctor Name *",
                        key=f"{prefix}doctor_name",
                    )
                with e2:
                    st.date_input(
                        "Treatment Start Date",
                        key=f"{prefix}treat_start",
                    )
                    st.date_input(
                        "Last Symptom Date",
                        key=f"{prefix}last_symptom",
                    )
                    st.text_input(
                        "Treating Doctor Contact",
                        key=f"{prefix}doctor_contact",
                        help="Phone or email",
                    )
            st.text_area(
                "Diagnostic Tests Performed",
                key=f"{prefix}tests",
                height=50,
            )
            st.text_area(
                "Treatment Received / Currently Receiving",
                key=f"{prefix}treatment",
                height=50,
            )
            st.text_area(
                "Symptoms Experienced",
                key=f"{prefix}symptoms",
                height=50,
            )
            st.text_input(
                "Symptom Frequency",
                key=f"{prefix}symptom_freq",
                help="e.g. daily, weekly, monthly",
            )
            st.text_area(
                "Hospitalisations for this Condition",
                key=f"{prefix}hospital",
                height=50,
            )
            st.text_input(
                "Time Off Work Due to this Condition",
                key=f"{prefix}time_off",
            )
            st.checkbox(
                "Lifestyle Affected?",
                key=f"{prefix}lifestyle_affected",
            )
            st.divider()

    # ------------------------------------------------------------------
    # Section F: Family History (conditional)
    # ------------------------------------------------------------------
    st.header("Section F: Family History")
    st.checkbox(
        "Do any immediate family members (parents, siblings) have or have had: "
        "heart disease, cancer, diabetes, stroke, or genetic conditions? *",
        key="_has_family_history",
    )

    if st.session_state.get("_has_family_history", False):
        st.header("Section F: Family History Details")
        st.caption("Provide details for each family member.")

        num_fam = st.session_state.get("_num_family", 0)
        add_fam = st.button("\U00002795 Add Family Member")
        if add_fam:
            num_fam += 1
            st.session_state["_num_family"] = num_fam

        for i in range(num_fam):
            prefix = f"_fam_{i}_"
            with st.container():
                f1, f2, f3 = st.columns(3)
                with f1:
                    st.selectbox(
                        "Relationship",
                        options=["father", "mother", "brother", "sister"],
                        key=f"{prefix}relationship",
                    )
                with f2:
                    st.text_input(
                        "Condition",
                        key=f"{prefix}condition",
                        help="e.g. Heart Disease, Melanoma",
                    )
                with f3:
                    st.number_input(
                        "Age at Diagnosis",
                        key=f"{prefix}age",
                        min_value=0,
                        step=1,
                    )
            st.divider()

    # ------------------------------------------------------------------
    # Section G: Lifestyle
    # ------------------------------------------------------------------
    st.header("Section G: Lifestyle")
    with st.container():
        g1, g2 = st.columns(2)
        with g1:
            st.checkbox(
                "Do you participate in any hazardous sports or pastimes? *",
                key="_has_hazardous_pursuits",
                help="Skydiving, rock climbing, motor racing, scuba diving, contact sports",
            )
            st.checkbox(
                "Do you use any recreational drugs? *",
                key="_recreational_drug_use",
            )
        with g2:
            st.checkbox(
                "Have you ever received medical treatment for alcohol dependency or drug abuse? *",
                key="_alcohol_drug_treatment",
            )
            st.checkbox(
                "Have you travelled to or do you plan to travel to any countries with known health or safety risks? *",
                key="_has_high_risk_travel",
            )

    if st.session_state.get("_has_hazardous_pursuits", False):
        st.subheader("Hazardous Pursuits Details")
        num_p = st.session_state.get("_num_pursuits", 0)
        add_p = st.button("\U00002795 Add Pursuit")
        if add_p:
            num_p += 1
            st.session_state["_num_pursuits"] = num_p

        for i in range(num_p):
            prefix = f"_pursuit_{i}_"
            with st.container():
                p1, p2, p3 = st.columns(3)
                with p1:
                    st.text_input(
                        "Activity",
                        key=f"{prefix}activity",
                        help="e.g. Scuba Diving",
                    )
                with p2:
                    st.text_input(
                        "Frequency",
                        key=f"{prefix}frequency",
                        help="e.g. Monthly, Weekly",
                    )
                with p3:
                    st.selectbox(
                        "Level",
                        options=["amateur", "professional"],
                        key=f"{prefix}level",
                    )
            st.divider()

    if st.session_state.get("_recreational_drug_use", False):
        st.text_area(
            "What, how often, and when was last use?",
            key="_drug_use_details",
            height=60,
        )

    if st.session_state.get("_has_high_risk_travel", False):
        st.text_area(
            "Travel details (countries, purpose, dates)",
            key="_high_risk_travel_details",
            height=60,
        )

    # ------------------------------------------------------------------
    # Section H: Financial
    # ------------------------------------------------------------------
    st.header("Section H: Financial")
    with st.container():
        h1, h2 = st.columns(2)
        with h1:
            st.number_input(
                "Total Net Worth (AUD)",
                key="_total_net_worth",
                min_value=0,
                step=10000,
                help="Required for sums insured > $1M",
            )
            st.text_area(
                "Financial Obligations Being Covered *",
                key="_financial_obligations",
                height=60,
                help="e.g. Mortgage, income replacement, business loan",
            )
        with h2:
            st.text_area(
                "Obligation End Dates *",
                key="_obligation_end_dates",
                height=60,
                help="e.g. Mortgage ends 2035, car loan ends 2028",
            )
            st.checkbox(
                "Have you ever been declared bankrupt or entered into a debt agreement? *",
                key="_has_bankruptcy",
            )
            st.checkbox(
                "Do you have any criminal convictions? *",
                key="_has_convictions",
            )

    if st.session_state.get("_has_bankruptcy", False):
        st.radio(
            "Bankruptcy Status",
            options=["None", "Current", "Recent", "Discharged"],
            key="_bankruptcy_status",
        )

    # ------------------------------------------------------------------
    # Compliance
    # ------------------------------------------------------------------
    st.header("Compliance")
    st.checkbox(
        "I acknowledge this is my Duty of Disclosure. I understand I must "
        "disclose all material information. *",
        key="_acknowledged",
    )

    st.divider()

    # Submit button
    col_submit, col_reset = st.columns([1, 1])
    with col_submit:
        if st.button("\U0001F4E9 Submit Application", type="primary", use_container_width=True):
            with st.spinner("\U0001F916 Running underwriting agents \u2014 assessing your application..."):
                _handle_submit()
    with col_reset:
        if st.button("\U0001F504 Reset Form", use_container_width=True):
            _reset_form()


def _handle_submit() -> None:
    """Validate form data and submit the application.

    Builds an :class:`Application` from session state, validates it,
    runs the agents, and stores results in session state.
    """
    # Clear previous results
    st.session_state["_errors"] = None
    st.session_state["_results"] = None

    # Basic required-field validation
    errors: list[str] = []
    if not st.session_state.get("_full_name", "").strip():
        errors.append("Full Name is required.")
    if not st.session_state.get("_occupation", "").strip():
        errors.append("Occupation is required.")
    if not st.session_state.get("_employer_name", "").strip():
        errors.append("Employer Name is required.")
    if not st.session_state.get("_contact_address", "").strip():
        errors.append("Contact Address is required.")
    if not st.session_state.get("_benefit_types"):
        errors.append("At least one benefit type is required.")
    if not st.session_state.get("_acknowledged", False):
        errors.append("Duty of Disclosure acknowledgment is required.")

    # Validate numeric fields
    height = st.session_state.get("_height_cm", 0)
    weight = st.session_state.get("_weight_kg", 0)
    if height and (height < 50 or height > 280):
        errors.append("Height must be between 50 and 280 cm.")
    if weight and (weight < 10 or weight > 500):
        errors.append("Weight must be between 10 and 500 kg.")
    if st.session_state.get("_annual_income", 0) <= 0:
        errors.append("Annual income must be greater than 0.")
    if st.session_state.get("_years_in_occupation", 0) < 0:
        errors.append("Years in occupation cannot be negative.")

    if errors:
        st.session_state["_errors"] = errors
        st.error("Please fix the following errors:")
        for err in errors:
            st.error(err)
        return

    # Build application and validate with Pydantic
    try:
        application = build_application_from_form()
        st.session_state["_application"] = application
    except ValidationError as exc:
        st.session_state["_errors"] = [str(exc)]
        st.error("Validation error:")
        for err in exc.errors():
            st.error(f"{err['loc']}: {err['msg']}")
        return

    # Run agents (no real LLM calls — deterministic only)
    try:
        _run_agents(application)
    except Exception as exc:
        st.session_state["_errors"] = [str(exc)]
        st.error("Error running agents:")
        st.error(str(exc))
        return

    st.session_state["_submitted"] = True

    # Auto-create conversation from results and save for debate view
    try:
        from underwriting.debate.chat_models import ChatMessage, Conversation
        from underwriting.debate.persistence import ConversationStore

        results = st.session_state.get("_results", {})
        application = st.session_state.get("_application")

        applicant_name = application.full_name if application else "Unknown Applicant"
        conv_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

        # Ensure chat_store exists
        if st.session_state.get("chat_store") is None:
            st.session_state.chat_store = ConversationStore()

        store = st.session_state.chat_store

        debate_log = results.get("debate_log", [])
        agent_assessments = results.get("agent_assessments", {})

        # Extract agent_assessments data for Conversation storage
        agent_assessments_data = {}
        for name, assessment in agent_assessments.items():
            agent_assessments_data[name] = {
                "agent_name": name,
                "risk_tier": assessment.risk_tier,
                "flags": assessment.flags,
                "recommendation": assessment.recommendation,
                "loading_range": assessment.loading_range,
                "confidence_score": assessment.confidence_score,
                "reasoning_summary": assessment.reasoning_summary,
                "additional_evidence_required": assessment.additional_evidence_required,
                "apra_references": assessment.apra_references,
            }

        conversation = Conversation(
            application_id=conv_id,
            applicant_name=applicant_name,
            debate_rounds=len(debate_log),
            final_decision=results.get("final_decision", ""),
            agents_participating=list(agent_assessments.keys()),
            agent_assessments=agent_assessments_data,
            applicant_data=application.model_dump(mode="json") if application else {},
        )

        # Add debate log entries as chat messages
        for entry in debate_log:
            msg = ChatMessage(
                sender=entry.get("agent", "Unknown"),
                content=entry.get("reasoning", ""),
                message_type="text",
                risk_tier_update=entry.get("updated_tier"),
                reasoning=entry.get("reasoning", ""),
            )
            conversation.add_message(msg)

        # Add initial system message
        conversation.add_system_message(
            f"Application submitted for {applicant_name}. "
            f"Final decision: {results.get('final_decision', 'Unknown')}. "
            f"Debate rounds: {conversation.debate_rounds}."
        )

        # Save conversation
        store.save(conversation)

        # Update session state to replace any old conversation
        st.session_state.chat_conversation_id = conv_id
        st.session_state.chat_conversation = conversation

        # Store application for chat interaction
        if application is not None:
            st.session_state.chat_application = application

        # Store agents for chat interaction
        st.session_state.chat_agents = {
            "Medical Agent": MedicalAgent(
                rules_path="rules/death/medical_rules.json",
                llm_client=LLMClient(config_path="./config.yaml")
                if os.path.exists("./config.yaml") else None,
            ),
            "Financial Agent": FinancialAgent(
                rules_path="rules/death/financial_rules.json",
                llm_client=LLMClient(config_path="./config.yaml")
                if os.path.exists("./config.yaml") else None,
            ),
            "Compliance Agent": ComplianceAgent(
                rules_path="rules/death/compliance_rules.json",
                llm_client=LLMClient(config_path="./config.yaml")
                if os.path.exists("./config.yaml") else None,
            ),
        }

    except Exception:
        import logging
        logging.warning("Failed to save conversation from Questionnaire submit")

    # Auto-navigate to Results & Debate on next page load
    st.session_state._pending_nav = "Results & Debate"
    st.rerun()


def _run_agents(application: Application) -> None:
    """Run the underwriting agents on the given application.

    Creates Medical, Financial, and Compliance agents with optional LLM
    enrichment, orchestrates debate, and stores results in session state.

    Args:
        application: The validated Application model.
    """
    # Attempt to load LLM client from config — falls back gracefully if
    # the endpoint is unreachable or config is missing.
    llm_client = None
    try:
        llm_client = LLMClient("./config.yaml")
        if not llm_client.is_available():
            llm_client = None
    except Exception:
        llm_client = None

    medical_agent = MedicalAgent(
        name="Medical Agent",
        rules_path="rules/death/medical_rules.json",
        llm_client=llm_client,
    )
    financial_agent = FinancialAgent(
        name="Financial Agent",
        rules_path="rules/death/financial_rules.json",
        llm_client=llm_client,
    )
    compliance_agent = ComplianceAgent(
        name="Compliance Agent",
        rules_path="rules/death/compliance_rules.json",
        llm_client=llm_client,
    )

    orchestrator = DebateOrchestrator(
        agents=[medical_agent, financial_agent, compliance_agent],
    )

    results = orchestrator.run(application)

    # Store results in session state
    st.session_state["_results"] = results


def _reset_form() -> None:
    """Clear all form-related session state keys."""
    keys_to_clear = [
        "_full_name", "_date_of_birth", "_gender", "_residency_status",
        "_contact_address", "_benefit_types", "_sum_insured_death",
        "_sum_insured_tpd", "_sum_insured_trauma", "_ip_monthly_benefit",
        "_ip_benefit_period", "_ip_agreed_value", "_has_other_policies",
        "_total_existing_policies", "_other_policy_details",
        "_previous_declination", "_occupation", "_employer_name",
        "_years_in_occupation", "_annual_income", "_has_hazardous_duties",
        "_hazardous_duties_desc", "_height_cm", "_weight_kg",
        "_smoker_status", "_cigarettes_per_day", "_years_smoked",
        "_years_since_quit", "_taking_medications", "_medication_details",
        "_has_medical_conditions", "_num_conditions", "_consumes_alcohol",
        "_standard_drinks_per_week", "_has_family_history", "_num_family",
        "_has_hazardous_pursuits", "_num_pursuits", "_recreational_drug_use",
        "_drug_use_details", "_alcohol_drug_treatment", "_has_high_risk_travel",
        "_high_risk_travel_details", "_total_net_worth", "_financial_obligations",
        "_obligation_end_dates", "_has_bankruptcy", "_bankruptcy_status",
        "_has_convictions", "_acknowledged",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["_submitted"] = False
    st.session_state["_results"] = None
    st.session_state["_errors"] = None
    st.rerun()


# ---------------------------------------------------------------------------
# Helpers for chat-style debate log
# ---------------------------------------------------------------------------

def _format_timestamp(iso_string: str) -> str:
    """Format an ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%H:%M")
    except (ValueError, TypeError):
        return iso_string[:16] if iso_string else ""


def _clean_chat_content(content: str) -> str:
    """Clean agent response content for display.

    If the content is a JSON response from the LLM (starts with ``{``
    and contains ``"response_text"``), extract just the response_text.
    Otherwise return the content as-is.
    """
    if not content or not content.strip():
        return content
    stripped = content.strip()
    if stripped.startswith("{") and '"response_text"' in stripped:
        try:
            import json as _json
            data = _json.loads(stripped)
            return data.get("response_text", content)
        except Exception:
            pass
    return content


def _render_chat_bubble(message: dict, agent_styles: dict) -> None:
    """Render a single chat message as a styled bubble using HTML/CSS.

    Args:
        message: Dict with sender, content, timestamp, risk_tier_update fields.
        agent_styles: Dict mapping sender names to style configs.
    """
    sender = message.get("sender", "unknown")
    content = message.get("content", "")
    timestamp = message.get("timestamp", "")
    tier_update = message.get("risk_tier_update")

    style = agent_styles.get(sender, agent_styles.get("user", {}))
    is_user = sender == "user"
    is_system = sender == "system"

    # Determine alignment and colors
    if is_user:
        align = "flex-end"
        bg_color = "#E3F2FD"
        border_color = "#90CAF9"
    elif is_system:
        align = "center"
        bg_color = "#F5F5F5"
        border_color = "#E0E0E0"
    else:
        align = "flex-start"
        color = style.get("color", "#2E86AB")
        bg_color = color + "15"  # 15% opacity
        border_color = color + "40"  # 40% opacity

    # Build emoji/name header
    emoji = style.get("emoji", "")
    name = sender

    # Build tier update badge
    tier_html = ""
    if tier_update:
        tier_html = f'<div style="margin-top:6px;padding-top:6px;border-top:1px solid #ddd;font-size:11px;color:#888;">Updated tier: <strong>{tier_update}</strong></div>'

    # Escape HTML in content to prevent injection
    import html as html_module
    safe_content = html_module.escape(content)

    # Build message bubble HTML
    bubble_html = f'''
    <div style="display: flex; justify-content: {align}; margin: 8px 0; width: 100%;">
        <div style="
            max-width: 75%;
            padding: 12px 16px;
            border-radius: 18px;
            background: {bg_color};
            border: 1px solid {border_color};
            margin-{"left" if is_user else "right"}: 8px;
            margin-{"right" if is_user else "left"}: 8px;
        ">
            <div style="font-size: 11px; color: #666; margin-bottom: 4px; display: flex; align-items: center; gap: 6px;">
                {emoji} <strong>{name}</strong>
                <span style="float: right; font-size: 10px; color: #999;">
                    {_format_timestamp(timestamp)}
                </span>
            </div>
            <div style="font-size: 14px; line-height: 1.5; white-space: pre-wrap;">
                {safe_content}
            </div>
            {tier_html}
        </div>
    </div>
    '''

    st.markdown(bubble_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page 2: Results & Debate (Combined)
# ---------------------------------------------------------------------------



def _normalize_results(
    results: dict | None,
    application=None,
    conversation=None,
) -> dict:
    """Normalize underwriting results into a consistent dict format.

    Handles results from three possible sources:
    - Direct ``results`` dict from orchestrator
    - ``conversation`` state from interactive debate
    - Empty defaults when neither is available

    ``AgentAssessment`` objects in *agent_assessments* are preserved as-is;
    raw dicts are converted via lazy import of the Pydantic model.

    Args:
        results: Dict returned by the underwriting engine, or ``None``.
        application: Application model (unused here, kept for API symmetry).
        conversation: Interactive debate Conversation, or ``None``.

    Returns:
        A dict with keys: ``final_decision``, ``decision_reasoning``,
        ``agent_assessments``, ``consensus_reached``, ``flags``,
        ``additional_evidence_required``.
    """
    from underwriting.agents.base_agent import AgentAssessment  # noqa: PLC0415

    # --- Source 1: direct results dict from orchestrator ---
    if results is not None and isinstance(results, dict):
        raw_assessments = results.get("agent_assessments", {})
        agent_assessments: dict[str, AgentAssessment] = {}
        for agent_name, assessment in raw_assessments.items():
            if isinstance(assessment, AgentAssessment):
                agent_assessments[agent_name] = assessment
            elif isinstance(assessment, dict):
                try:
                    agent_assessments[agent_name] = AgentAssessment(**assessment)
                except Exception:  # noqa: BLE001
                    agent_assessments[agent_name] = AgentAssessment(
                        agent_name=agent_name,
                        risk_tier="unknown",
                        recommendation="Unknown",
                        reasoning_summary="Could not parse assessment data.",
                    )
            else:
                agent_assessments[agent_name] = AgentAssessment(
                    agent_name=agent_name,
                    risk_tier="unknown",
                    recommendation="Unknown",
                    reasoning_summary="Unexpected assessment type.",
                )
        return {
            "final_decision": results.get("final_decision", "Unknown"),
            "decision_reasoning": results.get("decision_reasoning", ""),
            "agent_assessments": agent_assessments,
            "consensus_reached": results.get("consensus_reached", False),
            "flags": results.get("flags", []),
            "additional_evidence_required": results.get(
                "additional_evidence_required", []
            ),
        }

    # --- Source 2: conversation state from debate log ---
    if conversation is not None:
        raw_assessments = getattr(conversation, "agent_assessments", {})
        agent_assessments: dict[str, AgentAssessment] = {}
        for agent_name, assessment_data in raw_assessments.items():
            if isinstance(assessment_data, AgentAssessment):
                agent_assessments[agent_name] = assessment_data
            elif isinstance(assessment_data, dict):
                try:
                    agent_assessments[agent_name] = AgentAssessment(**assessment_data)
                except Exception:  # noqa: BLE001
                    agent_assessments[agent_name] = AgentAssessment(
                        agent_name=agent_name,
                        risk_tier="unknown",
                        recommendation="Unknown",
                        reasoning_summary="Could not parse assessment data.",
                    )
        return {
            "final_decision": conversation.final_decision or "Unknown",
            "decision_reasoning": "",
            "agent_assessments": agent_assessments,
            "consensus_reached": conversation.debate_rounds == 0,
            "flags": [],
            "additional_evidence_required": [],
        }

    # --- Source 3: nothing available ---
    return {
        "final_decision": "Unknown",
        "decision_reasoning": "",
        "agent_assessments": {},
        "consensus_reached": False,
        "flags": [],
        "additional_evidence_required": [],
    }


def _get_applicant_display_data(
    results: dict | None,
    application=None,
    conversation=None,
) -> dict:
    """Extract applicant display fields from the best available source.

    Prioritises the ``application`` model, falls back to ``conversation``,
    and returns placeholders when neither is available.

    Args:
        results: Underwriting results dict (unused here, kept for API symmetry).
        application: Application model instance, or ``None``.
        conversation: Debate Conversation, or ``None``.

    Returns:
        A dict with keys: ``name``, ``age``, ``bmi``, ``occupation_class``.
    """
    # --- Source 1: application model ---
    if application is not None:
        return {
            "name": application.full_name,
            "age": application.age,
            "bmi": application.bmi,
            "occupation_class": application.occupation_class.value,
        }

    # --- Source 2: conversation state ---
    if conversation is not None:
        return {
            "name": conversation.applicant_name,
            "age": "N/A",
            "bmi": "N/A",
            "occupation_class": "N/A",
        }

    # --- Source 3: nothing available ---
    return {
        "name": "Unknown",
        "age": "N/A",
        "bmi": "N/A",
        "occupation_class": "N/A",
    }


def _build_applicant_section_data(application):  # noqa: ANN001
    """Group Application model fields into labelled sections for display.

    Returns flat strings for simple fields, lists of dicts for list-type
    fields (medical conditions, family history, hazardous pursuits), and
    nested dicts for grouped fields (smoking details).

    Args:
        application: An Application model instance (or None).

    Returns:
        Dict mapping section_name -> dict of field_label -> value
        where value is str, list[dict], or dict[str, str].
        Returns empty dict if application is None.
    """
    if application is None:
        return {}

    def _fmt(val):  # noqa: ANN202
        """Format a scalar value for display. Lists are NOT flattened here."""
        if val is None or val == "":
            return "—"
        if isinstance(val, bool):
            return "Yes" if val else "No"
        if isinstance(val, float):
            if val == int(val):
                return f"{int(val):,}"
            return f"{val:,.1f}"
        if isinstance(val, int):
            return f"{val:,}"
        return str(val)

    from datetime import date as dt_date

    # ==================================================================
    # Section: Personal
    # ==================================================================
    personal: dict = {}  # type: ignore[var-annotated]
    personal["Full Name"] = _fmt(getattr(application, "full_name", ""))
    dob = getattr(application, "date_of_birth", None)
    if isinstance(dob, dt_date):
        personal["Date of Birth"] = dob.strftime("%d %b %Y")
        personal["Age"] = str(getattr(application, "age", "N/A"))
    else:
        personal["Date of Birth"] = str(dob) if dob else "—"
        personal["Age"] = "—"
    personal["Gender"] = _fmt(getattr(application, "gender", ""))
    personal["Residency"] = _fmt(getattr(application, "residency_status", ""))
    personal["Contact Address"] = _fmt(getattr(application, "contact_address", ""))

    # ==================================================================
    # Section: Cover Details
    # ==================================================================
    cover: dict = {}  # type: ignore[var-annotated]
    bt = getattr(application, "benefit_types", [])
    if bt:
        # Keep as list of strings for expander rendering
        cover["Benefit Types"] = [b.value for b in bt if hasattr(b, "value")]
    else:
        cover["Benefit Types"] = "—"
    cover["Sum Insured (Death)"] = _fmt(getattr(application, "sum_insured_death", None))
    cover["Sum Insured (TPD)"] = _fmt(getattr(application, "sum_insured_tpd", None))
    tpd_def = getattr(application, "tpd_definition", None)
    if tpd_def:
        cover["TPD Definition"] = tpd_def.value if hasattr(tpd_def, "value") else str(tpd_def)
    cover["Sum Insured (Trauma)"] = _fmt(getattr(application, "sum_insured_trauma", None))
    tcl = getattr(application, "trauma_condition_list", None)
    if tcl:
        cover["Trauma Conditions"] = tcl
    ip_benefit = getattr(application, "ip_monthly_benefit", None)
    cover["IP Monthly Benefit"] = _fmt(ip_benefit)
    ip_period = getattr(application, "ip_benefit_period", None)
    cover["IP Benefit Period"] = ip_period if ip_period else "—"
    ip_av = getattr(application, "ip_agreed_value", None)
    if ip_av is not None:
        cover["IP Agreed Value"] = "Yes" if ip_av else "No"
    cover["Has Other Policies"] = _fmt(getattr(application, "has_other_policies", False))
    existing = getattr(application, "total_existing_policies", 0)
    if existing:
        cover["Existing Policies"] = str(existing)
    other_pol = getattr(application, "other_policy_details", None)
    if other_pol and other_pol.strip():
        cover["Other Policy Details"] = other_pol
    cover["Previously Declined"] = _fmt(getattr(application, "previous_declination", False))

    # ==================================================================
    # Section: Occupation & Income
    # ==================================================================
    occupation: dict = {}  # type: ignore[var-annotated]
    occupation["Occupation"] = _fmt(getattr(application, "occupation", ""))
    occupation["Employer"] = _fmt(getattr(application, "employer_name", ""))
    occupation["Years in Occupation"] = _fmt(getattr(application, "years_in_occupation", None))
    occupation["Annual Income"] = _fmt(getattr(application, "annual_income", None))
    occ_class = getattr(application, "occupation_class", None)
    occupation["Occupation Class"] = occ_class.value if occ_class and hasattr(occ_class, "value") else "—"
    occupation["Hazardous Duties"] = _fmt(getattr(application, "has_hazardous_duties", False))
    haz_desc = getattr(application, "hazardous_duties_description", None)
    if haz_desc and str(haz_desc).strip():
        occupation["Hazardous Duties Details"] = str(haz_desc)

    # ==================================================================
    # Section: Health (includes Family History)
    # ==================================================================
    health: dict = {}  # type: ignore[var-annotated]
    health["Height (cm)"] = _fmt(getattr(application, "height_cm", None))
    health["Weight (kg)"] = _fmt(getattr(application, "weight_kg", None))
    health["BMI"] = _fmt(getattr(application, "bmi", None))

    # Smoking details — grouped
    smoker = getattr(application, "smoker_status", None)
    smoker_val = smoker.value if smoker and hasattr(smoker, "value") else "—"
    cpd = getattr(application, "cigarettes_per_day", None)
    ys = getattr(application, "years_smoked", None)
    ysq = getattr(application, "years_since_quit", None)
    smoking_details: dict[str, str] = {"Status": smoker_val}
    if smoker_val in ("Current", "Former"):
        smoking_details["Cigarettes / Day"] = str(cpd) if cpd else "—"
        smoking_details["Years Smoked"] = str(ys) if ys else "—"
    if smoker_val == "Former":
        smoking_details["Years Since Quit"] = str(ysq) if ysq else "—"
    health["Smoking"] = smoking_details

    health["Taking Medications"] = _fmt(getattr(application, "taking_medications", False))
    med_details = getattr(application, "medication_details", None)
    if med_details and str(med_details).strip():
        health["Medication Details"] = str(med_details)
    health["Has Medical Conditions"] = _fmt(getattr(application, "has_medical_conditions", False))
    mc = getattr(application, "medical_conditions", [])
    if mc:
        # Return list of dicts — each MedicalCondition becomes a dict of sub-fields
        conditions_list = []
        for cond in mc:
            cond_dict: dict[str, str] = {}
            cond_dict["Condition"] = getattr(cond, "condition_name", "Unknown")
            diag = getattr(cond, "diagnosis_date", None)
            cond_dict["Diagnosed"] = diag.strftime("%d %b %Y") if isinstance(diag, dt_date) else str(diag) if diag else "—"
            cond_dict["Doctor"] = getattr(cond, "treating_doctor_name", "—")
            cond_dict["Doctor Contact"] = getattr(cond, "treating_doctor_contact", "—")
            tests = getattr(cond, "diagnostic_tests", None)
            if tests:
                cond_dict["Diagnostic Tests"] = str(tests)
            treat_start = getattr(cond, "treatment_start_date", None)
            if treat_start:
                cond_dict["Treatment Start"] = treat_start.strftime("%d %b %Y") if isinstance(treat_start, dt_date) else str(treat_start)
            treat_desc = getattr(cond, "treatment_description", None)
            if treat_desc:
                cond_dict["Treatment"] = str(treat_desc)
            symptoms = getattr(cond, "symptoms", None)
            if symptoms:
                cond_dict["Symptoms"] = str(symptoms)
            freq = getattr(cond, "symptom_frequency", None)
            if freq:
                cond_dict["Symptom Frequency"] = str(freq)
            last_s = getattr(cond, "last_symptom_date", None)
            if last_s:
                cond_dict["Last Symptom"] = last_s.strftime("%d %b %Y") if isinstance(last_s, dt_date) else str(last_s)
            hosp = getattr(cond, "hospitalisations", None)
            if hosp:
                cond_dict["Hospitalisations"] = str(hosp)
            off_work = getattr(cond, "time_off_work", None)
            if off_work:
                cond_dict["Time Off Work"] = str(off_work)
            ls_aff = getattr(cond, "lifestyle_affected", None)
            if ls_aff is not None:
                cond_dict["Lifestyle Affected"] = "Yes" if ls_aff else "No"
            conditions_list.append(cond_dict)
        health["Medical Conditions"] = conditions_list
    else:
        health["Medical Conditions"] = "None"

    health["Consumes Alcohol"] = _fmt(getattr(application, "consumes_alcohol", False))
    drinks = getattr(application, "standard_drinks_per_week", None)
    if drinks and drinks > 0:
        health["Drinks / Week"] = str(drinks)

    # Family history — list of dicts
    fh = getattr(application, "family_history", [])
    if fh:
        family_list = []
        for item in fh:
            fam_dict: dict[str, str] = {}
            fam_dict["Relationship"] = getattr(item, "relationship", "—")
            fam_dict["Condition"] = getattr(item, "condition", "—")
            age_d = getattr(item, "age_at_diagnosis", None)
            fam_dict["Age at Diagnosis"] = str(age_d) if age_d else "—"
            family_list.append(fam_dict)
        health["Family History"] = family_list
    else:
        health["Family History"] = "None"

    # ==================================================================
    # Section: Lifestyle
    # ==================================================================
    lifestyle: dict = {}  # type: ignore[var-annotated]
    hp = getattr(application, "hazardous_pursuits", [])
    if hp:
        pursuit_list = []
        for p in hp:
            p_dict: dict[str, str] = {}
            p_dict["Activity"] = getattr(p, "activity", "—")
            p_dict["Frequency"] = getattr(p, "frequency", "—")
            p_dict["Level"] = getattr(p, "level", "—")
            pursuit_list.append(p_dict)
        lifestyle["Hazardous Pursuits"] = pursuit_list
    else:
        lifestyle["Hazardous Pursuits"] = "None"
    lifestyle["Recreational Drug Use"] = _fmt(getattr(application, "recreational_drug_use", False))
    drug_d = getattr(application, "drug_use_details", None)
    if drug_d and str(drug_d).strip():
        lifestyle["Drug Use Details"] = str(drug_d)
    lifestyle["Alcohol/Drug Treatment"] = _fmt(getattr(application, "alcohol_drug_treatment", False))
    lifestyle["High-Risk Travel"] = _fmt(getattr(application, "has_high_risk_travel", False))
    travel_d = getattr(application, "high_risk_travel_details", None)
    if travel_d and str(travel_d).strip():
        lifestyle["Travel Details"] = str(travel_d)

    # ==================================================================
    # Section: Financial
    # ==================================================================
    financial: dict = {}  # type: ignore[var-annotated]
    financial["Net Worth"] = _fmt(getattr(application, "total_net_worth", None))
    financial["Financial Obligations"] = _fmt(getattr(application, "financial_obligations", None))
    obl_end = getattr(application, "obligation_end_dates", None)
    if obl_end and str(obl_end).strip():
        financial["Obligation End Dates"] = str(obl_end)
    bk = getattr(application, "bankruptcy_status", "")
    financial["Bankruptcy Status"] = str(bk) if bk and bk != "None" else "None"
    prev_bk = getattr(application, "previous_bankruptcy", False)
    if prev_bk:
        financial["Previous Bankruptcy"] = "Yes"
    financial["Criminal Convictions"] = _fmt(getattr(application, "criminal_convictions", False))

    return {
        "Personal": personal,
        "Cover Details": cover,
        "Occupation & Income": occupation,
        "Health": health,
        "Lifestyle": lifestyle,
        "Financial": financial,
    }


def _generate_decision_summary(
    agent_assessments: dict,
    final_decision: str,
    llm_client=None,
    applicant_name: str = "",
    applicant_context: str = "",
) -> str:
    """Generate a plain-language summary of the underwriting decision.

    Builds a prompt from agent assessments (risk tiers, flags, matched rules,
    recommendations) and application context. Calls the LLM to produce a
    2-3 paragraph explanation tailored to the specific application. Falls
    back to a deterministic summary if the LLM is unavailable.

    Args:
        agent_assessments: Dict of agent name -> AgentAssessment or dict.
        final_decision: The final underwriting decision string.
        llm_client: Optional LLMClient instance for LLM-powered summary.
        applicant_name: Applicant's full name.
        applicant_context: Context string with age, occupation, BMI, etc.

    Returns:
        A plain-language summary of the decision.
    """
    # Build detailed assessment text with matched rules
    assessment_lines: list[str] = []
    for agent_name, assessment in agent_assessments.items():
        if "Compliance" in agent_name:
            continue  # Compliance is observer only
        if hasattr(assessment, "risk_tier"):
            risk_tier = assessment.risk_tier
            recommendation = assessment.recommendation
            reasoning = assessment.reasoning_summary
            flags = assessment.flags
            confidence = assessment.confidence_score
        elif isinstance(assessment, dict):
            risk_tier = assessment.get("risk_tier", "standard")
            recommendation = assessment.get("recommendation", "standard")
            reasoning = assessment.get("reasoning_summary", "")
            flags = assessment.get("flags", [])
            confidence = assessment.get("confidence_score", 1.0)
        else:
            continue

        line = (
            f"- **{agent_name}** (Risk Tier: {risk_tier.upper()}, "
            f"Recommendation: {recommendation}, Confidence: {confidence:.0%})"
        )
        if reasoning:
            line += f"\n  Reasoning: {reasoning[:300]}"
        if flags:
            for flag in flags[:10]:
                if isinstance(flag, dict):
                    rule_id = flag.get("rule_id", "?")
                    severity = flag.get("severity", "?")
                    desc = flag.get("description", "")
                    line += f"\n  Matched Rule [{severity}] {rule_id}: {desc}"
                else:
                    line += f"\n  Flag: {flag}"
        assessment_lines.append(line)

    assessments_text = "\n".join(assessment_lines) if assessment_lines else "No agent assessments available."

    # Build applicant context
    context = f"Applicant: {applicant_name or 'Unknown'}"
    if applicant_context:
        context += f" ({applicant_context})"

    # Build LLM prompt
    prompt = (
        "You are an Australian life insurance underwriting specialist. "
        "Write a detailed 2-3 paragraph summary explaining the underwriting decision "
        "for the following life insurance application. "
        "Focus on: which specific rules were breached, why each agent raised concerns, "
        "and how those factors led to the final decision. "
        "Use plain English that a non-expert applicant could understand. "
        "Be specific about the conditions and rules — do not use generic language.\n\n"
        f"{context}\n"
        f"Final Decision: {final_decision}\n\n"
        f"Agent Assessments with Matched Rules:\n{assessments_text}\n\n"
        "Explain clearly: (1) What specific factors were found, "
        "(2) Why they matter for underwriting, and "
        "(3) What the final decision means for the applicant."
    )

    # Try LLM first
    if llm_client is not None and llm_client.is_available():
        summary = llm_client.generate(prompt)
        if summary and summary != "[LLM unavailable - using deterministic fallback]":
            return summary

    # Deterministic fallback — still tailored with specific rules
    non_standard_agents = []
    standard_agents = []
    for agent_name, assessment in agent_assessments.items():
        if hasattr(assessment, "risk_tier"):
            tier = assessment.risk_tier
        elif isinstance(assessment, dict):
            tier = assessment.get("risk_tier", "standard")
        else:
            continue
        if tier == "standard":
            standard_agents.append(agent_name)
        else:
            non_standard_agents.append((agent_name, tier, assessment))

    name_str = applicant_name or "the applicant"

    if not non_standard_agents:
        return (
            f"Underwriting agents (Medical and Financial) reviewed "
            f"{name_str}'s application and found no significant risk factors. (Compliance Agent reviewed for regulatory observations only). "
            f"The application meets standard underwriting criteria, and a decision of "
            f"**{final_decision}** was reached."
        )

    agent_details = []
    for name, tier, assessment in non_standard_agents:
        detail = f"- **{name}** assessed risk tier as **{tier.upper()}**:"
        flags_data = []
        if hasattr(assessment, "flags"):
            flags_data = assessment.flags
        elif isinstance(assessment, dict):
            flags_data = assessment.get("flags", [])
        for flag in flags_data[:5]:
            if isinstance(flag, dict):
                rule_id = flag.get("rule_id", "")
                desc = flag.get("description", "")
                sev = flag.get("severity", "")
                detail += f"\n  - [{sev}] {rule_id}: {desc}"
        agent_details.append(detail)

    details = "\n".join(agent_details)

    return (
        f"The underwriting team assessed {name_str}'s application and reached a decision "
        f"of **{final_decision}**. Here is a summary of the key factors:\n\n"
        f"{details}\n\n"
        f"This decision is based on the combined assessments of specialist underwriting "
        f"agents (Medical and Financial). Compliance Agent provided regulatory "
        f"observations but did not influence the decision. "
        f"If you have questions or would "
        f"like to provide additional information, please consult an underwriter."
    )


# Color scheme for risk tier badges
_RISK_TIER_COLORS = {
    "standard": "#27ae60",
    "loading": "#f39c12",
    "refer": "#3498db",
    "decline": "#e74c3c",
}


def _render_non_standard_conditions(agent_assessments: dict) -> None:
    """Render non-standard risk tier conditions with color-coded badges.

    For each agent whose risk_tier != "standard", display a highlighted
    section with the agent name, color-coded tier badge, and the flags
    that explain the non-standard decision.

    Args:
        agent_assessments: Dict of agent name -> AgentAssessment or dict.
    """
    non_standard = []
    for agent_name, assessment in agent_assessments.items():
        if "Compliance" in agent_name:
            continue
        if hasattr(assessment, "risk_tier"):
            risk_tier = assessment.risk_tier
            flags = assessment.flags
        elif isinstance(assessment, dict):
            risk_tier = assessment.get("risk_tier", "standard")
            flags = assessment.get("flags", [])
        else:
            continue

        if risk_tier != "standard":
            non_standard.append((agent_name, risk_tier, flags))

    if not non_standard:
        return

    st.subheader("\u26A0\uFE0F Non-Standard Conditions")

    for agent_name, risk_tier, flags in non_standard:
        color = _RISK_TIER_COLORS.get(risk_tier, "#6c757d")
        tier_label = risk_tier.replace("_", " ").title()

        # Build HTML badge
        badge_html = (
            f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
            f'background-color:{color};color:#fff;font-size:12px;font-weight:600;'
            f'margin-left:8px;">{tier_label}</span>'
        )

        st.markdown(
            f"**{agent_name}** {badge_html}",
            unsafe_allow_html=True,
        )

        # Show flags/matched rules
        if flags:
            flag_items = []
            for flag in flags:
                if isinstance(flag, dict):
                    severity = flag.get("severity", "")
                    description = flag.get("description", flag.get("rule_id", ""))
                    flag_items.append(f"- <strong>[{severity}]</strong> {description}")
                else:
                    flag_items.append(f"- {flag}")
            st.markdown(
                "<div style='margin-left:20px;font-size:13px;color:#555;'>"
                + "<br>".join(flag_items)
                + "</div>",
                unsafe_allow_html=True,
            )

        st.divider()


def render_results_and_debate() -> None:
    """Display underwriting results and interactive debate log on one page.

    Results section at top (applicant summary, decision, agent assessments).
    Debate log section below (sidebar with conversations, chat bubbles, input).
    """
    # Initialize chat session state
    if "chat_conversation_id" not in st.session_state:
        st.session_state.chat_conversation_id = None
    if "chat_conversation" not in st.session_state:
        st.session_state.chat_conversation = None
    if "chat_store" not in st.session_state:
        try:
            from underwriting.debate.persistence import ConversationStore
            st.session_state.chat_store = ConversationStore()
        except Exception:
            st.session_state.chat_store = None
    if "chat_input_sent" not in st.session_state:
        st.session_state.chat_input_sent = False

    # If we just ran a pipeline, load the conversation
    if (st.session_state.get("chat_conversation_id") and
        st.session_state.get("chat_conversation") is None and
        st.session_state.get("chat_store")):
        st.session_state.chat_conversation = st.session_state.chat_store.load(
            st.session_state.chat_conversation_id
        )

    st.title("\U0001F4C1 Results & Debate")

    # ====================================================================
    # Application History — load / delete saved conversations
    # ====================================================================
    with st.expander("\U0001F4C1 Application History", expanded=False):
        if st.session_state.chat_store:
            conversations = st.session_state.chat_store.list_applications()

            if conversations:
                for conv in conversations:
                    c1, c2, c3 = st.columns([3, 1, 1])
                    with c1:
                        st.markdown(f"**{conv.get('applicant_name', 'Unknown')}**")
                        st.caption(
                            f"{conv.get('created_at', '')[:16]} \u2022 "
                            f"{conv.get('status', 'active')} \u2022 "
                            f"Decision: {conv.get('final_decision', 'N/A')}"
                        )
                    with c2:
                        if st.button("Load", key=f"load_{conv['application_id']}"):
                            st.session_state.chat_conversation_id = conv["application_id"]
                            st.session_state.chat_conversation = (
                                st.session_state.chat_store.load(conv["application_id"])
                            )
                            # Clear stale application data so conversation is the
                            # authoritative source for applicant display
                            st.session_state["_application"] = None
                            st.session_state["_results"] = None
                            st.session_state.test_q_results = None
                            # Reconstruct application from stored data for detail display
                            conversation = st.session_state.chat_conversation
                            saved_data = getattr(conversation, "applicant_data", None) if conversation else None
                            if saved_data:
                                try:
                                    from underwriting.application.schema import Application as AppModel
                                    st.session_state.chat_application = (
                                        AppModel.model_validate(saved_data)
                                    )
                                except Exception:
                                    st.session_state.chat_application = None
                            # Recreate agents for chat interaction
                            st.session_state.chat_agents = {
                                "Medical Agent": MedicalAgent(
                                    rules_path="rules/death/medical_rules.json",
                                    llm_client=LLMClient(config_path="./config.yaml")
                                    if os.path.exists("./config.yaml") else None,
                                ),
                                "Financial Agent": FinancialAgent(
                                    rules_path="rules/death/financial_rules.json",
                                    llm_client=LLMClient(config_path="./config.yaml")
                                    if os.path.exists("./config.yaml") else None,
                                ),
                                "Compliance Agent": ComplianceAgent(
                                    rules_path="rules/death/compliance_rules.json",
                                    llm_client=LLMClient(config_path="./config.yaml")
                                    if os.path.exists("./config.yaml") else None,
                                ),
                            }
                            st.rerun()
                    with c3:
                        if st.button("Delete", key=f"del_{conv['application_id']}"):
                            st.session_state.chat_store.delete(conv["application_id"])
                            if st.session_state.chat_conversation_id == conv["application_id"]:
                                st.session_state.chat_conversation_id = None
                                st.session_state.chat_conversation = None
                            st.rerun()
                st.divider()
            else:
                st.info("No saved conversations.")

            # New application button
            if st.button("\U0001F534 New Application"):
                st.session_state.chat_conversation_id = None
                st.session_state.chat_conversation = None
                st.session_state["_application"] = None
                st.session_state["_results"] = None
                st.session_state.test_q_results = None
                st.rerun()
        else:
            st.info("Conversation store not available.")

    # ====================================================================
    # Determine results source (priority order) — used by hero card + expanders
    # ====================================================================
    conversation = st.session_state.get("chat_conversation")
    main_results = st.session_state.get("_results")
    main_application = st.session_state.get("_application")
    test_results = st.session_state.get("test_q_results")

    # Priority: conversation > main_results > test_results
    active_results = None

    if conversation is not None:
        active_results = _normalize_results(None, None, conversation)
    elif main_results is not None:
        active_results = _normalize_results(main_results, main_application, None)
    elif test_results is not None:
        test_inner = test_results.get("results", test_results)
        active_results = _normalize_results(test_inner, main_application, None)

    if active_results is None or not active_results.get("agent_assessments"):
        st.info("No results available. Submit an application or run the Test Questionnaire pipeline.")
    else:
        # --- Extract common data for hero card and expanders ---
        display = _get_applicant_display_data(active_results, main_application, conversation)
        final_decision = active_results.get("final_decision", "Unknown")
        agent_assessments = active_results.get("agent_assessments", {})

        if conversation is not None:
            source_label = "Conversation / Debate"
        elif test_results is not None:
            source_label = "Test Questionnaire"
        else:
            source_label = "Main Questionnaire"

        # Non-standard count
        ns_count = 0
        for agent_name, a in agent_assessments.items():
            if "Compliance" in agent_name:
                continue
            tier = a.risk_tier if hasattr(a, "risk_tier") else a.get("risk_tier", "standard") if isinstance(a, dict) else "standard"
            if tier != "standard":
                ns_count += 1

        ns_text = f"\u26A0\uFE0F {ns_count} agent(s) flagged non-standard" if ns_count else "\u2705 All agents assessed as standard"

        # Decision color map
        _DC = {
            "Standard Offer": ("\u2705", "#27ae60"),
            "Offer with Loading/Exclusion": ("\u26A0\uFE0F", "#f39c12"),
            "Refer to Manual Underwriting": ("\U0001F50D", "#3498db"),
            "Decline": ("\U0001F534", "#e74c3c"),
        }
        d_icon, d_color = _DC.get(final_decision, ("\u2753", "#6c757d"))

        # ================================================================
        # HERO CARD
        # ================================================================
        card_html = f"""<div style="
            background: linear-gradient(135deg, {d_color} 0%, {d_color}dd 100%);
            border-radius: 16px; padding: 24px 28px; margin: 0 0 8px 0;
            color: #ffffff; box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        ">
            <div style="font-size:28px;font-weight:700;margin-bottom:4px;">
                {d_icon}  {final_decision}
            </div>
            <div style="font-size:15px;opacity:0.90;margin-bottom:12px;">
                {display["name"]}  \u00b7  {source_label}
            </div>
            <div style="font-size:13px;opacity:0.85;">
                {ns_text}
            </div>
        </div>"""
        st.markdown(card_html, unsafe_allow_html=True)

        # ================================================================
        # EXPANDER 1 \u2014 Application Details
        # ================================================================
        with st.expander("\U0001F4CB Application Details", expanded=False):
            # Try main application first, fall back to chat_application (from pipeline)
            app_data = main_application or st.session_state.get("chat_application")
            sections = _build_applicant_section_data(app_data)

            if sections:
                # Quick summary row
                c1, c2, c3, c4 = st.columns(4)
                personal_sec = sections.get("Personal", {})
                health_sec = sections.get("Health", {})
                occ_sec = sections.get("Occupation & Income", {})
                with c1:
                    st.metric("Name", personal_sec.get("Full Name", "N/A"))
                with c2:
                    st.metric("Age", personal_sec.get("Age", "N/A"))
                with c3:
                    st.metric("BMI", health_sec.get("BMI", "N/A"))
                with c4:
                    st.metric("Occupation Class", occ_sec.get("Occupation Class", "N/A"))

                # Nested expanders for each section
                section_icons = {
                    "Personal": "\U0001F464",
                    "Cover Details": "\U0001F4DD",
                    "Occupation & Income": "\U0001F4BC",
                    "Health": "\U0001FA7A",
                    "Lifestyle": "\U0001F3C3",
                    "Financial": "\U0001F4B0",
                }
                for sec_name, sec_fields in sections.items():
                    icon = section_icons.get(sec_name, "\U0001F4CB")
                    with st.expander(f"{icon} {sec_name}", expanded=False):
                        # Split into inline (simple) and block (nested) fields
                        inline_fields: list[tuple[str, str]] = []
                        block_fields: list[tuple[str, object]] = []

                        for label, value in sec_fields.items():
                            if isinstance(value, str):
                                inline_fields.append((label, value))
                            elif isinstance(value, list) and all(isinstance(x, str) for x in value):
                                # List of strings — join for inline display
                                joined = ", ".join(value) if value else "—"
                                inline_fields.append((label, joined))
                            else:
                                # List of dicts or nested dict — block display
                                block_fields.append((label, value))

                        # Render inline fields in 2-column layout
                        if inline_fields:
                            for i in range(0, len(inline_fields), 2):
                                cols = st.columns(2)
                                with cols[0]:
                                    lbl, val = inline_fields[i]
                                    if val:
                                        st.markdown(f"**{lbl}:** {val}")
                                if i + 1 < len(inline_fields):
                                    with cols[1]:
                                        lbl, val = inline_fields[i + 1]
                                        if val:
                                            st.markdown(f"**{lbl}:** {val}")

                        # Render block fields as sub-expanders
                        for label, value in block_fields:
                            if isinstance(value, dict):
                                # Nested dict (e.g., Smoking details)
                                with st.expander(f"\U0001F4CB {label}", expanded=False):
                                    sub_items = list(value.items())
                                    for j in range(0, len(sub_items), 2):
                                        sc = st.columns(2)
                                        with sc[0]:
                                            sl, sv = sub_items[j]
                                            if sv:
                                                st.markdown(f"**{sl}:** {sv}")
                                        if j + 1 < len(sub_items):
                                            with sc[1]:
                                                sl, sv = sub_items[j + 1]
                                                if sv:
                                                    st.markdown(f"**{sl}:** {sv}")
                            elif isinstance(value, list):
                                # List of dicts (e.g., Medical Conditions)
                                count = len(value)
                                with st.expander(
                                    f"\U0001F4CB {label} ({count} item{'s' if count != 1 else ''})",
                                    expanded=False,
                                ):
                                    for idx, item in enumerate(value):
                                        if isinstance(item, dict):
                                            if count > 1:
                                                item_name = item.get(list(item.keys())[0], f"Item {idx + 1}")
                                                st.markdown(f"**{idx + 1}. {item_name}**")
                                            # Show sub-fields in 2 columns
                                            sub = list(item.items())
                                            for j in range(0, len(sub), 2):
                                                sc = st.columns(2)
                                                with sc[0]:
                                                    sl, sv = sub[j]
                                                    if sv:
                                                        st.markdown(f"*{sl}:* {sv}")
                                                if j + 1 < len(sub):
                                                    with sc[1]:
                                                        sl, sv = sub[j + 1]
                                                        if sv:
                                                            st.markdown(f"*{sl}:* {sv}")
                                            if count > 1 and idx < count - 1:
                                                st.divider()
                                        else:
                                            st.markdown(f"- {item}")
            else:
                # Fallback when no application model available
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Name", display.get("name", "N/A"))
                with c2:
                    st.metric("Age", display.get("age", "N/A"))
                with c3:
                    st.metric("BMI", display.get("bmi", "N/A"))
                with c4:
                    st.metric("Occupation Class", display.get("occupation_class", "N/A"))

        # ================================================================
        # EXPANDER 2 \u2014 Full Assessment
        # ================================================================
        with st.expander("\U0001F4CA Full Assessment", expanded=True):
            # Decision source + reasoning
            st.caption(f"Source: {source_label} \u2014 Decision reached through structured agent debate")
            reasoning = active_results.get("decision_reasoning", "")
            if reasoning:
                st.info(reasoning)

            # Non-standard conditions
            _render_non_standard_conditions(agent_assessments)

            # Agent Assessments
            st.subheader("\U0001F916 Agent Assessments")
            if agent_assessments:
                for agent_name, assessment in agent_assessments.items():
                    if hasattr(assessment, "risk_tier"):
                        risk_tier = assessment.risk_tier
                        recommendation = assessment.recommendation
                        confidence_score = assessment.confidence_score
                        reasoning_summary = assessment.reasoning_summary
                        flags = assessment.flags
                        loading_range = assessment.loading_range
                        additional_evidence = assessment.additional_evidence_required
                    elif isinstance(assessment, dict):
                        risk_tier = assessment.get("risk_tier", "standard")
                        recommendation = assessment.get("recommendation", "standard")
                        confidence_score = assessment.get("confidence_score", 1.0)
                        reasoning_summary = assessment.get("reasoning_summary", "")
                        flags = assessment.get("flags", [])
                        loading_range = assessment.get("loading_range", [1.0, 1.0])
                        additional_evidence = assessment.get("additional_evidence_required", [])
                    else:
                        continue

                    is_compliance = "Compliance" in agent_name

                    if is_compliance:
                        # Compliance Agent — observer/informer only
                        expander_label = f"\U0001F6E1\uFE0F {agent_name} (Observer \u2014 Informational Only)"
                        severity_to_risk = {
                            "critical": "High", "high": "High",
                            "moderate": "Medium", "low": "Low",
                            "none": "None",
                        }
                        max_severity = "none"
                        if flags:
                            severity_order = {"critical": 4, "high": 3,
                                              "moderate": 2, "low": 1, "none": 0}
                            max_severity = max(
                                flags,
                                key=lambda f: severity_order.get(f.get("severity", "none"), 0),
                            ).get("severity", "none")
                        risk_level = severity_to_risk.get(max_severity, "None")

                        with st.expander(expander_label):
                            st.write(f"**Compliance Risk Level:** {risk_level}")
                            st.caption("(Compliance Agent does NOT influence the underwriting decision)")
                            if flags:
                                st.write(f"**Compliance Observations ({len(flags)}):**")
                                for flag in flags:
                                    st.write(f"- **[{flag.get('rule_id', '')}]** "
                                             f"({flag.get('severity', '')}): "
                                             f"{flag.get('description', '')}")
                            else:
                                st.success("No compliance gaps identified.")
                    else:
                        with st.expander(f"{agent_name} \u2014 {risk_tier.upper()}"):
                            st.write(f"**Recommendation:** {recommendation}")
                            st.write(f"**Confidence:** {confidence_score:.0%}")
                            st.write(f"**Reasoning:** {reasoning_summary}")
                            if flags:
                                st.write("**Matched Rules:**")
                                for flag in flags:
                                    st.markdown(
                                        f"<div style='font-size:1.05em; margin:4px 0;"
                                        f"padding:6px 10px;border-left:3px solid #f39c12;"
                                        f"background:#fffaf0;border-radius:4px;'>"
                                        f"<code style='font-size:0.95em;'>{flag.get('rule_id', '?')}</code> "
                                        f"<strong>({flag.get('severity', '')})</strong> &mdash; "
                                        f"{flag.get('description', '')}"
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )
                            if loading_range:
                                st.write(f"**Loading Range:** {loading_range[0]:.0%} to {loading_range[1]:.0%}")
                            if additional_evidence:
                                st.write("**Additional Evidence Required:**")
                                for item in additional_evidence:
                                    st.info(f"- {item}")

            # AI Summary
            if conversation is not None and getattr(conversation, "decision_summary", ""):
                summary_text = conversation.decision_summary
            else:
                llm_client = None
                try:
                    llm_client = LLMClient("./config.yaml")
                    if not llm_client.is_available():
                        llm_client = None
                except Exception:
                    llm_client = None
                applicant_context = ""
                if main_application is not None:
                    applicant_context = (
                        f"Age {main_application.age}, "
                        f"{main_application.occupation_class.value}, "
                        f"BMI {main_application.bmi}"
                    )
                summary_text = _generate_decision_summary(
                    agent_assessments,
                    final_decision,
                    llm_client,
                    applicant_name=display.get("name", ""),
                    applicant_context=applicant_context,
                )
                if conversation is not None:
                    conversation.decision_summary = summary_text
                    try:
                        if st.session_state.get("chat_store"):
                            st.session_state.chat_store.save(conversation)
                    except Exception:
                        pass
            st.info(f"**\U0001F916 AI Summary**\n\n{summary_text}")

            # Consensus
            consensus = active_results.get("consensus_reached", False)
            if consensus:
                st.success("\u2705 Consensus reached among all agents.")
            else:
                st.warning("\u26A0\uFE0F Disagreement detected. Debate was initiated.")

            # Flags + Evidence
            gflags = active_results.get("flags", [])
            if gflags:
                st.subheader("\U0001F6A8 Risk Flags")
                for flag in gflags:
                    st.warning(f"- {flag}")
            evidence_needed = active_results.get("additional_evidence_required", [])
            if evidence_needed:
                st.subheader("\U0001F4C4 Additional Evidence Required")
                for item in evidence_needed:
                    st.info(f"- {item}")

        # ================================================================
        # EXPANDER 3 \u2014 Debate Log
        # ================================================================
        with st.expander("\U0001F4AC Debate Log", expanded=False):
            col_main = st.container()
        
            with col_main:
                # Display conversation or empty state
                if st.session_state.chat_conversation:
                    conv = st.session_state.chat_conversation
        
                    # --- Decision Summary (always visible) ---
                    st.subheader(f"Conversation: {conv.applicant_name}")
                    caption_text = f"Created: {conv.created_at} \u2022 Rounds: {conv.debate_rounds} \u2022 Decision: {conv.final_decision}"
                    if getattr(conv, "evidence_re_evaluated", False):
                        caption_text += " \u2022 \u26A0\uFE0F Decision re-evaluated based on user-provided evidence"
                    elif getattr(conv, "user_evidence_applied", False):
                        caption_text += " \u2022 Evidence submitted (no tier change)"
                    st.caption(caption_text)
        
                    # --- Chat History (collapsible) ---
                    with st.expander("\U0001F4AC Full Debate Log & Chat History", expanded=True):
                        for msg in conv.messages:
                            msg_dict = {
                                "sender": msg.sender,
                                "content": _clean_chat_content(msg.content),
                                "timestamp": msg.timestamp,
                                "risk_tier_update": msg.risk_tier_update,
                            }
                            _render_chat_bubble(msg_dict, AGENT_STYLES)
        
                    st.divider()
        
                    # Re-evaluation toggle \u2014 default OFF
                    evidence_mode = st.checkbox(
                        "\U0001F50D Submit as evidence for re-evaluation",
                        value=False,
                        key="evidence_mode_toggle",
                        help="When checked, your input will be treated as new evidence "
                             "and agents will re-evaluate the application. The decision "
                             "will reflect the updated assessment.",
                    )
        
                    # Chat input
                    placeholder = (
                        "Describe the new evidence or information for re-evaluation..."
                        if evidence_mode
                        else "Ask an agent a question or provide additional evidence..."
                    )
                    user_input = st.chat_input(
                        placeholder,
                        key="debate_chat_input",
                    )
        
                    if user_input and st.session_state.chat_conversation_id:
                        # Process user input with agent responses
                        try:
                            from underwriting.agents.base_agent import AgentAssessment
                            from underwriting.debate.chat_models import ChatMessage
        
                            app = st.session_state.get("chat_application")
                            agents = st.session_state.get("chat_agents", {})
        
                            if app is None or not agents:
                                st.error("Application data not available. Please run a pipeline first.")
                            else:
                                # Load conversation from store
                                conversation = st.session_state.chat_store.load(
                                    st.session_state.chat_conversation_id,
                                )
        
                                # Add user message
                                user_msg = ChatMessage(
                                    sender="user",
                                    content=user_input,
                                    message_type="evidence" if evidence_mode else "question",
                                    is_user_input=True,
                                )
                                conversation.add_message(user_msg)
        
                                # Get agent responses
                                # Track previous risk_tier for debate detection
                                previous_tiers = {}
                                for agent_name, agent in agents.items():
                                    assessment_data = conversation.agent_assessments.get(
                                        agent_name, {},
                                    )
                                    previous_tiers[agent_name] = assessment_data.get(
                                        "risk_tier", "standard",
                                    )
        
                                for agent_name, agent in agents.items():
                                    assessment_data = conversation.agent_assessments.get(
                                        agent_name, {},
                                    )
                                    current_assessment = AgentAssessment(
                                        agent_name=agent_name,
                                        risk_tier=assessment_data.get("risk_tier", "standard"),
                                        flags=assessment_data.get("flags", []),
                                        recommendation=assessment_data.get(
                                            "recommendation", "standard",
                                        ),
                                        loading_range=assessment_data.get(
                                            "loading_range", [1.0, 1.0],
                                        ),
                                        confidence_score=assessment_data.get(
                                            "confidence_score", 1.0,
                                        ),
                                        reasoning_summary=assessment_data.get(
                                            "reasoning_summary", "",
                                        ),
                                        additional_evidence_required=assessment_data.get(
                                            "additional_evidence_required", [],
                                        ),
                                        apra_references=assessment_data.get(
                                            "apra_references", [],
                                        ),
                                    )
        
                                    response = agent.handle_user_message(
                                        application=app,
                                        current_assessment=current_assessment,
                                        user_message=user_input,
                                        conversation_history=conversation.messages[:-1],
                                    )
                                    conversation.add_message(response)
        
                                    # Always save modified assessment back to conversation
                                    conversation.agent_assessments[agent_name] = (
                                        current_assessment.model_dump()
                                    )
        
                                # Check if any agent's risk_tier changed
                                tier_changed = any(
                                    conversation.agent_assessments[agent_name].get("risk_tier")
                                    != previous_tiers[agent_name]
                                    for agent_name in agents
                                )
        
                                if tier_changed:
                                    # New evidence triggered re-evaluation. Debate round initiated.
                                    conversation.add_system_message(
                                        "New evidence triggered re-evaluation. Debate round initiated."
                                    )
        
                                    # Collect current assessments for rebuttal
                                    current_assessments = {}
                                    for agent_name, agent in agents.items():
                                        data = conversation.agent_assessments[agent_name]
                                        current_assessments[agent_name] = AgentAssessment(
                                            agent_name=agent_name,
                                            risk_tier=data.get("risk_tier", "standard"),
                                            flags=data.get("flags", []),
                                            recommendation=data.get("recommendation", "standard"),
                                            loading_range=data.get("loading_range", [1.0, 1.0]),
                                            confidence_score=data.get("confidence_score", 1.0),
                                            reasoning_summary=data.get("reasoning_summary", ""),
                                            additional_evidence_required=data.get(
                                                "additional_evidence_required", [],
                                            ),
                                            apra_references=data.get("apra_references", []),
                                        )
        
                                    # Run one-round debate: each agent generates rebuttal
                                    for agent_name, agent in agents.items():
                                        my_assessment = current_assessments[agent_name]
                                        other_assessments = [
                                            a for n, a in current_assessments.items()
                                            if n != agent_name
                                        ]
                                        rebuttal = agent.generate_rebuttal(
                                            application=app,
                                            my_assessment=my_assessment,
                                            other_assessments=other_assessments,
                                        )
                                        rebuttal_msg = ChatMessage(
                                            sender=agent_name,
                                            content=rebuttal.reasoning_summary,
                                            message_type="text",
                                            reasoning=rebuttal.reasoning_summary,
                                            risk_tier_update=rebuttal.risk_tier,
                                        )
                                        conversation.add_message(rebuttal_msg)
        
                                        # Update assessment with post-debate values
                                        conversation.agent_assessments[agent_name] = (
                                            rebuttal.model_dump()
                                        )
                                        current_assessments[agent_name] = rebuttal
        
                                    # Increment debate rounds
                                    conversation.debate_rounds += 1
        
                                    # Recalculate final_decision from updated assessments
                                    # Exclude Compliance Agent \u2014 it spots gaps, doesn't decide
                                    RISK_TIER_RANK = {"standard": 0, "loading": 1, "refer": 2, "decline": 3}
                                    DECISION_MAP = {0: "Standard Offer", 1: "Offer with Loading/Exclusion",
                                                    2: "Refer to Manual Underwriting", 3: "Refer to Manual Underwriting"}
                                    underwriting_data = {
                                        n: d for n, d in conversation.agent_assessments.items()
                                        if "Compliance" not in n
                                    }
                                    source = underwriting_data if underwriting_data else conversation.agent_assessments
                                    tiers = [data.get("risk_tier", "standard") for data in source.values()]
                                    ranks = [RISK_TIER_RANK.get(t, 0) for t in tiers]
                                    conversation.final_decision = DECISION_MAP.get(max(ranks), "Refer to Manual Underwriting")
        
                                # Mark evidence flags on conversation when evidence mode is on
                                if evidence_mode:
                                    conversation.user_evidence_applied = True
                                    if tier_changed:
                                        conversation.evidence_re_evaluated = True
                                        conversation.decision_summary = ""
                                        conversation.add_system_message(
                                            "Re-evaluation complete. Decision updated based on "
                                            "user-provided evidence."
                                        )
        
                                # Save updated conversation
                                st.session_state.chat_store.save(conversation)
                                st.session_state.chat_conversation = conversation
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error processing message: {e}")
        
                    # Delete button
                    st.divider()
                    if st.button("\U0001F5D1\uFE0F Delete This Conversation", type="secondary"):
                        st.session_state.chat_store.delete(st.session_state.chat_conversation_id)
                        st.session_state.chat_conversation_id = None
                        st.session_state.chat_conversation = None
                        st.rerun()
        
                else:
                    # Empty state
                    st.info(
                        "Select a conversation, submit a new application, "
                        "or run the Test Questionnaire pipeline."
                    )
        
                    # Show recent results if available
                    if "test_q_results" in st.session_state and st.session_state.test_q_results:
                        st.subheader("Recent Application")
                        res = st.session_state.test_q_results
                        if st.button("View in Chat"):
                            # If auto-converted by pipeline, just load it
                            if (st.session_state.get("chat_conversation_id") and
                                    st.session_state.get("chat_conversation")):
                                st.rerun()
                                return
                            # Otherwise convert results to conversation
                            from underwriting.debate.chat_models import Conversation, ChatMessage
        
                            conv_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
                            # Extract agent_assessments data for Conversation storage
                            agent_assessments = res.get("results", {}).get("agent_assessments", {})
                            agent_assessments_data = {}
                            for name, assessment in agent_assessments.items():
                                agent_assessments_data[name] = {
                                    "agent_name": name,
                                    "risk_tier": assessment.risk_tier,
                                    "flags": assessment.flags,
                                    "recommendation": assessment.recommendation,
                                    "loading_range": assessment.loading_range,
                                    "confidence_score": assessment.confidence_score,
                                    "reasoning_summary": assessment.reasoning_summary,
                                    "additional_evidence_required": assessment.additional_evidence_required,
                                    "apra_references": assessment.apra_references,
                                }
        
                            conv = Conversation(
                                application_id=conv_id,
                                applicant_name="Recent Application",
                                debate_rounds=len(res.get("results", {}).get("debate_log", [])),
                                final_decision=res.get("results", {}).get("final_decision", ""),
                                agent_assessments=agent_assessments_data,
                            )
        
                            for entry in res.get("results", {}).get("debate_log", []):
                                msg = ChatMessage(
                                    sender=entry.get("agent", "Unknown"),
                                    content=entry.get("reasoning", ""),
                                    risk_tier_update=entry.get("updated_tier"),
                                )
                                conv.add_message(msg)
        
                            store = st.session_state.chat_store
                            store.save(conv)
                            st.session_state.chat_conversation_id = conv_id
                            st.session_state.chat_conversation = conv
        
                            # Store agents for chat interaction (recreate from rules)
                            llm_client = None
                            if os.path.exists("./config.yaml"):
                                try:
                                    llm_client = LLMClient(config_path="./config.yaml")
                                except Exception:
                                    llm_client = None
                            st.session_state.chat_agents = {
                                "Medical Agent": MedicalAgent(
                                    rules_path="rules/death/medical_rules.json",
                                    llm_client=llm_client,
                                ),
                                "Financial Agent": FinancialAgent(
                                    rules_path="rules/death/financial_rules.json",
                                    llm_client=llm_client,
                                ),
                                "Compliance Agent": ComplianceAgent(
                                    rules_path="rules/death/compliance_rules.json",
                                    llm_client=llm_client,
                                ),
                            }
        
                            # Store a minimal application dict for handle_user_message
                            app_data = st.session_state.test_q_editable_data or {}
                            st.session_state.chat_application = _build_application_from_data(app_data)
        
                            st.rerun()


# ---------------------------------------------------------------------------
# Page 4: Test Questionnaire
# ---------------------------------------------------------------------------


def render_test_questionnaire() -> None:
    """Render the Test Questionnaire page."""
    st.title("\U0001F9EA Test Questionnaire")
    st.markdown("Select a pre-filled test questionnaire, edit values, and run the multi-agent underwriting pipeline.")

    if "test_q_selected_file" not in st.session_state:
        st.session_state.test_q_selected_file = None
    if "test_q_results" not in st.session_state:
        st.session_state.test_q_results = None
    if "test_q_editable_data" not in st.session_state:
        st.session_state.test_q_editable_data = None
    if "test_q_agent_selection" not in st.session_state:
        st.session_state.test_q_agent_selection = ["MedicalAgent", "FinancialAgent", "ComplianceAgent"]

    questionnaire_dir = Path("data/test_questionnaires")
    if questionnaire_dir.exists():
        yaml_files = sorted([f.name for f in questionnaire_dir.glob("*.yaml")])
    else:
        yaml_files = []

    if not yaml_files:
        st.warning("No questionnaire files found in data/test_questionnaires/. Create YAML files to get started.")
        return

    selected_file = st.selectbox("Select Questionnaire", yaml_files)

    if selected_file != st.session_state.test_q_selected_file:
        file_path = questionnaire_dir / selected_file
        try:
            from underwriting.test_questionnaire.models import QuestionnaireDefinition
            qd = QuestionnaireDefinition.from_yaml(str(file_path))
            st.session_state.test_q_selected_file = selected_file
            st.session_state.test_q_editable_data = qd.model_dump(mode='json')
            st.session_state.test_q_results = None
            st.rerun()
        except Exception as e:
            st.error(f"Error loading questionnaire: {e}")
            return

    if st.session_state.test_q_editable_data is None:
        return

    data = st.session_state.test_q_editable_data

    def safe_str(val, default=""):
        return val if val is not None else default

    def safe_int(val, default=0):
        return int(val) if val is not None else default

    def safe_float(val, default=0.0):
        return float(val) if val is not None else default

    def safe_bool(val, default=False):
        return bool(val) if val is not None else default

    def safe_index(options, val, default):
        v = val if val is not None else default
        try:
            return options.index(v)
        except ValueError:
            return 0

    st.subheader("Agent Selection")
    all_agents = ["MedicalAgent", "FinancialAgent", "ComplianceAgent"]
    selected_agents = st.multiselect(
        "Select agents to run",
        options=all_agents,
        default=st.session_state.test_q_agent_selection,
        key="test_q_agent_multiselect",
    )
    st.session_state.test_q_agent_selection = selected_agents

    st.subheader("Applicant Data")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", value=safe_str(data.get("full_name")))
        dob = st.date_input("Date of Birth", value=date.fromisoformat(safe_str(data.get("date_of_birth"), "1990-01-01")))
        gender = st.selectbox("Gender", options=["Male", "Female", "Non-binary"], index=safe_index(["Male", "Female", "Non-binary"], data.get("gender"), "Male"))
        residency = st.selectbox("Residency Status", options=["Australian Citizen", "Permanent Resident", "Temporary Visa"], index=safe_index(["Australian Citizen", "Permanent Resident", "Temporary Visa"], data.get("residency_status"), "Australian Citizen"))
        address = st.text_input("Contact Address", value=safe_str(data.get("contact_address")))

    with col2:
        occupation = st.text_input("Occupation", value=safe_str(data.get("occupation")))
        employer = st.text_input("Employer", value=safe_str(data.get("employer_name")))
        years_occ = st.number_input("Years in Occupation", value=safe_float(data.get("years_in_occupation"), 5.0), min_value=0.0, step=0.5)
        income = st.number_input("Annual Income ($)", value=safe_float(data.get("annual_income"), 100000.0), min_value=0.0, step=1000.0)
        contact = st.text_input("Contact Address (alt)", value=safe_str(data.get("contact_address")))

    st.subheader("Health")
    col3, col4 = st.columns(2)
    with col3:
        height = st.number_input("Height (cm)", value=safe_float(data.get("height_cm"), 170.0), min_value=0.0, step=0.1)
        weight = st.number_input("Weight (kg)", value=safe_float(data.get("weight_kg"), 70.0), min_value=0.0, step=0.1)
        smoker = st.selectbox("Smoker Status", options=["Never", "Former", "Current"], index=safe_index(["Never", "Former", "Current"], data.get("smoker_status"), "Never"))
    with col4:
        taking_meds = st.checkbox("Taking Medications", value=safe_bool(data.get("taking_medications")))
        consumes_alcohol = st.checkbox("Consumes Alcohol", value=safe_bool(data.get("consumes_alcohol")))

    st.subheader("Benefits")
    col5, col6 = st.columns(2)
    with col5:
        benefit_types = st.multiselect(
            "Benefit Types",
            options=["Death", "TPD", "Trauma/CI", "Income Protection"],
            default=data.get("benefit_types") or ["Death"],
        )
        sum_insured_death = st.number_input("Sum Insured Death ($)", value=safe_int(data.get("sum_insured_death"), 500000), min_value=0, step=10000)
    with col6:
        sum_insured_tpd = st.number_input("Sum Insured TPD ($)", value=safe_int(data.get("sum_insured_tpd"), 500000), min_value=0, step=10000)
        sum_insured_trauma = st.number_input("Sum Insured Trauma ($)", value=safe_int(data.get("sum_insured_trauma"), 0), min_value=0, step=10000)

    st.session_state.test_q_editable_data.update({
        "full_name": name,
        "date_of_birth": dob.isoformat(),
        "gender": gender,
        "residency_status": residency,
        "contact_address": address,
        "occupation": occupation,
        "employer_name": employer,
        "years_in_occupation": years_occ,
        "annual_income": income,
        "height_cm": height,
        "weight_kg": weight,
        "smoker_status": smoker,
        "taking_medications": taking_meds,
        "consumes_alcohol": consumes_alcohol,
        "benefit_types": benefit_types,
        "sum_insured_death": sum_insured_death,
        "sum_insured_tpd": sum_insured_tpd,
        "sum_insured_trauma": sum_insured_trauma if sum_insured_trauma > 0 else None,
    })

    st.subheader("Run Pipeline")
    if st.button("Run Underwriting Pipeline", type="primary"):
        with st.spinner("Running multi-agent pipeline..."):
            try:
                from underwriting.test_questionnaire.engine import TestQuestionnaireEngine
                from underwriting.test_questionnaire.models import QuestionnaireDefinition

                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    yaml.dump(st.session_state.test_q_editable_data, f, default_flow_style=False)
                    temp_path = f.name

                engine = TestQuestionnaireEngine(temp_path)
                results = engine.run(agent_names=selected_agents if selected_agents else None)

                app = engine.load().to_application()
                summary = engine.get_console_summary(app, results)

                st.session_state.test_q_results = {
                    "results": results,
                    "summary": summary,
                    "file_path": temp_path,
                }

                # Auto-create chat conversation from pipeline results
                try:
                    from underwriting.debate.chat_models import ChatMessage, Conversation
                    from underwriting.debate.persistence import ConversationStore

                    app_data = st.session_state.test_q_editable_data or {}
                    applicant_name = app_data.get("full_name", "Unknown Applicant")

                    conv_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

                    # Ensure chat_store exists
                    if st.session_state.get("chat_store") is None:
                        st.session_state.chat_store = ConversationStore()

                    store = st.session_state.chat_store

                    debate_log = results.get("debate_log", [])
                    agent_assessments = results.get("agent_assessments", {})

                    # Extract agent_assessments data for Conversation storage
                    agent_assessments_data = {}
                    for name, assessment in agent_assessments.items():
                        agent_assessments_data[name] = {
                            "agent_name": name,
                            "risk_tier": assessment.risk_tier,
                            "flags": assessment.flags,
                            "recommendation": assessment.recommendation,
                            "loading_range": assessment.loading_range,
                            "confidence_score": assessment.confidence_score,
                            "reasoning_summary": assessment.reasoning_summary,
                            "additional_evidence_required": assessment.additional_evidence_required,
                            "apra_references": assessment.apra_references,
                        }

                    conversation = Conversation(
                        application_id=conv_id,
                        applicant_name=applicant_name,
                        debate_rounds=len(debate_log),
                        final_decision=results.get("final_decision", ""),
                        agents_participating=list(agent_assessments.keys()),
                        agent_assessments=agent_assessments_data,
                        applicant_data=app.model_dump(mode="json"),
                    )

                    # Add debate log entries as chat messages
                    for entry in debate_log:
                        msg = ChatMessage(
                            sender=entry.get("agent", "Unknown"),
                            content=entry.get("reasoning", ""),
                            message_type="text",
                            risk_tier_update=entry.get("updated_tier"),
                            reasoning=entry.get("reasoning", ""),
                        )
                        conversation.add_message(msg)

                    # Add initial system message
                    conversation.add_system_message(
                        f"Pipeline completed for {applicant_name}. "
                        f"Final decision: {results.get('final_decision', 'Unknown')}. "
                        f"Debate rounds: {conversation.debate_rounds}."
                    )

                    # Save conversation
                    store.save(conversation)

                    # Update session state
                    st.session_state.chat_conversation_id = conv_id
                    st.session_state.chat_conversation = conversation

                    # Pre-generate decision summary so it's ready on Results page
                    try:
                        llm_client = LLMClient(config_path="./config.yaml") if os.path.exists("./config.yaml") else None
                        if llm_client is not None and not llm_client.is_available():
                            llm_client = None
                    except Exception:
                        llm_client = None

                    # Build applicant context
                    applicant_context = ""
                    try:
                        applicant_context = (
                            f"Age {app.age}, "
                            f"{app.occupation_class.value}, "
                            f"BMI {app.bmi}"
                        )
                    except Exception:
                        pass

                    summary_text = _generate_decision_summary(
                        agent_assessments_data,
                        results.get("final_decision", ""),
                        llm_client,
                        applicant_name=applicant_name,
                        applicant_context=applicant_context,
                    )
                    conversation.decision_summary = summary_text
                    store.save(conversation)
                    st.session_state.chat_conversation = conversation

                    # Store agents and application for chat interaction
                    st.session_state.chat_application = app
                    st.session_state.chat_agents = {
                        "Medical Agent": MedicalAgent(
                            rules_path="rules/death/medical_rules.json",
                            llm_client=LLMClient(config_path="./config.yaml")
                            if os.path.exists("./config.yaml") else None,
                        ),
                        "Financial Agent": FinancialAgent(
                            rules_path="rules/death/financial_rules.json",
                            llm_client=LLMClient(config_path="./config.yaml")
                            if os.path.exists("./config.yaml") else None,
                        ),
                        "Compliance Agent": ComplianceAgent(
                            rules_path="rules/death/compliance_rules.json",
                            llm_client=LLMClient(config_path="./config.yaml")
                            if os.path.exists("./config.yaml") else None,
                        ),
                    }

                except Exception as e:
                    import logging
                    logging.warning(f"Failed to save conversation: {e}")

                # Auto-navigate to Results & Debate on next page load
                st.session_state._pending_nav = "Results & Debate"
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                import traceback
                st.code(traceback.format_exc())

    if st.session_state.test_q_results is not None:
        res = st.session_state.test_q_results
        st.success("Pipeline complete!")

        st.subheader("Pipeline Summary")
        st.code(res["summary"], language="text")

        st.subheader("Underwriting Agent Assessments")
        for agent_name, assessment in res["results"].get("agent_assessments", {}).items():
            # Skip Compliance Agent — displayed separately as observer
            if "Compliance" in agent_name:
                continue
            with st.expander(f"{agent_name}"):
                st.write(f"**Risk Tier:** {assessment.risk_tier}")
                st.write(f"**Recommendation:** {assessment.recommendation}")
                st.write(f"**Confidence:** {assessment.confidence_score:.0%}")
                if assessment.flags:
                    st.write("**Flags:**")
                    for flag in assessment.flags:
                        st.write(f"- [{flag.get('severity', '?')}] {flag.get('rule_id', '?')}: {flag.get('description', '?')}")

        # Compliance Agent — displayed separately as observer/informer
        for agent_name, assessment in res["results"].get("agent_assessments", {}).items():
            if "Compliance" not in agent_name:
                continue
            with st.expander(f"\U0001F6E1\uFE0F {agent_name} (Observer — Informational Only)"):
                # Map severity to compliance risk level
                severity_to_risk = {"critical": "High", "high": "High",
                                    "moderate": "Medium", "low": "Low",
                                    "none": "None"}
                max_severity = "none"
                if assessment.flags:
                    severity_order = {"critical": 4, "high": 3, "moderate": 2, "low": 1, "none": 0}
                    max_severity = max(assessment.flags, key=lambda f: severity_order.get(f.get("severity", "none"), 0)).get("severity", "none")
                risk_level = severity_to_risk.get(max_severity, "None")

                st.write(f"**Compliance Risk Level:** {risk_level}")
                st.caption("(Compliance Agent does NOT influence the underwriting decision)")
                if assessment.flags:
                    st.write(f"**Compliance Observations ({len(assessment.flags)}):**")
                    for flag in assessment.flags:
                        st.write(f"- [{flag.get('severity', '?')}] **{flag.get('rule_id', '?')}**: {flag.get('description', '?')}")
                        if flag.get("regulatory_reference"):
                            st.caption(f"  Ref: {flag['regulatory_reference']}")
                else:
                    st.success("No compliance gaps identified.")

        debate_log = res["results"].get("debate_log", [])
        if debate_log:
            st.subheader("Debate Log")
            for entry in debate_log:
                st.write(f"Round {entry.get('round', '?')}: {entry.get('agent', '?')}")
        else:
            st.info("No debate needed — all agents reached consensus.")

        st.caption(f"Results saved to: {res['file_path']}")
    else:
        st.info("Select a questionnaire and click 'Run Underwriting Pipeline' to see results.")


# ---------------------------------------------------------------------------
# Page 5: Compliance Framework
# ---------------------------------------------------------------------------


def render_compliance_framework() -> None:
    """Render the standalone Compliance Framework reference page."""
    st.title("\U0001F6E1\uFE0F Compliance Framework")

    from underwriting.debate.compliance_summary import (
        generate_compliance_framework_summary,
        get_compliance_statistics,
    )

    stats = get_compliance_statistics()

    # Key metrics row
    if stats.get("total_rules", 0) > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rules Monitored", stats["total_rules"])
        with col2:
            st.metric("AI Ethics Principles", f"{stats['principles_covered']}/{stats['principles_total']}")
        with col3:
            st.metric("Regulatory Frameworks", stats["frameworks_covered"])
        with col4:
            st.metric("Informational", stats["informational_count"])

    compliance_md = generate_compliance_framework_summary()
    st.markdown(compliance_md)


# ---------------------------------------------------------------------------
# Page 6: Underwriting Rules Reference
# ---------------------------------------------------------------------------


def _load_rules_json(path: str) -> list:
    """Load rules from a JSON file."""
    import json as _json
    try:
        with open(path) as f:
            data = _json.load(f)
        return data.get("rules", [])
    except Exception:
        return []


def render_rules_reference() -> None:
    """Render the Underwriting Rules Reference page showing all rules."""
    st.title("\U0001F4DC Underwriting Rules Reference")
    st.markdown(
        "These are the deterministic rules evaluated by the Medical and Financial "
        "agents. Each rule's condition is checked against the applicant's data. "
        "Matched rules determine the risk tier, flags, and recommendations."
    )

    tab1, tab2 = st.tabs(["Medical Rules", "Financial Rules"])

    with tab1:
        st.subheader("Medical Underwriting Rules")
        medical_rules = _load_rules_json("rules/death/medical_rules.json")
        if medical_rules:
            st.caption(f"{len(medical_rules)} rules loaded")
            for rule in medical_rules:
                sev = rule.get("severity", "unknown")
                sev_color = {"critical": "#dc3545", "high": "#e8590c",
                              "moderate": "#fd7e14", "low": "#0c8599",
                              "none": "#28a745"}.get(sev, "#6c757d")
                with st.expander(
                    f"[{sev.upper()}] {rule['rule_id']}: {rule.get('description', '')[:80]}",
                    expanded=False,
                ):
                    st.markdown(f"**Rule ID:** `{rule['rule_id']}`")
                    st.markdown(f"**Category:** {rule.get('category', 'N/A')}")
                    st.markdown(f"**Severity:** :{'red' if sev in ('critical','high') else 'orange' if sev == 'moderate' else 'green'}[{sev}]{' ' if sev in ('critical','high') else ''}")
                    st.markdown(f"**Condition:** `{rule.get('condition', 'N/A')}`")
                    st.markdown(f"**Recommendation:** {rule.get('recommendation', 'N/A')}")
                    if rule.get("additional_evidence"):
                        st.markdown(f"**Evidence Required:** {', '.join(rule['additional_evidence'])}")
                    if rule.get("apra_ref"):
                        st.caption(f"APRA Ref: {rule['apra_ref']}")
        else:
            st.warning("No medical rules loaded.")

    with tab2:
        st.subheader("Financial Underwriting Rules")
        financial_rules = _load_rules_json("rules/death/financial_rules.json")
        if financial_rules:
            st.caption(f"{len(financial_rules)} rules loaded")
            for rule in financial_rules:
                sev = rule.get("severity", "unknown")
                with st.expander(
                    f"[{sev.upper()}] {rule['rule_id']}: {rule.get('description', '')[:80]}",
                    expanded=False,
                ):
                    st.markdown(f"**Rule ID:** `{rule['rule_id']}`")
                    st.markdown(f"**Category:** {rule.get('category', 'N/A')}")
                    st.markdown(f"**Severity:** {sev}")
                    st.markdown(f"**Condition:** `{rule.get('condition', 'N/A')}`")
                    st.markdown(f"**Recommendation:** {rule.get('recommendation', 'N/A')}")
                    if rule.get("additional_evidence"):
                        st.markdown(f"**Evidence Required:** {', '.join(rule['additional_evidence'])}")
                    if rule.get("apra_ref"):
                        st.caption(f"APRA Ref: {rule['apra_ref']}")
        else:
            st.warning("No financial rules loaded.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the Streamlit underwriting application.

    Displays navigation sidebar and renders the selected page.
    """
    load_config_yaml()  # noqa: F841

    # Sidebar navigation
    # Deferred navigation: apply pending page change before widget renders
    if "_pending_nav" in st.session_state:
        st.session_state._nav_page = st.session_state.pop("_pending_nav")
    page = st.sidebar.radio(
        "Navigation",
        options=[
            "Questionnaire", "Results & Debate",
            "Test Questionnaire", "Compliance Framework", "Rules Reference",
        ],
        index=0,
        key="_nav_page",
        format_func=lambda x: {
            "Questionnaire": "\U0001F4DD Questionnaire",
            "Results & Debate": "\U0001F4C1 Results & Debate",
            "Test Questionnaire": "\U0001F9EA Test Questionnaire",
            "Compliance Framework": "\U0001F6E1\uFE0F Compliance Framework",
            "Rules Reference": "\U0001F4DC Rules Reference",
        }.get(x, x),
    )

    if page == "Questionnaire":
        render_questionnaire()
    elif page == "Results & Debate":
        render_results_and_debate()
    elif page == "Test Questionnaire":
        render_test_questionnaire()
    elif page == "Compliance Framework":
        render_compliance_framework()
    elif page == "Rules Reference":
        render_rules_reference()


if __name__ == "__main__":
    main()
