"""Tests for application schema models."""
from datetime import date

import pytest
from pydantic import ValidationError

from underwriting.application.schema import (
    Application,
    BenefitType,
    FamilyHistoryCondition,
    HazardousPursuit,
    MedicalCondition,
    OccupationClass,
    SmokerStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STANDARD_APP = {
    "full_name": "John Doe",
    "date_of_birth": date(1990, 5, 15),
    "gender": "Male",
    "residency_status": "Australian Citizen",
    "contact_address": "123 Main St, Sydney NSW 2000",
    "benefit_types": [BenefitType.DEATH, BenefitType.TPD],
    "sum_insured_death": 500_000,
    "sum_insured_tpd": 500_000,
    "occupation": "Software Engineer",
    "employer_name": "Tech Corp",
    "years_in_occupation": 5.0,
    "annual_income": 95_000,
    "height_cm": 180.0,
    "weight_kg": 80.0,
    "smoker_status": SmokerStatus.NEVER,
    "taking_medications": False,
    "consumes_alcohol": False,
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestSmokerStatus:
    def test_enum_values(self):
        assert SmokerStatus.NEVER.value == "Never"
        assert SmokerStatus.FORMER.value == "Former"
        assert SmokerStatus.CURRENT.value == "Current"

    def test_from_string(self):
        assert SmokerStatus("Never") == SmokerStatus.NEVER
        assert SmokerStatus("Former") == SmokerStatus.FORMER
        assert SmokerStatus("Current") == SmokerStatus.CURRENT


class TestOccupationClass:
    def test_enum_values(self):
        assert OccupationClass.PROFESSIONAL.value == "Professional"
        assert OccupationClass.WHITE_COLLAR.value == "White Collar"
        assert OccupationClass.LIGHT_MANUAL.value == "Light Manual"
        assert OccupationClass.BLUE_COLLAR.value == "Blue Collar"
        assert OccupationClass.HEAVY_MANUAL.value == "Heavy Manual"
        assert OccupationClass.HAZARDOUS.value == "Hazardous"


class TestBenefitType:
    def test_enum_values(self):
        assert BenefitType.DEATH.value == "Death"
        assert BenefitType.TPD.value == "TPD"
        assert BenefitType.TRAUMA.value == "Trauma/CI"
        assert BenefitType.IP.value == "Income Protection"


# ---------------------------------------------------------------------------
# MedicalCondition
# ---------------------------------------------------------------------------

class TestMedicalCondition:
    def test_required_fields(self):
        cond = MedicalCondition(
            condition_name="Hypertension",
            diagnosis_date=date(2020, 1, 1),
            treating_doctor_name="Dr Smith",
            treating_doctor_contact="0400 000 000",
        )
        assert cond.condition_name == "Hypertension"
        assert cond.diagnostic_tests is None

    def test_optional_fields(self):
        cond = MedicalCondition(
            condition_name="Diabetes",
            diagnosis_date=date(2019, 6, 15),
            treating_doctor_name="Dr Jones",
            treating_doctor_contact="0400 111 111",
            diagnostic_tests="HbA1c test",
            treatment_start_date=date(2019, 7, 1),
            treatment_description="Metformin 500mg twice daily",
            symptoms="Increased thirst",
            symptom_frequency="Daily",
            last_symptom_date=date(2024, 12, 1),
            hospitalisations="None",
            time_off_work="None",
            lifestyle_affected=True,
        )
        assert cond.diagnostic_tests == "HbA1c test"
        assert cond.treatment_description == "Metformin 500mg twice daily"
        assert cond.symptoms == "Increased thirst"
        assert cond.lifestyle_affected is True


# ---------------------------------------------------------------------------
# FamilyHistoryCondition
# ---------------------------------------------------------------------------

class TestFamilyHistoryCondition:
    def test_creation(self):
        fh = FamilyHistoryCondition(
            relationship="father",
            condition="heart disease",
            age_at_diagnosis=55,
        )
        assert fh.relationship == "father"
        assert fh.condition == "heart disease"
        assert fh.age_at_diagnosis == 55


# ---------------------------------------------------------------------------
# HazardousPursuit
# ---------------------------------------------------------------------------

class TestHazardousPursuit:
    def test_amateur(self):
        hp = HazardousPursuit(activity="Scuba diving", frequency="Monthly", level="amateur")
        assert hp.level == "amateur"

    def test_professional(self):
        hp = HazardousPursuit(activity="Skydiving", frequency="Weekly", level="professional")
        assert hp.level == "professional"

    def test_invalid_level(self):
        with pytest.raises(ValidationError):
            HazardousPursuit(activity="Hiking", frequency="Weekly", level="extreme")


# ---------------------------------------------------------------------------
# Application — construction & validation
# ---------------------------------------------------------------------------

class TestApplicationConstruction:
    def test_standard_app(self):
        app = Application(**STANDARD_APP)
        assert app.full_name == "John Doe"
        assert app.age > 30  # born 1990
        assert app.bmi == 24.7  # 80 / (1.8^2)
        assert app.occupation_class == OccupationClass.PROFESSIONAL

    def test_required_fields(self):
        # Remove full_name to trigger ValidationError
        with pytest.raises(ValidationError) as exc:
            Application(**{k: v for k, v in STANDARD_APP.items() if k != "full_name"})
        assert "full_name" in str(exc.value)

    def test_gender_validation(self):
        with pytest.raises(ValidationError):
            Application(**{**STANDARD_APP, "gender": "Other"})  # type: ignore

    def test_residency_validation(self):
        with pytest.raises(ValidationError):
            Application(**{**STANDARD_APP, "residency_status": "Visitor"})  # type: ignore

    def test_smoker_status_validation(self):
        with pytest.raises(ValidationError):
            Application(**{**STANDARD_APP, "smoker_status": "Unknown"})  # type: ignore

    def test_default_values(self):
        app = Application(**STANDARD_APP)
        assert app.has_other_policies is False
        assert app.previous_declination is False
        assert app.has_family_history is False
        assert app.has_hazardous_pursuits is False
        assert app.recreational_drug_use is False
        assert app.alcohol_drug_treatment is False
        assert app.has_high_risk_travel is False
        assert app.previous_bankruptcy is False
        assert app.criminal_convictions is False
        assert app.duty_of_disclosure_acknowledged is False
        assert app.medical_conditions == []
        assert app.family_history == []
        assert app.hazardous_pursuits == []


# ---------------------------------------------------------------------------
# Computed fields
# ---------------------------------------------------------------------------

class TestComputedFields:
    def test_age(self):
        app = Application(**{**STANDARD_APP, "date_of_birth": date(1985, 3, 10)})
        # Age depends on today's date — just verify it's an int and reasonable
        assert isinstance(app.age, int)
        assert 35 <= app.age <= 120  # sanity range

    def test_age_birthday_not_yet(self):
        # Born Dec 31, 2000 — age should be 25 on Jan 1, 2026 (not 26)
        app = Application(**{**STANDARD_APP, "date_of_birth": date(2000, 12, 31)})
        today = date.today()
        expected = today.year - 2000 - 1  # birthday not yet
        assert app.age == expected

    def test_bmi(self):
        app = Application(**{**STANDARD_APP, "height_cm": 180, "weight_kg": 80})
        assert app.bmi == 24.7

    def test_bmi_overweight(self):
        app = Application(**{**STANDARD_APP, "height_cm": 170, "weight_kg": 95})
        assert app.bmi == 32.9  # 95 / (1.7^2)

    def test_bmi_underweight(self):
        app = Application(**{**STANDARD_APP, "height_cm": 180, "weight_kg": 55})
        assert app.bmi == 17.0  # 55 / (1.8^2)

    def test_occupation_class_professional(self):
        for occ in [
            "Doctor", "Lawyer", "Accountant", "Engineer",
            "Architect", "Consultant", "Analyst",
        ]:
            app = Application(**{**STANDARD_APP, "occupation": occ})
            assert app.occupation_class == OccupationClass.PROFESSIONAL, f"Failed for {occ}"

    def test_occupation_class_white_collar(self):
        for occ in ["Manager", "Administrator", "Sales Rep", "Retail Worker", "Clerk"]:
            app = Application(**{**STANDARD_APP, "occupation": occ})
            assert app.occupation_class == OccupationClass.WHITE_COLLAR, f"Failed for {occ}"

    def test_occupation_class_light_manual(self):
        for occ in ["Warehouse Worker", "Driver", "Machine Operator", "Cleaner"]:
            app = Application(**{**STANDARD_APP, "occupation": occ})
            assert app.occupation_class == OccupationClass.LIGHT_MANUAL, f"Failed for {occ}"

    def test_occupation_class_blue_collar(self):
        for occ in ["Mechanic", "Electrician", "Plumber", "Welder"]:
            app = Application(**{**STANDARD_APP, "occupation": occ})
            assert app.occupation_class == OccupationClass.BLUE_COLLAR, f"Failed for {occ}"

    def test_occupation_class_heavy_manual(self):
        for occ in ["Construction Worker", "Rigger", "Scaffolder", "Roofer"]:
            app = Application(**{**STANDARD_APP, "occupation": occ})
            assert app.occupation_class == OccupationClass.HEAVY_MANUAL, f"Failed for {occ}"

    def test_occupation_class_hazardous(self):
        for occ in ["Miner", "Offshore Worker", "Explosives Handler", "Underwater Diver"]:
            app = Application(**{**STANDARD_APP, "occupation": occ})
            assert app.occupation_class == OccupationClass.HAZARDOUS, f"Failed for {occ}"

    def test_occupation_class_default(self):
        app = Application(**{**STANDARD_APP, "occupation": "Unknown Job Title XYZ"})
        assert app.occupation_class == OccupationClass.WHITE_COLLAR


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------

class TestHelperMethods:
    def test_has_condition_cardiovascular(self):
        app = Application(**{
            **STANDARD_APP,
            "medical_conditions": [
                MedicalCondition(
                    condition_name="Hypertension",
                    diagnosis_date=date(2020, 1, 1),
                    treating_doctor_name="Dr Smith",
                    treating_doctor_contact="0400 000 000",
                ),
            ],
        })
        assert app.has_condition("cardiovascular_disease") is True

    def test_has_condition_no_match(self):
        app = Application(**{
            **STANDARD_APP,
            "medical_conditions": [
                MedicalCondition(
                    condition_name="Asthma",
                    diagnosis_date=date(2018, 3, 1),
                    treating_doctor_name="Dr Brown",
                    treating_doctor_contact="0400 222 222",
                ),
            ],
        })
        assert app.has_condition("cardiovascular_disease") is False

    def test_has_condition_empty_list(self):
        app = Application(**STANDARD_APP)
        assert app.has_condition("cardiovascular_disease") is False

    def test_has_condition_diabetes_type_1(self):
        app = Application(**{
            **STANDARD_APP,
            "medical_conditions": [
                MedicalCondition(
                    condition_name="Type 1 Diabetes",
                    diagnosis_date=date(2015, 6, 1),
                    treating_doctor_name="Dr Green",
                    treating_doctor_contact="0400 333 333",
                ),
            ],
        })
        assert app.has_condition("diabetes_type_1") is True
        assert app.has_condition("diabetes_type_2") is False

    def test_has_condition_diabetes_type_2(self):
        app = Application(**{
            **STANDARD_APP,
            "medical_conditions": [
                MedicalCondition(
                    condition_name="Type 2 Diabetes",
                    diagnosis_date=date(2018, 6, 1),
                    treating_doctor_name="Dr White",
                    treating_doctor_contact="0400 444 444",
                ),
            ],
        })
        assert app.has_condition("diabetes_type_2") is True
        assert app.has_condition("diabetes_type_1") is False

    def test_has_condition_mental_health(self):
        app = Application(**{
            **STANDARD_APP,
            "medical_conditions": [
                MedicalCondition(
                    condition_name="Anxiety Disorder",
                    diagnosis_date=date(2021, 1, 1),
                    treating_doctor_name="Dr Blue",
                    treating_doctor_contact="0400 555 555",
                ),
            ],
        })
        assert app.has_condition("mental_health") is True

    def test_has_condition_severe_mental_illness(self):
        app = Application(**{
            **STANDARD_APP,
            "medical_conditions": [
                MedicalCondition(
                    condition_name="Schizophrenia",
                    diagnosis_date=date(2019, 1, 1),
                    treating_doctor_name="Dr Red",
                    treating_doctor_contact="0400 666 666",
                ),
            ],
        })
        assert app.has_condition("severe_mental_illness") is True
        assert app.has_condition("mental_health") is True  # schizophrenia is also in mental_health

    def test_has_family_history_cardiovascular(self):
        app = Application(**{
            **STANDARD_APP,
            "has_family_history": True,
            "family_history": [
                FamilyHistoryCondition(
                    relationship="father",
                    condition="heart attack",
                    age_at_diagnosis=60,
                ),
            ],
        })
        assert app.check_family_history("cardiovascular_disease") is True

    def test_has_family_history_cancer(self):
        app = Application(**{
            **STANDARD_APP,
            "has_family_history": True,
            "family_history": [
                FamilyHistoryCondition(
                    relationship="mother",
                    condition="breast cancer",
                    age_at_diagnosis=50,
                ),
            ],
        })
        assert app.check_family_history("cancer") is True

    def test_has_family_history_no_match(self):
        app = Application(**{
            **STANDARD_APP,
            "has_family_history": True,
            "family_history": [
                FamilyHistoryCondition(
                    relationship="uncle",
                    condition="diabetes",
                    age_at_diagnosis=65,
                ),
            ],
        })
        assert app.check_family_history("cancer") is False

    def test_has_family_history_empty(self):
        app = Application(**STANDARD_APP)
        assert app.check_family_history("cancer") is False

    def test_family_age_at_diagnosis(self):
        app = Application(**{
            **STANDARD_APP,
            "has_family_history": True,
            "family_history": [
                FamilyHistoryCondition(
                    relationship="father",
                    condition="heart disease",
                    age_at_diagnosis=60,
                ),
                FamilyHistoryCondition(
                    relationship="mother",
                    condition="stroke",
                    age_at_diagnosis=55,
                ),
                FamilyHistoryCondition(
                    relationship="brother",
                    condition="diabetes",
                    age_at_diagnosis=40,
                ),
            ],
        })
        assert app.family_age_at_diagnosis == 40

    def test_family_age_at_diagnosis_empty(self):
        app = Application(**STANDARD_APP)
        assert app.family_age_at_diagnosis is None

    def test_has_multiple_policies_true(self):
        app = Application(**{**STANDARD_APP, "has_other_policies": True})
        assert app.has_multiple_policies is True

    def test_has_multiple_policies_false(self):
        app = Application(**STANDARD_APP)
        assert app.has_multiple_policies is False

    def test_total_sum_insured(self):
        app = Application(**{
            **STANDARD_APP,
            "sum_insured_death": 500_000,
            "sum_insured_tpd": 300_000,
            "sum_insured_trauma": 100_000,
        })
        assert app.total_sum_insured_across_all_policies == 900_000

    def test_total_sum_insured_partial(self):
        app = Application(**{
            **STANDARD_APP,
            "sum_insured_death": 500_000,
            "sum_insured_tpd": None,
        })
        assert app.total_sum_insured_across_all_policies == 500_000

    def test_total_sum_insured_none(self):
        app = Application(**{
            **{k: v for k, v in STANDARD_APP.items() if k not in ("sum_insured_death", "sum_insured_tpd", "sum_insured_trauma")},
            "sum_insured_death": None,
            "sum_insured_tpd": None,
            "sum_insured_trauma": None,
        })
        assert app.total_sum_insured_across_all_policies == 0

    def test_has_hazardous_pursuit_true(self):
        app = Application(**{
            **STANDARD_APP,
            "has_hazardous_pursuits": True,
            "hazardous_pursuits": [
                HazardousPursuit(activity="Scuba diving", frequency="Monthly", level="amateur"),
            ],
        })
        assert app.has_hazardous_pursuit("scuba") is True

    def test_has_hazardous_pursuit_case_insensitive(self):
        app = Application(**{
            **STANDARD_APP,
            "has_hazardous_pursuits": True,
            "hazardous_pursuits": [
                HazardousPursuit(activity="Skydiving", frequency="Weekly", level="professional"),
            ],
        })
        assert app.has_hazardous_pursuit("skydive") is True

    def test_has_hazardous_pursuit_false(self):
        app = Application(**{
            **STANDARD_APP,
            "has_hazardous_pursuits": True,
            "hazardous_pursuits": [
                HazardousPursuit(activity="Scuba diving", frequency="Monthly", level="amateur"),
            ],
        })
        assert app.has_hazardous_pursuit("hiking") is False

    def test_has_hazardous_pursuit_empty_list(self):
        app = Application(**STANDARD_APP)
        assert app.has_hazardous_pursuit("hiking") is False


# ---------------------------------------------------------------------------
# Integration / round-trip
# ---------------------------------------------------------------------------

class TestApplicationIntegration:
    def test_round_trip_serialization(self):
        app = Application(**STANDARD_APP)
        data = app.model_dump()
        restored = Application(**data)
        assert restored.full_name == app.full_name
        assert restored.bmi == app.bmi
        assert restored.age == app.age

    def test_benefit_types_enum(self):
        app = Application(**STANDARD_APP)
        assert BenefitType.DEATH in app.benefit_types
        assert app.benefit_types[0].value == "Death"

    def test_all_sections_present(self):
        """Verify all Sections A-H are represented in the model."""
        fields = Application.model_fields
        # Section A
        assert "full_name" in fields
        assert "date_of_birth" in fields
        assert "gender" in fields
        assert "residency_status" in fields
        assert "contact_address" in fields
        # Section B
        assert "benefit_types" in fields
        assert "sum_insured_death" in fields
        assert "has_other_policies" in fields
        # Section C
        assert "occupation" in fields
        assert "employer_name" in fields
        assert "annual_income" in fields
        # Section D
        assert "height_cm" in fields
        assert "weight_kg" in fields
        assert "smoker_status" in fields
        assert "medical_conditions" in fields
        # Section F
        assert "has_family_history" in fields
        assert "family_history" in fields
        # Section G
        assert "has_hazardous_pursuits" in fields
        assert "hazardous_pursuits" in fields
        # Section H
        assert "total_net_worth" in fields
        assert "previous_bankruptcy" in fields
        assert "criminal_convictions" in fields
        # Compliance
        assert "duty_of_disclosure_acknowledged" in fields
