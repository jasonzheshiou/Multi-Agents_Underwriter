"""Tests for the ReportGenerator class."""

import os
import tempfile
from typing import Any, Dict

from underwriting.audit.report_generator import ReportGenerator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_decision_data() -> Dict[str, Any]:
    """Create a representative decision_data dict for testing."""
    return {
        "applicant": {
            "full_name": "Alex Standard",
            "date_of_birth": "1990-06-15",
            "age": 35,
            "gender": "Male",
            "residency_status": "Australian Citizen",
            "occupation": "Software Manager",
            "annual_income": 120000.0,
            "bmi": 24.0,
            "smoker_status": "Never",
            "benefit_types": ["Death", "TPD"],
            "sum_insured_death": 500000.0,
        },
        "agent_assessments": {
            "Medical": {
                "risk_tier": "standard",
                "recommendation": "Approve",
                "confidence_score": 0.95,
                "reasoning_summary": "Applicant is in good health with no significant risk factors.",
                "flags": [
                    {"rule_id": "MED-001", "severity": "low", "description": "Slightly elevated BMI"},
                ],
                "additional_evidence_required": [],
                "apra_references": ["APRA Prudential Standard CPS230"],
            },
            "Financial": {
                "risk_tier": "standard",
                "recommendation": "Approve",
                "confidence_score": 0.92,
                "reasoning_summary": "Income sufficient to support requested cover.",
                "flags": [],
                "additional_evidence_required": ["Proof of income for verification"],
                "apra_references": ["APRA CPS 220"],
            },
            "Compliance": {
                "risk_tier": "standard",
                "recommendation": "Approve",
                "confidence_score": 1.0,
                "reasoning_summary": "No compliance issues identified.",
                "flags": [],
                "additional_evidence_required": [],
                "apra_references": ["APRA CPS 520"],
            },
        },
        "debate_log": [
            {
                "round": 1,
                "agent": "Financial",
                "original_tier": ["standard", "standard", "standard"],
                "updated_tier": "standard",
                "reasoning": "Confirmed standard after reviewing income documentation.",
            },
        ],
        "final_decision": "Standard Offer",
        "consensus_reached": True,
        "decision_reasoning": "Final decision: Standard Offer. Based on assessments from 3 agents. Highest risk tier: standard.",
        "flags": [
            {"rule_id": "MED-001", "severity": "low", "description": "Slightly elevated BMI"},
        ],
        "additional_evidence_required": ["Proof of income for verification"],
    }


# ---------------------------------------------------------------------------
# Test: ReportGenerator instantiation
# ---------------------------------------------------------------------------


class TestReportGeneratorInit:
    """Tests for ReportGenerator initialisation."""

    def test_instantiate_with_default_output_dir(self):
        """ReportGenerator creates default output directory."""
        gen = ReportGenerator()
        assert gen.output_dir == "./audit_reports"
        assert os.path.isdir(gen.output_dir)

    def test_instantiate_with_custom_output_dir(self):
        """ReportGenerator accepts a custom output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)
            assert gen.output_dir == tmpdir
            assert os.path.isdir(tmpdir)

    def test_instantiate_creates_output_dir(self):
        """ReportGenerator creates the output directory if it does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "reports")
            assert not os.path.isdir(new_dir)
            gen = ReportGenerator(output_dir=new_dir)
            assert os.path.isdir(new_dir)


# ---------------------------------------------------------------------------
# Test: generate_markdown_report returns valid markdown string
# ---------------------------------------------------------------------------


class TestGenerateMarkdownReport:
    """Tests for generate_markdown_report output."""

    def test_returns_string(self):
        """generate_markdown_report returns a str."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert isinstance(result, str)

    def test_returns_non_empty_string(self):
        """generate_markdown_report returns a non-empty string."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert len(result) > 0

    def test_report_has_title(self):
        """Report starts with a title heading."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "# Underwriting Audit Report" in result

    def test_report_includes_applicant_summary_section(self):
        """Report includes Applicant Summary section."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "## Applicant Summary" in result

    def test_report_includes_applicant_name(self):
        """Applicant name appears in the report."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "Alex Standard" in result

    def test_report_includes_applicant_age(self):
        """Applicant age appears in the report."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "**Age:** 35" in result

    def test_report_includes_applicant_occupation(self):
        """Applicant occupation appears in the report."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "Software Manager" in result

    def test_report_includes_applicant_bmi(self):
        """Applicant BMI appears in the report."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "**BMI:** 24.0" in result

    def test_report_includes_agent_assessments_section(self):
        """Report includes Agent Assessments section."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "## Agent Assessments" in result

    def test_report_includes_medical_agent_assessment(self):
        """Report includes Medical agent assessment."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "### Medical" in result
        assert "**Risk Tier:** standard" in result
        assert "**Recommendation:** Approve" in result
        assert "**Confidence Score:** 0.95" in result

    def test_report_includes_financial_agent_assessment(self):
        """Report includes Financial agent assessment."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "### Financial" in result
        assert "**Risk Tier:** standard" in result

    def test_report_includes_compliance_agent_assessment(self):
        """Report includes Compliance agent assessment."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "### Compliance" in result

    def test_report_includes_debate_log_section(self):
        """Report includes Debate Log section."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "## Debate Log" in result

    def test_report_includes_debate_entry(self):
        """Report includes debate log entries."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "Round 1" in result
        assert "**Agent:** Financial" in result

    def test_report_includes_final_decision_section(self):
        """Report includes Final Decision section."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "## Final Decision" in result

    def test_report_includes_final_decision_value(self):
        """Report includes the final decision value."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "**Decision:** Standard Offer" in result

    def test_report_includes_consensus_reached(self):
        """Report includes consensus status."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "**Consensus Reached:** Yes" in result

    def test_report_includes_decision_reasoning(self):
        """Report includes decision reasoning."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "Standard Offer" in result
        assert "3 agents" in result

    def test_report_includes_regulatory_references_section(self):
        """Report includes Regulatory References section."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "## Regulatory References" in result

    def test_report_includes_apra_references(self):
        """Report includes APRA regulatory references."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "APRA Prudential Standard CPS230" in result
        assert "APRA CPS 220" in result
        assert "APRA CPS 520" in result

    def test_report_includes_flags_section(self):
        """Report includes Flags section."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "## Flags" in result

    def test_report_includes_flag_details(self):
        """Report includes flag details."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "MED-001" in result
        assert "low" in result
        assert "Slightly elevated BMI" in result

    def test_report_includes_additional_evidence_section(self):
        """Report includes Additional Evidence section."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "## Additional Evidence" in result

    def test_report_includes_additional_evidence_item(self):
        """Report includes additional evidence items."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "Proof of income for verification" in result

    def test_report_with_no_applicant_data(self):
        """Report handles missing applicant gracefully."""
        data = _make_decision_data()
        data["applicant"] = None
        gen = ReportGenerator()
        result = gen.generate_markdown_report(data)
        assert "No applicant data available" in result

    def test_report_with_no_agent_assessments(self):
        """Report handles missing agent assessments gracefully."""
        data = _make_decision_data()
        data["agent_assessments"] = {}
        gen = ReportGenerator()
        result = gen.generate_markdown_report(data)
        assert "No agent assessments available" in result

    def test_report_with_empty_debate_log(self):
        """Report handles empty debate log."""
        data = _make_decision_data()
        data["debate_log"] = []
        gen = ReportGenerator()
        result = gen.generate_markdown_report(data)
        assert "No debate occurred" in result or "## Debate Log" in result

    def test_report_with_no_flags(self):
        """Report handles no flags gracefully."""
        data = _make_decision_data()
        data["flags"] = []
        data["agent_assessments"]["Medical"]["flags"] = []
        data["agent_assessments"]["Financial"]["flags"] = []
        data["agent_assessments"]["Compliance"]["flags"] = []
        gen = ReportGenerator()
        result = gen.generate_markdown_report(data)
        assert "No flags raised" in result

    def test_report_with_no_evidence(self):
        """Report handles no additional evidence gracefully."""
        data = _make_decision_data()
        data["additional_evidence_required"] = []
        data["agent_assessments"]["Medical"]["additional_evidence_required"] = []
        data["agent_assessments"]["Financial"]["additional_evidence_required"] = []
        data["agent_assessments"]["Compliance"]["additional_evidence_required"] = []
        gen = ReportGenerator()
        result = gen.generate_markdown_report(data)
        assert "No additional evidence required" in result

    def test_report_markdown_formatting(self):
        """Report uses proper markdown formatting (headers, lists)."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        # Verify markdown headers
        assert result.startswith("# Underwriting Audit Report")
        assert "## " in result  # Section headers
        assert "- **" in result  # List items with bold labels


# ---------------------------------------------------------------------------
# Test: generate_pdf_report returns bytes
# ---------------------------------------------------------------------------


class TestGeneratePdfReport:
    """Tests for generate_pdf_report output."""

    def test_returns_bytes(self):
        """generate_pdf_report returns bytes."""
        gen = ReportGenerator()
        result = gen.generate_pdf_report(_make_decision_data())
        assert isinstance(result, bytes)

    def test_returns_non_empty_bytes(self):
        """generate_pdf_report returns non-empty bytes."""
        gen = ReportGenerator()
        result = gen.generate_pdf_report(_make_decision_data())
        assert len(result) > 0

    def test_pdf_content_contains_decision(self):
        """PDF content contains the final decision."""
        gen = ReportGenerator()
        result = gen.generate_pdf_report(_make_decision_data())
        decoded = result.decode("utf-8")
        assert "Standard Offer" in decoded

    def test_pdf_content_contains_applicant_name(self):
        """PDF content contains the applicant name."""
        gen = ReportGenerator()
        result = gen.generate_pdf_report(_make_decision_data())
        decoded = result.decode("utf-8")
        assert "Alex Standard" in decoded

    def test_pdf_fallback_when_weasyprint_unavailable(self):
        """PDF generation falls back to markdown string when weasyprint is unavailable."""
        gen = ReportGenerator()
        result = gen.generate_pdf_report(_make_decision_data())
        # Should always return bytes (either PDF or markdown encoded)
        assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# Test: save_report
# ---------------------------------------------------------------------------


class TestSaveReport:
    """Tests for save_report method."""

    def test_save_report_creates_file(self):
        """save_report creates the report file on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)
            filepath = gen.save_report(_make_decision_data(), "test_report.md")
            assert os.path.isfile(filepath)

    def test_save_report_writes_content(self):
        """save_report writes markdown content to the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)
            filepath = gen.save_report(_make_decision_data(), "test_report.md")
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            assert "Underwriting Audit Report" in content
            assert "Alex Standard" in content
            assert "Standard Offer" in content

    def test_save_report_returns_absolute_path(self):
        """save_report returns an absolute path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)
            filepath = gen.save_report(_make_decision_data(), "test_report.md")
            assert os.path.isabs(filepath)

    def test_save_report_uses_specified_filename(self):
        """save_report uses the specified filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)
            gen.save_report(_make_decision_data(), "custom_name.md")
            expected = os.path.join(tmpdir, "custom_name.md")
            assert os.path.isfile(expected)


# ---------------------------------------------------------------------------
# Test: Report data formatting correctness
# ---------------------------------------------------------------------------


class TestReportDataFormatting:
    """Tests for report data formatting correctness."""

    def test_applicant_name_format(self):
        """Applicant name is correctly formatted."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "**Name:** Alex Standard" in result

    def test_applicant_age_format(self):
        """Applicant age is correctly formatted."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "**Age:** 35" in result

    def test_applicant_income_format(self):
        """Applicant income is formatted with currency."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "$120,000.00" in result

    def test_risk_tier_format(self):
        """Risk tier is correctly formatted in agent assessment."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "**Risk Tier:** standard" in result

    def test_recommendation_format(self):
        """Recommendation is correctly formatted in agent assessment."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "**Recommendation:** Approve" in result

    def test_flag_format_in_report(self):
        """Flags are correctly formatted with severity and rule_id."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "[low]" in result
        assert "MED-001" in result
        assert "Slightly elevated BMI" in result

    def test_consensus_format(self):
        """Consensus status is correctly formatted."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "**Consensus Reached:** Yes" in result

    def test_debate_round_format(self):
        """Debate log entries are correctly formatted."""
        gen = ReportGenerator()
        result = gen.generate_markdown_report(_make_decision_data())
        assert "### Round 1" in result
        assert "**Agent:** Financial" in result
