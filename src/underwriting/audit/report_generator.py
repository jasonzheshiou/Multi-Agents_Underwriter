"""Audit report generator for underwriting decisions."""

import os
from typing import Any, Dict, List, Optional


class ReportGenerator:
    """Generates audit reports (markdown and PDF) for underwriting decisions.

    Produces structured reports containing applicant summaries, agent
    assessments, debate logs, final decisions, and regulatory references.
    """

    DEFAULT_OUTPUT_DIR = "./audit_reports"

    def __init__(self, output_dir: Optional[str] = None) -> None:
        """Initialise the report generator.

        Args:
            output_dir: Directory to save generated reports. Defaults to
                ``./audit_reports``.
        """
        self.output_dir = output_dir or self.DEFAULT_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Markdown reports
    # ------------------------------------------------------------------

    def generate_markdown_report(self, decision_data: Dict[str, Any]) -> str:
        """Generate a markdown audit report from decision data.

        Args:
            decision_data: Dictionary containing the full underwriting
                decision context. Expected keys include ``applicant``,
                ``agent_assessments``, ``debate_log``, ``final_decision``,
                ``decision_reasoning``, ``flags``, and
                ``additional_evidence_required``.

        Returns:
            A markdown-formatted string representing the audit report.
        """
        lines: List[str] = []

        # Title
        lines.append("# Underwriting Audit Report")
        lines.append("")

        # Applicant summary
        lines.append("## Applicant Summary")
        lines.append("")
        applicant = decision_data.get("applicant", {})
        if applicant:
            lines.append(f"- **Name:** {applicant.get('full_name', 'N/A')}")
            lines.append(f"- **Date of Birth:** {applicant.get('date_of_birth', 'N/A')}")
            lines.append(f"- **Age:** {applicant.get('age', 'N/A')}")
            lines.append(f"- **Gender:** {applicant.get('gender', 'N/A')}")
            lines.append(f"- **Residency:** {applicant.get('residency_status', 'N/A')}")
            lines.append(f"- **Occupation:** {applicant.get('occupation', 'N/A')}")
            lines.append(f"- **Annual Income:** ${applicant.get('annual_income', 0):,.2f}")
            lines.append(f"- **BMI:** {applicant.get('bmi', 'N/A')}")
            lines.append(f"- **Smoker Status:** {applicant.get('smoker_status', 'N/A')}")
            benefits = applicant.get("benefit_types", [])
            lines.append(f"- **Benefits Requested:** {', '.join(str(b) for b in benefits)}")
            lines.append(f"- **Sum Insured (Death):** ${applicant.get('sum_insured_death', 0):,.2f}")
        else:
            lines.append("No applicant data available.")
        lines.append("")

        # Agent assessments
        lines.append("## Agent Assessments")
        lines.append("")
        agent_assessments = decision_data.get("agent_assessments", {})
        if agent_assessments:
            for agent_name, assessment in agent_assessments.items():
                if isinstance(assessment, dict):
                    risk_tier = assessment.get("risk_tier", "N/A")
                    recommendation = assessment.get("recommendation", "N/A")
                    confidence = assessment.get("confidence_score", "N/A")
                    reasoning = assessment.get("reasoning_summary", "N/A")
                    flags = assessment.get("flags", [])
                    evidence = assessment.get("additional_evidence_required", [])
                    apra_refs = assessment.get("apra_references", [])

                    lines.append(f"### {agent_name}")
                    lines.append("")
                    lines.append(f"- **Risk Tier:** {risk_tier}")
                    lines.append(f"- **Recommendation:** {recommendation}")
                    lines.append(f"- **Confidence Score:** {confidence}")
                    lines.append(f"- **Reasoning:** {reasoning}")
                    if flags:
                        lines.append("- **Flags:**")
                        for flag in flags:
                            if isinstance(flag, dict):
                                lines.append(
                                    f"  - [{flag.get('severity', 'N/A')}] "
                                    f"{flag.get('rule_id', 'N/A')}: "
                                    f"{flag.get('description', 'N/A')}"
                                )
                            else:
                                lines.append(f"  - {flag}")
                    if evidence:
                        lines.append("- **Additional Evidence Required:**")
                        for item in evidence:
                            lines.append(f"  - {item}")
                    if apra_refs:
                        lines.append("- **APRA References:**")
                        for ref in apra_refs:
                            lines.append(f"  - {ref}")
                else:
                    # Handle Pydantic model objects
                    lines.append(f"### {agent_name}")
                    lines.append("")
                    try:
                        data = assessment.dict() if hasattr(assessment, "dict") else vars(assessment)
                        lines.append(f"- **Risk Tier:** {data.get('risk_tier', 'N/A')}")
                        lines.append(f"- **Recommendation:** {data.get('recommendation', 'N/A')}")
                        lines.append(f"- **Confidence Score:** {data.get('confidence_score', 'N/A')}")
                        lines.append(f"- **Reasoning:** {data.get('reasoning_summary', 'N/A')}")
                    except Exception:
                        lines.append(f"- **Assessment:** {assessment}")
                lines.append("")
        else:
            lines.append("No agent assessments available.")
            lines.append("")

        # Debate log
        lines.append("## Debate Log")
        lines.append("")
        debate_log = decision_data.get("debate_log", [])
        if debate_log:
            for entry in debate_log:
                round_num = entry.get("round", "N/A")
                agent = entry.get("agent", "N/A")
                orig_tier = entry.get("original_tier", "N/A")
                updated_tier = entry.get("updated_tier", "N/A")
                reasoning = entry.get("reasoning", "N/A")
                lines.append(f"### Round {round_num}")
                lines.append("")
                lines.append(f"- **Agent:** {agent}")
                lines.append(f"- **Original Tiers:** {orig_tier}")
                lines.append(f"- **Updated Tier:** {updated_tier}")
                lines.append(f"- **Reasoning:** {reasoning}")
                lines.append("")
        else:
            lines.append("No debate occurred — consensus reached without debate.")
            lines.append("")

        # Final decision
        lines.append("## Final Decision")
        lines.append("")
        final_decision = decision_data.get("final_decision", "N/A")
        decision_reasoning = decision_data.get("decision_reasoning", "N/A")
        consensus = decision_data.get("consensus_reached", False)
        lines.append(f"- **Decision:** {final_decision}")
        lines.append(f"- **Consensus Reached:** {'Yes' if consensus else 'No'}")
        lines.append(f"- **Reasoning:** {decision_reasoning}")
        lines.append("")

        # Regulatory references
        lines.append("## Regulatory References")
        lines.append("")
        all_apra_refs: List[str] = []
        for assessment in agent_assessments.values():
            if isinstance(assessment, dict):
                refs = assessment.get("apra_references", [])
                all_apra_refs.extend(refs)
            else:
                try:
                    data = assessment.dict() if hasattr(assessment, "dict") else vars(assessment)
                    refs = data.get("apra_references", [])
                    all_apra_refs.extend(refs)
                except Exception:
                    pass
        if all_apra_refs:
            for ref in all_apra_refs:
                lines.append(f"- {ref}")
        else:
            lines.append("No specific regulatory references applied.")
        lines.append("")

        # Flags
        lines.append("## Flags")
        lines.append("")
        all_flags: List[Any] = decision_data.get("flags", [])
        if not all_flags:
            for assessment in agent_assessments.values():
                if isinstance(assessment, dict):
                    all_flags.extend(assessment.get("flags", []))
                else:
                    try:
                        data = assessment.dict() if hasattr(assessment, "dict") else vars(assessment)
                        all_flags.extend(data.get("flags", []))
                    except Exception:
                        pass
        if all_flags:
            for flag in all_flags:
                if isinstance(flag, dict):
                    lines.append(
                        f"- [{flag.get('severity', 'N/A')}] "
                        f"{flag.get('rule_id', 'N/A')}: "
                        f"{flag.get('description', 'N/A')}"
                    )
                else:
                    lines.append(f"- {flag}")
        else:
            lines.append("No flags raised.")
        lines.append("")

        # Additional evidence
        lines.append("## Additional Evidence")
        lines.append("")
        evidence = decision_data.get("additional_evidence_required", [])
        if not evidence:
            for assessment in agent_assessments.values():
                if isinstance(assessment, dict):
                    evidence.extend(assessment.get("additional_evidence_required", []))
                else:
                    try:
                        data = assessment.dict() if hasattr(assessment, "dict") else vars(assessment)
                        evidence.extend(data.get("additional_evidence_required", []))
                    except Exception:
                        pass
        if evidence:
            for item in evidence:
                lines.append(f"- {item}")
        else:
            lines.append("No additional evidence required.")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # PDF reports
    # ------------------------------------------------------------------

    def generate_pdf_report(self, decision_data: Dict[str, Any]) -> bytes:
        """Generate a PDF audit report from decision data.

        Attempts to convert the markdown report to PDF using ``markdown2``
        and ``weasyprint``. Falls back to returning the raw markdown string
        as bytes if those libraries are unavailable.

        Args:
            decision_data: Dictionary containing the full underwriting
                decision context.

        Returns:
            PDF content as ``bytes``. If PDF generation is unavailable,
            the markdown report is returned as ``bytes`` instead.
        """
        markdown_content = self.generate_markdown_report(decision_data)

        # Try PDF generation via weasyprint
        try:
            import markdown2  # type: ignore[import-not-found]
            from weasyprint import HTML  # type: ignore[import-not-found]

            html_content = markdown2.markdown(markdown_content)
            html_with_styles = (
                "<html><head><style>"
                "body { font-family: sans-serif; margin: 40px; }"
                "h1 { color: #2c3e50; }"
                "h2 { color: #34495e; margin-top: 24px; }"
                "h3 { color: #555; margin-top: 16px; }"
                "hr { border: none; border-top: 1px solid #ddd; margin: 16px 0; }"
                "ul { line-height: 1.6; }"
                "</style></head><body>"
                f"{html_content}"
                "</body></html>"
            )
            pdf_bytes = HTML(string=html_with_styles).write_pdf()
            return pdf_bytes
        except ImportError:
            # Fallback: return markdown as bytes
            return markdown_content.encode("utf-8")

    def save_report(self, decision_data: Dict[str, Any], filename: str) -> str:
        """Save the markdown report to disk.

        Args:
            decision_data: Dictionary containing the full underwriting
                decision context.
            filename: Output filename (e.g. ``audit_report.md``).

        Returns:
            Absolute path to the saved report file.
        """
        filepath = os.path.join(self.output_dir, filename)
        content = self.generate_markdown_report(decision_data)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return os.path.abspath(filepath)
