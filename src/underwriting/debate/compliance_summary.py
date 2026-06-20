"""Generate a compliance framework summary for display in the Streamlit UI.

Shows which Australian regulatory frameworks and AI Ethics Principles
are monitored by the Compliance Agent.
"""

import json
from pathlib import Path
from typing import Any


def _load_compliance_rules() -> list[dict[str, Any]]:
    """Load compliance rules from the JSON file.

    Returns:
        List of rule dicts from rules/death/compliance_rules.json,
        or an empty list if the file cannot be loaded.
    """
    rules_path = Path("rules/death/compliance_rules.json")
    if not rules_path.exists():
        return []
    try:
        with open(rules_path) as f:
            data = json.load(f)
        return data.get("rules", [])
    except (json.JSONDecodeError, OSError):
        return []


# ---------------------------------------------------------------------------
# Australia's 8 AI Ethics Principles (2019)
# ---------------------------------------------------------------------------

AI_PRINCIPLES = {
    "Human, Societal & Environmental Wellbeing": (
        "AI systems should benefit individuals, society, and the environment. "
        "Underwriting decisions should consider broader societal impacts."
    ),
    "Human-centred Values": (
        "AI systems should respect human rights, diversity, and individual autonomy. "
        "Plain-language communication and human-in-the-loop review are maintained."
    ),
    "Fairness": (
        "AI systems should be inclusive and accessible, and should not result in "
        "unfair discrimination. Anti-discrimination laws and LICOP mental health "
        "provisions are specifically monitored."
    ),
    "Privacy Protection & Security": (
        "AI systems should respect and uphold privacy rights and data protection. "
        "Private local LLM deployment ensures no data leaves the organisation."
    ),
    "Reliability & Safety": (
        "AI systems should reliably operate in accordance with their intended purpose. "
        "The deterministic rule engine provides consistent, reproducible assessments "
        "alongside any LLM enrichment."
    ),
    "Transparency & Explainability": (
        "There should be transparency and responsible disclosure so people can "
        "understand when they are being significantly impacted by AI. Full audit "
        "trail via debate log and JSONL logger."
    ),
    "Contestability": (
        "When an AI system significantly impacts a person, there should be an "
        "efficient process for people to challenge the use or outcomes. LICOP "
        "complaint handling and right-to-review provisions address this."
    ),
    "Accountability": (
        "People responsible for the different phases of the AI system lifecycle "
        "should be identifiable and accountable. Governance structures, disclosure "
        "acknowledgment, and conflict-of-interest checks are monitored."
    ),
}


def generate_compliance_framework_summary() -> str:
    """Generate a markdown summary of all compliance frameworks monitored.

    Returns a markdown string suitable for ``st.markdown()`` rendering
    in the Streamlit UI, showing:
    - Australia's 8 AI Ethics Principles with rule mappings
    - APRA prudential standards referenced
    - Other legislation/frameworks referenced
    """
    rules = _load_compliance_rules()
    if not rules:
        return "*No compliance rules loaded.*"

    # Group rules by AI principle
    by_principle: dict[str, list[dict]] = {}
    for rule in rules:
        principle = rule.get("ai_principle", "Uncategorised")
        by_principle.setdefault(principle, []).append(rule)

    # Group rules by framework
    frameworks: dict[str, list[str]] = {}
    for rule in rules:
        ref = rule.get("regulatory_reference", "")
        if " — " in ref:
            framework = ref.split(" — ")[0].strip()
        else:
            framework = ref
        frameworks.setdefault(framework, []).append(rule["rule_id"])

    lines: list[str] = []

    # Title
    lines.append("## \U0001F6E1\uFE0F Compliance Framework Summary")
    lines.append("")
    lines.append(
        f"The **Compliance Agent** monitors **{len(rules)} rules** across "
        f"Australian regulatory frameworks and AI Ethics Principles. "
        f"It acts as an **observer/informer** — spotting potential compliance "
        f"gaps without driving the underwriting decision."
    )
    lines.append("")

    # --- Section 1: AI Ethics Principles ---
    lines.append("### \U0001F916 Australia's 8 AI Ethics Principles")
    lines.append("")
    lines.append("| Principle | Rules | Status |")
    lines.append("|-----------|-------|--------|")

    for principle_name, description in AI_PRINCIPLES.items():
        mapped_rules = by_principle.get(principle_name, [])
        rule_ids = ", ".join(r["rule_id"] for r in mapped_rules) if mapped_rules else "—"
        status = (
            "\u2705 Monitored" if mapped_rules
            else "\u26A0\uFE0F Not yet mapped"
        )
        lines.append(f"| **{principle_name}** | {rule_ids} | {status} |")

    lines.append("")

    # --- Section 2: APRA Standards ---
    lines.append("### \U0001F4CB APRA Prudential Standards")
    lines.append("")
    apra_standards = [
        ("**CPS 220** — Risk Management", "CMP-D-002, CMP-D-010, CMP-D-031", "Governance, documentation, risk monitoring"),
        ("**CPS 230** — Operational Risk Management", "CMP-D-006", "Operational resilience for AI systems"),
        ("**CPS 234** — Information Security", "CMP-D-030", "Data security (private LLM deployment)"),
        ("**April 2026 AI Letter**", "CMP-D-008, CMP-D-031", "AI governance, explainability, board literacy, audit trail"),
    ]
    lines.append("| Standard | Rules | Coverage |")
    lines.append("|----------|-------|----------|")
    for std, rules_ref, coverage in apra_standards:
        lines.append(f"| {std} | {rules_ref} | {coverage} |")
    lines.append("")

    # --- Section 3: Other Legislation ---
    lines.append("### \u2696\uFE0F Australian Legislation & Codes")
    lines.append("")
    legislation = [
        "**Insurance Contracts Act 1984 (Cth)** — s.21A Duty of Disclosure, timely decision-making",
        "**Life Insurance Code of Practice (LICOP 2.0)** — Chapters 2, 3, 5, 7",
        "**Privacy Act 1988 (Cth)** — Schedule 1, APP 11 (Security of personal information)",
        "**Anti-Discrimination Acts** — Age (2004), Disability (1992), Racial (1975), Sex (1984)",
        "**ASIC Regulatory Guide 279** — Plain language requirements",
    ]
    for item in legislation:
        lines.append(f"- {item}")
    lines.append("")

    # --- Section 4: Default Status ---
    lines.append("### \u2705 Inherently Addressed (Private Local LLM)")
    lines.append("")
    lines.append(
        "Because this system runs on a **private local LLM** with no external "
        "API calls, the following are inherently addressed:"
    )
    lines.append("")
    lines.append("- **Data residency** — all data stays on-premises")
    lines.append("- **Encryption at rest** — ChromaDB and audit logs on local storage")
    lines.append("- **Access control** — local system authentication")
    lines.append("- **No third-party data sharing** — no external model providers")
    lines.append("- **APRA CPS 234** — satisfied by local-only deployment")
    lines.append("- **APRA AI Letter — Data Governance** — satisfied by private deployment")
    lines.append("")

    return "\n".join(lines)


def get_compliance_statistics() -> dict[str, Any]:
    """Return summary statistics about compliance rule coverage.

    Returns:
        Dict with keys: total_rules, principles_covered, frameworks_covered,
        high_severity_count, informational_count.
    """
    rules = _load_compliance_rules()
    if not rules:
        return {"total_rules": 0}

    principles = set()
    frameworks = set()
    high_count = 0
    info_count = 0

    for rule in rules:
        principle = rule.get("ai_principle", "")
        if principle:
            principles.add(principle)
        ref = rule.get("regulatory_reference", "")
        if ref and " — " in ref:
            frameworks.add(ref.split(" — ")[0].strip())
        sev = rule.get("severity", "low")
        if sev in ("high", "critical"):
            high_count += 1
        if rule.get("action", "") == "info":
            info_count += 1

    return {
        "total_rules": len(rules),
        "principles_covered": len(principles),
        "principles_total": 8,
        "frameworks_covered": len(frameworks),
        "high_severity_count": high_count,
        "informational_count": info_count,
    }
