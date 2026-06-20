"""Generate a professional HTML summary of a debate conversation."""

from html import escape
from typing import Any

from underwriting.debate.chat_models import Conversation

# Risk tier ranking — used to compute the final decision from agent assessments
_RISK_TIER_RANK: dict[str, int] = {
    "standard": 0,
    "loading": 1,
    "refer": 2,
    "decline": 3,
}

# Decision outcome mapping from highest risk tier rank
_DECISION_MAP: dict[int, str] = {
    0: "Standard Offer",
    1: "Offer with Loading/Exclusion",
    2: "Refer to Manual Underwriting",
    3: "Refer to Manual Underwriting",
}

# Severity ordering: lower number = more severe
_SEVERITY_ORDER: dict[str, int] = {
    "critical": 0,
    "high": 1,
    "moderate": 2,
    "low": 3,
}

# Risk tier color mapping
_TIER_COLORS: dict[str, str] = {
    "standard": "#28a745",
    "loading": "#fd7e14",
    "refer": "#ffc107",
    "decline": "#dc3545",
}


def _compute_decision_from_assessments(assessments: dict) -> str:
    """Compute the final underwriting decision from agent assessment data.

    Uses the same logic as ``DebateOrchestrator._produce_final_decision``:
    the most conservative (highest risk) tier across underwriting agents
    determines the outcome. The Compliance Agent is excluded — it spots
    compliance gaps but does not drive the underwriting decision.

    Args:
        assessments: Dict mapping agent names to their assessment dicts,
            each containing at least a ``"risk_tier"`` key.

    Returns:
        One of: ``"Standard Offer"``, ``"Offer with Loading/Exclusion"``,
        ``"Refer to Manual Underwriting"``, or ``"Pending"`` if there are
        no assessments.
    """
    if not assessments:
        return "Pending"
    # Exclude Compliance Agent from underwriting decision
    underwriting = {
        name: a for name, a in assessments.items()
        if "Compliance" not in name
    }
    source = underwriting if underwriting else assessments
    tiers = [a.get("risk_tier", "standard") for a in source.values()]
    ranks = [_RISK_TIER_RANK.get(t, 0) for t in tiers]
    highest_rank = max(ranks)
    return _DECISION_MAP.get(highest_rank, "Refer to Manual Underwriting")


def generate_debate_summary(conversation: Conversation) -> str:
    """Generate a professional HTML summary of the debate conversation.

    Args:
        conversation: The Conversation object with agent_assessments data.

    Returns:
        HTML string with inline CSS for the summary section, or "" if no data.
    """
    assessments = conversation.agent_assessments
    if not assessments:
        return ""

    # Separate underwriting agents from compliance (observer only)
    underwriting = {
        name: a for name, a in assessments.items()
        if "Compliance" not in name
    }
    compliance = {
        name: a for name, a in assessments.items()
        if "Compliance" in name
    }

    uw_names = sorted(underwriting.keys())
    uw_tiers = [underwriting[name]["risk_tier"] for name in uw_names] if uw_names else []
    all_same = len(set(uw_tiers)) == 1 if len(uw_tiers) >= 2 else True

    # Collect underwriting flags (risk-relevant) — top 5
    uw_flags: list[dict[str, Any]] = []
    for name in uw_names:
        for flag in underwriting[name].get("flags", []):
            uw_flags.append({
                "rule_id": flag.get("rule_id", ""),
                "severity": flag.get("severity", "low"),
                "description": flag.get("description", ""),
            })
    uw_flags.sort(key=lambda f: _SEVERITY_ORDER.get(f["severity"], 99))
    top_uw_flags = uw_flags[:5]

    # Collect compliance flags separately (process gaps, not risk)
    compliance_flags: list[dict[str, Any]] = []
    for name in sorted(compliance.keys()):
        for flag in compliance[name].get("flags", []):
            compliance_flags.append({
                "rule_id": flag.get("rule_id", ""),
                "severity": flag.get("severity", "low"),
                "description": flag.get("description", ""),
            })
    compliance_flags.sort(key=lambda f: _SEVERITY_ORDER.get(f["severity"], 99))

    # Build plain-language explanation (underwriting only)
    explanation_parts: list[str] = []
    for flag in top_uw_flags:
        desc = escape(str(flag["description"]))
        explanation_parts.append(desc)

    decision = escape(_compute_decision_from_assessments(assessments))

    if explanation_parts:
        explanation = (
            f"The application was flagged for {', '.join(explanation_parts[:-1])}"
            f"{' and ' + explanation_parts[-1] if len(explanation_parts) > 1 else ''},"
            f" resulting in a recommendation to {decision}."
        )
    else:
        explanation = f"No risk flags raised. Recommendation: {decision}."

    # Collect underwriting reasoning only
    reasoning_parts = []
    for name in uw_names:
        summary = underwriting[name].get("reasoning_summary", "")
        if summary:
            reasoning_parts.append(escape(str(summary)))

    # ---- Build HTML ----
    parts: list[str] = []

    # Card wrapper
    parts.append(
        '<div class="underwriting-summary-card">'
    )

    # 1. Decision Banner
    parts.append(
        f'<div class="underwriting-summary-banner">'
        f'<h2 style="margin:0;color:#fff;">{decision}</h2>'
        f'</div>'
    )

    # 1a. Evidence Re-evaluation Banner (if applicable)
    if getattr(conversation, "evidence_re_evaluated", False):
        parts.append(
            '<div class="underwriting-summary-evidence-banner" '
            'style="background:#fff3cd;border-left:4px solid #fd7e14;'
            'padding:10px 16px;margin:0;">'
            '<span style="font-size:0.92em;color:#856404;">'
            '&#9888;&#65039; <strong>Decision Re-evaluated</strong> &mdash; '
            'This decision incorporates user-provided evidence. '
            'The original pipeline assessment has been updated to reflect '
            'additional information submitted during the conversation.'
            '</span>'
            '</div>'
        )
    elif getattr(conversation, "user_evidence_applied", False):
        parts.append(
            '<div class="underwriting-summary-evidence-banner" '
            'style="background:#e7f3ff;border-left:4px solid #0c8599;'
            'padding:10px 16px;margin:0;">'
            '<span style="font-size:0.92em;color:#0b5e7a;">'
            '&#128269; <strong>Evidence Submitted</strong> &mdash; '
            'User-provided evidence was considered but did not change '
            'the risk tier. The decision reflects the original assessment.'
            '</span>'
            '</div>'
        )

    # 2. Risk Tier Comparison (underwriting agents only)
    parts.append('<div class="underwriting-summary-section">')
    parts.append('<h3 class="underwriting-summary-section-title">Underwriting Risk Tier Comparison</h3>')
    parts.append('<div class="underwriting-summary-tiers">')
    for name in uw_names:
        tier = underwriting[name]["risk_tier"]
        color = _TIER_COLORS.get(tier, "#6c757d")
        label = escape(str(tier)).capitalize()
        agent_label = escape(name)
        parts.append(
            f'<span class="underwriting-summary-tier-badge '
            f'underwriting-summary-{tier}" '
            f'style="background-color:{color};color:#fff;padding:4px 12px;'
            f'border-radius:12px;font-size:0.85em;margin:2px 4px 2px 0;">'
            f'{agent_label}: {label}</span>'
        )
    parts.append('</div></div>')

    # 3. Key Risk Flags (underwriting only)
    if top_uw_flags:
        parts.append('<div class="underwriting-summary-section">')
        parts.append('<h3 class="underwriting-summary-section-title">Key Risk Flags</h3>')
        parts.append('<ul class="underwriting-summary-flags">')
        for flag in top_uw_flags:
            sev = escape(str(flag["severity"])).lower()
            desc = escape(str(flag["description"]))
            rule = escape(str(flag["rule_id"]))
            parts.append(
                f'<li class="underwriting-summary-flag-item '
                f'underwriting-summary-severity-{sev}">'
                f'<span class="underwriting-summary-flag-severity '
                f'underwriting-summary-sev-{sev}">[{sev}]</span> '
                f'<strong>{rule}</strong>: {desc}'
                f'</li>'
            )
        parts.append('</ul></div>')

    # 4. Consensus / Debate Indicator
    parts.append('<div class="underwriting-summary-section">')
    parts.append('<h3 class="underwriting-summary-section-title">Agent Consensus</h3>')
    if all_same:
        parts.append(
            '<div class="underwriting-summary-consensus">'
            '<span style="color:#28a745;font-size:1.5em;">&#10004;</span> '
            '<strong>Consensus Reached</strong> — '
            f'All {len(uw_names)} underwriting agents agreed on the '
            f'{escape(str(uw_tiers[0])).lower()} tier.'
            '</div>'
        )
    else:
        parts.append(
            '<div class="underwriting-summary-debate">'
            '<span style="color:#fd7e14;font-size:1.5em;">&#9888;</span> '
            '<strong>Debate Warning</strong> — '
            'Underwriting agents disagreed on risk tier. '
            'Differences: '
            + ', '.join(
                f'{escape(name)} ({escape(str(underwriting[name]["risk_tier"]))})'
                for name in uw_names
            )
            + '.'
            '</div>'
        )
    parts.append('</div>')

    # 5. Compliance Observations (informational only — does not affect decision)
    if compliance:
        parts.append('<div class="underwriting-summary-section" style="background:#fff8e1;">')
        parts.append('<h3 class="underwriting-summary-section-title">&#128737; Compliance Observations (Informational)</h3>')
        parts.append('<p style="font-size:0.85em;color:#666;margin:0 0 8px 0;">')
        parts.append('The Compliance Agent monitors regulatory gaps but does <strong>not</strong> drive the underwriting decision.')
        parts.append('</p>')
        if compliance_flags:
            parts.append('<ul class="underwriting-summary-flags">')
            for flag in compliance_flags[:5]:
                sev = escape(str(flag["severity"])).lower()
                desc = escape(str(flag["description"]))
                rule = escape(str(flag["rule_id"]))
                parts.append(
                    f'<li class="underwriting-summary-flag-item '
                    f'underwriting-summary-severity-{sev}">'
                    f'<span class="underwriting-summary-flag-severity '
                    f'underwriting-summary-sev-{sev}">[{sev}]</span> '
                    f'<strong>{rule}</strong>: {desc}'
                    f'</li>'
                )
            parts.append('</ul>')
        else:
            parts.append('<p style="font-size:0.85em;color:#28a745;">&#10004; No compliance gaps identified.</p>')
        parts.append('</div>')

    # 6. Plain-Language Explanation
    parts.append('<div class="underwriting-summary-section">')
    parts.append('<h3 class="underwriting-summary-section-title">Summary</h3>')
    parts.append(f'<p class="underwriting-summary-explanation">{explanation}</p>')
    if reasoning_parts:
        parts.append('<div class="underwriting-summary-reasoning">')
        parts.append('<h4>Agent Reasoning</h4>')
        for name in uw_names:
            reason = underwriting[name].get("reasoning_summary", "")
            if reason:
                parts.append(
                    f'<div class="underwriting-summary-agent-reasoning">'
                    f'<strong>{escape(name)}:</strong> '
                    f'{escape(str(reason))}</div>'
                )
        parts.append('</div>')
    parts.append('</div>')

    parts.append('</div>')

    # Inline CSS scoped to underwriting-summary-*
    css = _build_css()
    return f'<style>{css}</style>' + ''.join(parts)


def _build_css() -> str:
    """Build inline CSS scoped to underwriting-summary classes."""
    return (
        '.underwriting-summary-card {'
        'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;'
        'max-width:720px;border:1px solid #dee2e6;border-radius:8px;'
        'overflow:hidden;background:#fff;box-shadow:0 2px 8px rgba(0,0,0,0.08);'
        '}'
        '.underwriting-summary-banner {'
        'background:linear-gradient(135deg,#007bff,#0056b3);'
        'padding:16px 20px;'
        '}'
        '.underwriting-summary-section {'
        'padding:12px 20px;border-bottom:1px solid #eee;'
        '}'
        '.underwriting-summary-section:last-child {border-bottom:none;}'
        '.underwriting-summary-section-title {'
        'font-size:1.05em;margin:0 0 8px 0;color:#333;'
        '}'
        '.underwriting-summary-tiers {display:flex;flex-wrap:wrap;align-items:center;}'
        '.underwriting-summary-tier-badge {'
        'font-weight:600;display:inline-block;'
        '}'
        '.underwriting-summary-flags {'
        'list-style:none;padding:0;margin:0;'
        '}'
        '.underwriting-summary-flag-item {'
        'padding:6px 0;border-bottom:1px solid #f0f0f0;'
        'font-size:0.92em;'
        '}'
        '.underwriting-summary-flag-item:last-child {border-bottom:none;}'
        '.underwriting-summary-flag-severity {'
        'font-weight:700;margin-right:6px;'
        '}'
        '.underwriting-summary-sev-critical {color:#dc3545;}'
        '.underwriting-summary-sev-high {color:#e8590c;}'
        '.underwriting-summary-sev-moderate {color:#fd7e14;}'
        '.underwriting-summary-sev-low {color:#0c8599;}'
        '.underwriting-summary-consensus {'
        'padding:10px 14px;background:#d4edda;border-radius:6px;'
        'font-size:0.93em;'
        '}'
        '.underwriting-summary-debate {'
        'padding:10px 14px;background:#fff3cd;border-radius:6px;'
        'font-size:0.93em;'
        '}'
        '.underwriting-summary-explanation {'
        'margin:0;font-size:0.95em;line-height:1.5;color:#495057;'
        '}'
        '.underwriting-summary-reasoning {'
        'margin-top:10px;'
        '}'
        '.underwriting-summary-agent-reasoning {'
        'padding:4px 0;font-size:0.88em;color:#555;'
        '}'
    )
