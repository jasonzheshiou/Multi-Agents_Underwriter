"""
Seed data for the ChromaDB knowledge vector store.

Contains regulatory documents that are ingested into ChromaDB
to provide a knowledge base for the underwriting agents.
"""

REGULATORY_DOCUMENTS = [
    # ------------------------------------------------------------------ #
    #  APRA CPS 220 — Risk Management (5 docs)                           #
    # ------------------------------------------------------------------ #
    {
        "id": "cps220-001",
        "text": (
            "APRA CPS 220 requires all APRA-regulated institutions to maintain a risk management "
            "framework that identifies, measures, monitors, and reports on material risks. For life "
            "insurers, this includes underwriting risk, insurance risk, operational risk, and "
            "strategic risk. The framework must include a Risk Appetite Statement approved by the Board."
        ),
        "metadata": {
            "source": "APRA CPS 220",
            "section": "Risk Management Framework",
            "date": "2025-01-01",
            "category": "risk_management",
            "regulatory_weight": "mandatory",
        },
    },
    {
        "id": "cps220-002",
        "text": (
            "Under CPS 220, the Board is ultimately responsible for the risk management framework. "
            "Senior management must implement the framework and ensure it is embedded in day-to-day "
            "operations. Risk management must be integrated into decision-making processes, including "
            "underwriting decisions."
        ),
        "metadata": {
            "source": "APRA CPS 220",
            "section": "Board Responsibilities",
            "date": "2025-01-01",
            "category": "governance",
            "regulatory_weight": "mandatory",
        },
    },
    {
        "id": "cps220-003",
        "text": (
            "CPS 220 requires institutions to establish and maintain a Risk Appetite Statement (RAS) "
            "that articulates the level of risk the institution is willing to accept. For underwriting, "
            "this includes tolerance for premium loadings, declination rates, and exposure to specific "
            "risk categories."
        ),
        "metadata": {
            "source": "APRA CPS 220",
            "section": "Risk Appetite",
            "date": "2025-01-01",
            "category": "risk_management",
            "regulatory_weight": "mandatory",
        },
    },
    {
        "id": "cps220-004",
        "text": (
            "Under CPS 220, material risks must be identified and assessed. In underwriting, material "
            "risks include: pre-existing medical conditions, hazardous pursuits, occupational risks, "
            "financial instability indicators, non-disclosure risk, and concentration risk across "
            "portfolios."
        ),
        "metadata": {
            "source": "APRA CPS 220",
            "section": "Material Risk Identification",
            "date": "2025-01-01",
            "category": "risk_assessment",
            "regulatory_weight": "mandatory",
        },
    },
    {
        "id": "cps220-005",
        "text": (
            "CPS 220 requires regular stress testing and scenario analysis. Underwriters should "
            "consider how adverse scenarios (economic downturn, pandemic, climate events) would affect "
            "the risk profile of individual applications and the portfolio as a whole."
        ),
        "metadata": {
            "source": "APRA CPS 220",
            "section": "Stress Testing",
            "date": "2025-01-01",
            "category": "risk_management",
            "regulatory_weight": "mandatory",
        },
    },
    # ------------------------------------------------------------------ #
    #  APRA CPS 230 — Operational Risk (3 docs)                           #
    # ------------------------------------------------------------------ #
    {
        "id": "cps230-001",
        "text": (
            "APRA CPS 230 requires institutions to manage operational risk, including risks arising "
            "from inadequate or failed internal processes, people, systems, or external events. AI "
            "and automated decision systems fall under operational risk and must be governed "
            "accordingly."
        ),
        "metadata": {
            "source": "APRA CPS 230",
            "section": "Operational Risk Management",
            "date": "2025-01-01",
            "category": "operational_risk",
            "regulatory_weight": "mandatory",
        },
    },
    {
        "id": "cps230-002",
        "text": (
            "CPS 230 requires service provider management. When an AI system or LLM is used in the "
            "underwriting process (even locally hosted), it constitutes a material service "
            "arrangement. Institutions must assess the risk of the AI provider, including model "
            "reliability, data handling, and business continuity."
        ),
        "metadata": {
            "source": "APRA CPS 230",
            "section": "Service Provider Management",
            "date": "2025-01-01",
            "category": "third_party_risk",
            "regulatory_weight": "mandatory",
        },
    },
    {
        "id": "cps230-003",
        "text": (
            "CPS 230 requires business continuity planning. Underwriting systems must have documented "
            "fallback procedures when automated systems (including AI) are unavailable. The fallback "
            "must ensure underwriting decisions can still be made with appropriate governance."
        ),
        "metadata": {
            "source": "APRA CPS 230",
            "section": "Business Continuity",
            "date": "2025-01-01",
            "category": "resilience",
            "regulatory_weight": "mandatory",
        },
    },
    # ------------------------------------------------------------------ #
    #  LICOP 2.0 (4 docs)                                                 #
    # ------------------------------------------------------------------ #
    {
        "id": "licop-001",
        "text": (
            "Under the Life Insurance Code of Practice (LICOP 2.0), insurers must conduct "
            "underwriting fairly and transparently. Decisions must be based on sound underwriting "
            "principles and relevant medical and financial evidence. All underwriting decisions "
            "must be communicated clearly to the applicant."
        ),
        "metadata": {
            "source": "LICOP 2.0",
            "section": "Underwriting Standards",
            "date": "2024-07-01",
            "category": "underwriting",
            "regulatory_weight": "code",
        },
    },
    {
        "id": "licop-002",
        "text": (
            "LICOP 2.0 requires insurers to use plain language in all communications with "
            "applicants. Complex underwriting terms must be explained. Decision letters must "
            "clearly state the reason for any non-standard decision, including specific risk "
            "factors and remediation options where applicable."
        ),
        "metadata": {
            "source": "LICOP 2.0",
            "section": "Plain Language",
            "date": "2024-07-01",
            "category": "communication",
            "regulatory_weight": "code",
        },
    },
    {
        "id": "licop-003",
        "text": (
            "LICOP 2.0 requires special handling for vulnerable customers. Vulnerability indicators "
            "include: age over 75, serious medical conditions, financial hardship, limited English "
            "proficiency, and cognitive impairment. Insurers must have a vulnerability policy and "
            "train staff to identify and support vulnerable customers."
        ),
        "metadata": {
            "source": "LICOP 2.0",
            "section": "Vulnerable Customers",
            "date": "2024-07-01",
            "category": "consumer_protection",
            "regulatory_weight": "code",
        },
    },
    {
        "id": "licop-004",
        "text": (
            "LICOP 2.0 requires underwriting decisions to be made within reasonable timeframes. "
            "Standard applications should be processed within 20 business days. Where additional "
            "evidence is required, the insurer must communicate expected timeframes to the applicant."
        ),
        "metadata": {
            "source": "LICOP 2.0",
            "section": "Decision Timeframes",
            "date": "2024-07-01",
            "category": "process",
            "regulatory_weight": "code",
        },
    },
    # ------------------------------------------------------------------ #
    #  APRA AI Guidance (3 docs)                                          #
    # ------------------------------------------------------------------ #
    {
        "id": "ai-001",
        "text": (
            "The APRA April 2026 AI Letter sets expectations for AI use in the financial sector. "
            "Key principles: (1) Board accountability for AI decisions, (2) explainability of AI "
            "outputs, (3) continuous validation of AI models, (4) human oversight proportionate to "
            "materiality, (5) data quality and bias management."
        ),
        "metadata": {
            "source": "APRA AI Letter",
            "section": "AI Governance Principles",
            "date": "2026-04-01",
            "category": "ai_governance",
            "regulatory_weight": "guidance",
        },
    },
    {
        "id": "ai-002",
        "text": (
            "APRA expects that AI systems used in underwriting must be explainable to both internal "
            "stakeholders and external regulators. This means: the factors influencing decisions must "
            "be identifiable, the decision logic must be traceable (audit trail), and plain-English "
            "explanations must be available for applicants."
        ),
        "metadata": {
            "source": "APRA AI Letter",
            "section": "Explainability",
            "date": "2026-04-01",
            "category": "ai_governance",
            "regulatory_weight": "guidance",
        },
    },
    {
        "id": "ai-003",
        "text": (
            "APRA expects continuous validation of AI models used in decision-making. This includes: "
            "monitoring for model drift, regular back-testing against actual outcomes, bias testing "
            "across demographic groups, and documentation of model limitations. AI models must be "
            "revalidated when underlying data or assumptions change."
        ),
        "metadata": {
            "source": "APRA AI Letter",
            "section": "Continuous Validation",
            "date": "2026-04-01",
            "category": "model_risk",
            "regulatory_weight": "guidance",
        },
    },
]
