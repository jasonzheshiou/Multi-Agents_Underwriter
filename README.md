# Multi-Agents Underwriting Rules Engine

**Reimagining the future of risk: a multi-agent underwriting system that showcases human-AI collaboration among actuaries, underwriters, and intelligent automation.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-orange.svg)]()

---

⚠️ **In Development — Expect Bugs & Inconsistencies**

This project is actively under development. You may encounter bugs, inconsistent methodologies, or incomplete features. Breaking changes between versions are possible.

**This is intentional** — the project demonstrates what can be built using local LLMs for actuarial underwriting work.

---

## 🤖 AI-Generated Project Showcase

**This entire application was developed using the local LLM `Qwen3.6-35B-A3B`.**

The purpose of this project is to demonstrate that a **full-featured Multi-Agent Underwriting Rules Engine** can be built entirely with:

- **Local LLM**: Qwen3.6-35B-A3B (no API keys, no cloud dependency)
- **Open-source tools**: Streamlit, Pydantic, ChromaDB, pytest
- **Iterative AI-assisted development**: Code generation, debugging, refactoring, and documentation were all handled by the LLM

---

## 🎬 Demo

<img src="https://github.com/jasonzheshiou/Multi-Agents_Underwriter/main/Animation.gif" alt="Multi-Agents_Underwriter Demo" width="100%">

---

## Why This Project Exists

Underwriting rules engines are the operational backbone of insurance — they encode the logic that decides what risks are acceptable, at what terms, and at what price. Building them is hard. Maintaining them across regulatory regimes, product lines, and distribution channels is harder. Explaining them to auditors, regulators, and business stakeholders is harder still.

**The core tension**: underwriting rules are both deeply actuarial (statistical, risk-based, evidence-driven) and deeply operational (procedural, exception-prone, context-dependent). Traditional rules engines handle the operational side well but struggle with actuarial reasoning. AI models handle the reasoning well but struggle with transparency, auditability, and governance.

**This project explores a third path**: a multi-agent architecture where specialised agents — Medical, Financial, and Compliance — evaluate insurance applications through a structured debate process, governed by a framework that prioritises auditability and professional accountability. The system doesn't replace actuarial judgment. It structures it, scales it, and makes it inspectable.

This is a **demonstration and reference implementation**, not a production underwriting system. Its purpose is to:

1. **Illustrate** how multi-agent system design principles can apply to actuarial work
2. **Demonstrate** human-AI cooperation patterns that preserve professional accountability
3. **Provide** a concrete, runnable example that actuaries and engineers can study, critique, and extend
4. **Provoke** conversation about what actuarial work looks like when AI is a collaborator, not a replacement

If you're an actuary wondering how AI fits into your professional practice — or an engineer building systems for regulated domains — this project is for you.

---

## IMPORTANT — READ BEFORE USE

**THIS PROJECT IS FOR EDUCATIONAL, RESEARCH, AND GOVERNANCE PROTOTYPING PURPOSES ONLY.**

- **NOT for production underwriting decisions.** This system must NOT be used to make real underwriting decisions on live insurance applications.
- **NOT a substitute for qualified actuarial or underwriting judgment.** All outputs are drafts for professional review.
- **Synthetic data only.** All testing uses synthetic applicant profiles. No real policyholder data is or should be used.
- **LLM output is not authoritative.** Any AI-generated text is clearly marked and must be reviewed by a qualified professional.
- **Rules are illustrative, not comprehensive.** The deterministic rules represent publicly documented underwriting principles only and do not constitute a complete underwriting manual.
- **NO WARRANTY.** This software is provided "as is", without warranty of any kind, express or implied. Use entirely at your own risk.

This project demonstrates what is possible with local LLMs and transparent, auditable AI-assisted underwriting governance. It is a conversation starter, not a finished product.

---

## Overview

This engine simulates how multiple underwriting agents collaborate to assess a life insurance application. Each agent specialises in a different risk domain, evaluates the applicant against a set of deterministic rules, and participates in a structured debate when assessments conflict.

### Key Features

- **Three specialised agents** — Medical, Financial, and Compliance
- **Structured debate protocol** — agents challenge and refine each other's assessments
- **Interactive chat interface** — underwriters and compliance officers communicate directly with agents, asking questions, submitting evidence, and exploring agent reasoning in real time
- **Deterministic rule engine** — JSON-based rules with transparent evaluation
- **Complete audit trail** — JSONL decision logging, markdown report generation, and saved chat communication for transparent compliance review
- **Interactive Streamlit UI** — application questionnaire, results dashboard, and debate visualisation
- **Multi-benefit support** — Death, TPD, Trauma/Critical Illness, and Income Protection
- **Local LLM integration** — optional enrichment via OpenAI-compatible API (Ollama, llama.cpp, vLLM)

---

## 🏛️ System Design Principles

This project is built around principles that define how agents, rules, and humans interact. These principles are the architecture — the code is their expression.

### 1. Professional Accountability Resides with the Actuary and Underwriter

No agent makes a binding underwriting decision. Agents propose, analyse, and explain — but authority resides with qualified professionals. Underwriters interact with agents through the chat interface to explore assessments, extract insights, and build their understanding. The actuary sets the principles, calibrates the rules, and makes the final call on complex cases. Every decision traces back to an accountable human.

Actuaries gain unprecedented transparency into how underwriting decisions are made at scale — understanding exactly which risk factors drive loadings, declines, and referrals across the portfolio. This visibility informs reserving assumptions, pricing calibration, and risk appetite frameworks with empirical evidence rather than estimates.

### 2. Transparency Is Non-Negotiable

Every agent action is explainable. The system logs *what* each agent did, *why* it did it, and *what evidence* it relied on — producing a complete, structured audit trail. This is not a feature. It's a design constraint. Regulators, auditors, and internal governance must be able to reconstruct the reasoning chain.

### 3. Separation of Concerns Through Agent Specialisation

Rather than one monolithic AI making underwriting calls, the system decomposes underwriting into distinct risk domains — each handled by a specialised agent. When agents disagree, they engage in a structured debate, challenging and refining each other's assessments. If disagreement persists, the case is escalated.

### 4. Rules Are First-Class Citizens

Underwriting rules are not buried in code. They are declarative JSON documents — versioned, testable, and human-readable. Rules can be authored by actuaries, reviewed by compliance, and executed by agents without translation through a developer.

### 5. Governance Is Built In, Not Bolted On

Agent permissions, escalation paths, decision thresholds, and override rules are part of the system's core design — not afterthoughts. The governance framework is as important as the inference engine.

### 6. Open by Default

Every component of this project is open-source. The reasoning: actuarial work underpins public trust in insurance markets. The tools that actuaries use to make decisions should be inspectable by the public, regulators, and the profession. Closed-source underwriting engines are a governance black box. This project argues that they shouldn't be.

---

## 🤝 Human-AI Cooperation Model

This project embodies a specific philosophy of human-AI cooperation:

### What This Is NOT

- **NOT an AI that replaces underwriters.** Agents propose; humans decide. The system exists to assist humans in serving humans — not to replace human judgment.
- **NOT a black-box ML model.** Every agent output is structured, logged, and explainable.
- **NOT an attempt to automate actuarial judgment.** The goal is to *augment* it — to handle the routine, surface the anomalies, and make the reasoning inspectable.
- **NOT "prompt-level safety."** Telling an AI to "follow the rules" is not governance. This system implements structural controls: debate, thresholds, escalation, and audit.

### How Underwriters and Agents Work Together

The interactive chat is where the cooperation happens. Underwriters and compliance officers can:

- **Ask questions** — probe agent reasoning, challenge assessments, request clarification on specific rules
- **Submit evidence** — provide additional information that agents re-evaluate against their rules
- **Extract insights** — surface patterns across applications, understand risk drivers, and build domain knowledge
- **Facilitate training** — new underwriters learn by observing how agents reason and by exploring "what-if" scenarios
- **Document decisions** — every communication, evidence update, and tier change is saved to the audit trail

This creates a virtuous cycle: agents become better calibrated through human feedback, underwriters gain deeper insight through structured agent dialogue, and compliance has a complete, transparent record of every interaction.

### How Actuaries Use the System

Beyond the underwriting workflow, the system provides actuaries with structured, auditable data that feeds directly into core actuarial processes:

- **Reserving** — understand the risk distribution across the portfolio: how many cases are standard, loaded, or referred, and what risk factors drive those classifications. This granular risk visibility supports more accurate IBNR and UPR reserve setting.
- **Assumption setting** — the debate output reveals exactly which risk factors (BMI, smoking, occupation class, medical history, family history) trigger adverse assessments and at what thresholds. Actuaries can calibrate mortality and morbidity assumptions against observed agent behaviour rather than industry tables alone.
- **Pricing** — loading ranges and risk-tier distributions provide empirical evidence for pricing models. Actuaries can see which combinations of risk factors produce which loading outcomes, enabling data-driven premium rate setting and product design.
- **Experience analysis** — structured, timestamped decision data enables actual-vs-expected analysis across every risk factor. Actuaries can monitor whether agent decisions align with claims experience over time and adjust rules accordingly.
- **Rule validation and scenario testing** — actuaries can test proposed rule changes against historical applications to quantify the impact on acceptance rates, loading distributions, and portfolio risk profile before changes go live.
- **Portfolio monitoring** — aggregate decision data reveals emerging risk patterns, shifts in applicant demographics, and changes in agent behaviour, enabling proactive risk management and early warning detection.
- **Model governance** — the transparent, deterministic rule engine means actuaries can validate that agents behave consistently with the intended risk framework. Every decision is reproducible and explainable, satisfying APRA's model governance expectations.
- **Regulatory reporting** — the complete audit trail provides evidence for APRA CPS 220 risk management reporting, CPS 230 operational risk disclosures, and Appointed Actuary investigations. Every risk decision is documented with the reasoning chain intact.

### The Cooperation Model

```
Human Actuary:       "I set the principles, calibrate the rules, and make
                     the final call. I use the structured decision data to
                     inform reserving, pricing, and assumption setting —
                     with complete transparency into how every risk is assessed."
Underwriter:         "I interact with agents, explore assessments, extract insights,
                     and build my professional judgment through structured dialogue."
Compliance Officer:  "I review the audit trail — every agent assessment, every
                     evidence update, every decision change is documented."
AI Agents:           "We apply the rules, explain our reasoning, respond to questions,
                     and document everything transparently."
Together:            "We handle more cases, with more consistency, deeper insight,
                     and better audit trails than either could alone."
```

The actuary and underwriter defines *what good underwriting looks like*. The underwriter engages with agents to *understand each case deeply*. The agents execute, explain, and document. When agents are uncertain — or when they disagree — the human steps in. This is not a weakness of the system. It is the system working as designed: **assisting humans to serve humans, not replacing people.**

---

<details>
<summary><strong>📦 Installation</strong></summary>
<br>

### Prerequisites

- Python 3.10 or later
- Git (optional, for cloning)

### Steps

```bash
# Clone the repository
git clone <repository-url>
cd Multi_Agents_Underwriting_Rules_Engine

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Or install in development mode (includes testing/linting tools)
pip install -e ".[dev]"
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit>=1.28.0` | Interactive web UI |
| `pydantic>=2.0.0` | Data validation and settings management |
| `pyyaml>=6.0` | YAML config and questionnaire file parsing |
| `chromadb>=0.4.0` | Vector store for knowledge persistence |
| `openai>=1.0.0` | OpenAI-compatible API client (optional LLM enrichment) |
| `pypdf2>=3.0.0` | PDF processing for document analysis |

**Development dependencies** (`pip install -e ".[dev]"`):
| Package | Purpose |
|---------|---------|
| `pytest>=7.0.0` | Test framework |
| `pytest-cov>=4.0.0` | Test coverage reporting |
| `ruff>=0.1.0` | Linting and formatting |

### Optional: LLM Backend

The engine works fully without an LLM (deterministic mode). To enable optional LLM enrichment:

1. Set up an OpenAI-compatible server (e.g., Ollama, llama.cpp, vLLM)
2. Edit `config.yaml` with your endpoint details:

```yaml
llm:
  baseURL: http://localhost:1234/v1
  model: your-model-name
  temperature: 0.1
  max_tokens: 2000
```

Or set environment variables:

```bash
set LLM_BASE_URL=http://localhost:1234/v1   # Windows
set LLM_MODEL=your-model-name               # Windows
# export LLM_BASE_URL=http://localhost:1234/v1   # macOS/Linux
# export LLM_MODEL=your-model-name               # macOS/Linux
```

</details>

---

<details>
<summary><strong>🚀 Usage</strong></summary>
<br>

### Streamlit Application

Launch the interactive web interface:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` and provides five pages via the sidebar:

1. **Questionnaire** — Full interactive underwriting form (Sections A through H). Fill in all fields and submit to run the multi-agent pipeline.
2. **Results & Debate** — View the underwriting outcome, agent assessments with matched rules, AI-generated decision summary, and an interactive debate log with chat interface for asking questions and submitting new evidence.
3. **Test Questionnaire** — Load pre-filled YAML questionnaires from `data/test_questionnaires/`, edit values, select agents, and run the pipeline with one click.
4. **Compliance Framework** — Reference page showing regulatory framework mappings, compliance statistics, and AI ethics principle coverage.
5. **Rules Reference** — Browse all deterministic underwriting rules (Medical and Financial) with conditions, severities, and regulatory references.

### Test Questionnaire (Recommended for Initial Exploration)

The fastest way to explore the system is with pre-filled test questionnaires. Load a YAML questionnaire, edit any values, and run the pipeline with one click — no need to fill out the full form:

```bash
python demo/test_questionnaire_cli.py data/test_questionnaires/healthy_applicant.yaml
```

Or use the interactive Test Questionnaire page in the Streamlit app:
1. Navigate to **Test Questionnaire** in the sidebar
2. Select a `.yaml` file from the dropdown — start with `healthy_applicant.yaml`
3. Edit any applicant values directly on the form to experiment with different risk profiles
4. Choose which agents to run (Medical, Financial, Compliance)
5. Click **Run Underwriting Pipeline**
6. Results auto-navigate to the Results & Debate page where you can interact with agents

Create your own test questionnaires by adding `.yaml` files to `data/test_questionnaires/`. See [TEST_INSTRUCTIONS.md](TEST_INSTRUCTIONS.md) for the YAML format and example profiles.

### Programmatic Usage

Run the engine directly from Python:

```python
import sys
sys.path.insert(0, "src")

from underwriting.application.schema import Application, BenefitType, SmokerStatus
from underwriting.agents.medical_agent import MedicalAgent
from underwriting.agents.financial_agent import FinancialAgent
from underwriting.agents.compliance_agent import ComplianceAgent
from underwriting.agents.debate_orchestrator import DebateOrchestrator
from datetime import date

# Build an application
app = Application(
    full_name="Jane Doe",
    date_of_birth=date(1990, 6, 15),
    gender="Female",
    residency_status="Australian Citizen",
    contact_address="123 Main St, Sydney NSW 2000",
    benefit_types=[BenefitType.DEATH],
    sum_insured_death=500000,
    occupation="Software Engineer",
    employer_name="Tech Corp",
    years_in_occupation=5.0,
    annual_income=120000.0,
    height_cm=165.0,
    weight_kg=68.0,
    smoker_status=SmokerStatus.NEVER,
)

# Run agents
medical_agent = MedicalAgent(name="Medical Agent", rules_path="rules/death/medical_rules.json")
financial_agent = FinancialAgent(name="Financial Agent", rules_path="rules/death/financial_rules.json")
compliance_agent = ComplianceAgent(name="Compliance Agent", rules_path="rules/death/compliance_rules.json")

orchestrator = DebateOrchestrator(agents=[medical_agent, financial_agent, compliance_agent])
results = orchestrator.run(app)

# Access results
print(f"Decision: {results['final_decision']}")
print(f"Reasoning: {results['decision_reasoning']}")
```

### Pipeline Demo

Run the end-to-end pipeline on a synthetic applicant:

```bash
python demo/pipeline.py
```

### Synthetic Data Generation

Generate test applicant profiles:

```bash
python data/generate_synthetic.py
```

</details>

---

<details>
<summary><strong>⚙️ Configuration</strong></summary>
<br>

### config.yaml

The main configuration file controls LLM settings and filesystem paths:

```yaml
llm:
  baseURL: http://192.168.1.59:1234/v1
  model: qwen/qwen3.6-35b-a3b
  context_window: 200000
  wait_time: 3600
  temperature: 0.1
  max_tokens: 2000

paths:
  chroma_persist_dir: ./src/underwriting/knowledge/chroma_db
  log_dir: ./audit_logs
```

### Environment Variable Overrides

Any config key containing `base_url`, `api_key`, `model`, or `path` (case-insensitive) can be overridden by an environment variable with the same name in uppercase:

```bash
set LLM_BASE_URL=http://localhost:8080/v1
set PATHS_LOG_DIR=./my-logs
```

### Rule Files

Rules live in `rules/<benefit_type>/` as JSON files. Each file contains a `rules` array. Every rule has:

- `rule_id` — unique identifier
- `condition` — Python expression evaluated against the applicant object
- `severity` — `low`, `moderate`, `high`, `critical`
- `recommendation` — `standard`, `loading`, `decline`, `refer`
- `loading_range` — `[min_rate, max_rate]` percentage multiplier
- `description` — human-readable explanation
- `apra_ref` — relevant regulatory reference (optional)

To add a new rule, append an object to the `rules` array in the relevant JSON file.

</details>

---

<details>
<summary><strong>📊 Decision Outcomes</strong></summary>
<br>

| Outcome | Criteria |
|---------|----------|
| **Standard Offer** | All agents pass; no flags raised |
| **Offer with Loading** | One or more agents flag moderate risk; loading applied within defined range |
| **Offer with Exclusion** | Specific risk excluded (e.g., back condition, hazardous pursuit) |
| **Request Additional Evidence** | Insufficient information for decision; specific evidence requested |
| **Refer to Manual Underwriting** | High complexity, conflicting signals, or critical severity flags |
| **Decline** | Risk exceeds acceptable thresholds |

</details>

---

<details>
<summary><strong>📜 Compliance Framework</strong></summary>
<br>

Insurance is a regulated industry. Underwriting systems must satisfy multiple compliance frameworks. This project is designed with the following compliance principles:

| Principle | Implementation |
|---|---|
| **Explainability** | Every agent decision includes a structured rationale referencing specific rules, data, and reasoning steps |
| **Auditability** | Complete, immutable audit trail from case submission to final decision |
| **Reproducibility** | Given the same inputs and rules, the system produces the same outputs |
| **Separability** | Agent domains are isolated — a change to pricing logic doesn't silently affect compliance logic |
| **Override Governance** | All human overrides are logged with reason, identity, and timestamp; override patterns are monitored |
| **Model Governance** | Agent behaviour can be tested against known cases; regression testing ensures agents don't drift |

These principles guide the design. Compliance with specific regulatory frameworks requires independent professional actuarial judgment.

### Regulatory Framework Mapping

This engine maps its rules and audit design to Australian regulatory frameworks:

| Regulation / Standard | Relevance | Implemented In |
|-----------------------|-----------|----------------|
| **Insurance Contracts Act 1984 (Cth)** | Duty of disclosure (s.21A), utmost good faith | Compliance Agent: CMP-D-001 |
| **Life Insurance Code of Practice (LICOP 2.0)** | Underwriting standards, plain language, mental health assessment, vulnerable customers, decision timeframes | Compliance Agent: CMP-D-002 to CMP-D-010 |
| **APRA CPS 220 Risk Management** | Risk management framework, material risk identification, measurement, monitoring, reporting | All agents, audit trail |
| **APRA CPS 230 Operational Risk Management** | Operational resilience, service provider management, extends to AI systems | Compliance Agent: CMP-D-006 |
| **APRA CPS 234 Information Security** | Data protection, encryption, access controls | Compliance Agent: CMP-D-020 |
| **Privacy Act 1988 (Cth), APP 11** | Security of personal information | Compliance Agent: CMP-D-020 |
| **APRA April 2026 AI Letter** | AI-specific expectations: governance, explainability, board literacy, continuous validation | Audit trail design, agent logging |

</details>

---

<details>
<summary><strong>🧪 Testing</strong></summary>
<br>

### Run the Test Suite

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src/underwriting --cov-report=term-missing
```

### Run Linting

```bash
ruff check .
```

### Run Type Checking

```bash
mypy src/
```

### Test Structure

| File | Purpose |
|------|---------|
| `tests/conftest.py` | Shared fixtures (synthetic applicants) |
| `tests/test_base_agent.py` | Base agent and assessment model tests |
| `tests/test_medical_agent.py` | Medical agent rule evaluation |
| `tests/test_financial_agent.py` | Financial agent rule evaluation |
| `tests/test_debate_orchestrator.py` | Debate coordination logic |
| `tests/test_decision_synthesis.py` | Decision aggregation |
| `tests/test_rule_engine.py` | Core rule engine |
| `tests/test_llm_client.py` | LLM client wrapper |
| `tests/test_config.py` | Configuration loading |
| `tests/test_application_schema.py` | Pydantic model validation |
| `tests/test_decision_logger.py` | Audit logging |
| `tests/test_report_generator.py` | Report generation |
| `tests/test_vector_store.py` | ChromaDB knowledge store |
| `tests/test_integration_agents.py` | Multi-agent integration |
| `tests/test_integration_orchestrator.py` | Full pipeline integration |
| `tests/rules/` | Per-rule-type tests (financial, medical, compliance, benefit limits) |

See [TEST_INSTRUCTIONS.md](TEST_INSTRUCTIONS.md) for the test questionnaire YAML format and example profiles.

</details>

---

<details>
<summary><strong>🌐 The Open-Source Actuarial Ecosystem</strong></summary>
<br>

This project is part of a broader initiative to build an open-source ecosystem for actuarial work.

**The thesis**: Actuarial science has a long tradition of professional collaboration — experience studies, mortality tables, industry working groups. But our *tools* have remained largely proprietary. In an era where AI is reshaping every profession, actuaries need open, inspectable tools more than ever. Open-source is not just a licensing choice. It is a governance choice.

### Cross-Repository References

| Repository | Purpose |
|------------|---------|
| **Experience Analysis Dashboard** | Industry benchmark data and claims experience analysis. Feeds comparative data into underwriting decisions. |
| **Standard Inquirer** | Compliance logic engine. Provides regulatory reference checks and policy condition lookups. |
| **Synthetic Life Insurance Data Generator** | Bulk test profile generation. Produces realistic synthetic applicant data for testing and validation. |

### What This Ecosystem Aims to Provide

- **Reference implementations** of actuarial systems that can be studied, critiqued, and adapted
- **Shared components** that reduce the cost of building well-governed actuarial technology
- **A community** of actuaries, engineers, and regulators who believe that the tools of our profession should be transparent

If you're an actuary, engineer, or regulator interested in this vision — get in touch.

</details>

---

<details>
<summary><strong>🤝 Contributing</strong></summary>
<br>

Contributions are welcome — especially from:

- **Actuaries** who can review the underwriting logic and compliance principles
- **Engineers** who can improve the multi-agent architecture
- **Regulators and compliance professionals** who can sharpen the governance framework
- **Anyone** who finds a bug, has an idea, or wants to discuss the intersection of actuarial science and AI

For now: open an issue, start a discussion, or reach out directly. A `CONTRIBUTING.md` will be added as the project matures.

</details>

---

<details>
<summary><strong>🏗️ Architecture</strong></summary>
<br>

```
                        +---------------------+
                        |   Streamlit UI      |
                        | (Questionnaire +    |
                        |  Results + Debate)  |
                        +---------+-----------+
                                  |
                                  v
                    +-------------+-------------+
                    |   DebateOrchestrator      |
                    |  (coordination + debate)  |
                    +-------------+-------------+
                              |     |     |
              +---------------+     |     +---------------+
              v                 v              v
    +-------------+   +-------------+   +---------------+
    |  Medical    |   |  Financial  |   |  Compliance   |
    |  Agent      |   |  Agent      |   |  Agent        |
    +-------------+   +-------------+   +---------------+
          |                 |                  |
          v                 v                  v
    +-------------+   +-------------+   +---------------+
    | Medical     |   | Financial   |   | Compliance    |
    | Rules JSON  |   | Rules JSON  |   | Rules JSON    |
    +-------------+   +-------------+   +---------------+
          |                 |                  |
          v                 v                  v
    +-------------------------------------------------+
    |          Deterministic Rule Engine              |
    |  (evaluate conditions → matched rules →         |
    |   risk tier + flags + loading range)            |
    +-------------------------------------------------+
                                  |
                                  v
                    +-------------+-------------+
                    |   DecisionSynthesizer    |
                    |  (combine → final        |
                    |   decision + reasoning)  |
                    +-------------+-------------+
                                  |
                                  v
                    +-------------+-------------+
                    |   Audit Logger           |
                    |  (JSONL + Markdown       |
                    |   report generation)     |
                    +---------------------------+
```

### Component Breakdown

| Component | Location | Purpose |
|-----------|----------|---------|
| Application Schema | `src/underwriting/application/schema.py` | Pydantic models for applicant data |
| Rule Engine | `src/underwriting/rules/rule_engine.py` | Deterministic rule evaluation |
| Medical Agent | `src/underwriting/agents/medical_agent.py` | Health risk assessment |
| Financial Agent | `src/underwriting/agents/financial_agent.py` | Financial risk assessment |
| Compliance Agent | `src/underwriting/agents/compliance_agent.py` | Regulatory compliance check |
| Debate Orchestrator | `src/underwriting/agents/debate_orchestrator.py` | Agent coordination and debate |
| Decision Synthesizer | `src/underwriting/agents/decision_synthesis.py` | Final decision aggregation |
| Audit Logger | `src/underwriting/audit/logger.py` | Structured JSONL logging |
| Report Generator | `src/underwriting/audit/report_generator.py` | Markdown report output |
| Vector Store | `src/underwriting/knowledge/vector_store.py` | ChromaDB knowledge persistence |
| LLM Client | `src/underwriting/llm/llm_client.py` | OpenAI-compatible API wrapper |
| Configuration | `src/underwriting/config.py` | YAML config + env var overrides |
| Rules (Death) | `rules/death/` | Medical, financial, compliance, benefit limit rules |
| Rules (TPD) | `rules/tpd/` | TPD-specific rules |
| Rules (Trauma) | `rules/trauma/` | Trauma/Critical Illness rules |
| Rules (IP) | `rules/ip/` | Income Protection rules |
| Synthetic Data | `data/generate_synthetic.py` | Test profile generator |

</details>

---

<details>
<summary><strong>📂 Project Structure</strong></summary>
<br>

```
Multi_Agents_Underwriting_Rules_Engine/
├── app.py                          # Streamlit application
├── config.yaml                     # LLM and path configuration
├── pyproject.toml                  # Project metadata and dependencies
├── requirements.txt                # Pip dependencies
├── LICENSE                         # MIT License
├── README.md                       # Project documentation
├── AGENTS.md                       # AI development guidelines
├── AI_GUIDE.md                     # AI usage guide
├── CONTRIBUTING.md                 # Contribution guidelines
├── DEVELOPMENT_RULES.md            # Coding standards
├── RELEASE_NOTES.md                # Release history
├── TEST_INSTRUCTIONS.md            # Test questionnaire format guide
├── plan.md                         # Project plan
├── src/underwriting/
│   ├── __init__.py
│   ├── config.py                   # Configuration loader
│   ├── agents/
│   │   ├── base_agent.py           # AgentAssessment model + BaseAgent
│   │   ├── medical_agent.py        # Medical risk evaluation
│   │   ├── financial_agent.py      # Financial risk evaluation
│   │   ├── compliance_agent.py     # Regulatory compliance check
│   │   ├── debate_orchestrator.py  # Agent debate coordination
│   │   └── decision_synthesis.py   # Final decision aggregation
│   ├── application/
│   │   └── schema.py               # Pydantic Application model
│   ├── rules/
│   │   └── rule_engine.py          # Deterministic rule evaluator
│   ├── debate/
│   │   ├── __init__.py
│   │   ├── chat_models.py          # Conversation + ChatMessage models
│   │   ├── chat_summary.py         # Debate summary generation
│   │   ├── compliance_summary.py   # Compliance summary generation
│   │   └── persistence.py          # JSON file-based conversation store
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── vector_store.py         # ChromaDB persistence
│   │   ├── chroma_db/              # Persisted knowledge base
│   │   └── ingestion/
│   │       ├── __init__.py
│   │       └── seeds.py            # Initial knowledge base seeds
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── llm_client.py           # OpenAI-compatible client
│   │   └── prompt_templates.py     # Agent prompt templates
│   ├── audit/
│   │   ├── __init__.py
│   │   ├── logger.py               # JSONL structured logging
│   │   └── report_generator.py     # Markdown report output
│   └── test_questionnaire/
│       ├── __init__.py
│       ├── engine.py               # Pipeline orchestration for test runs
│       └── models.py               # QuestionnaireDefinition model
├── rules/
│   ├── death/   # Death benefit rules
│   ├── tpd/     # TPD benefit rules
│   ├── trauma/  # Trauma/Critical Illness rules
│   └── ip/      # Income Protection rules
├── demo/
│   ├── pipeline.py                 # End-to-end pipeline demo
│   └── test_questionnaire_cli.py   # CLI test questionnaire runner
├── data/
│   ├── generate_synthetic.py       # Synthetic profile generator
│   ├── synthetic_applicants/        # Pre-generated test profiles (JSON)
│   ├── test_questionnaires/         # YAML test questionnaire files
│   └── chat_conversations/          # Saved debate conversations (JSON)
├── tests/
│   ├── conftest.py                 # Shared fixtures
│   ├── rules/                      # Per-rule-type tests
│   └── test_*.py                   # Test modules
├── audit_logs/                     # Generated JSONL audit output
└── audit_reports/                  # Generated JSON decision reports
```

</details>

---

## License

Copyright (c) 2026 [Your Name]

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

The MIT License was chosen deliberately. It is permissive, widely understood, and imposes minimal friction on adoption. The goal of this ecosystem is maximum reach — the more actuaries and engineers who can study, use, and improve these tools, the better.

---

## Disclaimer

This project is a **demonstration and educational tool**. It is not a production underwriting system. It has not been validated against any regulatory framework. It should not be used to make real underwriting decisions without independent actuarial review and professional certification.

The author is a Fellow of the Institute of Actuaries of Australia (FIAA), but this project represents personal work and does not reflect the views or endorsement of any employer, client, or professional body.

Use of this software is at your own risk. See the [LICENSE](LICENSE) for the full disclaimer of warranty.

---

*"The future of underwriting is not humans versus AI. It's humans with AI — assisting professionals to serve their customers better, with more insight, deeper understanding, and complete transparency. The tools we build together should be open for everyone to inspect, improve, and trust."*
