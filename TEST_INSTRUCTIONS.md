# Test Instructions — Multi-Agents Underwriting Rules Engine

How to test every feature of the Streamlit app: questionnaire, debate, evidence submission, and result re-evaluation.

---

## Quick Start

### Prerequisites

- Python 3.10+ installed
- Dependencies installed: `pip install -r requirements.txt`
- No LLM backend required (the engine works deterministically without one)

### Launch the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` with a sidebar navigation menu.

### Navigation Pages

| Page | Purpose |
|------|---------|
| **Questionnaire** | Full interactive underwriting form (Sections A-H). Submit to see results. |
| **Results & Debate** | View results, agent assessments, debate log, and chat with agents. |
| **Test Questionnaire** | Load pre-filled YAML questionnaires, edit values, run the pipeline. |
| **Compliance Framework** | Reference page for regulatory compliance mappings. |
| **Rules Reference** | Browse all deterministic medical and financial rules. |

---

## 1. Testing the Questionnaire

### 1.1 Full Manual Questionnaire (Questionnaire Page)

Navigate to **Questionnaire** in the sidebar. The form has 8 sections:

| Section | Fields | Notes |
|---------|--------|-------|
| **A: Personal & Demographic** | Full Name, DOB, Gender, Residency, Address | All required |
| **B: Cover Requested** | Benefit types (Death/TPD/Trauma/IP), sum insured per type, existing policies, previous declinations | Conditional fields appear based on selections |
| **C: Occupation & Income** | Occupation, Employer, Years in role, Annual income, Hazardous duties toggle | Income must be > 0 |
| **D: Health — General** | Height, Weight, Smoker status, Medications, Alcohol | Conditional fields for smoker details |
| **E: Pre-existing Conditions** | "Add Condition" button, condition name, diagnosis date, doctor, treatments, symptoms, hospitalisations, lifestyle impact | Only shown if Section D "medical conditions" checkbox is checked |
| **F: Family History** | Family member relationship, condition, age at diagnosis | Only shown if family history checkbox is checked |
| **G: Lifestyle** | Hazardous pursuits, recreational drugs, alcohol/drug treatment, high-risk travel | Conditional detail fields |
| **H: Financial** | Net worth, financial obligations, obligation end dates, bankruptcy, criminal convictions | Conditional bankruptcy status radio |
| **Compliance** | Duty of Disclosure acknowledgment | Must be checked to submit |

**Test Steps:**

1. Fill in all required fields (marked with `*`)
2. Test conditional logic: check/uncheck toggles to verify dependent fields appear/disappear
3. Try submitting with intentional errors:
   - Leave Full Name blank → should show error
   - Leave Occupation blank → should show error
   - Set height to 0 or 999 → should show validation error
   - Leave income at 0 → should show error
   - Don't check Duty of Disclosure → should show error
4. **Submit a valid application** → page shows success message
5. Navigate to **Results & Debate** to see the outcome

**Expected:** Validation errors display as red error messages. Successful submission stores results and navigable to Results page.

---

## 2. Testing the Test Questionnaire

### 2.1 Overview

The **Test Questionnaire** page lets you load pre-filled YAML questionnaire files, edit any values, select which agents to run, and execute the full multi-agent pipeline.

Navigate to **Test Questionnaire** in the sidebar.

### 2.2 Select a Questionnaire File

1. Use the dropdown to select a pre-filled `.yaml` file from `data/test_questionnaires/`
2. The form auto-populates with the YAML values
3. If no files exist, create one in `data/test_questionnaires/` (see Appendix A for YAML format)

### 2.3 Edit Applicant Data

Edit any field on the form:

| Field Group | Fields |
|-------------|--------|
| **Applicant Data** | Full Name, Date of Birth, Gender, Residency, Address |
| **Occupation** | Occupation, Employer, Years in Occupation, Annual Income |
| **Health** | Height, Weight, Smoker Status, Medications toggle, Alcohol toggle |
| **Benefits** | Benefit Types (multi-select), Sum Insured (Death, TPD, Trauma) |

### 2.4 Agent Selection

- Check/uncheck agents in the "Select agents to run" multi-select
- Options: MedicalAgent, FinancialAgent, ComplianceAgent
- Default: all three selected

**Test:** Deselect an agent (e.g., uncheck ComplianceAgent) and run the pipeline — verify only selected agents appear in results.

### 2.5 Run the Pipeline

1. Click **"Run Underwriting Pipeline"** button
2. Watch the spinner: "Running multi-agent pipeline..."
3. On completion:
   - Screen shows "Pipeline complete!"
   - Pipeline summary appears in a code block
   - Agent assessments display in expanders
   - Compliance Agent is shown separately as "Observer — Informational Only"
   - Debate log shows rounds (if any disagreement occurred)
4. **Auto-navigation:** You are automatically redirected to the **Results & Debate** page

**Expected:**
- Pipeline completes without errors (blue spinner → green success)
- Each agent shows: Risk Tier, Recommendation, Confidence percentage, Flags
- Non-standard conditions highlighted with orange/red badges
- If all agents agree: "No debate needed — all agents reached consensus"
- If agents disagree: debate rounds shown

---

## 3. Testing the Results Display

### 3.1 Hero Card

After running the pipeline, the **Results & Debate** page shows a colored hero card at the top:

| Decision | Card Color |
|----------|------------|
| Standard Offer | Green |
| Offer with Loading/Exclusion | Orange |
| Refer to Manual Underwriting | Blue |
| Decline | Red |

The card shows:
- Decision icon + text
- Applicant name + data source label
- Agent consensus status (all standard vs. agents flagged non-standard)

**Test:** Run different questionnaires to see different hero card colors.

### 3.2 Application Details Expander

Click **"Application Details"** to expand:
- Summary metrics row: Name, Age, BMI, Occupation Class
- Nested expanders for each section: Personal, Cover Details, Occupation & Income, Health, Lifestyle, Financial
- List-type fields (medical conditions, family history, hazardous pursuits) show as sub-expanders with item counts

**Test:** Verify all submitted fields appear correctly. Verify list items show properly in sub-expanders.

### 3.3 Full Assessment Expander

Click **"Full Assessment"** to expand (open by default):
- **Decision source** caption
- **Decision reasoning** (info box)
- **Non-Standard Conditions** — color-coded badges for each agent with non-standard risk tier
- **Agent Assessments** — individual expanders per agent showing:
  - Recommendation
  - Confidence score (percentage)
  - Reasoning summary
  - Matched Rules (with severity badges, rule IDs, descriptions)
  - Loading Range (if applicable)
  - Additional Evidence Required (if applicable)
- **AI Summary** — plain-language explanation (LLM-generated if available, deterministic fallback otherwise)
- **Consensus status** — green checkmark or orange warning
- **Risk Flags** and **Additional Evidence Required** sections (if any)

**Test:** Verify all agent assessments appear. Check that Compliance Agent is labeled "(Observer — Informational Only)" and shows "does NOT influence the underwriting decision."

---

## 4. Testing the Debate & Chat

### 4.1 The Debate Log Expander

Click **"Debate Log"** to expand:

**When a conversation is loaded:**
- Header shows: applicant name, created timestamp, debate rounds count, final decision
- If evidence was submitted: shows "Decision re-evaluated" or "Evidence submitted (no tier change)" indicators
- **"Full Debate Log & Chat History"** sub-expander shows all messages as styled chat bubbles:
  - Agent messages: colored bubbles with agent initials and emoji
  - User messages: blue bubbles, right-aligned
  - System messages: grey bubbles, centered
  - Risk tier updates shown as badges below agent messages

**When no conversation is loaded:**
- Info message: "Select a conversation, submit a new application, or run the Test Questionnaire pipeline."
- If test results exist: "Recent Application" section with "View in Chat" button

### 4.2 Application History

The "Application History" expander at the top of Results & Debate shows saved conversations:

- Each conversation shows: applicant name, timestamp, status, decision
- **Load** button: loads that conversation for chat interaction
- **Delete** button: removes the conversation
- **New Application** button: clears current conversation

**Test:**
1. Run a test questionnaire → conversation is auto-created
2. Verify it appears in Application History
3. Click **Load** → debate log populates with messages
4. Click **Delete** → conversation is removed

---

## 5. Testing Evidence & Chat Interaction

### 5.1 Asking Questions (Chat Mode)

When a conversation is loaded, a chat input appears at the bottom of the Debate Log:

1. **Leave "Submit as evidence for re-evaluation" UNCHECKED**
2. Type a question in the chat input (e.g., "What flagged the medical risk?")
3. Press Enter

**Expected:**
- Each agent processes the question via `handle_user_message()` and responds
- Agent replies appear as new chat bubbles
- Risk tiers do NOT change (questions are informational)
- No re-evaluation occurs

### 5.2 Submitting New Evidence (Evidence Mode)

This is the key feature — it tests the full re-evaluation loop.

1. **CHECK the "Submit as evidence for re-evaluation" checkbox**
2. Type evidence in the chat input
   - Example 1 (test tier change): "The applicant has been smoke-free for 5 more years than previously stated. Please re-evaluate."
   - Example 2 (test no change): "The applicant's height was actually 170cm, not 165cm. Please re-evaluate."
   - Example 3 (test evidence flag): "The applicant submits a medical report from Dr. Smith showing normal blood pressure readings for the past 2 years."
3. Press Enter

**Expected behavior:**

| Scenario | What Happens |
|----------|-------------|
| Evidence changes an agent's risk tier | System message: "New evidence triggered re-evaluation. Debate round initiated." → Each agent generates rebuttals → Final decision recalculated → Hero card updates |
| Evidence does NOT change any risk tier | Conversation shows "Evidence submitted (no tier change)" indicator. No debate round triggered. |
| Evidence mode is ON | Message is treated as evidence (not question). `evidence_mode` flag set on conversation. |

**Verify after evidence submission:**
1. Chat bubbles show agent responses with their updated reasoning
2. If tiers changed: rebuttal messages appear from each agent challenging others' assessments
3. **Hero card at the top of the page updates** — decision text, color, and status may change
4. "Decision re-evaluated based on user-provided evidence" indicator appears
5. **Full Assessment** expander reflects updated agent assessments
6. **AI Summary** regenerates to reflect the new decision

### 5.3 Step-by-Step Evidence Test Walkthrough

1. **Load a test questionnaire** with a risky applicant (e.g., smoker with high BMI, or hazardous occupation)
2. **Run the pipeline** → observe the initial decision (likely "Offer with Loading" or "Refer")
3. **Navigate to Results & Debate** → note the hero card decision
4. **Check "Submit as evidence for re-evaluation"**
5. **Enter exculpatory evidence** (information that should reduce risk):
   ```
   The applicant has completed a comprehensive medical exam by Dr. Chen showing:
   - Normal ECG and stress test results
   - Blood pressure 120/80
   - Cholesterol within normal range
   - BMI is now 28 (down from 32) due to a supervised weight management program
   - Smoking cessation confirmed for 24 months with nicotine replacement
   ```
6. **Observe** the re-evaluation cycle:
   - Each agent processes the evidence
   - System message about re-evaluation
   - Agent rebuttals if tiers changed
   - Hero card updates
7. **Verify** the decision may shift toward "Standard Offer" or lower loading

---

## 6. Testing Result Changes and Loading

### 6.1 Initial Load (Pipeline → Results)

When the pipeline completes:
- Results are stored in `st.session_state.test_q_results`
- A conversation is auto-created in the ConversationStore
- The page auto-navigates to Results & Debate
- The hero card, assessments, and debate log render from the conversation

### 6.2 Loading Saved Conversations

From **Application History**:
1. Click **Load** on any saved conversation
2. The conversation loads into `st.session_state.chat_conversation`
3. Results re-render from the loaded conversation data

### 6.3 Watching Results Update After Evidence

1. Load a conversation with a non-standard decision
2. Open **Full Assessment** and **Debate Log** expanders
3. Submit evidence (checkbox ON) that should change risk tiers
4. After `st.rerun()`, observe:
   - New chat bubbles appear in the debate log
   - Hero card changes color/text if decision changed
   - Agent assessment expanders update with new tiers/flags
   - AI Summary regenerates

**Edge case test:** Submit evidence that shouldn't change anything → verify "Evidence submitted (no tier change)" indicator appears without triggering re-evaluation.

---

## 7. Testing Across Pages

### 7.1 Navigation Flow

1. Start on **Test Questionnaire** → select file → edit → run pipeline
2. Auto-redirect to **Results & Debate** → view results
3. Switch to **Questionnaire** → fill form manually → submit
4. Return to **Results & Debate** → verify questionnaire results appear (different source label)
5. Switch to **Rules Reference** → browse rules → verify rule IDs match those shown in assessments
6. Switch to **Compliance Framework** → verify regulatory mappings

### 7.2 State Persistence

- Running a new pipeline does NOT delete previous conversations
- Application History preserves all saved conversations
- Switching pages preserves session state

---

## 8. Edge Cases & Error States

| Scenario | Expected Behavior |
|----------|-------------------|
| No questionnaire YAML files exist | Warning message: "No questionnaire files found..." |
| Submit form with no benefit types selected | Error: "At least one benefit type is required." |
| Deselect all agents in Test Questionnaire | Pipeline should handle gracefully (may default to all agents) |
| Chat input with no conversation loaded | Nothing happens (chat input hidden) |
| LLM backend unreachable | Agent responses fall back to deterministic mode. AI Summary uses deterministic fallback text. |
| Config file missing | LLM client gracefully returns `None`; everything works deterministically |
| Delete currently loaded conversation | Conversation cleared; page shows empty state |

---

## 9. Quick Smoke Test Checklist

- [ ] App launches: `streamlit run app.py`
- [ ] All 5 sidebar navigation items work
- [ ] Questionnaire form renders all sections
- [ ] Conditional fields appear/disappear correctly
- [ ] Validation errors show on incomplete submission
- [ ] Test Questionnaire loads YAML files
- [ ] Editing test questionnaire fields works
- [ ] Pipeline runs and shows results
- [ ] Hero card displays with correct color/decision
- [ ] Agent assessments show in expanders
- [ ] Compliance Agent shows as observer
- [ ] Debate log renders chat bubbles
- [ ] Chat input sends questions (evidence mode OFF)
- [ ] Chat input triggers re-evaluation (evidence mode ON)
- [ ] Hero card updates after evidence changes decision
- [ ] Application History loads/deletes conversations
- [ ] Rules Reference shows all rules
- [ ] Compliance Framework shows regulatory mappings
- [ ] No Python tracebacks in the Streamlit UI

---

## Appendix A: Creating Test Questionnaire YAML Files

Create files in `data/test_questionnaires/` with this format:

```yaml
full_name: "John Smith"
date_of_birth: "1985-06-15"
gender: "Male"
residency_status: "Australian Citizen"
contact_address: "42 Example Street, Sydney NSW 2000"

occupation: "Construction Worker"
employer_name: "BuildCorp Pty Ltd"
years_in_occupation: 8.0
annual_income: 85000.0

height_cm: 178.0
weight_kg: 95.0
smoker_status: "Current"
taking_medications: false
consumes_alcohol: true

benefit_types:
  - "Death"
  - "TPD"
sum_insured_death: 750000
sum_insured_tpd: 500000
sum_insured_trauma: null
```

**Suggested test profiles:**

| Profile | Key Traits | Expected Outcome |
|---------|-----------|-----------------|
| `healthy_applicant.yaml` | Non-smoker, healthy BMI, office job, standard cover | Standard Offer |
| `risky_smoker.yaml` | Current smoker, high BMI, hazardous occupation | Offer with Loading |
| `decline_case.yaml` | Multiple severe conditions, bankruptcy, criminal history | Decline or Refer |
| `borderline.yaml` | Moderate conditions, one agent flags | Refer or Loading |

## Appendix B: Debugging

If something isn't working:

1. Check the Streamlit terminal for Python tracebacks
2. Verify `config.yaml` exists (even with default/empty content)
3. Verify rule JSON files exist under `rules/death/`
4. Check `data/test_questionnaires/` has at least one `.yaml` file
5. Restart the Streamlit server: Ctrl+C then `streamlit run app.py` again
