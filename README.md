# Lead-Ops

> **OpenEnv-compliant RL environment for autonomous sales lead enrichment, MEDDIC qualification, and strategic routing.**

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.11-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()

---

## 🌐 Environment Description

Lead-Ops models the end-to-end lifecycle of a B2B sales lead as a reinforcement learning environment. An autonomous agent observes raw CRM data, takes actions to enrich and qualify leads, and receives reward signals based on data quality, MEDDIC coverage, and routing accuracy.

The environment uses a **real SQLite database** (no mock data), **Tavily API** for live web search, and **programmatic grading** (no LLM-as-a-judge).

### Environment Motivation

Sales Operations leaders face three critical challenges:
1. **Manual Lead Enrichment** — SDRs spend 30–40% of their time researching leads instead of selling
2. **Inconsistent MEDDIC Qualification** — Scoring varies wildly by rep experience and judgment
3. **Suboptimal Routing** — Round-robin assignment ignores deal fit, territory expertise, and win rates

Lead-Ops provides a controlled simulation where AI agents can learn to automate these tasks with measurable, deterministic performance metrics.

---

## 🎯 Tasks

| Task ID | Difficulty | Description | Max Steps |
|---|---|---|---|
| `enrich_lead` | Easy | Enrich a raw CRM lead with firmographic and contact intelligence | 10 |
| `meddic_qualify` | Medium | Score a lead against 6 MEDDIC pillars using interaction logs | 12 |
| `strategic_route` | Hard | Enrich + Qualify + Route to the optimal Account Executive | 15 |

---

## 📐 Observation Space

The agent observes a `LeadObservation` object representing the current CRM lead state:

| Field | Type | Description |
|---|---|---|
| `lead_id` | `str` | Unique lead identifier |
| `company_name` | `str` | Company name (may contain typos) |
| `industry` | `str?` | Industry vertical |
| `annual_revenue` | `float?` | Annual revenue (USD) |
| `employee_count` | `int?` | Number of employees |
| `website` | `str?` | Company website URL |
| `contact_name` | `str?` | Primary contact name |
| `contact_title` | `str?` | Contact job title |
| `contact_email` | `str?` | Contact email |
| `contact_linkedin` | `str?` | LinkedIn profile URL |
| `tech_stack` | `list[str]` | Known technologies used |
| `lead_source` | `LeadSource?` | Acquisition channel |
| `meddic_scores` | `MEDDICScores?` | MEDDIC qualification scores |
| `routing_result` | `RoutingResult?` | Routing assignment |
| `available_actions` | `list[AvailableAction]` | Actions the agent can take (prevents hallucinations) |

---

## 🎯 Action Space

Actions are a **discriminated union** (Search vs. Update) with mandatory **Chain-of-Thought** reasoning:

### Search Actions (read-only)
```json
{
  "action_type": "search",
  "thought": "I need to find the company's revenue and employee count.",
  "tool_name": "tavily_search",
  "query": "Stripe annual revenue 2024",
  "filters": {"max_results": 3},
  "confidence": 0.85
}
```

### Update Actions (write to CRM)
```json
{
  "action_type": "update",
  "thought": "Based on search results, updating revenue to $14B.",
  "tool_name": "update_lead",
  "field_updates": {"annual_revenue": 14000000000, "industry": "FinTech"},
  "reason": "Confirmed via Tavily search and public filings",
  "confidence": 0.9
}
```

### Available Tools

| Tool | Type | Description |
|---|---|---|
| `tavily_search` | Search | Real-time web search via Tavily API |
| `crm_lookup` | Search | Look up CRM records, accounts, and routing rules |
| `linkedin_enrich` | Search | Search LinkedIn for contact data |
| `read_logs` | Search | Read interaction logs (emails/calls) for MEDDIC signals |
| `update_lead` | Update | Update CRM lead fields with enriched data |
| `score_meddic` | Update | Set MEDDIC pillar scores (0.0–1.0) |
| `route_to_ae` | Update | Route lead to an Account Executive |
| `disqualify` | Update | Disqualify a lead from the pipeline |

---

## 🏆 Reward Function

Rewards are scalar values in `[0.0, 1.0]` with component breakdowns:

### Task 1 — Enrichment (Easy)
- **LinkedIn URL Match** (50%) — Fuzzy match against ground truth
- **Job Title Match** (50%) — Fuzzy match against ground truth
- **Step Penalty** (-0.02 per step)
- **Destructive Penalty** (-0.50 for deleting data)

### Task 2 — MEDDIC Qualification (Medium)
- **Economic Buyer Accuracy** (60%) — Agent score vs. interaction log signals
- **Identify Pain Accuracy** (40%) — Agent score vs. interaction log signals
- **Step Penalty** (-0.02 per step)

### Task 3 — Strategic Routing (Hard)
- **Enrichment Accuracy** (30%) — Revenue within 10% of ground truth
- **MEDDIC Completeness** (30%) — Fraction of 6 pillars scored
- **Routing Accuracy** (40%) — Correct AE assignment
- **Step Penalty** (-0.02 per step)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- A Tavily API key ([tavily.com](https://tavily.com))
- An OpenAI-compatible LLM endpoint

### Setup

```bash
# 1. Clone the repository
git clone <repo-url> && cd Lead-Ops

# 2. Create virtual environment
python -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your real API keys

# 5. Initialize and seed the database
python database_init.py
python scripts/dirty_seeder.py
python scripts/interaction_generator.py

# 6. Validate the setup
python check_compliance.py
```

### Run the Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Run Inference

```bash
# Set your model config
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o
export OPENAI_API_KEY=your-key

# Run against local server
python inference.py
```

### Docker

```bash
# Build
docker build -t leadops .

# Run
docker run -p 7860:7860 \
  -e HF_TOKEN=your-token \
  -e OPENAI_API_KEY=your-openai-key \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e TAVILY_API_KEY=your-key \
  leadops

# Validate endpoints against local container
./validate-submission.sh http://localhost:7860
```

---

## 📁 Project Structure

```
Lead-Ops/
├── app.py                    # Root runtime wrapper (binds to PORT=7860)
├── openenv.yaml              # OpenEnv manifest (port 7860)
├── models.py                 # Pydantic V2 data models (Action union, Observation)
├── config.py                 # Lazy-loaded environment config
├── db_models.py              # SQLAlchemy ORM (leads, accounts, logs, cache)
├── database_init.py          # Database initializer
├── session_manager.py        # Per-episode session isolation (/tmp clones)
├── environment.py            # LeadOpsEnv: reset(), step(), state()
├── actions.py                # Tool implementations (Tavily, CRM, MEDDIC)
├── grader.py                 # Programmatic graders (3 tasks)
├── inference.py              # Baseline inference script ([START/STEP/END])
├── validate-submission.sh    # Endpoint validator for local/HF URL
├── Dockerfile                # HF Spaces container (python:3.11-slim)
├── check_compliance.py       # OpenEnv compliance validator
├── requirements.txt          # Pinned dependencies
├── README.md                 # This file
├── .env.example              # Environment variable template
├── .gitignore                # Git ignore rules
├── data/
│   ├── solutions.json        # Ground truth for grading
│   ├── territory_routing.json # Routing rules (segment/territory/AE)
│   └── meddic_playbook.json  # MEDDIC framework reference
├── scripts/
│   ├── dirty_seeder.py       # Populates 50 dirty CRM leads
│   └── interaction_generator.py # Generates MEDDIC-signal email threads
├── server/
│   └── app.py                # FastAPI entrypoint
├── utils/
│   └── score_report.py       # Human-readable score reports
└── test_penalties.py         # Destructive action penalty tests
```

---

## 🗺️ Roadmap

- [x] **Phase 1:** Blueprint — manifest, models, config, compliance
- [x] **Phase 2:** Data Infrastructure — SQLite, seeder, interaction logs, sessions
- [x] **Phase 3:** API Server — FastAPI, tools, environment loop
- [x] **Phase 4:** Graders — programmatic scoring for all 3 tasks
- [x] **Phase 5:** Inference & Deployment — inference.py, Dockerfile
- [x] **Phase 6:** HF Space Hardening — health checks, wrapper entrypoint, validator script, Docker optimization

---

## 📄 License

MIT
