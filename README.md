# Support Ticket Triage & Escalation Copilot

**An end-to-end analytics and automation prototype for B2B customer service: find out which support work can be automated, quantify the business case, and demo an AI-style triage copilot.**

> All data is **synthetic** (fictional company "Kelvara Components", a B2B HVAC parts maker). No real customer data is used.

**Live demo:** _add your Streamlit Community Cloud link here_

<!-- Add 2-3 screenshots: docs/overview.png, docs/opportunities.png, docs/triage.png -->
<!-- ![Overview](docs/overview.png) -->

---

## The problem

A commercial support team handles thousands of tickets a year. Many are repetitive ("Where is my order?"), while others need a specialist. Without data, it is hard to say **what to automate first, how much it is worth, and where the process is failing customers**.

## What this project does

1. **Analyses** 12 months of support tickets with pandas (volume, response times, SLA breaches, reopen rates, CSAT).
2. **Scores automation opportunities** for each ticket type on value (hours saved) and feasibility (how repeatable the answer is), and recommends a solution pattern for each.
3. **Models the ROI** with adjustable assumptions (ticket volume, adoption, agent cost, implementation cost).
4. **Triages new tickets**: classifies raw customer messages, suggests a priority, decides whether to auto-reply, draft for review, or hand to a human, recommends escalation, and drafts a reply.
5. **Lays out a rollout plan** from prototype to production, including pilot, measurement, and guardrails.

## Key findings (on the synthetic data)

| Finding | Detail |
|---|---|
| Biggest, easiest target | "Order status" is ~29 % of tickets and takes a median of 4 minutes each |
| Coverage gap | Southern Europe's median first response (~142 min) is about 2x slower than other regions, with a 38 % SLA breach rate |
| Channel effect | Email first response (133 min median) is 5x slower than chat (26 min) |
| Peak load | Mondays carry ~40 % more tickets than other weekdays |
| Needs humans | Technical questions are reopened 21 % of the time and have among the lowest CSAT |
| Modelled saving | Automating the top 3 ticket types: ~183 hours/month, ~77k € per year, ~4 month payback at default assumptions |

The saving is a **model built on stated assumptions**, not a measured result (see Limitations).

## How it works

```
data/tickets.csv  →  analysis.py (pandas)  →  app.py (Streamlit dashboard)
                                    ↑
                triage.py (rules-based triage copilot)
```

In production, `load_data()` would read from a ticketing system API or data warehouse instead of a CSV; nothing else would need to change.

**Triage copilot output for each ticket:** category, confidence, priority, automation level (`Auto-reply`, `Draft for review`, `Human agent`), escalation recommendation, and a draft reply. When confidence is low, the ticket always goes to a human agent (safe fallback).

## Project structure

```
├── app.py               # Streamlit app (6 pages)
├── analysis.py          # pandas analysis and automation scoring
├── triage.py            # rules-based triage copilot
├── generate_data.py     # seeded synthetic data generator
├── data/
│   ├── tickets.csv          # 8,000 synthetic tickets
│   └── incoming_sample.csv  # 100 unlabelled messages for the batch triage demo
├── .streamlit/config.toml   # theme
└── requirements.txt
```

## Getting started

**Requirements:** Python 3.10+ and the packages in `requirements.txt` (pandas, numpy, streamlit, plotly).

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. install dependencies
python -m pip install -r requirements.txt

# 3. (optional) regenerate the synthetic data; the CSVs are already included
python generate_data.py

# 4. run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Run the commands from the project folder so the data paths resolve.

You can also run the analysis in the terminal with `python analysis.py` and the triage self-test with `python triage.py`.

## The data

`generate_data.py` creates 8,000 tickets over 12 months with a fixed random seed, so results are reproducible. Realistic patterns are built in (log-normal handle and resolution times, Monday and cooling-season peaks, slower response in one region and on email, CSAT that falls with slow resolution, SLA breaches, and reopens). Numbers in the CSV use the international format; the app displays them in European notation (e.g. `76.848 €`).

CSAT is on a **1 to 5 scale**. About 35 % of tickets have no rating, as in real surveys.

## Limitations (please read)

- **Synthetic data.** Findings demonstrate the method, not real-world results.
- **Circular accuracy.** The triage rules were written for the same message templates that generated the data, so the ~95 % category accuracy overstates real-world performance. Real emails (other languages, mixed topics, vague text) would score lower; the 100-message sample includes a few such cases, which correctly fall back to a human agent.
- **Assumptions drive the ROI.** Ticket volume (60,000 per year), adoption (60 %), agent cost (35 €/h), implementation cost (25,000 €), and the repeatability score of each ticket type are assumptions. A pilot would replace them with measured values.
- **Rules, not machine learning.** Triage uses keyword rules to keep the project simple, local, and explainable.

## From prototype to production

1. **Pilot (6 to 8 weeks):** run in suggestion mode for the top candidates (order status, returns); agents accept or edit drafts, nothing is sent automatically.
2. **Measure:** compare against a control group on handle time, first response, SLA breaches, reopens, CSAT, and suggestion acceptance.
3. **Scale:** enable auto-reply only where acceptance is high and CSAT holds; feed agent corrections back as training data; evaluate a trained classifier or LLM service.
4. **Guardrails:** human fallback for low confidence, no auto-replies on high-priority or technical cases, weekly quality sampling, GDPR review before using real data.

The same pattern (analyse, score, assist, pilot, measure) applies to **sales operations**: quote follow-up, order-intake emails, and lead routing.

## Tech stack

Python · pandas · NumPy · Streamlit · Plotly

## Skills demonstrated

Data analysis and data modelling · KPI design (SLA, CSAT, handle time) · process automation opportunity assessment · business case and ROI modelling · rules-based NLP triage · dashboarding · stakeholder-ready storytelling · rollout planning and change management
