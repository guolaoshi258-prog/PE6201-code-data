# PE6201 Lead Ranking — Code and Data

A reproducible coursework prototype for ranking sales leads under a daily 20% contact-capacity limit. Includes logistic regression, rule baselines, template explanations, five read-only tools, and agent/routed/workflow experiments.

## Run

Use Python 3.12. From this repository root:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/01_data_and_leakage.py
python scripts/02_method_comparison.py
python scripts/03_review_explain_cost.py
python scripts/04_agent_experiment.py
```

Before running script 1, download `olist_closed_deals_dataset.csv` from the [original Kaggle dataset](https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist) and place it in `data/raw/`. This public repository omits that raw table because it contains seller/SDR identifiers and detailed business attributes. The included processed tables let scripts 2–4 run without that download. Scripts 1–3 run locally. Script 4 uses the included 240 cached experimental runs; with this cache complete, it makes no new model calls. New experiments require your own `OPENROUTER_API_KEY` environment variable and incur API charges. Never commit a key. Generated output files are written to the current working directory or `outputs/` as specified by each script.

## Files

- `scripts/`: four executable experiments: data/leakage, method comparison, explanations/cost/monitoring, and LLM evaluation.
- `src/lead_tools.py`: scoring, daily capacity, explanations and five read-only tools. Evaluation-period outcomes are excluded from tool-visible records.
- `src/lead_agent.py`: agent loop, routed and fixed workflows, guard and deterministic fallback.
- `src/agent_eval.py`: task builder and automatic checks.
- `data/raw/`: public Olist marketing-qualified lead table; obtain the closed-deals table from its source as described above.
- `data/processed/`: derived modelling tables and fixed test requests.
- `outputs/agent_runs.jsonl`: cached model experiment records, not human ratings.

## Data attribution

Source: [Marketing Funnel by Olist](https://www.kaggle.com/olistbr/marketing-funnel-olist), licensed CC BY-NC-SA 4.0. Source data and its derivatives retain that licence. The closed-deals table is used for labels and leakage evaluation; ranking uses channel, landing page and weekday. The modelling window is January–May 2018; the forward evaluation trains on January–April and tests on May.

## Scope

This public repository contains code and research data. Personal reports, student identifiers, presentation video, and completed human/AI review sheets are excluded. Optional rating-sheet code can create blank review templates locally; no completed ratings are included. Scripts are the executable equivalents of the coursework notebooks. Code was developed with AI assistance.
