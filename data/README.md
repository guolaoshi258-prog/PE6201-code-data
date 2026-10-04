# Data provenance

Source: Marketing Funnel by Olist, https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist . The publisher describes it as anonymized public research data. Dataset licence: CC BY-NC-SA 4.0; source data and derivatives retain these terms.

- `raw/olist_marketing_qualified_leads_dataset.csv`: 8,000 public lead records (hashed lead/page IDs, date, channel).
- `processed/leads_clean.csv`: 5,998 January–May 2018 leads; channel, page, weekday, date, hashed lead ID, binary conversion label.
- `processed/leads_all.csv`: all 8,000 leads with the same derived fields for comparison.
- `processed/agent_tasks.json`: 20 fixed research requests for automatic tool-use evaluation.

The raw closed-deals table is intentionally not distributed here. Download `olist_closed_deals_dataset.csv` from the source and place it under `data/raw/` to reproduce script 1, including leakage checks. Scripts 2–4 can use the included processed data. No personal report, student identifier or human rating sheet is included.
