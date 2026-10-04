"""Executable project experiment: 01_data_and_leakage. Run from the repository root."""

try:
    display
except NameError:
    display = print




import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)

MQL_FILE = "olist_marketing_qualified_leads_dataset.csv"
DEALS_FILE = "olist_closed_deals_dataset.csv"

def find_file(name):
    """Find a data file whether the notebook runs from the repo root, from notebooks/, or in Colab."""
    for d in [".", "data/raw", "data/processed", "../data/raw", "../data/processed", "outputs", "../outputs"]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return name

MQL_FILE, DEALS_FILE = find_file(MQL_FILE), find_file(DEALS_FILE)
if not (os.path.exists(MQL_FILE) and os.path.exists(DEALS_FILE)):
    try:
        from google.colab import files   # only available inside Colab
        print("Please upload the two Olist CSV files.")
        files.upload()
    except ImportError:
        raise FileNotFoundError("Put the two CSV files in the same folder as this notebook.")

mql = pd.read_csv(MQL_FILE, parse_dates=["first_contact_date"])
deals = pd.read_csv(DEALS_FILE, parse_dates=["won_date"])

print("MQL table:  ", mql.shape)
print("Deals table:", deals.shape)
display(mql.head())
display(deals.head())


mql["converted"] = mql["mql_id"].isin(deals["mql_id"]).astype(int)

n_leads = len(mql)
n_conv = int(mql["converted"].sum())
print(f"Leads: {n_leads:,}   Conversions: {n_conv:,}   Conversion rate: {n_conv / n_leads:.1%}")

print("Duplicate mql_id in MQL table:", mql["mql_id"].duplicated().sum())
print("Deals with no matching lead:  ", (~deals["mql_id"].isin(mql["mql_id"])).sum())

top20 = int(round(0.2 * n_leads))
print(f"\nTop 20% of all leads = {top20:,} leads.")
print(f"If a 20% hold-out test set is used, it holds about {int(0.2 * n_conv)} conversions — "
      "small enough that one split is noisy.")


merged = mql.merge(deals, on="mql_id", how="left")
deal_cols = [c for c in deals.columns if c != "mql_id"]

rows = []
for c in deal_cols:
    filled = merged[c].notna().astype(int)
    rows.append({
        "column": c,
        "filled_for_converted": f"{filled[merged.converted == 1].mean():.0%}",
        "filled_for_not_converted": f"{filled[merged.converted == 0].mean():.0%}",
        "ROC-AUC of 'is filled'": round(roc_auc_score(merged["converted"], filled), 3),
    })
leak_table = pd.DataFrame(rows)
display(leak_table)
print("An AUC near 1.0 means the field alone gives away the answer -> excluded from every model.")


mql["origin"] = mql["origin"].fillna("missing")
mql["contact_month"] = mql["first_contact_date"].dt.to_period("M").astype(str)
mql["contact_weekday"] = mql["first_contact_date"].dt.day_name()

surviving = pd.DataFrame({
    "feature": ["origin", "landing_page_id", "contact_weekday", "contact_month"],
    "type": ["categorical", "categorical (ID)", "categorical", "time"],
    "distinct values": [mql["origin"].nunique(), mql["landing_page_id"].nunique(),
                        mql["contact_weekday"].nunique(), mql["contact_month"].nunique()],
    "note": ["marketing channel the lead came from",
             "page the lead signed up on — many rare values",
             "day of first contact",
             "not used as a predictor — see sections 6–7"],
})
display(surviving)


def rate_table(col, min_leads=0):
    t = (mql.groupby(col)["converted"]
            .agg(leads="count", conversions="sum")
            .assign(rate=lambda d: (d.conversions / d.leads).round(3))
            .sort_values("leads", ascending=False))
    return t[t.leads >= min_leads]

print("Conversion by origin:")
display(rate_table("origin"))

print("Conversion by weekday:")
display(rate_table("contact_weekday"))

lp = rate_table("landing_page_id")
print(f"Landing pages: {len(lp)} distinct. "
      f"{(lp.leads < 20).sum()} have fewer than 20 leads "
      f"(covering {lp.loc[lp.leads < 20, 'leads'].sum():,} leads) — too rare to estimate a rate reliably.")
display(lp.head(10))


conv_by_month = rate_table("contact_month").sort_index()
display(conv_by_month)

lag = (merged.loc[merged.converted == 1, "won_date"] - merged.loc[merged.converted == 1, "first_contact_date"]).dt.days
print(f"Days from first contact to won: median {lag.median():.0f}, 90th percentile {lag.quantile(0.9):.0f}")
print("Last first-contact date:", mql["first_contact_date"].max().date(),
      "| Last won date:", deals["won_date"].max().date())

fig, ax = plt.subplots(figsize=(9, 3.5))
conv_by_month["rate"].plot(kind="bar", ax=ax, color="#4C72B0")
ax.set_ylabel("conversion rate"); ax.set_title("Conversion rate by month of first contact")
plt.tight_layout(); plt.show()


print("won_date range:", deals["won_date"].min().date(), "to", deals["won_date"].max().date())
display(deals["won_date"].dt.to_period("M").value_counts().sort_index().rename("deals won").to_frame().T)

conv = merged[merged.converted == 1].copy()
conv["days_to_win"] = (conv["won_date"] - conv["first_contact_date"]).dt.days
by_cohort = conv.groupby(conv["first_contact_date"].dt.to_period("M"))["days_to_win"].agg(["count", "min", "median"])
display(by_cohort)
print("Deals with won_date before first contact (data quirk):", int((conv.days_to_win < 0).sum()))


window = (mql["first_contact_date"] >= "2018-01-01") & (mql["first_contact_date"] < "2018-06-01")
mql_w = mql[window]
print(f"Modelling window Jan–May 2018: {len(mql_w):,} leads, {int(mql_w.converted.sum())} conversions, "
      f"rate {mql_w.converted.mean():.1%}")
print(f"Dropped (2017 cohort): {int((~window).sum()):,} leads")
print(f"A 20% test set from this window holds about {int(0.2 * mql_w.converted.sum())} conversions.")

print("\nConversion by origin inside the window:")
display(mql_w.groupby("origin")["converted"].agg(leads="count", conversions="sum")
        .assign(rate=lambda d: (d.conversions / d.leads).round(3)).sort_values("leads", ascending=False))


clean = mql_w[["mql_id", "first_contact_date", "origin", "landing_page_id", "contact_weekday", "converted"]]
clean.to_csv("leads_clean.csv", index=False)
print("Saved leads_clean.csv:", clean.shape)

mql[clean.columns].to_csv("leads_all.csv", index=False)
print("Saved leads_all.csv (comparison only):", mql[clean.columns].shape)

