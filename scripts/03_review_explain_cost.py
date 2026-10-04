"""Executable project experiment: 03_review_explain_cost. Run from the repository root."""

try:
    display
except NameError:
    display = print




import os, re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 200)

FEATURES = ["origin", "landing_page_id", "contact_weekday"]
CAPACITY = 0.20
MIN_PAGE_LEADS = 20
SEED = 42

def find_file(name):
    """Find a data file whether the notebook runs from the repo root, from notebooks/, or in Colab."""
    for d in [".", "data/raw", "data/processed", "../data/raw", "../data/processed", "outputs", "../outputs"]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return name

df = pd.read_csv(find_file("leads_clean.csv"), parse_dates=["first_contact_date"])
train = df[df.first_contact_date < "2018-05-01"].copy()
test = df[df.first_contact_date >= "2018-05-01"].copy()
print(f"Train Jan–Apr: {len(train):,} leads, {int(train.converted.sum())} conversions")
print(f"Test  May:     {len(test):,} leads, {int(test.converted.sum())} conversions")

def make_logreg():
    return make_pipeline(
        OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_PAGE_LEADS),
        LogisticRegression(max_iter=2000))

model = make_logreg().fit(train[FEATURES], train["converted"])
test["score"] = model.predict_proba(test[FEATURES])[:, 1]
print(f"ROC-AUC on May: {roc_auc_score(test.converted, test.score):.3f}")


page_counts = train["landing_page_id"].value_counts()
test["page_history"] = test["landing_page_id"].map(page_counts).fillna(0).astype(int)

rng = np.random.default_rng(SEED)
test["called"] = False
for _, g in test.groupby(test.first_contact_date.dt.date):
    order = g.index[np.lexsort((rng.random(len(g)), -g["score"].values))]
    test.loc[order[:int(round(CAPACITY * len(g)))], "called"] = True
print(f"May: {int(test.called.sum())} call slots over {test.first_contact_date.dt.date.nunique()} days")

rows = []
for T in [1, 5, 10, 20, 50]:
    flag = test["page_history"] < T
    def miss_rate(mask):
        conv = test[mask & (test.converted == 1)]
        return (~conv["called"]).mean() if len(conv) else np.nan
    rows.append({
        "threshold T (training leads on page)": T,
        "flagged leads": int(flag.sum()),
        "flag rate": f"{flag.mean():.0%}",
        "conversions flagged": int(test.loc[flag, "converted"].sum()),
        "conv. rate flagged": f"{test.loc[flag, 'converted'].mean():.1%}",
        "conv. rate not flagged": f"{test.loc[~flag, 'converted'].mean():.1%}",
        "AUC flagged": round(roc_auc_score(test.loc[flag, "converted"], test.loc[flag, "score"]), 3)
                       if test.loc[flag, "converted"].nunique() == 2 else None,
        "AUC not flagged": round(roc_auc_score(test.loc[~flag, "converted"], test.loc[~flag, "score"]), 3),
        "converters missed, flagged": f"{miss_rate(flag):.0%}",
        "converters missed, not flagged": f"{miss_rate(~flag):.0%}",
    })
flag_table = pd.DataFrame(rows).set_index("threshold T (training leads on page)")
display(flag_table)


T_REVIEW = 20
test["review"] = test["page_history"] < T_REVIEW
print(f"May: {test.review.sum()} of {len(test)} leads flagged for review ({test.review.mean():.0%}); "
      f"they contain {int(test.loc[test.review, 'converted'].sum())} of {int(test.converted.sum())} conversions.")
lane = test[test.review & ~test.called].groupby(test.first_contact_date.dt.date).size()
slots = test.groupby(test.first_contact_date.dt.date)["called"].sum()
print(f"Review lane (flagged and not called): median {lane.median():.0f} leads a day vs {slots.median():.0f} call slots; "
      f"flagged leads that still got a call slot: {int((test.review & test.called).sum())}")


enc, lr = model.named_steps["onehotencoder"], model.named_steps["logisticregression"]
names = enc.get_feature_names_out(FEATURES)
coef = lr.coef_[0]
col_feature = np.array([next(f for f in FEATURES if n.startswith(f + "_")) for n in names])

def contributions(X):
    Z = enc.transform(X[FEATURES])
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    return pd.DataFrame({f: (Z[:, col_feature == f] * coef[col_feature == f]).sum(axis=1) for f in FEATURES},
                        index=X.index)

baseline = contributions(train).mean()
contrib = contributions(test) - baseline

origin_rate = train.groupby("origin")["converted"].mean()
page_stats = train.groupby("landing_page_id")["converted"].agg(["count", "mean"])
PRETTY = {"origin": "channel", "landing_page_id": "landing page", "contact_weekday": "weekday"}

def explain(row, c):
    parts = []
    if row["review"]:
        parts.append(f"REVIEW: landing page has only {row['page_history']} past leads, too few to judge on its own (it is pooled with other rare pages), so the score leans mainly on the channel.")
    else:
        n, r = page_stats.loc[row["landing_page_id"]]
        parts.append(f"Landing page {'raises' if c['landing_page_id'] > 0 else 'lowers'} the score "
                     f"({int(n)} past leads, {r:.0%} converted).")
    o_rate = origin_rate.get(row["origin"], np.nan)
    parts.append(f"Channel '{row['origin']}' {'raises' if c['origin'] > 0 else 'lowers'} it "
                 f"({o_rate:.0%} converted)." if not np.isnan(o_rate) else f"Channel '{row['origin']}' is new.")
    return " ".join(parts)

test["explanation"] = [explain(test.loc[i], contrib.loc[i]) for i in test.index]
display(test.sort_values("score", ascending=False)[["origin", "page_history", "score", "review", "explanation"]].head(6))


LEAKED = ["segment", "revenue", "sdr", "seller", "won", "catalog", "behaviour", "business type", "lead type", "gtin", "stock"]

def l1_checks(row, c):
    e = row["explanation"]
    checks = {}
    checks["length_ok"] = 0 < len(e.split()) <= 40
    checks["no_leaked_fields"] = not any(w in e.lower() for w in LEAKED)
    quoted = [int(x) for x in re.findall(r"(\d+)% converted", e)]
    expected = []
    if not row["review"]:
        expected.append(round(100 * page_stats.loc[row["landing_page_id"], "mean"]))
    if row["origin"] in origin_rate.index:
        expected.append(round(100 * origin_rate[row["origin"]]))
    checks["numbers_match_history"] = quoted == expected
    checks["review_flag_consistent"] = ("REVIEW" in e) == bool(row["review"])
    dir_ok = True
    m = re.search(r"Landing page (raises|lowers)", e)
    if m: dir_ok &= (m.group(1) == "raises") == (c["landing_page_id"] > 0)
    m = re.search(r"' (raises|lowers) it", e)
    if m: dir_ok &= (m.group(1) == "raises") == (c["origin"] > 0)
    checks["direction_matches_model"] = dir_ok
    return checks

l1 = pd.DataFrame([l1_checks(test.loc[i], contrib.loc[i]) for i in test.index], index=test.index)
l1_summary = l1.mean().map(lambda v: f"{v:.0%}").to_frame("pass rate")
l1_summary.loc["ALL CHECKS"] = f"{l1.all(axis=1).mean():.0%}"
print(f"L1 assertions on all {len(test)} May explanations:")
display(l1_summary)


top = test.sort_values("score", ascending=False)
sample = pd.concat([top[~top.review].head(5), top[top.review].head(3), top[~top.review].tail(2)])
sheet = sample[["mql_id", "origin", "page_history", "score", "converted", "explanation"]].copy()
sheet["score"] = sheet["score"].round(3)
sheet["rating_correct (1-3)"] = ""
sheet["rating_useful (1-3)"] = ""
sheet["comment"] = ""

import os
ai_rated_path = find_file("explanation_review_ai_rated.csv")
if os.path.exists(ai_rated_path):
    ai_rated = pd.read_csv(ai_rated_path)
    print("AI-assisted review (Codex), shown for comparison; it rated the earlier review-flag wording:")
    print(ai_rated[["rating_correct (1-3)", "rating_useful (1-3)"]].mean().round(2))

if os.path.exists(find_file("explanation_review_rated.csv")):
    rated = pd.read_csv(find_file("explanation_review_rated.csv"))
    print("My ratings (confirmed by me against the tool evidence, with AI-assisted presentation; not an independent blind study):")
    for col in ["rating_correct (1-3)", "rating_useful (1-3)"]:
        print(f"{col}: mean {rated[col].mean():.2f}, rated 3 in {(rated[col] == 3).mean():.0%} of cases")
    if os.path.exists(ai_rated_path):
        both = rated.merge(pd.read_csv(ai_rated_path), on="mql_id", suffixes=("", " AI"))
        for col in ["rating_correct (1-3)", "rating_useful (1-3)"]:
            print(f"Agreement with the AI reviewer on {col}: {(both[col] == both[col + ' AI']).sum()} of {len(both)}")
elif not os.path.exists(find_file("explanation_review.csv")):              # never overwrite a sheet someone may be filling in
    sheet.to_csv("explanation_review.csv", index=False)
    print("Wrote explanation_review.csv — fill in the ratings, save as explanation_review_rated.csv, and re-run this cell.")
display(sheet)


PRICE_IN_PER_M, PRICE_OUT_PER_M = 1.00, 5.00     # USD per million tokens (Claude Haiku 4.5)

example_prompt = (
    "You explain lead-ranking scores to an SDR manager in one or two plain sentences. "
    "Use only the facts given. Do not add facts. Do not change any number.\n"
    f"Rank: 3 of 40. Score: 0.24. Review flag: no.\n"
    f"Landing page: 912 past leads, 19% converted, contribution +0.52.\n"
    f"Channel: organic_search, 15% converted, contribution +0.18.\n"
    f"Weekday: Tuesday, contribution +0.03."
)
tokens_in = len(example_prompt) / 4          # rough rule of thumb: ~4 characters per token
tokens_out = 60                              # one or two sentences

daily = df.groupby(df.first_contact_date.dt.date).size()
leads_per_day = daily.median()
calls_per_list = leads_per_day * CAPACITY    # explanations only for the called 20%
cost_per_call = tokens_in / 1e6 * PRICE_IN_PER_M + tokens_out / 1e6 * PRICE_OUT_PER_M

rows = []
for scale in [1, 10, 100]:
    calls = calls_per_list * scale
    rows.append({"volume": f"{scale}× this sample", "leads per day": int(leads_per_day * scale),
                 "LLM calls per list": int(calls),
                 "USD per ranked list": round(calls * cost_per_call, 4),
                 "USD per year (250 working days)": round(calls * cost_per_call * 250, 2)})
print(f"~{tokens_in:.0f} input + {tokens_out} output tokens per call → USD {cost_per_call:.5f} per call")
print(f"Median leads per day in the data: {leads_per_day:.0f}")
display(pd.DataFrame(rows).set_index("volume"))



fair = (test.groupby("origin")
            .apply(lambda g: pd.Series({
                "leads": len(g),
                "called": int(g.called.sum()),
                "share called": f"{g.called.mean():.0%}",
                "conversions": int(g.converted.sum()),
                "conversions called": int((g.called & (g.converted == 1)).sum()),
                "share of conversions called": f"{(g.called & (g.converted == 1)).sum() / max(g.converted.sum(), 1):.0%}",
            }), include_groups=False)
            .sort_values("leads", ascending=False))
display(fair)


GAP_PTS, MIN_LEADS = 5, 30
drift = pd.DataFrame({
    "training leads": train.groupby("origin").size(),
    "training rate": train.groupby("origin")["converted"].mean(),
    "May leads": test.groupby("origin").size(),
    "May rate": test.groupby("origin")["converted"].mean(),
}).dropna(subset=["May leads"])
overall_shift = 100 * (test["converted"].mean() - train["converted"].mean())
drift["gap (points)"] = (100 * (drift["May rate"] - drift["training rate"])).round(1)
drift["gap beyond overall shift"] = (drift["gap (points)"] - overall_shift).round(1)
drift["flag"] = (drift["May leads"] >= MIN_LEADS) & (drift["gap beyond overall shift"].abs() > GAP_PTS)
drift = drift.sort_values("May leads", ascending=False)
for c in ["training rate", "May rate"]:
    drift[c] = (100 * drift[c]).round(1).astype(str) + "%"
drift[["training leads", "May leads"]] = drift[["training leads", "May leads"]].astype(int)
print(f"Overall conversion rate: {100 * train['converted'].mean():.1f}% in training, {100 * test['converted'].mean():.1f}% in May "
      f"(shift {overall_shift:+.1f} points)")
display(drift)
print("Flagged channels:", ", ".join(drift.index[drift.flag]) or "none")


final_model = make_logreg().fit(df[FEATURES], df["converted"])
artifacts = {
    "model": final_model,
    "page_counts": df["landing_page_id"].value_counts(),
    "page_rate": df.groupby("landing_page_id")["converted"].mean(),
    "origin_rate": df.groupby("origin")["converted"].mean(),
    "min_page_leads": MIN_PAGE_LEADS, "t_review": T_REVIEW, "capacity": CAPACITY,
}
joblib.dump(artifacts, "lead_ranker.joblib")
print("Saved lead_ranker.joblib")

may_days = test.groupby(test.first_contact_date.dt.date).size()
day = (may_days - may_days.median()).abs().idxmin()
lst = test[test.first_contact_date.dt.date == day].sort_values("score", ascending=False).copy()
n_call = int(round(CAPACITY * len(lst)))
lst["rank"] = range(1, len(lst) + 1)
lst["action"] = np.where(lst.index.isin(test.index[test.called]), "Call", np.where(lst["review"], "Review", "Later"))
lst.loc[(lst.action == "Call") & lst.review, "action"] = "Call (check)"
daily_list = lst[["rank", "action", "score", "origin", "explanation", "converted"]].rename(
    columns={"converted": "actually converted (hidden from Sarah)"})
daily_list["score"] = daily_list["score"].round(3)
daily_list.to_csv("sample_daily_list.csv", index=False)
print(f"Sample day {day}: {len(lst)} leads, {n_call} call slots, {int(lst.review.sum())} flagged for review.")
called = lst[lst.action.str.startswith("Call")]
print(f"Of the {n_call} called leads, {int(called.converted.sum())} actually converted; "
      f"the day had {int(lst.converted.sum())} conversions in total.")
display(daily_list)

