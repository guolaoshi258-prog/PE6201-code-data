"""Executable project experiment: 02_method_comparison. Run from the repository root."""

try:
    display
except NameError:
    display = print




import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)

FEATURES = ["origin", "landing_page_id", "contact_weekday"]
CAPACITY = 0.20
N_SPLITS = 30
SEED = 42
MIN_PAGE_LEADS = 20      # landing pages with fewer training leads are treated as "rare"

def find_file(name):
    """Find a data file whether the notebook runs from the repo root, from notebooks/, or in Colab."""
    for d in [".", "data/raw", "data/processed", "../data/raw", "../data/processed", "outputs", "../outputs"]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return name

df = pd.read_csv(find_file("leads_clean.csv"), parse_dates=["first_contact_date"])
print(f"{len(df):,} leads, {int(df.converted.sum())} conversions ({df.converted.mean():.1%})")


def capture_at(y, score, rng, frac=CAPACITY):
    """Return (conversions captured in the top `frac` of leads, total conversions)."""
    y = np.asarray(y); score = np.asarray(score, dtype=float)
    k = int(round(frac * len(y)))
    order = np.lexsort((rng.random(len(y)), -score))   # sort by score desc, random tie-break
    return int(y[order[:k]].sum()), int(y.sum())


def rule_channel(train, test):
    rates = train.groupby("origin")["converted"].mean()
    return test["origin"].map(rates).fillna(train["converted"].mean()).values

def rule_page_then_channel(train, test):
    page = train.groupby("landing_page_id")["converted"].agg(["mean", "count"])
    page_rate = page.loc[page["count"] >= MIN_PAGE_LEADS, "mean"]
    fallback = rule_channel(train, test)
    return test["landing_page_id"].map(page_rate).fillna(pd.Series(fallback, index=test.index)).values

def make_logreg():
    return make_pipeline(
        OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_PAGE_LEADS),
        LogisticRegression(max_iter=2000))

def make_gbm():
    return make_pipeline(
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                       encoded_missing_value=np.nan, min_frequency=MIN_PAGE_LEADS),
        HistGradientBoostingClassifier(categorical_features=[0, 1, 2], max_iter=200,
                                       learning_rate=0.05, max_leaf_nodes=15, random_state=SEED))

def ml_scores(make_model):
    def score(train, test):
        model = make_model().fit(train[FEATURES], train["converted"])
        return model.predict_proba(test[FEATURES])[:, 1]
    return score

METHODS = {
    "Rule A — channel": rule_channel,
    "Rule B — page, then channel": rule_page_then_channel,
    "ML 1 — logistic regression": ml_scores(make_logreg),
    "ML 2 — gradient boosting": ml_scores(make_gbm),
}


def evaluate(data, n_splits=N_SPLITS, seed=SEED):
    rng = np.random.default_rng(seed)
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=seed)
    rows = []
    for i, (tr, te) in enumerate(splitter.split(data, data["converted"])):
        train, test = data.iloc[tr], data.iloc[te]
        y = test["converted"].values
        rand = [capture_at(y, rng.random(len(y)), rng)[0] for _ in range(200)]
        rows.append({"split": i, "method": "Random", "captured": np.mean(rand),
                     "total": int(y.sum()), "auc": 0.5})
        for name, fn in METHODS.items():
            s = fn(train, test)
            cap, tot = capture_at(y, s, rng)
            rows.append({"split": i, "method": name, "captured": cap, "total": tot,
                         "auc": roc_auc_score(y, s)})
    res = pd.DataFrame(rows)
    res["capture_pct"] = 100 * res["captured"] / res["total"]
    return res

def summarise(res):
    order = ["Random"] + list(METHODS)
    s = (res.groupby("method")
            .agg(conversions_in_test=("total", "mean"),
                 captured_mean=("captured", "mean"),
                 capture_pct_mean=("capture_pct", "mean"),
                 capture_pct_min=("capture_pct", "min"),
                 capture_pct_max=("capture_pct", "max"),
                 capture_pct_sd=("capture_pct", "std"),
                 roc_auc_mean=("auc", "mean"))
            .reindex(order).round(1))
    s["roc_auc_mean"] = res.groupby("method")["auc"].mean().reindex(order).round(3)
    s["lift_vs_random"] = (s["capture_pct_mean"] / s.loc["Random", "capture_pct_mean"]).round(2)
    return s

t0 = time.time()
res = evaluate(df)
print(f"Ran {N_SPLITS} splits x {len(METHODS) + 1} methods in {time.time() - t0:.0f}s")
main_table = summarise(res)
display(main_table)



n_test = len(df) - int(len(df) * 0.8)          # StratifiedShuffleSplit(test_size=0.2) test-set size
k = int(round(CAPACITY * n_test))
pr = main_table[["captured_mean", "capture_pct_mean"]].copy()
pr["leads called"] = k
pr["precision@20%"] = (100 * pr["captured_mean"] / k).round(1).astype(str) + "%"
pr["recall@20% (= Capture@20%)"] = pr["capture_pct_mean"].round(1).astype(str) + "%"
print(f"Test set: {n_test:,} leads, {k} calls (20%); base conversion rate {100 * df.converted.mean():.1f}%")
display(pr.drop(columns="capture_pct_mean"))


order = ["Random"] + list(METHODS)
fig, ax = plt.subplots(figsize=(10, 4))
data = [res.loc[res.method == m, "capture_pct"].values for m in order]
import matplotlib
if tuple(int(x) for x in matplotlib.__version__.split(".")[:2]) >= (3, 10):
    ax.boxplot(data, orientation="horizontal", widths=0.5)
else:
    ax.boxplot(data, vert=False, widths=0.5)
for i, d in enumerate(data, start=1):
    ax.scatter(d, np.full(len(d), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(d)), s=10, alpha=0.6)
ax.set_yticks(range(1, len(order) + 1)); ax.set_yticklabels(order)
ax.axvline(20, color="grey", ls="--", lw=1)
ax.set_xlabel("Capture@20% (% of test conversions in the called 20%)")
ax.set_title(f"Capture@20% across {N_SPLITS} repeated splits")
plt.tight_layout(); plt.show()


wide = res.pivot(index="split", columns="method", values="captured")
best_rule = main_table.loc[["Rule A — channel", "Rule B — page, then channel"], "captured_mean"].idxmax()
rows = []
for ml in ["ML 1 — logistic regression", "ML 2 — gradient boosting"]:
    diff = wide[ml] - wide[best_rule]
    rows.append({"comparison": f"{ml}  vs  {best_rule}",
                 "ML wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()), "rule wins": int((diff < 0).sum()),
                 "mean difference (conversions)": round(diff.mean(), 1),
                 "range of difference": f"{int(diff.min())} to {int(diff.max())}"})
display(pd.DataFrame(rows))


train_t = df[df.first_contact_date < "2018-05-01"]
test_t = df[df.first_contact_date >= "2018-05-01"]
y = test_t["converted"].values
rng = np.random.default_rng(SEED)
rows = [{"method": "Random",
         "captured": round(np.mean([capture_at(y, rng.random(len(y)), rng)[0] for _ in range(1000)]), 1),
         "total": int(y.sum()), "auc": 0.5}]
for name, fn in METHODS.items():
    s = fn(train_t, test_t)
    cap, tot = capture_at(y, s, rng)
    rows.append({"method": name, "captured": cap, "total": tot, "auc": round(roc_auc_score(y, s), 3)})
time_table = pd.DataFrame(rows).set_index("method")
time_table["capture_pct"] = (100 * time_table.captured / time_table.total).round(1)
print(f"Train: {len(train_t):,} leads (Jan–Apr) · Test: {len(test_t):,} leads (May), {int(y.sum())} conversions")
display(time_table)

counts = train_t["landing_page_id"].value_counts()
seen = test_t["landing_page_id"].map(counts).fillna(0)
print(f"May leads on a landing page unseen in training: {(seen == 0).mean():.1%}; "
      f"on a page with < {MIN_PAGE_LEADS} training leads: {(seen < MIN_PAGE_LEADS).mean():.1%}")


days = list(test_t.groupby(test_t.first_contact_date.dt.date))
sizes = pd.Series({d: len(g) for d, g in days})
print(f"May: {len(days)} days, {int(sizes.median())} leads on a median day (range {sizes.min()}–{sizes.max()}), "
      f"{sum(int(round(CAPACITY * len(g))) for _, g in days)} call slots in total")

rows = []
scores = {"Random": None, **{name: pd.Series(fn(train_t, test_t), index=test_t.index) for name, fn in METHODS.items()}}
for name, s in scores.items():
    draws = []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        draws.append(sum(capture_at(g.converted, rng.random(len(g)) if s is None else s[g.index], rng)[0]
                         for _, g in days))
    rows.append({"method": name, "captured (mean of 200)": round(np.mean(draws), 1),
                 "min": min(draws), "max": max(draws), "total": int(test_t.converted.sum()),
                 "monthly top 20% (section 7)": time_table.loc[name, "captured"]})
daily_table = pd.DataFrame(rows).set_index("method")
daily_table["capture_pct"] = (100 * daily_table["captured (mean of 200)"] / daily_table["total"]).round(1)
calls = sum(int(round(CAPACITY * len(g))) for _, g in days)
daily_table["precision@20%"] = (100 * daily_table["captured (mean of 200)"] / calls).round(1)
display(daily_table)


df_all = pd.read_csv(find_file("leads_all.csv"), parse_dates=["first_contact_date"])
res_all = evaluate(df_all, n_splits=10)
all_table = summarise(res_all)
compare = pd.DataFrame({
    "Jan–May 2018 (main)": main_table["capture_pct_mean"],
    "All 8,000 leads": all_table["capture_pct_mean"],
})
print("Mean Capture@20% (%), 30 splits for the main window, 10 for the full data:")
display(compare)

y17 = df_all.first_contact_date.dt.year == 2017
pages_2018 = set(df_all.loc[~y17, "landing_page_id"])
print(f"2017 leads on landing pages never used in 2018: {(~df_all.loc[y17, 'landing_page_id'].isin(pages_2018)).mean():.0%}")



train, test = df.iloc[: int(0.8 * len(df))], df.iloc[int(0.8 * len(df)):]
rows = []
for name, fn in METHODS.items():
    t0 = time.perf_counter(); fn(train, test); t = time.perf_counter() - t0
    rows.append({"method": name, "fit + score time (s)": round(t, 3),
                 "per 1,000 leads (ms)": round(1000 * t / len(test) * 1000, 1)})
display(pd.DataFrame(rows).set_index("method"))
print("Effort (lines of method code): Rule A ~3, Rule B ~5, logistic regression ~4, gradient boosting ~6.")
print("Marginal cost per ranked list: effectively zero for all four (local CPU, < 1 second).")

