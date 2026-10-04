"""The five tools the lead-ranking agent may call.

Everything that is a number — scores, ranks, call slots, history rates — is computed here by code. The agent can only
read these outputs; it cannot change a score, a rank or the number of call slots.

Label isolation: the model is trained on January–April 2018 (labels needed for training). The May leads the agent works
on are loaded WITHOUT their `converted` column, so no tool can reveal whether a May lead converted.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

FEATURES = ["origin", "landing_page_id", "contact_weekday"]
CAPACITY = 0.20          # Sarah can call 20% of each day's leads
MIN_PAGE_LEADS = 20      # same threshold as Notebooks 2 and 3
T_REVIEW = 20            # landing page with fewer training leads -> Review flag
SEED = 42


def find_file(name):
    for d in [".", "data/raw", "data/processed", "../data/raw", "../data/processed", "outputs", "../outputs"]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return name


class LeadTools:
    def __init__(self, path=None):
        df = pd.read_csv(path or find_file("leads_clean.csv"), parse_dates=["first_contact_date"])
        df["lead_id"] = df["mql_id"].str[:8]          # short IDs are unique and cheaper for the model to copy
        train = df[df.first_contact_date < "2018-05-01"].copy()
        may = df[df.first_contact_date >= "2018-05-01"].drop(columns=["converted"]).copy()   # label isolation

        self.model = make_pipeline(
            OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_PAGE_LEADS),
            LogisticRegression(max_iter=2000)).fit(train[FEATURES], train["converted"])

        # Per-feature contribution = coefficient of the lead's category, relative to the average training lead
        enc, lr = self.model.named_steps["onehotencoder"], self.model.named_steps["logisticregression"]
        names = enc.get_feature_names_out(FEATURES)
        col_feature = np.array([next(f for f in FEATURES if n.startswith(f + "_")) for n in names])

        def contributions(X):
            Z = enc.transform(X[FEATURES])
            Z = Z.toarray() if hasattr(Z, "toarray") else Z
            return pd.DataFrame({f: (Z[:, col_feature == f] * lr.coef_[0][col_feature == f]).sum(axis=1)
                                 for f in FEATURES}, index=X.index)

        contrib = contributions(may) - contributions(train).mean()
        may["score"] = self.model.predict_proba(may[FEATURES])[:, 1]
        for f in FEATURES:
            may["c_" + f] = contrib[f]

        page_stats = train.groupby("landing_page_id")["converted"].agg(["count", "mean"])
        self.origin_rate = train.groupby("origin")["converted"].mean()
        self.origin_count = train["origin"].value_counts()
        may["page_history"] = may["landing_page_id"].map(page_stats["count"]).fillna(0).astype(int)
        may["page_rate"] = may["landing_page_id"].map(page_stats["mean"])
        may["review"] = may["page_history"] < T_REVIEW
        may["date"] = may["first_contact_date"].dt.date.astype(str)

        # Rank within each day and give the top 20% the call slots — exactly the policy evaluated in Notebook 2 (daily capacity)
        rng = np.random.default_rng(SEED)
        may["rank"], may["action"] = 0, ""
        self.slots = {}
        for d, g in may.groupby("date"):
            order = g.index[np.lexsort((rng.random(len(g)), -g["score"].values))]
            k = int(round(CAPACITY * len(g)))
            self.slots[d] = k
            may.loc[order, "rank"] = np.arange(1, len(g) + 1)
            called = order[:k]
            may.loc[order, "action"] = np.where(order.isin(called), "Call",
                                                np.where(may.loc[order, "review"], "Review", "Later"))
        self.leads = may.set_index("lead_id")
        self.train_rate = float(train["converted"].mean())

    # ------------------------------------------------------------------ helpers
    def _day(self, date):
        date = str(date)[:10]
        if date not in self.slots:
            return None, {"error": f"No leads for date {date}. Available dates: 2018-05-01 to 2018-05-31."}
        return date, None

    def _pct(self, x):
        return None if pd.isna(x) else f"{x:.0%}"

    def template_explanation(self, lead_id):
        r = self.leads.loc[lead_id]
        parts = []
        if r["review"]:
            parts.append(f"REVIEW: landing page has only {r['page_history']} past leads, too few to judge on its own "
                         f"(it is pooled with other rare pages), so the score leans mainly on the channel.")
        else:
            parts.append(f"Landing page {'raises' if r['c_landing_page_id'] > 0 else 'lowers'} the score "
                         f"({r['page_history']} past leads, {r['page_rate']:.0%} converted).")
        o = self.origin_rate.get(r["origin"], np.nan)
        parts.append(f"Channel '{r['origin']}' {'raises' if r['c_origin'] > 0 else 'lowers'} it ({o:.0%} converted)."
                     if not pd.isna(o) else f"Channel '{r['origin']}' is new.")
        return " ".join(parts)

    # ------------------------------------------------------------------ the five tools
    def rank_daily_leads(self, date):
        date, err = self._day(date)
        if err:
            return err
        g = self.leads[self.leads["date"] == date].sort_values("rank")
        return {
            "date": date, "n_leads": len(g), "call_slots": self.slots[date],
            "policy": "Top 20% of the day's leads by model score get the call slots. Review = thin evidence, "
                      "for Sarah's judgement; it does not take a call slot.",
            "leads": [{"lead_id": i, "rank": int(r["rank"]), "score": round(float(r["score"]), 3),
                       "action": r["action"], "review_flag": bool(r["review"]), "channel": r["origin"]}
                      for i, r in g.iterrows()],
        }

    def get_lead_evidence(self, lead_id):
        lead_id = str(lead_id).strip()[:8]
        if lead_id not in self.leads.index:
            return {"error": f"Unknown lead_id '{lead_id}'."}
        r = self.leads.loc[lead_id]
        o = self.origin_rate.get(r["origin"], np.nan)
        return {
            "lead_id": lead_id, "date": r["date"], "rank": int(r["rank"]), "of_leads_that_day": int((self.leads.date == r["date"]).sum()),
            "call_slots_that_day": self.slots[r["date"]], "score": round(float(r["score"]), 3), "action": r["action"],
            "average_training_conversion_rate": self._pct(self.train_rate),
            "landing_page": {"past_leads": int(r["page_history"]), "past_conversion_rate": self._pct(r["page_rate"]),
                             "effect_on_score": "raises" if r["c_landing_page_id"] > 0 else "lowers",
                             "contribution_log_odds": round(float(r["c_landing_page_id"]), 2)},
            "channel": {"name": r["origin"], "past_leads": int(self.origin_count.get(r["origin"], 0)),
                        "past_conversion_rate": self._pct(o),
                        "effect_on_score": "raises" if r["c_origin"] > 0 else "lowers",
                        "contribution_log_odds": round(float(r["c_origin"]), 2)},
            "weekday": {"name": r["contact_weekday"], "contribution_log_odds": round(float(r["c_contact_weekday"]), 2)},
            "review_flag": bool(r["review"]),
            "template_explanation": self.template_explanation(lead_id),
        }

    def get_review_flags(self, date):
        date, err = self._day(date)
        if err:
            return err
        g = self.leads[(self.leads["date"] == date) & self.leads["review"]].sort_values("rank")
        def reason(r):
            why = ("landing page never seen in training" if r["page_history"] == 0
                   else f"landing page has only {r['page_history']} training leads (threshold {T_REVIEW})")
            if r["origin"] in ("missing", "unknown"):
                why += f"; channel recorded as '{r['origin']}'"
            return why
        return {"date": date, "n_flagged": len(g), "n_leads": int((self.leads.date == date).sum()),
                "meaning": "The model has too little history to judge these landing pages on their own (rare pages are pooled), "
                           "so their score leans mainly on the channel.",
                "flagged": [{"lead_id": i, "rank": int(r["rank"]), "score": round(float(r["score"]), 3),
                             "action": r["action"], "reason": reason(r)} for i, r in g.iterrows()]}

    def get_channel_summary(self, date):
        date, err = self._day(date)
        if err:
            return err
        g = self.leads[self.leads["date"] == date]
        rows = []
        for ch, h in g.groupby("origin"):
            rows.append({"channel": ch, "leads": len(h), "called": int((h.action == "Call").sum()),
                         "flagged_for_review": int(h.review.sum()),
                         "training_conversion_rate": self._pct(self.origin_rate.get(ch, np.nan))})
        rows.sort(key=lambda x: -x["leads"])
        return {"date": date, "call_slots": self.slots[date], "channels": rows,
                "channels_with_zero_calls": [r["channel"] for r in rows if r["called"] == 0]}

    def validate_call_list(self, lead_ids, date):
        date, err = self._day(date)
        if err:
            return err
        ids = [str(x).strip()[:8] for x in (lead_ids if isinstance(lead_ids, list) else [lead_ids])]
        day = self.leads[self.leads["date"] == date]
        unknown = [i for i in ids if i not in day.index]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        unique = list(dict.fromkeys(i for i in ids if i in day.index))
        slots = self.slots[date]
        problems = []
        if unknown: problems.append(f"{len(unknown)} ID(s) are not leads on {date}: {unknown}")
        if dupes: problems.append(f"duplicate ID(s): {dupes}")
        if len(unique) > slots: problems.append(f"{len(unique)} distinct leads but only {slots} call slots — remove {len(unique) - slots}")
        displaced = [i for i in day[day.action == "Call"].index if i not in unique]
        return {"date": date, "valid": not problems, "call_slots": slots, "n_requested": len(ids),
                "n_distinct_valid": len(unique), "problems": problems,
                "flagged_for_review_in_list": [i for i in unique if day.loc[i, "review"]],
                "model_call_leads_left_out": displaced}

    # ------------------------------------------------------------------ dispatch
    def call(self, name, args):
        fn = {"rank_daily_leads": self.rank_daily_leads, "get_lead_evidence": self.get_lead_evidence,
              "get_review_flags": self.get_review_flags, "get_channel_summary": self.get_channel_summary,
              "validate_call_list": self.validate_call_list}.get(name)
        if fn is None:
            return {"error": f"Unknown tool '{name}'."}
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": f"Bad arguments for {name}: {e}"}


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "rank_daily_leads",
        "description": "Ranked leads for one day: score, rank, action (Call/Review/Later), review flag, channel, and the number of call slots.",
        "parameters": {"type": "object", "properties": {"date": {"type": "string", "description": "YYYY-MM-DD, May 2018"}},
                       "required": ["date"]}}},
    {"type": "function", "function": {
        "name": "get_lead_evidence",
        "description": "Why one lead has its score: its rank, the model's contribution from landing page, channel and weekday, and their training-history rates.",
        "parameters": {"type": "object", "properties": {"lead_id": {"type": "string", "description": "8-character lead ID"}},
                       "required": ["lead_id"]}}},
    {"type": "function", "function": {
        "name": "get_review_flags",
        "description": "Leads on one day whose landing page has too little training history to trust the score, with the reason.",
        "parameters": {"type": "object", "properties": {"date": {"type": "string"}}, "required": ["date"]}}},
    {"type": "function", "function": {
        "name": "get_channel_summary",
        "description": "Per marketing channel on one day: leads, leads called, leads flagged, training conversion rate; and channels that get no calls.",
        "parameters": {"type": "object", "properties": {"date": {"type": "string"}}, "required": ["date"]}}},
    {"type": "function", "function": {
        "name": "validate_call_list",
        "description": "Check a proposed call list for one day: unknown IDs, duplicates, and whether it fits the call slots.",
        "parameters": {"type": "object", "properties": {
            "lead_ids": {"type": "array", "items": {"type": "string"}}, "date": {"type": "string"}},
            "required": ["lead_ids", "date"]}}},
]
