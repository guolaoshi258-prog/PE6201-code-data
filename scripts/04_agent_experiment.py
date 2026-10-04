"""Executable project experiment: 04_agent_experiment. Run from the repository root."""

try:
    display
except NameError:
    display = print




import os, sys, json, time, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 120)

for p in ["src", "../src"]:
    if os.path.isdir(p):
        sys.path.insert(0, p)
from lead_tools import LeadTools, find_file
from lead_agent import (run_agent, run_routed, run_workflow, template_brief, apply_guard, make_client,
                        MAX_STEPS, TEMPERATURE)
from agent_eval import build_tasks, check

MODELS = {"low": "google/gemini-3.1-flash-lite", "mid": "anthropic/claude-haiku-4.5", "strong": "anthropic/claude-sonnet-5.5"}
AGENT_RUNS = 2
BUDGET_USD = 3.00                     # the experiment stops starting new runs once this much has been spent
OUT = "outputs" if os.path.isdir("outputs") else ("../outputs" if os.path.isdir("../outputs") else ".")
RUNS_FILE = os.path.join(OUT, "agent_runs.jsonl")

tools = LeadTools()
print(f"Tools ready: {len(tools.leads):,} May leads over {len(tools.slots)} days (no outcome column loaded).")
print(f"Settings: temperature {TEMPERATURE}, at most {MAX_STEPS} tool rounds, {AGENT_RUNS} agent runs per model and request.")


TASKS_FILE = find_file("agent_tasks.json")
if os.path.exists(TASKS_FILE):
    tasks = json.load(open(TASKS_FILE))
    print(f"Loaded {len(tasks)} tasks from {TASKS_FILE}")
else:
    tasks = build_tasks(tools)
    TASKS_FILE = os.path.join("data/processed" if os.path.isdir("data/processed") else
                              "../data/processed" if os.path.isdir("../data/processed") else ".", "agent_tasks.json")
    json.dump(tasks, open(TASKS_FILE, "w"), indent=1)
    print(f"Built and saved {len(tasks)} tasks to {TASKS_FILE}")
TASK = {t["id"]: t for t in tasks}
display(pd.DataFrame([{"id": t["id"], "request": t["request"],
                       "required tools": ", ".join(r["tool"] for r in t["required_tools"]) or "—"} for t in tasks]))


done = [json.loads(l) for l in open(RUNS_FILE)] if os.path.exists(RUNS_FILE) else []
have = {(r["mode"], r["model"], r["task"], r["run"]) for r in done}
jobs = [("agent", m, t["id"], k) for m in MODELS.values() for t in tasks for k in range(1, AGENT_RUNS + 1)]
jobs += [("workflow", m, t["id"], 1) for m in MODELS.values() for t in tasks]
jobs += [("routed", m, t["id"], 1) for m in MODELS.values() for t in tasks]
todo = [j for j in jobs if j not in have]
print(f"{len(done)} runs saved, {len(todo)} to do.")

try:
    client = make_client() if todo else None
except RuntimeError as e:
    client = None
    print(e, "— using saved runs only.")

lock, spent = threading.Lock(), sum(r["cost_usd"] for r in done)
def work(job):
    global spent
    mode, model, tid, k = job
    with lock:
        if spent >= BUDGET_USD:
            return None
    t = TASK[tid]
    rec = (run_agent(client, model, t["request"], tools) if mode == "agent"
           else run_routed(client, model, t["request"], tools) if mode == "routed"
           else run_workflow(client, model, t["request"], t["date"], tools))
    rec.update(task=tid, run=k, ts=time.strftime("%Y-%m-%d %H:%M:%S"))
    with lock:
        spent += rec["cost_usd"]
        with open(RUNS_FILE, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    return rec

if client and todo:
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        new = [r for r in ex.map(work, todo) if r]
    print(f"Ran {len(new)} new runs in {time.time() - t0:.0f}s. Total spend so far: USD {spent:.3f}")

runs = [json.loads(l) for l in open(RUNS_FILE)] if os.path.exists(RUNS_FILE) else []
def model_output(r):
    if "raw_answer" not in r:
        return r
    n = r.get("n_model_tool_calls", len(r["tool_calls"]))
    return {**r, "answer": r["raw_answer"], "tool_calls": r["tool_calls"][:n], "fallback": False}
runs = [apply_guard(model_output(r), TASK[r["task"]]["request"], tools) for r in runs]
print(f"Current guard applied to all {len(runs)} saved runs.")
runs += [dict(template_brief(t["date"], tools), task=t["id"], run=1) for t in tasks if t["kind"] == "daily_brief"]
print(f"{len(runs)} runs available for scoring (including {sum(r['mode'] == 'template' for r in runs)} template answers).")


ex = next((r for r in runs if r["mode"] == "agent" and r["model"] == MODELS["strong"] and r["task"] == "daily_brief_2"), None)
if ex:
    for i, c in enumerate(ex["tool_calls"], 1):
        print(f"{i}. {c['name']}({json.dumps(c['args'])})")
    print(f"\n{ex['steps']} model calls, {ex['tokens_in']:,} input + {ex['tokens_out']:,} output tokens, "
          f"USD {ex['cost_usd']:.4f}, {ex['latency_s']}s\n")
    print(ex["answer"])


def raw(r):
    """The model's own answer, before the guard replaced anything."""
    if "raw_answer" not in r:
        return r
    n = r.get("n_model_tool_calls", len(r["tool_calls"]))
    return {**r, "answer": r["raw_answer"], "tool_calls": r["tool_calls"][:n], "fallback": not r["raw_answer"].strip()}

rows = []
for r in runs:
    c = check(raw(r), TASK[r["task"]])
    rows.append({"mode": r["mode"], "model": r["model"], "task": r["task"], "kind": TASK[r["task"]]["kind"], "run": r["run"],
                 **{k: c[k] for k in ["tools_ok", "complete", "faithful", "constraints_ok", "pass_all"]},
                 "unsupported": ", ".join(c["unsupported_numbers"]), "fallback": raw(r)["fallback"], "error": r["error"],
                 "guard_blocked": bool(r.get("guard_blocked")), "blocked_for": ", ".join(r.get("guard_blocked", [])),
                 "delivered": check(r, TASK[r["task"]]),
                 "n_tool_calls": len(r["tool_calls"]), "steps": r["steps"], "tokens_in": r["tokens_in"],
                 "tokens_out": r["tokens_out"], "cost_usd": r["cost_usd"], "latency_s": r["latency_s"],
                 "retries": r["retries"]})
res = pd.DataFrame(rows)
res["answer_ok"] = res.complete & res.faithful & res.constraints_ok & ~res.fallback
tier = {v: k for k, v in MODELS.items()} | {"none": "—"}
res["system"] = res["mode"] + " · " + res["model"].map(tier).fillna(res["model"])

summary = (res.groupby("system")
              .agg(runs=("pass_all", "size"), pass_all=("pass_all", "mean"), answer_ok=("answer_ok", "mean"), tools_ok=("tools_ok", "mean"),
                   complete=("complete", "mean"), faithful=("faithful", "mean"), constraints_ok=("constraints_ok", "mean"),
                   fallback=("fallback", "sum"), errors=("error", lambda s: s.notna().sum()),
                   tool_calls=("n_tool_calls", "mean"), cost_per_request=("cost_usd", "mean"),
                   latency_s=("latency_s", "mean")))
for c in ["pass_all", "answer_ok", "tools_ok", "complete", "faithful", "constraints_ok"]:
    summary[c] = (100 * summary[c]).round(0).astype(int).astype(str) + "%"
summary["cost_per_request"] = summary["cost_per_request"].map(lambda v: f"{v:.4f}")
summary["latency_s"] = summary["latency_s"].round(1); summary["tool_calls"] = summary["tool_calls"].round(1)
display(summary)


by_kind = res.pivot_table(index="system", columns="kind", values="pass_all", aggfunc="mean")
display((100 * by_kind).round(0).astype("Int64").astype(str) + "%")


fails = res[(res.mode != "template") & ~res.pass_all]
cols = ["system", "task", "run", "tools_ok", "complete", "faithful", "constraints_ok", "unsupported", "error"]
display(fails[cols].sort_values(["task", "system"]).reset_index(drop=True))


ag = res[res["mode"] == "agent"].pivot_table(index=["system", "task"], columns="run", values="pass_all", aggfunc="first")
if ag.shape[1] >= 2:
    ag["agree"] = np.where(ag[1] == ag[2], np.where(ag[1], "both pass", "both fail"), "disagree")
    display(ag.groupby("system")["agree"].value_counts().unstack(fill_value=0))


AUDIT = {
    **{("adversarial_1", "agent", t, k): "strict criterion" for t in ["low", "mid"] for k in [1, 2]},
    **{("lead_question_3", "agent", "mid", k): "strict criterion" for k in [1, 2]},     # right answer, ID not repeated
    **{("channel_4", "agent", "mid", k): "derived number" for k in [1, 2]},
    ("channel_4", "agent", "strong", 2): "derived number",
    ("channel_4", "routed", "strong", 1): "derived number",                       # 80% = 4/5, 44% = 11/25, both correct
    **{("channel_1", m, "mid", 1): "strict criterion" for m in ["routed", "workflow"]},   # wrote "Paid search", not "paid_search"
}
def audit(r):
    if r.pass_all:
        return "pass"
    if r["mode"] in ("workflow", "routed") and r.answer_ok and not r.tools_ok:
        return "structural"
    return AUDIT.get((r.task, r["mode"], tier.get(r.model), r.run), "genuine")

res["audit"] = res.apply(audit, axis=1)
table = res[res["mode"] != "template"].groupby("system")["audit"].value_counts().unstack(fill_value=0)
table["genuine failure rate"] = (100 * table.get("genuine", 0) / table.sum(axis=1)).round(0).astype(int).astype(str) + "%"
display(table)
print("Genuine failures:")
display(res[res.audit == "genuine"][["system", "task", "run", "tools_ok", "complete", "faithful", "constraints_ok", "unsupported"]]
        .sort_values(["system", "task"]).reset_index(drop=True))


blk = res[res.guard_blocked]
print(f"Guard replaced {len(blk)} of {(res['mode'] != 'template').sum()} LLM answers.")
display(blk[["system", "task", "run", "blocked_for", "audit"]].reset_index(drop=True))
res["delivered_complete"] = [d["complete"] for d in res["delivered"]]
res["delivered_constraints"] = [d["constraints_ok"] for d in res["delivered"]]
display(res[res["mode"] != "template"].groupby("system")[["complete", "delivered_complete", "constraints_ok", "delivered_constraints"]]
        .mean().mul(100).round(0).astype(int).astype(str) + "%")


if (res["mode"] != "template").any():
    cost = res[res["mode"] != "template"].groupby(["system", res["kind"] == "daily_brief"])["cost_usd"].mean().unstack()
    cost.columns = ["per follow-up question", "per daily brief"]
    cost["per day (1 brief + 5 questions)"] = cost["per daily brief"] + 5 * cost["per follow-up question"]
    cost["per year (250 days)"] = 250 * cost["per day (1 brief + 5 questions)"]
    display(cost.map(lambda v: f"USD {v:.4f}" if v < 1 else f"USD {v:.2f}"))
    print(f"Total spent on this experiment: USD {res['cost_usd'].sum():.3f}")


pick = ["daily_brief_2", "lead_question_2", "override_2", "adversarial_1"]
sel = [r for r in runs if r["task"] in pick and r["run"] == 1 and r["mode"] in ("agent", "template")]
rng = np.random.default_rng(7)
order = rng.permutation(len(sel))
sheet = pd.DataFrame([{"row": i + 1, "request": TASK[sel[j]["task"]]["request"], "answer": sel[j]["answer"],
                       "rating_correct (1-3)": "", "rating_useful (1-3)": "", "comment": ""} for i, j in enumerate(order)])
key = pd.DataFrame([{"row": i + 1, "system": f"{sel[j]['mode']} · {tier.get(sel[j]['model'], sel[j]['model'])}",
                     "task": sel[j]["task"]} for i, j in enumerate(order)])
ai_rated_path = find_file("agent_review_ai_rated.csv")
if os.path.exists(ai_rated_path):
    ai_rated = pd.read_csv(ai_rated_path).merge(key, on="row")
    print("AI-assisted review (Codex, one reviewer with project context) — not a human blind L2 evaluation:")
    display(ai_rated.groupby("system")[["rating_correct (1-3)", "rating_useful (1-3)"]].mean().round(2))
    display(ai_rated[ai_rated["rating_correct (1-3)"] < 3][["row", "system", "task", "comment"]])

rated_path = find_file("agent_review_rated.csv")
if os.path.exists(rated_path):
    rated = pd.read_csv(rated_path).merge(key, on="row")
    print("My ratings (I rated every answer against the tool evidence; an AI assistant laid out the evidence and suggested points,\n"
          "and I confirmed or changed each score — not an independent blind study):")
    display(rated.groupby("system")[["rating_correct (1-3)", "rating_useful (1-3)"]].agg(["mean", "count"]).round(2))
    display(rated[rated["rating_correct (1-3)"] < 3][["row", "system", "task", "rating_correct (1-3)", "comment"]])
    if os.path.exists(ai_rated_path):
        both = rated.merge(pd.read_csv(ai_rated_path), on="row", suffixes=("", " AI"))
        for col in ["rating_correct (1-3)", "rating_useful (1-3)"]:
            same = (both[col] == both[col + " AI"]).sum()
            print(f"Agreement with the AI reviewer on {col}: {same} of {len(both)}")
elif sel and not os.path.exists(os.path.join(OUT, "agent_review.csv")):      # never overwrite a sheet someone may be filling in
    sheet.to_csv(os.path.join(OUT, "agent_review.csv"), index=False)
    key.to_csv(os.path.join(OUT, "agent_review_key.csv"), index=False)
    print(f"Wrote agent_review.csv ({len(sheet)} answers) and agent_review_key.csv to {OUT}. "
          "Rate the answers, save as agent_review_rated.csv, and re-run this cell.")

