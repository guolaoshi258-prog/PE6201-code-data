"""Answering Sarah's questions: the agent, the routed and fixed-workflow baselines, the guard and the fallback.

Four ways to answer one of Sarah's requests:
  1. agent     — the model decides which tools to call, in what order, and when to stop (up to MAX_STEPS rounds).
  2. routed    — no loop: code reads the dates, lead IDs and keywords in the request, calls the matching tools,
                 and the model writes one answer. This is the fair "no agent" baseline.
  3. workflow  — no loop: code always fetches the same evidence for the request's date, the model writes one answer.
  4. template  — no LLM at all (daily-brief requests only): code fills a fixed text template.

Every LLM answer passes a guard before it is returned: if it states a number that no tool output supports, or claims to
change the list or a score (the tools are read-only), it is replaced by a deterministic answer built by code from the same
tools (the original is kept in the log).

Models are reached through OpenRouter's OpenAI-compatible API, so one key covers models from several vendors.
"""

import os
import re
import json
import time
from pathlib import Path
from openai import OpenAI

from lead_tools import TOOL_SCHEMAS
from agent_eval import unsupported_numbers

MAX_STEPS = 8             # tool-calling rounds before the agent is stopped
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 1200

SYSTEM_PROMPT = """You support Sarah, the SDR manager of an e-commerce marketplace. Every day she can phone only 20% of the
new leads. A logistic-regression model has already scored and ranked the leads; you help her act on that ranking.

Rules:
- Use the tools to get facts. Every number you state (score, rank, rate, count) must come from a tool result in this conversation.
- You cannot change a score, a rank or the number of call slots. If asked to, say so and explain what Sarah can do instead
  (she may override the list by hand; check any proposed list with validate_call_list).
- A Review flag means the model has too little history on that lead's landing page. It is a prompt for Sarah's judgement,
  not a rejection.
- You have no contact details, company names or information about whether a lead later converted. Never invent them.
- Known limitations of the ranking: only three features exist (channel, landing page, weekday); channels recorded as
  'unknown' or 'missing' rank highest, which may reflect how leads were tagged rather than lead quality; some channels
  (e.g. social, email) are rarely or never called, so the system cannot learn if they improve.
- Be brief and concrete: lead IDs, ranks, scores and reasons. No more than about 200 words."""


def load_api_key():
    """OPENROUTER_API_KEY from the environment, or from a .env file in this folder or a parent folder."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    for d in [Path.cwd(), *Path.cwd().parents][:4]:
        env = d / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.strip().startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        return key
    raise RuntimeError("Set OPENROUTER_API_KEY (environment variable or a .env file) to run the agent.")


def make_client():
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=load_api_key())


def _complete(client, model, messages, tools=None):
    """One API call with a single retry. Returns (message, usage dict)."""
    kwargs = dict(model=model, messages=messages, temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS,
                  extra_body={"usage": {"include": True}})       # OpenRouter then reports the billed cost
    if tools:
        kwargs["tools"] = tools
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(**kwargs)
            u = resp.usage
            usage = {"in": u.prompt_tokens or 0, "out": u.completion_tokens or 0,
                     "cost": float(getattr(u, "cost", None) or (u.model_extra or {}).get("cost") or 0.0)}
            return resp.choices[0].message, usage, attempt
        except Exception as e:                                     # network error, rate limit, provider error
            if attempt == 1:
                raise
            time.sleep(3)


def _new_record(mode, model):
    return {"mode": mode, "model": model, "answer": "", "tool_calls": [], "steps": 0, "tokens_in": 0, "tokens_out": 0,
            "cost_usd": 0.0, "retries": 0, "error": None, "fallback": False, "latency_s": 0.0}


def _add_usage(rec, usage, retried):
    rec["tokens_in"] += usage["in"]; rec["tokens_out"] += usage["out"]
    rec["cost_usd"] += usage["cost"]; rec["retries"] += retried


def run_agent(client, model, request, tools):
    """The agent loop: the model chooses tools until it writes a final answer or hits MAX_STEPS."""
    rec = _new_record("agent", model)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": request}]
    t0 = time.time()
    try:
        for step in range(MAX_STEPS):
            msg, usage, retried = _complete(client, model, messages, TOOL_SCHEMAS)
            _add_usage(rec, usage, retried)
            rec["steps"] = step + 1
            if not msg.tool_calls:
                rec["answer"] = msg.content or ""
                break
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                out = tools.call(tc.function.name, args)
                rec["tool_calls"].append({"name": tc.function.name, "args": args, "output": out})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)})
        else:
            rec["error"] = f"step limit ({MAX_STEPS}) reached"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    rec["latency_s"] = round(time.time() - t0, 2)
    return apply_guard(rec, request, tools)


def run_workflow(client, model, request, date, tools):
    """Baseline without a loop: the same evidence is fetched for every request, then one model call."""
    rec = _new_record("workflow", model)
    t0 = time.time()
    ranked = tools.rank_daily_leads(date)
    calls = [("rank_daily_leads", {"date": date}, ranked),
             ("get_review_flags", {"date": date}, tools.get_review_flags(date)),
             ("get_channel_summary", {"date": date}, tools.get_channel_summary(date))]
    for lead in ranked.get("leads", [])[:3]:
        calls.append(("get_lead_evidence", {"lead_id": lead["lead_id"]}, tools.get_lead_evidence(lead["lead_id"])))
    rec["tool_calls"] = [{"name": n, "args": a, "output": o} for n, a, o in calls]
    evidence = "\n".join(f"{n}({json.dumps(a)}) -> {json.dumps(o)}" for n, a, o in calls)
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\nYou cannot call tools; all available evidence is below."},
                {"role": "user", "content": f"{request}\n\nEvidence:\n{evidence}"}]
    try:
        msg, usage, retried = _complete(client, model, messages)
        _add_usage(rec, usage, retried)
        rec["steps"] = 1
        rec["answer"] = msg.content or ""
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    rec["latency_s"] = round(time.time() - t0, 2)
    return apply_guard(rec, request, tools)


def template_brief(date, tools):
    """Non-LLM baseline for the daily brief: fixed text filled from the same tools."""
    rec = _new_record("template", "none")
    ranked, flags = tools.rank_daily_leads(date), tools.get_review_flags(date)
    rec["tool_calls"] = [{"name": "rank_daily_leads", "args": {"date": date}, "output": ranked},
                         {"name": "get_review_flags", "args": {"date": date}, "output": flags}]
    call = [l for l in ranked["leads"] if l["action"] == "Call"]
    lines = [f"Call list for {date} ({ranked['call_slots']} slots, {ranked['n_leads']} leads): "
             + ", ".join(f"{l['lead_id']} ({l['score']})" for l in call) + "."]
    for l in call[:3]:
        ev = tools.get_lead_evidence(l["lead_id"])
        rec["tool_calls"].append({"name": "get_lead_evidence", "args": {"lead_id": l["lead_id"]}, "output": ev})
        lines.append(f"#{l['rank']} {l['lead_id']}: {ev['template_explanation']}")
    lines.append(f"{flags['n_flagged']} leads are flagged for review: their landing page has too little history.")
    lines.append("Limitation: only channel, landing page and weekday are used; 'unknown' and 'missing' channels rank "
                 "highest, which may reflect tagging rather than lead quality.")
    rec["answer"] = "\n".join(lines)
    return rec


_DATE = re.compile(r"\b2018-\d\d-\d\d\b")
_ID = re.compile(r"\b[0-9a-f]{8}\b")
_CHANNEL_WORDS = ("channel", "social", "referral", "email", "organic", "paid", "direct", "display", "unknown")


def route(request):
    """Read a request the way a fixed router would: dates, lead IDs and a few keywords. Never looks at the test set."""
    dates, ids = _DATE.findall(request), _ID.findall(request)     # IDs kept as written, so duplicates reach the validator
    r = request.lower()
    if len(set(ids)) >= 2:
        kind = "list"
    elif len(ids) == 1:
        kind = "lead"
    elif any(w in r for w in _CHANNEL_WORDS):
        kind = "channel"
    else:
        kind = "day"
    return {"kind": kind, "date": dates[0] if dates else None, "ids": ids}


def _routed_calls(request, tools):
    """The tool calls a router makes for one request: (name, args, output) triples."""
    rt, calls = route(request), []
    def add(name, args):
        calls.append((name, args, tools.call(name, args)))
    if rt["kind"] == "list":
        add("validate_call_list", {"lead_ids": rt["ids"], "date": rt["date"]})
    elif rt["kind"] == "lead":
        add("get_lead_evidence", {"lead_id": rt["ids"][0]})
    elif rt["kind"] == "channel" and rt["date"]:
        add("get_channel_summary", {"date": rt["date"]})
    if rt["date"] and rt["kind"] in ("day", "list", "channel"):
        add("rank_daily_leads", {"date": rt["date"]})
    if rt["date"] and rt["kind"] == "day":
        add("get_review_flags", {"date": rt["date"]})
        for lead in calls[0][2].get("leads", [])[:3]:
            add("get_lead_evidence", {"lead_id": lead["lead_id"]})
    return calls


def run_routed(client, model, request, tools):
    """Fair baseline without a loop: code picks the tools from the request text, then one model call."""
    rec = _new_record("routed", model)
    t0 = time.time()
    calls = _routed_calls(request, tools)
    rec["tool_calls"] = [{"name": n, "args": a, "output": o} for n, a, o in calls]
    evidence = "\n".join(f"{n}({json.dumps(a)}) -> {json.dumps(o)}" for n, a, o in calls) or "(no tool matched this request)"
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\nYou cannot call tools; all available evidence is below."},
                {"role": "user", "content": f"{request}\n\nEvidence:\n{evidence}"}]
    try:
        msg, usage, retried = _complete(client, model, messages)
        _add_usage(rec, usage, retried)
        rec["steps"] = 1
        rec["answer"] = msg.content or ""
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    rec["latency_s"] = round(time.time() - t0, 2)
    return apply_guard(rec, request, tools)


# The assistant's tools are read-only, so any first-person promise to change the list or a score is false.
_CAPABILITY_CLAIM = re.compile(r"\bI(?:'ll| will|'ve| have| just)? (?:add|added|swap|swapped|put|move|moved|change|changed|"
                               r"raise|raised|update|updated|remove|removed|replace|replaced|book|booked|schedule|scheduled)\b",
                               re.IGNORECASE)


def apply_guard(rec, request, tools):
    """Runs before an answer reaches Sarah. An empty answer, one stating a number no tool output supports, or one
    claiming to change the list or a score, is replaced by a deterministic answer built by code.
    The model's original answer stays in the log."""
    rec.setdefault("guard_blocked", [])
    rec["n_model_tool_calls"] = len(rec["tool_calls"])
    if rec["answer"].strip():
        bad = unsupported_numbers(rec, request) + [m.group(0) for m in _CAPABILITY_CLAIM.finditer(rec["answer"])]
        if not bad:
            return rec
        rec["guard_blocked"] = bad
    rec["raw_answer"] = rec["answer"]
    rec["fallback"] = True
    text, calls = fallback_answer(request, tools)
    rec["answer"] = text
    rec["tool_calls"] = rec["tool_calls"] + [{"name": n, "args": a, "output": o} for n, a, o in calls]
    return rec


def fallback_answer(request, tools):
    """Deterministic answer from code alone: the list check, the lead's template explanation, or the template brief."""
    rt, calls = route(request), _routed_calls(request, tools)
    out = {n: o for n, a, o in calls}
    head = "[Checked answer, generated without the language model]\n"
    if rt["kind"] == "list" and "error" not in out["validate_call_list"]:
        v = out["validate_call_list"]
        body = ("Your list is valid: " if v["valid"] else "Your list has problems: " + "; ".join(v["problems"]) + ". ") + \
               f"{v['call_slots']} call slots on {v['date']}."
        if v["model_call_leads_left_out"]:
            body += " Left out of the model's call list: " + ", ".join(v["model_call_leads_left_out"]) + "."
        return head + body, calls
    if rt["kind"] == "lead" and "error" not in out["get_lead_evidence"]:
        e = out["get_lead_evidence"]
        return head + (f"Lead {e['lead_id']} is ranked {e['rank']} of {e['of_leads_that_day']} on {e['date']} "
                       f"({e['call_slots_that_day']} call slots), score {e['score']}, action {e['action']}. "
                       f"{e['template_explanation']}"), calls
    if rt["date"] and "error" not in out.get("rank_daily_leads", {"error": 1}):
        t = template_brief(rt["date"], tools)
        return head + t["answer"], calls + [(c["name"], c["args"], c["output"]) for c in t["tool_calls"]]
    return head + "I could not answer this. The day's ranked list and template explanations are unchanged.", calls
