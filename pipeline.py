"""
Log2Agent pipeline — all core functions.
Extracted from the validated notebook. UI and Docker import from here.
"""
import json, hashlib, ast, subprocess, sys, re, time, pickle
from pathlib import Path
import pandas as pd
import pm4py
from jinja2 import Template
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

def short(s):
    return s.replace("Request For Payment ", "").replace("Request Payment", "Pay")

CACHE_DIR = Path("llm_cache"); CACHE_DIR.mkdir(exist_ok=True)
_llm = None
import os

def _get_llm():
    global _llm
    if _llm is None:
        if os.getenv("OPENAI_API_KEY"):
            _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, seed=42)
        elif os.getenv("GROQ_API_KEY"):
            from langchain_groq import ChatGroq
            _llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        else:
            raise RuntimeError("Set OPENAI_API_KEY or GROQ_API_KEY in .env")
    return _llm

def llm_json(prompt, system="You are a process-mining analyst. Respond ONLY with valid JSON."):
    key = hashlib.sha256((system + "||" + prompt).encode()).hexdigest()[:16]
    cf = CACHE_DIR / f"{key}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    resp = _get_llm().invoke([("system", system), ("human", prompt)])
    text = resp.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    parsed = json.loads(text)
    cf.write_text(json.dumps(parsed, indent=2))
    return parsed

def ingest(path, col_map, keep_attrs=None):
    if path.endswith((".xes", ".xes.gz")):
        raw = pm4py.convert_to_dataframe(pm4py.read_xes(path))
    else:
        raw = pd.read_csv(path, sep=None, engine="python")
    df = raw.rename(columns=col_map)
    cols = ["case_id", "activity", "timestamp", "resource"] + (keep_attrs or [])
    df = df[[c for c in cols if c in df.columns]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    sizes = df.groupby("case_id").size()
    return df[df["case_id"].isin(sizes[sizes > 1].index)].reset_index(drop=True)

def to_pm(df):
    return df.rename(columns={"case_id": "case:concept:name",
                              "activity": "concept:name",
                              "timestamp": "time:timestamp"})

def quality_report(df):
    ev = df.groupby("case_id").size()
    return {"events": len(df), "cases": int(df["case_id"].nunique()),
            "activities": int(df["activity"].nunique()),
            "median_events_per_case": float(ev.median()),
            "time_range": [str(df["timestamp"].min()), str(df["timestamp"].max())]}

def mine(df_pm):
    net, im, fm = pm4py.discover_petri_net_inductive(df_pm)
    fit = pm4py.fitness_token_based_replay(df_pm, net, im, fm)
    prec = pm4py.precision_token_based_replay(df_pm, net, im, fm)
    return {"net": net, "im": im, "fm": fm,
            "fitness": fit["log_fitness"], "precision": prec}

def discover_map(df, top_k_variants=None, out_png=None):
    """Mine the directly-follows graph and render to PNG.
    top_k_variants: keep only the K most common paths (legibility); None = full map.
    Returns the PNG path. UI calls this — no mining logic in app.py."""
    import tempfile, os
    df_pm = to_pm(df)
    if top_k_variants:
        df_pm = pm4py.filter_variants_top_k(df_pm, top_k_variants)
    dfg, start, end = pm4py.discover_dfg(df_pm)
    if out_png is None:
        out_png = os.path.join(tempfile.gettempdir(), f"dfg_top{top_k_variants or 'all'}.png")
    pm4py.save_vis_dfg(dfg, start, end, out_png)
    return out_png

def transition_waits(df):
    d = df.sort_values(["case_id", "timestamp"]).copy()
    d["next_activity"] = d.groupby("case_id")["activity"].shift(-1)
    d["next_case"] = d.groupby("case_id")["case_id"].shift(-1)
    d["wait_hrs"] = (d.groupby("case_id")["timestamp"].shift(-1) - d["timestamp"]).dt.total_seconds()/3600
    d.loc[d["next_case"].isna(), "wait_hrs"] = pd.NA
    trans = d.dropna(subset=["next_activity"])
    return (trans.groupby(["activity", "next_activity"])["wait_hrs"]
            .agg(["count", "median", "mean"]).sort_values("median", ascending=False))

def throughput(df):
    span = df.groupby("case_id")["timestamp"].agg(["min", "max"])
    return (span["max"] - span["min"]).dt.total_seconds() / 86400

def analytics(df):
    tp = throughput(df)
    rw = (df.groupby(["case_id", "activity"]).size() > 1).groupby(level=0).any().sum()
    return {"variants": pm4py.get_variants(to_pm(df)),
            "waits": transition_waits(df),
            "throughput_days": {"median": float(tp.median()), "mean": float(tp.mean()),
                                "p90": float(tp.quantile(0.9))},
            "rework_pct": 100 * rw / df["case_id"].nunique()}


COST_PER_HR = 40
RISK_PENALTY = {"judgment": 0.5, "communication": 0.2, "deterministic": 0.0}

def classify_activities(activities, process_desc="approval process"):
    prompt = f"""Classify each business process activity into ONE category:
- "deterministic": rule-based, no human judgment
- "judgment": requires human decision/evaluation
- "communication": notifying/requesting info

Activities (from a {process_desc}):
{json.dumps(activities, indent=2)}

Return JSON: {{"activity name": {{"category": "...", "reason": "<8 words", "automatable": 0.0-1.0}}}}"""
    return llm_json(prompt)

def score_roi(df, classification, cost_per_hr=COST_PER_HR):
    d = df.sort_values(["case_id", "timestamp"]).copy()
    d["next_case"] = d.groupby("case_id")["case_id"].shift(-1)
    d["wait_hrs"] = (d.groupby("case_id")["timestamp"].shift(-1) - d["timestamp"]).dt.total_seconds()/3600
    d.loc[d["next_case"].isna(), "wait_hrs"] = pd.NA
    waits = d.groupby("activity")["wait_hrs"].median()
    freq = df["activity"].value_counts()
    rows = []
    for act in sorted(df["activity"].unique()):
        c = classification.get(act, {})
        cat = c.get("category", "judgment")
        auto = float(c.get("automatable", 0.1) or 0.1)
        f = int(freq.get(act) or 0)
        h = waits.get(act); h = float(h) if pd.notna(h) else 0.0
        roi = f * h * cost_per_hr * auto * (1 - RISK_PENALTY.get(cat, 0.5))
        rows.append({"activity": short(act), "cat": cat, "freq": f,
                     "med_hrs": round(h, 1), "auto": auto, "roi": round(roi)})
    return pd.DataFrame(rows).sort_values("roi", ascending=False).reset_index(drop=True)

IRREVERSIBLE = ("pay", "final_approved", "handled")

def needs_hitl(act):
    verb = act.lower().replace("request for payment", "").replace("request payment", "pay")
    return any(k in verb for k in IRREVERSIBLE)

def build_automation_spec(seg, classification, name="segment"):
    nodes = []
    for i, act in enumerate(seg):
        c = classification.get(act, {})
        slug = re.sub(r"[^a-z0-9]+", "_", short(act).lower()).strip("_")
        nodes.append({"activity": act, "short": short(act), "var": f"n{i}_{slug}",
                      "category": c.get("category", "judgment"),
                      "automatable": float(c.get("automatable", 0.1) or 0.1),
                      "hitl_gate": needs_hitl(act), "reason": c.get("reason", "")})
    return {"segment_name": name, "description": f"Automate {name}",
            "trigger": seg[0], "nodes": nodes, "cost_per_hr": COST_PER_HR}

def spec_to_dot(spec):
    """DOT string for the agent graph: linear chain, HITL-gated nodes red. UI renders it."""
    lines = ["digraph G {", '  rankdir=LR; node [shape=box, style=rounded, fontsize=11];',
             '  start [shape=circle, label="", width=0.15];',
             '  end [shape=doublecircle, label="", width=0.12];']
    prev = "start"
    for i, n in enumerate(spec["nodes"]):
        nid = f"n{i}"
        if n["hitl_gate"]:
            lines.append(f'  {nid} [label="{n["short"]}\\n(human gate)", color=red, fontcolor=red];')
        else:
            lines.append(f'  {nid} [label="{n["short"]}\\n(automated)", color=green];')
        lines.append(f"  {prev} -> {nid};")
        prev = nid
    lines += [f"  {prev} -> end;", "}"]
    return "\n".join(lines)

LANGGRAPH_TEMPLATE = Template(open("template.j2", encoding="utf-8").read())

def llm_fill_slots(spec_r):
    nodes_desc = [{"var": n["var"], "activity": n["short"], "category": n["category"],
                   "reason": n["reason"], "hitl_gate": n["hitl_gate"]} for n in spec_r["nodes"]]
    prompt = f"""Generate node bodies for a LangGraph agent automating: "{spec_r['description']}"
For EACH node, provide two Python snippets:
- "logic": 1-4 lines. FIRST line MUST be `result = stub()` (the mocked business-API call for this step). You may then adjust `result` using ONLY state keys: case_id (str), requested_amount (float), status (str), history (list). Do NOT reference any other key. NO imports/defs.
- "stub": a dict literal for a plausible return of that business API.
Nodes:
{json.dumps(nodes_desc, indent=2)}
Return JSON: {{"<var>": {{"logic": "...", "stub": {{...}}}}}}. Only vars: {[n['var'] for n in spec_r['nodes']]}"""
    return llm_json(prompt)

def inject_fills(spec_r, fills):
    s = json.loads(json.dumps(spec_r))
    for n in s["nodes"]:
        f = fills.get(n["var"], {})
        raw = (f.get("logic") or "").strip().replace("stub()", f"_stub_{n['var']}(state)")
        call = f"result = _stub_{n['var']}(state)"
        if f"_stub_{n['var']}" in raw:
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        else:
            keep = [ln.strip() for ln in raw.splitlines()
                    if ln.strip() and not re.match(r"result\s*=", ln.strip())]
            lines = [call] + keep
        n["logic"] = "\n    ".join(lines) if lines else call
        stub = f.get("stub")
        n["stub"] = repr(stub) if isinstance(stub, (dict, list)) else (stub.strip() if isinstance(stub, str) and stub.strip() else '{"ok": True}')
    return s

def smoke_test(path, timeout=30):
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True, text=True,
                              timeout=timeout, encoding="utf-8")
        return proc.returncode == 0, (proc.stdout + proc.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def _repair(spec_r, prev, err):
    prompt = f"""The generated agent failed. ERROR:
{err[:1000]}
PREVIOUS bodies:
{json.dumps(prev, indent=2)[:1500]}
Fix so it runs. Same rules. Return SAME JSON, vars: {[n['var'] for n in spec_r['nodes']]}"""
    return llm_json(prompt)

def compile_with_retry(spec_r, out_path, max_retries=3):
    fills = llm_fill_slots(spec_r)
    log = []
    for attempt in range(1, max_retries + 1):
        code = LANGGRAPH_TEMPLATE.render(**inject_fills(spec_r, fills))
        try:
            ast.parse(code)
        except SyntaxError as e:
            log.append(f"attempt {attempt}: AST fail L{e.lineno}"); fills = _repair(spec_r, fills, f"SyntaxError L{e.lineno}: {e.msg}"); continue
        Path(out_path).write_text(code, encoding="utf-8")
        ok, out = smoke_test(out_path)
        if ok:
            log.append(f"attempt {attempt}: PASS"); return code, True, log
        log.append(f"attempt {attempt}: smoke fail"); fills = _repair(spec_r, fills, out)
    return code, False, log

import importlib.util
from langgraph.types import Command

def load_agent(path):
    spec = importlib.util.spec_from_file_location("gen_agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run_case_auto(app, initial, decision="approve", max_resumes=10):
    config = {"configurable": {"thread_id": f"case-{initial['case_id']}"}}
    gates = []
    state = app.invoke(initial, config=config)
    for _ in range(max_resumes):
        if isinstance(state, dict) and state.get("__interrupt__"):
            gates.append(state["__interrupt__"][0].value.get("step", "?"))
            state = app.invoke(Command(resume=decision), config=config)
        else:
            break
    return state, gates

def reconstruct_cases(df, seg_activities, n=100, seed=42, amount_col="case:RequestedAmount"):
    import random
    seg_set = set(seg_activities)
    cases = []
    for cid, grp in df.sort_values("timestamp").groupby("case_id"):
        acts = list(grp["activity"])
        if not set(acts).issubset(seg_set):
            continue
        amt = float(grp[amount_col].iloc[0]) if amount_col in grp else 0.0
        cases.append({"case_id": cid, "requested_amount": amt,
                      "actual_activities": acts,
                      "actual_completed": seg_activities[-1] in acts})
    random.seed(seed)
    return random.sample(cases, min(n, len(cases)))

def validate_agent(app_builder, cases, policy):
    results = []
    for c in cases:
        app = app_builder()
        initial = {"case_id": c["case_id"], "requested_amount": c["requested_amount"],
                   "status": "new", "history": []}
        try:
            final, gates = run_case_auto(app, initial, decision=policy(c))
            done = final.get("history", [])
            halted = any("REJECTED" in h for h in done) or str(final.get("status","")).startswith("halted")
            agent_completed = not halted
            match = agent_completed == c["actual_completed"]
        except Exception as e:
            agent_completed, match, gates = None, False, [f"ERR:{type(e).__name__}"]
        results.append({"case_id": c["case_id"], "actual": c["actual_completed"],
                        "agent": agent_completed, "match": match})
    return results

def routing_savings(df, case_ids, seg_activities):
    """Median hand-off/queue time the agent removes, computed on the replayed cases.
    Routing transitions = consecutive segment steps. Returns None values if too little data."""
    pairs = set(zip(seg_activities, seg_activities[1:]))
    d = df[df["case_id"].isin(case_ids)].sort_values(["case_id", "timestamp"]).copy()
    d["next_act"] = d.groupby("case_id")["activity"].shift(-1)
    d["wait_h"] = (d.groupby("case_id")["timestamp"].shift(-1)
                   - d["timestamp"]).dt.total_seconds() / 3600
    m = d.dropna(subset=["next_act"])
    m = m[[(a, b) in pairs for a, b in zip(m["activity"], m["next_act"])]]
    if len(m) < 10:
        return {"per_handoff_hrs": None, "per_case_hrs": None, "per_case_days": None}
    per_handoff = float(m["wait_h"].median())
    per_case = float(m.groupby("case_id")["wait_h"].sum().median())
    return {"per_handoff_hrs": round(per_handoff, 1),
            "per_case_hrs": round(per_case, 1),
            "per_case_days": round(per_case / 24, 1)}

def validator_report(df, cases, results, mining_result, spec,
                     out_path="generated/validation_report.json"):
    agreement = sum(r["match"] for r in results) / len(results)
    auto_steps = [n["short"] for n in spec["nodes"] if not n["hitl_gate"]]
    gated = [n["short"] for n in spec["nodes"] if n["hitl_gate"]]
    ids = [c["case_id"] for c in cases]
    tp = throughput(df[df["case_id"].isin(ids)])
    fit = mining_result.get("fitness")
    prec = mining_result.get("precision")
    seg = [n["activity"] for n in spec["nodes"]]
    rs = routing_savings(df, ids, seg)
    report = {"agreement_pct": round(agreement * 100, 1),
            "n_cases_validated": len(results),
            "model_fitness": round(fit, 4) if fit is not None else "n/a",
            "model_precision": round(prec, 4) if prec is not None else "n/a",
            "current_median_cycle_days": round(float(tp.median()), 1),
            "median_routing_wait_per_handoff_hrs": rs["per_handoff_hrs"],
            "projected_routing_time_saved_hrs": rs["per_case_hrs"],
            "projected_routing_time_saved_days": rs["per_case_days"],
            "value_framing": ("Agent removes hand-off/queue delay between steps; humans retain "
                              "judgment at all financial gates. Savings = coordination latency, "
                              "not decision time."),
            "automatable_steps": auto_steps, "hitl_gated_steps": gated}
    if out_path:
        Path(out_path).parent.mkdir(exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report

from langchain_core.tools import tool

_repair_ctx = {"path": None, "last_error": ""}

def _set_repair_ctx(path):
    _repair_ctx["path"] = path
    _repair_ctx["last_error"] = ""

@tool
def run_smoke_test() -> str:
    """Execute the generated agent file in a sandbox. Returns PASS or the failure output."""
    ok, out = smoke_test(_repair_ctx["path"])
    _repair_ctx["last_error"] = "" if ok else out
    return "PASS" if ok else f"FAIL:\n{out[:800]}"

@tool
def read_traceback() -> str:
    """Return the last captured error from the most recent smoke test, cleaned."""
    err = _repair_ctx["last_error"]
    if not err:
        return "No error recorded. Run run_smoke_test first."
    return err[-800:]

@tool
def read_code(section: str = "all") -> str:
    """Read the current generated file. section='all' for whole file, or a function name to grep that def."""
    code = Path(_repair_ctx["path"]).read_text(encoding="utf-8")
    if section == "all":
        return code[:4000]
    lines = code.splitlines()
    for i, ln in enumerate(lines):
        if f"def {section}" in ln:
            return "\n".join(lines[i:i+15])
    return f"Section '{section}' not found. Use 'all' to see the file."

@tool
def patch_code(section: str, new_code: str) -> str:
    """Replace one function (named by section) with new_code. AST-validates before writing; rejects if it breaks parse."""
    code = Path(_repair_ctx["path"]).read_text(encoding="utf-8")
    lines = code.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if f"def {section}" in ln:
            start = i; break
    if start is None:
        return f"REJECTED: function '{section}' not found."
    end = len(lines)
    for j in range(start+1, len(lines)):
        if lines[j] and not lines[j][0].isspace() and lines[j].lstrip().startswith(("def ", "class ", "if __")):
            end = j; break
    candidate = "\n".join(lines[:start] + new_code.splitlines() + lines[end:])
    try:
        ast.parse(candidate)
    except SyntaxError as e:
        return f"REJECTED: patch causes SyntaxError L{e.lineno}: {e.msg}. Not written."
    Path(_repair_ctx["path"]).write_text(candidate, encoding="utf-8")
    return f"Patched '{section}' successfully. Run run_smoke_test to verify."

REPAIR_TOOLS = [run_smoke_test, read_traceback, read_code, patch_code]
def inject_bug(path):
    """Demo: deterministically break the first node (NameError) so the repair agent fixes it live."""
    p = Path(path)
    code = p.read_text(encoding="utf-8")
    for line in code.splitlines():
        if "result = _stub_" in line and "(state)" in line:
            broken = line.replace("(state)", "(stat)")
            p.write_text(code.replace(line, broken, 1), encoding="utf-8")
            return broken.strip()
    return None

from langgraph.prebuilt import create_react_agent

REPAIR_SYSTEM = """You are a code-repair agent. A generated Python file failed its smoke test.
Your goal: make it PASS, using the fewest tool calls.

Workflow:
1. run_smoke_test to see the current failure.
2. read_traceback to get the exact error.
3. read_code(section=<function name from the error>) to inspect the broken function.
4. patch_code(section, new_code) with a minimal fix — replace ONLY the broken function.
5. run_smoke_test again to confirm.

Rules:
- Patch the smallest thing that fixes the error. Do not rewrite unrelated code.
- The state TypedDict has EXACTLY: case_id, requested_amount, status, history. Never reference other keys.
- If patch_code returns REJECTED, read the reason and try a corrected patch.
- Stop as soon as run_smoke_test returns PASS.
"""

def repair_agent_fix(path, max_iters=5, timeout_s=60, on_event=None):
    """Run the ReAct repair agent on a failing file. Returns (passed, trace_of_tool_calls)."""
    _set_repair_ctx(path)
    agent = create_react_agent(_get_llm(), REPAIR_TOOLS, prompt=REPAIR_SYSTEM)

    trace = []
    def _log(line):
        trace.append(line)
        if on_event:
            on_event(line)
    start = time.time()
    ok, out = smoke_test(path)
    if ok:
        return True, ["already passing"]
    _repair_ctx["last_error"] = out

    msg = {"messages": [("human", f"The file at the sandbox failed. Error:\n{out[:600]}\nFix it.")]}
    config = {"recursion_limit": max_iters * 3}
    try:
        for chunk in agent.stream(msg, config=config, stream_mode="updates"):
            if time.time() - start > timeout_s:
                trace.append("TIMEOUT — falling back")
                break
            for node, update in chunk.items():
                for m in update.get("messages", []):
                    if hasattr(m, "tool_calls") and m.tool_calls:
                        for tc in m.tool_calls:
                            trace.append(f"CALL {tc['name']}({str(tc['args'])[:60]})")
                    elif m.__class__.__name__ == "ToolMessage":
                        trace.append(f"  -> {str(m.content)[:80]}")
            passed, _ = smoke_test(path)
            if passed:
                trace.append("PASS — repaired")
                return True, trace
    except Exception as e:
        trace.append(f"agent error: {type(e).__name__}: {str(e)[:80]}")

    passed, _ = smoke_test(path)
    return passed, trace

def compile_with_agent_repair(spec_r, out_path, use_agent=True, max_retries=3):
    """Compile the agent. If smoke fails:
       - use_agent=True: try the ReAct Repair Agent (visible trace), fall back to simple retry if it can't fix.
       - use_agent=False: simple retry loop only (the P4a path).
       Returns (code, passed, log) where log includes the repair trace if the agent ran."""
    fills = llm_fill_slots(spec_r)
    code = LANGGRAPH_TEMPLATE.render(**inject_fills(spec_r, fills))
    log = []

    try:
        ast.parse(code)
        Path(out_path).write_text(code, encoding="utf-8")
        ok, out = smoke_test(out_path)
    except SyntaxError as e:
        Path(out_path).write_text(code, encoding="utf-8")
        ok, out = False, f"SyntaxError L{e.lineno}: {e.msg}"

    if ok:
        log.append("compiled: PASS on first attempt")
        return code, True, log

    log.append("first attempt failed")

    if use_agent:
        log.append("--- Repair Agent engaged ---")
        fixed, trace = repair_agent_fix(out_path)
        log.extend(trace)
        if fixed:
            log.append("Repair Agent: SUCCESS")
            return Path(out_path).read_text(encoding="utf-8"), True, log
        log.append("Repair Agent could not fix -> falling back to simple retry")

    _, ok2, simple_log = compile_with_retry(spec_r, out_path, max_retries=max_retries)
    log.append("--- simple retry fallback ---")
    log.extend(simple_log)
    return Path(out_path).read_text(encoding="utf-8"), ok2, log

import asyncio as _asyncio

MOCKBANK_PATH = "mockbank_server.py"

def discover_mcp_tools(server_path=MOCKBANK_PATH):
    """One-shot connect to the MCP server, list tools, disconnect.
    Returns [(name, description), ...]. Synchronous wrapper over an async call."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def _list(path):
        params = StdioServerParameters(command="python", args=[path])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return [{"name": t.name, "description": t.description} for t in result.tools]

    try:
        return _asyncio.run(_list(server_path))
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            return ex.submit(lambda: _asyncio.run(_list(server_path))).result()

def match_activities_to_tools(activities, tools):
    """LLM matches each activity to the best-fitting MCP tool (or none).
    Returns {activity: {"tool": name_or_None, "confidence": 0-1, "reason": str}}.
    This match is FUZZY by nature — the UI must render it as editable, not asserted."""
    tool_lines = [f"- {t['name']}: {t['description']}" for t in tools]
    prompt = f"""Match each business process activity to the MCP tool that would implement it, if any.

Available MCP tools:
{chr(10).join(tool_lines)}

Activities to match:
{json.dumps([short(a) for a in activities], indent=2)}

For each activity, pick the single best-fitting tool, or null if no tool fits.
Return JSON: {{"<activity short name>": {{"tool": "<tool_name or null>", "confidence": 0.0-1.0, "reason": "<8 words>"}}}}"""
    raw = llm_json(prompt)
    out = {}
    short_to_full = {short(a): a for a in activities}
    for k, v in raw.items():
        full = short_to_full.get(k, k)
        out[full] = v
    return out

def tool_match_table(activities, matches, tools):
    """Build a display/edit table: activity, matched tool, confidence, tool_exists flag."""
    tool_names = {t["name"] for t in tools}
    rows = []
    for act in activities:
        m = matches.get(act, {})
        tool = m.get("tool")
        rows.append({
            "activity": short(act),
            "matched_tool": tool if tool in tool_names else "(none)",
            "confidence": round(float(m.get("confidence", 0) or 0), 2),
            "tool_exists": tool in tool_names,
            "reason": m.get("reason", ""),
        })
    return rows

def apply_tool_grounding(roi_df, match_table, no_tool_multiplier=0.5):
    """Merge the MCP tool audit into the ROI table (join on short activity name).
    Where no tool exists, automatability & ROI scale by no_tool_multiplier
    (visible assumption). Activities not in the audit keep their score
    (tool_exists = None = not audited). Always derives from the base roi_df,
    so re-applying never compounds the penalty."""
    m = pd.DataFrame(match_table)
    lookup = dict(zip(m["activity"], m["tool_exists"].astype(bool)))
    out = roi_df.copy()
    out["tool_exists"] = out["activity"].map(lookup)
    pen = out["tool_exists"].eq(False)
    out.loc[pen, "auto"] = (out.loc[pen, "auto"] * no_tool_multiplier).round(2)
    out.loc[pen, "roi"] = (out.loc[pen, "roi"] * no_tool_multiplier).round()
    return out.sort_values("roi", ascending=False).reset_index(drop=True)
