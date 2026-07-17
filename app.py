"""
Log2Agent — Streamlit UI. Thin wrapper over pipeline.py (zero logic rewritten here).
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import json, pickle
from pathlib import Path
import pipeline as pl

st.set_page_config(page_title="Log2Agent", layout="wide")

DATASETS = {
    "BPI 2020 — Request for Payment (fast)": {
        "path": "data/BPI Challenge 2020_ Request For Payment_1_all/RequestForPayment.xes.gz",
        "col_map": {"case:concept:name": "case_id", "concept:name": "activity",
                    "time:timestamp": "timestamp", "org:role": "resource"},
        "amount_col": "case:RequestedAmount",
        "process_desc": "Request-for-Payment approval process",
        "segment": ["Request For Payment SUBMITTED by EMPLOYEE",
                    "Request For Payment APPROVED by ADMINISTRATION",
                    "Request For Payment FINAL_APPROVED by SUPERVISOR",
                    "Request Payment", "Payment Handled"],
        "cache": None,
    },
    "BPI 2017 — Loan Applications (1.2M events, cached)": {
        "path": "data/BPI Challenge 2017_1_all/BPI Challenge 2017.xes.gz",
        "col_map": {"case:concept:name": "case_id", "concept:name": "activity",
                    "time:timestamp": "timestamp", "org:resource": "resource"},
        "amount_col": "case:RequestedAmount",
        "process_desc": "bank loan application process",
        "segment": None,
        "cache": "cache/bpi2017_mined.pkl",
    },
}

for key in ["df", "dataset_cfg", "analytics", "classification", "roi",
            "spec", "gen_code", "compile_log", "report"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.sidebar.title("Log2Agent")
st.sidebar.caption("Logs → mined process → ROI → deployable agent")

ds_name = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
st.session_state.dataset_cfg = DATASETS[ds_name]

page = st.sidebar.radio("Pipeline", [
    "0. Home", "1. Load your logs", "2. See your real process", "3. Where time is wasted",
    "4. What's worth automating", "5. Generate the agent", "6. Proof it works"])

st.sidebar.divider()
_steps = [("Logs loaded", st.session_state.df is not None),
          ("ROI scored", st.session_state.roi is not None),
          ("Agent compiled", st.session_state.gen_code is not None),
          ("Validated", st.session_state.report is not None)]
for label, ok in _steps:
    st.sidebar.caption(("✅ " if ok else "⬜ ") + label)
_next = next((l for l, ok in _steps if not ok), None)
if _next:
    st.sidebar.caption(f"**Next → {_next}**")

st.title(page)

if st.session_state.df is None and page not in ("0. Home", "1. Load your logs"):
    st.warning("Load a dataset on the **Load your logs** page first.")
    st.stop()

if page == "0. Home":
    st.write("**Log2Agent finds which parts of a business process are worth automating — "
             "then writes the automation.**")
    st.write("It mines your raw event logs to reconstruct the real process, ranks each step "
             "by automation ROI with every assumption visible, and compiles the best segment "
             "into a working LangGraph agent with human approval gates on financial steps.")
    st.write("Proof, not promises: the generated agent replays real historical cases and is "
             "scored on agreement with what actually happened.")
    st.graphviz_chart("""digraph { rankdir=LR; node [shape=box, style=rounded];
        "Raw logs" -> "Mined process" -> "ROI ranking" -> "Generated agent" -> "Replay validation" }""")
    st.caption("Pick a dataset in the sidebar, then start with **Load your logs**.")
    st.divider()
    st.subheader("One-click demo")
    st.caption("Runs steps 1–4 on the 2020 payment log from caches (keyless). "
               "Then walk pages 5–6 live: compile, inject-bug, replay.")
    cfg = st.session_state.dataset_cfg
    if not cfg.get("segment"):
        st.info("Select the BPI2020 dataset in the sidebar to enable the demo.")
    elif st.button("▶ Run demo (steps 1–4)", type="primary"):
        prog = st.status("Running demo...", expanded=True)
        prog.write("1/4 Ingesting log...")
        st.session_state.df = pl.ingest(cfg["path"], cfg["col_map"], keep_attrs=[cfg["amount_col"]])
        st.session_state["_cached_mine"] = None
        st.session_state["_cached_maps"] = {}
        st.session_state["_dfg_imgs"] = {}
        prog.write("2/4 Classifying activities (cached LLM)...")
        acts = sorted(st.session_state.df["activity"].unique())
        st.session_state.classification = pl.classify_activities(acts, cfg["process_desc"])
        prog.write("3/4 Scoring automation ROI...")
        st.session_state.roi = pl.score_roi(st.session_state.df, st.session_state.classification)
        prog.write("4/4 Tool audit (MCP discovery)...")
        try:
            disc = pl.discover_mcp_tools()
            m = pl.match_activities_to_tools(cfg["segment"], disc)
            st.session_state["_mcp_tools"] = disc
            st.session_state["_mcp_table"] = pl.tool_match_table(cfg["segment"], m, disc)
        except Exception as e:
            prog.write(f"MCP audit skipped ({type(e).__name__}) — run it on page 4.")
        prog.update(label="Demo ready — pages 1–4 are populated. Go to page 5 to compile.", state="complete")

elif page == "1. Load your logs":
    cfg = st.session_state.dataset_cfg
    st.write(f"**Source:** `{cfg['path']}`")
    st.write(f"**Process:** {cfg['process_desc']}")

    if cfg["cache"] and Path(cfg["cache"]).exists():
        st.info("This dataset has a pre-mined cache (1.2M events). Loading is instant.")

    if st.button("Load & ingest", type="primary"):
        with st.spinner("Ingesting..."):
            if cfg["cache"] and Path(cfg["cache"]).exists():
                with open(cfg["cache"], "rb") as f:
                    cached = pickle.load(f)
                st.session_state.df = cached["df_clean"]
                st.session_state["_cached_mine"] = {"net": cached["net"], "im": cached["im"],
                    "fm": cached["fm"], "fitness": cached.get("fitness"),
                    "precision": cached.get("precision")}
                st.session_state["_cached_maps"] = cached.get("maps") or {}
            else:
                st.session_state.df = pl.ingest(cfg["path"], cfg["col_map"],
                                                keep_attrs=[cfg["amount_col"]])
                st.session_state["_cached_mine"] = None
                st.session_state["_cached_maps"] = {}
            st.session_state["_dfg_imgs"] = {}
        st.success("Ingested.")

    if st.session_state.df is not None:
        df = st.session_state.df
        q = pl.quality_report(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cases", f"{q['cases']:,}")
        c2.metric("Events", f"{q['events']:,}")
        c3.metric("Activities", q["activities"])
        c4.metric("Median events/case", f"{q['median_events_per_case']:.0f}")
        st.caption(f"Time range: {q['time_range'][0][:10]} → {q['time_range'][1][:10]}")
        st.dataframe(df.head(20), width='stretch')

elif page == "2. See your real process":
    df = st.session_state.df
    st.write("Directly-follows graph mined from the log (frequency overlay).")
    full = st.toggle("Show full complexity (all variants)", value=False,
                     help="Default shows the 10 most common paths for legibility.")
    top_k = None if full else 10
    key = "full" if full else "top10"

    if st.button("Discover process map", type="primary"):
        cached_maps = st.session_state.get("_cached_maps") or {}
        if key in cached_maps:
            st.session_state.setdefault("_dfg_imgs", {})[key] = cached_maps[key]
        else:
            with st.spinner("Mining directly-follows graph..."):
                st.session_state.setdefault("_dfg_imgs", {})[key] = \
                    pl.discover_map(df, top_k_variants=top_k)

    img = (st.session_state.get("_dfg_imgs") or {}).get(key)
    if img:
        st.image(img, width='stretch')
        st.caption(("All variants shown. " if full else
                    "Top 10 variants shown — toggle above for everything. ")
                   + "Nodes = activities, edges = transitions; thicker/darker = more frequent path.")

elif page == "3. Where time is wasted":
    df = st.session_state.df
    if st.button("Run analytics", type="primary"):
        with st.spinner("Computing variants, bottlenecks, rework..."):
            st.session_state.analytics = pl.analytics(df)
    a = st.session_state.analytics
    if a:
        tp = a["throughput_days"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Distinct variants", f"{len(a['variants']):,}")
        c2.metric("Median cycle", f"{tp['median']:.1f} d")
        c3.metric("P90 cycle", f"{tp['p90']:.1f} d")
        c4.metric("Rework", f"{a['rework_pct']:.1f}%")

        st.subheader("Top variants")
        vlist = sorted(a["variants"].items(), key=lambda x: x[1], reverse=True)[:8]
        total = df["case_id"].nunique()
        vrows = [{"cases": cnt, "%": round(100*cnt/total, 1),
                  "path": " → ".join(pl.short(x)[:20] for x in trace[:6]) +
                          ("..." if len(trace) > 6 else "")}
                 for trace, cnt in vlist]
        st.dataframe(pd.DataFrame(vrows), width='stretch', hide_index=True)

        st.subheader("Slowest transitions (bottlenecks)")
        w = a["waits"]
        w = w[w["count"] >= 20].head(10).copy()
        w.index = [f"{pl.short(x[0])[:18]} → {pl.short(x[1])[:18]}" for x in w.index]
        st.dataframe(w[["count", "median"]].round(1), width='stretch')

elif page == "4. What's worth automating":
    df = st.session_state.df
    cfg = st.session_state.dataset_cfg
    st.write("LLM classifies each activity, then ranks by automation ROI.")
    cost = st.slider("Assumed loaded cost per hour (currency)", 10, 150, 40, 5,
                     help="Drives the savings estimate. Shown as an assumption, not a fixed truth.")

    if st.button("Classify & score", type="primary"):
        with st.spinner("Classifying activities (LLM, cached)..."):
            acts = sorted(df["activity"].unique())
            st.session_state.classification = pl.classify_activities(acts, cfg["process_desc"])
        with st.spinner("Scoring ROI..."):
            st.session_state.roi = pl.score_roi(df, st.session_state.classification, cost_per_hr=cost)

    if st.session_state.roi is not None:
        roi = st.session_state.roi
        st.subheader("Automation candidates (ranked by ROI)")
        st.caption("category from LLM; ROI = freq x median handling hrs x cost x automatability - risk. "
                   "All assumptions visible above.")
        st.dataframe(roi, width='stretch', hide_index=True,
                     column_config={"roi": st.column_config.NumberColumn("ROI", format="%d")})

        st.divider()
        st.subheader("Tool grounding (MCP)")
        st.caption("Automatability is stronger when a real tool exists to do the step. "
                   "This connects to the MockBank MCP server, discovers its tools, and matches them "
                   "to activities. The match is LLM-generated — review and edit it below.")
        if st.button("Discover tools & match"):
            with st.spinner("Connecting to MockBank MCP server (list_tools)..."):
                try:
                    disc = pl.discover_mcp_tools()
                    st.session_state["_mcp_tools"] = disc
                    seg = st.session_state.dataset_cfg.get("segment") or sorted(df["activity"].unique())[:6]
                    m = pl.match_activities_to_tools(seg, disc)
                    st.session_state["_mcp_table"] = pl.tool_match_table(seg, m, disc)
                    st.success(f"Discovered {len(disc)} tools from the MCP server.")
                except Exception as e:
                    st.error(f"MCP discovery failed: {e}")
        if st.session_state.get("_mcp_table"):
            st.write("**Discovered tools:** " +
                     ", ".join(t["name"] for t in st.session_state["_mcp_tools"]))
            st.caption("Edit the matched_tool / tool_exists cells to override the LLM's mapping.")
            edited = st.data_editor(pd.DataFrame(st.session_state["_mcp_table"]),
                                    width='stretch', hide_index=True, key="mcp_editor")
            grounded = int(edited["tool_exists"].sum())
            st.metric("Activities with a matching tool", f"{grounded}/{len(edited)}")
            st.info("Feasibility grounded in live tool discovery. Overrides are respected — "
                    "the table above is editable, not an opaque assertion.")
            mult = st.slider("Automatability multiplier when no tool exists (assumption)",
                             0.0, 1.0, 0.5, 0.05,
                             help="Activities without a discovered tool get automatability × this. "
                                  "Edit tool_exists above to override the audit.")
            roi_g = pl.apply_tool_grounding(st.session_state.roi,
                                            edited.to_dict("records"),
                                            no_tool_multiplier=mult)
            st.subheader("Grounded ranking (tool audit applied)")
            st.caption("Blank tool_exists = not in the audited segment — score unchanged.")
            st.dataframe(roi_g, width='stretch', hide_index=True,
                         column_config={"roi": st.column_config.NumberColumn("ROI", format="%d")})
            auto_rows = roi_g[roi_g["cat"].isin(["deterministic", "communication"])]["activity"].tolist()
            human_rows = roi_g[roi_g["cat"] == "judgment"]["activity"].tolist()
            st.success(f"**Automate:** {', '.join(auto_rows[:4])}")
            st.info(f"**Keep human:** {', '.join(human_rows[:4])} — judgment calls; the agent routes, humans decide.")

elif page == "5. Generate the agent":
    df = st.session_state.df
    cfg = st.session_state.dataset_cfg
    if st.session_state.classification is None:
        st.warning("Run **What's worth automating** first (needs activity classification).")
        st.stop()
    if not cfg.get("segment"):
        st.warning("No pre-defined automation segment for this dataset. "
                   "(2020 RfP has one; 2017 segment selection is future work.)")
        st.stop()

    seg = cfg["segment"]
    st.write("**Segment to automate:**")
    st.code(" → ".join(pl.short(a) for a in seg))

    spec = pl.build_automation_spec(seg, st.session_state.classification, name="agent")
    gated = [n["short"] for n in spec["nodes"] if n["hitl_gate"]]
    st.write(f"**HITL gates (auto-inserted on irreversible steps):** {', '.join(gated)}")
    st.graphviz_chart(pl.spec_to_dot(spec))
    st.caption("Green = agent acts alone. Red = execution pauses; a human approves in the UI before it proceeds.")

    use_agent = st.checkbox("Use Repair Agent (tool-using self-heal)", value=True,
                            help="ON: a ReAct agent debugs failures via tool calls, falls back to simple retry. OFF: simple retry loop only.")
    if st.button("Compile agent", type="primary"):
        Path("generated").mkdir(exist_ok=True)
        out = "generated/ui_agent.py"
        with st.spinner("LLM fills template → AST check → smoke test → repair..."):
            code, ok, log = pl.compile_with_agent_repair(spec, out, use_agent=use_agent)
        st.session_state.spec = spec
        st.session_state.gen_code = code
        st.session_state.compile_log = log
        st.session_state["_agent_path"] = out
        if ok:
            st.success(f"Compiled & smoke-tested in {len(log)} attempt(s).")
            st.info("What just happened: the LLM filled a locked code template → syntax-checked (AST) → "
                    "executed in a sandbox → self-repaired if broken. You're looking at verified, runnable code.")
        else:
            st.error("Compilation failed after retries (fallback would apply).")

    if st.session_state.compile_log:
        st.subheader("Compiler trace")
        for line in st.session_state.compile_log:
            st.text("  " + line)
    if st.session_state.gen_code:
        with st.expander("View generated LangGraph agent"):
            st.code(st.session_state.gen_code, language="python")

    if st.session_state.get("_agent_path") and st.session_state.gen_code:
        st.divider()
        st.subheader("Live repair demo")
        st.caption("Sabotage one node in the compiled agent, then watch the repair agent "
                   "fix it — tool call by tool call. Same flow it uses on real failures.")
        if st.button("Inject a bug & watch it self-repair"):
            broken = pl.inject_bug(st.session_state["_agent_path"])
            if not broken:
                st.error("No injectable line found — recompile first.")
            else:
                st.error(f"Sabotaged: `{broken}`  → NameError")
                with st.status("Repair agent working...", expanded=True) as status:
                    passed, trace = pl.repair_agent_fix(st.session_state["_agent_path"])
                    st.write(f"{len(trace)} repair events:")
                    for line in trace:
                        st.write(line)
                    if passed:
                        status.update(label="Repaired — smoke test PASS", state="complete")
                        st.session_state.gen_code = Path(
                            st.session_state["_agent_path"]).read_text(encoding="utf-8")
                    else:
                        status.update(label="Repair failed within bounds", state="error")

elif page == "6. Proof it works":
    df = st.session_state.df
    cfg = st.session_state.dataset_cfg
    if st.session_state.spec is None or not st.session_state.get("_agent_path"):
        st.warning("Compile an agent first (**Generate the agent** page).")
        st.stop()

    n = st.slider("Cases to replay", 20, 200, 60, 10)
    if st.button("Replay historical cases", type="primary"):
        seg = cfg["segment"]
        with st.spinner(f"Replaying {n} real cases through the generated agent..."):
            cases = pl.reconstruct_cases(df, seg, n=n, amount_col=cfg["amount_col"])
            agent_mod = pl.load_agent(st.session_state["_agent_path"])
            results = pl.validate_agent(agent_mod.build_graph, cases,
                                        lambda c: "approve" if c["actual_completed"] else "reject")
            if st.session_state.get("_cached_mine"):
                cm = st.session_state["_cached_mine"]
                if cm.get("fitness") is None:
                    st.warning("Cache predates conformance metrics — rebuild cache/bpi2017_mined.pkl.")
                mr = cm
            else:
                mr = pl.mine(pl.to_pm(df))
            st.session_state.report = pl.validator_report(df, cases, results, mr, st.session_state.spec)
    r = st.session_state.report
    if r:
        c1, c2, c3 = st.columns(3)
        c1.metric("Agreement", f"{r['agreement_pct']}%",
                  help="Agent outcome matches historical outcome (completed vs rejected).")
        c2.metric("Cases replayed", r["n_cases_validated"])
        c3.metric("Model fitness", r["model_fitness"])
        st.success(
            f"**Punchline:** {r['agreement_pct']:.0f}% agreement on {r['n_cases_validated']} replayed cases. "
            f"Median case takes {r['current_median_cycle_days']} days — effectively all of it is queue time "
            f"between steps, not work ({r['median_routing_wait_per_handoff_hrs']}h per hand-off). "
            f"The agent removes that wait — the savings ceiling; humans still approve every payment.")
        if r.get("projected_routing_time_saved_days"):
            st.metric("Routing time saved / case", 
                      f"{r['projected_routing_time_saved_days']} d")
        st.write(f"**Automated steps:** {', '.join(r['automatable_steps'])}")
        st.write(f"**Human-gated steps:** {', '.join(r['hitl_gated_steps'])}")
        st.info("Agreement = agent reproduces historical **routing** decisions. "
                "Judgment stays human (HITL). Savings = coordination latency, not decision-making.")
        with st.expander("Full evidence report (JSON)"):
            st.json(r)