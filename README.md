# Log2Agent

Mines an enterprise event log to reconstruct how a business process actually runs, ranks each step by automation ROI, generates a LangGraph agent for the best segment (with human approval gates on financial steps), and validates the agent by replaying historical cases through it.

## Demo
https://github.com/user-attachments/assets/599b486b-a64f-4eb9-83b9-0dbd13b81e87

## Motivation

Gartner projects that over 40 percent of agentic AI projects will be cancelled by 2027, mostly for unclear value and poor risk control. Usually the failure is picking the wrong thing to automate and having no evidence it works. Log2Agent picks the target from mined data instead of opinion, and scores the output by replaying history instead of a demo prompt.

## How it works

1. **Ingest.** Load an XES event log, map columns, get a quality report.
2. **Mine.** PM4Py reconstructs the process as a directly-follows graph and a Petri net, including paths absent from the official diagram.
3. **Analyze.** Hand-off wait times per step, cycle time per case.
4. **Rank.** An LLM classifies each activity (deterministic, communication, judgment), then ROI = frequency x median handling hours x cost x automatability, discounted by risk. Every assumption is an editable slider. An MCP tool audit against a mock bank server checks whether a real tool exists for each step and penalizes the score where none does; the match table is editable and your overrides win.
5. **Compile.** The chosen segment becomes LangGraph code: an LLM fills a locked Jinja2 template, the result is AST-checked and smoke-tested in a sandbox. Human-in-the-loop gates are auto-inserted on irreversible steps, so the agent pauses for a person before any payment moves.
6. **Repair.** If the generated code fails its smoke test, a bounded ReAct repair agent debugs it with four tools (run_smoke_test, read_traceback, read_code, patch_code), capped at 5 iterations or 60 seconds, then falls back to a plain retry. An "inject a bug" button on the compile page lets you watch this live.
7. **Validate.** Historical cases are replayed through the generated agent and its routing outcome is compared to what actually happened.

## Results

Measured on the BPI 2020 Request for Payment log (6,886 cases, 36,796 events), replaying 50 real cases through the generated agent:

| Metric | Value |
|---|---|
| Routing outcome agreement | 100% (50 of 50, approve and reject paths) |
| Mined model fitness | 1.0 |
| Mined model precision | 0.29 |
| Median cycle time | 7.1 days (on the replayed sample) |
| Median wait per hand-off | 29.3 hours |
| Steps automated | SUBMITTED, APPROVED by ADMINISTRATION |
| Steps gated for humans | FINAL_APPROVED, Pay, Payment Handled |

Scale test: the BPI 2017 loan log (1,202,267 events, 31,509 cases) mines end to end in about 50 minutes on a laptop, fitness 1.0, precision 0.14. Results are cached so the app loads it instantly.

Notes on the numbers:

- For straight-through cases, most of the 7.1-day cycle is queue time between people, not work. The agent removes the routing wait; that wait is the ceiling on savings, and the realized share depends on how much of each wait was queueing versus actual deliberation.
- Precision is low by design. The Inductive Miner guarantees fitness and soundness at the cost of permissive models; 0.2 to 0.5 is typical on real logs with rework, and 0.14 on the full unfiltered 1.2M-event log fits that pattern. The automation decision rests on frequency and wait analytics plus outcome replay, not on the Petri net's precision.

## Quickstart

```bash
git clone https://github.com/dLounce/log2agent.git
cd log2agent
pip install -r requirements.txt
```

Get the data (free, from 4TU.ResearchData at data.4tu.nl, search the dataset names):

- "BPI Challenge 2020" Request For Payment log → `data/BPI Challenge 2020_ Request For Payment_1_all/RequestForPayment.xes.gz`
- "BPI Challenge 2017" (optional, the big one) → `data/BPI Challenge 2017_1_all/BPI Challenge 2017.xes.gz`, then run `python build_cache.py` once (roughly an hour; mines and caches everything including rendered process maps)

Copy `.env.example` to `.env` and add an OpenAI key, or a Groq key as fallback (set `GROQ_API_KEY`; used only when no OpenAI key is present). The 2020 walkthrough runs without any key because every LLM response it needs is committed in `llm_cache/`. A live key is only needed for the repair demo and uncached prompts.

```bash
streamlit run app.py
```

Or with Docker: `docker compose up`.

## What is real and what is mocked

- The generated agents run for real: executed, smoke-tested, interrupted at human gates, replayed against history.
- The business APIs they call are mocks. Each node calls a stub returning a plausible hardcoded response; wiring to a real system means replacing the stubs.
- MCP tool discovery happens at design time and grounds the feasibility scoring. The generated agents do not call MCP tools at runtime yet; that is the known next step, scoped out for async lifecycle reasons.
- LLM outputs are JSON-mode prompting with parsing, not schema-enforced structured output, kept deterministic with temperature 0 and cached responses.
- Validation compares routing outcomes (did the case complete through the same path), not the content of human decisions.

## Repo layout

```
app.py               Streamlit UI, thin wrapper over pipeline.py
pipeline.py          mining, analytics, ROI, compiler, repair agent, validation
template.j2          the locked LangGraph code template the LLM fills
mockbank_server.py   FastMCP mock bank, target of tool discovery
build_cache.py       one-time offline mining of the 1.2M-event log
generated/           committed agent + validation report
llm_cache/           cached LLM responses for keyless demo reruns
```

## License

MIT
