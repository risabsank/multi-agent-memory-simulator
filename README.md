# Multi-Agent Memory Simulator

A Python simulation project that models **multi-agent collaborative memory** under different consistency/coherence strategies.

This repository is designed to be both:
- a practical sandbox for testing memory/coherence protocol behavior, and
- a portfolio-ready project that demonstrates systems thinking, simulation design, metrics engineering, and evaluation automation.

---
Modern AI and distributed systems frequently need to coordinate shared state across multiple workers/agents. This project explores that challenge by simulating:

- Multiple agents with local caches,
- A shared global memory,
- Event-driven read/write/sync/invalidate flows,
- Pluggable consistency protocols (strong/eventual/MESI/hybrid),
- Deterministic and LLM-assisted conflict judges,
- Reproducible workload generation, metrics aggregation, and visualization.

---

## Core capabilities

- **Discrete-event simulator** with scheduled operations and event tracing.
- **Protocol abstraction** for consistency behavior.
- **Built-in protocols**:
  - Write-through strong consistency
  - Eventual consistency
  - MESI-inspired coherence model
  - Hybrid strategy
- **Conflict judging system**:
  - Deterministic profiles (`strict`, `balanced`, `lenient`)
  - Optional LLM judge with deterministic fallback
- **Workload generation** for low/medium/high contention bursty traffic.
- **Evaluation matrix** across protocol × judge × regime × scale × seed.
- **Metric exports** (`raw`, `plot-ready`, `summary`) and generated plots.
- **Acceptance gate** for automated pass/fail thresholds in CI-style runs.

---

## Repository structure

```text
src/memory/
  simulator.py               # Core engine, scheduling, run/report logic
  model.py                   # Domain model (agents, memory, artifacts)
  events.py                  # Event definitions and queue
  workload.py                # Workload generation
  protocols/
    strong.py                # Strong write-through protocol
    eventual.py              # Eventual consistency protocol
    mesi.py                  # MESI-inspired protocol
    hybrid.py                # Hybrid protocol
    judges/
      deterministic.py       # Deterministic conflict judge
      llm.py                 # LLM-backed conflict judge + fallback

scenarios/
  trace.py                   # Minimal walkthrough run
  eval_matrix.py             # Full experiment matrix runner
  plot_judge_metrics.py      # Plot generation
  eval_pipeline.py           # End-to-end runner + acceptance gate
  generated/                 # Metrics CSV/JSON, generated workloads, plots

tests/
  test_*.py                  # Unit/integration-style behavior checks
```

---

## Quick start

### 1) Prerequisites

- Python **3.10+**
- `pip`
- (Optional) OpenAI API key for LLM-judge runs

### 2) Install

```bash
pip install -e .
```

### 3) Run a basic trace demo

```bash
PYTHONPATH=src python scenarios/trace.py
```
This prints:
- event-by-event execution trace,
- per-agent latency/hit/miss stats,
- final run report summary,
- final artifact state/version.

---

## Running tests

Using Makefile target:

```bash
make test
```

Equivalent direct command:

```bash
pytest tests/
```

---

## Evaluation workflow (recommended for portfolio demos)

### Option A: One-command end-to-end pipeline

```bash
python scenarios/eval_pipeline.py
```

What this does:
1. Runs `scenarios/eval_matrix.py` (unless `--skip-generate`),
2. Runs plotting (`scenarios/plot_judge_metrics.py`) (unless `--skip-plot`),
3. Applies acceptance thresholds on summary metrics,
4. Returns PASS/FAIL exit status for automation.

### Option B: Manual split

```bash
python scenarios/eval_matrix.py
python scenarios/plot_judge_metrics.py
```

### Useful pipeline flags

```bash
python scenarios/eval_pipeline.py \
  --max-stale-read-rate 0.30 \
  --min-read-your-writes 0.70 \
  --max-contested-ratio 0.90
```

---

## Generated outputs

After evaluation, see:

- `scenarios/generated/evaluation_metrics_raw.csv`
- `scenarios/generated/evaluation_metrics_plot_ready.csv`
- `scenarios/generated/evaluation_metrics_summary.csv`
- `scenarios/generated/evaluation_metrics_raw.json`
- `scenarios/generated/plots/...` (heatmaps + bar charts)
- `scenarios/generated/workloads/...` (replayable workload traces)

These artifacts are ideal for:
- attaching to a portfolio,
- discussing methodology in interviews,
- proving reproducibility and metric rigor.

---

## LLM judge configuration (optional)

LLM judging is supported through the OpenAI Responses API integration in `src/memory/protocols/judges/llm.py`.

Set your key before LLM runs:

```bash
export OPENAI_API_KEY="your_key_here"
```

Notes:
- Default model used by evaluation matrix is currently `gpt-4.1-mini`.
- LLM decisions enforce strict JSON schema parsing.
- Any parse/runtime/timeout issue falls back to deterministic judging, and failure categories are tracked in metrics.

---

## Next steps

- Add richer cache policies (LRU/LFU/size pressure stress tests).
- Add multi-level cache hierarchy.
- Add pluggable transport/network delay models.
- Track cost metrics for LLM judge calls.
- Add dashboard UI for interactive protocol exploration.
- Add CI job that runs fast smoke eval + threshold gate on every PR.
