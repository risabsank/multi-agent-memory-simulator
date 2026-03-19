## Evaluation process (protocols + judges)

This section defines a repeatable process to evaluate:
1. **Protocol behavior** (strong, eventual, MESI, hybrid, etc.).
2. **Judge behavior** (deterministic profiles vs LLM-based judging).

### 1) Evaluation goals

- Measure **correctness guarantees** under realistic multi-agent contention.
- Measure **performance and scalability** (latency, cache efficiency, event overhead).
- Measure **judge quality and reliability** (decision consistency, fallback rates, latency).
- Quantify **trade-offs** between stricter coherence and throughput/latency.

### 2) What we evaluate

#### A. Protocol-level outcomes
- Cache behavior: hit/miss profile and hit rate.
- Conflict handling: contested vs accepted writes.
- Consistency behavior: stale reads, visibility lag, convergence time.
- Session semantics: read-your-writes success, monotonic read violations.
- End-to-end cost: read/write latency distribution (avg/p50/p95/p99), total events.

#### B. Judge-level outcomes
- Provider usage mix (deterministic vs LLM provider counts).
- Decision outputs (reason code distributions, contested ratio).
- Reliability: fallback count and LLM failure categories.
- Cost/performance: average judge latency.

### 3) Experimental design

Run each experiment as a matrix over:
- **Protocol**: `strong`, `eventual`, `mesi`, `hybrid`.
- **Judge mode**: deterministic (`strict`, `balanced`, `lenient`) and LLM.
- **Workload regime**: low/medium/high contention bursty workloads.
- **Scale points**: vary number of agents, artifacts, and run duration.

Control variables:
- Keep simulator seed fixed per replicate group for reproducibility.
- Keep memory/cache sizing and base latencies fixed unless explicitly testing them.
- Run multiple seeds per config and aggregate mean + variance/confidence intervals.

### 4) Step-by-step execution flow

1. **Define the scenario set**
   - Select agent counts, artifact sets, contention intensity, and workload duration.
   - Use `generate_bursty_workload(...)` with fixed seeds for repeatability.

2. **Generate workloads**
   - For each regime, generate operation traces (read/write mixes, burst patterns).
   - Store config + seed with each run for exact replay.

3. **Run simulation matrix**
   - For each (protocol, judge, workload, scale, seed) combination:
     - initialize simulator and global memory;
     - schedule workload operations;
     - execute `sim.run()` and collect `sim.build_report()`.

4. **Collect raw outputs**
   - Persist run metadata, full report metrics, and optional event trace slices.
   - Keep judge diagnostics (`reason_code_counts`, `fallback_count`, `llm_failure_categories`).

5. **Aggregate and normalize**
   - Group by protocol + judge + workload regime.
   - Compute central tendency and spread (mean/std and/or percentile bands).
   - Normalize selected metrics per operation when comparing different run lengths.

6. **Graph and compare**
   - Produce the plots listed below.
   - Compare protocol families first, then judge variants within each protocol.

7. **Interpretation + decision**
   - Select best protocol/judge pairs by target profile:
     - low latency,
     - high consistency,
     - robust judge reliability.
   - Document known failure modes and operating envelopes.

### 5) Metrics to graph and what to compare

#### Core protocol comparison plots
- **Cache hit rate** (`cache_hit_rate`) by protocol and workload regime.
- **Latency curves** (`avg_read_latency`, `avg_write_latency`, plus p50/p95/p99) by protocol.
- **Consistency risk** (`stale_read_rate`, `avg_visibility_lag`, `avg_convergence_time`) by protocol.
- **Conflict pressure** (`contested_ratio`, `conflict_checks`) by protocol and contention level.

#### Judge-specific comparison plots
- **Judge latency** (`avg_judge_latency`) by judge mode/provider.
- **Fallback behavior** (`fallback_count`) by workload stress and judge mode.
- **Failure taxonomy** (`llm_failure_categories`) stacked bars for LLM runs.
- **Reason code distribution** (`reason_code_counts`) to inspect decision patterns.

#### Semantics validation plots
- **Read-your-writes success rate** (`read_your_writes_success_rate`) by protocol/judge.
- **Monotonic read violations** (`monotonic_read_violations`) by protocol/judge.
- **Staleness severity** (`avg_stale_version_gap`, `p95_stale_version_gap`, `max_stale_version_gap`).

#### Efficiency vs correctness trade-off plots
- Scatter/pareto chart:
  - x-axis: latency metric (e.g., `write_latency_p95`),
  - y-axis: correctness metric (e.g., `stale_read_rate` or monotonic violations),
  - color: protocol,
  - marker: judge mode.

### 6) Recommended acceptance criteria (example)

- `read_your_writes_success_rate` near 1.0 for strong-consistency targets.
- `monotonic_read_violations` = 0 for strict profiles.
- `fallback_count` below agreed threshold for judge stability.
- `write_latency_p95` and `read_latency_p95` within SLO budget.
- No dominant LLM failure category that indicates schema/runtime fragility.

### 7) Reporting template for each evaluation round

- **Setup**: protocol(s), judge(s), workload config, seed set, scale.
- **Tables**: aggregated metric summary (mean, std, p95).
- **Figures**: core protocol, judge, and trade-off plots.
- **Findings**: best/worst configurations and why.