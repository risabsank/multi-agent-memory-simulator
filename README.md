# Multi-Agent Memory Simulator (Milestone 0 Skeleton)

This repository contains a minimal version of a cache-coherent multi-agent memory simulator.

Implemented today:
- Discrete-event simulator
- One cache level per agent
- One protocol behavior: write-through to global memory
- Operations: `READ(artifact)` and `WRITE(artifact)`
- Event types:
  - `EV_READ_REQ`
  - `EV_READ_RESP`
  - `EV_WRITE_REQ`
  - `EV_WRITE_COMMIT`

Deferred:
- `SUMMARIZE`
- `SYNC`
- invalidation/reconciliation events
- eventual and task-scoped consistency variants
- multi-level cache policies

## Run

```bash
PYTHONPATH=src python scenarios/trace.py
```

## Extension points

- Add protocol strategy interface for eventual/task-scoped consistency.
- Add cache policy interface for capacity and eviction.
- Add dependency graph backend keyed by `artifact_id`.
