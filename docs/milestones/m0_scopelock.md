# Milestone 0 Scope Lock

## In scope

- Discrete-event engine with deterministic ordering `(time, eid)`.
- Single cache level per agent.
- Strong-on-write behavior via write-through commit to global memory.
- Operations: `READ`, `WRITE`.
- Event model: `EV_READ_REQ`, `EV_READ_RESP`, `EV_WRITE_REQ`, `EV_WRITE_COMMIT`.

## Next Steps

- Eventual consistency protocol.
- Task-scoped selective synchronization.
- `SUMMARIZE(artifact -> artifact)` operation.
- Conflict detection/reconciliation.
- Dependency semantics beyond storing dependency IDs.

## Extension interfaces for next milestone

- **Protocol interface**: decide propagation + sync requirements.
- **Cache policy interface**: capacity limits and eviction behavior.
- **Dependency graph backend**: adjacency list keyed by `artifact_id`.

## Implementation checklist

| Item | Status |
|---|---|
| Define minimal core entities (`Artifact`, `Agent`, `Task`, `GlobalMemory`) | Done |
| Implement event queue and four event kinds | Done |
| Implement `READ` hit/miss + global fetch latency | Done |
| Implement `WRITE` with version increment + global commit | Done |
| Add deterministic scenario with 2 agents, 1 shared artifact | Done |
| Print summary metrics (`hits`, `misses`, `avg_latency`) | Done |
