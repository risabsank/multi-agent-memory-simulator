# Semantic Cache Coherence Plan

## 1) Purpose
In CPU cache coherence, we ask whether multiple cores see the same value at the same address.
In this simulator, we ask whether multiple agents are acting on **compatible meaning** about shared world state.

This plan defines a protocol layer for semantic coherence so that the latest write is not automatically treated as safe truth.

---

## 2) Why this is needed
In multi-agent LLM systems, new information can be wrong even when recent.

Common failure modes:
- probabilistic writes (hallucinations, over-generalization)
- partial visibility (different agents observe different slices of reality)
- subtle incompatibilities (not exact opposites, but still inconsistent in practice)
- recency amplification (one bad update cascades if older evidence is overwritten)

So before shared memory becomes authoritative, writes should be checked for consistency, credibility, and action safety.

---

## 3) Core rule
A write is **coherent** iff it is either:
1. semantically consistent with accepted memory, **or**
2. explicitly marked as a competing hypothesis and handled by policy.

---

## 4) Hypothesis and objective
### Primary hypothesis
Improve decision reliability while keeping latency and operational complexity low enough to run continuously.

### Objective function
Evaluate each protocol using a weighted score over:
- reliability
- throughput
- latency
- cost

### Initial workloads
- incident triage
- code-debug collaboration
- planning

---

## 5) Memory item schema
Each memory item is the unit compared for compatibility.

Required fields:
- content
- claim type
- provenance (who/what generated it)
- confidence score
- timestamp + environment step
- scope
- dependencies

Recommended lifecycle fields:
- coherence state (`accepted`, `contested`, `provisional`, `deprecated`)
- TTL / validity window
- superseded-by pointer

---

## 6) Contradictions and stale memory
### 6.1 Contradiction definition
Memory items **A** and **B** contradict when:
- scopes overlap,
- both claims cannot be true under domain constraints,
- and the difference is **not** explained by different time window, component, or conditions.

### Contradiction types
- **Direct factual contradiction**: same variable, overlapping scope, incompatible values.
- **Causal/model contradiction**: two mutually exclusive root-cause claims.
- **Action/policy contradiction**: two plans cannot both execute safely.
- **Epistemic contradiction**: one claim says certain while another says unknown/unverified.

### 6.2 Staleness definition
A memory is stale if it is no longer reliable for the current decision due to time or state changes.

Staleness triggers:
- TTL expiration
- superseded by newer, higher-credibility evidence
- invalidated by state transition

---

## 7) Coherence violations
A coherence violation is a protocol-level event: a write is committed as **accepted shared memory** when it should not be.

Violation cases:
- contradicts accepted memory
- lacks minimum support for its claim type
- overwrites accepted memory with lower credibility under policy
- introduces unsafe-to-act directives without safety gating

---

## 8) Working semantic coherence system (no recovery protocol)
This system treats coherence as a **continuous write/read policy**, not as a separate recovery phase.

### 8.1 Components
- **Memory object model**
  - pairwise contradiction checks in §6.1
- **Coherence state machine**
  - allowed states: `provisional`, `accepted`, `contested`, `deprecated`
- **Policy gate**
  - deterministic decisions for write admission and state transitions
- **Retrieval filter**
  - task-aware ranking that suppresses unsafe or incoherent claims

### 8.2 Write path
For each incoming item `m_new`:
1. Validate required schema fields.
2. Retrieve overlapping-scope candidates from accepted + provisional memory.
3. Run contradiction detection.
4. Score credibility using provenance, confidence, and freshness.
5. Apply policy:
   - **No contradiction + min credibility** → `accepted`
   - **Contradiction + insufficient dominance** → `contested`
   - **Contradiction + strong dominance** → `accepted`, conflicting weaker items → `deprecated`
   - **Low support / malformed** → reject write

### 8.3 Read path
When serving context to agents:
1. Filter out `deprecated` items.
2. Include `accepted` first.
3. Include `provisional` only if relevant and clearly marked.
4. Include `contested` only when task requires uncertainty handling.
5. Block unsafe-to-act directives unless explicitly safety-approved.

### 8.4 Deterministic transition rules
- `provisional -> accepted`: passes contradiction + credibility policy.
- `accepted -> contested`: new conflicting peer with comparable credibility.
- `accepted -> deprecated`: superseded by stronger evidence or staleness trigger.
- `contested -> accepted`: tie broken by verification evidence or policy threshold.
- `contested -> deprecated`: loses comparison after new evidence.

Transitions must be auditable with reason codes (`contradiction`, `stale`, `superseded`, `low_support`, `safety_block`).

### 8.5 Minimum policy defaults
- Require contradiction checks on all write operations touching shared scope.
- Never auto-overwrite `accepted` with lower-credibility claims.
- Preserve competing hypotheses as `contested` instead of dropping them.
- Disallow action recommendations sourced only from `contested` memory.

---

## 9) Acceptance criteria
The system is considered operational when:
- contradiction-bearing writes are never silently committed as `accepted`;
- all committed items have explicit coherence state;
- retrieval omits unsafe directives lacking accepted support;
- p95 write-latency overhead from coherence checks stays within configured budget;
- audit logs allow replay of every state transition decision.

---

## 10) Deployment profile
- **Default profile**: soft gating with deterministic contesting and retrieval filtering.
- **High-risk profile**: hard gating for action-bearing claims.
- **Low-latency profile**: detect-only fallback with explicit reliability warning.