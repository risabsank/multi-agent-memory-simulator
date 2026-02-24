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
Semantic coherence improves reliability under bounded latency and cost.

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

## 8) Recovery success criteria
Recovery is successful if, within bounded latency and cost, the system achieves:
- no unresolved contradictions,
- selected memory aligns with ground-truth oracle above threshold,
- bounded blast radius from wrong memory,
- contested hypotheses resolved or quarantined.

---

## 9) Protocol components
- **Memory object model update**
- **Semantic conflict detector**
- **Coherence states**
- **Conflict-resolution policy engine**
- **Retrieval-time coherence filtering**

---

## 10) Protocol variants to evaluate
- **S0: No semantic coherence**
  - baseline behavior
- **S1: Detect-only**
  - flag contested claims, do not block writes
- **S2: Soft gating**
  - downweight contested claims unless highly task-relevant
- **S3: Hard gating + verification**
  - block contested claims from accepted shared memory until verification approval
- **S4: Adaptive semantic coherence**
  - dynamically switch policy strength using reliability/latency/cost feedback

