---
title: GPU/MPS-First Guards (Decision Memo)
---

# GPU/MPS-First Guards (Decision Memo)

This memo records the **vNext** decisions for making InvarLock guards usable on
large models (30B+ and beyond) where full-matrix SVD is operationally
impractical.

## Goals

- Make Spectral and RMT guard computations **device-resident** (CUDA/MPS-first).
- Make guard math **approximation-only** (iterative, matvec-based).
- Keep results **reproducible**: fixed budgets and deterministic sampling.
- Bind guard semantics to certificates via a **measurement contract** that is
  verify-time enforced.

## Decisions (vNext)

### 1) Remove strict/fast evidence modes

- There is a single canonical guard contract.
- Certificates and verification no longer use “strict vs fast” as an evidence
  class; they instead require a complete measurement contract.

### 2) Spectral contract: $\hat{\sigma}_{\max}$ + degeneracy proxies

- Primary signal remains baseline-relative per-family monitoring of
  $\hat{\sigma}_{\max}$ (largest singular value estimate).
- The σmin/condition-number check is replaced with **GPU-feasible
  degeneracy proxies**:
  - stable-rank drift (baseline-relative)
  - row/col norm collapse (baseline-relative)

**Note:** these proxies are reduced coverage compared to true σmin/κ semantics;
they are intended to catch collapse/rank loss in a scalable way.

### 3) RMT contract: activation edge risk score

- Replace “count of singular value outliers” with an activation **edge risk
  score**:
  - whiten activations (center + standardize)
  - estimate top singular value $\hat{\sigma}_{\max}$
  - normalize by the MP edge for the observed shape
- Acceptance is baseline-relative via an ε-band on score growth (per family).

### 4) Verification gate: measurement contract binding

- Certificates must record the measurement contract (estimator + sampling + dtype
  policy).
- Verification rejects certificates missing the contract or whose recorded
  contract hash does not match the resolved policy.

## Non-goals

- Backward compatibility with strict/fast certificates or policies.
- Full-spectrum or exact SVD computations in guard code paths.
