# Artifact Layout & Retention

InvarLock keeps intermediate run artifacts and long-lived certification evidence in a predictable directory structure. Use the conventions below when running the certification loop or automations so reports are reproducible and easy to audit.

---

## 1. Run Outputs (`runs/`)

Each invocation of `invarlock run` writes into a user-specified output directory (default `runs/<nickname>/`). A timestamped subdirectory is created for every attempt:

```text
runs/
  baseline/
    20251010_182515/
      report.json          # full RunReport (metrics, guard results, deltas)
      events.jsonl         # event log (prepare/edit/guard/eval phases)
      model.pt?            # optional checkpoint if `output.save_model=true`
    report.json            # convenience copy of the latest successful run
  quant8/
    20251010_151826/
      report.json
      events.jsonl
```

- `report.json` captures metrics, seeds, hashes, guard verdicts, and edit deltas. It is the canonical source for certificate generation.
- `events.jsonl` is an append-only log useful for debugging guard triggers or edit progress.
- Delete timestamped subdirectories only after copying the relevant `report.json` into a long-term location.

---

## 2. Reports & Certificates (`reports/`)

After validating a run, copy its canonical artifacts into `reports/<run-id>/`:

```text
reports/
  baseline/
    report.json            # normalized baseline report
  quant8_balanced/
    evaluation.cert.json   # `invarlock report --format cert`
    report.json            # (optional) copy of the edited RunReport
    metadata.json          # (optional) automation annotations
```

- Use deterministic directory names (e.g., `quant8_balanced_20251010T1509Z.json`) when archiving multiple attempts.
- Certificates reference the baseline report path, policy tier, policy digest, seeds, dataset/tokenizer hashes, and variance tap/targets. Keep baseline `report.json` next to edited certs so `invarlock verify` can resolve them without editing paths.
- Preserve `evaluation_windows` inside the baseline `report.json`. CI/Release baseline pairing is fail-closed: `invarlock run --baseline ...` refuses to proceed if the baseline window evidence is missing/invalid, and `invarlock verify` rejects certificates that are not provably paired.
- For a field-by-field description of the certificate bundle consult [Certificate Schema (v1)](certificate-schema.md).
- Do **not** commit raw model checkpoints or `events.jsonl`; they can contain large payloads and operational metadata. Store them in your artifact store if required.

---

## 3. Seeds, Hashes & Policy Digests

- Every `report.json` includes `meta.seeds` (Python/NumPy/Torch), `data.dataset_hash`, `data.tokenizer_hash`, and `auto.policy_digest`. Certificates preserve the same fields.
- External automations can persist the seed bundle separately (e.g., `runs/<name>/<ts>/seed_bundle.json`) but the RunReport already satisfies the retention requirement.
- When comparing runs, always reference certificates rather than regenerating them so the hashed policy digest and pairing statistics remain immutable.

---

## 4. Cleanup Checklist

1. Promote any PASS run into `reports/` (copy `report.json` and generated `cert.json`).
2. Record the mapping from `<run>` → `<baseline>` in your change log or tracker.
3. Delete stale timestamped subdirectories under `runs/` once evidence is archived.
4. Keep `reports/` under version control; exclude `runs/` and `reports_*` scratch directories via `.gitignore`.

Following these practices keeps long-lived evidence small, deterministic, and auditable while allowing `runs/` to remain a scratch space for repeated attempts.
