# Add InvarLock to an existing evaluation pipeline

!!! tip "User guide"

    **Outcome:** Compare existing baseline and candidate results across quality,
    cost or latency metrics and selected data slices, with a reusable CI gate.

    **Audience:** Engineers who already run evaluations and want explicit release
    criteria and a portable record of the decision.

    **Prerequisites:** Python 3.12 or newer on Linux or macOS, this source checkout,
    and per-case results from both releases. No model, GPU, container or evaluator
    SDK is required for comparison.

This workflow is **unreleased**. Install the checkout to try it; the published
0.15.0 wheel does not contain `invarlock-pipeline`.

## Get a complete first result

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install .
invarlock-pipeline init release-check --example extraction
invarlock-pipeline compare release-check/pipeline.json --output release-check/result
```

Expect exit code `0`, a JSON status with `decision: pass`, and five files under
`release-check/result`: `evidence.json`, `comparison.json`, `report.html`,
`summary.md` and `junit.xml`. Open the HTML report to inspect every metric and
slice, including counts, means, paired intervals and reasons for rejection.
`classification` and `judge` are the other runnable examples.

These examples contain 40 synthetic cases and illustrative thresholds. Replace
both runs and review the policy before using the decision. The judge example
contains recorded numbers; it does not call a judge or demonstrate judge quality.
Each invocation requires a new output directory, so a failed or repeated run
cannot silently replace a previous result.

## Capture your own results

Keep stable case IDs, task inputs, references and string slice tags identical
between releases. Put rendered prompts, retrieved context and generation
settings in `context`; these may change with the release and remain bound into
the evidence. The artifact identity should cover the actual deployed change:
model files or a content-addressed manifest of model, prompt and retrieval
configuration. A caller-supplied identity does not authenticate model execution.

The installed SDK accepts results from any evaluator:

```python
from pathlib import Path
import json
from invarlock.pipeline import make_run

# existing_results comes from your evaluator; keep failed cases in this list.
records = [
    {
        "id": result.case_id,
        "input": result.question,
        "expected": result.reference,
        "output": result.answer,
        "context": {"rendered_prompt": result.rendered_prompt},
        "metadata": {"category": result.category},
        "error": result.error,
    }
    for result in existing_results
]
run = make_run(
    records,
    source={"name": "our-evaluator", "version": "1.0.0"},
    run_id="candidate-build-42",
    artifact_digest=deployment_manifest_digest,
)
Path("candidate.json").write_text(json.dumps(run), encoding="utf-8")
```

The variable `deployment_manifest_digest` must be a real `sha256:` digest from
your build. `invarlock-pipeline digest deployment-manifest.json` hashes a regular
file without loading model weights into memory. Capture the baseline the same
way. Edit the generated project's two paths, or override them per CI run:

```bash
invarlock-pipeline compare release-check/pipeline.json \
  --baseline exports/baseline.json --candidate exports/candidate.json \
  --output artifacts/release-42
```

Project paths are relative to `pipeline.json`; command-line overrides are
relative to the current directory. Commit your approved policy and project
configuration. Exported answers and evidence may contain application data; apply
your normal retention and access rules before uploading them as CI artifacts.

## Import a native export

For supported native formats, import once using the installed parser:

```bash
invarlock-pipeline import candidate-log.json \
  --adapter inspect-json --source-version 0.3.254 \
  --run-id candidate-build-42 --artifact-digest "$CANDIDATE_ARTIFACT_DIGEST" \
  --output candidate.json
```

The environment variable is populated by your build, independently of the
export. Available adapters and their explicit format limits are documented in
[Pipeline contracts](../reference/pipeline-contracts.md#native-adapters).
Generic JSONL uses the same per-record keys as the SDK. Unsupported or ambiguous
formats fail with exit code `2`; use an explicit SDK mapping when your evaluator
produces a different shape. Aggregate scores alone cannot establish paired
release behavior.

## Choose metrics and slices

Edit `policy.json` before observing candidate results. Each metric specifies its
kind, direction, unit, mean aggregation, minimum count, maximum allowed
regression and maximum interval width. Optional absolute candidate mean bounds
prevent an equally poor baseline and candidate from passing your quality floor.
The generated examples include two metrics and one named slice. Every metric is
checked on the full schedule and on every selected slice; any failed check
blocks the gate.

| Kind | Useful for | Recomputed from outputs |
| --- | --- | --- |
| `exact_match` | Literal identifiers and labels | Yes |
| `normalized_match` | Labels with case or whitespace differences | Yes |
| `numeric_tolerance` | Numeric answers within declared absolute or relative tolerance | Yes |
| `json_fields` | Average accuracy of selected structured output fields | Yes |
| `json_exact` | Complete structured output correctness, with no extra fields | Yes |
| `token_f1` | Reference token overlap | Yes |
| `recorded` | Existing judge, human, cost, latency or external metric scores | Aggregation only |

`token_f1` is token overlap, not semantic correctness. `json_fields` checks only
the declared JSON pointers and averages their matches. Use `json_exact` with
`configuration: {}` for an all-or-nothing whole-document check. Scorer details and statistical assumptions are in
the [contract reference](../reference/pipeline-contracts.md).

For `normalized_match` and `token_f1`, approve an explicit
`configuration.unicode_version` matching the runtime used for comparison and
verification. Check it with
`python -c "import unicodedata; print(unicodedata.unidata_version)"`.
The generated classification example sets it for you. Reuse a matching runtime
when replaying evidence; a missing or different version fails the integration
instead of silently changing text scoring after a Python upgrade.

For recorded metrics, include per-record `scores`, declare `score_provenance` in
each run, and copy the approved provenance into the policy's
`accepted_provenance`. Units, source, version and rubric digest must match on both
sides. Judge and human judgments require a rubric digest. Changing the rubric or
judge version requires a deliberately revised policy and comparable runs.
Missing scores and upstream errors remain insufficient evidence; they are not
dropped to produce a smaller passing sample. Latency and cost currently use
means; percentile service-level objectives need their own supported metric.

## Use the result in CI

After your existing evaluator writes both exports and InvarLock is installed,
add this shell step:

```yaml
- name: Check release policy
  run: |
    invarlock-pipeline compare release-check/pipeline.json \
      --baseline exports/baseline.json --candidate exports/candidate.json \
      --output release-result
- name: Present release comparison
  if: always()
  run: |
    if test -f release-result/summary.md; then
      cat release-result/summary.md >> "$GITHUB_STEP_SUMMARY"
    fi
```

Preserve `release-result/` with your organization's artifact uploader and feed
`junit.xml` to its test reporter. Keep policy failures nonzero in the required
release check. Do not suppress them with `continue-on-error` or `|| true`.

| Exit | Meaning | Next action |
| --- | --- | --- |
| `0` | All policy checks pass | Continue the approval workflow |
| `1` | At least one regression-policy bound fails | Inspect the failed metric and slice; fix or explicitly reconsider the policy |
| `2` | Invalid input, incompatible export, publication or authentication failure | Correct the integration or trust inputs; no valid decision was produced |
| `3` | Insufficient count, missing measurements or an overly wide interval | Repair collection or collect enough representative cases |

A `regression` decision means the release did not establish the approved
non-regression bound. It does not necessarily mean a statistically proven
performance decrease. If a policy bound fails and the interval is also too wide,
the failing bound takes precedence. Counts and intervals remain visible.

## Sign and independently verify a handoff

Unsigned comparisons are useful inside the producing pipeline. For a recipient
handoff, generate a signing key once in a new directory:

```bash
invarlock-pipeline keygen release-keys
invarlock-pipeline compare release-check/pipeline.json \
  --output signed-result --signing-key release-keys/private.pem
```

Keep the private key outside the repository and CI artifacts. The recipient
must authorize the public key independently and obtain expected normalized runs
and policy from its approved workflow. Hash those approved run files:

```bash
invarlock-pipeline digest approved-baseline.json --run
invarlock-pipeline digest approved-candidate.json --run
invarlock-pipeline verify signed-result/evidence.json \
  --public-key approved-signer.pem --policy approved-policy.json \
  --expected-baseline "$APPROVED_BASELINE_RUN_DIGEST" \
  --expected-candidate "$APPROVED_CANDIDATE_RUN_DIGEST"
```

The expected digests identify complete normalized runs, not just model files.
Do not obtain them from the received evidence itself. Verification checks the
signature, independent run identities and policy, then recomputes the entire
comparison. Recorded judgments remain recorded judgments after authentication.

This pipeline evidence format does not replace the OCI-backed
[`evaluate → verify → report` transaction](getting-started.md). Use that path
when your recipient requires its execution, runtime and independent signed
receipt contracts. The new pipeline verifier returns an authenticated status;
it does not issue that core receipt. Consumers must explicitly support the
`invarlock/pipeline-evidence-v1` format and its verification requirements.
