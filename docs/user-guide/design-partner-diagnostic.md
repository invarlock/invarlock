# Design-Partner Diagnostic Runbook

Use this runbook for one bounded question: does one real transformed checkpoint
remain acceptable against one reviewer-selected baseline under independently
controlled acceptance inputs?

This is the smallest useful partner diagnostic. It uses the existing shared
compare wrapper; it does not introduce a second evaluation workflow.

## Applicability

Use this runbook only when both checkpoints are loadable through InvarLock's
current Hugging Face/PyTorch adapter path. The subject must be the output of a
real quantization, pruning, adapter, fine-tuning, or other transformation. A
copy, alias, or renamed baseline is not a valid subject.

Before the run, the reviewer records:

- the baseline model or path and its immutable revision;
- the distinct transformed subject model or path, its immutable revision when
  remote, the transformation kind, and the producer or reviewer receipt;
- a complete raw baseline report retained independently of the subject run;
- an acceptance policy pack maintained independently of the submitted bundle;
- the expected runtime-image digest obtained from reviewed build or release
  policy, not copied from the generated runtime manifest.

The baseline report, policy pack, and runtime digest are verifier trust inputs.
The submitted report cannot establish their independence.

## Prepare one reviewed case

From the repository root, copy the checked template to its ignored local name:

```bash
cp examples/integrations/design_partner_diagnostic/case.env.example \
  examples/integrations/design_partner_diagnostic/case.env
```

Edit `case.env`, then load and reject incomplete values:

```bash
set -a
source examples/integrations/design_partner_diagnostic/case.env
set +a

required=(
  CASE_ID BASELINE_MODEL BASELINE_REVISION BASELINE_ADAPTER
  SUBJECT_MODEL SUBJECT_ADAPTER SUBJECT_CHANGE_KIND
  SUBJECT_TRANSFORMATION_RECEIPT BASELINE_REPORT POLICY_PACK
  EXPECTED_RUNTIME_IMAGE_DIGEST PROFILE TIER EXPECTED_GUARD_AUTHORITY
  REPORT_OUT ALLOW_NETWORK
)
for name in "${required[@]}"; do
  value="${!name:-}"
  if [[ -z "$value" || "$value" == REPLACE_WITH_* || "$value" == *REPLACE_WITH_* ]]; then
    printf 'Set %s in case.env before running.\n' "$name" >&2
    exit 2
  fi
done

[[ "$BASELINE_MODEL" != "$SUBJECT_MODEL" ]] || {
  echo "Baseline and subject must be distinct inputs." >&2
  exit 2
}
[[ -f "$BASELINE_REPORT" && -f "$POLICY_PACK" ]] || {
  echo "Reviewer-supplied baseline report or policy pack is missing." >&2
  exit 2
}
if [[ -e "$SUBJECT_MODEL" || -L "$SUBJECT_MODEL" ]]; then
  [[ -z "$SUBJECT_REVISION" ]] || {
    echo "A local subject cannot also declare a remote revision." >&2
    exit 2
  }
else
  [[ "$SUBJECT_REVISION" =~ ^[0-9a-f]{40,64}$ ]] || {
    echo "A remote subject requires an immutable 40-64 character revision." >&2
    exit 2
  }
fi
[[ "$SUBJECT_CHANGE_KIND" =~ ^[a-z0-9]+([-_][a-z0-9]+)*$ ]] || {
  echo "The subject change kind must be a lowercase hyphen/underscore slug." >&2
  exit 2
}
[[ -f "$SUBJECT_TRANSFORMATION_RECEIPT" ]] || {
  echo "The transformation receipt is missing." >&2
  exit 2
}
[[ "$EXPECTED_RUNTIME_IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || {
  echo "The runtime-image digest must be a canonical sha256 digest." >&2
  exit 2
}
[[ "$EXPECTED_GUARD_AUTHORITY" == "enforce" ]] || {
  echo "This acceptance runbook requires all-enforce guard authority." >&2
  exit 2
}
```

Set `SUBJECT_REVISION=""` for a local subject. A remote subject revision must be
a full immutable commit identifier, not a branch or tag. The distinct-input
check catches only the simplest alias. The reviewer must still inspect the
transformation receipt and subject identity before accepting the case. The
handoff binder copies and hashes an opaque receipt; it does not validate a
producer-specific receipt schema or authenticate its claims.

## Run the comparison

Strict acceptance uses the CUDA/container lane. Add `--allow-network` only when
the reviewed inputs require downloads; leave it absent for local checkpoints.

```bash
compare_args=(
  examples/integrations/_shared/run_invarlock_compare.sh
  --baseline "$BASELINE_MODEL"
  --subject "$SUBJECT_MODEL"
  --baseline-revision "$BASELINE_REVISION"
  --baseline-adapter "$BASELINE_ADAPTER"
  --subject-adapter "$SUBJECT_ADAPTER"
  --baseline-report "$BASELINE_REPORT"
  --policy-pack "$POLICY_PACK"
  --expected-runtime-image-digest "$EXPECTED_RUNTIME_IMAGE_DIGEST"
  --profile "$PROFILE"
  --tier "$TIER"
  --lane cuda
  --report-out "$REPORT_OUT"
)
if [[ -n "$SUBJECT_REVISION" ]]; then
  compare_args+=(--subject-revision "$SUBJECT_REVISION")
fi
if [[ "$ALLOW_NETWORK" == "1" ]]; then
  compare_args+=(--allow-network)
fi
"${compare_args[@]}"
```

Because `--report-out` is explicit, the shared wrapper writes these exact
outputs beneath `$REPORT_OUT`:

| Output | Review use |
| --- | --- |
| `evaluation.report.json` | Canonical baseline-versus-subject report. |
| `verify.json` | Machine-readable strict verifier result, including the embedded report-bound unsigned receipt. |
| `evaluation.html` | Human-readable report. |
| `run_summary.txt` | Concise run and verifier status. |
| `run_command.txt` | Effective wrapper, evaluate, verify, and render commands. |
| `lane_artifact.json` | Effective lane, assurance, provenance, and device settings. |
| `runtime.manifest.json` | Container runtime provenance, when emitted by strict evaluation. |

The wrapper does not emit a separately signed receipt. Its `verify.json` is the
supported machine-readable verifier envelope and carries the byte-bound,
unsigned verification receipt under each result's `verification.receipt`.

## Build the review handoff

Generate the release-review Markdown and compact summary from the verified
report with the existing public handoff example:

```bash
examples/integrations/public_e2e/run_public_e2e_release_review.sh \
  --report "$REPORT_OUT/evaluation.report.json" \
  --baseline "$BASELINE_REPORT" \
  --policy-pack "$POLICY_PACK" \
  --expected-runtime-image-digest "$EXPECTED_RUNTIME_IMAGE_DIGEST" \
  --profile "$PROFILE" \
  --assurance strict \
  --runtime-provenance container \
  --output-dir "$REPORT_OUT/handoff" \
  --force || exit $?

"${PYTHON_BIN:-python3}" \
  examples/integrations/design_partner_diagnostic/bind_subject_handoff.py \
  create \
  --handoff-dir "$REPORT_OUT/handoff" \
  --subject-model "$SUBJECT_MODEL" \
  --subject-revision "$SUBJECT_REVISION" \
  --subject-change-kind "$SUBJECT_CHANGE_KIND" \
  --transformation-receipt "$SUBJECT_TRANSFORMATION_RECEIPT"
```

The handoff directory contains a copied `evaluation.report.json`,
`invarlock-verify.json`, `evaluation.html`, `release-review.md`,
`ci-summary.md`, `run_summary.txt`, and the independently supplied baseline and
policy snapshots. It also contains the model-card and MLflow exports documented
by the public end-to-end example. The diagnostic-specific binder adds:

| Artifact | Review use |
| --- | --- |
| `subject-transformation-receipt` | Byte-for-byte copy of the supplied transformation receipt. |
| `subject-handoff-binding.json` | Relative filenames, byte sizes, SHA-256 digests, declared change kind, and the exact typed subject identity from the report. |

The binder requires a canonical remote revision and checks it against both
`meta.model_identity` and `subject_ref.model_identity`. For local subjects it
records only the checkpoint-tree digest from the report, never the local path.
Its `verify` mode detects later report or receipt changes. This establishes
bundle consistency, not the truth or authenticity of an opaque transformation
receipt.

Without an authenticated signed evidence pack, the release-review export labels
the handoff `RECEIPT_BOUND_UNTRUSTED`. That label is expected: strict report
verification passed, but publisher authenticity was not established.

Check the acceptance fields and report binding directly:

```bash
"${PYTHON_BIN:-python3}" \
  examples/integrations/design_partner_diagnostic/bind_subject_handoff.py \
  verify \
  --handoff-dir "$REPORT_OUT/handoff" || exit $?

"${PYTHON_BIN:-python3}" - \
  "$REPORT_OUT/evaluation.report.json" \
  "$REPORT_OUT/verify.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
verify_path = Path(sys.argv[2])
report = json.loads(report_path.read_text(encoding="utf-8"))
verify = json.loads(verify_path.read_text(encoding="utf-8"))
expected_authority = {
    "spectral": "enforce",
    "rmt": "enforce",
    "variance": "enforce",
}
if report.get("resolved_policy", {}).get("guard_authority") != expected_authority:
    raise SystemExit("resolved policy is not all-enforce")
if report.get("assurance", {}).get("guard_authority") != expected_authority:
    raise SystemExit("assurance guard authority does not mirror policy")
if verify.get("summary", {}).get("ok") is not True:
    raise SystemExit("strict verifier summary is not ok")
results = verify.get("results")
if (
    not isinstance(results, list)
    or len(results) != 1
    or not isinstance(results[0], dict)
    or results[0].get("ok") is not True
):
    raise SystemExit("strict verifier result is missing or failed")
verification = results[0].get("verification", {})
runtime = verification.get("runtime_provenance", {})
if runtime.get("status") != "expected_image_digest_matched":
    raise SystemExit("runtime provenance did not match the independent digest")
receipt = verification.get("receipt", {})
report_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
if receipt.get("subject_report_sha256") != report_digest:
    raise SystemExit("verification receipt is not bound to this report")
print("strict all-enforce diagnostic accepted")
PY
```

## Success criteria

The case succeeds only when all of the following are true:

1. The reviewer confirmed one immutable baseline and one genuinely transformed,
   distinct subject from the transformation receipt and artifact identities;
   remote subject revision matches the report-bound typed identity.
2. The compare wrapper and handoff script both exit `0`.
3. `verify.json` and `handoff/invarlock-verify.json` report
   `summary.ok: true` with strict assurance and no failed result; each successful
   result includes an unsigned receipt bound to the evaluated report digest.
4. Runtime provenance reports `expected_image_digest_matched`, with the
   expected digest supplied independently by the reviewer.
5. `subject-handoff-binding.json` re-verifies the copied report and
   transformation receipt, and records the declared change kind without a
   local source path.
6. `run_summary.txt` records `status: success`, and the HTML and
   release-review Markdown describe the same report identity and verdict.
7. The resolved policy and policy pack use all-`enforce` guard authority for
   this acceptance run.

Preserve failed outputs. A failed run is a diagnostic result, not an invitation
to edit the report, weaken a threshold, or replace the independent trust inputs.

## Observe versus enforce

Shipped tiers default spectral, RMT, and variance guard authority to `enforce`.
That is the acceptance expectation for this runbook: a complete threshold
finding can block the result.

`observe` is an explicit reviewer-authorized policy override for investigation.
It can make a complete threshold finding that can be replayed non-blocking when the
report and independent policy pack bind the exact same authority map. It does
not waive primary-metric failure, drift, invariant failure, missing or degraded
evidence, unsupported measurements, monitor-only guards, or incomplete replay
facts. Keep an observe run as diagnostic evidence; it does not satisfy this
runbook's all-enforce success criterion.

## Evidence-pack boundary

The installed CLI exposes evidence-pack inspection and verification, but no
general-purpose evidence-pack build command. The repository's signed producer
is catalog-lane-specific and requires catalog bindings. Do not hand-assemble or
label this partner handoff as a signed evidence pack. End this workflow at the
strict verifier output and release-review bundle unless a supported producer
for this case type is added later.

## Non-goals

This diagnostic does not:

- validate TensorRT, ONNX, GGUF, CoreML, or a black-box serving endpoint;
- prove production serving parity, throughput, latency, or cost;
- certify general model safety, security, governance approval, or deployment
  readiness;
- create, train, transform, publish, sign, or deploy the subject;
- replace partner-specific acceptance criteria or independent review.

If the partner's actual artifact is outside the Hugging Face/PyTorch checkpoint
path, record that as an unsupported runtime boundary instead of converting it
silently and claiming the original runtime was evaluated.

## Related documentation

- [Integration examples](integrations.md)
- [Compare and evaluate](compare-and-evaluate.md)
- [Trust model](../assurance/14-trust-model.md)
- [Strict assurance checklist](../assurance/15-strict-assurance-checklist.md)

The repository's checked handoff example is
`examples/integrations/public_e2e/README.md`.
