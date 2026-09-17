# Frozen-answer judge example

This synthetic one-case fixture demonstrates offline import and report rendering.
It contains too few independent units to satisfy its policy and deliberately
returns `insufficient_evidence`. It is not a live provider qualification result.

Use this example to understand how frozen answers, a grading plan and retained
ratings become an evidence bundle you can inspect. The first exercise needs only
Python 3.12 or newer and the installed core InvarLock package. It needs no model,
GPU, API key or optional collector. Use example files from the same source
revision as your installed build.

## Run the offline fixture

From the repository root, copy the fixture inputs into a new workspace. The
measured-reference archives are not needed for this exercise:

```bash
mkdir judge-demo
cp examples/judge-measurements/*.json examples/judge-measurements/*.yaml \
  examples/judge-measurements/import_inspect.py judge-demo/
cd judge-demo
```

The main request connects these files:

| Input | Role |
| --- | --- |
| `baseline_run.json`, `subject_run.json` | One paired case with its original input, reference and frozen answers |
| `plan.json` | Rubric, judge identity, independent units and scheduled ratings |
| `measurements.json` | Retained illustrative calls and outcomes for replay |
| `analysis_policy.json` | Count, uncertainty and score requirements applied to those outcomes |

Check the inputs, publish unsigned local evidence, then render it:

```bash
invarlock evaluate request.yaml --preflight --json
invarlock evaluate request.yaml --unsigned --json
invarlock report evidence --html report.html --markdown report.md \
  --junit report.xml --json
```

Preflight writes no evidence. Evaluation creates `evidence/` and exits zero
because publication succeeded; the recorded decision is still
`insufficient_evidence`. Reporting creates the three requested views. Repeated
ratings of one answer do not create additional independent cases, so this
fixture cannot satisfy its uncertainty requirement. Do not loosen the policy to
make the example appear to qualify a model.

Outputs are no-clobber. Repeat the exercise in a fresh workspace, or select new
evidence and report paths. Unsigned evidence is useful for this local exercise
but cannot establish recipient acceptance. The signed handoff is described below.

## Use existing evaluator captures

Captured `invarlock/evaluation-request-v2` requests can select
`comparison.metric: judge` and the same `invarlock/native-judge-policy-v1` recipe
used by the native starter. InvarLock derives the finalized plan and analysis
policy from the frozen case records. Use `comparison.judge.measurements` for
retained measurements, or omit it to collect through the core collector.
The [captured-results guide](../../docs/user-guide/captured-results.md#judge-captured-answers)
shows the complete route and explicit text projection for structured exports.
This directory's v3 requests instead take an already finalized plan and analysis
policy. Neither captured-answer route grants native model execution assurance.

The text profile grades string inputs and answers. References remain retained
but are omitted from judge requests by default. Set `prompt.reference_mode` to
`per_case` in a finalized plan, or `plan.prompt.reference_mode` in a recipe, to
send each string reference as a separate field. Rebind every rendered-request
and plan digest after changing this choice; it is not a presentation-only option.
The generic core `prepare_evaluator_judge` and `import_judge_sources` APIs described
in the [judge reference](../../docs/reference/judge-measurements.md) support
caller-owned collection without requiring Inspect.

## Import expanded Inspect events offline

The core collector imports the closed `invarlock/inspect-judge-export-v1`
projection, including the expanded model events and provider request/response
fields. It does not accept arbitrary Inspect `.eval` archives or summary scores.
Install core from the repository root, without the SDK extra:

```bash
python -m pip install .
```

From the copied example directory, exercise the synthetic expanded-event fixture:

```bash
python import_inspect.py --export inspect-export.json --collection collection-inspect.json --output measurements-inspect.json
invarlock evaluate request-inspect.yaml --preflight --json
invarlock evaluate request-inspect.yaml --unsigned --json
invarlock report evidence-inspect --html report-inspect.html --json
```

No provider key or network call is needed. The two completed trial slots remain
bound to the same frozen runs and approved plan. The one-case policy still
returns `insufficient_evidence`; this fixture makes no hosted execution claim.
For a retained supported export, provide its path and the independently selected
collection configuration with `--export` and `--collection`. Use `--plan`,
`--baseline-run` and `--subject-run` when the frozen inputs have different names.
Point the evaluation request's `comparison.measurements` at the imported file.

`--root` defaults to the current directory. Input paths resolve relative to that
root; `--output` must be a new relative path inside it, through existing real
directories. Existing files, symlinks and parent traversal are rejected. Reads
are bounded to 64 MiB for plans, 128 MiB per frozen run, 1 MiB for collection
settings and 384 MiB for the export. Import preserves retained event values and
missing or failed trials; it neither calls a model nor fills missing results.
Unsupported event settings, changed answers and mismatched plan or collection
bindings fail before publication.

## Collect new judgments

`request-collect.yaml` selects installed collection for already frozen answers.
The committed `example-judge` identity is synthetic and cannot make live calls.
For real collection, freeze an approved supported hosted judge and matching
collection settings. Install core with the collection SDKs from the repository
root:

```bash
python -m pip install ".[judge]"
```

Then return to the prepared workspace with the updated plan, policy and
collection settings. Use a fresh evidence destination if you already ran the
offline fixture, since both requests initially name `evidence`:

```bash
# Supply OPENAI_API_KEY through your secret manager.
invarlock evaluate request-collect.yaml --preflight --json
invarlock evaluate request-collect.yaml --signing-key signer-private.pem --json
```

The installed command constructs the pinned model and resumes through a private
workspace. Review its call, token, cost and timeout limits before execution.
Missing dependencies or credentials fail preflight without a provider call.
Installed collection requires exactly Inspect `0.3.263`, OpenAI `3.13.0` and
`httpx==0.28.1`. Remove `OPENAI_BASE_URL` and `OPENAI_API_BASE` entirely; custom
endpoints and empty endpoint overrides are rejected. Review the model-specific
sampling and reasoning requirements in the [collection guide](collection.md).
The optional `execution.collection.workspace` defaults to
`<output.evidence>.judge-work`, `scorer_id` to `judge`, and
`invocation_timeout_seconds` to 3600. Use a private, stable workspace to resume
without repeating admitted calls.
For native model execution and automatic answer freezing, use
[`metric: judge`](../native-judge/README.md) instead. Imported frozen answers do
not claim native runtime provenance.

## Publish signed evidence and verify it

Add `--fail-on-policy` to evaluation for exit 7 on the inconclusive policy result.
The unsigned evidence cannot establish recipient acceptance. To publish signed
evidence, use a fresh output directory and `--signing-key signer-private.pem`
instead of `--unsigned`. With your own Ed25519 signing key, publish the same
fixture into a new directory beneath the workspace:

```bash
invarlock evaluate request.yaml --output signed-evidence \
  --signing-key signer-private.pem --json
```

Signing authenticates the recorded result; the one-case decision remains
insufficient evidence. Verification additionally requires an independently
maintained judge recipient policy, not a policy copied from submitted evidence.
Prepare `recipient-policy.json` using the [judge recipient contract](../../docs/reference/judge-measurements.md).
It is not created by the offline commands above.
When retaining a verification receipt, supply an independent verifier key and
identity together:

```bash
invarlock verify signed-evidence \
  --trust-profile recipient-policy.json \
  --receipt verification.receipt.json \
  --verifier-signing-key verifier-private.pem \
  --verifier-identity release-verifier \
  --json
```

The receipt remains outside `signed-evidence`. It authenticates the complete
local bounded-result record and recipient-policy digest; it does not broaden the
fixed-benchmark claim.

See the [judge reference](../../docs/reference/judge-measurements.md) for the
contract boundary, statistical assumptions and recipient verification command.

## Measured references

The [completed Luna held-out reference](references/k2-32b-luna-xhigh-heldout/README.md)
is the main measured judge example: 10,260 complete ratings, both frozen policies
met, signed offline replay, and a completed reference-label comparison with
retained disagreements. It does not establish general judge accuracy or model
improvement.

The [corrected Sol pilot](references/k2-32b-pilot/README.md) and
[Luna xHigh pilot](references/k2-32b-luna-xhigh-pilot/README.md) retain 480
complete ratings each. Their smaller analyses remain insufficient evidence
under unchanged interval-width policies. The original incomplete Sol attempt is
retained with the corrected pilot as historical failure evidence, not a
recommended configuration or completed qualification.

For the optional study's outcome-blind corpus selection, original pilot plans,
reference-rating sheets and candidate final plans, see
[Freeze a K2 judge reference](K2-REFERENCE.md). That frozen-answer archive contains
no judge outcomes. Its original study inputs remain unchanged alongside the
separately retained completed comparison.
