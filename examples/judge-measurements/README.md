# Frozen-answer judge example

This synthetic one-case fixture demonstrates offline import and report rendering.
It contains too few independent units to satisfy its policy and deliberately
returns `insufficient_evidence`. It is not a live provider qualification result.

Copy this directory to a fresh working directory, then run:

```bash
invarlock evaluate request-collect.yaml --preflight --json
invarlock evaluate request.yaml --preflight --json
invarlock evaluate request.yaml --unsigned --json
invarlock report evidence --html report.html --markdown report.md --junit report.xml --json
```

## Import expanded Inspect events offline

The optional add-in imports the closed `invarlock/inspect-judge-export-v1`
projection, including the expanded model events and provider request/response
fields. It does not accept arbitrary Inspect `.eval` archives or summary scores.
Install the matching packages from the repository root, without the SDK extra:

```bash
python -m pip install .
python -m pip install addins/inspect_judge
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

`request-collect.yaml` demonstrates the execution-free collection preflight. Its
synthetic `example-judge` identity is deliberately not callable. For a
live run, freeze a real supported hosted-model identity in the plan, set the
matching grader and current account limits in `collection.json`, then install the
optional package and run the maintained collector:

```bash
# From the repository root for this matching checkout:
python -m pip install .
python -m pip install 'addins/inspect_judge[inspect]'
export OPENAI_API_KEY=your-key-from-a-secret-store
python examples/judge-measurements/collect.py \
  --root /path/to/copied-example \
  --execute-collection
```

The script loads `plan.json`, `collection.json` and both frozen runs from the
directory selected by `--root`. It resumes through the private
`judge-checkpoint` directory
and writes a new `measurements-collected.json` for `judge_import`. Review the
preflight and every call, token and cost cap before supplying
`--execute-collection`. Custom provider URLs are outside this qualified example.
Keep provider credentials in the collector environment. Never put them in the
request, plan, checkpoint or retained evidence.

Add `--fail-on-policy` to evaluation for exit 7 on the inconclusive policy result.
The unsigned evidence cannot establish recipient acceptance. To publish signed
evidence, use a fresh output directory and `--signing-key signer-private.pem`
instead of `--unsigned`. Verification additionally requires an independently
maintained judge recipient policy, not a policy copied from submitted evidence.
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

For the real K2 Horizon 32B frozen-answer corpus, outcome-blind selection,
pilot plans, reviewer sheets and pending final plans, see
[Freeze a K2 judge reference](K2-REFERENCE.md). That retained reference contains
no judge-model results yet; its final plans remain unavailable for execution
until the pilot rubric review is recorded.
