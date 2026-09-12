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

`request-collect.yaml` demonstrates the execution-free collection preflight. Its
synthetic `example-judge` identity is deliberately not callable. For a
live run, freeze a real supported hosted-model identity in the plan, set the
matching grader and current account limits in `collection.json`, then install the
optional package and run the maintained collector:

```bash
python -m pip install 'invarlock-inspect-judge[inspect]==0.15.0'
export OPENAI_API_KEY=your-key-from-a-secret-store
python collect.py --execute-collection
```

The script loads `plan.json`, `collection.json` and both frozen runs from the
current directory. It resumes through the private `judge-checkpoint` directory
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

See the [judge reference](../../docs/reference/judge-measurements.md) for the
contract boundary, statistical assumptions and recipient verification command.

For the real K2 Horizon 32B frozen-answer corpus, outcome-blind selection,
pilot plans, reviewer sheets and pending final plans, see
[Freeze a K2 judge reference](K2-REFERENCE.md). That retained reference contains
no judge-model results yet; its final plans remain unavailable for execution
until the pilot rubric review is recorded.
