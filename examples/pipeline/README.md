# Existing pipeline example

Install the unreleased source checkout or its candidate wheel, then run:

```bash
invarlock-pipeline init release-check --example extraction
invarlock-pipeline compare release-check/pipeline.json --output release-check/result
```

The installed command creates a complete 40-case synthetic example, a reusable
project and policy, and quality plus latency checks on the full dataset and a
selected category. Also try `--example classification` or `--example judge`.
The examples are illustrative; they do not establish deployment quality.

Follow the [integration guide](../../docs/user-guide/pipeline-integration.md) to
replace these records with your evaluator's outputs, set approved thresholds,
use the SDK or native parsers and add the gate to CI.

The distribution smoke test runs outside the source checkout, clears Python's
source-path override and exercises all examples, signing, independent
verification, reports, repeated destinations and every decision exit code:

```bash
python examples/pipeline/wheel_smoke.py --cli invarlock-pipeline
```

It requires an installed candidate wheel. All generated examples, keys and
reports stay in a temporary directory that is removed on completion.
