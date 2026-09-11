# Five-minute signed-evidence check

This wheel-user example verifies one retained signed evidence pack using
independently supplied anchors, creates a new verifier-signed receipt, and
renders a self-contained HTML report. It runs on a regular CPU without a model,
container engine, networked service, or source-tree import.

From an empty directory:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install invarlock
INVARLOCK_VERSION="$(python -c 'from importlib.metadata import version; print(version("invarlock"))')"

curl -fsSLO \
  "https://github.com/invarlock/invarlock/archive/refs/tags/v${INVARLOCK_VERSION}.tar.gz" &&
tar -xzf "v${INVARLOCK_VERSION}.tar.gz" --strip-components=3 \
  "invarlock-${INVARLOCK_VERSION}/examples/quickstart" \
  "invarlock-${INVARLOCK_VERSION}/examples/acceptance-handoff/golden" &&

python run.py --fixture golden
```

This recipe requires the matching released tag archive; a missing archive is an
error, not permission to use a mutable branch. For local builds, use the examples
from the exact checkout that built the installed wheel, even if its package
version has not changed. See [Matching wheels and examples](../../docs/user-guide/getting-started.md#matching-wheels-and-examples).

The successful command prints `Decision: pass` and writes:

- `verification.result.json`, the machine-readable verifier result;
- `verification.receipt.json`, the newly signed independent receipt; and
- `evidence.html`, the HTML report.

The script creates a one-use demonstration verifier key and deletes it after
receipt issuance. Production verifiers retain their keys in an appropriate
secret manager and supply recipient-owned policy, artifact, schedule, runtime,
and signer anchors. The retained fixture establishes only the exact signed
comparison it contains.
