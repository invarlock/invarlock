# Five-minute signed-evidence check

This wheel-user example verifies one retained signed evidence pack using
independently supplied anchors, creates a new verifier-signed receipt, and
renders a self-contained HTML report. It runs on a regular CPU without a model,
container engine, networked service, or source-tree import.

From an empty directory:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install "invarlock==0.15.0"

curl -fsSLO \
  https://github.com/invarlock/invarlock/archive/refs/tags/v0.15.0.tar.gz
tar -xzf v0.15.0.tar.gz --strip-components=3 \
  invarlock-0.15.0/examples/quickstart \
  invarlock-0.15.0/examples/acceptance-handoff/golden

python run.py --fixture golden
```

The successful command prints `Decision: pass` and writes:

- `verification.result.json`, the machine-readable verifier result;
- `verification.receipt.json`, the newly signed independent receipt; and
- `evidence.html`, the human-readable rendering.

The script creates a one-use demonstration verifier key and deletes it after
receipt issuance. Production verifiers retain their keys in an appropriate
secret manager and supply recipient-owned policy, artifact, schedule, runtime,
and signer anchors. The retained fixture establishes only the exact signed
comparison it contains.
