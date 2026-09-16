# Five-minute signed-evidence check

Start here if you want to see what a reviewer receives and how to check it.
You will verify an included signed comparison, create a new verification receipt
and open an HTML report. The comparison has already been run; this example does
not execute a model.

You need Python 3.12 or newer, a terminal, `curl`, `tar` and network access to
install the package and download its matching examples. Verification itself runs
offline on a regular CPU, without a GPU, container engine or provider account.

## Run the check

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

## Inspect the result

The successful command prints `Decision: pass` and writes:

- `verification.result.json`, the machine-readable verifier result;
- `verification.receipt.json`, the newly signed independent receipt; and
- `evidence.html`, the HTML report.

Open `evidence.html` in your browser. The report presents the comparison and its
policy result; `verification.receipt.json` is the separate signed record of the
verification you just performed.

## Use your own evidence

The script creates a one-use demonstration verifier key and deletes it after
receipt issuance. Production verifiers retain their keys in an appropriate
secret manager and supply recipient-owned policy, artifact, schedule, runtime,
and signer anchors. The retained fixture establishes only the exact signed
comparison it contains.

Here, **anchors** means the identities and hashes the recipient expects before
opening the evidence. The example supplies those from its retained fixture. For
your own handoff, follow the [evidence and verification guide](../../docs/user-guide/evidence-and-verification.md)
to supply your own expected inputs and trusted signer. To evaluate new records,
continue with the [captured-results example](../captured-results/README.md).
