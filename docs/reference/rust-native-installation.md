# Experimental native Linux commands

The native distribution provides two executables. `invarlock-native-verify`
contains offline evaluation, verification, reporting and packaging. It has no
process launcher or Python discovery. `invarlock-native` offers those commands
and an explicitly selected Python compatibility worker.

The distribution is experimental and is developed separately from `staging/next`.
Use the architecture matching your Linux host: `x86_64` or `aarch64`. The bundle's
`manifest.json` records its build architecture, C library version, source archive
identity and file hashes. Qualification on the build image does not establish
compatibility with older C library releases or Alpine/musl.

## Install and inspect

Extract the archive into a new directory, then verify its checksums before use:

```sh
mkdir native-invarlock
cd native-invarlock
tar -xzf /path/to/invarlock-native-linux-ARCH.tar.gz
sha256sum -c SHA256SUMS
./bin/invarlock-native-verify --version
./bin/invarlock-native-verify --capabilities
./bin/invarlock-native diagnostics
```

Run the binaries from this directory or copy `bin/` and `share/` into a prefix you
own. Keep the license notices, dependency inventory and SBOM with the binaries.
The checksums establish integrity relative to this bundle; they do not establish
publisher authenticity. Approve the archive digest through your distribution
channel before installing it.

## Offline workflows

```sh
./bin/invarlock-native evaluate request.yaml --signing-key signer.pem --preflight --json
./bin/invarlock-native evaluate request.yaml --signing-key signer.pem --json
./bin/invarlock-native verify evidence --trust-profile trust.json --receipt receipt.json --json
./bin/invarlock-native report evidence --html report.html --markdown report.md --junit report.xml --json
./bin/invarlock-native report evidence --review-context context.json --package handoff --json
```

Captured exports, retained runtime imports, and retained judge imports run
without Python. All eight export adapter entry points and nineteen evaluator
profiles use native parsing. A provider import validates retained identities and
receipts; it does not load the model. Model execution, judge collection and
installed custom scorers or providers require their compatibility integration.

The native CLI supports the signing, verifier and explicit trust-pin environment
variables used by the Python CLI. Explicit options take precedence. Use
`evaluate --help`, `verify --help` or `report --help` for the supported options.
`--json` selects the machine contract. Default output uses plain native terminal
presentation; report content is printed as Markdown rather than Rich's terminal
layout. Generated Markdown, HTML, JUnit and evidence files retain their existing
contracts. Native version/help text identifies the native application.

## Explicit Python compatibility

Prepare a Python 3.12 environment containing this branch's InvarLock package,
the matching `invarlock-native-experiment` wheel and the integrations needed for
the request. Select its interpreter explicitly:

```sh
./bin/invarlock-native evaluate request.yaml \
  --python-worker /absolute/venv/bin/python \
  --worker-timeout-seconds 3600 --signing-key signer.pem --json
```

The command passes a bounded request to `invarlock.native_compat_worker`. That
worker requires the native extension and selects the Rust execution backend.
It retains Python for SDK and custom-plugin contracts. Ordinary authorization
flags, including installed-scorer and network authorization, still apply.
There is no interpreter search or automatic fallback. The control process owns
the worker deadline, bounded output capture and cancellation of its process
group. The default deadline is 600 seconds; choose a bound appropriate for your
request. Live provider behavior requires separate qualification with that
provider, model and deployment environment.

## Reproduce a bundle

Build both binaries with the pinned Rust toolchain and locked dependency graph,
then run `native/tools/package_binaries.py` on that Linux architecture. Supply
the exact source archive used for qualification with `--source-archive`, the
release directory with `--binaries`, and a fresh archive path with `--output`.
The tool writes a deterministic archive, hashes and a source-bound manifest.
Python is needed by this build helper, not by the installed offline executables.
