# Release Verification

This page covers verification of published InvarLock package artifacts and
source tags. It does **not** describe evidence-pack verification. Evidence packs
remain the mechanism for evaluation evidence; release verification is about the
published wheel/sdist pair and the tagged source tree that produced them.

## What a tagged release contains

For a tagged release, treat these surfaces as authoritative:

- the git tag `vX.Y.Z`, which resolves to an immutable commit SHA
- the PyPI wheel (`*.whl`)
- the PyPI source distribution (`*.tar.gz`)

The release workflow validates release tags before publishing, resolves the
requested tag to a resolved commit SHA, rebuilds from that commit, generates an
SBOM from the installed release surface, records gitleaks output, and publishes
the distributions through trusted publishing.

## Recommended verification flow

1. Resolve the tag to its commit SHA and confirm the tag points at the expected
   source tree.
2. Fetch the PyPI metadata for `invarlock==X.Y.Z` and compare the published
   filenames and sha256 digests against the files you download.
3. Install the wheel in a fresh virtual environment and smoke-test the CLI
   surface you rely on.
4. If you are doing maintainer or audit review, inspect the tagged `release`
   workflow run and confirm the publish job, install-surface SBOM generation,
   and provenance steps completed from the same resolved commit SHA.

Example metadata check:

```bash
python - <<'PY'
import json
import urllib.request

version = "X.Y.Z"
with urllib.request.urlopen(f"https://pypi.org/pypi/invarlock/{version}/json") as response:
    payload = json.load(response)

for item in payload["urls"]:
    print(item["filename"], item["digests"]["sha256"])
PY
```

Example smoke install:

```bash
tmpdir="$(mktemp -d /tmp/invarlock-release-verify.XXXXXX)"
python3 -m venv "$tmpdir/venv"
"$tmpdir/venv/bin/python" -m pip install --upgrade pip
"$tmpdir/venv/bin/python" -m pip install --no-cache-dir "invarlock==X.Y.Z"
"$tmpdir/venv/bin/python" -c "import invarlock; print(invarlock.__version__)"
"$tmpdir/venv/bin/invarlock" --help
```

## Notes

- Release verification and evidence-pack verification are intentionally separate.
- Manual release dispatch rejects malformed, missing, or non-tag release refs
  before build/publish begins.
- Release tag resolution peels annotated tags to immutable commit SHAs before
  build/publish begins.
- The install-surface SBOM describes the installed release surface, not the CI
  tool environment.
- Public package consumers should rely on PyPI artifacts plus source tags;
  maintainers can use GitHub Actions run logs and uploaded workflow artifacts
  for deeper audit trails.
