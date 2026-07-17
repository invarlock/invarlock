# Release and distribution verification

InvarLock ships five coordinated Python distributions:

!!! info "Reference"

    - **Surface:** Core and first-party add-in distributions, release checks, provenance, and installation verification
    - **Stability:** Coordinated version and ABI compatibility rules are public; the release workflow may evolve while preserving those checks
    - **Use this page when:** Building, publishing, installing, or independently checking an InvarLock release set

| Distribution | Role |
| --- | --- |
| `invarlock` | Core engine, contracts, Hugging Face provider, verifier, renderer |
| `invarlock-runtime-gguf` | Optional GGUF/`llama.cpp` provider |
| `invarlock-runtime-tensorrt-llm` | Optional TensorRT-LLM provider |
| `invarlock-runtime-hf-vision-text` | Optional Hugging Face vision-text provider |
| `invarlock-diagnostics` | Optional observation-only numeric diagnostics |

All five use the same release version. Provider add-ins declare that exact core
dependency and must also match runtime-provider ABI `1` when loaded.

| Compatibility dimension | Required check |
| --- | --- |
| Package version | Core and selected first-party distributions use the same release version |
| Dependency metadata | Add-in's bounded `invarlock` requirement contains that version |
| Provider ABI | Installed core and provider instance both report ABI `1` |
| Entry point | Exact first-party name resolves to the expected module and class |
| Native runtime | Artifact/backend/image/device identities match the intended deployment |

## What the release workflow checks

For a tagged or explicitly selected release commit, the repository workflow:

1. resolves the exact tag commit;
2. requires the workflow event commit to equal the resolved tag commit;
3. runs the complete repository, coverage, documentation, contract, and
   workflow gates;
4. scans the release history range for secrets;
5. builds one wheel and source distribution for the core and every first-party
   add-in;
6. validates every archive against the exact checkout and runs the release
   preflight again from a clean detached checkout;
7. runs `twine check` on every distribution;
8. installs the built wheels together in a clean environment;
9. exercises the public CLI, all provider conformance commands, diagnostics,
   and entry-point discovery;
10. audits the installed dependency surface and generates an SBOM;
11. attaches build-provenance attestations to the distributions before
   publication; and
12. for a TestPyPI release, verifies every hosted archive against the build
    ledger, installs the hosted wheels together, repeats the conformance smoke,
    and records the immutable run and ledger authorized for promotion.

These checks authenticate and exercise the package set. They do not qualify a
specific model artifact, runtime image, accelerator, dataset, or evidence pack.
Those belong to the evaluate/verify trust model.

The workflow resolves and checks out the release tag's exact commit before it
builds. A tag push validates the candidate but does not publish it. Publication
is an explicit manual action against an existing tag, and the manual workflow
must be dispatched with that tag as its workflow ref so the event commit and
resolved tag commit are identical. TestPyPI publication uses
the distributions built and gated by that run. Production publication requires
the successful TestPyPI run ID and reuses that run's immutable distribution
artifact after authenticating its workflow result, release tag, commit, and
digest ledger. The distribution and promotion artifacts are retained for 14
days, so production promotion must complete within that interval.

Build provenance uses GitHub's [artifact-attestation
mechanism](https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations).
An attestation is useful only when the release verifier validates its subject digest,
signer/workflow identity, and trusted transparency or provenance policy.

## Local preflight before a tag

Run the release checks from a clean checkout whose `HEAD` is the candidate
commit. The coordinated install smoke builds the core and all four optional
packages, installs their pinned base dependency closure and all five wheels in
a disposable virtual environment, runs `pip check`, and exercises provider
discovery and conformance without the checkout or user site on `sys.path`.
The smoke selects the maintained Python 3.12 or 3.13 lock for the invoking
interpreter and fails closed when no matching lock exists.
The read-only preflight then inspects the core wheel and source distribution
against a separately generated `sha256sum`-format manifest, installs the
candidate wheel in isolation, exercises its runtime surface, and reruns the
public-evidence audit:

```bash
make addins-install-smoke

HASH_MANIFEST="$(mktemp)"
trap 'rm -f -- "$HASH_MANIFEST"' EXIT
(cd dist && shasum -a 256 invarlock-*.whl invarlock-*.tar.gz) \
  > "$HASH_MANIFEST"
RELEASE_SHA="$(git rev-parse HEAD)"

make release-preflight RELEASE_PREFLIGHT_ARGS="\
  --release-sha ${RELEASE_SHA} \
  --expected-version X.Y.Z \
  --dist-dir dist \
  --hash-manifest "$HASH_MANIFEST" \
  --json"
```

Use `sha256sum` instead of `shasum -a 256` where that is the platform tool.
Keep the temporary manifest outside the checkout: preflight rejects untracked
files as well as modified files. The manifest must contain only the two core
artifacts by base name. `X.Y.Z` is
the candidate version without a leading `v`; preflight rejects a dirty checkout,
a different `HEAD`, unexpected artifacts, metadata/content mismatch, or a hash
change. A passing result is release-candidate evidence, not authorization to
tag or publish.

The local preflight intentionally validates the core pair in depth while
`make dist-check` validates every archive against its checkout source and
`make addins-install-smoke` plus the release workflow validate the coordinated
five-package install. Keep both kinds of gate; do not describe the core-only
JSON result as proof that every add-in archive was independently inspected by
preflight.

## Test index and production publication

For a candidate tag, use a manual TestPyPI publication first, selecting the
release tag itself as the workflow ref. Leave `promotion_run_id` empty for this
run. The workflow
downloads all five published wheels from the index, compares each download with
the digest returned by that index, installs the set together, and reruns the
CLI, diagnostics, provider-conformance, and entry-point smoke. Review the
provenance bundle and SBOM produced by the same tag workflow. A successful
smoke produces a `testpypi-promotion` artifact bound to the exact release tag,
commit, distribution ledger, and workflow run.

Publish to the production index only when the TestPyPI smoke, local preflight,
tag-to-commit check, release notes, and security review all refer to the same
candidate. Start a separate manual production run with that successful
TestPyPI run's numeric ID as `promotion_run_id`, again selecting the release
tag as the workflow ref. The production job retrieves the immutable
distribution artifact from that run and rejects a failed run, a different
workflow, tag, commit, target, or digest ledger. Production publication remains
a separate trusted-environment action; copying a TestPyPI URL or its downloaded
bytes is not by itself publication authorization.

Configure a protected `v*` tag ruleset that blocks updates and deletion, and
protect the `pypi` environment with required reviewers and an appropriate
deployment policy. These repository controls provide the authorization layer
around the workflow's commit, artifact, and promotion checks.

After production publication, download the five wheels again from the
production index, compare hosted digests, install them together in a clean
environment, and repeat the conformance smoke. Reconcile the published
filenames, version, source tag, provenance subjects, and release assets before
announcing completion.

## Installation and provenance checks

Before installing a release in a controlled environment:

1. select one exact version for core and any first-party add-ins;
2. obtain hashes from a trusted release record or package index response;
3. download artifacts without installing them;
4. verify every downloaded SHA-256 digest;
5. install with hash enforcement where the package-management workflow supports
   it; and
6. run `invarlock --version` and the conformance command for each installed
   runtime add-in.

Example discovery checks after installation:

```bash
invarlock --version
invarlock-gguf-conformance
invarlock-tensorrt-llm-conformance
```

Each conformance command must report `ok: true`, its expected provider name,
and the ABI accepted by the installed core. A conformance pass verifies the
install surface and lightweight provider contract, not a native runtime model
run. Before qualification fan-out, produce and strictly verify one signed
canary through the exact digest-pinned runtime image. Retain its evidence,
signed receipt, and verifier-owned trust profile for the maintained readiness
and evidence targets. Reuse is limited to that exact image digest; a canary does
not establish model-specific load, memory, backend, or execution success.

An example hash-enforced download/install flow is:

```bash
python -m pip download --only-binary=:all: --dest wheelhouse \
  'invarlock==X.Y.Z' \
  'invarlock-runtime-gguf==X.Y.Z' \
  'invarlock-runtime-tensorrt-llm==X.Y.Z'

# Populate requirements.lock with the independently verified hashes, then:
python -m pip install --require-hashes -r requirements.lock
```

`X.Y.Z` and the lockfile are placeholders. Never generate the lockfile from
unverified local downloads and then treat the same downloads as independently
verified.

## Published artifact checklist

| Artifact | Verify |
| --- | --- |
| Wheel | Filename/version, index SHA-256, archive integrity, metadata, entry points |
| Source distribution | Filename/version, index SHA-256, archive integrity, expected source surface |
| Provenance bundle | Subject digest matches each distribution and trusted workflow identity |
| SBOM | Generated for the installed release surface and associated with the same build |
| First-party set | Exactly one coordinated version of core and selected add-ins |

PyPI's index responses can supply hosted distribution digests, but the trust
decision still belongs to the installer's package and provenance policy.

### Offline review bundle

The repository can package already collected distributions, per-artifact
Sigstore sidecars, the GitHub provenance bundle, and the CycloneDX SBOM for an
offline release verifier:

```bash
scripts/release/make_offline_bundle.sh \
  --version X.Y.Z \
  --tag vX.Y.Z \
  --repo OWNER/REPO \
  --dist-dir release-material/dist \
  --sbom release-material/sbom.json \
  --provenance-dir release-material/provenance \
  --output-dir release-material/offline
```

This script assembles existing material; it does not fetch or manufacture
provenance. Every distribution must already have a Sigstore sidecar. Inspect
the generated `release_manifest.json`, verify each file digest, then follow the
bundle's `README.txt` identity and issuer checks. The current GitHub workflow
uploads a build-provenance bundle but does not automatically create this
offline archive, so maintainers must deliberately collect compatible sidecars
and run the assembler.

## Stop, recover, and supersede

If any pre-publication gate fails, stop publication, preserve the failing
artifacts and logs privately, fix the source or workflow, and cut a fresh
candidate. Never replace an artifact while retaining its filename, tag, or
version.

If a defect is discovered after publication:

1. stop recommending and promoting the affected coordinated version;
2. assess whether all five distributions or only selected files require an index
   yank, and record the user-visible reason;
3. preserve the tag, provenance, hashes, and incident record needed to explain
   existing installations;
4. fix forward under a new version rather than overwriting published files;
5. rerun the complete local, TestPyPI, provenance, and production checks; and
6. reconcile documentation and public-evidence links to the replacement.

Yanking is a discovery warning, not remote uninstallation or revocation of
bytes already downloaded. Security-sensitive compromise also requires the key,
runtime, or dependency response described in the security documentation.

## Keep the signature domains separate

Three signature or provenance domains can appear in a deployment:

- package build provenance associates distributions with a release build;
- the evidence signature authenticates one evidence manifest; and
- the verifier signature authenticates one receipt and its independent anchors.

None substitutes for another. Package provenance does not approve evaluation
results, an evidence signer does not choose verifier trust anchors, and a
verification receipt does not attest how a Python wheel was built.

## Version compatibility

Do not infer compatibility from package names alone. Confirm all of:

- coordinated first-party package versions;
- the add-in's declared core version range;
- exact runtime-provider ABI equality;
- conformance-command success; and
- for native providers, compatibility of the authenticated artifact, pinned
  runtime image, runner/backend, device, and compute capability.

Evidence format versions are independent of package versions. A package update
may preserve an existing format exactly; a breaking artifact interpretation
requires a new format identifier and explicit reader support.

## Reproducibility boundaries

The workflow proves that its uploaded distributions came from one selected
release commit under the recorded build workflow. It does not claim bit-for-bit
reproducibility across arbitrary builders. Operators needing that stronger
property must independently rebuild the same source, control the full build
environment, and compare each wheel and source-distribution digest.

Likewise, installing a verified wheel does not validate a runtime image. OCI
image digests, provider/backend identities, model artifacts, schedules, and
evidence signatures remain separate dependency chains checked by evaluation and
verification.

## Related documentation

- [Runtime providers](runtime-providers.md) defines provider ABI compatibility
  and first-party add-in conformance.
- [Public contracts](contracts.md) separates evidence-format versioning from
  package releases.
- [Architecture](architecture.md) distinguishes distribution provenance from
  evidence creation and independent acceptance.
- [Environment variables](environment.md) lists the runtime-image and native
  resource inputs checked after installation.
