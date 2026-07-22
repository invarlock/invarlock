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

For a tagged release commit or an explicitly selected pre-tag candidate, the
repository workflow:

1. binds a pre-tag validation to the workflow event commit and declared package
   version, or resolves the exact tag commit for a tag build or publication;
2. requires a tag build or publication event commit to equal the resolved tag
   commit;
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
11. records all ten archives in one SHA-256 ledger and attaches build-provenance
    attestations during the tag run; and
12. after a complete TestPyPI or PyPI publication, verifies every hosted
    archive against that tag-run ledger, installs the hosted wheels together,
    and repeats the conformance smoke.

These checks authenticate and exercise the package set. They do not qualify a
specific model artifact, runtime image, accelerator, dataset, or evidence pack.
Those belong to the evaluate/verify trust model.

Before tagging, dispatch the release workflow from the candidate branch with
`publish` disabled, `release_tag` empty, and `candidate_version` set to the
package version without a leading `v`. This runs the complete Linux build and
release gates against the workflow event commit without creating an
authoritative candidate or publishing anything. The validation job has
read-only repository permissions. OIDC and attestation write permissions are
granted only to the short tag-only job that downloads the already validated
archive set, rechecks its ledger, and creates the provenance statement.
Publication preparation likewise runs without an identity token. Each
environment-gated publish job downloads only its two previously validated archives,
rechecks the immutable tag, and invokes the pinned trusted-publishing action;
it does not check out or execute candidate Python code.

For a tag build or publication, the workflow resolves and checks out the
release tag's exact commit before it builds. A tag push validates and builds
the authoritative candidate but does not publish it. Publication is an
explicit manual action against an existing tag. The manual workflow must be
dispatched with that tag as its workflow ref and the successful tag-run ID as
`candidate_run_id`, so the event commit, resolved tag commit, workflow-run
identity, and downloaded artifact agree.
Manual publication does not rebuild the archives. The tagged distribution and
provenance artifacts are retained for 14 days, so publication must complete
within that interval.

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
tag or publish. Run the non-publishing branch workflow after these local checks
to exercise the same release surface on the hosted Linux runner before creating
a release tag.

The local preflight intentionally validates the core pair in depth while
`make dist-check` validates every archive against its checkout source and
`make addins-install-smoke` plus the release workflow validate the coordinated
five-package install. Keep both kinds of gate; do not describe the core-only
JSON result as proof that every add-in archive was independently inspected by
preflight.

## Test index and production publication

TestPyPI is an optional publication and installation smoke, not the source of
the production candidate. To use it, dispatch the release workflow from the
release tag, select `testpypi`, and provide the successful tag workflow run's
numeric ID as `candidate_run_id`. The workflow authenticates that the supplied
run is a successful tag-push execution of the release workflow at the exact tag
and commit. It then downloads that run's immutable distribution artifact,
checks the closed ten-file set and its ledger, publishes through the five
project-scoped TestPyPI identities, verifies all hosted archives, installs the
five hosted wheels together, and reruns the CLI, diagnostics,
provider-conformance, and entry-point smoke.

Production uses the same tagged candidate directly. Once local preflight,
tag-to-commit checks, release notes, security review, provenance, and the tag
workflow are complete, dispatch the workflow again from the release tag,
select `pypi`, and provide the same tag workflow run ID as `candidate_run_id`.
Use the default `complete` publication phase for an ordinary release.
The production jobs consume the exact archives built by the tag run; they do
not rebuild them or depend on TestPyPI state. A stale, incomplete, or
filename-colliding TestPyPI project therefore cannot silently select or alter a
production candidate. Before upload, each publication job downloads and checks
any already hosted files for its exact version against the candidate ledger.
Absent files may be uploaded and ledger-identical files may be skipped, making
an interrupted multi-project publication safe to resume. Any conflicting
filename, metadata digest, or downloaded bytes fail before upload, and the
post-publication verifier cannot turn a mixed or partially replaced release
into a successful run.

PyPI permits only three pending trusted publishers at once. When a coordinated
release creates four new add-in projects for the first time, use the production
`bootstrap` phase to publish the core plus the diagnostics, GGUF, and
vision-text distributions. After those pending publishers become ordinary
project publishers, register the TensorRT-LLM publisher and dispatch the
`finish` phase with the same release tag and `candidate_run_id`. The finish run
first confirms that all eight bootstrap archives are hosted byte-for-byte from
that ledger, publishes only TensorRT-LLM, then verifies all ten hosted archives
and installs all five wheels together. The two special phases are rejected for
TestPyPI; later releases use `complete` and publish all five projects in one
dispatch.

Configure a protected `v*` tag ruleset that blocks updates and deletion. Protect
each project-scoped PyPI environment with required reviewers and an appropriate
deployment policy. These repository controls provide the authorization layer
around the workflow's commit, tag-run artifact, ledger, and trusted-publisher
identity.

After production publication, the workflow downloads all ten archives from
PyPI, compares their hashes with the tag-run ledger, installs the five hosted
wheels together in a clean environment, and repeats the conformance smoke.
Reconcile the published filenames, version, source tag, provenance subjects,
and release assets before announcing completion.

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
