# Release and distribution verification

InvarLock ships one Python distribution with public optional-dependency extras:

> **Reference**
>
> **Surface:** The core distribution, release checks, provenance, and installation verification
>
> **Stability:** Coordinated version and ABI compatibility rules are public; the release workflow may evolve while preserving those checks
>
> **Use this page when:** Building, publishing, installing, or independently checking an InvarLock release set

| Distribution | Role |
| --- | --- |
| `invarlock` | Engine, contracts, all maintained runtime providers, bounded judge collection/import, observation-only diagnostics, verifier, and renderer |

The wheel and source distribution include the repository license text; the
distribution gate checks their contents and metadata.

Live judge collection uses the core `invarlock[judge]` extra, which installs
pinned Inspect, OpenAI/OpenRouter, Anthropic, Google, and `httpx` dependencies.
Offline judge import and replay require only the core wheel.

All built-in providers are versioned with `invarlock` and must match
runtime-provider ABI `1` when loaded. Optional third-party providers remain
independently versioned and are accepted only through the explicit extension
policy.

| Compatibility dimension | Required check |
| --- | --- |
| Package version | The wheel and source distribution use the same release version |
| Dependency metadata | Optional extras contain only user-facing runtime dependencies |
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
5. builds exactly one wheel and source distribution for `invarlock`;
6. validates every archive against the exact checkout and runs the release
   preflight again from a clean detached checkout, including full replay of
   all seven retained public signed evidence packs and all four retained
   evaluator-qualification transactions through the isolated candidate-wheel
   CLI;
7. exercises all 19 maintained evaluator profiles and both import routes through
   an independently installed recipient, then replays the 184 signed packs from
   the retained evaluator campaigns without new model or judge calls;
8. runs `twine check` on both archives;
9. installs the wheel alone in a clean environment outside the checkout and
   exercises the CLI, captured consumer, offline judge consumer, diagnostics,
   and provider entry-point discovery;
10. exercises the public CLI, all provider conformance commands, diagnostics,
   and entry-point discovery;
11. audits the installed dependency surface and generates an SBOM;
12. records the wheel and source archive in one SHA-256 ledger and attaches build-provenance
    attestations during the tag run;
13. after a complete TestPyPI or PyPI publication, verifies both hosted
    archives against that tag-run ledger, installs the hosted wheel,
   and repeats the conformance smoke; and
14. after a verified production PyPI run, publishes the
    documentation from that exact tag source to its immutable version path,
    `latest`, and `stable` in one serialized `gh-pages` commit.

These checks authenticate and exercise the package set. They do not qualify a
specific model artifact, runtime image, accelerator, dataset, or evidence pack.
Those belong to the evaluate/verify trust model.

Before tagging, dispatch the release workflow from the candidate branch with
`publish` disabled, `release_tag` empty, and `candidate_version` set to the
package version without a leading `v`. This runs the complete Linux build and
release gates against the workflow event commit without creating an
authoritative candidate or publishing anything. The validation job has
read-only repository permissions. Attestation write access and the OIDC token
used for provenance are granted only to the short tag-only job that downloads
the already validated archive set, rechecks its ledger, and creates the
provenance statement.
Publication preparation likewise runs without an identity token. The
environment-gated publish job downloads the two previously validated archives,
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
commit. The install smoke builds the one distribution, installs its pinned base
dependency closure and candidate wheel in a disposable virtual environment,
runs `pip check`, and exercises diagnostics, provider discovery, and conformance
without the checkout or user site on `sys.path`.
The smoke selects the maintained Python 3.12 or 3.13 lock for the invoking
interpreter and fails closed when no matching lock exists.
The read-only preflight then inspects the core wheel and source distribution
against a separately generated `sha256sum`-format manifest, installs the
candidate wheel in isolation, exercises its runtime surface, replays all eleven
retained release-evidence packs through `verify` and `report`, and reruns the
public-evidence audit. Separate closed reference sets must exactly cover the
seven public-evidence directories and four evaluator-qualification transaction
directories, so adding or removing a carrier requires an explicit
release-compatibility update. The evaluator replay must preserve the exact
declared outcome: one policy pass and three integrity-valid policy rejections.

The designated Qwen3.8 reference can be exercised directly against the current
source-tree verifier with `make release-reference-journey`. Use
`make release-public-evidence-compatibility` for the seven public packs,
`make release-evaluator-qualification-compatibility` for the four evaluator
transactions, or `make release-retained-evidence-compatibility` for both closed
sets. The release preflight remains authoritative for the isolated
candidate-wheel path.

Run the local release gates with:

```bash
make install-smoke

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
change. Each retained replay supplies a fresh verifier key and an external,
digest-pinned policy and trust profile. It requires the expected artifact,
runtime, schedule, request, and evidence-signer anchors, a freshly signed
receipt, and byte-identical repeated HTML reports. The machine-readable result
records every pack-manifest digest plus its fresh receipt and deterministic
report digests. It does not rewrite checked-in evidence or receipts, rerun model
or evaluator inference or conversion, or generalize any retained transaction
to other artifacts or policies. A passing result is release-candidate
compatibility evidence, not recipient authorization to deploy. Run the
non-publishing branch workflow after these local checks to exercise the same
release surface on the hosted Linux runner before creating a release tag.

Local preflight validates the one wheel/sdist pair against the checkout,
including version, source contents, metadata and license. The external hash
manifest and isolated execution/replay consumer cover that same pair;
`make dist-check` also validates the archives against the checkout. Provider
discovery runs from the minimal installed wheel in `make install-smoke`, where
NumPy and Pillow are asserted absent. The same target creates a second isolated
environment from the hash-pinned optional-feature lock and runs diagnostics and
vision-text conformance there.

The captured wheel consumer checks the exact three-command root, all three
synthetic starters, signed trust-v2 handoffs and receipt-v3
authentication, unsigned rejection, adverse policy gates, repeated destinations,
unchanged evidence bytes, and report-v2 `requested_outputs`/`written_outputs`.
The judge core-wheel consumer imports the committed bounded fixture, publishes
signed evidence, computes independent recipient pins before publication, and
replays verification and all three report formats. It preserves the fixture's
`insufficient_evidence` decision and checks wrong signer, plan, subject, and
unsigned evidence rejection. It runs without provider SDK extras and requires
Inspect and OpenAI SDK modules to be absent. Provider SDK collection is qualified
separately by `make inspect-judge-sdk-test`; Langfuse export is qualified by
`make langfuse-sdk-test`. Offline replay needs only core.

`make evaluator-parity-test` installs the candidate wheel into an independent
recipient, exercises all 19 maintained evaluator profiles across the supported
scorer and import routes, and replays the four retained fresh-campaign archives.
Those archives contain 76 sentinel exact-match/NLL packs, 52 sentinel judge
packs, 32 priority exact-match/NLL packs and 24 priority judge packs. The gate
authenticates and preserves their original outcomes; it does not rerun models or
call judge providers.

Routine pull-request jobs run the same 114 installed profile journeys on Python
3.12 and 3.13 with retained campaign replay disabled. The interpreter-independent
campaign archives replay once in the full release workflow, avoiding duplicate
work while keeping the release gate complete.

The three-scorer consumer additionally exercises exact-match, normalized-NLL and
judge selection through captured v2 requests and public SDK capture/import helpers.
Its copied fixture inventory includes `collection.json`; tests exercise that
exact inventory outside the checkout. The NLL contract fixtures are synthetic.
The separate [Harness likelihood reference](https://github.com/invarlock/invarlock/blob/main/examples/captured-results/references/harness-likelihood/README.md)
retains real CPU measurements and an installed signed journey for one pinned
same-model profile. Its file hashes, executed capture source, regenerated anchors,
receipt and model identity in reports are replayed by the example tests.

Retained native release consumers request HTML and Markdown and validate report
v2 with `kind: runtime`; native console-only and HTML-only callers use the same
result contract.
Validation preserves historical native receipts, upstream fixture bytes, and
recorded K2 qualification status. Optional runtime tests are not
substitutes for observed inference.

For released-wheel examples, follow
[Matching wheels and examples](../user-guide/getting-started.md#matching-wheels-and-examples):
select the tag archive from `importlib.metadata.version("invarlock")`. Local
wheels require their exact build checkout's examples, with no mutable-branch
fallback when a matching release archive is unavailable.

## Test index and production publication

TestPyPI rehearses publication and installation using the production candidate.
For changes to package descriptions, packaging, installation or publication,
complete this rehearsal and inspect the rendered project pages before production.
Other releases can use it as an optional additional check. Dispatch the release workflow from the
release tag, select `testpypi`, and provide the successful tag workflow run's
numeric ID as `candidate_run_id`. The workflow authenticates that the supplied
run is a successful tag-push execution of the release workflow at the exact tag
and commit. It then downloads that run's immutable distribution artifact,
checks the closed two-file set and its ledger, publishes through the
project-scoped TestPyPI identity, verifies both hosted archives, installs the
hosted wheel, and reruns the CLI, diagnostics,
provider-conformance, and entry-point smoke.

A successful upload does not check presentation. Extract the description from
both the wheel and source archive, confirm their parity, and render the exact
text with the PyPI Markdown renderer before tagging. Check that the logo and
workflow diagram are image elements, rather than escaped HTML code. Inspect the
actual version-specific TestPyPI pages for those images, headings, tables, code
and release links, then repeat the page check on production. Record these manual
checks separately from CI; the workflow does not enforce the visual review.

Check the distribution's trusted publisher and matching GitHub environment on
both indexes before dispatch. TestPyPI registration is separate from production
registration.
Submitting a publisher registration, satisfying a deployment approval and
successfully publishing are separate steps.

Production uses the same tagged candidate directly. Once local preflight,
tag-to-commit checks, release notes, security review, provenance, and the tag
workflow are complete, dispatch the workflow again from the release tag,
select `pypi`, and provide the same tag workflow run ID as `candidate_run_id`.
The production job consumes the exact archives built by the tag run; it does
not rebuild them or depend on TestPyPI state. A stale, incomplete, or
filename-colliding TestPyPI project therefore cannot silently select or alter a
production candidate. Before upload, the publication job downloads and checks
any already hosted files for its exact version against the candidate ledger.
Absent files may be uploaded and ledger-identical files may be skipped, making
an interrupted publication safe to resume. Any conflicting
filename, metadata digest, or downloaded bytes fail before upload, and the
post-publication verifier cannot turn a partial or replaced release
into a successful run.

Every publication follows one complete path: upload, hosted digest verification,
installed-wheel smoke, and, for production, documentation publication. A retry
uses that same path and the same authenticated candidate.

Configure a protected `v*` tag ruleset that blocks updates and deletion. Protect
each project-scoped PyPI environment with an appropriate release authorization
and deployment policy. These repository controls provide the authorization layer
around the workflow's commit, tag-run artifact, ledger, and trusted-publisher
identity.

After production publication, the workflow downloads both archives from PyPI,
compares their hashes with the tag-run ledger, installs the hosted wheel in a
clean environment, and repeats the conformance smoke.
Only after that smoke succeeds does the production workflow invoke the reusable
documentation publisher. The publisher removes the leading `v` from the
validated release tag, builds from the caller's exact tag commit, and updates
the versioned path plus `latest` while making `stable` redirect to the immutable
version. TestPyPI publication cannot update Pages, and one global
documentation concurrency group prevents release and branch publishers from
racing their `gh-pages` pushes.
Reconcile the published filenames, version, source tag, provenance subjects,
and release assets before announcing completion.

## Installation and provenance checks

Before installing a release in a controlled environment:

1. select one exact `invarlock` version;
2. obtain hashes from a trusted release record or package index response;
3. download artifacts without installing them;
4. verify every downloaded SHA-256 digest;
5. install with hash enforcement where the package-management workflow supports
   it; and
6. run `invarlock --version` and the provider conformance commands exposed by
   the installed package.

Example discovery checks after installation:

```bash
invarlock --version
python -m invarlock.runtime_providers.llama_cpp_conformance --help
python -m invarlock.runtime_providers.tensorrt_llm_conformance --help
```

Each conformance command must report `ok: true`, its expected provider name,
and the ABI accepted by the installed core. A conformance pass verifies the
install surface and lightweight provider contract, not a native runtime model
run. Before qualification fan-out, produce and strictly verify one signed
canary through the exact digest-pinned runtime image. Retain its evidence,
signed receipt, original verifier-owned trust profile and referenced verifier
private key for the maintained readiness and evidence targets. Reuse requires
matching image, providers, task, acceptance binding and CPU/CUDA device class;
see [canary compatibility](runtime-providers.md). A canary does not establish
model-specific load, memory, backend, or execution success.

An example hash-enforced download/install flow is:

```bash
python -m pip download --only-binary=:all: --dest wheelhouse \
  'invarlock==X.Y.Z'

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
| Package set | Exactly one coordinated `invarlock` wheel/source pair |

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
  --certificate-identity "https://github.com/OWNER/REPO/.github/workflows/SIGNING_WORKFLOW.yml@refs/tags/vX.Y.Z" \
  --dist-dir release-material/dist \
  --sbom release-material/sbom.json \
  --provenance-dir release-material/provenance \
  --output-dir release-material/offline
```

This script assembles existing material; it does not fetch or manufacture
provenance. Supply the independently approved certificate identity of the actual
signing workflow; a repository OIDC subject is not its certificate identity.
The assembler inventories the release distribution directory. Every
distribution must have its own adjacent Sigstore sidecar; unlisted files and
symbolic links are rejected. When the release `SHA256SUMS` ledger is present,
its entries must match every distribution path and digest; the manifest retains
the ledger as a supporting file. Inspect the generated `release_manifest.json`,
verify each file digest, then follow the bundle's `README.txt` identity and issuer checks. The current GitHub workflow
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
2. assess whether the release files require an index yank, and record the
   user-visible reason;
3. preserve the tag, provenance, hashes, and incident record needed to explain
   existing installations;
4. fix forward under a new version rather than overwriting published files;
5. rerun the complete local, provenance, production and any selected TestPyPI checks; and
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
- the package's declared optional dependency boundaries;
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
  and built-in provider conformance.
- [Public contracts](contracts.md) separates evidence-format versioning from
  package releases.
- [Architecture](architecture.md) distinguishes distribution provenance from
  evidence creation and independent acceptance.
- [Environment variables](environment.md) lists the runtime-image and native
  resource inputs checked after installation.
