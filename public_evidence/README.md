# Public Evidence

This directory contains the compact current-evidence index for the maintained
evaluation lanes in `contracts/evidence_catalog_v1.json`. Full evidence packs
are distributed as the GitHub Release asset named and hash-bound by
`catalog_evidence_index.json`.

All 39 catalog lanes use `edit_name: noop` and evaluate the same declared
checkpoint as baseline and subject. They are compatibility runs that exercise
the evidence mechanics: model loading, input materialization, paired
evaluation, guard/report generation, strict verification, and packaging for
each lane. Strictly verified evidence is currently available for 31 lanes; the
other 8 remain `not_created`. Transformation detection and guard effectiveness
require separate runs with an actual edited subject and are outside the claim
made by these catalog results.

The 31 available packs were produced with the frozen v1 claim set. The current
verifier accepts them through its explicit v1 compatibility path, so they
remain valid strictly verified artifacts for their declared runs. They do not
exercise the v2 `guard_authority` fields, and this index does not imply that
they do.

The catalog and support matrix remain useful before a lane has been run: they
define the model, adapter, preset, input materialization, execution policy, and
required artifact roles. Catalog evidence is added to the compact current index
only after the corresponding run has completed and passed strict verification
under the verification contract applicable to its declared claim set.
Historical observation indexes are identified and scoped separately below.

## Evidence status

| Status | Meaning |
| --- | --- |
| `not_created` | The maintained catalog lane is defined for evaluation; current evidence has not yet been created. |
| `available` | Current evidence is listed in the compact index and available from its hash-bound release asset. |

The status for every maintained lane is recorded in
`contracts/support_matrix.json`. Until a current artifact is available, the
human-readable label is **Evidence not yet created**.

## Artifact shape

Each available lane's archive entry supplies the artifact roles declared by its
catalog entry, including the evaluation report, runtime manifest, final
verdict, source and input provenance, resolved configuration, preset,
independent baseline, and policy pack. The index records each logical path,
artifact digest and size, plus the containing archive's URL, digest, size, and
root.

Current evidence content is additive: publishing one lane changes that lane's
status and adds its artifacts while the remaining catalog rows continue to read
`not_created`. The complete evidence tree is distributed as one shared archive,
so rebuilding it changes the digest and size and requires refreshing the shared
archive binding recorded by every available entry. Regenerate the compact index
atomically with the archive instead of editing only the new lane's entry.

The current index can contain `public_evidence/published_basis/...` paths for
immutable assets published before the terminology update. Those values are
historical archive bindings, not a support tier. Future archives produced after
this contract release use `public_evidence/catalog_evidence/`.

## Historical guard-scenario observations

[guard_scenario_observations.json](guard_scenario_observations.json) hash-binds
selected historical Mistral-7B observations in an immutable release asset. The
file explicitly records that those reports predate the current report schema
and do not carry current strict assurance. Its content-aware checker validates
the archive and recomputes the indexed primary-metric, spectral, RMT, and
variance observations; it does not upgrade the reports to the current contract.

Current verdict behavior is characterized separately by deterministic fixture
tests in
[`test_evidence_pack_verdict_guard_value_contract.py`](../tests/evidence_packs/test_evidence_pack_verdict_guard_value_contract.py).
Those synthetic tests cover primary-metric-pass cases with baseline-relative
spectral, RMT, and variance findings plus a negative control. They test the
current scenario/verdict contract only: they are neither model-run evidence nor
empirical guard-effectiveness evidence.

## Verification

Download the release asset recorded in `catalog_evidence_index.json`, verify its
SHA-256 and size, and unpack it at the repository root. Then use the public
verifier commands to inspect an available artifact:

```bash
invarlock advanced evidence-pack inspect PATH --json
invarlock advanced evidence-pack verify PATH --strict --json
```

The canonical JSON artifacts are the review source of truth. HTML reports are
rendered views of the same evaluation result. Maintainers can also verify the
download automatically with:

```bash
python scripts/checks/check_public_evidence.py --fetch-external-assets
```
