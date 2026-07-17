# Publish public evidence

Public evidence is a curated index of immutable `invarlock/evidence-pack-v1` directories
and independently signed verification receipts. An empty index intentionally
says **Evidence not yet created**; publication begins only after a pack and
receipt pass strict verification and disclosure review.

!!! tip "User guide"

    **Outcome:** Add one strictly verified evidence entry, publish it locally or as a hash-bound release asset, and keep the compact wheel index synchronized.
    **Audience:** Evidence maintainers preparing public, independently reviewable InvarLock artifacts.
    **Prerequisites:** An immutable signed evidence pack, a separately signed verification receipt, independently maintained anchors, permission to publish every record, and a clean repository checkout.

## Publication boundary

The source tree has two carriers:

| Carrier | Contains | Intended use |
| --- | --- | --- |
| Repository-local entry | Full evidence directory, receipt, and metadata beneath `public_evidence/evidence/SLUG/` | Small packs that are appropriate for Git history |
| External release asset | Archive available over HTTPS plus digest metadata in the index | Large packs retained outside Git while remaining content-addressed |

Installed wheels contain only `evidence_index.json`. They do not silently
embed full packs. The index is a discovery and transport-integrity surface; it
does not replace `invarlock verify` or the caller's independent artifact,
schedule, policy, runtime, evidence-signer, and verifier anchors.

## Prepare a local entry

Choose a stable, portable slug and stage exactly three objects:

```text
public_evidence/evidence/example-comparison/
├── evidence/                       # immutable invarlock/evidence-pack-v1 directory
├── verification.receipt.json      # separately signed receipt
└── evidence.meta.json              # publication metadata
```

`evidence.meta.json` is a small closed publication record:

```json
{
  "artifact_paths": {
    "evidence_pack": "evidence",
    "verification_receipt": "verification.receipt.json"
  },
  "format_version": "invarlock/public-evidence-meta-v1",
  "summary": "Authenticated baseline-versus-subject comparison under the published policy"
}
```

The pack must be a real directory, the receipt must be a regular file beside
it, and neither may be a symbolic link. The metadata paths are relative to the
entry and cannot escape it.

Before indexing, independently repeat strict verification into a fresh receipt
destination outside the pack and validate the received receipt with
`verify_signed_verification_receipt`. An evidence signature, a report-local pass,
or an existing receipt alone is not sufficient.

## Disclosure review

Review the entire pack and receipt before public staging. In particular, search
record text, model outputs, locators, summaries, runtime descriptions, and error
messages for:

- private or proprietary prompts and outputs;
- access tokens, credentials, private keys, or signed download URLs;
- absolute user, host, cache, or temporary paths;
- private host names, addresses, or environment dumps;
- unpublished model or dataset identifiers; and
- operational notes that do not belong in public evidence.

Cryptographic validity does not make content safe to disclose. If redaction
would change authenticated bytes, do not edit the pack; create a new evaluation
from publication-safe source material.

## Build and audit the index

For repository-local entries, rebuild the compact package index and run the
public evidence audit:

```bash
make public-evidence-sync
make public-evidence-audit
```

The generated entry records:

- slug, logical path, summary, and evidence class;
- pack kind, file count, total bytes, and control-file hashes; and
- receipt path, byte size, and SHA-256 digest.

This repository audit validates publication structure, local carrier bytes,
the receipt's Ed25519 signature and embedded verifier-key fingerprint, and the
receipt-to-manifest binding. Authorization remains with independently managed
artifact, schedule, policy, runtime, evidence-signer, and verifier anchors.
Perform strict verification and independently anchored receipt validation
before indexing, and retain those review inputs outside the submitted pack.

The synchronizer rebuilds the source index from local metadata, preserves
validated external-only entries, and writes the same bytes to the package-owned
index. Every local directory must appear in the index; external-only entries
may coexist without empty placeholder directories. Reported artifact sizes,
file counts, and hashes are checked against each local carrier.

## Use an external release asset

For a large pack, create a deterministic archive containing the immutable
evidence directory and signed receipt. Publish the archive as a release asset,
then record an HTTPS URL and independently computed SHA-256 in the corresponding
artifact summary's `external_asset` object:

```json
"external_asset": {
  "sha256": "sha256:VERIFIED_ARCHIVE_DIGEST",
  "url": "https://github.com/OWNER/REPOSITORY/releases/download/TAG/ASSET"
}
```

Use a real lowercase digest in the index. The shown value, owner, repository,
tag, and asset name are placeholders. The audit requires HTTPS and a canonical
`sha256:` value when the local artifact is absent.

An external-only entry is authored directly in
`public_evidence/evidence_index.json`, because there is no local
`evidence.meta.json` from which the synchronizer could derive it. Include the
same closed entry and artifact-summary fields produced for a local entry, but
give each absent pack/receipt summary an `external_asset`. Then run
`make public-evidence-sync` to preserve that entry while rebuilding local
entries and to write both index copies. Then run `make public-evidence-audit` to
validate both copies. Do not create an empty local entry directory: each local
directory must contain the actual pack, receipt, and metadata. When an external
and local entry share a slug, the locally derived record replaces the external
record.

An external archive remains subject to the same pack and receipt checks after
download:

1. obtain the expected archive digest from the independently maintained index;
2. download the exact asset without executing it;
3. hash the downloaded bytes and compare the result;
4. extract into a new bounded directory without following links;
5. run `invarlock verify` with independent acceptance anchors; and
6. validate the signed receipt against the expected verifier identity and key.

The release asset URL is a carrier locator, not identity. If an asset is
re-uploaded or moved, its digest and publication record must be revalidated.

## Empty-state behavior

When no evidence entries exist, retain the checked-in index:

```json
{
  "carrier_policy": {"installed_wheel": "compact_index_only"},
  "entries": [],
  "evidence_count": 0,
  "evidence_file_count": 0,
  "evidence_size_bytes": 0,
  "format_version": "invarlock/public-evidence-index-v1",
  "status": "not_created",
  "status_label": "Evidence not yet created"
}
```

Do not replace the empty state with historical failures, batch-run notes, or
unverified placeholders. Entries appear only when publishable evidence exists.

## Final publication checklist

- the pack is unchanged after evidence signing;
- strict verification succeeds with caller-owned anchors;
- the signed receipt independently validates and remains outside the pack;
- disclosure review covers every byte intended for publication;
- the local or external carrier is content-addressed and immutable;
- `make public-evidence-sync` makes only the expected index change;
- `make public-evidence-audit` succeeds; and
- repository documentation and release notes describe only evidence that is
  actually available.

For artifact roles, see [Evidence artifacts](../reference/artifacts.md). For
acceptance semantics, see [Evidence and verification](evidence-and-verification.md).
