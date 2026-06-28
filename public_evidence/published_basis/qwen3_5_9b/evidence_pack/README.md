# InvarLock Evidence Pack

This evidence pack bundles reports, summary reports, and metadata for offline
verification. No model weights are included.

Evidence level: high
Review summary: clean_reports=1, error_injection_reports=0, failed_reports=0, profile=release.

Why it might be wrong:
- Nested report verification succeeded for the bundled clean reports, but evidence readers should still inspect the underlying evaluation.report.json files.
- Error-injection reports are expected-failure evidence and should not be interpreted as clean PASS runs.
- The pack is ready for strict verification; signed manifest and checksum sealing are present.
- Signer fingerprint: sha256:044075dad4fd8c4d117f2617046120387b22df214d828c779f971446e66bc0ca

## Verify

1. Verify the manifest signature and strict pack integrity:
   invarlock advanced evidence-pack verify <pack-dir> --strict

2. Verify file checksums:
   sha256sum -c checksums.sha256
   # macOS: shasum -a 256 -c checksums.sha256

3. Verify report integrity:
   invarlock verify --json reports/**/evaluation.report.json

Or use:
  invarlock advanced evidence-pack verify <pack-dir> --strict
Repo workflow alternative:
  scripts/evidence_packs/verify_pack.sh --pack <pack-dir> --strict
