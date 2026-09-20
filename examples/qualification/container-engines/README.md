# Container engine execution reference

> **Surface:** Retained Docker and Podman CUDA execution evidence and signed
> verification receipts.
>
> **Stability:** Historical evidence bound to the source, wheel, models, images
> and policies recorded in `provenance.json`.
>
> **Use this page when:** Checking which container paths were exercised, or
> replaying their signed results without running models.

This reference retains 14 successful evidence packs from Docker 29.8.1 and
rootful Podman 5.7.0 on Ubuntu 26.04 with one NVIDIA H100 80GB HBM3. It establishes
execution and offline verification for these exact configurations. It is not a
model-quality qualification, a rootless GPU qualification, or a release
qualification. The separate [public evidence index](../../../public_evidence/README.md)
retains the broader model comparisons.

## Replay without a GPU

Use Python 3.12 or newer with an installed InvarLock wheel and its core
dependencies. See the [installation guide](../../../docs/user-guide/getting-started.md)
for matching a wheel to its source checkout. Replay needs no container engine,
model download, GPU dependency, signing key or external service. From the
repository root:

```bash
python -I examples/qualification/container-engines/replay.py
```

Expected output has `"ok": true`, `"pack_count": 14`, and successful evidence and
receipt checks for every row. The script authenticates the complete archive
against a fixed SHA-256 before extracting it into a temporary directory. It
uses the installed package, replays each policy and checks the existing signed
receipt with its retained public key; it does not issue new receipts.

For manual inspection, first check the pin, then extract:

```bash
cd examples/qualification/container-engines
shasum -a 256 -c reference.sha256
python -m zipfile -e reference.zip extracted-reference
```

The archive contains 373 files, including a manifest of every payload file's
size and hash. It is 344,031 bytes compressed and 721,833 bytes expanded.
Entries are sorted, use fixed timestamps and regular-file modes, and preserve
all original signed bytes. Do not edit the extracted evidence or trust profiles.
The historical trust profiles mention private signing-key filenames because
receipt signatures bind those profiles; only public keys are distributed and
replay never opens the private-key paths.

The repository archive pin and retained public trust inputs are the recipient's
reference anchors. A recipient must decide to trust those anchors independently;
signatures establish byte integrity and the declared signer, not independent
attestation of the host or hardware.

## What was exercised

| Path | Docker | Podman | Retained result and limit |
| --- | --- | --- | --- |
| Native exact match | CUDA execution and installed-wheel lane | Same | Two signed packs per engine; deterministic tiny model fixture |
| Native normalized NLL | CUDA execution and installed-wheel lane | Same | Two signed packs per engine; likelihood plumbing fixture |
| Native judge | CUDA answer generation | Same | One signed pack per engine; fixed offline judge measurements, no live judge calls |
| TensorRT-LLM | BF16/FP8 evaluation | Same engine bytes evaluated again | One signed pack per engine; 102 generated records |
| Vision-text | Qwen2-VL 2B/7B evaluation | Same checkpoint bytes evaluated again | One signed pack per engine; four fixed color-grid cases |

All three unchanged native scorer tests passed under each engine with no skips.
Native exact-match and normalized-NLL tests each also exercised their isolated
installed-wheel qualification lane. Both engines produced byte-identical paired
record payloads for the native exact-match, native NLL, TensorRT and vision rows.
GPU work was serialized. Docker used its explicit GPU selection; Podman used
NVIDIA Container Device Interface (CDI) devices, with no engine fallback.

TensorRT preparation built the BF16 baseline with Docker and the FP8 subject
with Podman. Both evaluations then used the same seven engine/input files,
checked by hash before and after execution. Both sides answered 52 of 102
records correctly. The delta was 0 percentage points (pp), with a 95% paired
interval of approximately −4.99 to +4.99 pp. The frozen policy required a lower
bound of at least −10 pp, interval width at most 20 pp, at least 102 records and
side accuracy at least 0.40. Both engine runs passed that policy.

Vision used Qwen2-VL-2B-Instruct as baseline and Qwen2-VL-7B-Instruct as subject.
They answered 1/4 and 3/4 cases correctly: +50 pp, with a 95% paired interval of
approximately −13.55 to +78.91 pp. The unchanged tutorial policy allowed a lower
bound of −100 pp and width of 200 pp with four records. Both runs passed this
permissive policy; these four cases do not establish a general quality advantage.

Each TensorRT and vision example completed `evaluate`, `verify` and `report`
with exit status zero. These execution results are distinct from the recorded
policy verdicts above. All 14 retained packs and signed receipts also passed a
separate offline replay with the exact installed wheel identified below.

## Provenance and limits

[provenance.json](provenance.json) records full source, wheel, model revision,
engine-file, image manifest/configuration and root filesystem layer identities,
plus the original policies' observed results. The executed source was
`59d9bbc048d1fe10083e912b32912445cb7bb974`; its wheel was named
`invarlock-0.16.1-py3-none-any.whl` with SHA-256
`f7663dd4a177e9497146461805a386e09e8fd696f57fb377acfd5f7664cc788e`.
That filename does not identify the published release wheel. Offline replay
checked 183 installed files against that exact wheel and used core dependencies
only. Later source revisions and wheels require their own execution evidence.

Docker reported OCI manifest identities while Podman reported configuration
identities. Transport retained the same original manifests, configuration
bytes and ordered filesystem layers; Podman execution used authenticated
manifest-addressed references. Equality of the engines' displayed image IDs
was neither assumed nor used as proof of equivalent image contents.

Earlier setup attempts stopped on an incorrect image-ID equality assumption,
wrapper interpreter lookup, generated wheelhouse cleanup, missing container
init support, restrictive extracted-source permissions, or missing host Pillow.
Those failures are retained in the full execution archive; this compact public
bundle includes the successful signed results and this summary of setup causes.
The source contents, model settings and policies were not retuned after model
outcomes. No model files, engine binaries, wheels, private keys, host addresses
or full execution logs are included here.
