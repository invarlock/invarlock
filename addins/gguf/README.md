# InvarLock GGUF runtime add-in

This optional distribution connects authenticated GGUF artifacts to InvarLock's
public runtime-provider ABI through a pinned local `llama.cpp` executable. The
InvarLock core package remains responsible for portable GGUF identity and
independent evidence verification.

## Install and check discovery

```bash
PYTHON=/path/to/venv/bin/python
"$PYTHON" -m pip install invarlock-runtime-gguf
"$PYTHON" -m invarlock_addins.gguf.conformance
```

The conformance command returns JSON with `"ok":true`, provider `llama_cpp`,
and the ABI version accepted by the installed core package. Installing the
wheel also registers `llama_cpp` in the `invarlock.runtime_providers` entry-point
group; no core source change is required.

The install above is the standalone conformance path. Maintained qualification
does not treat first-party packages already installed in that environment as
the candidate. `CANDIDATE_WHEEL_MANIFEST` must authenticate the exact core and
GGUF wheels built from the qualified source commit. The driver extracts those
wheels into a private candidate site and loads them before any first-party code
visible to `PYTHON`. `PYTHON` supplies only their third-party dependencies;
`QUALIFICATION_DRIVER_PYTHON` launches the standard-library qualification
driver. Using one isolated dependency environment for both variables is the
simplest configuration.

After `make dist-check`, create the manifest with the maintained no-clobber
helper. Its destination parent must already exist, and the destination must be
new:

```bash
install -d -m 700 qualification
python scripts/qualification_candidate_wheels.py \
  --wheel dist/invarlock-*.whl \
  --wheel dist/addins/invarlock_runtime_gguf-*.whl \
  --output "$PWD/qualification/gguf-candidate-wheels.json"
export CANDIDATE_WHEEL_MANIFEST="$PWD/qualification/gguf-candidate-wheels.json"
```

The helper records absolute wheel paths and SHA-256 digests. The qualification
driver additionally requires unique, version-aligned maintained distributions
whose contents match the authenticated source archive. Recreate the manifest
after any wheel changes, and pass the same manifest to canary, preflight, and
evidence qualification.

## Build the runtime image

Build from the repository root. The `llama.cpp` build stage deliberately has
no mutable package-index default: select and pin a Debian snapshot timestamp
in `YYYYMMDDTHHMMSSZ` form, then record it with the resulting image digest.

```bash
SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"
LLAMA_CPP_APT_SNAPSHOT=PINNED_YYYYMMDDTHHMMSSZ
SOURCE_COMMIT="$(git rev-parse HEAD)"
SOURCE_BUNDLE=/absolute/path/to/invarlock-source.tar
SOURCE_BUNDLE_SHA256=sha256:PINNED_SOURCE_BUNDLE_DIGEST

make -C addins/gguf build \
  IMAGE=invarlock-gguf:candidate \
  SOURCE_COMMIT="$SOURCE_COMMIT" \
  SOURCE_BUNDLE="$SOURCE_BUNDLE" \
  SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256" \
  SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
  LLAMA_CPP_APT_SNAPSHOT="$LLAMA_CPP_APT_SNAPSHOT" \
  BUILD_STATEMENT=/absolute/path/to/gguf-build.json
```

The add-in Makefile groups its build, import smoke, and complete
evaluate-to-verify qualification targets:

```bash
make -C addins/gguf build \
  SOURCE_COMMIT=0123456789abcdef0123456789abcdef01234567 \
  SOURCE_BUNDLE=/absolute/path/to/invarlock-source.tar \
  SOURCE_BUNDLE_SHA256=sha256:... \
  LLAMA_CPP_APT_SNAPSHOT=PINNED_YYYYMMDDTHHMMSSZ
make -C addins/gguf smoke
```

Resolve the candidate to an immutable registry digest before creating strict
evidence. The local tag is only a build handle.

The image applies the tracked
`runtime/llama-completion-user-output.patch` to the pinned b10015 source. Normal
EOG termination remains enabled, while llama.cpp's human-readable
`[end of text]` console marker is kept out of the generated-text stream. The
provider rejects that marker if an unpatched executable emits it; it never
guesses whether marker-shaped bytes were backend control or model-authored text.

## Derive request settings

Do not hand-copy artifact or backend digests. Derive the complete runtime spec
from the exact GGUF file, executable, and source archive that will be used. The
inspection itself executes native code, so this Python must run inside the
digest-pinned, network-disabled image built from
[`runtime/Dockerfile`](runtime/Dockerfile), not in an ordinary host process:

```python
from pathlib import Path
import json

from invarlock_addins.gguf.provider import LlamaCppProvider
from invarlock_addins.gguf.session import LlamaCppRuntimeBindings

bindings = LlamaCppRuntimeBindings(
    gguf_path=Path("/inputs/model.gguf"),
    executable_path=Path("/opt/llama.cpp/llama-completion"),
    source_archive_path=Path(
        "/opt/llama.cpp/source/llama.cpp-b10015.tar.gz"
    ),
)
spec = LlamaCppProvider().inspect_runtime_spec(
    bindings,
    seed=0,
    context_length=2048,
    batch_size=1,
    cpu_threads=16,
    prompt_batch_size=512,
    prompt_microbatch_size=512,
    max_output_tokens=64,
    timeout_seconds=300,
)
print(
    json.dumps(
        {"model_id": spec.model_id, "settings": dict(spec.settings)},
        indent=2,
        sort_keys=True,
    )
)
```

After resolving the built image to an immutable digest, the inspection pattern
is:

```bash
IMAGE=registry.example/invarlock-gguf@sha256:PINNED_IMAGE_DIGEST
DIGEST=sha256:PINNED_IMAGE_DIGEST

docker run --rm --network none \
  --env INVARLOCK_CONTAINER_EXECUTION=1 \
  --env INVARLOCK_RUNTIME_IMAGE="$IMAGE" \
  --env INVARLOCK_RUNTIME_IMAGE_DIGEST="$DIGEST" \
  --mount type=bind,src="$PWD/inputs",dst=/inputs,readonly \
  --entrypoint python "$IMAGE" /inputs/inspect.py
```

The Makefile keeps `IMAGE` as the convenient build and smoke handle. Its
`qualify-*` targets use `QUALIFICATION_IMAGE`, which defaults to
`IMAGE_DIGEST` so the mutable local tag never enters strict qualification.
Set `QUALIFICATION_IMAGE` explicitly to either the exact local
`sha256:...` config ID or a portable `repository@sha256:...` reference.

Use `spec.model_id` and `spec.settings` for a request side whose
`runtime.provider` is `llama_cpp`. Evaluation requires the same files as
root-confined runtime resources and the same authenticated container boundary.
Local paths are execution inputs and are not published in canonical evidence.

## Qualify one concrete runtime

The conformance command checks package discovery and ABI shape; it does not
load a GGUF model. The runtime canary is one real, minimal InvarLock
transaction over the inspected model and backend:

1. retain the printed `model_id` and `settings` as pinned request input;
2. create a one-record canonical schedule and an exact-match policy;
3. evaluate baseline and subject in independently pinned side workers, with the
   resource variables documented in the main runtime-provider guide;
4. let the host validate both side results and sign the evidence bundle without
   mounting the signing key into either worker;
5. retain the reviewed preflight `request_digest`, then verify the resulting pack
   with that digest and independently recorded artifact, schedule, runtime-image,
   and evidence-signer anchors; and
6. require both the evaluation and strict verification to finish successfully.

With those pinned request and expected-anchor files in place, the qualification
uses the same public transactions as the core package:

```bash
invarlock evaluate request.yaml \
  --signing-key evidence-signer.pem \
  --preflight --json
invarlock evaluate request.yaml --signing-key evidence-signer.pem
invarlock verify evidence/ \
  --trust-profile trust/trust-inputs.json \
  --receipt verification.receipt.json
```

Place the reviewed preflight `request_digest` in the trust profile's `anchors`
object. GGUF verification fails closed without it and emits a v2 signed receipt
that binds the same request identity.

Bootstrap that transaction once for the exact image with
`make -C addins/gguf qualify-canary`. Retain its `EVIDENCE`, strictly verified
`RECEIPT`, and verifier-owned `TRUST_PROFILE`. For every later request using
the same image digest, pass those paths to `qualify-preflight` and
`qualify-evidence` as `CANARY_EVIDENCE`, `CANARY_RECEIPT`, and
`CANARY_TRUST_PROFILE`. A different image digest requires a new signed canary.
Set `QUALIFICATION_DEVICE`, `QUALIFICATION_CPUS`,
`QUALIFICATION_MEMORY_MIB`, and `QUALIFICATION_USER` for the bounded worker
environment and keep them unchanged through canary, preflight, and evidence.
For GGUF, the device is normally `cpu`.

With the canary request, trust profile, source bundle, and output parent already
prepared, the bootstrap command is:

```bash
INVARLOCK_GGUF_RESOURCE_ROOT="$PWD/gguf-runtime" \
INVARLOCK_GGUF_BACKEND_EXECUTABLE=bin/llama-completion \
INVARLOCK_GGUF_BACKEND_SOURCE=source/llama.cpp-source.tar \
make -C addins/gguf qualify-canary \
  REQUEST="$PWD/canary/request.yaml" \
  SIGNING_KEY="$PWD/private/evidence.key" \
  QUALIFICATION_IMAGE="$IMAGE" IMAGE_DIGEST="$DIGEST" \
  EVIDENCE="$PWD/canary/evidence" \
  TRUST_PROFILE="$PWD/canary/trust-inputs.json" \
  RECEIPT="$PWD/canary/verification-receipt.json" \
  SUMMARY="$PWD/canary/qualification-summary.json" \
  CANDIDATE_WHEEL_MANIFEST="$CANDIDATE_WHEEL_MANIFEST" \
  QUALIFICATION_DEVICE=cpu \
  SOURCE_COMMIT="$SOURCE_COMMIT" SOURCE_BUNDLE="$SOURCE_BUNDLE" \
  SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256"
```

Run `make -C addins/gguf qualify-preflight` for each target request. It
reverifies the signed canary and performs execution-free checks without loading
the target model. After it succeeds, `make -C addins/gguf qualify-evidence`
evaluates and verifies with `REQUEST`, `SIGNING_KEY`, `QUALIFICATION_IMAGE`,
`IMAGE_DIGEST`,
`EVIDENCE`, `TRUST_PROFILE`, `RECEIPT`, `SUMMARY`,
`CANDIDATE_WHEEL_MANIFEST`, the three `CANARY_*`
inputs, `SOURCE_COMMIT`, and the actual `SOURCE_BUNDLE` plus
`SOURCE_BUNDLE_SHA256`. The source bundle must be the Git
archive for that commit, its execution files must match the checkout, and the
image must carry matching source labels. The verifier-owned trust profile
contains the independently sourced policy, artifact, schedule, runtime,
signer, verifier, and scorer-authorization inputs; the target never derives
them from the pack. Qualification verifies the exact evidence path, normalized
request, signed receipt, and pack identity before writing the private summary.

Set `INVARLOCK_GGUF_RESOURCE_ROOT`,
`INVARLOCK_GGUF_BACKEND_EXECUTABLE`, and
`INVARLOCK_GGUF_BACKEND_SOURCE` exactly as shown in the main runtime-provider
guide. `REPORT` is optional. Receipt, report, and summary destinations must be
fresh, distinct, outside the evidence path, and beneath existing non-symlinked
directories.

Use a model and prompt whose expected output is fixed before qualification.
Changing the GGUF file, `llama.cpp` executable, source archive, image digest,
CPU identity, prompt, or decoding settings invalidates the qualification and
requires a new transaction. Qualification therefore binds the caller's actual
model, prompt, artifact, backend, and CPU combination. The signed canary catches
image-level failure before fan-out; it does not prove that a different GGUF
artifact loads, fits memory, or completes successfully.

## Public Python surface

The provider package intentionally keeps its public surface small:

- `LlamaCppProvider` implements runtime-provider ABI `1` and derives the
  complete `ModelRuntimeSpec` through `inspect_runtime_spec`;
- `LlamaCppRuntimeBindings` names the authenticated GGUF, executable, and
  source-archive inputs used for inspection and execution;
- `LlamaCppBackendInspection` is the typed result of backend inspection; and
- `LlamaCppExecutionError` reports a native execution failure that could not
  produce a scoring observation.

The remaining session helpers are implementation details of the first-party
add-in. Applications should enter through the provider ABI rather than invoke
`LlamaCppSession` directly.
