# InvarLock GGUF runtime add-in

This optional distribution connects authenticated GGUF artifacts to InvarLock's
public runtime-provider ABI through a pinned local `llama.cpp` executable. The
InvarLock core package remains responsible for portable GGUF identity and
independent evidence verification.

## Install and check discovery

```bash
python -m pip install invarlock-runtime-gguf
invarlock-gguf-conformance
```

The conformance command returns JSON with `"ok":true`, provider `llama_cpp`,
and the ABI version accepted by the installed core package. Installing the
wheel also registers `llama_cpp` in the `invarlock.runtime_providers` entry-point
group; no core source change is required.

## Build the runtime image

Build from the repository root. The `llama.cpp` build stage deliberately has
no mutable package-index default: select and pin a Debian snapshot timestamp
in `YYYYMMDDTHHMMSSZ` form, then record it with the resulting image digest.

```bash
SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"
LLAMA_CPP_APT_SNAPSHOT=PINNED_YYYYMMDDTHHMMSSZ

docker build \
  --file addins/gguf/runtime/Dockerfile \
  --build-arg SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
  --build-arg LLAMA_CPP_APT_SNAPSHOT="$LLAMA_CPP_APT_SNAPSHOT" \
  --tag invarlock-gguf:candidate \
  .
```

The maintained add-in targets keep the build, import smoke, and full
evaluate-to-verify qualification together without expanding the core Makefile:

```bash
make -C addins/gguf build \
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
5. verify the resulting pack with independently recorded artifact, schedule,
   runtime-image, and evidence-signer anchors; and
6. require both the evaluation and strict verification to finish successfully.

With those pinned request and expected-anchor files in place, the qualification
uses the same public transactions as the core package:

```bash
invarlock evaluate request.yaml --signing-key evidence-signer.pem
invarlock verify evidence/ \
  --policy acceptance.json \
  --expected-baseline-artifact sha256:EXPECTED_BASELINE_ARTIFACT_DIGEST \
  --expected-subject-artifact sha256:EXPECTED_SUBJECT_ARTIFACT_DIGEST \
  --expected-schedule sha256:EXPECTED_CANONICAL_SCHEDULE_DIGEST \
  --expected-baseline-runtime sha256:EXPECTED_BASELINE_IMAGE_DIGEST \
  --expected-subject-runtime sha256:EXPECTED_SUBJECT_IMAGE_DIGEST \
  --expected-signer sha256:EXPECTED_EVIDENCE_SIGNER_FINGERPRINT \
  --receipt verification.receipt.json \
  --verifier-signing-key verifier.pem \
  --verifier-identity VERIFIER_IDENTITY
```

The same transaction is available as `make -C addins/gguf qualify-evidence`;
the target requires the request, independent anchors, and both signing-key
paths as explicit variables and refuses to derive trust anchors from the pack.

Use a model and prompt whose expected output is fixed before qualification.
Changing the GGUF file, `llama.cpp` executable, source archive, image digest,
CPU identity, prompt, or decoding settings invalidates the qualification and
requires a new transaction. InvarLock does not ship a model fixture because a
fixture would not qualify the operator's actual artifact/backend combination.

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
