# InvarLock TensorRT-LLM runtime add-in

This optional distribution connects authenticated TensorRT-LLM engine bundles
to InvarLock's public runtime-provider ABI. Vendor imports and CUDA execution
remain isolated behind the add-in's pinned runner protocol; InvarLock core keeps
portable engine identity and independent evidence verification.

## Install and check discovery

```bash
PYTHON=/path/to/venv/bin/python
"$PYTHON" -m pip install invarlock-runtime-tensorrt-llm
"$PYTHON" -m invarlock_addins.tensorrt_llm.conformance
```

The conformance command returns JSON with `"ok":true`, provider
`tensorrt_llm`, and the ABI version accepted by the installed core package. The
wheel registers that provider through the `invarlock.runtime_providers`
entry-point group.

The install above is the standalone conformance path. Maintained qualification
does not treat first-party packages already installed in that environment as
the candidate. `CANDIDATE_WHEEL_MANIFEST` must authenticate the exact core and
TensorRT-LLM wheels built from the qualified source commit. The driver extracts
those wheels into a private candidate site and loads them before any
first-party code visible to `PYTHON`. `PYTHON` supplies only their third-party
dependencies; vendor and CUDA dependencies remain in the runtime image.
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
  --wheel dist/addins/invarlock_runtime_tensorrt_llm-*.whl \
  --output "$PWD/qualification/tensorrt-llm-candidate-wheels.json"
export CANDIDATE_WHEEL_MANIFEST="$PWD/qualification/tensorrt-llm-candidate-wheels.json"
```

The helper records absolute wheel paths and SHA-256 digests. The qualification
driver additionally requires unique, version-aligned maintained distributions
whose contents match the authenticated source archive. Recreate the manifest
after any wheel changes, and pass the same manifest to canary, preflight, and
evidence qualification.

## Build the runtime image

Build from the repository root so the Dockerfile can include the core and
add-in distributions from the same checkout:

```bash
SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"
SOURCE_COMMIT="$(git rev-parse HEAD)"
SOURCE_BUNDLE=/absolute/path/to/invarlock-source.tar
SOURCE_BUNDLE_SHA256=sha256:PINNED_SOURCE_BUNDLE_DIGEST

make -C addins/tensorrt_llm build \
  IMAGE=invarlock-tensorrt-llm:candidate \
  SOURCE_COMMIT="$SOURCE_COMMIT" \
  SOURCE_BUNDLE="$SOURCE_BUNDLE" \
  SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256" \
  SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
  BUILD_STATEMENT=/absolute/path/to/tensorrt-llm-build.json
```

The maintained add-in targets keep image construction and GPU qualification
beside the optional package:

```bash
make -C addins/tensorrt_llm build \
  IMAGE=invarlock-tensorrt-llm:candidate \
  SOURCE_COMMIT=0123456789abcdef0123456789abcdef01234567 \
  SOURCE_BUNDLE=/absolute/path/to/invarlock-source.tar \
  SOURCE_BUNDLE_SHA256=sha256:...

IMAGE="$(docker image inspect --format '{{.Id}}' invarlock-tensorrt-llm:candidate)"
DIGEST="$IMAGE"

make -C addins/tensorrt_llm smoke IMAGE="$IMAGE"
make -C addins/tensorrt_llm canary \
  IMAGE="$IMAGE" \
  IMAGE_DIGEST="$DIGEST" \
  INPUT_ROOT=/absolute/path/to/qualified-inputs \
  ENGINE_BUNDLE=engine \
  TOKENIZER_CONTRACT=tokenizer-contract.json \
  EXPECTED_ENGINE_TREE_SHA256=EXPECTED_BARE_SHA256 \
  EXPECTED_TOKENIZER_SHA256=EXPECTED_BARE_SHA256 \
  EXPECTED_OUTPUT_SHA256=EXPECTED_BARE_SHA256 \
  CANARY_TMPFS_GIB=8
```

Before Docker is invoked, the host preflight authenticates the complete closed
engine layout (`config.json` plus the declared non-empty rank engines), compares
its derived tree digest, hashes and validates the closed tokenizer contract,
and checks the remaining expected digests and tmpfs bound. It also requires a
canonical immutable image reference and an absolute input root with no comma,
control character, or symlinked path component. The canonical root emitted by
that check is passed immediately to Docker by the same recipe, so
deterministically invalid inputs do not allocate a GPU.

Using the config ID as both `IMAGE` and `DIGEST` is the explicit local-only
mode. It addresses the immutable image in the current container-engine store;
it is not a portable registry manifest identity.

For cross-host qualification, push the image and select the digest entry for
that exact repository rather than selecting a positional `RepoDigests` value:

```bash
REGISTRY=registry.example
REPOSITORY="$REGISTRY/invarlock-tensorrt-llm"
TAG="$REPOSITORY:candidate"
docker tag invarlock-tensorrt-llm:candidate "$TAG"
docker push "$TAG"
docker pull "$TAG"
IMAGE="$(
  docker image inspect "$TAG" |
    python -c 'import json,sys; repository=sys.argv[1]; entries=json.load(sys.stdin)[0].get("RepoDigests") or []; matches=[entry for entry in entries if entry.rpartition("@")[0] == repository]; len(matches) == 1 or sys.exit("expected exactly one digest for repository"); print(matches[0])' "$REPOSITORY"
)"
DIGEST="${IMAGE##*@}"
```

Use those registry `IMAGE` and `DIGEST` values for smoke, canary, and evidence
qualification on another host.

The Makefile keeps `IMAGE` as the build, smoke, and standalone-canary handle.
Its signed `qualify-*` targets instead use `QUALIFICATION_IMAGE`, which
defaults to `IMAGE_DIGEST`. Set it explicitly to an exact local `sha256:...`
config ID or to the portable `repository@sha256:...` reference selected above.

Bootstrap a signed end-to-end transaction once for the exact image with
`make -C addins/tensorrt_llm qualify-canary`. Retain its evidence, strictly
verified receipt, and verifier-owned trust profile. Supply those paths to each
later `qualify-preflight` and `qualify-evidence` call as `CANARY_EVIDENCE`,
`CANARY_RECEIPT`, and `CANARY_TRUST_PROFILE`. A different image digest requires
a new signed canary.
Set `QUALIFICATION_DEVICE` to an explicit GPU such as `cuda:0`, then set
`QUALIFICATION_CPUS`, `QUALIFICATION_MEMORY_MIB`, and `QUALIFICATION_USER` for
the bounded workers. Keep all four values unchanged through canary, preflight,
and evidence.

With the canary request, trust profile, source bundle, and output parent already
prepared, the bootstrap command is:

```bash
INVARLOCK_TENSORRT_LLM_RESOURCE_ROOT="$PWD/tensorrt-runtime" \
INVARLOCK_TENSORRT_LLM_TOKENIZER_CONTRACT=tokenizer-contract.json \
make -C addins/tensorrt_llm qualify-canary \
  REQUEST="$PWD/canary/request.yaml" \
  SIGNING_KEY="$PWD/private/evidence.key" \
  QUALIFICATION_IMAGE="$IMAGE" IMAGE_DIGEST="$DIGEST" \
  EVIDENCE="$PWD/canary/evidence" \
  TRUST_PROFILE="$PWD/canary/trust-inputs.json" \
  RECEIPT="$PWD/canary/verification-receipt.json" \
  SUMMARY="$PWD/canary/qualification-summary.json" \
  CANDIDATE_WHEEL_MANIFEST="$CANDIDATE_WHEEL_MANIFEST" \
  QUALIFICATION_DEVICE=cuda:0 \
  SOURCE_COMMIT="$SOURCE_COMMIT" SOURCE_BUNDLE="$SOURCE_BUNDLE" \
  SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256"
```

Run `make -C addins/tensorrt_llm qualify-preflight` for each target request
before allocating a full engine run. It reverifies the signed canary and checks
the target configuration without starting a model worker.
`make -C addins/tensorrt_llm qualify-evidence` then runs the public
evaluate-to-verify transaction. Supply the request, evidence key, image
reference and digest, evidence destination, verifier-owned `TRUST_PROFILE`,
receipt and summary destinations, `CANDIDATE_WHEEL_MANIFEST`, the three
`CANARY_*` inputs, and the
`SOURCE_COMMIT`, `SOURCE_BUNDLE`, and `SOURCE_BUNDLE_SHA256`; the trust profile
binds the independently sourced verification anchors. The Git archive,
execution tree, image source labels, normalized request, evidence pack, and
independently validated signed receipt must all agree before qualification
completes.

Set `INVARLOCK_TENSORRT_LLM_RESOURCE_ROOT` and
`INVARLOCK_TENSORRT_LLM_TOKENIZER_CONTRACT` exactly as shown in the main
runtime-provider guide. `REPORT` is optional. Receipt, report, and summary
destinations must be fresh, distinct, outside the evidence path, and beneath
existing non-symlinked directories.

The Dockerfile pins its NVIDIA base manifest. Qualify the finished candidate by
an explicit local config ID or an exact repository manifest on the intended GPU.

## Derive request settings

Build the request-side runtime spec from the exact engine bundle, tokenizer
contract, and pinned runner instead of transcribing engine metadata. The
inspection probes the vendor runner and CUDA runtime, so this Python must run
inside the digest-pinned, network-disabled image built from
[`runtime/Dockerfile`](runtime/Dockerfile), with the target GPU available:

```python
from pathlib import Path
import json

from invarlock_addins.tensorrt_llm.provider import TensorRTLLMProvider
from invarlock_addins.tensorrt_llm.session import TensorRTLLMRuntimeBindings

bindings = TensorRTLLMRuntimeBindings(
    engine_bundle_path=Path("/inputs/engine"),
    tokenizer_contract_path=Path("/inputs/tokenizer-contract.json"),
    runner_executable_path=Path(
        "/opt/invarlock/bin/tensorrt-llm-runner"
    ),
)
spec = TensorRTLLMProvider().inspect_runtime_spec(
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
IMAGE=registry.example/invarlock-tensorrt-llm@sha256:PINNED_IMAGE_DIGEST
DIGEST=sha256:PINNED_IMAGE_DIGEST

docker run --rm --network none --gpus all \
  --read-only \
  --cap-drop=ALL \
  --security-opt no-new-privileges \
  --pids-limit 1024 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=8g \
  --env INVARLOCK_CONTAINER_EXECUTION=1 \
  --env INVARLOCK_RUNTIME_IMAGE="$IMAGE" \
  --env INVARLOCK_RUNTIME_IMAGE_DIGEST="$DIGEST" \
  --mount type=bind,src="$PWD/inputs",dst=/inputs,readonly \
  --entrypoint python "$IMAGE" /inputs/inspect.py
```

Use `spec.model_id` and `spec.settings` for a request side whose
`runtime.provider` is `tensorrt_llm`. Evaluation requires the same bundle and
tokenizer contract as root-confined resources plus the authenticated outer
container on a matching CUDA compute capability. The official runner remains
image-owned and cannot be replaced by a caller-projected file. Vendor imports
remain inside the runner boundary; they are not dependencies of the core wheel.

## Qualify a candidate image and engine

The package includes a real provider-path canary. Run it inside the candidate
image, on the target GPU, after recording the engine-tree, tokenizer, and
expected-output digests through an independent trust path:

```bash
python -m invarlock_addins.tensorrt_llm.canary \
  --engine-bundle /inputs/engine \
  --tokenizer-contract /inputs/tokenizer-contract.json \
  --runner /opt/invarlock/bin/tensorrt-llm-runner \
  --expected-engine-tree-sha256 EXPECTED_BARE_SHA256 \
  --expected-tokenizer-sha256 EXPECTED_BARE_SHA256 \
  --expected-output-sha256 EXPECTED_BARE_SHA256
```

Set `INVARLOCK_RUNTIME_IMAGE`, `INVARLOCK_RUNTIME_IMAGE_DIGEST`, and
`INVARLOCK_CONTAINER_EXECUTION=1` on the container exactly as in the inspection
example. Use a bounded temporary filesystem large enough for the engine snapshot;
the maintained target accepts `CANARY_TMPFS_GIB` from 4 through 64. The canary
authenticates the official runner, checks the target CUDA
facts, opens two fresh provider sessions, requires byte-identical output and
evidence records, and returns one compact JSON object with `"ok":true`. It is
a focused backend smoke, not the signed canary prerequisite. Use
`qualify-canary` to produce the evidence, receipt, and trust-profile binding
required before maintained readiness or evidence qualification. That signed
canary prevents image-level fan-out, but another engine can still fail to load,
fit memory, match its compute capability, or complete successfully.

## Runner protocol

`invarlock-tensorrt-llm-runner` is a bounded process protocol used by the
provider. It has exactly two modes:

| Invocation | Input | Output | Exit status |
| --- | --- | --- | --- |
| `--invarlock-runtime-info-v1` | No standard input | One `invarlock/tensorrt-llm-runner-info-v1` JSON line | `0`, or `70` on a closed runtime failure |
| `--invarlock-score-v1` | One canonical JSON request on standard input, at most 1 MiB | One `invarlock/tensorrt-llm-runner-response-v1` JSON line | `0`, or `70` on a closed runtime failure |
| Any other argument shape | None | None | `64` |

The score mode authenticates regular engine and tokenizer files, enforces
closed JSON and resource limits, checks deterministic scalar settings and the
container/network boundary, and rejects unbounded or malformed output. This is
an internal provider transport, not a replacement for the public
`evaluate -> verify -> report` workflow.

## Public Python surface

- `TensorRTLLMProvider` implements runtime-provider ABI `1` and derives a
  complete `ModelRuntimeSpec` through `inspect_runtime_spec`.
- `TensorRTLLMRuntimeBindings` names the engine, tokenizer contract, and runner
  inputs used for inspection and execution.
- `qualify_candidate` and `TensorRTLLMCanaryError` implement candidate-image
  qualification through the real provider path.
- `TensorRTLLMRunnerError` describes a closed runner-protocol failure.

Session and execution modules are implementation details. Applications should
enter through the provider ABI or the documented canary rather than construct a
session directly.
