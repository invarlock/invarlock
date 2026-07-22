# Runtime-security API

The public runtime-security facades expose the small set of primitives used to
observe strict execution context and to harden an embedding process. They are
defense-in-depth tools; they do not turn a Python process or container into a
trusted execution environment.

!!! info "Reference"

    - **Surface:** `invarlock.runtime_security` and `invarlock.security`
    - **Stability:** Documented names are public; kernel, container, and socket behavior remains platform-dependent
    - **Use this page when:** Embedding the engine, constructing runtime manifests, or applying process-local network and temporary-file controls

## Runtime observation facade

`invarlock.runtime_security` exports constants, immutable manifest value types,
and pure observations over the current environment.

| Export | Return or value | Meaning |
| --- | --- | --- |
| `ALLOW_NETWORK_ENV` | `INVARLOCK_ALLOW_NETWORK` | Process-local network switch name |
| `ALLOW_REMOTE_CODE_ENV` | `INVARLOCK_ALLOW_REMOTE_CODE` | Remote-code switch name |
| `ALLOW_THIRD_PARTY_PLUGINS_ENV` | `INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS` | Arbitrary provider-discovery switch name |
| `CONTAINER_EXECUTION_ENV` | `INVARLOCK_CONTAINER_EXECUTION` | Explicit container-intent switch name |
| `RUNTIME_IMAGE_ENV` | `INVARLOCK_RUNTIME_IMAGE` | Current side worker's OCI image reference input |
| `RUNTIME_IMAGE_DIGEST_ENV` | `INVARLOCK_RUNTIME_IMAGE_DIGEST` | Current side worker's pinned image digest input |
| `RUNTIME_MANIFEST_FILENAME` | `runtime.manifest.json` | Fixed sidecar name |
| `RUNTIME_MANIFEST_VERSION` | `1` | Runtime manifest implementation version |
| `network_allowed()` | `bool` | Whether the network switch is truthy |
| `remote_code_allowed()` | `bool` | Whether remote code is explicitly enabled |
| `third_party_plugins_allowed()` | `bool` | Whether arbitrary provider discovery is enabled |
| `running_inside_container()` | `bool` | Whether explicit container intent is set |
| `current_execution_mode()` | `host` or `container` | Intent-derived execution label |
| `strict_container_boundary_present()` | `bool` | Explicit intent plus kernel-visible container evidence |
| `resolve_runtime_image_digest()` | digest or `None` | Validated explicit or embedded digest |
| `resolve_runtime_image()` | image reference | Reference normalized with its declared digest |

`RuntimeManifestExecution` is the immutable execution-fact record written into
a runtime manifest. `RuntimeProviderManifestFiles` names the artifact identity,
scoring observation, and provider receipt files that the manifest binds.

### Image resolution

```python
from invarlock.runtime_security import (
    resolve_runtime_image,
    resolve_runtime_image_digest,
    strict_container_boundary_present,
)

if not strict_container_boundary_present():
    raise RuntimeError("strict execution requires an observed container")

image = resolve_runtime_image()
digest = resolve_runtime_image_digest()
if digest is None:
    raise RuntimeError("strict execution requires a runtime digest")
```

When no image input is present, `resolve_runtime_image()` returns the mutable
development fallback `ghcr.io/invarlock/invarlock-runtime:latest`. That value is
not acceptable for strict evidence. The strict transaction separately requires
a canonical digest, a digest-bearing reference, and agreement between them.
Callers must not interpret the fallback as a pinned runtime identity.

`running_inside_container()` reads explicit intent only.
`strict_container_boundary_present()` additionally checks `/.dockerenv`,
`/run/.containerenv`, or bounded cgroup markers. Neither function proves which
image ran, supplies host attestation, or validates the container engine.

The host launcher resolves common and per-side image inputs before starting
workers. Inside each worker, these two common variables identify only that
side's selected image. The evidence-signing and verifier keys remain in the
host process and are not part of the worker runtime-security environment.

## Process-security facade

`invarlock.security` exposes a socket guard and secure temporary-directory
helpers.

| Export | Behavior |
| --- | --- |
| `NetworkGuard` | Installs or restores a process-global outbound socket guard |
| `enforce_network_policy(allow)` | Applies the process-global policy under a lock |
| `enforce_default_security()` | Reads `INVARLOCK_ALLOW_NETWORK`; default is deny |
| `network_policy_allows()` | Reports the current process/context policy |
| `temporarily_allow_network()` | Context-local, nested temporary bypass |
| `secure_tempdir(prefix, base_dir)` | Creates mode `0700` storage and removes it on exit |
| `is_secure_path(path)` | Checks that a path exists with exact mode `0700` |

```python
from invarlock.security import enforce_default_security, secure_tempdir

enforce_default_security()
with secure_tempdir(prefix="invarlock-review-") as workspace:
    # Write transient, non-public review material beneath workspace.
    ...
```

The network guard replaces Python's socket constructors in the current
process. It does not constrain subprocesses, native extensions that bypass
Python sockets, another process in the same container, or the host. Apply
container or infrastructure egress controls as the authoritative boundary.

`temporarily_allow_network()` is intended for tightly scoped embedding code.
Do not use it around provider execution or evidence creation. Context-local
permission does not authorize a strict evidence path; strict execution observes
the environment switches independently and requires network disabled.

## Failure and concurrency behavior

- malformed image digests or disagreeing reference/digest inputs raise
  `RuntimeError`;
- socket attempts rejected by the guard raise `RuntimeError`;
- nested temporary network scopes restore the previous context on exit;
- repeated guard installation and restoration are idempotent;
- `secure_tempdir` attempts cleanup even when the body raises; and
- exact `0700` mode is a local filesystem check, not proof of storage
  encryption or host isolation.

## Container user and resource boundary

Repository runtime images inherit the base image's default user, which may be
root. The automatic `invarlock evaluate` launcher overrides that default with
numeric non-root `65532:65532`, applies four CPUs and 65536 MiB per worker, and
accepts validated caller-owned overrides through the CLI or environment. It
also supplies read-only input mounts, one isolated writable output mount, and
only the selected device access. Submitted request data cannot choose or relax
these controls.

For a manual container invocation, apply equivalent controls explicitly:

```bash
docker run --rm --network none --read-only \
  --user 10001:10001 \
  --cpus 4 --memory 65536m \
  --mount type=bind,src="$PWD/inputs",dst=/inputs,readonly \
  --mount type=bind,src="$PWD/output",dst=/output \
  IMAGE_BY_DIGEST ...
```

Validate the selected image and mounted paths under that identity before the
real transaction. The automatic launcher also derives a finite process
deadline from the validated per-record timeout and schedule length, caps it at
24 hours, and stops or kills the exact engine-issued container ID on expiry.
TensorRT-LLM images also inherit NVIDIA's entrypoint and need the target GPU
device; least privilege must preserve only those required capabilities.

## Related documentation

- [Environment variables](environment.md) defines the operator inputs these
  helpers observe.
- [Security practices](../security/best-practices.md) places process controls
  inside the larger deployment boundary.
- [Threat model](../security/threat-model.md) states the attacks these helpers
  do and do not address.
