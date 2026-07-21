# Hugging Face vision-text

Run a complete image-and-text comparison with one command:

```console
make example-hf-vision-text
```

The command downloads the immutable Qwen2-VL 2B and 7B revisions, verifies the
previously qualified checkpoint-tree, tokenizer, and processor commitments,
builds the canonical CUDA image and the optional
`invarlock-runtime-hf-vision-text` layer from the exact Git commit, then runs
`invarlock evaluate`, `invarlock verify`, and `invarlock report`.

The tutorial uses one generated 96×96 PNG with four color quadrants. Four
records address those same bytes by a content ID, byte length, media type, and
SHA-256. The runtime resolves that authenticated ID from the prepared content
store; the schedule contains no host path or URL.

## Requirements

- Linux with Docker or Podman and the NVIDIA container runtime;
- one NVIDIA GPU with at least 24 GB of available memory;
- `uv` and Git;
- network access for the first dependency, image, and checkpoint download;
- about 35 GB of temporary disk for the two checkpoints, build layers, and
  transaction outputs; and
- a clean committed checkout so the runtime can authenticate its source.

The first run is dominated by downloads and image construction. Later runs can
reuse the local dependency, model, and image caches. Model execution is offline
after the snapshots and runtime layers have been materialized. Select a GPU or
an explicit new workspace with `EXAMPLE_ARGS`:

```console
make example-hf-vision-text \
  EXAMPLE_ARGS="--runtime-device cuda:1 --workspace /tmp/invarlock-vision"
```

The command prints the workspace, evidence pack, signed verification receipt,
and HTML report paths when it completes.

## Inspect the transaction without GPU work

Preparation-only mode creates the request, four-record schedule, content store,
policy, signing keys, and independent trust profile. It does not download the
checkpoints, build an image, initialize CUDA, or execute either model:

```console
make example-hf-vision-text \
  EXAMPLE_ARGS="--prepare-only --workspace /tmp/invarlock-vision-inputs"
```

The model directories contain only coordinate markers in this mode, not model
weights. The request shows their immutable Hugging Face revisions and the exact
commitments required by full execution, and its repository-confined paths can
still be inspected through the normal request loader.

This four-record journey demonstrates the integration and trust boundary. It
does not support a general model-quality conclusion. Public vision-text
evidence uses a separately prepared 400-record, domain-balanced schedule with
stricter precision requirements.
