# Native Linux optimization comparison

`launch.py` prepares and runs the existing four-candidate protocol on Linux. It
compares the installed Python reference, Rust standalone from commit `09e23090`,
the candidate Rust binding, and candidate Rust standalone. Both native versions
are compiled on the selected Linux architecture. The old macOS executable is
never copied or used.

This launcher has not yet completed a Linux build or qualification. The cached
`python:3.12-slim` image alone is insufficient: it lacks Rust and the required
offline build/dependency caches. No image pull, container, build, or measurement
is started by creating these files or running `--help`.

## Prepare the inputs on the current host

Commit the intended native changes before selecting `--after`. Export uses Git
archives of committed `native/` sources and excludes working-tree changes. The
experiment harness is copied and hashed separately from the current checkout;
its original nine files must match the retained metadata.

```sh
python3 experiments/rust/linux/launch.py export \
  --repository /path/to/invarlock \
  --after COMMITTED_CANDIDATE_REVISION \
  --metadata /path/to/retained-artifacts/metadata.json \
  --fixtures /path/to/retained-artifacts/fixtures-with-reports.json \
  --output /path/to/invarlock-linux-inputs
```

The new output directory contains native source archives, the fixed pure-Python
wheel and dependency lock, the allowlisted harness, matrix, launcher, and only
fixture files enumerated by the frozen manifest. It does not copy the repository's
Git database, host home directory, Cargo configuration/credentials, or unrelated
evidence. Fixture PEMs are synthetic experiment keys. Retain this bundle locally;
do not publish its private-key files.

All fixture file bytes, including signed manifests, remain unchanged. On Linux,
only the copied fixture manifest's paths are rebound to the staged fixture root.
Original manifest bytes, file hashes and provenance remain in the input bundle.
Unexpected absolute paths in request configurations fail preparation.

## Linux prerequisites and storage

Provision these separately before invoking `prepare`:

- Native CPython 3.12 with `venv` and pip; Rust/Cargo 1.94.0 with the matching
  native Linux target; Maturin 1.12.0; a working C toolchain and GNU `stat`.
- Offline Cargo source caches covering both retained lockfiles. Do not copy a
  whole host Cargo home. Use a prepared Linux cache or independently curated
  registry/vendor sources; retain any configuration that changes source lookup.
- A directory containing Linux wheels for every requirement in the frozen
  `release-install-py312.txt`. Preparation installs with `--require-hashes` and
  `--no-index`; macOS-only wheels cannot satisfy native Linux requirements.
- A new work directory on ext2/ext3/ext4 storage with adequate capacity for two
  release builds, caches, installed environments and retained attempts. This
  wrapper rejects overlay, `VirtioFS` host sharing, `tmpfs` and unknown filesystem
  types for the work directory. It makes no storage or elapsed-time guarantee.

On a Linux machine, copy the input bundle and dependency wheelhouse locally, then:

```sh
python3.12 /path/to/invarlock-linux-inputs/launch.py prepare \
  --inputs /path/to/invarlock-linux-inputs \
  --work /ext-volume/invarlock-comparison \
  --wheelhouse /path/to/linux-wheelhouse \
  --host-description 'actual CPU, physical RAM and host operating system' \
  --vm-description 'bare metal, or actual hypervisor and guest allocation' \
  --storage-description 'actual disk, filesystem, and any virtual disk backing'

python3.12 /ext-volume/invarlock-comparison/inputs/launch.py run \
  --work /ext-volume/invarlock-comparison --name primary
```

Preparation builds both locked native source archives, creates a new installed
Python environment, and freezes the new Linux executable/wheel identities. Cargo
and wheel installation are offline. Failed preparation leaves its logs and work
directory intact; diagnose it before creating another generation.

The primary run includes **32 qualifications and 128 timings**, with four balanced
rounds, seed 4127, a 300-second process timeout and a 7,200-second
dispatch-admission budget.
Keep the machine otherwise idle during the run. Builds, package installation and
profiling must finish first. Before each round, the runner requires 1,200 seconds
of remaining budget for four process timeouts. This is an admission reserve, not
a hard wall-clock cap: validation, capture and cleanup can extend elapsed time.
It does not guarantee completion on a slower Linux host.

For qualification only, add `--qualification-only` and choose a separate name.
That creates 32 attempts and does not complete the comparison. A later primary
run performs its own 32 qualifications; the two runs remain separate evidence.
There is no automatic retry, qualification reuse or continuation. Retain every
failed/stopped run and do not pool its timings into a replacement run.

## Docker Desktop on this ARM64 Mac

Use a locally prepared **Linux ARM64** image with the prerequisites above. A
Python slim image can be its starting point, but preparing that image and any
network downloads are separate work. Set `IMAGE` to its immutable local image ID
and record its provenance. These commands are an operator recipe, not operations
that have been performed:

```sh
IMAGE=sha256:REPLACE_WITH_PREPARED_LOCAL_IMAGE_ID
BUNDLE=/path/to/invarlock-linux-inputs
LINUX_WHEELS=/absolute/path/to/linux-arm64-wheelhouse

docker volume create invarlock-linux-evidence
docker create --pull=never --platform linux/arm64 --network none \
  --name invarlock-linux --cpus 4 --memory 12g \
  --mount type=volume,source=invarlock-linux-evidence,target=/work \
  "$IMAGE" sleep infinity
docker cp "$BUNDLE" invarlock-linux:/work/input
docker cp "$LINUX_WHEELS" invarlock-linux:/work/linux-wheels
docker start invarlock-linux
docker inspect invarlock-linux > /path/to/invarlock-linux-container.json
docker image inspect "$IMAGE" > /path/to/invarlock-linux-image.json
docker exec invarlock-linux uname -m

docker exec invarlock-linux python3.12 /work/input/launch.py prepare \
  --inputs /work/input --work /work/comparison \
  --wheelhouse /work/linux-wheels \
  --host-description 'Apple M2 Max; fill in current macOS and physical RAM' \
  --vm-description 'Docker Desktop Linux ARM64 VM; fill in VM CPU/RAM; container 4 CPUs and 12 GiB' \
  --storage-description 'named volume on guest ext filesystem; record Docker virtual disk physical location and host filesystem'
docker exec invarlock-linux python3.12 /work/comparison/inputs/launch.py run \
  --work /work/comparison --name primary
```

Confirm `uname -m` is `aarch64` and the recorded Rust host matches it. Adjust the
example resource allocation to what the configured VM can actually provide and
record that allocation before dispatch. The launcher records Linux architecture,
tool versions, CPU/memory information, cgroup allocation and mount information
before measurement. Save the Docker inspection files beside exported evidence.

Inputs, wheels, executable artifacts and output files are copied to the named
volume before timing. No TB4 host bind mount participates in the measured fixture
or output I/O. A named volume still uses the Linux VM's virtual disk; record
whether that disk itself lives on the internal disk or external TB4/APFS storage.
The Python base interpreter and system libraries remain in the container's Linux
image layers. These results describe that ARM64 VM configuration, not bare-metal
Linux or production x86 servers. Cold-cache behavior, thermal stability, live
providers/GPUs and customer installation experience remain unqualified.

## Retain results and clean up

After execution stops, export the complete directory, including unsuccessful
attempts, build/install logs, input/source archives, wheels and environment facts:

```sh
docker cp invarlock-linux:/work/comparison /absolute/new/linux-evidence-directory
docker stop invarlock-linux
docker rm invarlock-linux
```

Verify the exported files and keep their hashes before deleting the volume.
`docker volume rm invarlock-linux-evidence` deletes the remaining Linux copy;
run it only after retention is confirmed. Do not delete unrelated images,
containers, volumes or caches. Exported absolute paths retain their original
Linux identities; inspecting the files on macOS does not rerun the benchmark.

The original 128 macOS timings and optimized-Python control remain historical.
Report Linux results separately with their own denominator and environment.
The eight fixture journeys establish only their exercised behavior; they do not
establish large native/judge capacity or all-input equivalence.
