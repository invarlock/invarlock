# Resource-aware qualification campaigns

The campaign runner schedules independent capture and validation jobs on a
single Linux Docker host. It reserves concrete GPU UUIDs, CPU IDs, host memory,
and concurrent disk-work slots. A two-GPU job starts only when both devices are
available. An exclusive job runs alone, which is useful for controlled latency
measurements. Eight CPUs and 32 GiB of host memory remain outside the declared
capacity for the operating system and control process.

This is a repository qualification tool. It has no private-platform dependency
and does not change the installed InvarLock evaluation or evidence contracts.
A successful job means its command and declared artifact checks completed; it
is not a model qualification or policy acceptance decision.

## Prepare a campaign

Use a committed source bundle and an already reviewed immutable Docker image.
Run the controller as a dedicated non-root account with Docker access. Docker
access remains privileged host access; container isolation is defense in depth.
The controller requires Python 3.12 or newer. Its scheduling code uses only the
standard library. Inputs must be local; containers have no network access.

A manifest has format `invarlock/campaign-schedule-v1`, an ID, a `host` capacity
object, and a list of jobs. Each job declares dependencies, resource counts,
timeout, estimated duration, workload identity, immutable container image,
command arguments, read-only input mounts, and expected output files. The
validator in `campaign_scheduling.py` rejects unknown fields, dependency cycles,
impossible allocations and unsafe mount paths. File mounts may carry an expected
`sha256` value. Directory contents need an explicit validation job.

For existing frozen K2 plans, build a preflight manifest from Python:

```python
import json
from pathlib import Path
from examples.qualification.campaign_k2 import make_manifest

host = json.loads(Path("host-capacity.json").read_text())
manifest = make_manifest(
    Path("frozen-plan.json"),
    {"baseline": Path("/srv/models/baseline"),
     "candidate": Path("/srv/models/candidate")},
    host,
    Path("/srv/results/sentinel-attempt-1"),
    Path("examples/qualification/campaign_k2.py"),
)
Path("campaign.json").write_text(json.dumps(manifest, indent=2) + "\n")
```

The host object declares `gpu_ids`, `cpu_ids`, `memory_mib`, `io_slots`, an
absolute UTC `deadline_epoch`, `reserve_seconds`, and `hourly_cost_usd`.
Use actual GPU UUIDs and CPU IDs observed on the host. For an 80-CPU, two-H100
host, a starting capacity is 72 CPUs, 280 GiB host memory, two disk-work slots,
and a 600-second cleanup reserve. Workload memory and CPU limits still need
representative validation. Increasing declared capacity does not prove that
co-located jobs retain their previous throughput.

The K2 adapter creates two independent prepare → capture → validate chains.
It verifies each frozen snapshot before and after capture, preserves the
historical plan/image, and hashes the separately mounted adapter. CPU validation
can overlap other jobs. Captures reserve one or two GPUs according to the plan.
The adapter currently generates preflight campaigns; a broader portfolio
forecast is not an executable qualification campaign by itself.

## Forecast and execute

From the committed checkout:

```bash
python -m examples.qualification.campaign_execution forecast \
  --manifest campaign.json --output forecast.json
python -m examples.qualification.campaign_execution run \
  --manifest campaign.json --output /srv/results/sentinel-attempt-1
```

The controller uses a fixed host lock and refuses unrelated active GPU work or
unreconciled campaign containers. It passes explicit GPU UUIDs, CPU sets, memory
and swap limits to Docker and checks the resulting container configuration.
CPU jobs use the ordinary runtime without GPU requests. The image must already
exist locally with the exact declared Docker image ID.

State, status transitions, container inspection, logs and output hashes are
retained. Only job outputs are writable inside the container. Normal Docker
calls stop at the deadline minus the cleanup reserve; removal attempts share
the remaining cleanup budget. SIGINT and SIGTERM stop admission and attempt
owned-container cleanup. An unresponsive Docker daemon can still require
operator intervention; uncertain cleanup prevents further admission. Individual
job timeouts are checked at control boundaries. Container removal latency can
overrun an individual timeout while remaining bounded by the global cleanup
budget.

Restarting the same command verifies completed results and removes interrupted
owned containers. Interrupted jobs remain cancelled, with their descendants
blocked. To retry, create an explicit new attempt with a fresh output directory;
do not delete failed evidence or interpret partial completion as success.
The lock coordinates this runner, not arbitrary host processes.

## Interpret timing evidence

`forecast.json` describes the full manifest. `remaining.json` excludes completed
work and separately lists blocked or deferred jobs. If active work exceeds its
estimate, expected remaining time is unavailable and the timeout remainder is
shown explicitly. The scheduling heuristic prioritizes long ready jobs; its
hardware lower bound is not a claim of globally optimal scheduling.

Successful, complete, semantically ready sentinels can update durations only
for matching workload and resource identities. In K2 preflights, semantic
readiness checks valid nonempty final responses; answer quality still requires
independent policy evaluation. Empty or failed captures remain diagnostics and
do not calibrate the forecast. The workload key binds the runtime, snapshot,
protocol, adapter and co-execution mode. New tasks, context lengths, generation
budgets, batching or model revisions need new representative sentinels.

Declared estimates and observed minima/medians/maxima are planning scenarios,
not statistical confidence intervals. Scheduling independent one-GPU jobs can
reduce host time; it does not establish an inference batching speedup. Provider
billing continues until the rented host is stopped through the provider.
