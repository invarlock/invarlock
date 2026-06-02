# Shared Integration Assets

These files define the common shape for target-specific integration examples.
They are intentionally independent of optional third-party backends.

## Files

| File | Role |
| --- | --- |
| `evidence-scope.md` | Wording and mode boundaries for public integration examples. |
| `expected-artifacts.md` | Artifact checklist for runnable examples. |
| `preflight.sh` | Host-lane device checks, host default resolution, and artifact-lane labels. |
| `run_invarlock_compare.sh` | Shared compare/verify/render wrapper for HF-loadable baseline and subject paths. |
| `create_source_archive.sh` | Source-only archive helper for remote outreach-style validation. |

## Preflight Checklist

Run these from the repository root or from an environment where `invarlock` is
installed:

```bash
invarlock doctor
invarlock advanced plugins list --json
```

For optional target backends, verify the Python import before promising a
runnable example:

```bash
python -c "import importlib.util; print(importlib.util.find_spec('gptqmodel') is not None)"
```

Use the relevant module name for each target, such as `torchao`, `peft`,
`optimum`, `llmcompressor`, `lm_eval`, `vllm`, or `bitsandbytes`.

## Source-Only Archive

Outreach reviewers usually start from a GitHub source archive or a cloned
branch, not from a local checkout with generated artifacts. For local pre-PR
validation, build the same source-only shape with:

```bash
examples/integrations/_shared/create_source_archive.sh \
  --output /tmp/invarlock-current-source.tgz
```

The helper uses `git archive` when the checkout is clean. If the checkout has
pending changes, it archives tracked, modified, staged, and untracked
non-ignored files from the worktree. The worktree path sets
`COPYFILE_DISABLE=1` and uses `--no-xattrs` when the local `tar` supports it,
so macOS extended-attribute headers are not written into the tarball.

Use `--committed` for external outreach archives. Use `--include-worktree` only
for local pre-PR validation or when intentionally sharing uncommitted changes.

## Shared Compare Wrapper

The shared script expects an already loadable baseline and subject:

```bash
examples/integrations/_shared/run_invarlock_compare.sh \
  --baseline sshleifer/tiny-gpt2 \
  --subject ./models/tiny-gpt2-subject \
  --report-out ./reports/integration-smoke \
  --allow-network
```

Use the same run-lane shape in target README files when the lanes are
meaningful. The CLI keeps simple shortcuts, while generated artifacts record a
canonical lane label:

| Artifact lane label | Purpose | Required flags |
| --- | --- | --- |
| `cuda-container-strict` | Primary review path: runtime manifest, container provenance, and strict verifier assurance on a CUDA host. | `--lane cuda` |
| `cuda-host-off` | Secondary comparison path: local CUDA dependency bring-up without strict container evidence. | `--lane host --device cuda` |
| `cpu-host-off` | Secondary comparison path: local non-CUDA dependency bring-up and quick smoke runs when supported by the target backend. | `--lane host --device cpu` |

Use `--assurance off` only for local backend debugging or when documenting a
concrete blocker.

Host lanes run preflight before model materialization and evaluation. Missing
CUDA availability, compiler, or Python header requirements should fail early
with a prerequisite message instead of failing inside the backend runtime.

The wrapper default mode is strict/container-backed. For a host-side
exploratory run, pass the host lane and device explicitly:

```bash
examples/integrations/_shared/run_invarlock_compare.sh \
  --baseline sshleifer/tiny-gpt2 \
  --subject ./models/tiny-gpt2-subject \
  --report-out ./reports/integration-host-smoke \
  --lane host \
  --device cpu \
  --allow-network
```

## Expected Run Output

The shared wrapper prints a concise completion block after the verbose
`invarlock evaluate`, `verify`, and optional HTML-render steps complete:

```text
InvarLock integration run complete
  status: success
  lane: cuda-container-strict
  report: <report-out>/evaluation.report.json
  verify: <report-out>/verify.json
  verify status: ok
  runtime provenance: verified, declared=container, verified=true
  html: <report-out>/evaluation.html
  lane artifact: <report-out>/lane_artifact.json
  summary: <report-out>/run_summary.txt
```

If a run fails, the wrapper prints the lane label, report directory, command
log, and summary path. Check the prerequisite message first. If the failure
happened during evaluation or verification, replay the concrete command from
`run_command.txt`.

Target runners also print compact progress markers before backend preparation,
metadata collection, comparison, verification, and rendering. The detailed
`invarlock evaluate` output remains visible between those markers, including
phase banners, model loading, guard execution, primary metrics, and pass/fail
status.
