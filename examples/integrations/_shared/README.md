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
| `create_source_archive.sh` | Source-only archive helper for sharing reproducible example inputs. |
| `validate_source_matrix_artifacts.py` | Replays strict verification with independent acceptance inputs and validates generated artifacts against `source_matrix.json`. |

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

When sharing an example, start from a GitHub source archive or a cloned branch,
not from a local checkout with generated artifacts. To validate the same
source-only shape locally, run:

```bash
examples/integrations/_shared/create_source_archive.sh \
  --output /tmp/invarlock-current-source.tgz
```

The helper uses `git archive` when the checkout is clean. If the checkout has
pending changes, it archives tracked, modified, staged, and untracked
non-ignored files from the worktree. The worktree path sets
`COPYFILE_DISABLE=1` and uses `--no-xattrs` when the local `tar` supports it,
so macOS extended-attribute headers are not written into the tarball.

Use `--committed` when sharing an archive. Use `--include-worktree` only when
deliberately including local changes in an archive.

## Shared Compare Wrapper

The shared script expects an already loadable baseline and subject:

```bash
examples/integrations/_shared/run_invarlock_compare.sh \
  --baseline sshleifer/tiny-gpt2 \
  --subject ./models/tiny-gpt2-subject \
  --report-out ./reports/integration-example \
  --baseline-report /path/to/raw-baseline-report.json \
  --policy-pack /path/to/acceptance-policy-pack.json \
  --expected-runtime-image-digest "$TRUSTED_RUNTIME_IMAGE_DIGEST" \
  --allow-network
```

Supply the raw baseline report and policy pack independently of the subject run.
Set `TRUSTED_RUNTIME_IMAGE_DIGEST` from build/release policy or another channel
independent of the generated report and runtime manifest. The wrapper also
accepts these inputs through its wrapper-only
`INVARLOCK_ACCEPTANCE_BASELINE_REPORT`, `INVARLOCK_ACCEPTANCE_POLICY_PACK`, and
`INVARLOCK_EXPECTED_RUNTIME_IMAGE_DIGEST` environment variables. Strict mode
fails closed unless all three inputs are present and valid.

Use the same run-lane shape in example README files when the lanes are
meaningful. The CLI keeps simple shortcuts, while generated artifacts record a
canonical lane label:

| Artifact lane label | Purpose | Required flags |
| --- | --- | --- |
| `cuda-container-strict` | Primary evidence path: runtime manifest, container provenance, and strict verifier assurance on a CUDA host. | `--lane cuda` |
| `cuda-host-off` | Secondary comparison path: local CUDA dependency bring-up without strict container evidence. | `--lane host --device cuda` |
| `cpu-host-off` | Secondary comparison path: local non-CUDA dependency setup and quick compatibility runs when supported by the target backend. | `--lane host --device cpu` |

Use `--assurance off` only for local backend investigation or when documenting a
concrete blocker. Its output is diagnostic and cannot serve as strict or release
evidence.

Host lanes run preflight before model materialization and evaluation. Missing
CUDA availability, compiler, or Python header requirements should fail early
with a prerequisite message instead of failing inside the backend runtime.

The wrapper default mode is strict/container-backed. For a host-side
exploratory run, pass the host lane and device explicitly:

```bash
examples/integrations/_shared/run_invarlock_compare.sh \
  --baseline sshleifer/tiny-gpt2 \
  --subject ./models/tiny-gpt2-subject \
  --report-out ./reports/integration-host-example \
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
  runtime provenance: expected_image_digest_matched, declared=container, verified=true
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
