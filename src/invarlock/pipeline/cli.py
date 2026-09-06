"""Installed front door for adding release checks to an existing evaluator."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import typer
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_json import read_regular_file_bytes
from invarlock.pipeline.adapters import load_run
from invarlock.pipeline.cases import (
    canonical_case_set,
    case_set_digest,
    validate_run_case_set,
)
from invarlock.pipeline.contracts import (
    MAX_EVIDENCE_BYTES,
    PipelineError,
    digest,
    read_json,
    validate,
    write_directory,
    write_new,
)
from invarlock.pipeline.evidence import create_evidence, verify_evidence
from invarlock.pipeline.report import render_html, render_junit, render_markdown
from invarlock.pipeline.templates import example_project
from invarlock.security import enforce_default_security

app = typer.Typer(
    name="invarlock-pipeline",
    add_completion=False,
    no_args_is_help=True,
    help="Check existing evaluation results without rerunning inference. Recorded judgments remain explicit.",
)
EXIT_CODES = {"pass": 0, "regression": 1, "insufficient_evidence": 3}


@app.callback()
def root() -> None:
    enforce_default_security()


def _fail(exc: Exception) -> None:
    typer.echo(
        canonical_json_bytes(
            {"status": "integration_error", "message": str(exc), "exit_code": 2}
        ).decode(),
        nl=False,
    )
    raise typer.Exit(2) from exc


def _private(path: Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(
        read_regular_file_bytes(path, label="signing key", max_bytes=65536),
        password=None,
    )
    if not isinstance(key, Ed25519PrivateKey):
        raise PipelineError("signing key must be Ed25519")
    return key


@app.command()
def init(
    directory: Path, example: str = typer.Option("classification", "--example")
) -> None:
    """Create a complete synthetic example and editable, reusable project files."""
    try:
        baseline, candidate, policy = example_project(example)
        artifacts = {}
        for name, value in (
            ("baseline.json", baseline),
            ("candidate.json", candidate),
            ("policy.json", policy),
            (
                "pipeline.json",
                {
                    "format": "invarlock/pipeline-project-v1",
                    "baseline": {"path": "baseline.json", "adapter": "invarlock"},
                    "candidate": {"path": "candidate.json", "adapter": "invarlock"},
                    "policy": "policy.json",
                },
            ),
        ):
            artifacts[name] = canonical_json_bytes(value)
        artifacts["README.txt"] = (
            b"These are synthetic records and illustrative thresholds, not customer results.\nReplace both runs with your captured exports and review policy.json before reliance.\nUse a new output directory for each comparison. No inference or external service is required.\n"
        )
        write_directory(directory, artifacts)
        typer.echo(
            f"Created {example} example. Run: invarlock-pipeline compare {directory / 'pipeline.json'} --output {directory / 'result'}"
        )
    except (ValueError, OSError) as exc:
        _fail(exc)


@app.command(name="import")
def import_export(
    export: Path,
    output: Path = typer.Option(..., "--output"),
    adapter: str = typer.Option("jsonl", "--adapter"),
    source_version: str = typer.Option(..., "--source-version"),
    run_id: str = typer.Option(..., "--run-id"),
    artifact_digest: str = typer.Option(..., "--artifact-digest"),
    provenance: Path | None = typer.Option(None, "--provenance"),
) -> None:
    """Normalize a native export once; supply release identities from your pipeline."""
    try:
        run = load_run(
            export,
            adapter=adapter,
            source={"name": adapter, "version": source_version},
            run_id=run_id,
            artifact_digest=artifact_digest,
            score_provenance=read_json(provenance) if provenance else None,
        )
        write_new(output, canonical_json_bytes(run))
        typer.echo(f"Imported {len(run['records'])} records to {output}")
    except (ValueError, OSError) as exc:
        _fail(exc)


def _project_run(
    specification: dict[str, Any], root_path: Path, override: Path | None
) -> dict[str, Any]:
    options = {
        k: v
        for k, v in specification.items()
        if k not in {"path", "expected_run_digest"}
    }
    path = override if override is not None else root_path / specification["path"]
    run = load_run(path, **options)
    expected = specification.get("expected_run_digest")
    if expected is not None and digest(run) != expected:
        raise PipelineError(f"{path}: run digest does not match expected_run_digest")
    return run


@app.command(name="case-set")
def freeze_case_set(
    cases: Path,
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Validate explicitly planned cases and print the digest for your policy."""
    try:
        value = canonical_case_set(read_json(cases))
        expected = case_set_digest(value)
        if output is not None:
            write_new(output, canonical_json_bytes(value))
        typer.echo(
            canonical_json_bytes(
                {
                    "format": "invarlock/pipeline-case-set-digest-v1",
                    "case_count": len(value["cases"]),
                    "expected_case_set_digest": expected,
                    "output": str(output) if output is not None else None,
                }
            ).decode(),
            nl=False,
        )
    except (ValueError, OSError) as exc:
        _fail(exc)


@app.command()
def compare(
    project: Path,
    output: Path = typer.Option(..., "--output"),
    signing_key: Path | None = typer.Option(
        None, "--signing-key", envvar="INVARLOCK_PIPELINE_SIGNING_KEY"
    ),
    baseline: Path | None = typer.Option(None, "--baseline"),
    candidate: Path | None = typer.Option(None, "--candidate"),
) -> None:
    """Check all metrics/slices and write JSON, HTML, Markdown and JUnit reports."""
    try:
        config = read_json(project)
        validate(config, "project")
        base = _project_run(config["baseline"], project.parent, baseline)
        subject = _project_run(config["candidate"], project.parent, candidate)
        policy = read_json(project.parent / config["policy"])
        if isinstance(policy, dict) and "expected_case_set_digest" in policy:
            validate(policy, "policy")
            for run in (base, subject):
                validate_run_case_set(run, policy["expected_case_set_digest"])
        evidence = create_evidence(
            base, subject, policy, _private(signing_key) if signing_key else None
        )
        result = evidence["comparison"]
        artifacts = {
            "evidence.json": canonical_json_bytes(evidence),
            "comparison.json": canonical_json_bytes(result),
            "report.html": render_html(result).encode(),
            "summary.md": render_markdown(result).encode(),
            "junit.xml": render_junit(result),
        }
        write_directory(output, artifacts)
        typer.echo(
            canonical_json_bytes(
                {
                    "decision": result["decision"],
                    "bindings": result["bindings"],
                    "authentication": "signed" if signing_key else "unsigned_local",
                    "output": str(output),
                    "exit_code": EXIT_CODES[result["decision"]],
                }
            ).decode(),
            nl=False,
        )
    except (ValueError, OSError, OverflowError) as exc:
        _fail(exc)
    raise typer.Exit(EXIT_CODES[result["decision"]])


@app.command()
def keygen(directory: Path) -> None:
    """Create reusable local signing material; authorize its public key independently."""
    private_key = directory / "private.pem"
    public_key = directory / "public.pem"
    try:
        key = Ed25519PrivateKey.generate()
        private_data = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        public_data = key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        write_directory(
            directory, {"private.pem": private_data, "public.pem": public_data}
        )
        typer.echo(
            f"Created {private_key} and {public_key}. Recipient authorization must use an independent channel."
        )
    except (ValueError, OSError) as exc:
        _fail(exc)


@app.command()
def verify(
    evidence: Path,
    public_key: Path = typer.Option(..., "--public-key"),
    policy: Path = typer.Option(..., "--policy"),
    expected_baseline: str = typer.Option(..., "--expected-baseline"),
    expected_candidate: str = typer.Option(..., "--expected-candidate"),
) -> None:
    """Authenticate and replay using recipient-owned expected inputs, never pack keys."""
    try:
        key = serialization.load_pem_public_key(
            read_regular_file_bytes(public_key, label="public key", max_bytes=65536)
        )
        if not isinstance(key, Ed25519PublicKey):
            raise PipelineError("verification key must be Ed25519")
        result = verify_evidence(
            read_json(evidence, max_bytes=MAX_EVIDENCE_BYTES),
            public_key=key,
            policy=read_json(policy),
            expected_baseline=expected_baseline,
            expected_candidate=expected_candidate,
        )
        typer.echo(
            canonical_json_bytes(
                {
                    "authenticated": True,
                    "decision": result["decision"],
                    "exit_code": EXIT_CODES[result["decision"]],
                }
            ).decode(),
            nl=False,
        )
    except (ValueError, OSError, OverflowError) as exc:
        _fail(exc)
    raise typer.Exit(EXIT_CODES[result["decision"]])


@app.command(name="digest")
def file_digest(path: Path, run: bool = typer.Option(False, "--run")) -> None:
    """Hash a regular artifact or identity manifest without loading model weights."""
    import os
    import stat

    try:
        if run:
            from invarlock.pipeline.contracts import digest

            value_run = read_json(path)
            validate(value_run, "run")
            typer.echo(digest(value_run))
            return
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise PipelineError("artifact must be a regular file")
            value = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
            after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise PipelineError("artifact changed during hashing")
        typer.echo("sha256:" + value.hexdigest())
    except (ValueError, OSError) as exc:
        _fail(exc)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
