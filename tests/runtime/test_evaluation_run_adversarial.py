from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import invarlock.evaluation_run as evaluation_run
from invarlock.core.evaluation_request import (
    ArtifactRequest,
    ComparisonRequest,
    ComparisonSideRequest,
    EvaluationRequest,
    ExecutionRequest,
    OutputRequest,
    RuntimeRequest,
)
from invarlock.core.registry import CoreRegistry
from invarlock.core.runtime_provider import RuntimeProvider
from invarlock.evaluation_runtime import RuntimeResourceResolver

_BASELINE_DIGEST = "sha256:" + "a" * 64
_SUBJECT_DIGEST = "sha256:" + "b" * 64


def _request(root: Path) -> EvaluationRequest:
    def side(name: str) -> ComparisonSideRequest:
        checkpoint = root / name
        checkpoint.mkdir()
        return ComparisonSideRequest(
            artifact=ArtifactRequest(
                path=checkpoint,
                model_id=f"org/{name}",
                locator=f"artifact:{name}",
            ),
            runtime=RuntimeRequest(
                provider="hf_transformers",
                settings={"batch_size": 1},
            ),
        )

    dataset = root / "dataset.jsonl"
    policy = root / "policy.yaml"
    dataset.write_text("{}\n", encoding="utf-8")
    policy.write_text("tier: release\n", encoding="utf-8")
    return EvaluationRequest(
        format_version="invarlock/evaluation-request-v1",
        root=root,
        comparison=ComparisonRequest(
            baseline=side("baseline"),
            subject=side("subject"),
            dataset=dataset,
            policy=policy,
            task="text_causal",
            metric="exact_match",
        ),
        execution=ExecutionRequest(
            mode="run",
            records=None,
            schedule=None,
            baseline=None,
            subject=None,
        ),
        output=OutputRequest(evidence=root / "evidence"),
    )


class _Provider:
    name = "hf_transformers"

    def __init__(self) -> None:
        self.prepared: list[tuple[object, object]] = []

    def prepare_execution(self, spec: object, resources: object) -> object:
        self.prepared.append((spec, resources))
        return {"prepared": True}


class _Registry:
    def __init__(self, provider: _Provider) -> None:
        self.provider = provider
        self.lookups: list[str] = []

    def get_runtime_provider(self, name: str) -> RuntimeProvider:
        self.lookups.append(name)
        return cast(RuntimeProvider, self.provider)


class _Resolver:
    def __init__(self) -> None:
        self.roles: list[str] = []

    def resolve(self, **kwargs: Any) -> object:
        self.roles.append(cast(str, kwargs["role"]))
        return {"role": kwargs["role"]}


def _fake_runner(
    digest_for_role: Callable[[str], str | None],
    calls: list[dict[str, object]],
) -> Callable[..., object]:
    def run(**kwargs: Any) -> object:
        call = dict(kwargs)
        call["schedule_bytes"] = cast(Path, kwargs["schedule_path"]).read_bytes()
        calls.append(call)
        role = cast(str, kwargs["role"])
        output = cast(Path, kwargs["output_directory"])
        output.mkdir()
        report = output / "report.json"
        manifest = output / "runtime.manifest.json"
        config = output / "run.yaml"
        report.write_bytes(f"{role}-report".encode())
        manifest.write_bytes(f"{role}-manifest".encode())
        config.write_bytes(f"{role}-config".encode())
        evidence = SimpleNamespace(
            artifact_identity_bytes=f"{role}-identity".encode(),
            receipt_bytes=f"{role}-receipt".encode(),
            scoring_observation_bytes=f"{role}-observation".encode(),
            receipt=SimpleNamespace(outer_image_digest=digest_for_role(role)),
        )
        return SimpleNamespace(
            report_path=report,
            manifest_path=manifest,
            config_path=config,
            evidence=evidence,
        )

    return run


def test_execute_runtime_comparison_binds_both_sides_and_copies_exact_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    registry = _Registry(provider)
    resolver = _Resolver()
    calls: list[dict[str, object]] = []
    digests = {"baseline": _BASELINE_DIGEST, "subject": _SUBJECT_DIGEST}
    monkeypatch.setattr(
        evaluation_run,
        "run_evidence_side",
        _fake_runner(digests.get, calls),
    )

    result = evaluation_run.execute_runtime_comparison(
        _request(tmp_path),
        registry=cast(CoreRegistry, registry),
        resource_resolver=cast(RuntimeResourceResolver, resolver),
        schedule_bytes=b'{"format":"schedule"}\n',
        policy_digest="sha256:" + "c" * 64,
    )

    assert registry.lookups == ["hf_transformers", "hf_transformers"]
    assert resolver.roles == ["baseline", "subject"]
    assert [call["role"] for call in calls] == ["baseline", "subject"]
    assert all(call["schedule_bytes"] == b'{"format":"schedule"}\n' for call in calls)
    assert result.baseline.run_report == b"baseline-report"
    assert result.baseline.runtime_manifest == b"baseline-manifest"
    assert result.baseline.provider_receipt == b"baseline-receipt"
    assert result.subject.runtime_config == b"subject-config"
    assert result.subject.artifact_identity == b"subject-identity"
    assert result.subject.scoring_observation == b"subject-observation"
    assert result.baseline_runtime_digest == _BASELINE_DIGEST
    assert result.subject_runtime_digest == _SUBJECT_DIGEST


def test_execute_runtime_comparison_rejects_receipt_without_outer_image_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        evaluation_run,
        "run_evidence_side",
        _fake_runner(
            lambda role: _BASELINE_DIGEST if role == "baseline" else None,
            calls,
        ),
    )

    with pytest.raises(ValueError, match="subject provider receipt lacks"):
        evaluation_run.execute_runtime_comparison(
            _request(tmp_path),
            registry=cast(CoreRegistry, _Registry(provider)),
            resource_resolver=cast(RuntimeResourceResolver, _Resolver()),
            schedule_bytes=b"{}\n",
            policy_digest="sha256:" + "c" * 64,
        )

    assert [call["role"] for call in calls] == ["baseline", "subject"]
