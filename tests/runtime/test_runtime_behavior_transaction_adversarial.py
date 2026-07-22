from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import invarlock.runtime_behavior.transaction as transaction
from invarlock.core.runtime_provider import (
    ModelRuntimeSpec,
    RuntimeExecutionContext,
    RuntimeProvider,
)
from invarlock.runtime_behavior.transaction import RuntimeEvidenceError
from invarlock.runtime_security_helpers import RuntimeManifestExecution

_DIGEST = "sha256:" + "a" * 64


def _context(
    *,
    strict: bool = True,
    allow_network: bool = False,
    digest: str | None = _DIGEST,
) -> RuntimeExecutionContext:
    return RuntimeExecutionContext(
        strict=strict,
        allow_network=allow_network,
        container_image_digest=digest,
        device_kind="cpu",
        artifact_identity_sha256="b" * 64,
    )


def test_output_resolution_rejects_ambiguous_missing_and_existing_destinations(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeEvidenceError, match="must name a directory"):
        transaction._resolved_output(Path("."))

    with pytest.raises(RuntimeEvidenceError, match="parent must be a real directory"):
        transaction._resolved_output(tmp_path / "missing" / "evidence")

    parent_file = tmp_path / "parent-file"
    parent_file.write_bytes(b"not a directory")
    with pytest.raises(RuntimeEvidenceError, match="parent must be a real directory"):
        transaction._resolved_output(parent_file / "evidence")

    existing = tmp_path / "evidence"
    existing.mkdir()
    with pytest.raises(RuntimeEvidenceError, match="already exists"):
        transaction._resolved_output(existing)


@pytest.mark.parametrize(
    ("image_ref", "expected"),
    [
        (_DIGEST, True),
        (f"registry/runtime@{_DIGEST}", True),
        ("registry/runtime:latest", False),
        (f"@{_DIGEST}", False),
        (None, False),
    ],
)
def test_runtime_image_reference_must_bind_the_exact_digest(
    image_ref: object,
    expected: bool,
) -> None:
    assert transaction._image_ref_matches_digest(image_ref, _DIGEST) is expected


@pytest.mark.parametrize(
    ("context", "message"),
    [
        (_context(strict=False), "strict offline"),
        (_context(allow_network=True), "strict offline"),
    ],
)
def test_observed_execution_requires_strict_offline_context(
    context: RuntimeExecutionContext,
    message: str,
) -> None:
    with pytest.raises(RuntimeEvidenceError, match=message):
        transaction._observed_execution(context)


def test_observed_execution_requires_kernel_container_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(transaction, "strict_container_boundary_present", lambda: False)
    with pytest.raises(RuntimeEvidenceError, match="actual container"):
        transaction._observed_execution(_context())


def test_observed_execution_rejects_enabled_third_party_provider_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "true")

    with pytest.raises(RuntimeEvidenceError, match="third-party provider discovery"):
        transaction._observed_execution(_context())


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ("INVARLOCK_ALLOW_NETWORK", "network access"),
        ("INVARLOCK_ALLOW_REMOTE_CODE", "remote code loading"),
    ],
)
def test_observed_execution_rejects_process_runtime_opt_ins(
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
    message: str,
) -> None:
    monkeypatch.setenv(environment, "true")

    with pytest.raises(RuntimeEvidenceError, match=message):
        transaction._observed_execution(_context())


def test_observed_execution_requires_pinned_matching_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(transaction, "strict_container_boundary_present", lambda: True)
    with pytest.raises(RuntimeEvidenceError, match="pinned outer image"):
        transaction._observed_execution(_context(digest=None))

    monkeypatch.setattr(
        transaction,
        "resolve_runtime_image_digest",
        lambda: "sha256:" + "c" * 64,
    )
    monkeypatch.setattr(
        transaction,
        "resolve_runtime_image",
        lambda: f"registry/runtime@{'sha256:' + 'c' * 64}",
    )
    with pytest.raises(RuntimeEvidenceError, match="does not match"):
        transaction._observed_execution(_context())

    monkeypatch.setattr(transaction, "resolve_runtime_image_digest", lambda: _DIGEST)
    monkeypatch.setattr(
        transaction, "resolve_runtime_image", lambda: "registry/runtime:latest"
    )
    with pytest.raises(RuntimeEvidenceError, match="must embed"):
        transaction._observed_execution(_context())


def test_observed_execution_returns_closed_manifest_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "false")
    monkeypatch.setattr(transaction, "strict_container_boundary_present", lambda: True)
    monkeypatch.setattr(transaction, "resolve_runtime_image_digest", lambda: _DIGEST)
    monkeypatch.setattr(
        transaction,
        "resolve_runtime_image",
        lambda: f"registry/runtime@{_DIGEST}",
    )

    execution = transaction._observed_execution(_context())

    assert execution.image_digest == _DIGEST
    assert execution.execution_mode == "container"
    assert execution.container_execution is True
    assert execution.allow_network is False
    assert execution.allow_remote_code is False
    assert execution.allow_third_party_plugins is False


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"provider_name": "other"}, "provider is invalid"),
        ({"artifact_identity_sha256": "c" * 64}, "artifact binding is invalid"),
        ({"schedule_sha256": "d" * 64}, "schedule binding is invalid"),
        (
            {"records": (SimpleNamespace(record_id="two", input_sha256="e" * 64),)},
            "order does not match",
        ),
    ],
)
def test_observation_must_preserve_every_provider_and_schedule_binding(
    override: dict[str, object],
    message: str,
) -> None:
    schedule = SimpleNamespace(
        schedule_sha256="f" * 64,
        records=(SimpleNamespace(record_id="one", input_sha256="e" * 64),),
    )
    values: dict[str, object] = {
        "provider_name": "fixture_provider",
        "artifact_identity_sha256": "b" * 64,
        "schedule_sha256": "f" * 64,
        "records": (SimpleNamespace(record_id="one", input_sha256="e" * 64),),
    }
    values.update(override)

    with pytest.raises(RuntimeEvidenceError, match=message):
        transaction._validate_observation_bindings(
            schedule=schedule,
            provider_name="fixture_provider",
            artifact_sha256="b" * 64,
            observation=SimpleNamespace(**values),
        )


@pytest.mark.parametrize(
    ("role", "spec", "policy_digest", "error_type", "message"),
    [
        (
            "other",
            ModelRuntimeSpec("fixture", "model"),
            "sha256:" + "1" * 64,
            RuntimeEvidenceError,
            "role",
        ),
        ("baseline", object(), "sha256:" + "1" * 64, TypeError, "ModelRuntimeSpec"),
        (
            "baseline",
            ModelRuntimeSpec("fixture", "model"),
            "sha256:bad",
            RuntimeEvidenceError,
            "policy_digest",
        ),
    ],
)
def test_run_side_rejects_invalid_public_arguments_before_provider_execution(
    tmp_path: Path,
    role: str,
    spec: object,
    policy_digest: str,
    error_type: type[Exception],
    message: str,
) -> None:
    provider = cast(RuntimeProvider, SimpleNamespace(name="fixture"))
    with pytest.raises(error_type, match=message):
        transaction.run_evidence_side(
            role=cast(transaction.RuntimeSideRole, role),
            provider=provider,
            spec=cast(ModelRuntimeSpec, spec),
            context=_context(),
            schedule_path=tmp_path / "schedule.json",
            policy_digest=policy_digest,
            output_directory=tmp_path / "evidence",
        )


class _Session:
    def __init__(self, observation: object, receipt: object) -> None:
        self.observation = observation
        self.receipt = receipt
        self.closed = False

    def score(self, _batch: object) -> object:
        return self.observation

    def runtime_receipt(self) -> object:
        return self.receipt

    def close(self) -> None:
        self.closed = True


class _Provider:
    def __init__(
        self,
        *,
        name: str = "fixture",
        capability_name: str = "fixture",
        session: _Session | None = None,
        fail_open: bool = False,
    ) -> None:
        self.name = name
        self.capability = SimpleNamespace(
            provider_name=capability_name, metrics=("exact_match",)
        )
        self.session = session
        self.fail_open = fail_open

    def validate_config(self, _spec: ModelRuntimeSpec) -> None:
        return None

    def capabilities(self) -> object:
        return self.capability

    def identify_artifact(self, _spec: ModelRuntimeSpec) -> object:
        return {"model": "fixture"}

    def open(
        self, _spec: ModelRuntimeSpec, _context: RuntimeExecutionContext
    ) -> object:
        if self.fail_open:
            raise RuntimeError("backend refused to open")
        assert self.session is not None
        return self.session


def _execution_prerequisites(monkeypatch: pytest.MonkeyPatch) -> object:
    schedule = SimpleNamespace(
        schedule_sha256="f" * 64,
        records=(SimpleNamespace(record_id="one", input_sha256="e" * 64),),
        evaluation_batch=lambda _metric="exact_match": object(),
    )
    monkeypatch.setattr(
        transaction,
        "_observed_execution",
        lambda _context: RuntimeManifestExecution(
            execution_mode="container",
            container_execution=True,
            image_ref=f"registry/runtime@{_DIGEST}",
            image_digest=_DIGEST,
            allow_network=False,
            allow_remote_code=False,
            allow_third_party_plugins=False,
        ),
    )
    monkeypatch.setattr(
        transaction, "load_runtime_behavioral_schedule", lambda _path: schedule
    )
    monkeypatch.setattr(
        transaction, "artifact_identity_sha256", lambda _value: "b" * 64
    )
    return schedule


def test_run_side_rejects_provider_spec_batch_capability_and_artifact_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _execution_prerequisites(monkeypatch)
    context = _context()

    cases = (
        (
            _Provider(name="other"),
            ModelRuntimeSpec("fixture", "model", {"batch_size": 1}),
            context,
            "provider and model spec",
        ),
        (
            _Provider(),
            ModelRuntimeSpec("fixture", "model", {"batch_size": 2}),
            context,
            "batch_size=1",
        ),
        (
            _Provider(capability_name="other"),
            ModelRuntimeSpec("fixture", "model", {"batch_size": 1}),
            context,
            "capabilities identity",
        ),
        (
            _Provider(),
            ModelRuntimeSpec("fixture", "model", {"batch_size": 1}),
            RuntimeExecutionContext(
                strict=True,
                allow_network=False,
                container_image_digest=_DIGEST,
                device_kind="cpu",
                artifact_identity_sha256="c" * 64,
            ),
            "artifact identity",
        ),
    )
    for provider, spec, runtime_context, message in cases:
        with pytest.raises(RuntimeEvidenceError, match=message):
            transaction.run_evidence_side(
                role="baseline",
                provider=cast(RuntimeProvider, provider),
                spec=spec,
                context=runtime_context,
                schedule_path=tmp_path / "schedule.json",
                policy_digest="sha256:" + "1" * 64,
                output_directory=tmp_path / "evidence",
            )


def test_run_side_rejects_metric_outside_provider_capabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _execution_prerequisites(monkeypatch)

    with pytest.raises(RuntimeEvidenceError, match="does not support metric"):
        transaction.run_evidence_side(
            role="baseline",
            provider=cast(RuntimeProvider, _Provider()),
            spec=ModelRuntimeSpec("fixture", "model", {"batch_size": 1}),
            context=_context(),
            schedule_path=tmp_path / "schedule.json",
            policy_digest="sha256:" + "1" * 64,
            output_directory=tmp_path / "evidence",
            metric="normalized_nll_per_utf8_byte",
        )


def _session_and_provider(
    *,
    receipt_capabilities: object | None = None,
    allow_network: bool = False,
    receipt_seed: int = 0,
    device_kind: str = "cpu",
) -> tuple[_Session, _Provider]:
    observation = SimpleNamespace(
        provider_name="fixture",
        artifact_identity_sha256="b" * 64,
        schedule_sha256="f" * 64,
        records=(SimpleNamespace(record_id="one", input_sha256="e" * 64),),
    )
    capability = SimpleNamespace(provider_name="fixture", metrics=("exact_match",))
    receipt = SimpleNamespace(
        capabilities=receipt_capabilities or capability,
        execution_settings=SimpleNamespace(
            allow_network=allow_network,
            seed=receipt_seed,
            context_length=8,
            batch_size=1,
            max_output_tokens=1,
            timeout_seconds=30,
        ),
        device=SimpleNamespace(device_kind=device_kind),
    )
    session = _Session(observation, receipt)
    provider = _Provider(session=session)
    provider.capability = capability
    return session, provider


@pytest.mark.parametrize(
    ("provider_factory", "settings", "message"),
    [
        (
            lambda: _session_and_provider(
                receipt_capabilities=SimpleNamespace(provider_name="other")
            ),
            {"batch_size": 1},
            "receipt capabilities",
        ),
        (
            lambda: _session_and_provider(allow_network=True),
            {"batch_size": 1},
            "offline execution",
        ),
        (
            lambda: _session_and_provider(receipt_seed=0),
            {"batch_size": 1, "seed": 1},
            "setting 'seed'",
        ),
        (
            lambda: _session_and_provider(device_kind="cuda"),
            {"batch_size": 1},
            "device does not match",
        ),
    ],
)
def test_run_side_closes_session_before_rejecting_receipt_contract_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_factory: Callable[[], tuple[_Session, _Provider]],
    settings: dict[str, str | int | float | bool | None],
    message: str,
) -> None:
    _execution_prerequisites(monkeypatch)
    session, provider = provider_factory()

    with pytest.raises(RuntimeEvidenceError, match=message):
        transaction.run_evidence_side(
            role="subject",
            provider=cast(RuntimeProvider, provider),
            spec=ModelRuntimeSpec("fixture", "model", settings),
            context=_context(),
            schedule_path=tmp_path / "schedule.json",
            policy_digest="sha256:" + "1" * 64,
            output_directory=tmp_path / "evidence",
        )
    assert session.closed is True


def test_run_side_closes_prepared_context_when_provider_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _execution_prerequisites(monkeypatch)
    closed: list[bool] = []
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=_DIGEST,
        device_kind="cpu",
        artifact_identity_sha256="b" * 64,
        close_callback=lambda: closed.append(True),
    )

    with pytest.raises(RuntimeError, match="refused to open"):
        transaction.run_evidence_side(
            role="baseline",
            provider=cast(RuntimeProvider, _Provider(fail_open=True)),
            spec=ModelRuntimeSpec("fixture", "model", {"batch_size": 1}),
            context=context,
            schedule_path=tmp_path / "schedule.json",
            policy_digest="sha256:" + "1" * 64,
            output_directory=tmp_path / "evidence",
        )
    assert closed == [True]


@pytest.mark.parametrize("failure_phase", ["score", "receipt"])
def test_run_side_preserves_primary_failure_when_session_cleanup_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    _execution_prerequisites(monkeypatch)
    primary = RuntimeError(f"primary {failure_phase} failure")

    class FailingSession(_Session):
        def score(self, _batch: object) -> object:
            if failure_phase == "score":
                raise primary
            return self.observation

        def runtime_receipt(self) -> object:
            if failure_phase == "receipt":
                raise primary
            return self.receipt

        def close(self) -> None:
            self.closed = True
            raise OSError("private cleanup detail /must/not/escape")

    session = FailingSession(object(), object())
    provider = _Provider(session=session)

    with pytest.raises(
        RuntimeError, match=f"primary {failure_phase} failure"
    ) as caught:
        transaction.run_evidence_side(
            role="baseline",
            provider=cast(RuntimeProvider, provider),
            spec=ModelRuntimeSpec("fixture", "model", {"batch_size": 1}),
            context=_context(),
            schedule_path=tmp_path / "schedule.json",
            policy_digest="sha256:" + "1" * 64,
            output_directory=tmp_path / "evidence",
        )

    assert session.closed is True
    assert caught.value is primary
    assert caught.value.__notes__ == [
        transaction._RUNTIME_CLEANUP_FAILURE_NOTE  # noqa: SLF001
    ]
    assert "/must/not/escape" not in "\n".join(caught.value.__notes__)


def test_run_side_preserves_open_failure_when_context_cleanup_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _execution_prerequisites(monkeypatch)
    primary = RuntimeError("primary open failure")
    cleanup_calls: list[bool] = []

    class FailingProvider(_Provider):
        def open(
            self, _spec: ModelRuntimeSpec, _context: RuntimeExecutionContext
        ) -> object:
            raise primary

    def fail_cleanup() -> None:
        cleanup_calls.append(True)
        raise OSError("private context cleanup /must/not/escape")

    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=_DIGEST,
        device_kind="cpu",
        artifact_identity_sha256="b" * 64,
        close_callback=fail_cleanup,
    )

    with pytest.raises(RuntimeError, match="primary open failure") as caught:
        transaction.run_evidence_side(
            role="baseline",
            provider=cast(RuntimeProvider, FailingProvider()),
            spec=ModelRuntimeSpec("fixture", "model", {"batch_size": 1}),
            context=context,
            schedule_path=tmp_path / "schedule.json",
            policy_digest="sha256:" + "1" * 64,
            output_directory=tmp_path / "evidence",
        )

    assert caught.value is primary
    assert cleanup_calls == [True]
    assert caught.value.__notes__ == [
        transaction._RUNTIME_CLEANUP_FAILURE_NOTE  # noqa: SLF001
    ]


def test_run_side_raises_cleanup_failure_when_execution_succeeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _execution_prerequisites(monkeypatch)
    cleanup = OSError("cleanup-only failure")

    class CleanupFailingSession(_Session):
        def close(self) -> None:
            self.closed = True
            raise cleanup

    session = CleanupFailingSession(object(), object())

    with pytest.raises(OSError, match="cleanup-only failure") as caught:
        transaction.run_evidence_side(
            role="baseline",
            provider=cast(RuntimeProvider, _Provider(session=session)),
            spec=ModelRuntimeSpec("fixture", "model", {"batch_size": 1}),
            context=_context(),
            schedule_path=tmp_path / "schedule.json",
            policy_digest="sha256:" + "1" * 64,
            output_directory=tmp_path / "evidence",
        )

    assert caught.value is cleanup
    assert session.closed is True
