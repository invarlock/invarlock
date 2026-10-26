"""Project workflow values into owned native judge evidence transactions."""

from __future__ import annotations

import base64
import copy
import importlib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock.judge_measurement_types import (
    JudgeAnalysisPolicyDocument,
    JudgeMeasurementPlan,
    JudgeMeasurements,
)
from invarlock.native_backend import transact as _dispatch


class AnalysisProjection(Protocol):
    def to_dict(self) -> dict[str, Any]: ...


class Publication(Protocol):
    @property
    def path(self) -> Path: ...
    @property
    def analysis_result(self) -> AnalysisProjection: ...


class Publisher(Protocol):
    def __call__(
        self,
        output: Path,
        *,
        plan: JudgeMeasurementPlan,
        measurements: JudgeMeasurements,
        baseline_run: dict[str, Any],
        subject_run: dict[str, Any],
        analysis_policy: JudgeAnalysisPolicyDocument,
        signing_key: Path | Ed25519PrivateKey | None = None,
        signer_identity: str | None = None,
        native_capture: dict[str, Any] | None = None,
    ) -> Publication: ...


def transact(operation: str, config: dict[str, Any], **options: Any) -> dict[str, Any]:
    # The successful native transaction supplies this operation's validated payload.
    return cast(dict[str, Any], _dispatch(operation, config, **options)["payload"])


def prepare(policy_bytes: bytes, schedule: dict[str, Any]) -> dict[str, Any]:
    return transact(
        "prepare-native-judge",
        {"policy_json": policy_bytes.decode("utf-8"), "schedule": schedule},
    )


def finalize(
    recipe: dict[str, Any], baseline_run: dict[str, Any], subject_run: dict[str, Any]
) -> tuple[JudgeMeasurementPlan, JudgeAnalysisPolicyDocument]:
    result = transact(
        "finalize-native-judge",
        {"recipe": recipe, "baseline": baseline_run, "subject": subject_run},
    )
    return cast(JudgeMeasurementPlan, result["plan"]), cast(
        JudgeAnalysisPolicyDocument, result["policy"]
    )


def validate_capture(capture: dict[str, Any]):
    result = transact("validate-native-capture", capture)
    return result["baseline"], result["subject"]


def create_capture(
    normalized_request,
    schedule,
    policy_bytes,
    recipe,
    baseline,
    subject,
    observations=(),
):
    from invarlock.evidence_pack_contract import (
        MAX_EVIDENCE_BYTES,
        MAX_OBSERVATION_BYTES,
        MAX_OBSERVATIONS,
        RuntimeSideEvidence,
    )

    from .native_capture import NATIVE_CAPTURE_FORMAT, NATIVE_CAPTURE_MAX_BYTES

    if len(policy_bytes) > 4 * 1024 * 1024 or len(observations) > MAX_OBSERVATIONS:
        raise ValueError("native capture policy or observation allowance exceeded")
    if any(
        type(item.payload) is not bytes or len(item.payload) > MAX_OBSERVATION_BYTES
        for item in observations
    ):
        raise ValueError("native capture observation exceeds its byte allowance")
    names = sorted(field.name for field in fields(RuntimeSideEvidence))
    raw = [getattr(side, name) for side in (baseline, subject) for name in names]
    if (
        any(
            type(value) is not bytes or len(value) > MAX_EVIDENCE_BYTES for value in raw
        )
        or sum(map(len, raw)) * 4 // 3 > NATIVE_CAPTURE_MAX_BYTES
    ):
        raise ValueError(
            "native capture side files exceed their bounded byte allowance"
        )
    capture = {
        "format": NATIVE_CAPTURE_FORMAT,
        "normalized_request": normalized_request,
        "schedule": schedule,
        "policy_base64": base64.b64encode(policy_bytes).decode("ascii"),
        "recipe": recipe,
        "observations": [
            {
                "id": value.observation_id,
                "scope": value.scope,
                "kind": value.kind,
                "payload_base64": base64.b64encode(value.payload).decode("ascii"),
            }
            for value in sorted(observations, key=lambda value: value.observation_id)
        ],
        **{
            role: {
                name: base64.b64encode(getattr(side, name)).decode("ascii")
                for name in names
            }
            for role, side in (("baseline", baseline), ("subject", subject))
        },
    }
    validate_capture(capture)
    return capture


class _AnalysisProjection:
    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.value)


def publish(
    output: Path,
    *,
    plan: JudgeMeasurementPlan,
    measurements: JudgeMeasurements,
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
    analysis_policy: JudgeAnalysisPolicyDocument,
    signing_key: Path | Ed25519PrivateKey | None = None,
    signer_identity: str | None = None,
    native_capture: dict[str, Any] | None = None,
) -> Publication:
    key_bytes = None
    if isinstance(signing_key, Ed25519PrivateKey):
        from cryptography.hazmat.primitives import serialization

        key_bytes = signing_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    result = transact(
        "publish-judge",
        {
            "destination": output,
            "plan": plan,
            "measurements": measurements,
            "baseline_run": baseline_run,
            "subject_run": subject_run,
            "analysis_policy": analysis_policy,
            "signer_identity": signer_identity,
            **(
                {"signing_key_path": signing_key}
                if isinstance(signing_key, Path)
                else {}
            ),
            **(
                {"native_capture": native_capture} if native_capture is not None else {}
            ),
        },
        signing_key_bytes=key_bytes,
    )
    return SimpleNamespace(
        path=Path(result["path"]),
        envelope=result["envelope"],
        analysis_result=_AnalysisProjection(result["analysis_result"]),
    )


# The selected workspace capability is scoped to this invocation. It is not a
# process-global backend switch and is never propagated to unrelated threads.

_WORKSPACE: ContextVar[tuple[Path, Any] | None] = ContextVar(
    "native_judge_workspace", default=None
)


def active_workspace():
    return _WORKSPACE.get()


def _workspace_path(path: Path):
    active = active_workspace()
    if active is None or Path(path).absolute().parent != active[0]:
        raise ValueError("workspace file is outside the active native capability")
    return active[1], Path(path).name


@contextmanager
def locked_workspace(path: Path):
    invarlock_native = importlib.import_module("invarlock_native")

    root = Path(path).absolute()
    workspace = invarlock_native.Workspace(str(root))
    token = _WORKSPACE.set((root, workspace))
    try:
        workspace.verify()
        yield workspace.verify
        workspace.verify()
    finally:
        _WORKSPACE.reset(token)
        workspace.close()


def retain_identity(path: Path, value: dict[str, Any]):
    from .contracts import canonical_payload

    workspace, name = _workspace_path(path)
    workspace.retain_identity(name, canonical_payload(value))


def read_file(path: Path, limit: int):
    workspace, name = _workspace_path(path)
    return workspace.read(name, limit)


def write_file_no_replace(path: Path, payload: bytes) -> None:
    workspace, name = _workspace_path(path)
    workspace.write(name, payload)
