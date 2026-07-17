from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import invarlock.evaluation_side_worker as worker
from invarlock.core.runtime_provider import (
    RuntimeProvider,
    build_runtime_behavioral_schedule_from_material,
    canonical_runtime_behavioral_schedule_json,
)
from invarlock.evaluation_side_worker import RuntimeSideWorkerError, execute_job
from invarlock.evidence_pack_contract import canonical_json_bytes

_DIGEST = "sha256:" + "a" * 64


def _job(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    resources = tmp_path / "resources"
    resources.mkdir()
    (resources / "model").mkdir()
    schedule = tmp_path / "schedule.json"
    schedule.write_bytes(
        canonical_runtime_behavioral_schedule_json(
            build_runtime_behavioral_schedule_from_material(
                dataset_identity={
                    "provider": "local",
                    "dataset_name": None,
                    "config_name": None,
                    "revision": None,
                    "split": "qualification",
                },
                records=[
                    {
                        "record_id": "record/1",
                        "input_text": "Return A",
                        "expected_output": "A",
                    }
                ],
            )
        )
    )
    payload: dict[str, object] = {
        "format_version": "invarlock/runtime-side-job-v1",
        "role": "baseline",
        "provider": "hf_transformers",
        "model_id": "local/model",
        "settings": {"batch_size": 1},
        "metric": "exact_match",
        "policy_digest": _DIGEST,
        "resource_root": str(resources),
        "primary_artifact": "model",
        "support_resources": {},
        "device_kind": "cpu",
        "image_digest": _DIGEST,
        "schedule": str(schedule),
        "output": str(tmp_path / "output"),
    }
    job = tmp_path / "job.json"
    job.write_bytes(canonical_json_bytes(payload))
    return job, payload


def test_worker_executes_one_closed_job_without_a_signing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, _payload = _job(tmp_path)
    observed: dict[str, Any] = {}

    class Provider:
        name = "hf_transformers"

        def prepare_execution(self, spec: object, resources: object) -> object:
            observed["spec"] = spec
            observed["resources"] = resources
            return {"context": True}

    class Registry:
        def get_runtime_provider(self, name: str) -> RuntimeProvider:
            assert name == "hf_transformers"
            return cast(RuntimeProvider, Provider())

    def run(**kwargs: object) -> object:
        observed.update(kwargs)
        output = cast(Path, kwargs["output_directory"])
        output.mkdir()
        return SimpleNamespace(directory=output)

    monkeypatch.setattr(worker, "CoreRegistry", Registry)
    monkeypatch.setattr(worker, "run_evidence_side", run)

    assert execute_job(job) == tmp_path / "output"
    assert observed["role"] == "baseline"
    assert observed["metric"] == "exact_match"
    resources = observed["resources"]
    assert resources.container_image_digest == _DIGEST
    assert resources.primary_artifact == "model"


def test_worker_input_preflight_rejects_before_model_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, payload = _job(tmp_path)
    schedule = build_runtime_behavioral_schedule_from_material(
        dataset_identity={
            "provider": "local",
            "dataset_name": None,
            "config_name": None,
            "revision": None,
            "split": "qualification",
        },
        records=[
            {
                "record_id": "record/1",
                "input_text": "Return A",
                "expected_output": "A",
            }
        ],
    )
    Path(cast(str, payload["schedule"])).write_bytes(
        canonical_runtime_behavioral_schedule_json(schedule)
    )

    class Provider:
        name = "hf_transformers"

        def validate_evaluation_inputs(
            self, _spec: object, _resources: object, _schedule: object
        ) -> None:
            raise ValueError("authenticated content is unavailable")

        def prepare_execution(self, _spec: object, _resources: object) -> object:
            pytest.fail("model preparation must not run after input-preflight failure")

    class Registry:
        def get_runtime_provider(self, _name: str) -> RuntimeProvider:
            return cast(RuntimeProvider, Provider())

    monkeypatch.setattr(worker, "CoreRegistry", Registry)
    monkeypatch.setattr(
        worker,
        "run_evidence_side",
        lambda **_kwargs: pytest.fail("runtime scoring must not start"),
    )

    with pytest.raises(RuntimeSideWorkerError, match="authenticated content"):
        execute_job(job)
    assert not (tmp_path / "output").exists()


def test_worker_scores_the_exact_schedule_snapshot_validated_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, payload = _job(tmp_path)

    def schedule(record_id: str):  # noqa: ANN202
        return build_runtime_behavioral_schedule_from_material(
            dataset_identity={
                "provider": "local",
                "dataset_name": None,
                "config_name": None,
                "revision": None,
                "split": "qualification",
            },
            records=[
                {
                    "record_id": record_id,
                    "input_text": "Return A",
                    "expected_output": "A",
                }
            ],
        )

    original = schedule("record/original")
    replacement = schedule("record/replaced")
    schedule_path = Path(cast(str, payload["schedule"]))
    schedule_path.write_bytes(canonical_runtime_behavioral_schedule_json(original))

    class Provider:
        name = "hf_transformers"

        def validate_evaluation_inputs(
            self, _spec: object, _resources: object, observed: object
        ) -> None:
            assert observed.records[0].record_id == "record/original"  # type: ignore[attr-defined]
            schedule_path.write_bytes(
                canonical_runtime_behavioral_schedule_json(replacement)
            )

        def prepare_execution(self, _spec: object, _resources: object) -> object:
            return {"context": True}

    class Registry:
        def get_runtime_provider(self, _name: str) -> RuntimeProvider:
            return cast(RuntimeProvider, Provider())

    def run(**kwargs: object) -> object:
        validated = kwargs["_validated_schedule"]
        assert validated.records[0].record_id == "record/original"  # type: ignore[attr-defined]
        assert schedule_path.read_bytes() == canonical_runtime_behavioral_schedule_json(
            replacement
        )
        output = cast(Path, kwargs["output_directory"])
        output.mkdir()
        return SimpleNamespace(directory=output)

    monkeypatch.setattr(worker, "CoreRegistry", Registry)
    monkeypatch.setattr(worker, "run_evidence_side", run)

    assert execute_job(job) == tmp_path / "output"


def test_worker_rejects_noncanonical_unknown_or_mixed_jobs(tmp_path: Path) -> None:
    job, payload = _job(tmp_path)
    job.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(RuntimeSideWorkerError, match="canonical JSON"):
        execute_job(job)

    payload["foreign"] = True
    job.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(RuntimeSideWorkerError, match="unexpected fields"):
        execute_job(job)


def test_worker_process_refuses_signing_key_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    job, _payload = _job(tmp_path)
    monkeypatch.setenv("INVARLOCK_EVIDENCE_SIGNING_KEY", "/secret/key.pem")
    assert worker.main([str(job)]) == 2
    assert "refuses evidence-signing key" in capsys.readouterr().err


def test_worker_refuses_preexisting_output(tmp_path: Path) -> None:
    job, _payload = _job(tmp_path)
    (tmp_path / "output").mkdir()
    with pytest.raises(RuntimeSideWorkerError, match="new directory"):
        execute_job(job)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format_version", "invarlock/runtime-side-job-v0", "unsupported"),
        ("role", "candidate", "role is invalid"),
        ("provider", 7, "non-string bindings"),
        ("policy_digest", "sha256:bad", "policy digest is invalid"),
        ("image_digest", "sha256:bad", "image digest is invalid"),
        ("device_kind", "metal", "device must be cpu or cuda"),
        ("settings", [], "settings must be an object"),
        ("support_resources", {"tokenizer": 7}, "values must be strings"),
    ],
)
def test_worker_rejects_malformed_or_unauthorized_bindings(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    case_root = tmp_path / field
    case_root.mkdir()
    job, payload = _job(case_root)
    payload[field] = value
    job.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(RuntimeSideWorkerError, match=message):
        execute_job(job)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("resource_root", "relative", "absolute path"),
        ("resource_root", "noncanonical", "not canonical"),
        ("schedule", "missing", "unavailable"),
        ("resource_root", "schedule", "must be a directory"),
        ("schedule", "resources", "must be a regular file"),
        ("output", "file-parent", "new directory"),
    ],
)
def test_worker_rejects_unsafe_or_wrong_kind_paths(
    tmp_path: Path, field: str, replacement: str, message: str
) -> None:
    case_root = tmp_path / replacement
    case_root.mkdir()
    job, payload = _job(case_root)
    resources = case_root / "resources"
    schedule = case_root / "schedule.json"
    replacements = {
        "relative": "relative/path",
        "noncanonical": f"{resources}/../resources",
        "missing": str(case_root / "missing.json"),
        "schedule": str(schedule),
        "resources": str(resources),
    }
    if replacement == "file-parent":
        parent = case_root / "not-a-directory"
        parent.write_text("occupied", encoding="utf-8")
        payload[field] = str(parent / "output")
    else:
        payload[field] = replacements[replacement]
    job.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(RuntimeSideWorkerError, match=message):
        execute_job(job)


def test_worker_main_closes_argument_and_execution_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert worker.main([]) == 2
    assert "exactly one job path" in capsys.readouterr().err

    monkeypatch.setattr(
        worker,
        "execute_job",
        lambda _path: (_ for _ in ()).throw(RuntimeSideWorkerError("closed failure")),
    )
    assert worker.main([str(tmp_path / "job.json")]) == 2
    assert "closed failure" in capsys.readouterr().err

    monkeypatch.setattr(worker, "execute_job", lambda _path: tmp_path / "output")
    assert worker.main([str(tmp_path / "job.json")]) == 0
