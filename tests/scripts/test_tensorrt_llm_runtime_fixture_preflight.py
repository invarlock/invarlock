from __future__ import annotations

from pathlib import Path

import pytest

from tests.scripts._tensorrt_llm_fixture_support import fixture

_REAL_VALIDATE_HARDWARE = fixture._validate_hardware


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("all", "device=1"),
        ("device=0,1", "device=2"),
        ("device=0", "device=0"),
        ("0", "1"),
        ("device=0\n--privileged", "device=1"),
    ],
)
def test_gpu_selectors_fail_closed(first: str, second: str) -> None:
    with pytest.raises(fixture.TensorRTLLMFixtureError):
        fixture._validate_selectors(first, second)


def test_gpu_selectors_accept_indices_and_uuids() -> None:
    assert fixture._validate_selectors("device=0", "device=1") == (
        "device=0",
        "device=1",
    )
    assert fixture._validate_selectors(
        "device=GPU-01234567-89ab-cdef-0123-456789abcdef",
        "device=GPU-fedcba98-7654-3210-fedc-ba9876543210",
    )


def test_exact_base_hardware_probe_uses_closed_argv() -> None:
    commands: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...], **_kwargs: object):
        commands.append(command)
        return (
            0,
            b"GPU-01234567-89ab-cdef-0123-456789abcdef, 9.0\n",
            b"",
        )

    assert fixture._boundary.probe_base_hardware(
        engine="docker", selector="device=0", run_captured=run
    ) == ("GPU-01234567-89ab-cdef-0123-456789abcdef", "9.0")
    assert commands == [
        (
            "docker",
            "run",
            "--rm",
            "--gpus",
            "device=0",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--entrypoint",
            "nvidia-smi",
            fixture._boundary.BASE_IMAGE,
            "--query-gpu=uuid,compute_cap",
            "--format=csv,noheader,nounits",
        )
    ]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ((2, b"", b"failed"), "preflight failed"),
        ((0, b"", b"warning"), "preflight failed"),
        (
            (
                0,
                b"GPU-01234567-89ab-cdef-0123-456789abcdef, 9.0\n"
                b"GPU-fedcba98-7654-3210-fedc-ba9876543210, 9.0\n",
                b"",
            ),
            "exactly one GPU",
        ),
        ((0, b"not-a-uuid, 9.0\n", b""), "preflight is invalid"),
        (
            (0, b"GPU-01234567-89ab-cdef-0123-456789abcdef, bad\n", b""),
            "preflight is invalid",
        ),
        ((0, b"\xff", b""), "preflight is invalid"),
    ],
)
def test_exact_base_hardware_probe_rejects_invalid_results(
    response: tuple[int, bytes, bytes], message: str
) -> None:
    with pytest.raises(fixture.TensorRTLLMFixtureError, match=message):
        fixture._boundary.probe_base_hardware(
            engine="docker",
            selector="device=0",
            run_captured=lambda *_args, **_kwargs: response,
        )
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="explicit GPU"):
        fixture._boundary.probe_base_hardware(
            engine="docker",
            selector="all",
            run_captured=lambda *_args, **_kwargs: response,
        )


def test_hardware_resolution_rejects_aliases_and_wrong_compute_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    same_uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
    results = iter(((same_uuid, "9.0"), (same_uuid, "9.0")))
    monkeypatch.setattr(
        fixture._boundary,
        "probe_base_hardware",
        lambda **_kwargs: next(results),
    )
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="physical devices"):
        _REAL_VALIDATE_HARDWARE(
            engine="docker", selectors=("device=0", f"device={same_uuid}")
        )

    results = iter(
        (
            (same_uuid, "9.0"),
            ("GPU-fedcba98-7654-3210-fedc-ba9876543210", "8.0"),
        )
    )
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="compute capability"):
        _REAL_VALIDATE_HARDWARE(engine="docker", selectors=("device=0", "device=1"))

    results = iter(
        (
            (same_uuid, "9.0"),
            ("GPU-fedcba98-7654-3210-fedc-ba9876543210", "9.0"),
        )
    )
    assert _REAL_VALIDATE_HARDWARE(
        engine="docker", selectors=("device=0", "device=1")
    ) == (
        (same_uuid, "9.0"),
        ("GPU-fedcba98-7654-3210-fedc-ba9876543210", "9.0"),
    )


def test_full_flow_preflight_validates_all_inputs_before_image_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    inventory = fixture._model_inventory_sha256(model)
    hardware_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        fixture,
        "_validate_hardware",
        lambda *, engine, selectors: (
            hardware_calls.append((engine, selectors[0]))
            or (
                ("GPU-01234567-89ab-cdef-0123-456789abcdef", "9.0"),
                ("GPU-fedcba98-7654-3210-fedc-ba9876543210", "9.0"),
            )
        ),
    )
    result = fixture.preflight_flow(
        engine="docker",
        image="candidate:qualified",
        stable_tag="invarlock-runtime:tensorrt-llm-local",
        source_date_epoch="1784073600",
        smoke_selector="device=0",
        model=model,
        output=tmp_path / "fixture",
        selectors=("device=0", "device=1"),
        expected_model_inventory_sha256=inventory,
    )
    assert result == {
        "format_version": fixture.PREFLIGHT_FORMAT,
        "gpu_count": 2,
        "model_inventory_sha256": inventory,
        "ok": True,
        "target_compute_capability": "9.0",
    }
    assert hardware_calls == [("docker", "device=0")]

    monkeypatch.setattr(
        fixture,
        "_validate_hardware",
        lambda **_kwargs: pytest.fail("hardware must not run for invalid inputs"),
    )
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="lowercase sha256"):
        fixture.preflight_flow(
            engine="docker",
            image="candidate:qualified",
            stable_tag="invarlock-runtime:tensorrt-llm-local",
            source_date_epoch="1784073600",
            smoke_selector="device=0",
            model=model,
            output=tmp_path / "fixture",
            selectors=("device=0", "device=1"),
            expected_model_inventory_sha256="not-a-digest",
        )
    existing_output = tmp_path / "existing-fixture"
    existing_output.mkdir()
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="must be new"):
        fixture.preflight_flow(
            engine="docker",
            image="candidate:qualified",
            stable_tag="invarlock-runtime:tensorrt-llm-local",
            source_date_epoch="1784073600",
            smoke_selector="device=0",
            model=model,
            output=existing_output,
            selectors=("device=0", "device=1"),
            expected_model_inventory_sha256=inventory,
        )
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="reviewed inventory"):
        fixture.preflight_flow(
            engine="docker",
            image="candidate:qualified",
            stable_tag="invarlock-runtime:tensorrt-llm-local",
            source_date_epoch="1784073600",
            smoke_selector="device=0",
            model=model,
            output=tmp_path / "fixture",
            selectors=("device=0", "device=1"),
            expected_model_inventory_sha256="f" * 64,
        )
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="inside the model"):
        fixture.preflight_flow(
            engine="docker",
            image="candidate:qualified",
            stable_tag="invarlock-runtime:tensorrt-llm-local",
            source_date_epoch="1784073600",
            smoke_selector="device=0",
            model=model,
            output=model / "fixture",
            selectors=("device=0", "device=1"),
            expected_model_inventory_sha256=inventory,
        )
