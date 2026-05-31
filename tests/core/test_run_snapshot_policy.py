from __future__ import annotations

import pytest

import invarlock.core.run_snapshot_contract as mod
from invarlock.core.run_snapshot_contract import (
    choose_snapshot_mode,
    estimate_model_bytes,
    resolve_snapshot_config,
)


class _FakeTensor:
    def __init__(self, bytes_count: int) -> None:
        self._bytes_count = bytes_count

    def element_size(self) -> int:
        return 1

    def nelement(self) -> int:
        return self._bytes_count


class _FakeModel:
    def named_parameters(self):
        return [("w", _FakeTensor(1024)), ("u", _FakeTensor(2048))]

    def named_buffers(self):
        return [("b", _FakeTensor(512))]


class _BadTensor:
    def element_size(self) -> int:
        raise TypeError("boom")

    def nelement(self) -> int:
        return 1


class _BadModel:
    def named_parameters(self):
        return [("bad", _BadTensor())]

    def named_buffers(self):
        return [("bad-buffer", _BadTensor())]


def test_resolve_snapshot_config_extracts_nested_mapping() -> None:
    out = resolve_snapshot_config(
        {"snapshot": {"mode": "chunked", "threshold_mb": 10.0}},
        to_serialisable_dict_fn=lambda obj: obj if isinstance(obj, dict) else {},
    )

    assert out == {"mode": "chunked", "threshold_mb": 10.0}


def test_resolve_snapshot_config_propagates_unexpected_serializer_failures() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        resolve_snapshot_config(
            object(),
            to_serialisable_dict_fn=lambda _obj: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
        )


def test_resolve_snapshot_config_fails_closed_on_type_errors() -> None:
    out = resolve_snapshot_config(
        object(),
        to_serialisable_dict_fn=lambda _obj: (_ for _ in ()).throw(TypeError("boom")),
    )

    assert out == {}


def test_resolve_snapshot_config_fails_closed_on_non_mapping_serialiser_output() -> (
    None
):
    assert (
        resolve_snapshot_config(
            {"snapshot": {"mode": "bytes"}},
            to_serialisable_dict_fn=lambda _obj: [],
        )
        == {}
    )


def test_resolve_snapshot_config_fails_closed_on_non_mapping_snapshot_output() -> None:
    out = resolve_snapshot_config(
        {"snapshot": {"mode": "bytes"}},
        to_serialisable_dict_fn=lambda obj: [] if obj == {"mode": "bytes"} else obj,
    )

    assert out == {}


def test_resolve_snapshot_config_fails_closed_on_nested_snapshot_serialiser_error() -> (
    None
):
    out = resolve_snapshot_config(
        {"snapshot": {"mode": "bytes"}},
        to_serialisable_dict_fn=lambda obj: (
            (_ for _ in ()).throw(TypeError("boom"))
            if obj == {"mode": "bytes"}
            else obj
        ),
    )

    assert out == {}


def test_estimate_model_bytes_sums_parameters_and_buffers() -> None:
    assert estimate_model_bytes(_FakeModel()) == 3584
    assert estimate_model_bytes(object()) == 0


def test_estimate_model_bytes_ignores_bad_tensor_metadata() -> None:
    assert estimate_model_bytes(_BadModel()) == 0


def test_estimate_model_bytes_returns_zero_on_broken_parameter_iterator() -> None:
    class _BrokenModel:
        def named_parameters(self):
            raise TypeError("boom")

    assert estimate_model_bytes(_BrokenModel()) == 0


def test_choose_snapshot_mode_honors_explicit_config_then_falls_back() -> None:
    assert (
        choose_snapshot_mode(
            snapshot_config={"mode": "bytes"},
            env_mode="chunked",
            supports_bytes=True,
            supports_chunked=True,
            estimated_model_mb=128.0,
            available_ram_mb=512.0,
            disk_free_mb=2048.0,
        )
        == "bytes"
    )

    assert (
        choose_snapshot_mode(
            snapshot_config={"mode": "bytes"},
            env_mode=None,
            supports_bytes=False,
            supports_chunked=True,
            estimated_model_mb=128.0,
            available_ram_mb=512.0,
            disk_free_mb=2048.0,
        )
        == "chunked"
    )


def test_choose_snapshot_mode_auto_prefers_chunked_under_ram_pressure() -> None:
    out = choose_snapshot_mode(
        snapshot_config={},
        env_mode="auto",
        supports_bytes=True,
        supports_chunked=True,
        estimated_model_mb=600.0,
        available_ram_mb=512.0,
        disk_free_mb=2048.0,
        env_ram_fraction="0.4",
    )

    assert out == "chunked"


def test_choose_snapshot_mode_auto_prefers_bytes_for_small_models() -> None:
    out = choose_snapshot_mode(
        snapshot_config={},
        env_mode="auto",
        supports_bytes=True,
        supports_chunked=True,
        estimated_model_mb=8.0,
        available_ram_mb=2048.0,
        disk_free_mb=2048.0,
        env_ram_fraction="0.4",
    )

    assert out == "bytes"


def test_choose_snapshot_mode_returns_reload_when_no_snapshot_support_exists() -> None:
    out = choose_snapshot_mode(
        snapshot_config={"mode": "bytes"},
        env_mode=None,
        supports_bytes=False,
        supports_chunked=False,
        estimated_model_mb=256.0,
        available_ram_mb=1024.0,
        disk_free_mb=1024.0,
    )

    assert out == "reload"


def test_choose_snapshot_mode_honors_explicit_chunked_request_and_reload_fallback() -> (
    None
):
    assert (
        choose_snapshot_mode(
            snapshot_config={"mode": "chunked"},
            env_mode=None,
            supports_bytes=True,
            supports_chunked=True,
            estimated_model_mb=64.0,
            available_ram_mb=512.0,
            disk_free_mb=512.0,
        )
        == "chunked"
    )
    assert (
        choose_snapshot_mode(
            snapshot_config={"mode": "chunked"},
            env_mode=None,
            supports_bytes=False,
            supports_chunked=False,
            estimated_model_mb=64.0,
            available_ram_mb=512.0,
            disk_free_mb=512.0,
        )
        == "reload"
    )


def test_choose_snapshot_mode_honors_env_requested_mode_when_cfg_mode_absent() -> None:
    assert (
        choose_snapshot_mode(
            snapshot_config={},
            env_mode="chunked",
            supports_bytes=True,
            supports_chunked=True,
            estimated_model_mb=64.0,
            available_ram_mb=512.0,
            disk_free_mb=512.0,
        )
        == "chunked"
    )


def test_choose_snapshot_mode_uses_threshold_and_margin_fallbacks_when_ram_unknown() -> (
    None
):
    out = choose_snapshot_mode(
        snapshot_config={"threshold_mb": "bad", "disk_free_margin_ratio": "bad"},
        env_mode="auto",
        supports_bytes=False,
        supports_chunked=True,
        estimated_model_mb=900.0,
        available_ram_mb=0.0,
        disk_free_mb=1200.0,
        env_threshold_mb="800",
    )

    assert out == "chunked"


def test_choose_snapshot_mode_falls_back_on_invalid_env_threshold_values() -> None:
    out = choose_snapshot_mode(
        snapshot_config={},
        env_mode="auto",
        supports_bytes=True,
        supports_chunked=False,
        estimated_model_mb=32.0,
        available_ram_mb=0.0,
        disk_free_mb=0.0,
        env_ram_fraction="bad",
        env_threshold_mb="bad",
    )

    assert out == "bytes"


def test_choose_snapshot_mode_prefers_bytes_when_disk_is_tight_for_chunked() -> None:
    out = choose_snapshot_mode(
        snapshot_config={"disk_free_margin_ratio": 2.0},
        env_mode="auto",
        supports_bytes=True,
        supports_chunked=True,
        estimated_model_mb=600.0,
        available_ram_mb=512.0,
        disk_free_mb=700.0,
        env_ram_fraction="0.4",
    )

    assert out == "bytes"


def test_choose_snapshot_mode_uses_cfg_ram_fraction_and_bytes_fallback_chunked_path() -> (
    None
):
    out = choose_snapshot_mode(
        snapshot_config={"ram_fraction": "0.9"},
        env_mode="auto",
        supports_bytes=True,
        supports_chunked=True,
        estimated_model_mb=850.0,
        available_ram_mb=1000.0,
        disk_free_mb=2000.0,
    )

    assert out == "chunked"


def test_choose_snapshot_mode_covers_requested_mode_and_auto_fallback_edges() -> None:
    assert (
        choose_snapshot_mode(
            snapshot_config={"mode": "other"},
            env_mode="auto",
            supports_bytes=True,
            supports_chunked=False,
            estimated_model_mb=8.0,
            available_ram_mb=128.0,
            disk_free_mb=128.0,
        )
        == "bytes"
    )
    assert (
        choose_snapshot_mode(
            snapshot_config={},
            env_mode="auto",
            supports_bytes=False,
            supports_chunked=True,
            estimated_model_mb=8.0,
            available_ram_mb=128.0,
            disk_free_mb=128.0,
        )
        == "chunked"
    )
    assert (
        choose_snapshot_mode(
            snapshot_config={},
            env_mode="auto",
            supports_bytes=False,
            supports_chunked=False,
            estimated_model_mb=8.0,
            available_ram_mb=128.0,
            disk_free_mb=128.0,
        )
        == "reload"
    )


def test_choose_snapshot_mode_falls_back_to_bytes_for_unknown_requested_mode() -> None:
    assert (
        mod._requested_snapshot_mode(
            "bytes-ish", supports_bytes=True, supports_chunked=False
        )
        == "bytes"
    )
