from __future__ import annotations

import json
from pathlib import Path

import pytest

import invarlock.policy_pack as policy_pack_mod
from invarlock.policy_pack import (
    build_policy_pack,
    compute_policy_pack_digest,
    exercise_policy_pack_bytes,
    load_policy_pack,
    verify_policy_pack,
    write_policy_pack,
)


def test_policy_pack_digest_verification_round_trip(tmp_path: Path) -> None:
    pack = build_policy_pack(
        tier="balanced",
        resolved_policy={"metrics": {"pm_ratio": {"ratio_limit_base": 1.1}}},
        overrides=[{"path": "metrics.pm_ratio.ratio_limit_base", "value": 1.1}],
        compatibility={"support_tiers": ["published_basis"]},
        approval={"owner": "oss"},
    )
    out = tmp_path / "policy-pack.json"
    write_policy_pack(out, pack)

    loaded = load_policy_pack(out)
    assert verify_policy_pack(loaded) == []


def test_policy_pack_verification_rejects_digest_mismatch() -> None:
    pack = build_policy_pack(
        tier="balanced",
        resolved_policy={"metrics": {"pm_ratio": {"ratio_limit_base": 1.1}}},
        overrides=[{"path": "metrics.pm_ratio.ratio_limit_base", "value": 1.1}],
        compatibility={"support_tiers": ["published_basis"]},
    )
    pack["policy_digest"] = "0000000000000000"
    errors = verify_policy_pack(pack)
    assert any("policy digest mismatch" in error for error in errors)


def test_policy_pack_verification_rejects_format_mismatch() -> None:
    pack = build_policy_pack(
        tier="balanced",
        resolved_policy={"metrics": {"pm_ratio": {"ratio_limit_base": 1.1}}},
    )
    pack["format"] = "wrong-format"

    errors = verify_policy_pack(pack)
    assert any("policy pack format must be policy-pack-v1" in error for error in errors)


def test_policy_pack_load_yaml_and_normalize_override_shapes(tmp_path: Path) -> None:
    yaml_pack = tmp_path / "policy-pack.yaml"
    yaml_pack.write_text(
        "\n".join(
            [
                "format: policy-pack-v1",
                "tier: balanced",
                "resolved_policy:",
                "  metrics:",
                "    pm_ratio:",
                "      ratio_limit_base: 1.1",
                "overrides:",
                "  ratio_limit_base: 1.1",
                "policy_digest: placeholder",
                "compatibility:",
                "  support_tiers:",
                "    - published_basis",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_policy_pack(yaml_pack)
    assert loaded["tier"] == "balanced"

    assert policy_pack_mod._normalize_overrides(None) == []
    assert policy_pack_mod._normalize_overrides({"path": 1}) == [
        {"path": "path", "value": 1}
    ]
    assert policy_pack_mod._normalize_overrides([{"path": "a", "value": 1}, "raw"]) == [
        {"path": "a", "value": 1},
        {"value": "raw"},
    ]
    assert policy_pack_mod._normalize_overrides("raw") == [{"value": "raw"}]


def test_policy_pack_structured_text_loader_supports_json_and_yaml() -> None:
    assert policy_pack_mod._load_structured_text(
        '{"tier":"balanced"}', suffix=".json"
    ) == {"tier": "balanced"}
    assert policy_pack_mod._load_structured_text(
        "tier: balanced\n", suffix=".yaml"
    ) == {"tier": "balanced"}


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"{",
        b"tier: balanced\nresolved_policy:\n  metrics: []\n",
        bytes(range(256)),
        json.dumps(
            {
                "format": "policy-pack-v1",
                "tier": "balanced",
                "resolved_policy": {"metrics": {"pm_ratio": {"ratio_limit_base": 1.1}}},
                "overrides": [],
                "policy_digest": "placeholder",
                "compatibility": {"support_tiers": ["published_basis"]},
            }
        ).encode("utf-8"),
    ],
)
def test_policy_pack_fuzz_target_handles_arbitrary_bytes(payload: bytes) -> None:
    exercise_policy_pack_bytes(payload)


def test_policy_pack_structured_text_loader_normalizes_yaml_overflow(
    monkeypatch,
) -> None:
    def _boom(_: str):
        raise OverflowError("int too large")

    monkeypatch.setattr(policy_pack_mod.yaml, "safe_load", _boom)
    with pytest.raises(
        ValueError, match="policy pack could not be decoded as JSON/YAML"
    ):
        policy_pack_mod._load_structured_text("tier: balanced\n", suffix=".yaml")


def test_policy_pack_build_defaults_and_metadata() -> None:
    pack = build_policy_pack(
        tier="aggressive",
        resolved_policy={"metrics": {"pm_ratio": {"ratio_limit_base": 1.2}}},
        metadata={"author": "tests"},
    )

    assert pack["format"] == "policy-pack-v1"
    assert pack["compatibility"] == {"support_tiers": ["published_basis"]}
    assert pack["metadata"] == {"author": "tests"}
    assert "approval" not in pack
    assert pack["policy_digest"] == compute_policy_pack_digest(
        resolved_policy=pack["resolved_policy"],
        overrides=pack["overrides"],
    )


def test_policy_pack_load_rejects_non_mapping_payload(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")

    with pytest.raises(
        ValueError, match="policy pack must decode to a JSON/YAML object"
    ):
        load_policy_pack(bad)


def test_policy_pack_verify_rejects_non_mapping_and_non_list_overrides() -> None:
    assert verify_policy_pack(["not", "a", "mapping"]) == [
        "policy pack must be a mapping"
    ]

    pack = build_policy_pack(
        tier="balanced",
        resolved_policy={"metrics": {"pm_ratio": {"ratio_limit_base": 1.1}}},
    )
    pack["overrides"] = {"path": "metrics.pm_ratio.ratio_limit_base", "value": 1.1}

    errors = verify_policy_pack(pack)
    assert "overrides must be an ordered list" in errors


def test_policy_pack_verify_rejects_bad_resolved_policy_type() -> None:
    pack = build_policy_pack(
        tier="balanced",
        resolved_policy={"metrics": {"pm_ratio": {"ratio_limit_base": 1.1}}},
    )
    pack["resolved_policy"] = []

    errors = verify_policy_pack(pack)
    assert "resolved_policy must be an object" in errors


def test_policy_pack_verify_captures_schema_validation_error(monkeypatch) -> None:
    pack = build_policy_pack(
        tier="balanced",
        resolved_policy={"metrics": {"pm_ratio": {"ratio_limit_base": 1.1}}},
    )

    class FakeJsonSchema:
        @staticmethod
        def validate(*, instance, schema) -> None:
            raise RuntimeError("schema boom")

    monkeypatch.setattr(policy_pack_mod, "jsonschema", FakeJsonSchema)
    errors = verify_policy_pack(pack)
    assert any("schema validation failed: schema boom" in error for error in errors)


def test_policy_pack_verify_skips_schema_when_unavailable(monkeypatch) -> None:
    pack = build_policy_pack(
        tier="balanced",
        resolved_policy={"metrics": {"pm_ratio": {"ratio_limit_base": 1.1}}},
    )

    monkeypatch.setattr(policy_pack_mod, "jsonschema", None)
    monkeypatch.setattr(policy_pack_mod, "load_policy_pack_schema", lambda: {})
    assert verify_policy_pack(pack) == []
