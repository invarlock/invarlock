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


def test_policy_pack_digest_binds_dataset_compatibility() -> None:
    pack = build_policy_pack(
        tier="balanced",
        resolved_policy={"metrics": {"pm_ratio": {"ratio_limit_base": 1.1}}},
        compatibility={
            "support_tiers": ["published_basis"],
            "dataset_identity": {
                "provider": "hf_text",
                "dataset_name": "Salesforce/wikitext",
                "config_name": "wikitext-2-raw-v1",
                "revision": "a" * 40,
                "split": "validation",
            },
        },
    )
    pack["compatibility"]["dataset_identity"]["provider"] = "local_jsonl"

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


def test_policy_pack_load_yaml_and_reject_malformed_override_shapes(
    tmp_path: Path,
) -> None:
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
                "  - path: metrics.pm_ratio.ratio_limit_base",
                "    value: 1.1",
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
    assert policy_pack_mod._normalize_overrides([{"path": "a", "value": 1}]) == [
        {"path": "a", "value": 1}
    ]
    for malformed in ({"path": 1}, [{"path": "a"}], ["raw"], "raw"):
        with pytest.raises(ValueError, match="exact path/value"):
            policy_pack_mod._normalize_overrides(malformed)


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
    def _boom(*_args, **_kwargs):
        raise OverflowError("int too large")

    monkeypatch.setattr(policy_pack_mod.yaml, "load", _boom)
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
        tier="aggressive",
        resolved_policy=pack["resolved_policy"],
        overrides=pack["overrides"],
        metadata=pack["metadata"],
    )
    assert len(pack["policy_digest"]) == len("sha256:") + 64


def test_policy_pack_load_rejects_non_mapping_payload(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")

    with pytest.raises(
        ValueError, match="policy pack must decode to a JSON/YAML object"
    ):
        load_policy_pack(bad)


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_policy_pack_load_rejects_duplicate_object_members(
    tmp_path: Path, suffix: str
) -> None:
    path = tmp_path / f"duplicate{suffix}"
    if suffix == ".json":
        path.write_text('{"format":"first","format":"second"}', encoding="utf-8")
    else:
        path.write_text("format: first\nformat: second\n", encoding="utf-8")

    with pytest.raises(ValueError, match="could not be decoded|duplicate key"):
        load_policy_pack(path)


def test_policy_pack_load_rejects_yaml_merge_keys(tmp_path: Path) -> None:
    path = tmp_path / "merged.yaml"
    path.write_text(
        "defaults: &defaults\n  support_tiers: [published_basis]\n"
        "compatibility:\n  <<: *defaults\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="could not be decoded"):
        load_policy_pack(path)


def test_policy_pack_load_rejects_yaml_aliases(tmp_path: Path) -> None:
    path = tmp_path / "aliases.yaml"
    path.write_text(
        "resolved_policy: &policy\n  threshold: 1.0\nmetadata: *policy\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="aliases"):
        load_policy_pack(path)


def test_policy_pack_load_rejects_non_finite_yaml_number(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.yaml"
    path.write_text("resolved_policy:\n  threshold: .nan\n", encoding="utf-8")

    with pytest.raises(ValueError, match="could not be decoded"):
        load_policy_pack(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_policy_pack_rejects_non_finite_values(value: float) -> None:
    pack = build_policy_pack(tier="balanced", resolved_policy={})
    pack["resolved_policy"] = {"threshold": value}

    assert any("non-finite" in error for error in verify_policy_pack(pack))


@pytest.mark.parametrize(
    ("section", "field"),
    [
        (None, "legacy_compatibility"),
        ("compatibility", "legacy_approval"),
        ("approval", "reviewed"),
    ],
)
def test_policy_pack_rejects_unknown_authority_fields(
    section: str | None, field: str
) -> None:
    pack = build_policy_pack(
        tier="balanced", resolved_policy={}, approval={"owner": "acceptance-authority"}
    )
    target = pack if section is None else pack[section]
    target[field] = True

    assert any("unknown fields" in error for error in verify_policy_pack(pack))


def test_policy_pack_digest_binds_approval_and_metadata() -> None:
    pack = build_policy_pack(
        tier="balanced",
        resolved_policy={},
        approval={"owner": "acceptance-authority"},
        metadata={"source": "review"},
    )
    pack["approval"]["owner"] = "producer"
    pack["metadata"]["source"] = "producer"

    assert any("policy digest mismatch" in error for error in verify_policy_pack(pack))


@pytest.mark.parametrize(
    "overrides",
    [
        [{"path": "", "value": 1}],
        [{"path": "a..b", "value": 1}],
        [{"path": "a", "value": 1}, {"path": "a", "value": 2}],
        [{"path": "a", "value": 1, "approved": True}],
    ],
)
def test_policy_pack_rejects_ambiguous_overrides(
    overrides: list[dict[str, object]],
) -> None:
    if all(set(item) == {"path", "value"} for item in overrides):
        pack = build_policy_pack(tier="balanced", resolved_policy={})
        pack["overrides"] = overrides
        assert any("override" in error for error in verify_policy_pack(pack))
    else:
        with pytest.raises(ValueError, match="exact path/value"):
            build_policy_pack(tier="balanced", resolved_policy={}, overrides=overrides)


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


def test_policy_pack_verify_remains_strict_when_schema_library_is_unavailable(
    monkeypatch,
) -> None:
    pack = build_policy_pack(
        tier="balanced",
        resolved_policy={"metrics": {"pm_ratio": {"ratio_limit_base": 1.1}}},
    )

    monkeypatch.setattr(policy_pack_mod, "jsonschema", None)
    monkeypatch.setattr(policy_pack_mod, "load_policy_pack_schema", lambda: {})
    assert verify_policy_pack(pack) == []

    pack["compatibility"]["unreviewed_legacy_mode"] = True
    errors = verify_policy_pack(pack)
    assert any("compatibility contains unknown fields" in error for error in errors)


def test_policy_pack_rejects_retired_v2_format() -> None:
    pack = build_policy_pack(tier="balanced", resolved_policy={})
    pack["format"] = "policy-pack-v2"

    assert any("policy-pack-v1" in error for error in verify_policy_pack(pack))


@pytest.mark.parametrize(
    ("text", "accepted"),
    [
        ("value: true\n", True),
        ("value: false\n", False),
        ("value: 1\n", 1),
        ("value: null\n", None),
    ],
)
def test_policy_yaml_accepts_only_canonical_json_scalars(
    text: str, accepted: object
) -> None:
    assert policy_pack_mod._load_structured_text(text, suffix=".yaml") == {
        "value": accepted
    }


@pytest.mark.parametrize(
    "text",
    [
        "value: YES\n",
        "value: 0x10\n",
        "value: ~\n",
        "1: value\n",
        "value:\n  <<: {}\n",
    ],
)
def test_policy_yaml_rejects_ambiguous_scalar_or_mapping_syntax(text: str) -> None:
    with pytest.raises(ValueError, match="could not be decoded"):
        policy_pack_mod._load_structured_text(text, suffix=".yaml")


def test_policy_pack_manual_shape_validator_covers_every_nested_authority() -> None:
    base = build_policy_pack(tier="balanced", resolved_policy={})
    cases: list[tuple[dict[str, object], str]] = []

    def changed(mutator) -> dict[str, object]:
        payload = json.loads(json.dumps(base))
        mutator(payload)
        return payload

    cases.extend(
        [
            (changed(lambda p: p["resolved_policy"].update({"": 1})), "object keys"),
            (
                changed(lambda p: p["resolved_policy"].update({"bad": {1, 2}})),
                "unsupported value type",
            ),
            (
                changed(lambda p: p["compatibility"].update({"support_tiers": []})),
                "non-empty",
            ),
            (
                changed(
                    lambda p: p["compatibility"].update(
                        {"support_tiers": ["published_basis", 1]}
                    )
                ),
                "non-empty strings",
            ),
            (
                changed(
                    lambda p: p["compatibility"].update(
                        {"support_tiers": ["published_basis", "published_basis"]}
                    )
                ),
                "unique",
            ),
            (
                changed(
                    lambda p: p["compatibility"].update(
                        {
                            "support_tiers": [
                                "supported_experimental",
                                "published_basis",
                            ]
                        }
                    )
                ),
                "canonical sorted order",
            ),
            (
                changed(
                    lambda p: p["compatibility"].update({"support_tiers": ["unknown"]})
                ),
                "unsupported value",
            ),
            (
                changed(lambda p: p.update({"compatibility": {}})),
                "support_tiers is required",
            ),
            (
                changed(
                    lambda p: p["compatibility"].update(
                        {"adapter_families": ["z", "a"]}
                    )
                ),
                "canonical sorted order",
            ),
            (
                changed(
                    lambda p: p["compatibility"].update(
                        {"dataset_identity": {"provider": "local"}}
                    )
                ),
                "must contain exactly",
            ),
            (
                changed(
                    lambda p: p["compatibility"].update(
                        {
                            "dataset_identity": {
                                "provider": "",
                                "dataset_name": None,
                                "config_name": None,
                                "revision": None,
                                "split": "",
                            }
                        }
                    )
                ),
                "must be non-empty",
            ),
            (
                changed(
                    lambda p: p["compatibility"].update(
                        {
                            "dataset_identity": {
                                "provider": "local",
                                "dataset_name": "",
                                "config_name": None,
                                "revision": None,
                                "split": "validation",
                            }
                        }
                    )
                ),
                "must be null or non-empty",
            ),
            (changed(lambda p: p.update({"approval": {}})), "non-empty object"),
            (
                changed(lambda p: p.update({"metadata": []})),
                "metadata must be an object",
            ),
            (
                changed(
                    lambda p: p.update(
                        {"overrides": [{"path": "a", "value": 1, "legacy": True}]}
                    )
                ),
                "exactly path and value",
            ),
        ]
    )
    for payload, expected in cases:
        assert any(expected in error for error in verify_policy_pack(payload)), payload


def test_policy_pack_public_builders_reject_ambiguous_inputs() -> None:
    with pytest.raises(ValueError, match="compatibility must be an object"):
        compute_policy_pack_digest(
            tier="balanced",
            resolved_policy={},
            overrides=[],
            compatibility=[],  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="approval must be an object"):
        compute_policy_pack_digest(
            tier="balanced",
            resolved_policy={},
            overrides=[],
            approval=[],  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="metadata must be an object"):
        compute_policy_pack_digest(
            tier="balanced",
            resolved_policy={},
            overrides=[],
            metadata=[],  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="invalid policy pack digest input"):
        compute_policy_pack_digest(tier="unknown", resolved_policy={}, overrides=[])

    for field, value in (
        ("compatibility", []),
        ("approval", []),
        ("metadata", []),
    ):
        with pytest.raises(ValueError, match=f"{field} must be an object"):
            build_policy_pack(
                tier="balanced",
                resolved_policy={},
                **{field: value},  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError, match="invalid policy pack"):
        build_policy_pack(tier="unknown", resolved_policy={})

    digest = compute_policy_pack_digest(
        tier="balanced",
        resolved_policy={},
        overrides=[],
        approval={"owner": "acceptance-authority"},
        metadata={"source": "review"},
    )
    assert digest.startswith("sha256:")


def test_policy_pack_snapshot_helpers_reject_unsafe_or_non_json_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid_utf8 = tmp_path / "invalid.json"
    invalid_utf8.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="could not be decoded"):
        load_policy_pack(invalid_utf8)

    monkeypatch.setattr(
        policy_pack_mod,
        "_load_structured_file_snapshot",
        lambda _path: (b"{}", {"bad": {1, 2}}),
    )
    with pytest.raises(ValueError, match="unsupported value type"):
        policy_pack_mod.read_policy_pack_snapshot(tmp_path / "unused.json")

    monkeypatch.setattr(
        policy_pack_mod,
        "_load_structured_file",
        lambda _path: {"bad": {1, 2}},
    )
    with pytest.raises(ValueError, match="unsupported value type"):
        policy_pack_mod.load_policy_input(tmp_path / "unused.json")
