from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from invarlock.evidence_pack_policy import (
    POLICY_RELATIVE_PATH,
    load_valid_policy_pack_snapshot,
    policy_manifest_entry,
    verify_policy_material,
    write_canonical_policy_pack,
)
from invarlock.policy_pack import build_policy_pack, write_policy_pack


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_policy_pack_fixture(tmp_path: Path) -> tuple[Path, Path]:
    pack_dir = tmp_path / "pack"
    sealed_path = pack_dir / POLICY_RELATIVE_PATH
    external_path = tmp_path / "acceptance-policy.json"
    payload = build_policy_pack(
        tier="balanced",
        resolved_policy={"metrics": {"accuracy": {"delta_min_pp": -1.0}}},
    )
    write_canonical_policy_pack(sealed_path, payload)
    write_policy_pack(external_path, payload)
    report_path = pack_dir / "reports/report-001/evaluation.report.json"
    _write_json(report_path, {"assurance": {"mode": "strict"}})
    checksums_path = pack_dir / "checksums.sha256"
    checksums_path.write_text(
        "\n".join(
            [
                f"{_sha256(sealed_path)}  {POLICY_RELATIVE_PATH}",
                f"{_sha256(report_path)}  reports/report-001/evaluation.report.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        pack_dir / "manifest.json",
        {
            "format": "evidence-pack-v1",
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": _sha256(checksums_path),
            "verification": {"report_assurance": "strict"},
            "verification_policy_pack": policy_manifest_entry(sealed_path),
        },
    )
    return pack_dir, external_path


def test_missing_policy_material_fails_before_manifest_binding(tmp_path: Path) -> None:
    missing = tmp_path / "missing-policy.json"

    raw, payload, errors = load_valid_policy_pack_snapshot(
        missing, label="acceptance policy"
    )

    assert raw is None
    assert payload is None
    assert errors == [f"acceptance policy not found: {missing}"]
    with pytest.raises(ValueError, match="invalid verification policy pack"):
        policy_manifest_entry(missing)


def test_strict_policy_material_matches_independently_supplied_copy(
    tmp_path: Path,
) -> None:
    pack_dir, external_path = _write_policy_pack_fixture(tmp_path)

    result = verify_policy_material(
        pack_dir,
        report_assurance="strict",
        acceptance_policy_path=external_path,
    )

    assert result.errors == ()
    assert result.required is True
    assert result.policy_pack_path == pack_dir / "policy/policy-pack.json"
    assert isinstance(result.policy_digest, str)


def test_strict_policy_material_rejects_missing_independent_anchor(
    tmp_path: Path,
) -> None:
    pack_dir, _external_path = _write_policy_pack_fixture(tmp_path)

    result = verify_policy_material(
        pack_dir,
        report_assurance="strict",
        acceptance_policy_path=None,
    )

    assert any(
        "independently supplied --policy-pack" in error for error in result.errors
    )


def test_strict_policy_material_rejects_different_acceptance_policy(
    tmp_path: Path,
) -> None:
    pack_dir, external_path = _write_policy_pack_fixture(tmp_path)
    write_policy_pack(
        external_path,
        build_policy_pack(
            tier="balanced",
            resolved_policy={"metrics": {"accuracy": {"delta_min_pp": -99.0}}},
        ),
    )

    result = verify_policy_material(
        pack_dir,
        report_assurance="strict",
        acceptance_policy_path=external_path,
    )

    assert any("does not exactly match" in error for error in result.errors)


def test_policy_material_rejects_tampered_sealed_bytes(tmp_path: Path) -> None:
    pack_dir, external_path = _write_policy_pack_fixture(tmp_path)
    sealed_path = pack_dir / POLICY_RELATIVE_PATH
    payload = json.loads(sealed_path.read_text(encoding="utf-8"))
    payload["metadata"] = {"tampered": True}
    _write_json(sealed_path, payload)

    result = verify_policy_material(
        pack_dir,
        report_assurance="strict",
        acceptance_policy_path=external_path,
    )

    assert any("digest mismatch" in error for error in result.errors)
    assert any("not bound by checksums" in error for error in result.errors)


def test_policy_material_rejects_undeclared_extra_file(tmp_path: Path) -> None:
    pack_dir, external_path = _write_policy_pack_fixture(tmp_path)
    _write_json(pack_dir / "policy/extra.json", {})

    result = verify_policy_material(
        pack_dir,
        report_assurance="strict",
        acceptance_policy_path=external_path,
    )

    assert any("undeclared policy material" in error for error in result.errors)
