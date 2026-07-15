from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

from scripts.checks.public_evidence_checks.guard_scenarios import (
    check_historical_guard_scenario_observations,
)


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True).encode() + b"\n"


def _digest(data: bytes, *, prefixed: bool = True) -> str:
    value = hashlib.sha256(data).hexdigest()
    return f"sha256:{value}" if prefixed else value


def _build_fixture(
    tmp_path: Path,
    *,
    bad_rmt_math: bool = False,
    bad_ve_baseline: bool = False,
    bad_ve_subject_math: bool = False,
    bad_ve_input: bool = False,
    bad_manifest_model_id: bool = False,
    bad_manifest_revision: bool = False,
    missing_manifest_scenario: bool = False,
    unlisted_payload: bool = False,
) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    index = json.loads(
        (repo_root / "public_evidence/guard_scenario_observations.json").read_text()
    )
    observations = {item["scenario_id"]: item for item in index["observations"]}
    common_caps = [
        {
            "type": "family_z_cap",
            "selected": True,
            "module": "model.layers.0.self_attn.q_proj",
        },
        {
            "type": "family_z_cap",
            "selected": True,
            "module": "model.layers.0.self_attn.k_proj",
        },
    ]

    members: dict[str, bytes] = {}
    for scenario, observation in observations.items():
        report: dict[str, object] = {
            "validation": {"primary_metric_acceptable": True},
            "primary_metric": {
                "ratio_vs_baseline": observation["primary_metric"]["ratio_vs_baseline"]
            },
            "assurance": {"mode": "off"},
        }
        if scenario.startswith("spectral_moderate_scale_mlp"):
            report["spectral"] = {
                "caps_applied": 3,
                "violations": [
                    *common_caps,
                    {
                        "type": "family_z_cap",
                        "selected": True,
                        "module": "model.layers.31.mlp.up_proj",
                    },
                ],
            }
        elif scenario.startswith("spectral_moderate_scale_attn"):
            report["spectral"] = {
                "caps_applied": 2,
                "violations": common_caps,
            }
        data = _json_bytes(report)
        members[observation["report"]["path"]] = data
        observation["report"]["sha256"] = _digest(data)

    rmt = observations["rmt_norm_noise_l31_ffn_up_b030"]
    current = 1.005 if bad_rmt_math else 1.02
    rmt_payload = {
        "stable": False,
        "epsilon_violations": [
            {
                "family": "ffn",
                "module": "model.layers.31.mlp.up_proj",
                "edge_base": 1.0,
                "edge_cur": current,
                "epsilon": 0.01,
                "allowed": 1.01,
                "delta": current - 1.0,
            }
        ],
    }
    data = _json_bytes(rmt_payload)
    members[rmt["sidecar"]["path"]] = data
    rmt["sidecar"]["sha256"] = _digest(data)

    ve = observations["ve_mlp_scale_skew_l31_down_s090"]
    subject = {
        "signal": True,
        "ppl_no_ve": 0.0 if bad_ve_input else 100.0,
        "ppl_with_ve": 99.0,
        "abs_improvement": 0.5 if bad_ve_subject_math else 1.0,
        "ab_gain": 0.01,
    }
    baseline = {
        "signal": bad_ve_baseline,
        "ppl_no_ve": 100.0,
        "ppl_with_ve": 101.0,
        "ab_gain": -0.01,
        "abs_improvement": -1.0,
    }
    for key, payload in (("sidecar", subject), ("baseline_sidecar", baseline)):
        data = _json_bytes(payload)
        members[ve[key]["path"]] = data
        ve[key]["sha256"] = _digest(data)

    root = index["source_asset"]["manifest"]["path"].rsplit("/", 1)[0]
    manifest = {
        "artifact_root": root,
        "files": [
            {
                "path": PurePosixPath(path).relative_to(root).as_posix(),
                "sha256": _digest(data, prefixed=False),
                "size_bytes": len(data),
            }
            for path, data in sorted(members.items())
        ],
        "schema": "invarlock.public_evidence.guard_value_manifest.v2",
        "source_run": {
            "model_id": (
                "example/other-model" if bad_manifest_model_id else index["model"]["id"]
            ),
            "model_revision": (
                "0" * 40 if bad_manifest_revision else index["model"]["revision"]
            ),
            "scenario_ids": [
                scenario
                for position, scenario in enumerate(observations)
                if not missing_manifest_scenario or position != 0
            ],
        },
    }
    manifest_data = _json_bytes(manifest)
    manifest_path = index["source_asset"]["manifest"]["path"]
    members[manifest_path] = manifest_data
    index["source_asset"]["manifest"]["sha256"] = _digest(manifest_data)
    if unlisted_payload:
        members[f"{root}/unlisted.json"] = b"{}\n"

    archive_path = tmp_path / "guard-observations.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, data in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    archive_data = archive_path.read_bytes()
    index["source_asset"].update(
        {
            "name": archive_path.name,
            "url": f"https://github.com/invarlock/invarlock/releases/download/test/{archive_path.name}",
            "size_bytes": len(archive_data),
            "sha256": _digest(archive_data),
        }
    )
    index_path = tmp_path / "guard_scenario_observations.json"
    index_path.write_bytes(_json_bytes(index))
    return index_path, archive_path


def test_historical_guard_scenario_checker_replays_closed_archive(
    tmp_path: Path,
) -> None:
    index_path, archive_path = _build_fixture(tmp_path)
    errors: list[str] = []
    assert check_historical_guard_scenario_observations(
        errors,
        tmp_path,
        index_path=index_path,
        asset_path=archive_path,
    )
    assert errors == []


def test_historical_guard_scenario_checker_rejects_current_assurance_claim(
    tmp_path: Path,
) -> None:
    index_path, archive_path = _build_fixture(tmp_path)
    index = json.loads(index_path.read_text())
    index["current_assurance"]["current_strict_assurance"] = True
    index_path.write_bytes(_json_bytes(index))
    errors: list[str] = []
    assert not check_historical_guard_scenario_observations(
        errors, tmp_path, index_path=index_path, asset_path=archive_path
    )
    assert any("must not claim current strict assurance" in error for error in errors)


def test_historical_guard_scenario_checker_recomputes_rmt_and_ve(
    tmp_path: Path,
) -> None:
    for name, kwargs, expected in (
        ("rmt", {"bad_rmt_math": True}, "RMT epsilon violation arithmetic"),
        ("ve", {"bad_ve_baseline": True}, "true subject-vs-baseline signal"),
        (
            "ve-arithmetic",
            {"bad_ve_subject_math": True},
            "VE subject gain arithmetic",
        ),
        (
            "ve-input",
            {"bad_ve_input": True},
            "VE subject perplexities must be finite and positive",
        ),
    ):
        case = tmp_path / name
        case.mkdir()
        index_path, archive_path = _build_fixture(case, **kwargs)
        errors: list[str] = []
        assert not check_historical_guard_scenario_observations(
            errors, case, index_path=index_path, asset_path=archive_path
        )
        assert any(expected in error for error in errors)


def test_historical_guard_scenario_checker_binds_manifest_source_run(
    tmp_path: Path,
) -> None:
    for name, kwargs, expected in (
        (
            "model-id",
            {"bad_manifest_model_id": True},
            "model_id does not match the index",
        ),
        (
            "revision",
            {"bad_manifest_revision": True},
            "model_revision does not match the index",
        ),
        (
            "scenario",
            {"missing_manifest_scenario": True},
            "does not bind every indexed scenario",
        ),
    ):
        case = tmp_path / name
        case.mkdir()
        index_path, archive_path = _build_fixture(case, **kwargs)
        errors: list[str] = []
        assert not check_historical_guard_scenario_observations(
            errors, case, index_path=index_path, asset_path=archive_path
        )
        assert any(expected in error for error in errors)


def test_historical_guard_scenario_checker_rejects_archive_tamper(
    tmp_path: Path,
) -> None:
    index_path, archive_path = _build_fixture(tmp_path)
    archive_path.write_bytes(archive_path.read_bytes() + b"tamper")
    errors: list[str] = []
    assert not check_historical_guard_scenario_observations(
        errors, tmp_path, index_path=index_path, asset_path=archive_path
    )
    assert any("size or sha256" in error for error in errors)


def test_historical_guard_scenario_checker_binds_indexed_signal_values(
    tmp_path: Path,
) -> None:
    for name, scenario, field, value, expected in (
        (
            "rmt",
            "rmt_norm_noise_l31_ffn_up_b030",
            "stable",
            True,
            "RMT indexed signal values",
        ),
        (
            "ve",
            "ve_mlp_scale_skew_l31_down_s090",
            "baseline_signal",
            True,
            "VE indexed signal values",
        ),
    ):
        case = tmp_path / name
        case.mkdir()
        index_path, archive_path = _build_fixture(case)
        index = json.loads(index_path.read_text())
        observation = next(
            item for item in index["observations"] if item["scenario_id"] == scenario
        )
        observation["signal"][field] = value
        index_path.write_bytes(_json_bytes(index))
        errors: list[str] = []
        assert not check_historical_guard_scenario_observations(
            errors, case, index_path=index_path, asset_path=archive_path
        )
        assert any(expected in error for error in errors)


def test_historical_guard_scenario_checker_rejects_unlisted_payload(
    tmp_path: Path,
) -> None:
    index_path, archive_path = _build_fixture(tmp_path, unlisted_payload=True)
    errors: list[str] = []
    assert not check_historical_guard_scenario_observations(
        errors, tmp_path, index_path=index_path, asset_path=archive_path
    )
    assert any("not closed over its artifact root" in error for error in errors)
