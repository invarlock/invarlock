from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.cli.commands import inputs as inputs_commands
from invarlock.evidence_catalog import EvidenceCatalogError


def test_evidence_catalog_human_output_covers_valid_and_invalid_catalogs(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    catalog = Path(__file__).resolve().parents[2] / "contracts/evidence_catalog_v1.json"

    valid = runner.invoke(
        app, ["advanced", "evidence-catalog", "validate", str(catalog)]
    )
    invalid = runner.invoke(
        app,
        ["advanced", "evidence-catalog", "validate", str(tmp_path / "missing.json")],
    )

    assert valid.exit_code == 0
    assert valid.stdout.startswith("Evidence catalog valid: 39 entries, sha256:")
    assert invalid.exit_code == 2
    assert "Evidence catalog is invalid" in invalid.output


def test_catalog_input_binding_writes_exclusively_and_translates_collisions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[dict[str, object]] = []

    def build_binding(**kwargs: object) -> dict[str, object]:
        captured.append(kwargs)
        return {"ok": True, "lane_id": "lane-a"}

    monkeypatch.setattr(
        inputs_commands, "build_evaluation_input_binding", build_binding
    )
    output = tmp_path / "nested/binding.json"
    inputs_commands.binding_command(
        catalog="catalog.json",
        lane="lane-a",
        resolved_inputs="resolved.json",
        preset="preset.yaml",
        input_materialization="dataset-evidence.json",
        out=str(output),
    )

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "lane_id": "lane-a",
        "ok": True,
    }
    assert captured == [
        {
            "catalog_path": Path("catalog.json"),
            "lane_id": "lane-a",
            "resolved_inputs_path": Path("resolved.json"),
            "preset_path": Path("preset.yaml"),
            "input_materialization_path": Path("dataset-evidence.json"),
        }
    ]
    with pytest.raises(typer.BadParameter, match="binding.json"):
        inputs_commands.binding_command(
            catalog="catalog.json",
            lane="lane-a",
            resolved_inputs="resolved.json",
            preset="preset.yaml",
            input_materialization=None,
            out=str(output),
        )


def test_catalog_materialize_command_requires_network_and_emits_closed_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with pytest.raises(typer.BadParameter, match="allow-network"):
        inputs_commands.materialize_command(
            catalog="catalog.json",
            lane="lane-a",
            out=str(tmp_path / "dataset"),
            allow_network=False,
            json_out=False,
        )

    monkeypatch.setattr(
        inputs_commands,
        "materialize_catalog_input",
        lambda **_kwargs: {"ok": True, "lane_id": "lane-a"},
    )
    inputs_commands.materialize_command(
        catalog="catalog.json",
        lane="lane-a",
        out=str(tmp_path / "dataset"),
        allow_network=True,
        json_out=True,
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["format_version"] == "catalog-input-materialize-v1"
    assert payload["ok"] is True
    inputs_commands.materialize_command(
        catalog="catalog.json",
        lane="lane-a",
        out=str(tmp_path / "dataset"),
        allow_network=True,
        json_out=False,
    )
    assert capsys.readouterr().out == ""

    def fail_materialization(**_kwargs: object) -> dict[str, object]:
        raise EvidenceCatalogError("pinned input is unavailable")

    monkeypatch.setattr(
        inputs_commands, "materialize_catalog_input", fail_materialization
    )
    with pytest.raises(typer.Exit) as json_exit:
        inputs_commands.materialize_command(
            catalog="catalog.json",
            lane="lane-a",
            out=str(tmp_path / "failed"),
            allow_network=True,
            json_out=True,
        )
    assert json_exit.value.exit_code == 2
    failure = json.loads(capsys.readouterr().out)
    assert failure["ok"] is False
    assert failure["errors"] == ["pinned input is unavailable"]

    with pytest.raises(typer.Exit) as human_exit:
        inputs_commands.materialize_command(
            catalog="catalog.json",
            lane="lane-a",
            out=str(tmp_path / "failed"),
            allow_network=True,
            json_out=False,
        )
    assert human_exit.value.exit_code == 2
    assert capsys.readouterr().out == ""


def test_catalog_prepare_command_binds_optional_materialization_and_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    captured: list[dict[str, object]] = []

    def prepare(**kwargs: object) -> dict[str, object]:
        captured.append(kwargs)
        return {"ok": True, "lane_id": "lane-a"}

    monkeypatch.setattr(inputs_commands, "prepare_catalog_preset", prepare)
    inputs_commands.prepare_command(
        catalog="catalog.json",
        lane="lane-a",
        resolved_inputs="resolved.json",
        preset="preset.yaml",
        materialization_dir="dataset",
        out=str(tmp_path / "prepared.yaml"),
        json_out=True,
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["format_version"] == "catalog-input-prepare-v1"
    assert payload["ok"] is True
    assert captured[0]["materialization_dir"] == Path("dataset")
    inputs_commands.prepare_command(
        catalog="catalog.json",
        lane="lane-a",
        resolved_inputs="resolved.json",
        preset="preset.yaml",
        materialization_dir=None,
        out=str(tmp_path / "prepared-without-materialization.yaml"),
        json_out=False,
    )
    assert captured[1]["materialization_dir"] is None
    assert capsys.readouterr().out == ""

    def fail_prepare(**_kwargs: object) -> dict[str, object]:
        raise EvidenceCatalogError("resolved inputs do not match")

    monkeypatch.setattr(inputs_commands, "prepare_catalog_preset", fail_prepare)
    with pytest.raises(typer.Exit) as json_exit:
        inputs_commands.prepare_command(
            catalog="catalog.json",
            lane="lane-a",
            resolved_inputs="resolved.json",
            preset="preset.yaml",
            materialization_dir=None,
            out=str(tmp_path / "failed.yaml"),
            json_out=True,
        )
    assert json_exit.value.exit_code == 2
    failure = json.loads(capsys.readouterr().out)
    assert failure["ok"] is False
    assert failure["errors"] == ["resolved inputs do not match"]

    with pytest.raises(typer.Exit) as human_exit:
        inputs_commands.prepare_command(
            catalog="catalog.json",
            lane="lane-a",
            resolved_inputs="resolved.json",
            preset="preset.yaml",
            materialization_dir=None,
            out=str(tmp_path / "failed.yaml"),
            json_out=False,
        )
    assert human_exit.value.exit_code == 2
    assert capsys.readouterr().out == ""
