from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import jsonschema  # type: ignore
from typer.testing import CliRunner


def test_provider_schema_local_jsonl_requires_path(tmp_path: Path) -> None:
    schema = json.loads(
        (Path.cwd() / "tests/schemas/dataset_provider.schema.json").read_text("utf-8")
    )
    # Missing file/path/data_files should fail schema
    cfg = {"kind": "local_jsonl", "text_field": "text"}
    try:
        jsonschema.validate(instance=cfg, schema=schema)
        ok = True
    except jsonschema.ValidationError:  # type: ignore[attr-defined]
        ok = False
    assert ok is False


def test_doctor_reports_missing_local_jsonl_file(tmp_path: Path) -> None:
    # Prepare a minimal config referencing non-existent file
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
model:
  id: gpt2
  adapter: hf_gpt2
dataset:
  provider:
    kind: local_jsonl
    file: does_not_exist.jsonl
output:
  dir: runs
        """
    )

    class _Provider:
        def estimate_capacity(self, **kwargs):
            return {"available_nonoverlap": 0}

    with (
        patch("invarlock.core.registry.get_registry") as mock_registry,
        patch("invarlock.model_profile.detect_model_profile"),
        patch(
            "invarlock.model_profile.resolve_tokenizer",
            return_value=(object(), "tokhash"),
        ),
        patch("invarlock.eval.data.get_provider", return_value=_Provider()),
        patch(
            "invarlock.cli.commands.run._resolve_metric_and_provider",
            return_value=("ppl_causal", "local_jsonl", {}),
        ),
    ):
        reg = Mock()
        reg.list_adapters.return_value = []
        reg.list_edits.return_value = []
        reg.list_guards.return_value = []
        reg.get_plugin_info.return_value = {
            "module": "invarlock.adapters",
            "entry_point": "",
        }
        mock_registry.return_value = reg

        from invarlock.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, ["doctor", "--config", str(cfg), "--json"])

    payload = json.loads(result.stdout)
    codes = {f.get("code") for f in payload.get("findings", [])}
    assert "D011" in codes, "Expected missing file error (D011)"
