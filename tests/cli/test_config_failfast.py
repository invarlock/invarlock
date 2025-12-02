import json
from pathlib import Path

import pytest
import yaml
from click.exceptions import Exit as ClickExit

from invarlock.cli.commands.run import run_command


@pytest.fixture()
def minimal_cfg(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
model:
  adapter: hf_gpt2
  id: gpt2
edit:
  name: quant_rtn
  plan: {}

dataset:
  provider: synthetic
  id: synthetic
  split: validation
  seq_len: 16
  stride: 8
  preview_n: 2
  final_n: 2

guards:
  order: [invariants]
  invariants: {}

output:
  dir: runs

"""
    )
    return cfg_path


def test_empty_edit_name_fails(minimal_cfg: Path, tmp_path: Path) -> None:
    cfg_dict = yaml.safe_load(minimal_cfg.read_text())
    cfg_dict["edit"]["name"] = ""
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    with pytest.raises(ClickExit):
        run_command(
            config=str(cfg_path),
            device="cpu",
            profile="ci",
            out=str(tmp_path / "runs"),
        )


def test_missing_evaluation_windows_with_baseline_fails(
    minimal_cfg: Path, tmp_path: Path
) -> None:
    baseline_report = tmp_path / "baseline.json"
    baseline_report.write_text(json.dumps({"evaluation_windows": {}}))
    with pytest.raises(ClickExit):
        run_command(
            config=str(minimal_cfg),
            device="cpu",
            profile="release",
            out=str(tmp_path / "runs"),
            baseline=str(baseline_report),
        )
