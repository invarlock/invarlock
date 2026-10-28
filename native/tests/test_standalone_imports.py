"""Native public imports preserve reference runs and published evidence bytes."""

import base64
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from invarlock.evaluation_records.adapters import _parse_run_bytes
from invarlock.evaluation_transaction import (
    evaluate_request_file,
    preflight_evaluation_request,
)
from tests.cli.test_import_journey import _key, _materialize_request
from tests.evaluation_records.test_langfuse_adapter import export as langfuse_export
from tests.evaluation_records.test_scalar_integrations import native_entry

ROOT = Path(__file__).resolve().parents[2]
BIN = Path(os.environ["CARGO_TARGET_DIR"]) / "debug/invarlock-native-verify"


def native(raw, options):
    return subprocess.run(
        [str(BIN), "import-evaluation-run", "--request", "-"],
        input=json.dumps(
            {"raw_base64": base64.b64encode(raw).decode(), "options": options}
        ),
        capture_output=True,
        text=True,
        env={"PATH": "/nonexistent"},
    )


@pytest.mark.parametrize(
    "name",
    [
        "deepeval",
        "ragas",
        "lighteval",
        "hugging-face-evaluate",
        "autoevals",
        "openevals",
        "arize-phoenix-evals",
        "opik",
    ],
)
@pytest.mark.parametrize(
    "mode", ["plain", "error", "projection", "score_conflict", "duplicate"]
)
def test_scalar_import_values_and_rejections(name, mode):
    row = native_entry(name)
    options = {
        "adapter": "evaluator-native-json",
        "source": {"name": name, "version": "1"},
        "run_id": "run",
        "artifact_digest": "sha256:" + "a" * 64,
    }
    if mode == "error":
        row["error"] = "failed"
    if mode == "projection":
        options["input_projection"] = {
            "kind": "json-pointer",
            "pointer": "/context/upstream_record/id",
        }
    if mode == "score_conflict":
        row["metadata"]["invarlock_scores"] = {"x": True}
    rows = [row, row] if mode == "duplicate" else [row]
    raw = json.dumps(rows).encode()
    try:
        expected = _parse_run_bytes(raw, **options)
    except ValueError:
        assert native(raw, options).returncode != 0
    else:
        got = native(raw, options)
        assert got.returncode == 0, got.stdout + got.stderr
        assert json.dumps(json.loads(got.stdout), sort_keys=True) == json.dumps(
            expected, sort_keys=True
        )


@pytest.mark.parametrize(
    "mode", ["preflight", "publish", "bad_records", "unknown_setting", "wrong_identity"]
)
def test_runtime_import_public_journey(tmp_path, mode):
    root = tmp_path.resolve()
    material = _materialize_request(root)
    path = Path(material["request"])
    key, _ = _key(root / "signer.pem")
    request = yaml.safe_load(path.read_text())
    if mode == "unknown_setting":
        request["comparison"]["baseline"]["runtime"]["settings"]["unknown"] = True
    if mode == "wrong_identity":
        request["comparison"]["baseline"]["runtime"]["settings"][
            "checkpoint_tree_sha256"
        ] = "f" * 64
    if mode == "bad_records":
        p = Path(material["records"])
        p.write_bytes(p.read_bytes() + b" ")
    request["output"]["evidence"] = "python-out"
    path.write_text(yaml.safe_dump(request))
    func = (
        preflight_evaluation_request if mode == "preflight" else evaluate_request_file
    )
    try:
        expected = json.loads(func(path, signing_key_path=key).as_json())
    except ValueError:
        expected = None
    request["output"]["evidence"] = "rust-out"
    path.write_text(yaml.safe_dump(request))
    args = [str(BIN), "evaluate", str(path), "--signing-key", str(key), "--json"]
    if mode == "preflight":
        args.append("--preflight")
    process = subprocess.run(
        args, capture_output=True, text=True, env={"PATH": "/nonexistent"}
    )
    if expected is None:
        assert process.returncode != 0, process.stdout
        assert not (root / "rust-out").exists()
        return
    assert process.returncode == 0, process.stdout + process.stderr
    actual = json.loads(process.stdout)
    if mode == "preflight":
        expected["output"] = "rust-out"
    else:
        expected["evidence"] = str(root / "rust-out")
    assert actual == expected
    if mode == "publish":

        def files(p):
            return {
                x.relative_to(p).as_posix(): x.read_bytes()
                for x in p.rglob("*")
                if x.is_file()
            }

        assert files(root / "python-out") == files(root / "rust-out")


def test_langfuse_production_array_bound():
    value = langfuse_export()
    value["result"]["run_evaluations"] = [
        {"name": str(i), "value": 1.0} for i in range(50001)
    ]
    options = {
        "adapter": "langfuse-json",
        "source": {"name": "langfuse", "version": "4.14.1"},
        "run_id": "run",
        "artifact_digest": "sha256:" + "a" * 64,
    }
    raw = json.dumps(value).encode()
    with pytest.raises(ValueError):
        _parse_run_bytes(raw, **options)
    assert native(raw, options).returncode != 0
