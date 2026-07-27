from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "evaluator-qualification"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def test_retained_matrix_requalifies_offline() -> None:
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE / "matrix.py"), "verify"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("verified ") == 12


def test_matrix_is_twelve_real_upstream_execution_profiles() -> None:
    matrix = _load(EXAMPLE / "matrix.json")
    profiles = matrix["profiles"]
    assert isinstance(profiles, list)
    assert len(profiles) == 12
    expected = {
        "autoevals",
        "deepeval",
        "garak",
        "hugging-face-evaluate",
        "inspect-ai",
        "lighteval",
        "lm-evaluation-harness",
        "mlflow",
        "openevals",
        "promptfoo",
        "pydantic-evals",
        "ragas",
    }
    assert {profile["profile_id"] for profile in profiles} == expected

    for profile in profiles:
        artifact = EXAMPLE / "artifacts" / profile["profile_id"]
        raw = _load(artifact / "upstream-output.json")
        result = _load(artifact / "qualification-result.json")
        assert raw["upstream"] == profile["upstream"]
        assert isinstance(raw["entrypoint"], str) and raw["entrypoint"]
        assert raw["entrypoint"] != "precomputed"
        assert result["profile_id"] == profile["profile_id"]
        if profile["authority"]["mode"] == "deterministic_per_record":
            assert [record["score"] for record in raw["records"]] == [1.0, 0.0]
            assert result["authority"] == "verdict_authority"
            assert result["record_count"] == 2
        else:
            assert "summary" in raw
            assert result["authority"] == "observation_only"
            assert result["record_count"] == 0


def test_python_execution_evidence_names_the_pinned_upstream_package() -> None:
    matrix = _load(EXAMPLE / "matrix.json")
    for profile in matrix["profiles"]:
        if profile["upstream"]["ecosystem"] != "pypi":
            continue
        raw = _load(
            EXAMPLE / "artifacts" / profile["profile_id"] / "upstream-output.json"
        )
        inventory = {item["name"]: item["version"] for item in raw["environment"]}
        package_name = profile["upstream"]["name"].lower().replace("_", "-")
        assert inventory[package_name] == profile["upstream"]["version"]


def test_promptfoo_execution_binds_registry_integrity() -> None:
    raw = _load(EXAMPLE / "artifacts" / "promptfoo" / "upstream-output.json")
    package = raw["environment"][0]
    declaration = dict(
        line.split("=", 1)
        for line in (EXAMPLE / "locks" / "promptfoo.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    assert declaration["package"] == f"promptfoo@{package['version']}"
    assert declaration["integrity"] == package["integrity"]
    assert declaration["shasum"] == package["shasum"]


def test_public_retained_artifacts_do_not_contain_local_paths_or_secrets() -> None:
    forbidden = (
        b"/users/",
        b"/private/tmp/",
        b"root@",
        b"authorization:",
        b"api_key",
        b"bearer ",
    )
    for path in (EXAMPLE / "artifacts").rglob("*.json"):
        payload = path.read_bytes().lower()
        for marker in forbidden:
            assert marker not in payload, path
