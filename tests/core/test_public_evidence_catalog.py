from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from invarlock.evidence_catalog import EvidenceCatalogError, load_evidence_catalog
from invarlock.public_contracts import load_support_matrix

ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "contracts" / "evidence_catalog_v1.json"


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value)) if value else set()
    return set()


def test_public_evidence_catalog_is_exact_and_contains_no_operational_placement() -> (
    None
):
    catalog = load_evidence_catalog(CATALOG_PATH)
    support = load_support_matrix()
    expected_ids = {
        lane["lane_id"]
        for lane in support["lanes"]
        if lane.get("support_tier") == "maintained_catalog"
    }
    assert len(catalog.entries) == 39
    assert set(catalog.entries) == expected_ids
    assert {lane_id: entry["slug"] for lane_id, entry in catalog.entries.items()} == {
        lane_id: lane_id for lane_id in expected_ids
    }
    assert all(entry["required_artifacts"] for entry in catalog.entries.values())
    for entry in catalog.entries.values():
        execution = entry["execution"]
        assert execution["profile"] == "release"
        assert (execution["preview_n"], execution["final_n"]) == (400, 400)
        assert execution["tier"] == "balanced"
        assert execution["assurance_mode"] == "strict"
        assert execution["execution_mode"] == "container"
        assert execution["edit_name"] == "noop"
    assert not _keys(catalog.payload) & {
        "endpoint",
        "gpu",
        "host",
        "node",
        "placement",
        "queue",
        "remote",
        "resource",
        "scheduler",
    }


def test_public_evidence_catalog_is_reproducible_and_packaged_verbatim() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/evidence_catalog/build_public_catalog.py", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    packaged = ROOT / "src/invarlock/_data/contracts/evidence_catalog_v1.json"
    assert packaged.read_bytes() == CATALOG_PATH.read_bytes()
    assert json.loads(packaged.read_text(encoding="utf-8"))["entry_count"] == 39


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile", "ci"),
        ("preview_n", 240),
        ("final_n", 240),
        ("profile_sha256", "sha256:" + ("0" * 64)),
    ],
)
def test_public_catalog_rejects_execution_policy_downgrades(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    payload["entries"][0]["execution"][field] = value
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvidenceCatalogError, match="execution"):
        load_evidence_catalog(catalog_path)
