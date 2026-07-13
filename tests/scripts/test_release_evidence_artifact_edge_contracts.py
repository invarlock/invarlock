from __future__ import annotations

import json
from pathlib import Path

from tests.scripts._support_release_evidence_check import (
    release_checker_module,
    repo_root,
    write_bundle_manifest,
)


def test_release_evidence_check_rejects_guard_sbom_and_bundle_edges(
    tmp_path: Path,
) -> None:
    module = release_checker_module(repo_root())
    failures: list[str] = []

    guard_json = tmp_path / "guard-validation-smoke.json"
    guard_md = tmp_path / "guard-validation-smoke.md"
    guard_json.write_text(
        json.dumps({"schema": "unexpected", "rate_rows": "not-a-list"}),
        encoding="utf-8",
    )
    guard_md.write_text("", encoding="utf-8")
    module._validate_guard_validation(
        json_path=guard_json,
        markdown_path=guard_md,
        failures=failures,
    )
    assert any("top-level fields must match v1 exactly" in item for item in failures)

    failures.clear()
    sbom = tmp_path / "sbom.json"
    sbom.write_text("[]", encoding="utf-8")
    module._validate_sbom(sbom, failures)
    assert any("SBOM must be a JSON object" in item for item in failures)

    failures.clear()
    missing_bundle_dir = tmp_path / "missing-bundles"
    module._validate_offline_bundle(missing_bundle_dir, failures)
    assert any("offline release bundle missing" in item for item in failures)

    failures.clear()
    bundle_dir = tmp_path / "bundles"
    bundle_dir.mkdir()
    (bundle_dir / "invalid.tar.gz").write_text("not a tarball", encoding="utf-8")
    write_bundle_manifest(bundle_dir, "non-object.tar.gz", [])
    write_bundle_manifest(bundle_dir, "bad-schema.tar.gz", {"schema": "bad"})
    write_bundle_manifest(
        bundle_dir,
        "empty-distributions.tar.gz",
        {"schema": "invarlock/release-offline-bundle-v1", "distributions": []},
    )
    write_bundle_manifest(
        bundle_dir,
        "missing-wheel.tar.gz",
        {
            "schema": "invarlock/release-offline-bundle-v1",
            "distributions": [{"path": "dist/invarlock-0.9.0.tar.gz"}],
        },
    )
    write_bundle_manifest(
        bundle_dir,
        "missing-sdist.tar.gz",
        {
            "schema": "invarlock/release-offline-bundle-v1",
            "distributions": [{"path": "dist/invarlock-0.9.0-py3-none-any.whl"}],
        },
    )
    module._validate_offline_bundle(bundle_dir, failures)
    assert any("offline release bundle invalid" in item for item in failures)
    assert any("manifest must be an object" in item for item in failures)
    assert any("schema is not recognized" in item for item in failures)
    assert any("has no distributions" in item for item in failures)
    assert any("missing wheel distribution" in item for item in failures)
    assert any("missing sdist distribution" in item for item in failures)
