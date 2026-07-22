import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path


def test_generate_sbom_invokes_cyclonedx_for_install_surface(tmp_path: Path):
    script_path = Path("scripts/security/generate_sbom.sh")
    assert script_path.exists(), "SBOM generator script missing"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record_path = tmp_path / "cyclonedx-args.json"
    fake_cyclonedx = bin_dir / "cyclonedx-py"
    fake_cyclonedx.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "${CYCLO_RECORD}"
out=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--output-file" ]]; then
    out="$2"
    shift 2
  else
    shift
  fi
done
printf '{"bomFormat":"CycloneDX"}\n' > "$out"
""",
        encoding="utf-8",
    )
    fake_cyclonedx.chmod(0o755)
    sbom_path = tmp_path / "sbom.json"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "CYCLO_RECORD": str(record_path),
    }

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--scope",
            "install-surface",
            "--python",
            sys.executable,
            str(sbom_path),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "SBOM written to" in result.stdout
    args = record_path.read_text(encoding="utf-8").splitlines()
    assert args[:2] == ["environment", sys.executable]
    assert "--spec-version" in args
    assert args[args.index("--spec-version") + 1] == "1.4"
    assert "--output-format" in args
    assert args[args.index("--output-format") + 1] == "JSON"
    assert "--output-file" in args
    assert args[args.index("--output-file") + 1] == str(sbom_path)
    assert json.loads(sbom_path.read_text(encoding="utf-8"))["bomFormat"] == "CycloneDX"


def test_generate_sbom_rejects_unknown_scope_before_tool_lookup() -> None:
    script_path = Path("scripts/security/generate_sbom.sh")

    result = subprocess.run(
        ["bash", str(script_path), "--scope", "unknown"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--scope must be environment" in result.stderr


def test_offline_bundle_script_builds_manifested_tarball(tmp_path: Path):
    script_path = Path("scripts/release/make_offline_bundle.sh")
    assert script_path.exists(), "offline bundle generator script missing"

    dist_dir = tmp_path / "dist"
    provenance_dir = tmp_path / "provenance"
    out_dir = tmp_path / "bundle"
    dist_dir.mkdir()
    provenance_dir.mkdir()
    (dist_dir / "invarlock-1.2.3-py3-none-any.whl").write_text(
        "wheel bytes\n", encoding="utf-8"
    )
    (dist_dir / "invarlock-1.2.3-py3-none-any.whl.sigstore.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (provenance_dir / "attestation.intoto.jsonl").write_text("{}\n", encoding="utf-8")
    sbom_path = tmp_path / "sbom.json"
    sbom_path.write_text('{"bomFormat":"CycloneDX"}\n', encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--version",
            "1.2.3",
            "--tag",
            "v1.2.3",
            "--repo",
            "invarlock/invarlock",
            "--dist-dir",
            str(dist_dir),
            "--sbom",
            str(sbom_path),
            "--provenance-dir",
            str(provenance_dir),
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Offline release bundle written to" in result.stdout
    bundle_path = out_dir / "invarlock-1.2.3-offline-bundle.tar.gz"
    with tarfile.open(bundle_path, "r:gz") as bundle:
        manifest_member = bundle.getmember(
            "invarlock-1.2.3-offline-bundle/release_manifest.json"
        )
        manifest_file = bundle.extractfile(manifest_member)
        assert manifest_file is not None
        manifest = json.loads(manifest_file.read().decode("utf-8"))

    assert manifest["schema"] == "invarlock/release-offline-bundle-v1"
    assert manifest["bundle"]["tag"] == "v1.2.3"
    assert manifest["verification"]["certificate_identity"] == (
        "repo:invarlock/invarlock@refs/tags/v1.2.3"
    )
    assert [row["path"] for row in manifest["distributions"]] == [
        "dist/invarlock-1.2.3-py3-none-any.whl"
    ]
    assert manifest["distributions"][0]["sigstore_sidecars"] == [
        "dist/invarlock-1.2.3-py3-none-any.whl.sigstore.json"
    ]
    assert manifest["sbom"]["path"] == "invarlock-1.2.3-sbom.cdx.json"
    assert manifest["provenance_bundles"][0]["path"] == (
        "provenance/attestation.intoto.jsonl"
    )
