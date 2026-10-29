"""Publish a complete delivery atomically, leaving the original package unchanged."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from invarlock.client_review_files import copy_evidence, file_hash, require_unchanged
from invarlock.filesystem.paths import pinned_directory
from invarlock.filesystem.staged_directory import staged_directory
from invarlock.report_provenance import write_archive
from invarlock.report_publication import validate_report_destinations


def _print(root: Path, browser_module: Path, paper: str) -> None:
    subprocess.run(
        [
            "node",
            str(Path(__file__).parent / "_data/client-review-print.cjs"),
            str(browser_module.absolute()),
            str(root),
            paper,
        ],
        check=True,
        timeout=120,
        capture_output=True,
        text=True,
    )


def export_client_review(
    package: Path, *, output: Path, browser_module: Path, paper: str = "A4"
) -> dict[str, Any]:
    """Export PDFs plus full/summary archives; no final directory on interruption."""
    if paper not in {"A4", "Letter"}:
        raise ValueError("supported paper sizes are A4 and Letter")
    package, output = Path(package).absolute(), Path(output).absolute()
    validate_report_destinations({"delivery": output}, evidence=package)
    with staged_directory(
        output, prefix=".report-delivery-", create_parents=True
    ) as stage:
        with pinned_directory(package) as source:
            state, originals = copy_evidence(source, stage.path / "package")
            root = stage.path / "package"
            manifest = json.loads((root / "package-manifest.json").read_text())
            files = manifest["files"]
            if any(
                name.startswith(("/", "../")) or ".." in Path(name).parts
                for name in files
            ):
                raise ValueError("unsafe package inventory")
            if set(originals) != set(files) | {
                "package-manifest.json",
                "handoff.zip",
            } or any(
                file_hash(root / name) != digest for name, digest in files.items()
            ):
                raise ValueError("package inventory is stale or incomplete")
            _print(root, browser_module, paper)
            # The snapshot and exact PDF hashes bind a distribution copy; no self-hashing claim.
            hashes = {
                p.relative_to(stage.path).as_posix(): file_hash(p)
                for p in sorted(stage.path.rglob("*"))
                if p.is_file()
            }
            summary = {
                "format": "invarlock/summary-delivery-v1",
                "evidence_statement_sha256": manifest["evidence_statement_sha256"],
                "snapshot_sha256": manifest["snapshot_sha256"],
                "scope": "Summary only. Inputs, outputs and replay evidence withheld; obtain the identity-matched companion package. Authored context and declared model names may still be sensitive.",
                "files": {"package/brief.pdf": hashes["package/brief.pdf"]},
            }
            (stage.path / "summary-inventory.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            write_archive(
                stage.path,
                stage.path / "summary.zip",
                ["package/brief.pdf", "summary-inventory.json"],
            )
            hashes.update(
                {
                    name: file_hash(stage.path / name)
                    for name in ["summary-inventory.json", "summary.zip"]
                }
            )
            inventory = {
                "format": "invarlock/client-review-delivery-v1",
                "snapshot_sha256": manifest["snapshot_sha256"],
                "evidence_statement_sha256": manifest["evidence_statement_sha256"],
                "scope": "All delivery files except this inventory and delivery.zip. Unsigned identities do not confer authority.",
                "files": hashes,
            }
            (stage.path / "delivery-inventory.json").write_text(
                json.dumps(inventory, indent=2) + "\n"
            )
            write_archive(
                stage.path,
                stage.path / "delivery.zip",
                [*hashes, "delivery-inventory.json"],
            )
            if any(
                file_hash(root / name) != digest for name, digest in originals.items()
            ):
                raise ValueError("original package changed during export")
            require_unchanged(source, state)
        stage.publish()
        stage.require_published_binding()
    return {
        "output": str(output),
        "snapshot_sha256": manifest["snapshot_sha256"],
        "archive": str(output / "delivery.zip"),
        "summary": str(output / "summary.zip"),
    }
