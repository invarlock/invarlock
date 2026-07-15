#!/usr/bin/env python3
"""Validate and canonically mirror the authoritative public evidence catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from invarlock.evidence_catalog import load_evidence_catalog  # noqa: E402
from invarlock.strict_yaml import load_yaml_object  # noqa: E402

AUTHORITATIVE_CATALOG = REPO_ROOT / "contracts" / "evidence_catalog_v1.json"
PACKAGED_CATALOG = (
    REPO_ROOT / "src" / "invarlock" / "_data" / "contracts" / "evidence_catalog_v1.json"
)


def _load_support_matrix() -> dict[str, Any]:
    payload = json.loads(
        (REPO_ROOT / "contracts" / "support_matrix.json").read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("lanes"), list):
        raise ValueError("support matrix is malformed")
    return payload


def build_catalog() -> dict[str, object]:
    """Return the checked static catalog; do not derive entries from run machinery."""

    catalog = load_evidence_catalog(AUTHORITATIVE_CATALOG)
    support = _load_support_matrix()
    published = {
        str(lane["lane_id"]): lane
        for lane in support["lanes"]
        if isinstance(lane, dict) and lane.get("support_tier") == "maintained_catalog"
    }
    if len(published) != 39 or set(catalog.entries) != set(published):
        raise ValueError(
            "public evidence catalog must equal the exact 39-entry maintained catalog"
        )

    config_root = (REPO_ROOT / "configs").resolve(strict=True)
    for lane_id, entry in catalog.entries.items():
        model = entry["model"]
        preset = entry["preset"]
        if not isinstance(model, dict) or not isinstance(preset, dict):
            raise ValueError(f"catalog entry is malformed: {lane_id}")
        if model.get("adapter") != published[lane_id].get("adapter"):
            raise ValueError(
                f"catalog adapter disagrees with support matrix: {lane_id}"
            )

        preset_path = (REPO_ROOT / str(preset["path"])).resolve(strict=True)
        try:
            preset_path.relative_to(config_root)
        except ValueError as exc:
            raise ValueError(f"catalog preset is outside configs: {lane_id}") from exc
        if not preset_path.is_file():
            raise ValueError(f"catalog preset is not a regular file: {lane_id}")
        observed_digest = (
            "sha256:" + hashlib.sha256(preset_path.read_bytes()).hexdigest()
        )
        if observed_digest != preset.get("sha256"):
            raise ValueError(f"catalog preset digest mismatch: {lane_id}")
        preset_payload = load_yaml_object(preset_path, label="catalog preset")
        preset_model = (
            preset_payload.get("model") if isinstance(preset_payload, dict) else None
        )
        if not isinstance(preset_model, dict) or preset_model.get(
            "adapter"
        ) != model.get("adapter"):
            raise ValueError(f"catalog preset adapter mismatch: {lane_id}")

    return catalog.payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=AUTHORITATIVE_CATALOG,
        help="Canonical mirror destination; entries always come from the static catalog.",
    )
    args = parser.parse_args()
    rendered = json.dumps(build_catalog(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != rendered:
            return 1
        if args.output == AUTHORITATIVE_CATALOG and (
            PACKAGED_CATALOG.read_text(encoding="utf-8") != rendered
        ):
            return 1
        return 0
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
