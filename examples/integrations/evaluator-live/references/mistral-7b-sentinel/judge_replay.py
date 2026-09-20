"""Replay the original public judge receipts without SDKs or model calls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _read_archive(path: Path, catalog: dict) -> dict[str, bytes]:
    spec = importlib.util.spec_from_file_location(
        "sentinel_reference_replay", HERE / "replay.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_archive(
        path,
        expected_sha256=catalog["sha256"].removeprefix("sha256:"),
        maximum_bytes=catalog["bytes"],
        maximum_expanded_bytes=catalog["expanded_bytes"],
        maximum_members=catalog["members"],
    )


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def replay(output: Path, *, require_installed: bool = True) -> dict:
    """Verify pinned inventories and reproduce every original signed decision."""
    import invarlock
    from invarlock.judge_measurements.acceptance import (
        replay_signed_judge_verification_receipt,
    )

    installed = Path(invarlock.__file__).resolve()
    if require_installed:
        _require(
            installed.is_relative_to(Path(sysconfig.get_path("purelib")).resolve()),
            "Use an independently installed core-only recipient.",
        )
    catalog = json.loads((HERE / "judge-reference.json").read_bytes())
    members = _read_archive(HERE / "judge-reference.zip", catalog["archive"])
    manifest = json.loads(members["archive-manifest.json"])["files"]
    _require(
        set(members) == set(manifest) | {"archive-manifest.json"},
        "Archive inventory differs.",
    )
    for name, pin in manifest.items():
        raw = members[name]
        _require(
            len(raw) == pin["bytes"] and _digest(raw) == pin["sha256"],
            "Archive member differs.",
        )
    reference = json.loads(members["reference.json"])
    _require(reference == catalog["reference"], "Pinned replay map differs.")
    companion = _read_archive(HERE / "captures.zip", catalog["companion_archive"])
    links = json.loads(members["model-captures.json"])
    for roles in links["captures"].values():
        for capture in roles.values():
            for name, pin in capture["files"].items():
                raw = companion[capture["capture_member"] + "/" + name]
                _require(
                    len(raw) == pin["bytes"] and _digest(raw) == pin["sha256"],
                    "Model capture link differs.",
                )
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve(strict=True)
    for name, raw in members.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    results = []
    for pack in reference["packs"]:
        entry = output / pack["directory"]
        for name in (
            "plan.json",
            "analysis_policy.json",
            "baseline_run.json",
            "subject_run.json",
        ):
            _require(
                json.loads((entry / "frozen" / name).read_bytes())
                == json.loads((entry / "evidence" / name).read_bytes()),
                "Frozen judge input differs.",
            )
        _require(
            _digest((entry / "frozen/plan.json").read_bytes()) == pack["plan_sha256"],
            "Frozen plan bytes differ.",
        )
        measured = json.loads((entry / "evidence/measurements.json").read_bytes())
        sources = {source["source_id"]: source for source in measured["sources"]}
        for source in pack["event_sources"]:
            raw = members[source["member"]]
            original = sources[source["source_id"]]
            _require(
                raw == original["content"].encode("utf-8")
                and len(raw) == original["byte_size"]
                and _digest(raw) == source["sha256"],
                "Original normalized SDK event source differs.",
            )
        result = replay_signed_judge_verification_receipt(
            entry / "recipient/verification.receipt.json",
            evidence_path=entry / "evidence",
            recipient_policy_path=entry / "recipient/trust.json",
            expected_verifier_identity=pack["verifier"]["identity"],
            expected_verifier_fingerprint=pack["verifier"]["signing_key_fingerprint"],
        ).to_dict()
        _require(
            all(
                result[key] == value
                for key, value in (pack["expected"] | pack["signer"]).items()
            ),
            "Original signed receipt or evidence replay differs: " + pack["id"],
        )
        results.append({"id": pack["id"], "verification": result})
    report = {
        "archive_sha256": catalog["archive"]["sha256"],
        "installed_package": str(installed),
        "counts": reference["counts"],
        "original_receipts_replayed": len(results),
        "results": results,
    }
    (output / "replay-results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from invarlock.security import enforce_network_policy

    enforce_network_policy(False)
    result = replay(args.output)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "results"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
