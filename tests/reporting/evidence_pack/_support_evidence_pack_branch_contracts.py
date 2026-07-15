from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import invarlock.evidence_pack_policy as policy
from invarlock.policy_pack import build_policy_pack
from invarlock.reporting.verify_contract import VerifyExecutionResult, VerifyOutcome


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _minimal_baseline_pack(tmp_path: Path) -> tuple[Path, str, str]:
    pack = tmp_path / "pack"
    report_rel = "reports/model/clean/evaluation.report.json"
    baseline_rel = "baselines/model/evaluation.report.json"
    _write_json(pack / report_rel, {})
    _write_json(pack / baseline_rel, {})
    digest = _sha256(pack / baseline_rel)
    (pack / "checksums.sha256").write_text(
        f"{digest}  {baseline_rel}\n", encoding="utf-8"
    )
    _write_json(
        pack / "manifest.json",
        {
            "verification": {"report_assurance": "off"},
            "verification_baselines": [
                {
                    "name": "baseline-1",
                    "path": baseline_rel,
                    "digest": f"sha256:{digest}",
                    "report_paths": [report_rel],
                }
            ],
        },
    )
    return pack, report_rel, baseline_rel


def _write_policy_fixture(tmp_path: Path) -> tuple[Path, Path]:
    pack = tmp_path / "pack"
    sealed = pack / policy.POLICY_RELATIVE_PATH
    acceptance_policy = tmp_path / "acceptance-policy.json"
    payload = build_policy_pack(tier="balanced", resolved_policy={"metrics": {}})
    policy.write_canonical_policy_pack(sealed, payload)
    policy.write_canonical_policy_pack(acceptance_policy, payload)
    digest = _sha256(sealed)
    (pack / "checksums.sha256").write_text(
        f"{digest}  {policy.POLICY_RELATIVE_PATH}\n", encoding="utf-8"
    )
    _write_json(
        pack / "manifest.json",
        {
            "verification": {"report_assurance": "strict"},
            policy.POLICY_MANIFEST_FIELD: policy.policy_manifest_entry(sealed),
        },
    )
    return pack, acceptance_policy


def _canonical_report(
    path: str, *, digest: str = "a" * 64, run_id: str | None = "run"
) -> dict[str, Any]:
    return {
        "path": path,
        "report_sha256": digest,
        "run_id": run_id,
        "report_id": run_id,
    }


def _verify_result(outcome: VerifyOutcome, payload: object) -> VerifyExecutionResult:
    return VerifyExecutionResult(outcome=outcome, payload=payload, diagnostics=())
