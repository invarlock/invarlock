from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


def _helpers():
    from invarlock import runtime_security_helpers as helpers

    return helpers


def write_runtime_manifest(
    report_path: str | os.PathLike[str],
    *,
    config_path: str | os.PathLike[str] | None = None,
    config_payload: Any | None = None,
    extra: dict[str, Any] | None = None,
    execution: Any | None = None,
) -> Path:
    helpers = _helpers()
    report = Path(report_path).resolve()
    digest, digest_source = helpers._config_digest(
        config_path=config_path, config_payload=config_payload
    )
    runtime_execution = execution or helpers.RuntimeManifestExecution(
        execution_mode=helpers.current_execution_mode(),
        container_execution=helpers.running_inside_container(),
        image_ref=helpers.resolve_runtime_image(),
        image_digest=helpers.resolve_runtime_image_digest(),
        allow_network=helpers.network_allowed(),
        allow_remote_code=helpers.remote_code_allowed(),
        allow_third_party_plugins=helpers.third_party_plugins_allowed(),
    )
    manifest: dict[str, Any] = {
        "manifest_version": helpers.RUNTIME_MANIFEST_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "verifier_contract_version": helpers.RUNTIME_VERIFIER_CONTRACT_VERSION,
        "report": {
            "path": str(report),
            "filename": report.name,
            "sha256": helpers._sha256_path(report),
        },
        "config": {
            "path": str(Path(config_path).resolve())
            if config_path is not None
            else None,
            "sha256": digest,
            "source": digest_source,
        },
        "execution_mode": runtime_execution.execution_mode,
        "runtime": {
            "image_ref": helpers._attested_runtime_image_ref(
                runtime_execution.image_ref,
                runtime_execution.image_digest,
            ),
            "image_digest": runtime_execution.image_digest,
            "container_execution": runtime_execution.container_execution,
            "allow_network": runtime_execution.allow_network,
            "allow_remote_code": runtime_execution.allow_remote_code,
            "allow_third_party_plugins": (runtime_execution.allow_third_party_plugins),
        },
    }
    if isinstance(extra, dict) and extra:
        manifest["context"] = helpers._json_safe(extra)
    manifest_path = report.parent / str(helpers.RUNTIME_MANIFEST_FILENAME)
    manifest_path.write_text(
        json.dumps(helpers._json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_runtime_manifest(
    report_path: str | os.PathLike[str],
) -> Any:
    helpers = _helpers()
    report = Path(report_path)
    result_type = cast("type[Any]", helpers.RuntimeManifestLoadResult)
    manifest_path = report.parent / str(helpers.RUNTIME_MANIFEST_FILENAME)
    if not manifest_path.exists():
        return result_type(
            path=manifest_path,
            payload=None,
            issue_code=helpers.RuntimeManifestLoadIssueCode.MISSING,
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError:
        return result_type(
            path=manifest_path,
            payload=None,
            issue_code=helpers.RuntimeManifestLoadIssueCode.READ_FAILED,
            issue_message=f"unable to read {manifest_path.name}",
        )
    except json.JSONDecodeError:
        return result_type(
            path=manifest_path,
            payload=None,
            issue_code=helpers.RuntimeManifestLoadIssueCode.INVALID_JSON,
            issue_message=f"{manifest_path.name} is not valid JSON",
        )
    if not isinstance(payload, dict):
        return result_type(
            path=manifest_path,
            payload=None,
            issue_code=helpers.RuntimeManifestLoadIssueCode.INVALID_PAYLOAD,
            issue_message=f"{manifest_path.name} must decode to a JSON object",
        )
    return result_type(path=manifest_path, payload=payload)
