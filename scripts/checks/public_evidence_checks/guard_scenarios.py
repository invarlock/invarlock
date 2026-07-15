"""Closed validation for immutable historical guard-scenario observations.

This module deliberately does not pass old reports through the current report
verifier.  The index says that they predate the current contract, and this
checker preserves that boundary while independently rehashing the archive and
recomputing the narrow observations that the index records.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

INDEX_FILENAME = "guard_scenario_observations.json"
INDEX_FORMAT = "invarlock/guard-scenario-observations-v1"
MANIFEST_FORMAT = "invarlock.public_evidence.guard_value_manifest.v2"
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ARCHIVE_PREFIX = PurePosixPath("public_evidence/published_basis")

EXPECTED_OBSERVATIONS = {
    "spectral_moderate_scale_mlp_l31_up_s112": (
        "spectral",
        "subtle_detectable",
        "frozen_scenario_verdict",
    ),
    "spectral_moderate_scale_attn_l31_o_s105": (
        "spectral",
        "negative_control",
        "frozen_negative_control",
    ),
    "rmt_norm_noise_l31_ffn_up_b030": (
        "rmt",
        "subtle_detectable",
        "hash_bound_observation",
    ),
    "ve_mlp_scale_skew_l31_down_s090": (
        "variance",
        "subtle_detectable",
        "hash_bound_observation",
    ),
}


def _label(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _load_json_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    def _closed_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(data, object_pairs_hook=_closed_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label}: JSON root must be an object")
    return payload


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _safe_archive_path(raw: object, *, label: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label}: path must be a non-empty string")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.parts[:2] != ARCHIVE_PREFIX.parts
    ):
        raise ValueError(f"{label}: path must stay below {ARCHIVE_PREFIX.as_posix()}")
    return path.as_posix()


def _artifact_ref(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label}: artifact reference shape is invalid")
    path = _safe_archive_path(value.get("path"), label=label)
    digest = value.get("sha256")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{label}: sha256 must be a prefixed lowercase digest")
    return {"path": path, "sha256": digest}


def _validate_current_assurance(value: object, *, label: str) -> None:
    assurance = value
    if not isinstance(assurance, dict) or set(assurance) != {
        "current_report_schema_compatible",
        "current_strict_assurance",
        "reason",
    }:
        raise ValueError(f"{label}: current_assurance shape is invalid")
    if assurance.get("current_report_schema_compatible") is not False:
        raise ValueError(
            f"{label}: old reports must not claim current schema compatibility"
        )
    if assurance.get("current_strict_assurance") is not False:
        raise ValueError(
            f"{label}: old reports must not claim current strict assurance"
        )
    if not isinstance(assurance.get("reason"), str) or not assurance["reason"].strip():
        raise ValueError(f"{label}: current_assurance reason is required")


def _validate_model(value: object, *, label: str) -> None:
    model = value
    if not isinstance(model, dict) or set(model) != {"id", "revision"}:
        raise ValueError(f"{label}: model shape is invalid")
    if model.get("id") != "mistralai/Mistral-7B-v0.1":
        raise ValueError(f"{label}: unexpected historical model id")
    if (
        not isinstance(model.get("revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", model["revision"]) is None
    ):
        raise ValueError(f"{label}: model revision must be a full git digest")


def _validate_source_asset(value: object, *, label: str) -> None:
    source = value
    if not isinstance(source, dict) or set(source) != {
        "archive_root",
        "manifest",
        "name",
        "sha256",
        "size_bytes",
        "url",
    }:
        raise ValueError(f"{label}: source_asset shape is invalid")
    if source.get("archive_root") != ARCHIVE_PREFIX.as_posix():
        raise ValueError(f"{label}: source archive root is invalid")
    _artifact_ref(source.get("manifest"), label=f"{label}: source manifest")
    if not isinstance(source.get("name"), str) or not source["name"].endswith(
        ".tar.gz"
    ):
        raise ValueError(f"{label}: source asset name is invalid")
    if (
        not isinstance(source.get("sha256"), str)
        or SHA256_RE.fullmatch(source["sha256"]) is None
    ):
        raise ValueError(f"{label}: source asset sha256 is invalid")
    if (
        isinstance(source.get("size_bytes"), bool)
        or not isinstance(source.get("size_bytes"), int)
        or source["size_bytes"] <= 0
    ):
        raise ValueError(f"{label}: source asset size is invalid")
    url = source.get("url")
    if (
        not isinstance(url, str)
        or not url.startswith("https://github.com/")
        or url.rsplit("/", 1)[-1] != source["name"]
    ):
        raise ValueError(f"{label}: source asset URL is invalid")


def _validate_observations(value: object, *, label: str) -> list[dict[str, Any]]:
    observations = value
    if not isinstance(observations, list) or len(observations) != len(
        EXPECTED_OBSERVATIONS
    ):
        raise ValueError(f"{label}: exactly four historical observations are required")
    indexed: dict[str, dict[str, Any]] = {}
    for position, item in enumerate(observations):
        item_label = f"{label}: observation[{position}]"
        if not isinstance(item, dict):
            raise ValueError(f"{item_label}: must be an object")
        scenario = item.get("scenario_id")
        if not isinstance(scenario, str) or scenario in indexed:
            raise ValueError(f"{item_label}: scenario_id is missing or duplicated")
        expected = EXPECTED_OBSERVATIONS.get(scenario)
        if (
            expected is None
            or tuple(item.get(key) for key in ("guard", "intent", "artifact_class"))
            != expected
        ):
            raise ValueError(f"{item_label}: scenario classification is invalid")
        expected_fields = {
            "artifact_class",
            "guard",
            "intent",
            "primary_metric",
            "report",
            "scenario_id",
            "signal",
        }
        if expected[0] in {"rmt", "variance"}:
            expected_fields.add("sidecar")
        if expected[0] == "variance":
            expected_fields.add("baseline_sidecar")
        if set(item) != expected_fields:
            raise ValueError(f"{item_label}: observation shape is invalid")
        _artifact_ref(item.get("report"), label=f"{item_label}: report")
        if "sidecar" in item:
            _artifact_ref(item.get("sidecar"), label=f"{item_label}: sidecar")
        if "baseline_sidecar" in item:
            _artifact_ref(
                item.get("baseline_sidecar"), label=f"{item_label}: baseline sidecar"
            )
        metric = item.get("primary_metric")
        if (
            not isinstance(metric, dict)
            or set(metric) != {"acceptable", "ratio_vs_baseline"}
            or metric.get("acceptable") is not True
            or not _finite(metric.get("ratio_vs_baseline"))
        ):
            raise ValueError(f"{item_label}: primary metric observation is invalid")
        if not isinstance(item.get("signal"), dict):
            raise ValueError(f"{item_label}: signal observation is invalid")
        indexed[scenario] = item
    if set(indexed) != set(EXPECTED_OBSERVATIONS):
        raise ValueError(f"{label}: historical scenario set is incomplete")
    return [indexed[scenario] for scenario in EXPECTED_OBSERVATIONS]


def _validate_index(payload: dict[str, Any], *, label: str) -> list[dict[str, Any]]:
    if set(payload) != {
        "claim_class",
        "current_assurance",
        "format_version",
        "model",
        "observations",
        "scope",
        "source_asset",
    }:
        raise ValueError(f"{label}: top-level shape is invalid")
    if payload.get("format_version") != INDEX_FORMAT:
        raise ValueError(f"{label}: format_version must be {INDEX_FORMAT}")
    if payload.get("claim_class") != "historical_observation":
        raise ValueError(f"{label}: claim_class must be historical_observation")
    _validate_current_assurance(payload.get("current_assurance"), label=label)
    _validate_model(payload.get("model"), label=label)
    scope = payload.get("scope")
    if (
        not isinstance(scope, list)
        or not scope
        or any(not isinstance(item, str) or not item.strip() for item in scope)
    ):
        raise ValueError(f"{label}: scope must contain non-empty strings")
    _validate_source_asset(payload.get("source_asset"), label=label)
    return _validate_observations(payload.get("observations"), label=label)


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _number(value: object) -> float:
    assert not isinstance(value, bool) and isinstance(value, int | float)
    return float(value)


def _close(left: object, right: object) -> bool:
    return (
        _finite(left)
        and _finite(right)
        and math.isclose(_number(left), _number(right), rel_tol=1e-9, abs_tol=1e-12)
    )


def _read_members(archive: Path) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    try:
        with tarfile.open(archive, "r:gz") as handle:
            seen: set[str] = set()
            for member in handle:
                name = _safe_archive_path(member.name, label="archive member")
                if name in seen:
                    raise ValueError(f"archive contains duplicate member {name}")
                seen.add(name)
                if member.isdir():
                    continue
                if not member.isfile() or member.issym() or member.islnk():
                    raise ValueError(f"archive member {name} is not a regular file")
                stream = handle.extractfile(member)
                if stream is None:
                    raise ValueError(f"archive member {name} is unreadable")
                members[name] = stream.read()
    except (tarfile.TarError, OSError) as exc:
        raise ValueError(f"source asset is not a readable tar archive ({exc})") from exc
    return members


def _checked_member(
    members: dict[str, bytes], ref: dict[str, Any], *, label: str
) -> bytes:
    path = ref["path"]
    data = members.get(path)
    if data is None:
        raise ValueError(f"{label}: archive member is missing: {path}")
    if _sha256(data) != ref["sha256"]:
        raise ValueError(f"{label}: indexed sha256 does not match: {path}")
    return data


def _verify_manifest(
    members: dict[str, bytes],
    source: dict[str, Any],
    *,
    model: dict[str, Any],
    selected_scenario_ids: set[str],
) -> None:
    manifest_ref = source["manifest"]
    manifest_bytes = _checked_member(members, manifest_ref, label="source manifest")
    manifest = _load_json_bytes(manifest_bytes, label="source manifest")
    if (
        set(manifest) != {"artifact_root", "files", "schema", "source_run"}
        or manifest.get("schema") != MANIFEST_FORMAT
    ):
        raise ValueError("source manifest shape or schema is invalid")
    source_run = manifest.get("source_run")
    if not isinstance(source_run, dict):
        raise ValueError("source manifest source_run must be an object")
    if source_run.get("model_id") != model["id"]:
        raise ValueError("source manifest model_id does not match the index")
    if source_run.get("model_revision") != model["revision"]:
        raise ValueError("source manifest model_revision does not match the index")
    scenario_ids = source_run.get("scenario_ids")
    if (
        not isinstance(scenario_ids, list)
        or any(not isinstance(item, str) or not item for item in scenario_ids)
        or len(scenario_ids) != len(set(scenario_ids))
    ):
        raise ValueError("source manifest scenario_ids must be unique strings")
    missing_scenarios = sorted(selected_scenario_ids - set(scenario_ids))
    if missing_scenarios:
        raise ValueError(
            "source manifest does not bind every indexed scenario "
            f"(missing={missing_scenarios})"
        )
    artifact_root = _safe_archive_path(
        manifest.get("artifact_root"), label="manifest artifact_root"
    )
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("source manifest files must be a non-empty list")
    seen: set[str] = set()
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"source manifest file[{position}] shape is invalid")
        relative = entry.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
        ):
            raise ValueError(f"source manifest file[{position}] path is unsafe")
        full_path = (PurePosixPath(artifact_root) / relative).as_posix()
        if full_path in seen:
            raise ValueError(f"source manifest duplicates {relative}")
        seen.add(full_path)
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if (
            not isinstance(digest, str)
            or HEX_SHA256_RE.fullmatch(digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValueError(f"source manifest file[{position}] summary is invalid")
        data = members.get(full_path)
        if (
            data is None
            or len(data) != size
            or hashlib.sha256(data).hexdigest() != digest
        ):
            raise ValueError(
                f"source manifest does not match archive member {relative}"
            )
    manifest_path = source["manifest"]["path"]
    actual_payloads = {
        path
        for path in members
        if path.startswith(f"{artifact_root}/") and path != manifest_path
    }
    if actual_payloads != seen:
        extras = sorted(actual_payloads - seen)
        missing = sorted(seen - actual_payloads)
        raise ValueError(
            "source manifest is not closed over its artifact root "
            f"(unlisted={extras}, missing={missing})"
        )


def _check_report(report: dict[str, Any], observation: dict[str, Any]) -> None:
    validation = report.get("validation")
    metric = report.get("primary_metric")
    if (
        not isinstance(validation, dict)
        or validation.get("primary_metric_acceptable") is not True
    ):
        raise ValueError(
            f"{observation['scenario_id']}: report did not record a PM pass"
        )
    if not isinstance(metric, dict) or not _close(
        metric.get("ratio_vs_baseline"),
        observation["primary_metric"]["ratio_vs_baseline"],
    ):
        raise ValueError(
            f"{observation['scenario_id']}: PM ratio does not match the index"
        )


def _spectral_signal(report: dict[str, Any]) -> tuple[int, set[str]]:
    spectral = report.get("spectral")
    if (
        not isinstance(spectral, dict)
        or isinstance(spectral.get("caps_applied"), bool)
        or not isinstance(spectral.get("caps_applied"), int)
    ):
        raise ValueError("spectral report does not contain a cap count")
    modules = {
        item["module"]
        for item in spectral.get("violations", [])
        if isinstance(item, dict)
        and item.get("type") == "family_z_cap"
        and item.get("selected") is True
        and isinstance(item.get("module"), str)
    }
    return spectral["caps_applied"], modules


def _load_observation_reports(
    observations: list[dict[str, Any]], members: dict[str, bytes]
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for observation in observations:
        scenario_id = observation["scenario_id"]
        report = _load_json_bytes(
            _checked_member(members, observation["report"], label=scenario_id),
            label=scenario_id,
        )
        _check_report(report, observation)
        reports[scenario_id] = report
    return reports


def _verify_spectral_observations(
    positive: dict[str, Any],
    negative: dict[str, Any],
    reports: dict[str, dict[str, Any]],
) -> None:
    positive_caps, positive_modules = _spectral_signal(reports[positive["scenario_id"]])
    negative_caps, negative_modules = _spectral_signal(reports[negative["scenario_id"]])
    for observation, caps in ((positive, positive_caps), (negative, negative_caps)):
        signal = observation["signal"]
        if set(signal) != {
            "baseline_caps_applied",
            "delta_caps_applied",
            "new_cap_modules",
            "subject_caps_applied",
        }:
            raise ValueError(
                f"{observation['scenario_id']}: spectral signal shape is invalid"
            )
        if (
            signal["subject_caps_applied"] != caps
            or signal["subject_caps_applied"] - signal["baseline_caps_applied"]
            != signal["delta_caps_applied"]
        ):
            raise ValueError(
                f"{observation['scenario_id']}: spectral cap arithmetic is invalid"
            )
    expected_new = set(positive["signal"]["new_cap_modules"])
    if (
        positive_modules - negative_modules != expected_new
        or len(expected_new) != positive["signal"]["delta_caps_applied"]
    ):
        raise ValueError("spectral positive observation is not baseline-relative")
    if (
        negative["signal"]["delta_caps_applied"] != 0
        or negative["signal"]["new_cap_modules"]
        or positive["signal"]["baseline_caps_applied"] != negative_caps
    ):
        raise ValueError(
            "spectral negative control records a positive baseline-relative signal"
        )


def _verify_rmt_observation(
    observation: dict[str, Any], members: dict[str, bytes]
) -> None:
    signal = observation["signal"]
    if set(signal) != {"epsilon_violations_min", "stable"}:
        raise ValueError("RMT indexed signal shape is invalid")
    if (
        signal.get("stable") is not False
        or isinstance(signal.get("epsilon_violations_min"), bool)
        or not isinstance(signal.get("epsilon_violations_min"), int)
        or signal["epsilon_violations_min"] < 1
    ):
        raise ValueError("RMT indexed signal values are invalid")
    scenario_id = observation["scenario_id"]
    rmt_probe = _load_json_bytes(
        _checked_member(members, observation["sidecar"], label=scenario_id),
        label=scenario_id,
    )
    violations = rmt_probe.get("epsilon_violations")
    if (
        rmt_probe.get("stable") is not signal["stable"]
        or not isinstance(violations, list)
        or len(violations) < signal["epsilon_violations_min"]
    ):
        raise ValueError("RMT observation does not contain the indexed unstable signal")
    for violation in violations:
        if not isinstance(violation, dict):
            raise ValueError("RMT epsilon violation is invalid")
        base, current, epsilon = (
            violation.get("edge_base"),
            violation.get("edge_cur"),
            violation.get("epsilon"),
        )
        if not (
            _finite(base)
            and _number(base) > 0
            and _finite(current)
            and _finite(epsilon)
            and _number(epsilon) >= 0
        ):
            raise ValueError("RMT epsilon violation numbers are invalid")
        if (
            not _close(
                violation.get("allowed"), _number(base) * (1.0 + _number(epsilon))
            )
            or not _close(
                violation.get("delta"), _number(current) / _number(base) - 1.0
            )
            or _number(current) <= _number(violation["allowed"])
        ):
            raise ValueError("RMT epsilon violation arithmetic is invalid")


def _verify_ve_gain(payload: dict[str, Any], *, role: str) -> None:
    ppl_no_ve = payload.get("ppl_no_ve")
    ppl_with_ve = payload.get("ppl_with_ve")
    if not (
        _finite(ppl_no_ve)
        and _number(ppl_no_ve) > 0
        and _finite(ppl_with_ve)
        and _number(ppl_with_ve) > 0
    ):
        raise ValueError(f"VE {role} perplexities must be finite and positive")
    abs_improvement = _number(ppl_no_ve) - _number(ppl_with_ve)
    ab_gain = abs_improvement / _number(ppl_no_ve)
    if not _close(payload.get("abs_improvement"), abs_improvement) or not _close(
        payload.get("ab_gain"), ab_gain
    ):
        raise ValueError(f"VE {role} gain arithmetic is invalid")


def _verify_ve_observation(
    observation: dict[str, Any], members: dict[str, bytes]
) -> None:
    signal = observation["signal"]
    if set(signal) != {"baseline_signal", "subject_signal"}:
        raise ValueError("VE indexed signal shape is invalid")
    if (
        signal.get("subject_signal") is not True
        or signal.get("baseline_signal") is not False
    ):
        raise ValueError("VE indexed signal values are invalid")
    scenario_id = observation["scenario_id"]
    subject = _load_json_bytes(
        _checked_member(members, observation["sidecar"], label=scenario_id),
        label=scenario_id,
    )
    baseline_label = f"{scenario_id} baseline"
    baseline = _load_json_bytes(
        _checked_member(members, observation["baseline_sidecar"], label=baseline_label),
        label=baseline_label,
    )
    if (
        subject.get("signal") is not signal["subject_signal"]
        or baseline.get("signal") is not signal["baseline_signal"]
    ):
        raise ValueError("VE observation is not a true subject-vs-baseline signal")
    _verify_ve_gain(subject, role="subject")
    _verify_ve_gain(baseline, role="baseline")
    if not (
        _number(subject["ab_gain"]) > 0 and _number(subject["abs_improvement"]) > 0
    ):
        raise ValueError("VE subject signal lacks positive measured gain")
    if not (
        _number(baseline["ab_gain"]) < 0 and _number(baseline["abs_improvement"]) < 0
    ):
        raise ValueError("VE baseline does not preserve the observed absence of signal")


def _verify_semantics(
    observations: list[dict[str, Any]], members: dict[str, bytes]
) -> None:
    reports = _load_observation_reports(observations, members)
    _verify_spectral_observations(observations[0], observations[1], reports)
    _verify_rmt_observation(observations[2], members)
    _verify_ve_observation(observations[3], members)


def check_historical_guard_scenario_observations(
    errors: list[str],
    root: Path,
    *,
    fetch_external_assets: bool = False,
    index_path: Path | None = None,
    asset_path: Path | None = None,
) -> bool:
    """Validate the local index and, when requested, independently replay it."""

    path = index_path or root / INDEX_FILENAME
    label = _label(path, root)
    try:
        payload = _load_json_bytes(path.read_bytes(), label=label)
        observations = _validate_index(payload, label=label)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        return False
    if not fetch_external_assets and asset_path is None:
        return True

    source = payload["source_asset"]
    temporary: Any | None = None
    try:
        archive = asset_path
        if archive is None:
            temporary = tempfile.NamedTemporaryFile(
                prefix="invarlock-guard-observations-", suffix=".tar.gz"
            )
            with urllib.request.urlopen(source["url"], timeout=60) as response:  # noqa: S310 - closed HTTPS URL validated above
                temporary.write(response.read())
                temporary.flush()
            archive = Path(temporary.name)
        data = archive.read_bytes()
        if len(data) != source["size_bytes"] or _sha256(data) != source["sha256"]:
            raise ValueError(
                "historical guard source asset size or sha256 does not match the index"
            )
        members = _read_members(archive)
        _verify_manifest(
            members,
            source,
            model=payload["model"],
            selected_scenario_ids={item["scenario_id"] for item in observations},
        )
        _verify_semantics(observations, members)
    except (OSError, ValueError) as exc:
        errors.append(f"{label}: {exc}")
        return False
    finally:
        if temporary is not None:
            temporary.close()
    return True


__all__ = ["check_historical_guard_scenario_observations"]
