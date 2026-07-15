from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.release import release_preflight as preflight


def _config(tmp_path: Path) -> preflight.ReleasePreflightConfig:
    root = tmp_path / "checkout"
    root.mkdir(exist_ok=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname='invarlock'\nversion='1.2.3'\n", encoding="utf-8"
    )
    return preflight.ReleasePreflightConfig(
        repo_root=root,
        release_sha="a" * 40,
        expected_version="1.2.3",
        dist_dir=root / "dist",
        hash_manifest=root / "hashes.txt",
    )


def _inventory() -> dict[str, object]:
    return {
        "category": "runtime-providers",
        "items": [
            {
                "name": name,
                "entry_point": name,
                "entry_point_group": "invarlock.runtime_providers",
                "kind": "runtime_provider",
                "origin": "builtin",
                "status": "ready",
                **expected,
            }
            for name, expected in preflight._FIRST_PARTY_RUNTIME_PROVIDERS.items()
        ],
    }


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_git_output_returns_stripped_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, " value \n", ""),
    )
    assert preflight._git_output(tmp_path, "rev-parse", "HEAD") == "value"


def test_dependency_bridge_rejects_missing_parent_and_child_sites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight.site, "getsitepackages", lambda: [])
    with pytest.raises(
        preflight.ReleasePreflightError, match="environment is unavailable"
    ):
        preflight._install_isolated_dependency_bridge(tmp_path / "environment")

    parent = tmp_path / "locked" / "site-packages"
    parent.mkdir(parents=True)
    monkeypatch.setattr(preflight.site, "getsitepackages", lambda: [str(parent)])
    with pytest.raises(
        preflight.ReleasePreflightError, match="site-packages directory"
    ):
        preflight._install_isolated_dependency_bridge(tmp_path / "environment")


def test_dependency_bridge_normalizes_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "locked" / "site-packages"
    parent.mkdir(parents=True)
    environment = tmp_path / "environment"
    child = (
        environment
        / "lib"
        / f"python{preflight.sys.version_info.major}.{preflight.sys.version_info.minor}"
        / "site-packages"
    )
    child.mkdir(parents=True)
    monkeypatch.setattr(preflight.site, "getsitepackages", lambda: [str(parent)])

    original = Path.write_text

    def fail_write(self: Path, *args: object, **kwargs: object) -> int:
        if self.name == "invarlock-release-dependencies.pth":
            raise OSError("disk failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(preflight.ReleasePreflightError, match="unable to bind"):
        preflight._install_isolated_dependency_bridge(environment)


def test_installed_wheel_command_requires_zero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed = subprocess.CompletedProcess([], 0, "ok", "")
    monkeypatch.setattr(
        preflight, "_run_isolated_wheel_command", lambda *_args, **_kwargs: completed
    )
    assert (
        preflight._require_successful_installed_wheel_command(
            ["candidate"], cwd=tmp_path, label="probe"
        )
        is completed
    )

    monkeypatch.setattr(
        preflight,
        "_run_isolated_wheel_command",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", "private"),
    )
    with pytest.raises(preflight.ReleasePreflightError, match="probe failed"):
        preflight._require_successful_installed_wheel_command(
            ["candidate"], cwd=tmp_path, label="probe"
        )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("not-json", "did not return JSON"),
        ("[]", "invalid payload"),
        ('{"category":"runtime-providers","items":{}}', "omitted provider items"),
        (
            '{"category":"runtime-providers","items":[]}',
            "exactly the three",
        ),
    ],
)
def test_runtime_provider_inventory_rejects_malformed_shapes(
    payload: str, expected: str
) -> None:
    with pytest.raises(preflight.ReleasePreflightError, match=expected):
        preflight._validate_runtime_provider_inventory(payload)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"not-json", "unreadable"),
        (b"[]", "not a JSON object"),
        (b'{"value": 1}', "not canonical JSON"),
    ],
)
def test_canonical_json_file_rejects_unreadable_nonobject_and_pretty_json(
    tmp_path: Path, payload: bytes, expected: str
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(payload)
    with pytest.raises(preflight.ReleasePreflightError, match=expected):
        preflight._require_canonical_json_file(artifact, label="artifact")


def test_canonical_json_file_accepts_exact_encoding(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(_canonical({"ok": True}))
    assert preflight._require_canonical_json_file(artifact, label="artifact") == {
        "ok": True
    }


@pytest.mark.parametrize("failure", ["schedule", "policy"])
def test_installed_cli_smoke_rejects_invalid_generated_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    schedule = tmp_path / "runtime-schedule.json"

    def command(
        invoked: list[str], *, cwd: Path, label: str, timeout: int = 60
    ) -> subprocess.CompletedProcess[str]:
        del cwd, label, timeout
        if invoked[-1] == "--help":
            return subprocess.CompletedProcess(
                invoked,
                0,
                "build-schedule prepare-binding build-policy run-side verify-pair",
                "",
            )
        if "runtime-providers" in invoked:
            return subprocess.CompletedProcess(invoked, 0, json.dumps(_inventory()), "")
        output = Path(invoked[invoked.index("--out") + 1])
        if "build-schedule" in invoked:
            value = {
                "format_version": (
                    "wrong"
                    if failure == "schedule"
                    else "invarlock/runtime-behavioral-schedule-v1"
                ),
                "records": [{"record_id": "release-smoke-1"}],
            }
        else:
            value = {
                "format": "wrong" if failure == "policy" else "policy-pack-v3",
                "behavioral_claim": {
                    "schedule_sha256": preflight.hashlib.sha256(
                        schedule.read_bytes()
                    ).hexdigest()
                },
            }
        output.write_bytes(_canonical(value))
        return subprocess.CompletedProcess(invoked, 0, "{}", "")

    monkeypatch.setattr(
        preflight, "_require_successful_installed_wheel_command", command
    )
    with pytest.raises(preflight.ReleasePreflightError, match=f"invalid {failure}"):
        preflight._smoke_installed_wheel_cli(tmp_path / "invarlock", cwd=tmp_path)


def test_wheel_probe_rejects_checkout_local_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    inside = config.repo_root / "temporary"
    inside.mkdir()

    class TemporaryDirectory:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> str:
            return str(inside)

        def __exit__(self, *_args: object) -> None:
            pass

    monkeypatch.setattr(preflight.tempfile, "TemporaryDirectory", TemporaryDirectory)
    with pytest.raises(preflight.ReleasePreflightError, match="outside checkout"):
        preflight._probe_installed_wheel(config, tmp_path / "candidate.whl")


def test_wheel_probe_rejects_import_outside_created_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    completed = subprocess.CompletedProcess([], 0, "{}", "")
    imported = preflight.InstalledWheelImport(
        module_file=config.repo_root / "elsewhere" / "invarlock" / "__init__.py",
        module_version="1.2.3",
        distribution_name="invarlock",
        distribution_version="1.2.3",
        distribution_root=config.repo_root / "elsewhere",
    )
    monkeypatch.setattr(
        preflight, "_run_isolated_wheel_command", lambda *_args, **_kwargs: completed
    )
    monkeypatch.setattr(
        preflight, "_install_isolated_dependency_bridge", lambda _path: None
    )
    monkeypatch.setattr(preflight, "_require_executable_file", lambda *_args: None)
    monkeypatch.setattr(preflight, "_parse_import_probe", lambda _payload: imported)

    with pytest.raises(preflight.ReleasePreflightError, match="isolated environment"):
        preflight._probe_installed_wheel(config, tmp_path / "candidate.whl")
