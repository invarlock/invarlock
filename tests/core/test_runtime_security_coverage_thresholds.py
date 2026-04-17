from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import invarlock.runtime_security as runtime_security
import invarlock.runtime_security_helpers as runtime_security_helpers


def _plan(
    argv: list[str],
    *,
    needs_mirror: bool = False,
) -> runtime_security.ContainerLaunchPlan:
    return runtime_security.ContainerLaunchPlan(
        argv=tuple(argv),
        argv_mounts=(),
        needs_cwd_host_mirror=needs_mirror,
        gpu_passthrough=False,
    )


def test_runtime_flag_value_preserves_unknown_env_under_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_UNKNOWN_FLAG", "from-env")

    with runtime_security.runtime_allowances_scope(allow_network=True):
        assert (
            runtime_security_helpers._runtime_flag_value("INVARLOCK_UNKNOWN_FLAG")
            == "from-env"
        )


def test_runtime_provenance_image_ref_keeps_digest_and_allows_unverified_provenance_override() -> (
    None
):
    digest_pinned = "ghcr.io/invarlock/runtime:test@sha256:" + ("a" * 64)
    assert (
        runtime_security_helpers._runtime_provenance_image_ref(digest_pinned, None)
        == digest_pinned
    )

    with runtime_security.runtime_allowances_scope(allow_unverified_provenance=True):
        assert (
            runtime_security_helpers._runtime_provenance_image_ref(
                "ghcr.io/invarlock/runtime:test",
                None,
            )
            == "ghcr.io/invarlock/runtime:test"
        )


def test_inspect_container_image_parses_repo_digest_and_image_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_security_helpers.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='["ghcr.io/invarlock/invarlock-runtime:test@sha256:abc"]\nsha256:def\n',
        ),
        raising=True,
    )

    assert runtime_security_helpers._inspect_container_image("docker", "img") == (
        True,
        "sha256:abc",
    )

    monkeypatch.setattr(
        runtime_security_helpers.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="not-json\nsha256:def\n",
        ),
        raising=True,
    )

    assert runtime_security_helpers._inspect_container_image("docker", "img") == (
        True,
        "sha256:def",
    )


def test_inspect_container_image_handles_digestless_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_security_helpers.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="[]\nimage-id\n"),
        raising=True,
    )

    assert runtime_security_helpers._inspect_container_image("docker", "img") == (
        True,
        None,
    )


def test_inspect_container_image_handles_empty_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_security_helpers.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=""),
        raising=True,
    )

    assert runtime_security_helpers._inspect_container_image("docker", "img") == (
        True,
        None,
    )


def test_inspect_container_image_ignores_repo_digests_without_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_security_helpers.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='["ghcr.io/invarlock/runtime:test"]\nimage-id\n',
        ),
        raising=True,
    )

    assert runtime_security_helpers._inspect_container_image("docker", "img") == (
        True,
        None,
    )


def test_build_container_python_command_adds_cwd_host_mirror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    script_path = repo_root / "scripts" / "run.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("# stub\n", encoding="utf-8")

    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image",
        lambda: "ghcr.io/invarlock/runtime:test",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image_digest",
        lambda: "sha256:abc",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "container_image_available_locally",
        lambda image, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "network_allowed",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_container_pythonpath_entries",
        lambda *, cwd: ([], []),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_delegated_env_pairs",
        lambda *, cwd: ({}, []),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_host_nvidia_visible",
        lambda: False,
        raising=True,
    )

    command = runtime_security.build_container_python_command(
        script_path,
        _plan(["--help"], needs_mirror=True),
    )

    cwd = str(repo_root.resolve())
    assert f"{cwd}:{cwd}" in command


def test_delegate_python_script_to_container_surfaces_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_security_helpers,
        "build_container_python_command",
        lambda script_path, plan: ["docker", "run", "python", str(script_path)],
        raising=True,
    )

    def _run(command, check=False, timeout=None):
        raise runtime_security_helpers.subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(runtime_security_helpers.subprocess, "run", _run, raising=True)

    with pytest.raises(RuntimeError, match="timed out"):
        runtime_security.delegate_python_script_to_container(
            "scripts/evidence_packs/python/run_from_config.py",
            _plan(["--config", "demo.yaml"]),
        )


def test_build_container_command_skips_network_none_when_network_allowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image",
        lambda: "ghcr.io/invarlock/runtime:test",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image_digest",
        lambda: "sha256:container",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "network_allowed",
        lambda: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_container_pythonpath_entries",
        lambda *, cwd: ([], []),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_delegated_env_pairs",
        lambda *, cwd: ({}, []),
        raising=True,
    )

    command = runtime_security.build_container_command(_plan(["evaluate", "--help"]))

    assert "--network" not in command


def test_load_runtime_manifest_reports_read_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text('{"ok": true}\n', encoding="utf-8")
    manifest_path = report_path.parent / runtime_security.RUNTIME_MANIFEST_FILENAME
    manifest_path.write_text('{"ok": true}\n', encoding="utf-8")

    original_read_text = Path.read_text

    def _read_text(self: Path, *args, **kwargs) -> str:
        if self == manifest_path:
            raise OSError("boom")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _read_text, raising=True)

    result = runtime_security.load_runtime_manifest(report_path)

    assert result.payload is None
    assert (
        result.issue_code == runtime_security.RuntimeManifestLoadIssueCode.READ_FAILED
    )


def test_iter_external_symlink_target_mounts_skips_targets_already_covered_by_cwd(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    covered_target = cwd / "models" / "baseline.bin"
    covered_target.parent.mkdir(parents=True)
    covered_target.write_text("ok\n", encoding="utf-8")

    external_root = tmp_path / "external-links"
    external_root.mkdir()
    link_path = external_root / "baseline-link"
    link_path.symlink_to(covered_target)

    assert (
        runtime_security_helpers._iter_external_symlink_target_mounts(
            link_path, cwd=cwd
        )
        == []
    )
