from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import invarlock.cli.runtime_launch_plan as runtime_launch_plan
import invarlock.runtime_security as runtime_security


def test_config_digest_and_load_runtime_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"model":"gpt2"}\n', encoding="utf-8")
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text('{"ok":true}\n', encoding="utf-8")

    digest, source = runtime_security._config_digest(config_path=config_path)
    assert digest is not None
    assert source == "file"

    digest, source = runtime_security._config_digest(
        config_payload={"model": {"id": "gpt2"}}
    )
    assert digest is not None
    assert source == "inline"

    digest, source = runtime_security._config_digest()
    assert digest is None
    assert source == "missing"

    result = runtime_security.load_runtime_manifest(report_path)
    assert result.path.name == runtime_security.RUNTIME_MANIFEST_FILENAME
    assert result.payload is None
    assert result.issue_code == runtime_security.RuntimeManifestLoadIssueCode.MISSING

    result.path.write_text("{invalid", encoding="utf-8")
    result = runtime_security.load_runtime_manifest(report_path)
    assert result.payload is None
    assert (
        result.issue_code == runtime_security.RuntimeManifestLoadIssueCode.INVALID_JSON
    )

    result.path.write_text('["not-a-dict"]', encoding="utf-8")
    result = runtime_security.load_runtime_manifest(report_path)
    assert result.payload is None
    assert (
        result.issue_code
        == runtime_security.RuntimeManifestLoadIssueCode.INVALID_PAYLOAD
    )

    result.path.write_text('{"ok": true}', encoding="utf-8")
    result = runtime_security.load_runtime_manifest(report_path)
    assert result.payload == {"ok": True}
    assert result.issue_code is None


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


def test_config_digest_falls_back_to_payload_when_path_is_missing() -> None:
    digest, source = runtime_security._config_digest(
        config_path="missing.yaml",
        config_payload={"model": {"id": "demo"}},
    )

    assert digest is not None
    assert source == "inline"


def test_reset_runtime_allowances_clears_scoped_policy(monkeypatch) -> None:
    monkeypatch.setenv(runtime_security.ALLOW_NETWORK_ENV, "1")
    runtime_security.reset_runtime_allowances()

    token = runtime_security.apply_runtime_allowances(allow_network=True)
    try:
        assert runtime_security.network_allowed() is True
    finally:
        runtime_security.reset_runtime_allowances(token)

    assert runtime_security.network_allowed() is True


def test_runtime_allowances_scope_restores_previous_policy(monkeypatch) -> None:
    monkeypatch.setenv(runtime_security.ALLOW_NETWORK_ENV, "1")
    runtime_security.reset_runtime_allowances()

    with runtime_security.runtime_allowances_scope(allow_network=True):
        assert runtime_security.network_allowed() is True

    assert runtime_security.network_allowed() is True


def test_apply_runtime_allowances_can_disable_prior_allowances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = runtime_security.apply_runtime_allowances(
        allow_network=True,
        allow_host_execution=True,
        allow_remote_code=True,
        allow_third_party_plugins=True,
        allow_unattested_artifacts=True,
    )

    try:
        assert runtime_security.network_allowed() is True
        assert runtime_security.host_execution_allowed() is True
        assert runtime_security.remote_code_allowed() is True
        assert runtime_security.third_party_plugins_allowed() is True
        assert runtime_security.unattested_artifacts_allowed() is True

        runtime_security.apply_runtime_allowances(
            allow_network=False,
            allow_host_execution=False,
            allow_remote_code=False,
            allow_third_party_plugins=False,
            allow_unattested_artifacts=False,
        )
        assert runtime_security.network_allowed() is False
        assert runtime_security.host_execution_allowed() is False
        assert runtime_security.remote_code_allowed() is False
        assert runtime_security.third_party_plugins_allowed() is False
        assert runtime_security.unattested_artifacts_allowed() is False
    finally:
        runtime_security.reset_runtime_allowances(token)


def test_build_runtime_security_policy_applies_request_scoped_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = runtime_security.build_runtime_security_policy(
        allow_network=True,
        allow_host_execution=True,
        allow_third_party_plugins=True,
        allow_remote_code=True,
        allow_unattested_artifacts=True,
    )

    assert policy == runtime_security.RuntimeSecurityPolicy(
        allow_network=True,
        allow_host_execution=True,
        allow_third_party_plugins=True,
        allow_remote_code=True,
        allow_unattested_artifacts=True,
    )

    monkeypatch.delenv(runtime_security.ALLOW_NETWORK_ENV, raising=False)
    monkeypatch.delenv(runtime_security.ALLOW_HOST_EXECUTION_ENV, raising=False)
    monkeypatch.delenv(runtime_security.ALLOW_REMOTE_CODE_ENV, raising=False)
    monkeypatch.delenv(runtime_security.ALLOW_THIRD_PARTY_PLUGINS_ENV, raising=False)
    monkeypatch.delenv(runtime_security.ALLOW_UNATTESTED_ARTIFACTS_ENV, raising=False)
    try:
        runtime_security.apply_runtime_allowances(policy=policy)

        assert runtime_security.network_allowed() is True
        assert runtime_security.host_execution_allowed() is True
        assert runtime_security.remote_code_allowed() is True
        assert runtime_security.third_party_plugins_allowed() is True
        assert runtime_security.unattested_artifacts_allowed() is True
    finally:
        runtime_security.reset_runtime_allowances()


def test_inspect_container_image_timeout_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def _run(command, capture_output=False, text=False, check=False, timeout=None):
        seen["timeout"] = timeout
        raise runtime_security.subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(runtime_security.subprocess, "run", _run, raising=True)

    exists, digest = runtime_security._inspect_container_image("docker", "demo:latest")

    assert exists is False
    assert digest is None
    assert seen["timeout"] == runtime_security._CONTAINER_INSPECT_TIMEOUT_SECONDS


def test_flag_occurrences_cover_split_and_inline_forms() -> None:
    argv = [
        "evaluate",
        "--out",
        "reports",
        "--baseline-report=baseline.json",
        "-c",
        "config.yaml",
        "--ignored",
        "value",
    ]

    assert runtime_launch_plan._iter_flag_occurrences(
        argv,
        flags={"--out", "--baseline-report", "-c"},
    ) == [
        (1, "--out", "reports", 2),
        (3, "--baseline-report", "baseline.json", None),
        (4, "-c", "config.yaml", 5),
    ]


def test_runtime_security_helpers_cover_empty_and_deduplicated_path_sets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert runtime_security._coerce_bool(None) is None
    assert runtime_security._iter_absolute_pythonpath_entries() == []

    cwd = tmp_path / "repo"
    parent_mount = tmp_path / "external"
    child_mount = parent_mount / "nested"
    cwd.mkdir()
    child_mount.mkdir(parents=True)

    assert runtime_security._container_pythonpath_entries(cwd=cwd) == (
        ["/workspace/src"],
        [],
    )
    assert runtime_security._minimize_mounts([child_mount, parent_mount]) == [
        parent_mount
    ]


def test_replace_flag_value_updates_split_and_inline_tokens() -> None:
    argv = ["--out", "reports", "--baseline-report=baseline.json"]

    runtime_launch_plan._replace_flag_value(
        argv,
        token_index=0,
        flag="--out",
        value_index=1,
        new_value="artifacts",
    )
    runtime_launch_plan._replace_flag_value(
        argv,
        token_index=2,
        flag="--baseline-report",
        value_index=None,
        new_value="latest.json",
    )

    assert argv == ["--out", "artifacts", "--baseline-report=latest.json"]


def test_path_helpers_resolve_workspace_and_membership(tmp_path: Path) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    nested = cwd / "nested" / "report.json"
    external = tmp_path / "outside" / "artifact.json"

    assert runtime_security._absolute_host_path("nested/report.json", cwd=cwd) == nested
    assert runtime_security._absolute_host_path(external, cwd=cwd) == external
    assert runtime_security._path_is_within(nested, cwd) is True
    assert runtime_security._path_is_within(external, cwd) is False
    assert (
        runtime_security._workspace_path(cwd / "nested", cwd=cwd) == "/workspace/nested"
    )


def test_mount_root_helpers_cover_files_directories_and_resolved_targets(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    file_path = root_dir / "weights.bin"
    file_path.write_text("payload\n", encoding="utf-8")
    link_path = tmp_path / "weights-link"
    link_path.symlink_to(file_path)

    assert runtime_security._mount_root_for_path(root_dir) == root_dir
    assert runtime_security._mount_root_for_path(file_path) == root_dir
    assert runtime_security._mount_root_for_resolved_path(link_path) == root_dir


def test_iter_external_symlink_target_mounts_ignores_in_workspace_targets(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    inside_target = cwd / "models" / "baseline"
    inside_target.parent.mkdir(parents=True)
    inside_target.write_text("ok\n", encoding="utf-8")
    link_path = cwd / "baseline-link"
    link_path.symlink_to(inside_target)

    assert (
        runtime_security._iter_external_symlink_target_mounts(link_path, cwd=cwd) == []
    )


def test_iter_external_symlink_target_mounts_finds_external_targets_recursively(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    external_root = tmp_path / "external-cache"
    external_root.mkdir()
    target = external_root / "artifact.bin"
    target.write_text("bin\n", encoding="utf-8")

    direct_link = cwd / "artifact-link"
    direct_link.symlink_to(target)
    assert runtime_security._iter_external_symlink_target_mounts(
        direct_link, cwd=cwd
    ) == [external_root]

    tree = cwd / "tree"
    nested = tree / "nested"
    nested.mkdir(parents=True)
    deep_link = nested / "deep-link"
    deep_link.symlink_to(target)
    assert (
        runtime_security._iter_external_symlink_target_mounts(
            tree, cwd=cwd, recursive=False
        )
        == []
    )
    assert runtime_security._iter_external_symlink_target_mounts(tree, cwd=cwd) == [
        external_root
    ]


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
        runtime_security._iter_external_symlink_target_mounts(link_path, cwd=cwd) == []
    )


def test_iter_absolute_pythonpath_entries_filters_relative_empty_and_duplicates(
    monkeypatch, tmp_path: Path
) -> None:
    abs_a = tmp_path / "lib-a"
    abs_b = tmp_path / "lib-b"
    abs_a.mkdir()
    abs_b.mkdir()
    monkeypatch.setenv(
        "PYTHONPATH",
        runtime_security.os.pathsep.join(
            [
                "",
                str(abs_a),
                "relative-entry",
                str(abs_a),
                str(abs_b),
            ]
        ),
    )

    assert runtime_security._iter_absolute_pythonpath_entries() == [
        abs_a.resolve(),
        abs_b.resolve(),
    ]


def test_container_pythonpath_entries_map_workspace_and_external_mounts(
    monkeypatch, tmp_path: Path
) -> None:
    cwd = tmp_path / "repo"
    inside = cwd / "src"
    external = tmp_path / "shared-lib"
    inside.mkdir(parents=True)
    external.mkdir()
    monkeypatch.setenv(
        "PYTHONPATH",
        runtime_security.os.pathsep.join([str(inside), str(external)]),
    )

    entries, mounts = runtime_security._container_pythonpath_entries(cwd=cwd)

    assert entries == ["/workspace/src", str(external.resolve())]
    assert mounts == [external]


def test_normalize_output_and_local_model_paths_cover_inside_outside_and_missing(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    inside_model = cwd / "models" / "edited"
    inside_model.parent.mkdir(parents=True)
    inside_model.write_text("weights\n", encoding="utf-8")
    external_root = tmp_path / "external-model"
    external_root.mkdir()
    missing = cwd / "missing-model"

    normalized_output, output_mounts = (
        runtime_security._normalize_output_path_for_container(
            "reports",
            cwd=cwd,
        )
    )
    assert normalized_output == "reports"
    assert output_mounts == set()

    external_output = external_root / "report.json"
    normalized_external_output, external_output_mounts = (
        runtime_security._normalize_output_path_for_container(
            str(external_output),
            cwd=cwd,
        )
    )
    assert normalized_external_output == str(external_output.resolve())
    assert external_output_mounts == {external_root}

    normalized_inside_model, inside_mounts, treated_inside = (
        runtime_security._normalize_local_model_path_for_container(
            str(inside_model),
            cwd=cwd,
        )
    )
    assert normalized_inside_model == "/workspace/models/edited"
    assert inside_mounts == set()
    assert treated_inside is True

    normalized_missing, missing_mounts, treated_missing = (
        runtime_security._normalize_local_model_path_for_container(
            str(missing),
            cwd=cwd,
        )
    )
    assert normalized_missing == str(missing)
    assert missing_mounts == set()
    assert treated_missing is False

    normalized_external_model, external_mounts, treated_external = (
        runtime_security._normalize_local_model_path_for_container(
            str(external_root),
            cwd=cwd,
        )
    )
    assert normalized_external_model == str(external_root.resolve())
    assert external_mounts == {external_root}
    assert treated_external is True


def test_normalize_config_path_for_container_scans_dependencies_and_wraps_errors(
    monkeypatch, tmp_path: Path
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    config_path = cwd / "config.yaml"
    config_path.write_text("profile: ci\n", encoding="utf-8")
    local_data = cwd / "data" / "windows.json"
    local_data.parent.mkdir(parents=True)
    local_data.write_text("[]\n", encoding="utf-8")
    external_root = tmp_path / "datasets"
    external_root.mkdir()
    external_config = external_root / "overlay.yaml"
    external_ref = external_root / "windows.bin"
    external_config.write_text("profile: ci\n", encoding="utf-8")
    external_ref.write_text("payload\n", encoding="utf-8")

    monkeypatch.setattr(
        runtime_security,
        "inspect_config_dependencies",
        lambda _path: SimpleNamespace(
            config_paths=[config_path, external_config],
            referenced_paths=[local_data, external_ref],
        ),
        raising=True,
    )

    normalized, mounts, needs_mirror = (
        runtime_security._normalize_config_path_for_container(
            "config.yaml",
            cwd=cwd,
            scan_dependencies=True,
        )
    )
    assert normalized == str(config_path.resolve())
    assert mounts == {external_root}
    assert needs_mirror is True

    monkeypatch.setattr(
        runtime_security,
        "inspect_config_dependencies",
        lambda _path: (_ for _ in ()).throw(ValueError("broken dependency scan")),
        raising=True,
    )
    with pytest.raises(RuntimeError, match="not mountable"):
        runtime_security._normalize_config_path_for_container(
            "config.yaml",
            cwd=cwd,
            scan_dependencies=True,
        )


def test_normalize_delegated_argv_rewrites_paths_and_collects_mounts(
    monkeypatch, tmp_path: Path
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    config_path = cwd / "config.yaml"
    config_path.write_text("profile: dev\n", encoding="utf-8")
    inside_model = cwd / "subject"
    inside_model.mkdir()
    external_root = tmp_path / "external-baseline"
    external_root.mkdir()

    monkeypatch.setattr(
        runtime_security,
        "inspect_config_dependencies",
        lambda _path: SimpleNamespace(config_paths=[config_path], referenced_paths=[]),
        raising=True,
    )

    plan = runtime_launch_plan.normalize_delegated_argv(
        [
            "evaluate",
            "--config",
            "config.yaml",
            "--out",
            "reports",
            "--baseline",
            str(external_root),
            "--subject",
            str(inside_model),
        ],
        cwd=cwd,
    )

    assert list(plan.argv) == [
        "evaluate",
        "--config",
        str(config_path.resolve()),
        "--out",
        "reports",
        "--baseline",
        str(external_root.resolve()),
        "--subject",
        "/workspace/subject",
    ]
    assert list(plan.argv_mounts) == [external_root]
    assert plan.needs_cwd_host_mirror is True


def test_path_env_value_and_delegated_env_pairs_translate_workspace_paths(
    monkeypatch, tmp_path: Path
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    inside_tmp = cwd / "tmp-cache"
    inside_tmp.mkdir()
    external_tmp = tmp_path / "external-tmp"
    external_tmp.mkdir()

    monkeypatch.setenv(runtime_security.ALLOW_NETWORK_ENV, "1")
    monkeypatch.setenv(runtime_security.ALLOW_REMOTE_CODE_ENV, "0")
    monkeypatch.setenv(runtime_security.ALLOW_THIRD_PARTY_PLUGINS_ENV, "1")
    monkeypatch.setenv(runtime_security.ALLOW_UNATTESTED_ARTIFACTS_ENV, "0")
    monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "audit")
    monkeypatch.setenv("INVARLOCK_EVALUATE_TMP_DIR", str(inside_tmp))
    monkeypatch.setenv("TMPDIR", str(external_tmp))
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")

    with runtime_security.runtime_allowances_scope(
        allow_network=True,
        allow_remote_code=True,
        allow_third_party_plugins=False,
        allow_unattested_artifacts=True,
    ):
        translated_inside, inside_mounts = (
            runtime_security._path_env_value_for_container(
                str(inside_tmp),
                cwd=cwd,
            )
        )
        translated_external, external_mounts = (
            runtime_security._path_env_value_for_container(
                str(external_tmp),
                cwd=cwd,
            )
        )
        assert translated_inside == "/workspace/tmp-cache"
        assert inside_mounts == []
        assert translated_external == str(external_tmp.resolve())
        assert external_mounts == [external_tmp]

        env_pairs, mounts = runtime_security._delegated_env_pairs(cwd=cwd)
        assert env_pairs[runtime_security.ALLOW_NETWORK_ENV] == "1"
        assert env_pairs[runtime_security.ALLOW_REMOTE_CODE_ENV] == "1"
        assert env_pairs[runtime_security.ALLOW_THIRD_PARTY_PLUGINS_ENV] == "0"
        assert env_pairs[runtime_security.ALLOW_UNATTESTED_ARTIFACTS_ENV] == "1"
        assert env_pairs["INVARLOCK_SNAPSHOT_MODE"] == "audit"
        assert env_pairs["INVARLOCK_EVALUATE_TMP_DIR"] == "/workspace/tmp-cache"
        assert env_pairs["TMPDIR"] == str(external_tmp.resolve())
        assert "INVARLOCK_TINY_RELAX" not in env_pairs
        assert mounts == [external_tmp]

    translated_inside, inside_mounts = runtime_security._path_env_value_for_container(
        str(inside_tmp),
        cwd=cwd,
    )
    translated_external, external_mounts = (
        runtime_security._path_env_value_for_container(
            str(external_tmp),
            cwd=cwd,
        )
    )
    assert translated_inside == "/workspace/tmp-cache"
    assert inside_mounts == []
    assert translated_external == str(external_tmp.resolve())
    assert external_mounts == [external_tmp]


def test_runtime_verifier_binary_falls_back_to_default_when_no_candidates_exist(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    module_path = repo_root / "src" / "invarlock" / "runtime_security.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text("# stub\n", encoding="utf-8")
    script_dir = tmp_path / "venv" / "bin"
    script_dir.mkdir(parents=True, exist_ok=True)
    python_bin = script_dir / "python"
    python_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    python_bin.chmod(0o755)

    monkeypatch.delenv(runtime_security.RUNTIME_VERIFIER_BINARY_ENV, raising=False)
    monkeypatch.setattr(runtime_security, "__file__", str(module_path), raising=False)
    monkeypatch.setattr(
        runtime_security.sys, "executable", str(python_bin), raising=True
    )
    monkeypatch.setattr(
        runtime_security.sys, "argv", [str(script_dir / "cli")], raising=True
    )

    assert (
        runtime_security.runtime_verifier_binary()
        == runtime_security.RUNTIME_VERIFIER_BINARY_DEFAULT
    )
