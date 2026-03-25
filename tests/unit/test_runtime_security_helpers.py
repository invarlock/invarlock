from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import invarlock.runtime_security as runtime_security


def test_runtime_bool_helpers_and_execution_mode(monkeypatch) -> None:
    monkeypatch.setenv(runtime_security.ALLOW_NETWORK_ENV, "1")
    monkeypatch.setenv(runtime_security.ALLOW_HOST_EXECUTION_ENV, "0")
    monkeypatch.setenv(runtime_security.ALLOW_REMOTE_CODE_ENV, "yes")
    monkeypatch.setenv(runtime_security.ALLOW_UNATTESTED_ARTIFACTS_ENV, "true")
    monkeypatch.setenv(runtime_security.ALLOW_THIRD_PARTY_PLUGINS_ENV, "1")
    monkeypatch.setenv(runtime_security.CONTAINER_EXECUTION_ENV, "1")

    assert runtime_security._coerce_bool("on") is True
    assert runtime_security._coerce_bool("off") is False
    assert runtime_security._coerce_bool("maybe") is None
    assert runtime_security.network_allowed() is True
    assert runtime_security.host_execution_allowed() is False
    assert runtime_security.remote_code_allowed() is True
    assert runtime_security.unattested_artifacts_allowed() is True
    assert runtime_security.third_party_plugins_allowed() is True
    assert runtime_security.running_inside_container() is True
    assert runtime_security.current_execution_mode() == "container"


def test_serialize_canonical_json_normalizes_supported_types() -> None:
    OptionInfo = type("OptionInfo", (), {})
    option = OptionInfo()
    option.default = Path("payload.json")

    payload = {
        "path": Path("artifact.txt"),
        "values": {3, 1},
        "nested": [Path("nested.txt"), option, SimpleNamespace(answer=42)],
    }

    encoded = runtime_security.serialize_canonical_json(payload)
    decoded = json.loads(encoded)

    assert decoded["path"] == "artifact.txt"
    assert sorted(decoded["values"]) == [1, 3]
    assert decoded["nested"][0] == "nested.txt"
    assert decoded["nested"][1] == "payload.json"
    assert decoded["nested"][2].startswith("namespace(")


def test_resolve_runtime_image_digest_prefers_explicit_and_embedded_digest(
    monkeypatch,
) -> None:
    monkeypatch.setenv(runtime_security.RUNTIME_IMAGE_DIGEST_ENV, "sha256:explicit")
    assert runtime_security.resolve_runtime_image_digest() == "sha256:explicit"

    monkeypatch.delenv(runtime_security.RUNTIME_IMAGE_DIGEST_ENV, raising=False)
    monkeypatch.setenv(
        runtime_security.RUNTIME_IMAGE_ENV,
        "ghcr.io/invarlock/invarlock-runtime:test@sha256:embedded",
    )
    assert runtime_security.resolve_runtime_image_digest() == "sha256:embedded"


def test_resolve_runtime_image_digest_uses_inspection_when_needed(monkeypatch) -> None:
    monkeypatch.delenv(runtime_security.RUNTIME_IMAGE_DIGEST_ENV, raising=False)
    monkeypatch.delenv(runtime_security.RUNTIME_IMAGE_ENV, raising=False)
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "ghcr.io/invarlock/invarlock-runtime:test",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "_inspect_container_image",
        lambda engine, image: (True, "sha256:inspected"),
        raising=True,
    )

    assert runtime_security.resolve_runtime_image_digest() == "sha256:inspected"


def test_resolve_runtime_image_prefers_explicit_local_and_default(monkeypatch) -> None:
    monkeypatch.setenv(
        runtime_security.RUNTIME_IMAGE_ENV,
        "ghcr.io/invarlock/invarlock-runtime:explicit",
    )
    assert (
        runtime_security.resolve_runtime_image()
        == "ghcr.io/invarlock/invarlock-runtime:explicit"
    )

    monkeypatch.delenv(runtime_security.RUNTIME_IMAGE_ENV, raising=False)
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image, engine=None: image == runtime_security.RUNTIME_IMAGE_LOCAL_DEFAULT,
        raising=True,
    )
    assert (
        runtime_security.resolve_runtime_image()
        == runtime_security.RUNTIME_IMAGE_LOCAL_DEFAULT
    )

    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image, engine=None: False,
        raising=True,
    )
    assert runtime_security.resolve_runtime_image() == runtime_security.RUNTIME_IMAGE_DEFAULT


def test_inspect_container_image_parses_repo_digest_and_image_id(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_security.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='["ghcr.io/invarlock/invarlock-runtime:test@sha256:abc"]\nsha256:def\n',
        ),
        raising=True,
    )
    assert runtime_security._inspect_container_image("docker", "img") == (
        True,
        "sha256:abc",
    )

    monkeypatch.setattr(
        runtime_security.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="not-json\nsha256:def\n",
        ),
        raising=True,
    )
    assert runtime_security._inspect_container_image("docker", "img") == (
        True,
        "sha256:def",
    )


def test_inspect_container_image_handles_failures_and_digestless_images(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_security.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2, stdout=""),
        raising=True,
    )
    assert runtime_security._inspect_container_image("docker", "img") == (False, None)

    monkeypatch.setattr(
        runtime_security.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="[]\nimage-id\n"),
        raising=True,
    )
    assert runtime_security._inspect_container_image("docker", "img") == (True, None)


def test_container_engine_and_device_helpers(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_security.shutil,
        "which",
        lambda name: "/usr/bin/podman" if name == "podman" else None,
        raising=True,
    )
    assert runtime_security.resolve_container_engine() == "podman"

    assert runtime_security._requested_device(["evaluate"]) == "auto"
    assert runtime_security._requested_device(["run"]) == "auto"
    assert runtime_security._requested_device(["verify"]) is None
    assert runtime_security._requested_device(["evaluate", "--device", "CUDA"]) == "cuda"
    assert runtime_security._requested_device(["evaluate", "--device"]) is None

    monkeypatch.setattr(
        runtime_security,
        "_host_nvidia_visible",
        lambda: True,
        raising=True,
    )
    assert runtime_security._needs_gpu_passthrough(["evaluate"]) is True
    assert runtime_security._needs_gpu_passthrough(["evaluate", "--device", "cpu"]) is False

    monkeypatch.setattr(
        runtime_security,
        "_host_nvidia_visible",
        lambda: False,
        raising=True,
    )
    assert runtime_security._needs_gpu_passthrough(["evaluate", "--device", "cuda"]) is False


def test_container_image_available_locally_and_runtime_verifier_binary(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: None,
        raising=True,
    )
    assert runtime_security.container_image_available_locally("img") is False

    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "_inspect_container_image",
        lambda engine, image: (True, None),
        raising=True,
    )
    assert runtime_security.container_image_available_locally("img") is True

    verifier = tmp_path / runtime_security.RUNTIME_VERIFIER_BINARY_DEFAULT
    verifier.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv(runtime_security.RUNTIME_VERIFIER_BINARY_ENV, str(verifier))
    assert runtime_security.runtime_verifier_binary() == str(verifier)


def test_runtime_verifier_binary_finds_repo_and_script_dir_candidates(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    module_path = repo_root / "src" / "invarlock" / "runtime_security.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text("# stub\n", encoding="utf-8")
    debug_binary = (
        repo_root / "target" / "debug" / runtime_security.RUNTIME_VERIFIER_BINARY_DEFAULT
    )
    debug_binary.parent.mkdir(parents=True, exist_ok=True)
    debug_binary.write_text("#!/bin/sh\n", encoding="utf-8")
    debug_binary.chmod(0o755)

    monkeypatch.delenv(runtime_security.RUNTIME_VERIFIER_BINARY_ENV, raising=False)
    monkeypatch.setattr(runtime_security, "__file__", str(module_path), raising=False)
    assert runtime_security.runtime_verifier_binary() == str(debug_binary)

    debug_binary.unlink()
    script_dir = tmp_path / "venv" / "bin"
    script_dir.mkdir(parents=True, exist_ok=True)
    python_bin = script_dir / "python"
    python_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    python_bin.chmod(0o755)
    script_binary = script_dir / runtime_security.RUNTIME_VERIFIER_BINARY_DEFAULT
    script_binary.write_text("#!/bin/sh\n", encoding="utf-8")
    script_binary.chmod(0o755)
    monkeypatch.setattr(runtime_security.sys, "executable", str(python_bin), raising=True)
    monkeypatch.setattr(runtime_security.sys, "argv", [str(script_dir / "cli")], raising=True)
    assert runtime_security.runtime_verifier_binary() == str(script_binary)


def test_apply_runtime_allowances_and_delegate_current_process(monkeypatch) -> None:
    seen: list[bool] = []

    monkeypatch.setattr(
        "invarlock.security.enforce_network_policy",
        lambda enabled: seen.append(enabled),
        raising=False,
    )
    runtime_security.apply_runtime_allowances(
        allow_network=True,
        allow_host_execution=True,
        allow_third_party_plugins=True,
        allow_remote_code=True,
        allow_unattested_artifacts=True,
    )

    assert seen == [True]
    assert runtime_security.network_allowed() is True
    assert runtime_security.host_execution_allowed() is True
    assert runtime_security.remote_code_allowed() is True
    assert runtime_security.unattested_artifacts_allowed() is True
    assert runtime_security.third_party_plugins_allowed() is True

    monkeypatch.setattr(
        runtime_security,
        "build_container_command",
        lambda argv=None: ["docker", "run"],
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security.subprocess,
        "run",
        lambda command, check=False: SimpleNamespace(returncode=7),
        raising=True,
    )
    assert runtime_security.delegate_current_process_to_container(["evaluate"]) == 7


def test_write_runtime_manifest_records_runtime_context(
    monkeypatch, tmp_path: Path
) -> None:
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text('{"ok": true}\n', encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("profile: release\n", encoding="utf-8")

    monkeypatch.setattr(
        runtime_security,
        "current_execution_mode",
        lambda: "container",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "ghcr.io/invarlock/invarlock-runtime:test",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image_digest",
        lambda: "sha256:attested",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "running_inside_container",
        lambda: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "network_allowed",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "remote_code_allowed",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "third_party_plugins_allowed",
        lambda: False,
        raising=True,
    )

    manifest_path = runtime_security.write_runtime_manifest(
        report_path,
        config_path=config_path,
        extra={"note": "demo", "path": report_path},
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["execution_mode"] == "container"
    assert payload["runtime"]["image_ref"] == "ghcr.io/invarlock/invarlock-runtime:test"
    assert payload["runtime"]["image_digest"] == "sha256:attested"
    assert payload["runtime"]["container_execution"] is True
    assert payload["config"]["source"] == "file"
    assert payload["context"]["note"] == "demo"
    assert payload["context"]["path"] == str(report_path)


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

    manifest_path, payload = runtime_security.load_runtime_manifest(report_path)
    assert manifest_path.name == runtime_security.RUNTIME_MANIFEST_FILENAME
    assert payload is None

    manifest_path.write_text("{invalid", encoding="utf-8")
    _, payload = runtime_security.load_runtime_manifest(report_path)
    assert payload is None

    manifest_path.write_text('["not-a-dict"]', encoding="utf-8")
    _, payload = runtime_security.load_runtime_manifest(report_path)
    assert payload is None

    manifest_path.write_text('{"ok": true}', encoding="utf-8")
    _, payload = runtime_security.load_runtime_manifest(report_path)
    assert payload == {"ok": True}


def test_set_env_flag_only_writes_when_explicitly_enabled(monkeypatch) -> None:
    flag = "INVARLOCK_TEST_RUNTIME_FLAG"
    monkeypatch.delenv(flag, raising=False)

    runtime_security._set_env_flag(flag, False)
    assert runtime_security.os.environ.get(flag) is None

    runtime_security._set_env_flag(flag, None)
    assert runtime_security.os.environ.get(flag) is None

    runtime_security._set_env_flag(flag, True)
    assert runtime_security.os.environ[flag] == "1"


def test_iter_path_args_and_flag_occurrences_cover_split_and_inline_forms() -> None:
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

    assert runtime_security._iter_path_args(argv) == [
        Path("reports"),
        Path("baseline.json"),
        Path("config.yaml"),
    ]
    assert runtime_security._iter_flag_occurrences(
        argv,
        flags={"--out", "--baseline-report", "-c"},
    ) == [
        (1, "--out", "reports", 2),
        (3, "--baseline-report", "baseline.json", None),
        (4, "-c", "config.yaml", 5),
    ]


def test_replace_flag_value_updates_split_and_inline_tokens() -> None:
    argv = ["--out", "reports", "--baseline-report=baseline.json"]

    runtime_security._replace_flag_value(
        argv,
        token_index=0,
        flag="--out",
        value_index=1,
        new_value="artifacts",
    )
    runtime_security._replace_flag_value(
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
    assert runtime_security._workspace_path(cwd / "nested", cwd=cwd) == "/workspace/nested"


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

    assert runtime_security._iter_external_symlink_target_mounts(link_path, cwd=cwd) == []


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
    assert runtime_security._iter_external_symlink_target_mounts(direct_link, cwd=cwd) == [
        external_root
    ]

    tree = cwd / "tree"
    nested = tree / "nested"
    nested.mkdir(parents=True)
    deep_link = nested / "deep-link"
    deep_link.symlink_to(target)
    assert runtime_security._iter_external_symlink_target_mounts(
        tree, cwd=cwd, recursive=False
    ) == []
    assert runtime_security._iter_external_symlink_target_mounts(tree, cwd=cwd) == [
        external_root
    ]


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

    normalized_output, output_mounts = runtime_security._normalize_output_path_for_container(
        "reports",
        cwd=cwd,
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

    normalized, mounts, needs_mirror = runtime_security._normalize_config_path_for_container(
        "config.yaml",
        cwd=cwd,
        scan_dependencies=True,
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

    rewritten, mounts, needs_mirror = runtime_security._normalize_delegated_argv(
        [
            "evaluate",
            "--config",
            "config.yaml",
            "--out",
            "reports",
            "--baseline",
            str(external_root),
            "--edited",
            str(inside_model),
        ],
        cwd=cwd,
    )

    assert rewritten == [
        "evaluate",
        "--config",
        str(config_path.resolve()),
        "--out",
        "reports",
        "--baseline",
        str(external_root.resolve()),
        "--edited",
        "/workspace/subject",
    ]
    assert mounts == [external_root]
    assert needs_mirror is True


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

    translated_inside, inside_mounts = runtime_security._path_env_value_for_container(
        str(inside_tmp),
        cwd=cwd,
    )
    translated_external, external_mounts = runtime_security._path_env_value_for_container(
        str(external_tmp),
        cwd=cwd,
    )
    assert translated_inside == "/workspace/tmp-cache"
    assert inside_mounts == []
    assert translated_external == str(external_tmp.resolve())
    assert external_mounts == [external_tmp]

    env_pairs, mounts = runtime_security._delegated_env_pairs(cwd=cwd)
    assert env_pairs[runtime_security.ALLOW_NETWORK_ENV] == "1"
    assert env_pairs[runtime_security.ALLOW_REMOTE_CODE_ENV] == "0"
    assert env_pairs[runtime_security.ALLOW_THIRD_PARTY_PLUGINS_ENV] == "1"
    assert env_pairs["INVARLOCK_SNAPSHOT_MODE"] == "audit"
    assert env_pairs["INVARLOCK_EVALUATE_TMP_DIR"] == "/workspace/tmp-cache"
    assert env_pairs["TMPDIR"] == str(external_tmp.resolve())
    assert mounts == [external_tmp]


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
    monkeypatch.setattr(runtime_security.sys, "executable", str(python_bin), raising=True)
    monkeypatch.setattr(runtime_security.sys, "argv", [str(script_dir / "cli")], raising=True)

    assert (
        runtime_security.runtime_verifier_binary()
        == runtime_security.RUNTIME_VERIFIER_BINARY_DEFAULT
    )


def test_build_container_command_raises_when_no_engine_is_available(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: None,
        raising=True,
    )

    with pytest.raises(RuntimeError, match="no container engine"):
        runtime_security.build_container_command(["evaluate", "--help"])
