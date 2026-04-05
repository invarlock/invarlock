from __future__ import annotations

from pathlib import Path


def test_clusterfuzz_dockerfile_pins_base_image_digest() -> None:
    text = (Path.cwd() / ".clusterfuzzlite" / "Dockerfile").read_text(encoding="utf-8")

    assert (
        "FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:d78540f9e04918ea3c903f5280c597aa48643a6e01d2b86c38f4101807433212"
        in text
    )


def test_clusterfuzz_build_uses_hash_locked_installs() -> None:
    text = (Path.cwd() / ".clusterfuzzlite" / "build.sh").read_text(encoding="utf-8")

    assert (
        "python3 -m pip install --require-hashes -r requirements/workflows/clusterfuzzlite-py311.txt"
        in text
    )
    assert "python3 -m pip install --ignore-requires-python ." not in text
    assert (
        'python3 -m pip install --ignore-requires-python --no-deps --require-hashes -r "$wheel_requirements"'
        in text
    )
    assert "python3 -m build --wheel --no-isolation" in text


def test_clusterfuzz_wrapper_sets_contract_root_for_fuzzers() -> None:
    text = (Path.cwd() / ".clusterfuzzlite" / "build.sh").read_text(encoding="utf-8")

    assert "INVARLOCK_CONTRACTS_ROOT" in text
    assert (
        'workspace_root=\\${GITHUB_WORKSPACE:-\\$(CDPATH= cd -- "\\$this_dir/.." && pwd)}'
        in text
    )
    assert 'export INVARLOCK_CONTRACTS_ROOT="\\$workspace_root/contracts"' in text


def test_clusterfuzz_pyinstaller_bundles_contracts_for_fuzzers() -> None:
    text = (Path.cwd() / ".clusterfuzzlite" / "build.sh").read_text(encoding="utf-8")

    assert '--add-data "$SRC/invarlock/contracts:contracts"' in text


def test_clusterfuzz_requirements_are_hash_locked() -> None:
    text = (
        Path.cwd() / "requirements" / "workflows" / "clusterfuzzlite-py311.txt"
    ).read_text(encoding="utf-8")

    for package in (
        "atheris==",
        "build==",
        "jsonschema==",
        "pyinstaller==",
        "pyyaml==",
    ):
        assert package in text
    assert "--hash=sha256:" in text


def test_refresh_pinned_requirements_generates_runtime_and_clusterfuzz_locks() -> None:
    text = (Path.cwd() / "scripts" / "refresh_pinned_requirements.sh").read_text(
        encoding="utf-8"
    )

    assert (
        'compile_req_platform \\\n  "${WORKFLOW_DIR}/runtime-image.in" \\\n'
        '  "${WORKFLOW_DIR}/runtime-image-py312.txt"'
    ) in text
    assert (
        'compile_req_platform \\\n  "${WORKFLOW_DIR}/runtime-image.in" \\\n'
        '  "${WORKFLOW_DIR}/runtime-image-py312-cu128.txt"'
    ) in text
    assert (
        'compile_req_platform \\\n  "${WORKFLOW_DIR}/runtime-image.in" \\\n'
        '  "${WORKFLOW_DIR}/runtime-image-py312-aarch64.txt"'
    ) in text
    assert text.count("--torch-backend cpu") == 2
    assert text.count("--torch-backend cu128") == 1
    assert (
        'compile_req_platform \\\n  "${WORKFLOW_DIR}/clusterfuzzlite.in" \\\n'
        '  "${WORKFLOW_DIR}/clusterfuzzlite-py311.txt"'
    ) in text
