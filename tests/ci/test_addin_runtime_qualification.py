from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from invarlock import __version__ as INVARLOCK_VERSION
from invarlock.runtime_providers.tensorrt_llm_identity import (
    read_tensorrt_llm_artifact_identity,
)
from scripts import runtime_qualification

ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "b" * 40
SOURCE_BUNDLE_BYTES = b"qualification source bundle\n"
SOURCE_BUNDLE_DIGEST = "sha256:" + hashlib.sha256(SOURCE_BUNDLE_BYTES).hexdigest()
IMAGE_DIGEST = "sha256:" + ("a" * 64)
PACK_DIGEST = "sha256:" + ("b" * 64)
BASELINE_ARTIFACT = "sha256:" + ("c" * 64)
SUBJECT_ARTIFACT = "sha256:" + ("d" * 64)
POLICY_DIGEST = "sha256:" + ("e" * 64)
BASELINE_RUNTIME = "sha256:" + ("f" * 64)
SUBJECT_RUNTIME = "sha256:" + ("0" * 64)
SCHEDULE_DIGEST = "sha256:" + ("1" * 64)
SIGNER_FINGERPRINT = "sha256:" + ("2" * 64)
TRUST_DIGEST = "sha256:" + ("3" * 64)
VERIFIER_FINGERPRINT = "sha256:" + ("4" * 64)
VERIFIER_IDENTITY = "qualification-verifier"
_ADDIN_PROVIDERS = {
    "gguf": (
        "llama_cpp",
        "invarlock-runtime-gguf",
        "invarlock_addins.gguf.provider:LlamaCppProvider",
    ),
    "multimodal": (
        "hf_vision_text",
        "invarlock-runtime-hf-vision-text",
        "invarlock_addins.multimodal.provider:HFVisionTextProvider",
    ),
    "tensorrt_llm": (
        "tensorrt_llm",
        "invarlock-runtime-tensorrt-llm",
        "invarlock_addins.tensorrt_llm.provider:TensorRTLLMProvider",
    ),
}


def _makefile(addin: str) -> str:
    return ROOT.joinpath("addins", addin, "Makefile").read_text(encoding="utf-8")


def _assert_frozen_source_labels(dockerfile: str) -> None:
    assert (
        'org.opencontainers.image.revision="${INVARLOCK_SOURCE_COMMIT}"' in dockerfile
    )
    assert (
        'dev.invarlock.source-bundle-sha256="${INVARLOCK_SOURCE_BUNDLE_SHA256}"'
        in dockerfile
    )


def test_gguf_runtime_has_reproducible_build_smoke_and_shared_qualification() -> None:
    makefile = _makefile("gguf")
    dockerfile = ROOT.joinpath("addins/gguf/runtime/Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "LLAMA_CPP_APT_SNAPSHOT is required" in makefile
    assert "APT::Snapshot=${LLAMA_CPP_APT_SNAPSHOT}" in dockerfile
    assert "Acquire::Check-Valid-Until=false" in dockerfile
    assert "LLAMA_CPP_BUILD_JOBS must be an integer from 1 to 8" in makefile
    assert "addins/gguf/runtime/Dockerfile" in makefile
    assert "scripts/authenticated_runtime_build.py" in makefile
    assert '--source-bundle "$(SOURCE_BUNDLE)"' in makefile
    assert '--statement "$(BUILD_STATEMENT)"' in makefile
    assert "--network none" in makefile
    assert "llama-completion" in makefile
    assert "raise SystemExit(main())" in makefile
    assert "RUNTIME_DEVICE ?= cpu" in makefile
    _assert_frozen_source_labels(dockerfile)
    _assert_shared_qualification_wrapper(makefile, "gguf")


@pytest.mark.parametrize("jobs", ["0", "9", "forty"])
def test_gguf_build_rejects_invalid_parallelism_before_authenticated_build(
    jobs: str,
) -> None:
    completed = subprocess.run(
        [
            "make",
            "-C",
            str(ROOT / "addins/gguf"),
            "build",
            "SOURCE_COMMIT=" + ("a" * 40),
            "SOURCE_BUNDLE=/unreachable/source.tar",
            "SOURCE_BUNDLE_SHA256=sha256:" + ("b" * 64),
            "LLAMA_CPP_APT_SNAPSHOT=20260715T000000Z",
            "LLAMA_CPP_BUILD_JOBS=" + jobs,
            "PYTHON=/unreachable/python",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "LLAMA_CPP_BUILD_JOBS must be an integer from 1 to 8" in completed.stderr
    assert "authenticated_runtime_build.py" not in completed.stdout


def test_gguf_build_rejects_malformed_snapshot_before_authenticated_build() -> None:
    completed = subprocess.run(
        [
            "make",
            "-C",
            str(ROOT / "addins/gguf"),
            "build",
            "SOURCE_COMMIT=" + ("a" * 40),
            "SOURCE_BUNDLE=/unreachable/source.tar",
            "SOURCE_BUNDLE_SHA256=sha256:" + ("b" * 64),
            "LLAMA_CPP_APT_SNAPSHOT=latest",
            "LLAMA_CPP_BUILD_JOBS=8",
            "PYTHON=/unreachable/python",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "LLAMA_CPP_APT_SNAPSHOT must use YYYYMMDDTHHMMSSZ" in completed.stderr
    assert "authenticated_runtime_build.py" not in completed.stdout


def test_tensorrt_runtime_has_gpu_build_smoke_canary_and_shared_qualification() -> None:
    makefile = _makefile("tensorrt_llm")
    readme = ROOT.joinpath("addins/tensorrt_llm/README.md").read_text(encoding="utf-8")
    dockerfile = ROOT.joinpath("addins/tensorrt_llm/runtime/Dockerfile").read_text(
        encoding="utf-8"
    )
    preflight = ROOT.joinpath("scripts/tensorrt_llm_canary_preflight.py").read_text(
        encoding="utf-8"
    )
    assert "addins/tensorrt_llm/runtime/Dockerfile" in makefile
    assert "scripts/authenticated_runtime_build.py" in makefile
    assert '--source-bundle "$(SOURCE_BUNDLE)"' in makefile
    assert '--statement "$(BUILD_STATEMENT)"' in makefile
    assert "--network none --gpus all" in makefile
    assert "torch.cuda.is_available" in makefile
    assert "--entrypoint /opt/invarlock/bin/vendor-python" in makefile
    assert "raise SystemExit(main())" in makefile
    assert "-m invarlock_addins.tensorrt_llm.canary" in makefile
    assert "--read-only" in makefile
    assert "--cap-drop=ALL" in makefile
    assert "--security-opt no-new-privileges" in makefile
    assert "--user 65532:65532" in makefile
    assert "scripts/tensorrt_llm_canary_preflight.py" in makefile
    assert "CANARY_TMPFS_GIB must be an integer from 4 to 64" in preflight
    assert 'canonical_input_root="$$(PYTHONNOUSERSITE=1' in makefile
    assert "src:$(REPOSITORY_ROOT)/addins/tensorrt_llm/src" in makefile
    assert "src=$$canonical_input_root,dst=/inputs,readonly" in makefile
    _assert_frozen_source_labels(dockerfile)
    _assert_shared_qualification_wrapper(makefile, "tensorrt_llm")
    assert "INVARLOCK_TENSORRT_LLM_RUNNER" not in dockerfile
    assert 'IMAGE="$IMAGE"' in readme
    assert 'IMAGE_DIGEST="$DIGEST"' in readme
    assert 'DIGEST="$IMAGE"' in readme
    assert "docker tag invarlock-tensorrt-llm:candidate" in readme
    assert "index .RepoDigests 0" not in readme
    assert 'entry.rpartition("@")[0] == repository' in readme


def test_multimodal_runtime_has_conformance_and_shared_qualification() -> None:
    makefile = _makefile("multimodal")
    dockerfile = ROOT.joinpath("addins/multimodal/runtime/Dockerfile").read_text(
        encoding="utf-8"
    )
    dockerignore = ROOT.joinpath(".dockerignore").read_text(encoding="utf-8")
    assert "addins/multimodal/runtime/Dockerfile" in makefile
    assert "BASE_IMAGE must use a named repository@sha256 reference" in makefile
    assert "scripts/authenticated_runtime_build.py" in makefile
    assert '--source-bundle "$(SOURCE_BUNDLE)"' in makefile
    assert '--statement "$(BUILD_STATEMENT)"' in makefile
    assert '--require-base-source-labels "$(BASE_IMAGE)"' in makefile
    assert "--network none --gpus all" in makefile
    assert "torch.cuda.is_available" in makefile
    assert "torchvision.__version__ == '0.26.0+cu128'" in makefile
    assert "Qwen2VLImageProcessor" in makefile
    assert "Qwen2VLVideoProcessor" in makefile
    _assert_frozen_source_labels(dockerfile)
    assert "conformance:" in makefile
    assert "INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT" in makefile
    assert "INVARLOCK_HF_VISION_TEXT_CONTENT_STORE" in makefile
    _assert_shared_qualification_wrapper(makefile, "multimodal")
    assert "RUNTIME_BASE_IMAGE" in dockerfile
    assert "multimodal-runtime-py312.txt" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "--no-deps" in dockerfile
    assert "https://download.pytorch.org/whl/cu128" in dockerfile
    assert "torchvision.__version__ == '0.26.0+cu128'" in dockerfile
    assert 'ENTRYPOINT ["python", "-m", "invarlock"]' in dockerfile
    for build_input in (
        "!requirements/workflows/multimodal-runtime-py312.txt",
        "!addins/multimodal/pyproject.toml",
        "!addins/multimodal/README.md",
        "!addins/multimodal/src/**",
        "!addins/multimodal/runtime/Dockerfile",
    ):
        assert build_input in dockerignore


def test_multimodal_build_rejects_a_raw_base_config_id_before_python_or_docker() -> (
    None
):
    completed = subprocess.run(
        [
            "make",
            "-C",
            str(ROOT / "addins/multimodal"),
            "build",
            "SOURCE_COMMIT=" + ("a" * 40),
            "SOURCE_BUNDLE=/unreachable/source.tar",
            "SOURCE_BUNDLE_SHA256=sha256:" + ("b" * 64),
            "BASE_IMAGE=sha256:" + ("c" * 64),
            "IMAGE=invarlock-hf-vision-text:local",
            "PYTHON=/unreachable/python",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "BASE_IMAGE must use a named repository@sha256 reference" in (
        completed.stderr
    )
    assert "authenticated_runtime_build.py" not in completed.stdout


def test_source_tree_test_environments_include_multimodal_host_dependencies() -> None:
    """Clean developer and CI installs must run vision-text host conformance."""

    project = tomllib.loads(ROOT.joinpath("pyproject.toml").read_text(encoding="utf-8"))
    optional = project["project"]["optional-dependencies"]
    for extra in ("dev", "ci"):
        assert any(
            str(requirement).lower().startswith("pillow>=")
            for requirement in optional[extra]
        ), f"{extra} must install Pillow for multimodal host preflight tests"
    for python_tag in ("312", "313"):
        lock = ROOT.joinpath(
            "requirements", "workflows", f"ci-hf-py{python_tag}.txt"
        ).read_text(encoding="utf-8")
        assert "pillow==" in lock.lower()


def _assert_shared_qualification_wrapper(makefile: str, addin: str) -> None:
    _provider_name, distribution_name, _entry_point_value = _ADDIN_PROVIDERS[addin]
    assert "qualification-host-check:" in makefile
    assert "qualification host check failed for PYTHON=$(PYTHON)" in makefile
    assert (
        f"matching core (invarlock) and add-in ({distribution_name}) wheels" in makefile
    )
    assert "PYTHONPATH/importability alone is insufficient" in makefile
    assert "conformance: qualification-host-check" in makefile
    assert "qualify-preflight: qualification-host-check" not in makefile
    assert "qualify-evidence: qualification-host-check" not in makefile
    assert "qualify-canary: qualification-host-check" not in makefile
    assert makefile.count("scripts/runtime_qualification.py") == 3
    assert (
        makefile.count(
            '"$(QUALIFICATION_DRIVER_PYTHON)" -I -S scripts/runtime_qualification.py'
        )
        == 3
    )
    assert "scripts/runtime_qualification.py canary" in makefile
    assert "scripts/runtime_qualification.py readiness" in makefile
    assert "scripts/runtime_qualification.py run" in makefile
    assert makefile.count('--python "$(PYTHON)"') == 3
    assert makefile.count('--source-commit "$(SOURCE_COMMIT)"') == 4
    assert makefile.count('--source-bundle "$(SOURCE_BUNDLE)"') == 4
    assert makefile.count('--source-bundle-sha256 "$(SOURCE_BUNDLE_SHA256)"') == 4
    assert (
        makefile.count('--candidate-wheel-manifest "$(CANDIDATE_WHEEL_MANIFEST)"') == 3
    )
    assert makefile.count('--canary-evidence "$(CANARY_EVIDENCE)"') == 2
    assert makefile.count('--canary-receipt "$(CANARY_RECEIPT)"') == 2
    assert makefile.count('--canary-trust-profile "$(CANARY_TRUST_PROFILE)"') == 2
    assert "QUALIFICATION_DEVICE ?= $(RUNTIME_DEVICE)" in makefile
    assert "QUALIFICATION_DRIVER_PYTHON ?= $(PYTHON)" in makefile
    if addin in {"gguf", "multimodal", "tensorrt_llm"}:
        assert "QUALIFICATION_IMAGE ?= $(IMAGE_DIGEST)" in makefile
        assert makefile.count('--runtime-image "$(QUALIFICATION_IMAGE)"') == 3
        assert '--runtime-image "$(IMAGE)"' not in makefile
    assert makefile.count('--runtime-device "$(QUALIFICATION_DEVICE)"') == 3
    assert makefile.count('--runtime-cpus "$(QUALIFICATION_CPUS)"') == 3
    assert makefile.count('--runtime-memory-mib "$(QUALIFICATION_MEMORY_MIB)"') == 3
    assert makefile.count('--runtime-user "$(QUALIFICATION_USER)"') == 3
    assert '--summary "$(SUMMARY)"' in makefile
    assert "-m invarlock evaluate" not in makefile
    assert "-m invarlock verify" not in makefile
    assert f'PYTHONPATH="src:addins/{addin}/src"' not in makefile
    assert makefile.count("PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH=") == 2
    assert (
        "conformance: qualification-host-check\n"
        '\tPYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "$(PYTHON)" -m '
    ) in makefile


def _qualification_host_site(
    tmp_path: Path,
    addin: str,
    *,
    include_distribution_metadata: bool,
) -> Path:
    site = tmp_path / (
        f"{addin}-host-site-"
        + ("installed" if include_distribution_metadata else "source-only")
    )
    shutil.copytree(
        ROOT / "addins" / addin / "src" / "invarlock_addins",
        site / "invarlock_addins",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )
    if include_distribution_metadata:
        provider_name, distribution_name, entry_point_value = _ADDIN_PROVIDERS[addin]
        dist_info = site / (
            f"{distribution_name.replace('-', '_')}-{INVARLOCK_VERSION}.dist-info"
        )
        dist_info.mkdir()
        dist_info.joinpath("METADATA").write_text(
            "Metadata-Version: 2.4\n"
            f"Name: {distribution_name}\n"
            f"Version: {INVARLOCK_VERSION}\n",
            encoding="utf-8",
        )
        dist_info.joinpath("entry_points.txt").write_text(
            f"[invarlock.runtime_providers]\n{provider_name} = {entry_point_value}\n",
            encoding="utf-8",
        )
    return site


def _write_installed_python(tmp_path: Path, site: Path) -> Path:
    selected = tmp_path / "qualification venv" / "bin" / "python"
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "import runpy\n"
        "import sys\n"
        "inherited = {entry for entry in os.environ.get('PYTHONPATH', '').split(os.pathsep) if entry}\n"
        "position = max((index + 1 for index, entry in enumerate(sys.path) if entry in inherited), default=0)\n"
        f"sys.path[position:position] = [{str(ROOT / 'src')!r}, {str(site)!r}]\n"
        "arguments = sys.argv[1:]\n"
        "if arguments[:1] == ['-c']:\n"
        "    sys.argv = ['-c', *arguments[2:]]\n"
        "    exec(arguments[1], {'__name__': '__main__'})\n"
        "elif arguments[:1] == ['-m']:\n"
        "    sys.argv = [arguments[1], *arguments[2:]]\n"
        "    runpy.run_module(arguments[1], run_name='__main__', alter_sys=True)\n"
        "else:\n"
        "    raise SystemExit('test interpreter accepts only -c or -m')\n",
        encoding="utf-8",
    )
    selected.chmod(0o700)
    return selected


@pytest.mark.parametrize("addin", tuple(_ADDIN_PROVIDERS))
@pytest.mark.parametrize("include_distribution_metadata", (False, True))
def test_qualification_host_check_requires_approved_distribution_metadata(
    tmp_path: Path,
    addin: str,
    include_distribution_metadata: bool,
) -> None:
    site = _qualification_host_site(
        tmp_path,
        addin,
        include_distribution_metadata=include_distribution_metadata,
    )
    selected = _write_installed_python(tmp_path, site)
    environment = {
        **os.environ,
        "INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS": "0",
        "PYTHONPATH": str(site),
    }

    completed = subprocess.run(
        [
            "make",
            "-C",
            str(ROOT / "addins" / addin),
            "qualification-host-check",
            f"PYTHON={selected}",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    if include_distribution_metadata:
        assert completed.returncode == 0, completed.stderr or completed.stdout
    else:
        assert completed.returncode != 0
        provider_name, distribution_name, _entry_point_value = _ADDIN_PROVIDERS[addin]
        assert f"Unknown runtime provider {provider_name!r}" in completed.stderr
        assert (
            f"qualification host check failed for PYTHON={selected}" in completed.stderr
        )
        assert (
            f"matching core (invarlock) and add-in ({distribution_name}) wheels"
            in completed.stderr
        )
        assert "PYTHONPATH/importability alone is insufficient" in completed.stderr


@pytest.mark.parametrize("addin", tuple(_ADDIN_PROVIDERS))
def test_conformance_uses_the_installed_provider_not_injected_source(
    tmp_path: Path,
    addin: str,
) -> None:
    site = _qualification_host_site(
        tmp_path,
        addin,
        include_distribution_metadata=True,
    )
    selected = _write_installed_python(tmp_path, site)
    injected = tmp_path / "injected source" / "invarlock_addins" / addin
    injected.mkdir(parents=True)
    injected.parent.joinpath("__init__.py").write_text("", encoding="utf-8")
    injected.joinpath("__init__.py").write_text("", encoding="utf-8")
    injected.joinpath("conformance.py").write_text(
        "raise RuntimeError('injected source executed')\n", encoding="utf-8"
    )

    environment = {**os.environ, "PYTHONPATH": str(injected.parents[1])}
    direct = subprocess.run(
        [selected, "-m", f"invarlock_addins.{addin}.conformance"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert direct.returncode != 0
    assert "injected source executed" in direct.stderr

    completed = subprocess.run(
        [
            "make",
            "-C",
            str(ROOT / "addins" / addin),
            "conformance",
            f"PYTHON={selected}",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    result = next(
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.lstrip().startswith("{")
    )
    assert result["ok"] is True
    assert "injected source executed" not in completed.stderr


def test_optional_runtime_qualification_remains_addin_owned() -> None:
    root_makefile = ROOT.joinpath("Makefile").read_text(encoding="utf-8")
    assert "runtime-image-gguf" not in root_makefile
    assert "runtime-image-tensorrt-llm" not in root_makefile


def _write_python_recorder(tmp_path: Path) -> tuple[Path, Path]:
    tool_directory = tmp_path / "tool directory"
    tool_directory.mkdir()
    executable = tool_directory / "python target"
    selected = tmp_path / "qualification venv" / "bin" / "python"
    selected.parent.mkdir(parents=True)
    log = tmp_path / "qualification invocations.jsonl"
    executable.write_text(
        f"""#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

arguments = sys.argv[1:]
entry = {{
    "argv": arguments,
    "invoked_as": sys.argv[0],
    "pythonpath": os.environ.get("PYTHONPATH"),
    "content_store": os.environ.get("INVARLOCK_HF_VISION_TEXT_CONTENT_STORE"),
    "resource_root": os.environ.get("INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT"),
}}
with open(os.environ["QUALIFICATION_LOG"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(entry, sort_keys=True) + "\\n")

if arguments and arguments[0].endswith("qualification_precheck.py"):
    json.load(sys.stdin)
    print(json.dumps({{
        "artifact_digests": {{
            "baseline": "{BASELINE_ARTIFACT}",
            "subject": "{SUBJECT_ARTIFACT}",
        }},
        "evidence_signer_fingerprint": "{SIGNER_FINGERPRINT}",
        "format_version": "invarlock/qualification-precheck-v1",
        "ok": True,
        "policy_digest": "{POLICY_DIGEST}",
        "runtime_digests": {{
            "baseline": "{BASELINE_RUNTIME}",
            "subject": "{SUBJECT_RUNTIME}",
        }},
        "schedule_digest": "{SCHEDULE_DIGEST}",
        "trust_profile_digest": "{TRUST_DIGEST}",
        "verifier_fingerprint": "{VERIFIER_FINGERPRINT}",
        "verifier_identity": "{VERIFIER_IDENTITY}",
    }}, sort_keys=True))
elif arguments[:3] == ["-m", "invarlock", "evaluate"]:
    if "--preflight" in arguments:
        print(json.dumps({{
            "format_version": "invarlock/evaluation-preflight-v1",
            "ok": True,
            "output": os.environ["QUALIFICATION_EVIDENCE"],
        }}, sort_keys=True))
    else:
        print(json.dumps({{
            "evidence": os.environ["QUALIFICATION_EVIDENCE"],
            "format_version": "invarlock/evaluation-result-v1",
            "ok": True,
            "pack_manifest_digest": "{PACK_DIGEST}",
        }}, sort_keys=True))
elif arguments[:3] == ["-m", "invarlock", "verify"]:
    receipt = Path(arguments[arguments.index("--receipt") + 1])
    receipt.write_text("signed receipt\\n", encoding="utf-8")
    print(json.dumps({{
        "anchors": {{
            "artifact_digests": {{
                "baseline": "{BASELINE_ARTIFACT}",
                "subject": "{SUBJECT_ARTIFACT}",
            }},
            "policy_digest": "{POLICY_DIGEST}",
            "runtime_digests": {{
                "baseline": "{BASELINE_RUNTIME}",
                "subject": "{SUBJECT_RUNTIME}",
            }},
            "schedule_digest": "{SCHEDULE_DIGEST}",
            "signer_fingerprint": "{SIGNER_FINGERPRINT}",
        }},
        "format_version": "invarlock/evidence-pack-verification-v1",
        "ok": True,
        "pack_manifest_digest": "{PACK_DIGEST}",
        "signed_receipt": receipt.name,
        "trust_profile_digest": "{TRUST_DIGEST}",
        "verifier_fingerprint": "{VERIFIER_FINGERPRINT}",
        "verifier_identity": "{VERIFIER_IDENTITY}",
    }}, sort_keys=True))
elif arguments[:3] == ["-m", "invarlock", "report"]:
    report = Path(arguments[arguments.index("--html") + 1])
    report.write_text("<html>qualification report</html>\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    selected.symlink_to(executable)
    return selected, log


def _write_driver_recorder(tmp_path: Path, host_site: Path) -> tuple[Path, Path, Path]:
    driver = tmp_path / "qualification driver"
    selected = _write_installed_python(tmp_path, host_site)
    log = tmp_path / "driver invocation.json"
    driver.write_text(
        f"""#!{sys.executable}
import json
import os
import stat
import sys
from pathlib import Path

arguments = sys.argv[1:]
isolated = arguments[:2] == ["-I", "-S"]
if isolated:
    arguments = arguments[2:]
Path(os.environ["QUALIFICATION_DRIVER_LOG"]).write_text(json.dumps({{
    "argv": arguments,
    "isolated": isolated,
    "pythonpath": os.environ.get("PYTHONPATH"),
    "content_store": os.environ.get("INVARLOCK_HF_VISION_TEXT_CONTENT_STORE"),
    "resource_root": os.environ.get("INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT"),
}}), encoding="utf-8")
mode = arguments[1]
if mode in {{"run", "canary"}}:
    summary = Path(arguments[arguments.index("--summary") + 1])
    summary.write_text(json.dumps({{"mode": mode, "stage": "complete"}}), encoding="utf-8")
    summary.chmod(stat.S_IRUSR | stat.S_IWUSR)
    if "--report" in arguments:
        Path(arguments[arguments.index("--report") + 1]).write_text(
            "<html>qualification report</html>\\n", encoding="utf-8"
        )
print(json.dumps({{"format_version": "invarlock/runtime-qualification-v1", "mode": mode, "ok": True, "stage": "ready" if mode == "readiness" else "complete"}}))
""",
        encoding="utf-8",
    )
    driver.chmod(0o700)
    return driver, selected, log


def _qualification_case(
    tmp_path: Path, addin: str, *, target: str
) -> tuple[list[str], dict[str, Path | str], dict[str, str], Path]:
    host_site = _qualification_host_site(
        tmp_path,
        addin,
        include_distribution_metadata=True,
    )
    driver, selected, log = _write_driver_recorder(tmp_path, host_site)
    inputs = tmp_path / "qualification inputs"
    inputs.mkdir()
    request = inputs / "request file.yaml"
    request.write_text(
        "format_version: invarlock/evaluation-request-v1\n", encoding="utf-8"
    )
    source_bundle = inputs / "source bundle.tar.gz"
    source_bundle.write_bytes(SOURCE_BUNDLE_BYTES)
    candidate_manifest = inputs / "candidate wheels.json"
    candidate_manifest.write_text("{}\n", encoding="utf-8")
    paths: dict[str, Path | str] = {
        "python": selected,
        "request": request,
        "signing_key": inputs / "evidence signer.pem",
        "evidence": inputs / "evidence directory",
        "trust_profile": inputs / "trust profile.json",
        "receipt": inputs / "verification receipt.json",
        "canary_evidence": inputs / "canary evidence directory",
        "canary_receipt": inputs / "canary verification receipt.json",
        "canary_trust_profile": inputs / "canary trust profile.json",
        "source_bundle": source_bundle,
        "candidate_manifest": candidate_manifest,
        "summary": inputs / "qualification summary.json",
        "report": inputs / "human report.html",
        "resource_root": inputs / "vision resources",
        "content_store": "image content store",
    }
    paths["host_pythonpath"] = os.pathsep.join((str(ROOT / "src"), str(host_site)))
    arguments = [
        "make",
        "-C",
        str(ROOT / "addins" / addin),
        target,
        f"QUALIFICATION_DRIVER_PYTHON={driver}",
        f"PYTHON={paths['python']}",
        f"REQUEST={paths['request']}",
        f"SIGNING_KEY={paths['signing_key']}",
        f"EVIDENCE={paths['evidence']}",
        f"TRUST_PROFILE={paths['trust_profile']}",
        f"RECEIPT={paths['receipt']}",
        f"CANARY_EVIDENCE={paths['canary_evidence']}",
        f"CANARY_RECEIPT={paths['canary_receipt']}",
        f"CANARY_TRUST_PROFILE={paths['canary_trust_profile']}",
        f"SOURCE_COMMIT={SOURCE_COMMIT}",
        f"SOURCE_BUNDLE={paths['source_bundle']}",
        f"SOURCE_BUNDLE_SHA256={SOURCE_BUNDLE_DIGEST}",
        f"CANDIDATE_WHEEL_MANIFEST={paths['candidate_manifest']}",
        f"SUMMARY={paths['summary']}",
        "IMAGE=registry.example/candidate@" + IMAGE_DIGEST,
        f"IMAGE_DIGEST={IMAGE_DIGEST}",
        "CONTAINER_ENGINE=docker",
        "QUALIFICATION_CPUS=12",
        "QUALIFICATION_MEMORY_MIB=98304",
        "QUALIFICATION_USER=65532:65532",
    ]
    if addin == "multimodal":
        arguments.extend(
            (
                f"REPORT={paths['report']}",
                f"RESOURCE_ROOT={paths['resource_root']}",
                f"CONTENT_STORE={paths['content_store']}",
            )
        )
    elif addin == "tensorrt_llm":
        arguments.append("QUALIFICATION_DEVICE=cuda:1")
    environment = {
        **os.environ,
        "QUALIFICATION_DRIVER_LOG": str(log),
        "PYTHONPATH": str(paths["host_pythonpath"]),
    }
    return arguments, paths, environment, log


def _logged(log: Path) -> list[dict[str, Any]]:
    return [json.loads(log.read_text(encoding="utf-8"))]


def _assert_real_driver_required_options(argv: list[str], mode: str) -> None:
    parser = runtime_qualification._parser()  # noqa: SLF001
    mode_action = next(action for action in parser._actions if action.dest == "mode")
    mode_parser = mode_action.choices[mode]
    required = {
        option
        for action in mode_parser._actions
        if action.required
        for option in action.option_strings
    }
    assert required <= set(argv)


def _assert_isolated_driver(invocation: dict[str, Any]) -> None:
    assert invocation["isolated"] is True


@pytest.mark.parametrize("addin", ("gguf", "multimodal", "tensorrt_llm"))
def test_qualification_readiness_uses_selected_venv_without_execution(
    tmp_path: Path,
    addin: str,
) -> None:
    arguments, paths, environment, log = _qualification_case(
        tmp_path, addin, target="qualify-preflight"
    )

    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    result = next(
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.lstrip().startswith("{")
    )
    assert result["mode"] == "readiness"
    assert result["stage"] == "ready"
    invocations = _logged(log)
    assert len(invocations) == 1
    _assert_isolated_driver(invocations[0])
    argv = invocations[0]["argv"]
    assert str(argv[0]).endswith("scripts/runtime_qualification.py")
    assert argv[1] == "readiness"
    assert argv[argv.index("--python") + 1] == str(paths["python"])
    assert argv[argv.index("--source-commit") + 1] == SOURCE_COMMIT
    assert argv[argv.index("--source-bundle") + 1] == str(paths["source_bundle"])
    assert argv[argv.index("--candidate-wheel-manifest") + 1] == str(
        paths["candidate_manifest"]
    )
    assert argv[argv.index("--canary-evidence") + 1] == str(paths["canary_evidence"])
    assert argv[argv.index("--runtime-cpus") + 1] == "12"
    assert argv[argv.index("--runtime-memory-mib") + 1] == "98304"
    assert argv[argv.index("--runtime-user") + 1] == "65532:65532"
    assert not Path(paths["summary"]).exists()
    _assert_real_driver_required_options(argv, "readiness")
    _assert_addin_environment(invocations, addin, paths)


@pytest.mark.parametrize("addin", ("gguf", "multimodal", "tensorrt_llm"))
def test_qualification_run_forwards_provenance_and_private_summary_inputs(
    tmp_path: Path,
    addin: str,
) -> None:
    arguments, paths, environment, log = _qualification_case(
        tmp_path, addin, target="qualify-evidence"
    )

    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    invocations = _logged(log)
    assert len(invocations) == 1
    _assert_isolated_driver(invocations[0])
    argv = invocations[0]["argv"]
    assert str(argv[0]).endswith("scripts/runtime_qualification.py")
    assert argv[1] == "run"
    assert argv[argv.index("--python") + 1] == str(paths["python"])
    assert argv[argv.index("--source-bundle-sha256") + 1] == SOURCE_BUNDLE_DIGEST
    assert argv[argv.index("--candidate-wheel-manifest") + 1] == str(
        paths["candidate_manifest"]
    )
    assert argv[argv.index("--canary-receipt") + 1] == str(paths["canary_receipt"])
    assert argv[argv.index("--runtime-cpus") + 1] == "12"
    assert argv[argv.index("--runtime-memory-mib") + 1] == "98304"
    assert argv[argv.index("--runtime-user") + 1] == "65532:65532"
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert summary == {"mode": "run", "stage": "complete"}
    assert Path(paths["summary"]).stat().st_mode & 0o777 == 0o600
    _assert_real_driver_required_options(argv, "run")
    _assert_addin_environment(invocations, addin, paths)
    if addin == "multimodal":
        assert Path(paths["report"]).is_file()


@pytest.mark.parametrize("addin", ("gguf", "multimodal", "tensorrt_llm"))
def test_qualification_canary_bootstrap_has_no_prior_canary_input(
    tmp_path: Path,
    addin: str,
) -> None:
    arguments, paths, environment, log = _qualification_case(
        tmp_path, addin, target="qualify-canary"
    )

    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    invocation = _logged(log)[0]
    _assert_isolated_driver(invocation)
    argv = invocation["argv"]
    assert argv[1] == "canary"
    assert "--canary-evidence" not in argv
    assert argv[argv.index("--runtime-cpus") + 1] == "12"
    assert argv[argv.index("--runtime-memory-mib") + 1] == "98304"
    assert argv[argv.index("--runtime-user") + 1] == "65532:65532"
    assert argv[argv.index("--candidate-wheel-manifest") + 1] == str(
        paths["candidate_manifest"]
    )
    assert Path(paths["summary"]).is_file()
    _assert_real_driver_required_options(argv, "canary")


@pytest.mark.parametrize("addin", ("gguf", "tensorrt_llm"))
def test_qualification_derives_immutable_image_reference_from_local_build_tag(
    tmp_path: Path,
    addin: str,
) -> None:
    arguments, _paths, environment, log = _qualification_case(
        tmp_path, addin, target="qualify-canary"
    )
    arguments.append("IMAGE=invarlock-runtime:mutable-local-build")

    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    invocation = _logged(log)[0]["argv"]
    assert invocation[invocation.index("--runtime-image") + 1] == IMAGE_DIGEST
    assert invocation[invocation.index("--runtime-image-digest") + 1] == IMAGE_DIGEST
    assert "invarlock-runtime:mutable-local-build" not in invocation


def _assert_addin_environment(
    invocations: list[dict[str, Any]],
    addin: str,
    paths: dict[str, Path | str],
) -> None:
    assert all(entry["pythonpath"] == paths["host_pythonpath"] for entry in invocations)
    if addin == "multimodal":
        assert all(
            entry["resource_root"] == str(paths["resource_root"])
            for entry in invocations
        )
        assert all(
            entry["content_store"] == paths["content_store"] for entry in invocations
        )


@pytest.mark.parametrize("addin", ("gguf", "multimodal", "tensorrt_llm"))
@pytest.mark.parametrize(
    ("target", "missing", "message"),
    (
        ("qualify-preflight", "SOURCE_BUNDLE=", "SOURCE_BUNDLE is required"),
        (
            "qualify-preflight",
            "CANDIDATE_WHEEL_MANIFEST=",
            "CANDIDATE_WHEEL_MANIFEST is required",
        ),
        ("qualify-evidence", "SUMMARY=", "SUMMARY is required"),
        ("qualify-evidence", "CANARY_RECEIPT=", "CANARY_RECEIPT is required"),
    ),
)
def test_qualification_wrapper_rejects_missing_provenance_or_summary_input(
    tmp_path: Path,
    addin: str,
    target: str,
    missing: str,
    message: str,
) -> None:
    arguments, _, environment, log = _qualification_case(tmp_path, addin, target=target)
    arguments.append(missing)

    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert message in completed.stderr
    assert not log.exists()


def _tensorrt_canary_case(
    tmp_path: Path,
    *,
    values: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str], Path, Path]:
    recorder, log = _write_python_recorder(tmp_path)
    input_root = tmp_path / "canary inputs"
    input_root.mkdir()
    engine = input_root / "engine bundle"
    engine.mkdir()
    engine.joinpath("config.json").write_text(
        json.dumps(
            {
                "build_config": {
                    "max_batch_size": 8,
                    "max_input_len": 128,
                    "max_seq_len": 256,
                },
                "pretrained_config": {
                    "architecture": "LlamaForCausalLM",
                    "dtype": "float16",
                    "mapping": {"pp_size": 1, "tp_size": 1, "world_size": 1},
                },
                "version": "1.0.0",
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    engine.joinpath("rank0.engine").write_bytes(b"serialized-engine-fixture")
    tokenizer = input_root / "tokenizer contract.json"
    tokenizer.write_text(
        json.dumps(
            {
                "add_special_tokens": False,
                "clean_up_tokenization_spaces": False,
                "eos_token_id": 1,
                "format_version": "invarlock/tensorrt-llm-tokenizer-contract-v1",
                "pad_token_id": 0,
                "skip_special_tokens": True,
                "tokenizer_json": {
                    "model": {"type": "BPE"},
                    "version": "1.0",
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    tokenizer_sha256 = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    engine_tree_sha256 = read_tensorrt_llm_artifact_identity(
        engine,
        target_compute_capability="9.0",
        tokenizer_metadata_sha256=tokenizer_sha256,
    ).engine_bundle_tree_sha256
    digest = "sha256:" + ("a" * 64)
    arguments = {
        "CANARY_TMPFS_GIB": "8",
        "ENGINE_BUNDLE": "engine bundle",
        "EXPECTED_ENGINE_TREE_SHA256": engine_tree_sha256,
        "EXPECTED_OUTPUT_SHA256": "d" * 64,
        "EXPECTED_TOKENIZER_SHA256": tokenizer_sha256,
        "IMAGE": "registry.example/invarlock/candidate@" + digest,
        "IMAGE_DIGEST": digest,
        "INPUT_ROOT": str(input_root),
        "TOKENIZER_CONTRACT": "tokenizer contract.json",
    }
    arguments.update(values or {})
    command = [
        "make",
        "-C",
        str(ROOT / "addins/tensorrt_llm"),
        "canary",
        f"PYTHON={sys.executable}",
        f"CONTAINER_ENGINE={recorder}",
        *(f"{name}={value}" for name, value in arguments.items()),
    ]
    environment = {
        **os.environ,
        "QUALIFICATION_EVIDENCE": "unused",
        "QUALIFICATION_LOG": str(log),
    }
    return command, environment, log, input_root


def _run_tensorrt_canary(
    tmp_path: Path,
    *,
    values: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    command, environment, log, input_root = _tensorrt_canary_case(
        tmp_path, values=values
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, log, input_root


def _replace_make_assignment(
    command: list[str],
    name: str,
    value: str,
) -> list[str]:
    prefix = name + "="
    return [prefix + value if item.startswith(prefix) else item for item in command]


def test_tensorrt_canary_uses_authenticated_canonical_root_with_spaced_paths(
    tmp_path: Path,
) -> None:
    completed, log, input_root = _run_tensorrt_canary(tmp_path)
    digest = "sha256:" + ("a" * 64)
    image = "registry.example/invarlock/candidate@" + digest

    assert completed.returncode == 0, completed.stderr or completed.stdout
    invocations = _logged(log)
    assert len(invocations) == 1
    argv = invocations[0]["argv"]
    assert f"type=bind,src={input_root},dst=/inputs,readonly" in argv
    assert image in argv
    assert "/inputs/engine bundle" in argv
    assert "/inputs/tokenizer contract.json" in argv
    assert f"INVARLOCK_RUNTIME_IMAGE={digest}" in argv


def test_tensorrt_canary_rejects_an_image_not_bound_to_its_digest(
    tmp_path: Path,
) -> None:
    digest = "sha256:" + ("a" * 64)
    completed, log, _input_root = _run_tensorrt_canary(
        tmp_path,
        values={
            "IMAGE": "registry.example/candidate:mutable",
            "IMAGE_DIGEST": digest,
        },
    )

    assert completed.returncode == 2
    assert "IMAGE must be" in completed.stderr
    assert not log.exists()


@pytest.mark.parametrize(
    ("values", "message"),
    (
        (
            {"EXPECTED_ENGINE_TREE_SHA256": "0" * 64},
            "engine bundle does not match",
        ),
        (
            {"EXPECTED_TOKENIZER_SHA256": "0" * 64},
            "tokenizer contract does not match",
        ),
        (
            {"IMAGE": "candidate image@" + IMAGE_DIGEST},
            "IMAGE must be",
        ),
        (
            {"IMAGE": "registry.example/invarlock,candidate@" + IMAGE_DIGEST},
            "IMAGE must be",
        ),
        (
            {"IMAGE": "registry.example/invarlock/candidate@@" + IMAGE_DIGEST},
            "IMAGE must be",
        ),
    ),
)
def test_tensorrt_canary_rejects_unauthenticated_bytes_or_image_before_container(
    tmp_path: Path,
    values: dict[str, str],
    message: str,
) -> None:
    completed, log, _input_root = _run_tensorrt_canary(tmp_path, values=values)

    assert completed.returncode == 2
    assert message in completed.stderr
    assert not log.exists()


def test_tensorrt_canary_rejects_malformed_tokenizer_before_container(
    tmp_path: Path,
) -> None:
    command, environment, log, input_root = _tensorrt_canary_case(tmp_path)
    tokenizer = input_root / "tokenizer contract.json"
    tokenizer.write_text('{"unexpected":true}\n', encoding="utf-8")
    command = _replace_make_assignment(
        command,
        "EXPECTED_TOKENIZER_SHA256",
        hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
    )

    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "fields are not closed" in completed.stderr
    assert not log.exists()


@pytest.mark.parametrize("root_kind", ("comma", "symlink_parent"))
def test_tensorrt_canary_rejects_unsafe_input_root_before_container(
    tmp_path: Path,
    root_kind: str,
) -> None:
    command, environment, log, input_root = _tensorrt_canary_case(tmp_path)
    if root_kind == "comma":
        unsafe_root = tmp_path / "canary,inputs"
        input_root.rename(unsafe_root)
    else:
        real_parent = tmp_path / "real-parent"
        real_parent.mkdir()
        real_root = real_parent / "inputs"
        input_root.rename(real_root)
        linked_parent = tmp_path / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        unsafe_root = linked_parent / "inputs"
    command = _replace_make_assignment(command, "INPUT_ROOT", str(unsafe_root))

    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert not log.exists()


@pytest.mark.parametrize(
    ("values", "message"),
    (
        (
            {
                "IMAGE": "sha256:" + ("a" * 63),
                "IMAGE_DIGEST": "sha256:" + ("a" * 63),
            },
            "IMAGE_DIGEST must be a canonical sha256 image digest",
        ),
        (
            {"EXPECTED_ENGINE_TREE_SHA256": "B" * 64},
            "EXPECTED_ENGINE_TREE_SHA256 must be a lowercase sha256 digest",
        ),
        (
            {"EXPECTED_TOKENIZER_SHA256": "c" * 63},
            "EXPECTED_TOKENIZER_SHA256 must be a lowercase sha256 digest",
        ),
        (
            {"EXPECTED_OUTPUT_SHA256": "not-a-digest"},
            "EXPECTED_OUTPUT_SHA256 must be a lowercase sha256 digest",
        ),
        (
            {"INPUT_ROOT": "relative/input-root"},
            "INPUT_ROOT must be an absolute path",
        ),
        (
            {"INPUT_ROOT": "/definitely/missing/invarlock-canary-inputs"},
            "INPUT_ROOT must be an existing non-symlink directory",
        ),
        (
            {"ENGINE_BUNDLE": "../escape"},
            "ENGINE_BUNDLE must be a portable relative path",
        ),
        (
            {"TOKENIZER_CONTRACT": "nested\\tokenizer.json"},
            "TOKENIZER_CONTRACT must be a portable relative path",
        ),
        (
            {"CANARY_TMPFS_GIB": "3"},
            "CANARY_TMPFS_GIB must be an integer from 4 to 64",
        ),
        (
            {"CANARY_TMPFS_GIB": "eight"},
            "CANARY_TMPFS_GIB must be an integer from 4 to 64",
        ),
    ),
)
def test_tensorrt_canary_rejects_invalid_host_inputs_before_container_execution(
    tmp_path: Path,
    values: dict[str, str],
    message: str,
) -> None:
    completed, log, _input_root = _run_tensorrt_canary(tmp_path, values=values)

    assert completed.returncode == 2
    assert message in completed.stderr
    assert not log.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("root_symlink", "INPUT_ROOT contains a symlink"),
        (
            "engine_symlink",
            "engine identity cannot be authenticated",
        ),
        (
            "tokenizer_symlink",
            "TOKENIZER_CONTRACT must exist beneath INPUT_ROOT without symbolic links",
        ),
        ("engine_file", "engine identity cannot be authenticated"),
        ("tokenizer_directory", "TOKENIZER_CONTRACT must be a stable regular file"),
    ),
)
def test_tensorrt_canary_rejects_unsafe_resource_types_before_container_execution(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    command, environment, log, input_root = _tensorrt_canary_case(tmp_path)
    if mutation == "root_symlink":
        linked_root = tmp_path / "linked inputs"
        linked_root.symlink_to(input_root, target_is_directory=True)
        command = [
            f"INPUT_ROOT={linked_root}" if item.startswith("INPUT_ROOT=") else item
            for item in command
        ]
    elif mutation == "engine_symlink":
        shutil.rmtree(input_root / "engine bundle")
        target = tmp_path / "engine target"
        target.mkdir()
        input_root.joinpath("engine bundle").symlink_to(
            target, target_is_directory=True
        )
    elif mutation == "tokenizer_symlink":
        input_root.joinpath("tokenizer contract.json").unlink()
        target = tmp_path / "tokenizer target.json"
        target.write_text("{}\n", encoding="utf-8")
        input_root.joinpath("tokenizer contract.json").symlink_to(target)
    elif mutation == "engine_file":
        shutil.rmtree(input_root / "engine bundle")
        input_root.joinpath("engine bundle").write_text(
            "not a directory\n", encoding="utf-8"
        )
    else:
        input_root.joinpath("tokenizer contract.json").unlink()
        input_root.joinpath("tokenizer contract.json").mkdir()

    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert message in completed.stderr
    assert not log.exists()
