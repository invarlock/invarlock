from __future__ import annotations

from pathlib import Path

from scripts import runtime_qualification

ROOT = Path(__file__).resolve().parents[2]


def test_qualification_accepts_only_the_core_candidate_distribution() -> None:
    assert runtime_qualification._CANDIDATE_DISTRIBUTION_SOURCES == {
        "invarlock": ("src/invarlock", "invarlock")
    }


def test_qualification_source_inventory_is_core_only() -> None:
    assert runtime_qualification._is_execution_source("src/invarlock/cli/app.py")
    assert runtime_qualification._is_execution_source(
        "scripts/runtime_qualification.py"
    )
    assert not runtime_qualification._is_execution_source("legacy/runtime/provider.py")
    assert not runtime_qualification._is_execution_source("README.md")


def test_candidate_probe_requires_the_core_distribution() -> None:
    assert 'expected.get("invarlock")' in runtime_qualification._CANDIDATE_PROBE
    assert "candidate distribution discovery does not match manifest" in (
        runtime_qualification._CANDIDATE_PROBE
    )


def test_provider_runtime_images_bind_source_identity_and_read_only_execution() -> None:
    for filename in (
        "Dockerfile.gguf",
        "Dockerfile.hf-vision-text",
        "Dockerfile.tensorrt-llm",
    ):
        dockerfile = (ROOT / "runtime" / filename).read_text(encoding="utf-8")
        assert (
            'org.opencontainers.image.revision="${INVARLOCK_SOURCE_COMMIT}"'
            in dockerfile
        )
        assert (
            'dev.invarlock.source-bundle-sha256="${INVARLOCK_SOURCE_BUNDLE_SHA256}"'
            in dockerfile
        )
        assert "--require-hashes" in dockerfile
        assert "--no-deps" in dockerfile


def test_runtime_build_and_qualification_targets_use_authenticated_inputs() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    qualification = (ROOT / "scripts/authenticated_runtime_build.py").read_text(
        encoding="utf-8"
    )
    assert "runtime-image" in makefile
    assert "runtime-qualification-canary" in makefile
    assert "--source-bundle-sha256" in makefile
    assert "source_bundle_sha256" in qualification
    assert "INVARLOCK_ALLOW_NETWORK" in (
        ROOT / "scripts/runtime_qualification.py"
    ).read_text(encoding="utf-8")
