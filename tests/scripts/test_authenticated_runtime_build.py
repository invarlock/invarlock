from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import authenticated_runtime_build

ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "scripts" / "authenticated_runtime_build.py"
SOURCE_SCRIPT = ROOT / "scripts" / "qualification_source.py"


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.joinpath("runtime").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test"],
        check=True,
    )
    repository.joinpath("source.txt").write_text("authenticated\n", encoding="utf-8")
    repository.joinpath("runtime", "Dockerfile").write_text(
        "FROM scratch\nARG INVARLOCK_SOURCE_COMMIT\n"
        "ARG INVARLOCK_SOURCE_BUNDLE_SHA256\n"
        'LABEL org.opencontainers.image.revision="${INVARLOCK_SOURCE_COMMIT}" \\\n'
        '  dev.invarlock.source-bundle-sha256="${INVARLOCK_SOURCE_BUNDLE_SHA256}"\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repository, commit


def _bundle(
    tmp_path: Path, repository: Path, commit: str, name: str
) -> tuple[Path, str]:
    bundle = tmp_path / name
    completed = subprocess.run(
        [
            sys.executable,
            str(SOURCE_SCRIPT),
            "create",
            "--repository",
            str(repository),
            "--commit",
            commit,
            "--output",
            str(bundle),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return bundle, json.loads(completed.stdout)["source_bundle_sha256"]


def _fake_engine(tmp_path: Path, name: str = "fake-container-engine") -> Path:
    engine = tmp_path / name
    engine.write_text(
        """#!/usr/bin/env python3
import json
import os
import pathlib
import sys

arguments = sys.argv[1:]
calls = os.environ.get("FAKE_CALLS")
if calls:
    with pathlib.Path(calls).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(arguments) + "\\n")
if arguments[:2] == ["image", "inspect"]:
    image = arguments[2]
    if os.environ.get("FAKE_INSPECT_EXIT") or image == os.environ.get(
        "FAKE_INSPECT_EXIT_IMAGE"
    ):
        print("inspection failed", file=sys.stderr)
        raise SystemExit(int(os.environ.get("FAKE_INSPECT_EXIT", "17")))
    output_image = os.environ.get("FAKE_OUTPUT_IMAGE", "candidate:local")
    commit = os.environ["FAKE_SOURCE_COMMIT"]
    digest = os.environ["FAKE_SOURCE_DIGEST"]
    base_layer = "sha256:" + "a" * 64
    layers = [base_layer]
    identity = "sha256:" + "a" * 64
    if image == output_image:
        identity = os.environ.get("FAKE_INSPECT_IMAGE_ID", os.environ["FAKE_IMAGE_ID"])
        layers.append("sha256:" + "c" * 64)
        if os.environ.get("FAKE_FINAL_BASE_MISMATCH"):
            layers[0] = "sha256:" + "b" * 64
    if image == os.environ.get("FAKE_MISMATCH_IMAGE"):
        commit = "0" * 40
    payload = {"Config": {"Labels": {
        "org.opencontainers.image.revision": commit,
        "dev.invarlock.source-bundle-sha256": digest,
    }}, "RootFS": {"Layers": layers}}
    if not os.environ.get("FAKE_OMIT_INSPECT_IMAGE_ID"):
        payload["Id"] = identity
    if image == output_image and os.environ.get("FAKE_CONTAINERD_IMAGE_ID"):
        manifest = os.environ["FAKE_CONTAINERD_IMAGE_ID"]
        payload["Id"] = manifest
        payload["Descriptor"] = {
            "digest": manifest,
            "annotations": {"config.digest": os.environ["FAKE_IMAGE_ID"]},
        }
    print(json.dumps([payload]))
    raise SystemExit(0)
if arguments and arguments[0] == "build":
    pathlib.Path(os.environ["FAKE_CONTEXT"]).write_bytes(sys.stdin.buffer.read())
    pathlib.Path(os.environ["FAKE_ARGUMENTS"]).write_text(
        json.dumps(arguments), encoding="utf-8"
    )
    image_identity = os.environ.get("FAKE_IMAGE_ID", "sha256:" + "c" * 64)
    pathlib.Path(arguments[arguments.index("--iidfile") + 1]).write_text(
        image_identity, encoding="ascii"
    )
    raise SystemExit(int(os.environ.get("FAKE_BUILD_EXIT", "0")))
raise SystemExit(91)
""",
        encoding="utf-8",
    )
    engine.chmod(0o700)
    return engine


def _command(
    *,
    repository: Path,
    commit: str,
    bundle: Path,
    digest: str,
    engine: Path,
) -> list[str]:
    return [
        sys.executable,
        str(BUILD_SCRIPT),
        "--repository",
        str(repository),
        "--source-commit",
        commit,
        "--source-bundle",
        str(bundle),
        "--source-bundle-sha256",
        digest,
        "--container-engine",
        str(engine),
        "--dockerfile",
        "runtime/Dockerfile",
        "--image",
        "candidate:local",
        "--build-arg",
        "SOURCE_DATE_EPOCH=1",
    ]


def _environment(tmp_path: Path, *, commit: str, digest: str) -> dict[str, str]:
    return {
        **os.environ,
        "FAKE_SOURCE_COMMIT": commit,
        "FAKE_SOURCE_DIGEST": digest,
        "FAKE_CONTEXT": str(tmp_path / "context.tar"),
        "FAKE_ARGUMENTS": str(tmp_path / "arguments.json"),
        "FAKE_CALLS": str(tmp_path / "container-engine-calls.jsonl"),
        "FAKE_IMAGE_ID": "sha256:" + "c" * 64,
    }


@dataclass(frozen=True)
class BuildHarness:
    root: Path
    repository: Path
    commit: str
    bundle: Path
    digest: str
    engine: Path

    @classmethod
    def create(cls, tmp_path: Path) -> BuildHarness:
        repository, commit = _repository(tmp_path)
        bundle, digest = _bundle(tmp_path, repository, commit, "source.tar")
        return cls(tmp_path, repository, commit, bundle, digest, _fake_engine(tmp_path))

    def command(
        self,
        *,
        commit: str | None = None,
        bundle: Path | None = None,
        digest: str | None = None,
    ) -> list[str]:
        return _command(
            repository=self.repository,
            commit=commit or self.commit,
            bundle=bundle or self.bundle,
            digest=digest or self.digest,
            engine=self.engine,
        )

    def environment(
        self,
        *,
        commit: str | None = None,
        digest: str | None = None,
        overrides: dict[str, str] | None = None,
    ) -> dict[str, str]:
        environment = _environment(
            self.root,
            commit=commit or self.commit,
            digest=digest or self.digest,
        )
        environment.update(overrides or {})
        return environment

    def run(
        self,
        command: list[str] | None = None,
        *,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command or self.command(),
            check=False,
            capture_output=True,
            text=True,
            env=environment or self.environment(),
        )


@pytest.fixture
def harness(tmp_path: Path) -> BuildHarness:
    return BuildHarness.create(tmp_path)


def test_build_consumes_the_authenticated_archive_not_the_dirty_checkout(
    harness: BuildHarness,
) -> None:
    harness.repository.joinpath("source.txt").write_text(
        "dirty checkout\n", encoding="utf-8"
    )

    statement = harness.root / "build-statement.json"
    completed = harness.run([*harness.command(), "--statement", str(statement)])

    assert completed.returncode == 0, completed.stderr
    with tarfile.open(harness.root / "context.tar", mode="r:") as archive:
        source = archive.extractfile("source.txt")
        assert source is not None
        assert source.read() == b"authenticated\n"
    arguments = json.loads(
        (harness.root / "arguments.json").read_text(encoding="utf-8")
    )
    assert arguments[-1] == "-"
    assert f"INVARLOCK_SOURCE_COMMIT={harness.commit}" in arguments
    assert f"INVARLOCK_SOURCE_BUNDLE_SHA256={harness.digest}" in arguments
    assert json.loads(statement.read_text(encoding="utf-8")) == {
        "base_image": None,
        "build_arguments": {"SOURCE_DATE_EPOCH": "1"},
        "dockerfile": {
            "path": "runtime/Dockerfile",
            "sha256": "sha256:"
            + hashlib.sha256(
                (harness.repository / "runtime" / "Dockerfile").read_bytes()
            ).hexdigest(),
        },
        "format_version": "invarlock/runtime-image-build-v1",
        "image": "candidate:local",
        "ok": True,
        "platform": None,
        "runtime_image_id": "sha256:" + "c" * 64,
        "source_bundle_sha256": harness.digest,
        "source_commit": harness.commit,
    }


def test_docker_build_disables_nondeterministic_default_provenance(
    tmp_path: Path,
) -> None:
    repository, commit = _repository(tmp_path)
    bundle, digest = _bundle(tmp_path, repository, commit, "source.tar")
    engine = _fake_engine(tmp_path, "docker")

    completed = subprocess.run(
        _command(
            repository=repository,
            commit=commit,
            bundle=bundle,
            digest=digest,
            engine=engine,
        ),
        check=False,
        capture_output=True,
        text=True,
        env=_environment(tmp_path, commit=commit, digest=digest),
    )

    assert completed.returncode == 0, completed.stderr
    arguments = json.loads((tmp_path / "arguments.json").read_text(encoding="utf-8"))
    assert arguments[:3] == ["build", "--provenance=false", "--iidfile"]


def test_containerd_manifest_identity_is_bound_to_the_recorded_config_digest(
    harness: BuildHarness,
) -> None:
    statement = harness.root / "build-statement.json"
    manifest = "sha256:" + "d" * 64

    completed = harness.run(
        [*harness.command(), "--statement", str(statement)],
        environment=harness.environment(
            overrides={"FAKE_CONTAINERD_IMAGE_ID": manifest}
        ),
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(statement.read_text(encoding="utf-8"))["runtime_image_id"] == (
        manifest
    )


def test_build_statement_is_no_clobber(harness: BuildHarness) -> None:
    statement = harness.root / "build-statement.json"
    statement.write_text("caller-owned\n", encoding="utf-8")

    completed = harness.run([*harness.command(), "--statement", str(statement)])

    assert completed.returncode != 0
    assert "destination already exists" in completed.stderr
    assert statement.read_text(encoding="utf-8") == "caller-owned\n"


def test_build_statement_rejects_a_symlinked_parent(harness: BuildHarness) -> None:
    real_parent = harness.root / "real-statements"
    real_parent.mkdir()
    linked_parent = harness.root / "linked-statements"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    completed = harness.run(
        [
            *harness.command(),
            "--statement",
            str(linked_parent / "build-statement.json"),
        ]
    )

    assert completed.returncode != 0
    assert "parent must be one real directory" in completed.stderr
    assert not (real_parent / "build-statement.json").exists()


def test_build_statement_requires_an_existing_parent(harness: BuildHarness) -> None:
    statement = harness.root / "missing" / "build-statement.json"

    completed = harness.run([*harness.command(), "--statement", str(statement)])

    assert completed.returncode != 0
    assert "parent is unavailable" in completed.stderr
    assert not statement.exists()


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses directory write permissions",
)
def test_build_statement_rejects_an_unwritable_destination(
    harness: BuildHarness,
) -> None:
    statement_root = harness.root / "statements"
    statement_root.mkdir(mode=0o500)
    statement = statement_root / "build-statement.json"

    try:
        completed = harness.run([*harness.command(), "--statement", str(statement)])
    finally:
        statement_root.chmod(0o700)

    assert completed.returncode != 0
    assert "could not be published" in completed.stderr
    assert not statement.exists()


def test_build_rejects_an_arbitrary_nonexistent_source_commit(
    harness: BuildHarness,
) -> None:
    nonexistent = "b" * 40

    completed = harness.run(
        harness.command(commit=nonexistent),
        environment=harness.environment(commit=nonexistent),
    )

    assert completed.returncode != 0
    assert "source reference does not identify one Git commit" in completed.stderr
    assert not (harness.root / "arguments.json").exists()


def test_build_rejects_a_bundle_from_a_different_commit(tmp_path: Path) -> None:
    repository, first_commit = _repository(tmp_path)
    repository.joinpath("source.txt").write_text("second commit\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-qam", "second"], check=True
    )
    second_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    bundle, digest = _bundle(tmp_path, repository, second_commit, "second.tar")
    engine = _fake_engine(tmp_path)

    completed = subprocess.run(
        _command(
            repository=repository,
            commit=first_commit,
            bundle=bundle,
            digest=digest,
            engine=engine,
        ),
        check=False,
        capture_output=True,
        text=True,
        env=_environment(tmp_path, commit=first_commit, digest=digest),
    )

    assert completed.returncode != 0
    assert "source bundle does not bind the selected commit" in completed.stderr
    assert not (tmp_path / "arguments.json").exists()


@pytest.mark.parametrize(
    ("attack", "expected_error", "engine_used"),
    (
        (
            "base_label_mismatch",
            "source label 'org.opencontainers.image.revision' does not match",
            False,
        ),
        (
            "reserved_source_argument",
            "source identity build arguments are driver-owned",
            False,
        ),
        (
            "missing_dockerfile",
            "runtime Dockerfile is absent from source bundle",
            False,
        ),
        (
            "final_label_mismatch",
            "source label 'org.opencontainers.image.revision' does not match",
            True,
        ),
        (
            "final_base_mismatch",
            "filesystem does not derive from the authenticated base image",
            True,
        ),
        (
            "invalid_image_identity",
            "container build recorded an invalid image identity",
            True,
        ),
        (
            "different_tag_identity",
            "does not identify the recorded build result",
            True,
        ),
        (
            "malformed_tag_identity",
            "inspection is missing its image identity",
            True,
        ),
        (
            "missing_tag_identity",
            "inspection is missing its image identity",
            True,
        ),
        (
            "final_inspection_failure",
            "is unavailable for source-label verification: inspection failed",
            True,
        ),
        ("invalid_platform", "runtime build platform is invalid", False),
        (
            "mutable_base",
            "base runtime image must use a canonical repository@sha256 reference",
            False,
        ),
        (
            "raw_config_base",
            "base runtime image must use a canonical repository@sha256 reference",
            False,
        ),
        (
            "base_inspection_failure",
            "is unavailable for source-label verification: inspection failed",
            False,
        ),
    ),
)
def test_build_rejects_source_and_engine_boundary_attacks(
    harness: BuildHarness,
    attack: str,
    expected_error: str,
    engine_used: bool,
) -> None:
    command = harness.command()
    overrides: dict[str, str] = {}
    base = "registry.example/base@sha256:" + "a" * 64
    if attack == "base_label_mismatch":
        command.extend(
            (
                "--require-base-source-labels",
                base,
                "--build-arg",
                f"RUNTIME_BASE_IMAGE={base}",
            )
        )
        overrides["FAKE_MISMATCH_IMAGE"] = base
    elif attack == "reserved_source_argument":
        command.extend(("--build-arg", f"INVARLOCK_SOURCE_COMMIT={harness.commit}"))
    elif attack == "missing_dockerfile":
        command[command.index("runtime/Dockerfile")] = "runtime/missing.Dockerfile"
    elif attack == "final_label_mismatch":
        overrides["FAKE_MISMATCH_IMAGE"] = "candidate:local"
    elif attack == "final_base_mismatch":
        command.extend(
            (
                "--require-base-source-labels",
                base,
                "--build-arg",
                f"RUNTIME_BASE_IMAGE={base}",
            )
        )
        overrides["FAKE_FINAL_BASE_MISMATCH"] = "1"
    elif attack == "invalid_image_identity":
        overrides["FAKE_IMAGE_ID"] = "candidate:local"
    elif attack == "different_tag_identity":
        overrides["FAKE_INSPECT_IMAGE_ID"] = "sha256:" + "d" * 64
    elif attack == "malformed_tag_identity":
        overrides["FAKE_INSPECT_IMAGE_ID"] = "candidate:local"
    elif attack == "missing_tag_identity":
        overrides["FAKE_OMIT_INSPECT_IMAGE_ID"] = "1"
    elif attack == "final_inspection_failure":
        overrides["FAKE_INSPECT_EXIT_IMAGE"] = "candidate:local"
    elif attack == "invalid_platform":
        command.extend(("--platform", "darwin/arm64"))
    elif attack == "mutable_base":
        command.extend(("--require-base-source-labels", "registry.example/base:latest"))
    elif attack == "raw_config_base":
        raw_config = "sha256:" + "a" * 64
        command.extend(
            (
                "--require-base-source-labels",
                raw_config,
                "--build-arg",
                f"RUNTIME_BASE_IMAGE={raw_config}",
            )
        )
    elif attack == "base_inspection_failure":
        command.extend(
            (
                "--require-base-source-labels",
                base,
                "--build-arg",
                f"RUNTIME_BASE_IMAGE={base}",
            )
        )
        overrides["FAKE_INSPECT_EXIT"] = "17"

    completed = harness.run(
        command, environment=harness.environment(overrides=overrides)
    )

    assert completed.returncode != 0
    assert expected_error in completed.stderr
    assert (harness.root / "arguments.json").exists() is engine_used


def test_build_binds_the_authenticated_base_to_the_from_argument(
    harness: BuildHarness,
) -> None:
    base = "registry.example/invarlock/runtime@sha256:" + "a" * 64
    statement = harness.root / "build-statement.json"
    command = [
        *harness.command(),
        "--require-base-source-labels",
        base,
        "--build-arg",
        f"RUNTIME_BASE_IMAGE={base}",
        "--statement",
        str(statement),
    ]

    completed = harness.run(command)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(statement.read_text(encoding="utf-8"))
    assert payload["base_image"] == base
    assert payload["build_arguments"]["RUNTIME_BASE_IMAGE"] == base


@pytest.mark.parametrize(
    "consumed",
    [None, "registry.example/invarlock/runtime@sha256:" + "b" * 64],
)
def test_build_rejects_a_missing_or_different_consumed_base_before_engine(
    harness: BuildHarness, consumed: str | None
) -> None:
    authenticated = "registry.example/invarlock/runtime@sha256:" + "a" * 64
    command = [
        *harness.command(),
        "--require-base-source-labels",
        authenticated,
    ]
    if consumed is not None:
        command.extend(("--build-arg", f"RUNTIME_BASE_IMAGE={consumed}"))

    completed = harness.run(command)

    assert completed.returncode != 0
    assert (
        "authenticated base runtime image must match the RUNTIME_BASE_IMAGE "
        "build argument" in completed.stderr
    )
    assert not (harness.root / "container-engine-calls.jsonl").exists()


def test_build_accepts_a_tagged_named_digest_for_a_from_argument(
    harness: BuildHarness,
) -> None:
    base = "python:3.12-slim@sha256:" + "a" * 64

    completed = harness.run(
        [*harness.command(), "--build-arg", f"WHEEL_BUILD_BASE={base}"]
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "argument_name",
    [
        "RUNTIME_BASE_IMAGE",
        "RUNTIME_BUILD_BASE_IMAGE",
        "WHEEL_BUILD_BASE",
        "LLAMA_CPP_BUILD_BASE",
    ],
)
def test_build_rejects_raw_ids_for_dockerfile_from_arguments_before_engine(
    harness: BuildHarness, argument_name: str
) -> None:
    command = [
        *harness.command(),
        "--build-arg",
        f"{argument_name}=sha256:" + "a" * 64,
    ]

    completed = harness.run(command)

    assert completed.returncode != 0
    assert (
        f"runtime build argument {argument_name} must use a canonical "
        "repository@sha256 reference" in completed.stderr
    )
    assert not (harness.root / "container-engine-calls.jsonl").exists()


@pytest.mark.parametrize(
    "image",
    [
        "candidate",
        "sha256:" + "a" * 64,
        "registry.example/invarlock/runtime@sha256:" + "a" * 64,
        "Registry.example/invarlock/runtime:local",
        "registry.example/invarlock/runtime:bad tag",
    ],
)
def test_build_rejects_non_tag_output_images_before_engine(
    harness: BuildHarness, image: str
) -> None:
    command = harness.command()
    command[command.index("--image") + 1] = image

    completed = harness.run(command)

    assert completed.returncode != 0
    assert "runtime image name must be a canonical repository tag" in completed.stderr
    assert not (harness.root / "container-engine-calls.jsonl").exists()


def test_build_propagates_engine_failure_without_inspecting_a_candidate(
    tmp_path: Path,
) -> None:
    repository, commit = _repository(tmp_path)
    bundle, digest = _bundle(tmp_path, repository, commit, "source.tar")
    engine = _fake_engine(tmp_path)
    environment = _environment(tmp_path, commit=commit, digest=digest)
    environment["FAKE_BUILD_EXIT"] = "23"

    completed = subprocess.run(
        _command(
            repository=repository,
            commit=commit,
            bundle=bundle,
            digest=digest,
            engine=engine,
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 23
    assert (tmp_path / "arguments.json").exists()


@pytest.mark.parametrize("value", ("", " docker", "docker\n"))
def test_engine_rejects_noncanonical_names(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        authenticated_runtime_build.shutil, "which", lambda _value: None
    )
    with pytest.raises(SystemExit, match="container engine is invalid"):
        authenticated_runtime_build._engine(value)


def test_engine_requires_an_available_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        authenticated_runtime_build.shutil, "which", lambda _value: None
    )
    with pytest.raises(SystemExit, match="container engine is unavailable"):
        authenticated_runtime_build._engine("missing-engine")


@pytest.mark.parametrize("value", ("", "/Dockerfile", "../Dockerfile", "a//Dockerfile"))
def test_dockerfile_path_must_be_canonical_and_archive_relative(value: str) -> None:
    with pytest.raises(SystemExit, match="canonical archive-relative"):
        authenticated_runtime_build._relative_dockerfile(value)


def test_build_arguments_reject_invalid_and_repeated_names() -> None:
    with pytest.raises(SystemExit, match="build argument is invalid"):
        authenticated_runtime_build._build_arguments(["lowercase=value"])
    with pytest.raises(SystemExit, match="is repeated"):
        authenticated_runtime_build._build_arguments(["VALUE=one", "VALUE=two"])


@pytest.mark.parametrize(
    ("raw", "message"),
    (
        (b"x" * (1024 * 1024 + 1), "size limit"),
        (b'{"Id":"one","Id":"two"}', "duplicate key"),
        (b"not-json", "strict JSON"),
        (b"[]", "identify one image"),
        (b"[{},{}]", "identify one image"),
        (b"1", "identify one image"),
    ),
)
def test_image_inspection_parser_rejects_ambiguous_payloads(
    raw: bytes, message: str
) -> None:
    with pytest.raises(SystemExit, match=message):
        authenticated_runtime_build._inspect_payload(raw)


def _inspect_payload(**updates: object) -> dict[str, object]:
    digest = "sha256:" + "a" * 64
    payload: dict[str, object] = {
        "Id": digest,
        "Config": {"Labels": {"label": "value"}},
        "RootFS": {"Layers": [digest]},
    }
    payload.update(updates)
    return payload


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (_inspect_payload(Descriptor="bad"), "invalid descriptor"),
        (
            _inspect_payload(Descriptor={"digest": "sha256:" + "b" * 64}),
            "descriptor does not match",
        ),
        (
            _inspect_payload(
                Descriptor={
                    "digest": "sha256:" + "a" * 64,
                    "annotations": "bad",
                }
            ),
            "annotations are invalid",
        ),
        (
            _inspect_payload(
                Descriptor={
                    "digest": "sha256:" + "a" * 64,
                    "annotations": {"config.digest": "bad"},
                }
            ),
            "config digest is invalid",
        ),
        (_inspect_payload(Config=None), "missing Config"),
        (_inspect_payload(Config={"Labels": {"label": 1}}), "missing source labels"),
        (_inspect_payload(RootFS="bad"), "invalid RootFS"),
        (_inspect_payload(RootFS={"Layers": ["bad"]}), "missing filesystem layers"),
    ),
)
def test_image_metadata_rejects_malformed_engine_inventory(
    payload: dict[str, object],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        authenticated_runtime_build.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps([payload]).encode(), stderr=b""
        ),
    )
    with pytest.raises(SystemExit, match=message):
        authenticated_runtime_build._image_metadata("engine", "image")


@pytest.mark.parametrize("payload", (b"", b"not-a-digest", b"\xff", b"x" * 129))
def test_built_image_identity_requires_a_small_ascii_digest(
    tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "identity"
    path.write_bytes(payload)
    with pytest.raises(SystemExit, match="invalid image identity"):
        authenticated_runtime_build._built_image_identity(path)
    with pytest.raises(SystemExit, match="did not record"):
        authenticated_runtime_build._built_image_identity(tmp_path / "missing")


def test_deterministic_build_options_apply_only_to_docker() -> None:
    assert authenticated_runtime_build._deterministic_build_options("/bin/docker") == (
        "--provenance=false",
    )
    assert authenticated_runtime_build._deterministic_build_options("/bin/podman") == ()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("INVARLOCK_RUN_CONTAINER_SMOKE") != "1",
    reason="set INVARLOCK_RUN_CONTAINER_SMOKE=1 to exercise a real container engine",
)
def test_real_container_engine_builds_from_the_authenticated_tar_context(
    tmp_path: Path,
) -> None:
    engine = shutil.which(os.environ.get("INVARLOCK_CONTAINER_ENGINE", "docker"))
    if engine is None:
        pytest.fail("container engine was explicitly requested but is unavailable")
    available = subprocess.run(
        [engine, "info"], check=False, capture_output=True, timeout=30
    )
    if available.returncode != 0:
        pytest.fail(
            "container engine was explicitly requested but its daemon is unavailable"
        )
    repository, commit = _repository(tmp_path)
    bundle, digest = _bundle(tmp_path, repository, commit, "source.tar")
    images = (
        f"invarlock-authenticated-runtime-build-a:{os.getpid()}",
        f"invarlock-authenticated-runtime-build-b:{os.getpid()}",
    )
    command = _command(
        repository=repository,
        commit=commit,
        bundle=bundle,
        digest=digest,
        engine=Path(engine),
    )
    command[command.index("candidate:local")] = images[0]

    try:
        first = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert first.returncode == 0, first.stderr
        second_command = list(command)
        second_command[second_command.index(images[0])] = images[1]
        second = subprocess.run(
            second_command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert second.returncode == 0, second.stderr
        image_ids = [
            subprocess.run(
                [engine, "image", "inspect", "--format", "{{.Id}}", image],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
            for image in images
        ]
        assert image_ids[0] == image_ids[1]
        inspected = subprocess.run(
            [engine, "image", "inspect", images[0]],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        labels = json.loads(inspected.stdout)[0]["Config"]["Labels"]
        assert labels["org.opencontainers.image.revision"] == commit
        assert labels["dev.invarlock.source-bundle-sha256"] == digest
    finally:
        subprocess.run(
            [engine, "image", "rm", "--force", *images],
            check=False,
            capture_output=True,
            timeout=30,
        )
