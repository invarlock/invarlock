"""Authenticate reproducible dependency derivation and its failure boundaries."""

from __future__ import annotations

import hashlib
import io
import runpy
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.security import build_cache_free_lm_eval_wheel as wheel_tools
from scripts.security import build_hardened_accelerate_wheel as builder

MODELING = """def load_state_dict(checkpoint_file, device_map=None):
    if checkpoint_file.endswith(".safetensors"):
        return "safe"
    return "bin"

def load_checkpoint_in_model(model, checkpoint):
    checkpoint_files = None
    index_filename = None
    # Original path selection is replaced by descriptor-backed selection.
    # Logic for missing/unexpected keys goes here.
    for checkpoint_file in checkpoint_files:
        model.append(load_state_dict(checkpoint_file))
    return model
"""


def write_wheel(path, files, root):
    payloads = dict(files)
    record = f"{root}/RECORD"
    payloads[record] = wheel_tools._record(payloads, record)
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in payloads.items():
            archive.writestr(name, content)
    return path


@pytest.fixture
def source(tmp_path, monkeypatch):
    path = tmp_path / builder.UPSTREAM_WHEEL_NAME
    root = builder.UPSTREAM_DIST_INFO
    write_wheel(
        path,
        {
            "accelerate/__init__.py": b'__version__ = "1.14.0"\n',
            "accelerate/utils/modeling.py": MODELING.encode(),
            "accelerate/big_modeling.py": b"# Original dispatch implementation\n",
            f"{root}/METADATA": b"Metadata-Version: 2.4\nName: accelerate\nVersion: 1.14.0\nRequires-Dist: torch>=2.0\n\n",
            f"{root}/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
            f"{root}/licenses/LICENSE": b"Original upstream license\n",
        },
        root,
    )
    monkeypatch.setattr(
        builder, "UPSTREAM_WHEEL_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    return path


def authorize_wheel(path, monkeypatch):
    monkeypatch.setattr(
        builder, "DERIVED_WHEEL_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_reproducible_authenticated_build_preserves_unmodified_payload(
    source, tmp_path, monkeypatch
):
    first = builder.build_wheel(source, tmp_path / "one")
    second = builder.build_wheel(source, tmp_path / "two")
    assert first.read_bytes() == second.read_bytes()
    authorize_wheel(first, monkeypatch)
    assert (
        builder.verify_hardened_wheel(first)["upstream_wheel_sha256"]
        == builder.UPSTREAM_WHEEL_SHA256
    )
    with zipfile.ZipFile(first) as derived, zipfile.ZipFile(source) as original:
        assert all(
            info.compress_type == zipfile.ZIP_STORED for info in derived.infolist()
        )
        assert derived.read("accelerate/big_modeling.py") == original.read(
            "accelerate/big_modeling.py"
        )
        assert derived.read(
            f"{builder.DERIVED_DIST_INFO}/licenses/LICENSE"
        ) == original.read(f"{builder.UPSTREAM_DIST_INFO}/licenses/LICENSE")
        assert b"Requires-Dist: torch>=2.0" in derived.read(
            f"{builder.DERIVED_DIST_INFO}/METADATA"
        )
        namespace = {}
        exec(derived.read("accelerate/utils/modeling.py"), namespace)
        checkpoint = tmp_path / "model.safetensors"
        checkpoint.write_bytes(b"data")
        assert namespace["load_checkpoint_in_model"]([], checkpoint) == ["safe"]


def test_verification_rejects_absent_link_changed_hash_and_identity(
    source, tmp_path, monkeypatch
):
    wheel = builder.build_wheel(source, tmp_path / "derived")
    with pytest.raises(builder.DerivationError, match="SHA-256"):
        builder.verify_hardened_wheel(wheel)
    for name in ("absent", "linked"):
        path = tmp_path / name
        if name == "linked":
            path.symlink_to(wheel)
        with pytest.raises(builder.DerivationError, match="regular file"):
            builder.verify_hardened_wheel(path)
    wrong = tmp_path / "wrong.whl"
    write_wheel(
        wrong,
        {
            f"{builder.DERIVED_DIST_INFO}/METADATA": b"Name: unexpected\nVersion: 1.14.0+invarlock.1\n"
        },
        builder.DERIVED_DIST_INFO,
    )
    authorize_wheel(wrong, monkeypatch)
    with pytest.raises(builder.DerivationError, match="identity"):
        builder.verify_hardened_wheel(wrong)


@pytest.mark.parametrize(
    "mutated",
    [
        MODELING.replace("def load_state_dict(", "def other("),
        MODELING.replace('checkpoint_file.endswith(".safetensors")', "False"),
        MODELING.replace("def load_checkpoint_in_model(", "def other("),
        MODELING.replace("    index_filename = None\n", ""),
        MODELING.replace("    # Logic for missing/unexpected keys goes here.\n", ""),
        MODELING.replace(
            "    checkpoint_files = None\n    index_filename = None\n", ""
        ).replace(
            "    # Logic for missing/unexpected keys goes here.\n",
            "    # Logic for missing/unexpected keys goes here.\n    checkpoint_files = None\n    index_filename = None\n",
        ),
    ],
)
def test_unexpected_upstream_loader_shape_cannot_be_patched(mutated):
    with pytest.raises(builder.DerivationError, match="changed|reordered"):
        builder._patch_modeling(mutated.encode())


def test_version_patch_requires_exact_single_identity():
    with pytest.raises(builder.DerivationError, match="version changed"):
        builder._patch_version(b'__version__ = "1.15.0"')


def test_bootstrap_uses_authenticated_existing_or_explicit_input(
    source, tmp_path, monkeypatch
):
    wheelhouse = tmp_path / "wheelhouse"
    reference = builder.build_wheel(source, tmp_path / "reference")
    authorize_wheel(reference, monkeypatch)
    actual = builder.bootstrap(wheelhouse, source=source)
    assert actual.read_bytes() == reference.read_bytes()
    assert builder.bootstrap(wheelhouse) == actual
    actual.write_bytes(b"tampered")
    with pytest.raises(builder.DerivationError, match="SHA-256"):
        builder.bootstrap(wheelhouse, source=source)


def test_bootstrap_download_authentication_cache_and_link_rejection(
    source, tmp_path, monkeypatch
):
    reference = builder.build_wheel(source, tmp_path / "reference")
    authorize_wheel(reference, monkeypatch)
    calls = []

    def download(url, *, timeout):
        calls.append((url, timeout))
        return io.BytesIO(source.read_bytes())

    monkeypatch.setattr(builder.urllib.request, "urlopen", download)
    wheelhouse = tmp_path / "downloaded"
    assert builder.bootstrap(wheelhouse).read_bytes() == reference.read_bytes()
    assert calls == [(builder.UPSTREAM_WHEEL_URL, 30)]
    (wheelhouse / builder.DERIVED_WHEEL_NAME).unlink()
    assert builder.bootstrap(wheelhouse).read_bytes() == reference.read_bytes()
    assert len(calls) == 1
    (wheelhouse / builder.DERIVED_WHEEL_NAME).unlink()
    cache = wheelhouse / ".upstream" / builder.UPSTREAM_WHEEL_NAME
    cache.unlink()
    cache.symlink_to(source)
    with pytest.raises(builder.DerivationError, match="symlink"):
        builder.bootstrap(wheelhouse)
    monkeypatch.setattr(
        builder.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(b"wrong upstream"),
    )
    with pytest.raises(builder.DerivationError, match="downloaded upstream"):
        builder.bootstrap(tmp_path / "bad-download")
    assert not (
        tmp_path / "bad-download" / ".upstream" / builder.UPSTREAM_WHEEL_NAME
    ).exists()


def test_cli_commands_and_errors(source, tmp_path, monkeypatch, capsys):
    destination = tmp_path / "cli"
    assert (
        builder.main(
            [
                "build-wheel",
                "--input",
                str(source),
                "--output-directory",
                str(destination),
            ]
        )
        == 0
    )
    wheel = destination / builder.DERIVED_WHEEL_NAME
    authorize_wheel(wheel, monkeypatch)
    assert builder.main(["verify-wheel", "--input", str(wheel)]) == 0
    assert '"upstream_version": "1.14.0"' in capsys.readouterr().out
    assert (
        builder.main(
            [
                "bootstrap",
                "--input",
                str(source),
                "--output-directory",
                str(destination),
            ]
        )
        == 0
    )
    assert builder.main(["verify-wheel", "--input", str(tmp_path / "absent")]) == 2
    assert "ERROR:" in capsys.readouterr().err


def test_direct_script_entrypoint_fails_closed_without_artifact(tmp_path, monkeypatch):
    script = Path(builder.__file__)
    monkeypatch.syspath_prepend(str(script.parent))
    monkeypatch.setattr(
        sys, "argv", [str(script), "verify-wheel", "--input", str(tmp_path / "absent")]
    )
    with pytest.raises(SystemExit) as status:
        runpy.run_path(str(script), run_name="__main__")
    assert status.value.code == 2


def test_verification_parses_the_authenticated_snapshot(source, tmp_path, monkeypatch):
    wheel = builder.build_wheel(source, tmp_path / "derived")
    authorize_wheel(wheel, monkeypatch)
    original = Path.read_bytes

    def replace_after_read(path):
        payload = original(path)
        if path == wheel:
            path.write_bytes(b"replacement is not a wheel")
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    proof = builder.verify_hardened_wheel(wheel)
    assert proof["wheel_sha256"] == builder.DERIVED_WHEEL_SHA256
    assert original(wheel) == b"replacement is not a wheel"
