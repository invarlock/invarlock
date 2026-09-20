"""Real-journey controls fail closed before launching external execution."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from examples.integrations import modelkit_real_journey as journey


def test_binary_digest_is_checked_before_execution(tmp_path):
    binary = tmp_path / "kit"
    binary.write_bytes(b"unreviewed executable")
    with pytest.raises(ValueError, match="checksum"):
        journey.check_kit(binary, "0" * 64, tmp_path)


def test_nonmatching_version_is_rejected(tmp_path, monkeypatch):
    binary = tmp_path / "kit"
    binary.write_bytes(b"reviewed executable")
    monkeypatch.setattr(journey, "command", lambda *a, **k: "Version: 1.16.0")
    with pytest.raises(ValueError, match="1.15.0"):
        journey.check_kit(
            binary, hashlib.sha256(binary.read_bytes()).hexdigest(), tmp_path
        )


def test_recipient_transfer_copies_without_hardlinks(tmp_path):
    publisher = tmp_path / "publisher"
    publisher.mkdir()
    (publisher / "blob").write_bytes(b"publisher bytes")
    recipient = tmp_path / "recipient"
    journey.transfer(publisher, recipient)
    (publisher / "blob").write_bytes(b"later publisher mutation")
    assert (recipient / "blob").read_bytes() == b"publisher bytes"
    assert (publisher / "blob").stat().st_ino != (recipient / "blob").stat().st_ino


def test_transfer_rejects_symlink(tmp_path):
    publisher = tmp_path / "publisher"
    publisher.mkdir()
    (publisher / "link").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink"):
        journey.transfer(publisher, tmp_path / "recipient")


def test_smoke_rejects_failed_acceptance_before_loading(tmp_path):
    with pytest.raises(ValueError, match="acceptance"):
        journey.inference_smoke(
            tmp_path / "recipient.json",
            {"accepted": False},
            tmp_path / "backend",
            "0" * 64,
            tmp_path,
            prompt="Hello",
            tokens=1,
        )


def test_command_records_failed_output(tmp_path):
    with pytest.raises(RuntimeError, match="exit 7"):
        journey.command(
            [sys.executable, "-c", "print('failure'); raise SystemExit(7)"],
            tmp_path / "run",
            tmp_path,
        )
    assert "failure" in (tmp_path / "run.stdout").read_text()
    assert journey.read_json(tmp_path / "run.command.json")["exit_code"] == 7


@pytest.mark.parametrize("engine", ["docker", "podman"])
@pytest.mark.parametrize("bare_id", [False, True])
def test_smoke_binds_exact_unpacked_file_and_backend(
    tmp_path, monkeypatch, engine, bare_id
):

    model = tmp_path / "model"
    model.mkdir()
    artifact = model / "model.gguf"
    artifact.write_bytes(b"GGUF synthetic fixture")
    request = tmp_path / "recipient.json"
    journey.write_json(
        request,
        {
            "sides": {
                "subject": {
                    "candidate": "model",
                    "artifact_file": "model.gguf",
                    "content_digest": journey.digest(artifact),
                }
            }
        },
    )
    calls = []
    image = "sha256:" + "a" * 64

    def command(argv, *args, **kwargs):
        calls.append(argv)
        if "inspect" in argv:
            return json.dumps(
                [{"Id": image.removeprefix("sha256:") if bare_id else image}]
            )
        return " Paris" if "--model" in argv else "reviewed-version"

    monkeypatch.setattr(journey, "command", command)
    result = journey.inference_smoke(
        request,
        {"accepted": True},
        image,
        "/bin/llama-completion",
        tmp_path,
        prompt="The capital of France is",
        tokens=2,
        container_engine=engine,
    )
    assert result["generated_text"] == " Paris"
    assert "/model/model.gguf" in calls[2]
    assert calls[2][calls[2].index("--n-gpu-layers") + 1] == "0"
    assert calls[2][calls[2].index("--network") + 1] == "none"
    assert calls[2][calls[2].index("--user") + 1] == "65534:65534"
    assert result["image_id"] == image
    assert result["container_engine"] == engine
    assert calls[0] == [engine, "image", "inspect", image]
    assert all(call[0] == engine for call in calls)
    for call in calls[1:]:
        assert call[:3] == [engine, "run", "--rm"]
        assert "--read-only" in call
        assert call[call.index("--cap-drop") + 1] == "ALL"
        assert call[call.index("--security-opt") + 1] == "no-new-privileges"
        assert call[call.index("--memory") + 1] == "12g"
        assert call[call.index("--cpus") + 1] == "4"
        assert call[call.index("--pids-limit") + 1] == "128"
        assert call[call.index("--mount") + 1] == (
            f"type=bind,source={model},target=/model,readonly"
        )
        assert call[call.index("--entrypoint") + 2] == image
    artifact.write_bytes(b"changed file")
    with pytest.raises(ValueError, match="changed before"):
        journey.inference_smoke(
            request,
            {"accepted": True},
            image,
            "/bin/llama-completion",
            tmp_path,
            prompt="Hi",
            tokens=1,
        )


@pytest.mark.parametrize("engine", ["docker", "podman"])
@pytest.mark.parametrize("image", ["latest", "model:v1", "sha256:bad", "a" * 64])
def test_smoke_rejects_mutable_runtime(tmp_path, image, engine):
    with pytest.raises(ValueError, match="immutable"):
        journey.inference_smoke(
            tmp_path / "request.json",
            {"accepted": True},
            image,
            "/bin/llama-completion",
            tmp_path,
            prompt="Hi",
            tokens=1,
            container_engine=engine,
        )


def test_policy_rejection_does_not_authorize_loading_untrusted_bytes(tmp_path):
    with pytest.raises(ValueError, match="acceptance"):
        journey.inference_smoke(
            tmp_path / "request.json",
            {"accepted": False, "technical_integrity_ok": False, "exit_code": 1},
            "sha256:" + "a" * 64,
            "/bin/llama-completion",
            tmp_path,
            prompt="Hi",
            tokens=1,
            allow_policy_rejection=True,
        )


def test_run_refuses_reusing_output_directory(tmp_path):
    with pytest.raises(FileExistsError):
        journey.run(SimpleNamespace(output=tmp_path))


@pytest.mark.parametrize("engine", ["containerd", "auto", "/usr/bin/docker", ""])
def test_smoke_rejects_unsupported_engine_before_execution(tmp_path, engine):
    with pytest.raises(ValueError, match="container engine"):
        journey.inference_smoke(
            tmp_path / "recipient.json",
            {"accepted": True},
            "sha256:" + "a" * 64,
            "/bin/llama-completion",
            tmp_path,
            prompt="Hi",
            tokens=1,
            container_engine=engine,
        )


@pytest.mark.parametrize("engine", ["docker", "podman"])
@pytest.mark.parametrize(
    "inspection",
    [
        [],
        {},
        [{"Id": "sha256:" + "b" * 64}],
        [{"Id": "a" * 63}],
        [{"Id": "sha256:" + "a" * 64 + "\n"}],
        [{"Id": None}],
        [{}],
        [{"Id": "a" * 64}, {"Id": "a" * 64}],
        [None],
    ],
)
def test_smoke_rejects_unbound_image_inspection(
    tmp_path, monkeypatch, engine, inspection
):
    model = tmp_path / "model"
    model.mkdir()
    artifact = model / "model.gguf"
    artifact.write_bytes(b"GGUF synthetic fixture")
    request = tmp_path / "recipient.json"
    journey.write_json(
        request,
        {
            "sides": {
                "subject": {
                    "candidate": "model",
                    "artifact_file": "model.gguf",
                    "content_digest": journey.digest(artifact),
                }
            }
        },
    )
    calls = []

    def command(argv, *args, **kwargs):
        calls.append(argv)
        return json.dumps(inspection)

    monkeypatch.setattr(journey, "command", command)
    with pytest.raises(ValueError, match="image.*identity"):
        journey.inference_smoke(
            request,
            {"accepted": True},
            "sha256:" + "a" * 64,
            "/bin/llama-completion",
            tmp_path,
            prompt="Hi",
            tokens=1,
            container_engine=engine,
        )
    assert calls == [[engine, "image", "inspect", "sha256:" + "a" * 64]]


@pytest.mark.parametrize("engine", ["docker", "podman"])
def test_cli_passes_selected_engine_to_journey(tmp_path, monkeypatch, engine):
    observed = []

    def run(args):
        observed.append(args.container_engine)
        return {"accepted": {"accepted": True, "exit_code": 0}}

    monkeypatch.setattr(journey, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "modelkit_real_journey",
            "--request",
            "request.json",
            "--kit",
            "kit",
            "--kit-sha256",
            "a" * 64,
            "--python",
            sys.executable,
            "--smoke-image",
            "sha256:" + "a" * 64,
            "--output",
            str(tmp_path),
            "--container-engine",
            engine,
        ],
    )
    assert journey.main() == 0
    assert observed == [engine]


def test_cli_rejects_unsupported_engine(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["modelkit_real_journey", "--container-engine", "auto"]
    )
    with pytest.raises(SystemExit) as exc:
        journey.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("engine", ["docker", "podman"])
def test_missing_engine_does_not_fall_back(tmp_path, monkeypatch, engine):
    model = tmp_path / "model"
    model.mkdir()
    artifact = model / "model.gguf"
    artifact.write_bytes(b"GGUF synthetic fixture")
    request = tmp_path / "recipient.json"
    journey.write_json(
        request,
        {
            "sides": {
                "subject": {
                    "candidate": "model",
                    "artifact_file": "model.gguf",
                    "content_digest": journey.digest(artifact),
                }
            }
        },
    )
    calls = []

    def unavailable(argv, *args, **kwargs):
        calls.append(argv)
        raise FileNotFoundError(engine)

    monkeypatch.setattr(journey, "command", unavailable)
    with pytest.raises(FileNotFoundError, match=engine):
        journey.inference_smoke(
            request,
            {"accepted": True},
            "sha256:" + "a" * 64,
            "/bin/llama-completion",
            tmp_path,
            prompt="Hi",
            tokens=1,
            container_engine=engine,
        )
    assert calls == [[engine, "image", "inspect", "sha256:" + "a" * 64]]


@pytest.fixture
def real_native_journey(tmp_path):
    """A fake package CLI transports a genuinely signed native fixture."""
    from pathlib import Path

    from examples.integrations import modelkit_handoff as handoff
    from examples.run_acceptance_handoff import run_handoff
    from invarlock.core.checkpoint_identity import checkpoint_tree_sha256

    source = tmp_path / "source"
    run_handoff(source)
    incoming = source / "handoff"
    recipient = source / "recipient"
    anchors = journey.read_json(recipient / "trust/technical-anchors.json")
    request = {
        "format": "invarlock/example-modelkit-recipient-v1",
        "sides": {
            role: {
                "candidate": str(incoming / "artifacts" / role),
                "content_digest": checkpoint_tree_sha256(incoming / "artifacts" / role),
            }
            for role in ("baseline", "subject")
        },
        "evidence": str(incoming / "evidence"),
        "technical_policy": str(incoming / "policy/acceptance.json"),
        "technical_anchors": {
            key: anchors[key]
            for key in (
                "artifact_digests",
                "schedule_digest",
                "runtime_digests",
                "evidence_signer_fingerprint",
            )
        },
        "envelope": str(incoming / "acceptance.dsse.json"),
        "recipient_policy": str(recipient / "policy.json"),
        "trusted_public_keys": {
            anchors["envelope_signer_fingerprint"]: str(
                recipient / "trust/envelope-signer.public.pem"
            )
        },
    }
    request_path = tmp_path / "source-recipient.json"
    journey.write_json(request_path, request)
    kit = tmp_path / "kit"
    # This subprocess implements only the exercised local pack/inspect/unpack
    # protocol. Production package verification checks every emitted OCI blob.
    kit.write_text(
        f"#!{sys.executable}\n"
        + """
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

args = sys.argv[1:]
if args == ["version"]:
    print("Version: 1.15.0 COMMIT")
    raise SystemExit(0)
assert args[:1] == ["--config"] and args[2:4] == ["--progress", "none"]
store = Path(args[1]) / "storage"
blobs = store / "blobs" / "sha256"
blobs.mkdir(parents=True, exist_ok=True)
args = args[4:]
tags_path = store / "tags.json"
tags = json.loads(tags_path.read_text()) if tags_path.exists() else {}

def blob(data):
    digest = hashlib.sha256(data).hexdigest()
    (blobs / digest).write_bytes(data)
    return {"digest": "sha256:" + digest, "size": len(data)}

def document(value):
    return blob(json.dumps(value, sort_keys=True).encode())

if args[0] == "pack":
    context = Path(args[1])
    assert args[2] == "--tag" and args[4:] == ["--compression", "none"]
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for path in sorted((context / "model").rglob("*")):
            if path.is_file():
                info = tarfile.TarInfo(path.relative_to(context).as_posix())
                data = path.read_bytes()
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    layer = blob(archive.getvalue())
    layer["mediaType"] = MODEL_TAR
    config = document({
        "manifestVersion": "1.0.0",
        "package": {"description": (context / "Kitfile").read_text()},
        "model": {"path": "model", "digest": layer["digest"], "diffId": layer["digest"]},
    })
    config["mediaType"] = CONFIG_MEDIA
    manifest = document({
        "schemaVersion": 2, "mediaType": MANIFEST_MEDIA,
        "artifactType": ARTIFACT_MEDIA, "config": config, "layers": [layer],
    })
    tags[args[3]] = manifest["digest"]
    tags_path.write_text(json.dumps(tags))
elif args[0] == "inspect":
    print(json.dumps({"digest": tags[args[1]]}))
elif args[0] == "unpack":
    assert args[2:5] == ["--filter", "model", "--dir"]
    selected = args[1].split("@", 1)[1]
    manifest = json.loads((blobs / selected.removeprefix("sha256:")).read_bytes())
    layer = blobs / manifest["layers"][0]["digest"].removeprefix("sha256:")
    with tarfile.open(layer) as tar:
        tar.extractall(args[5], filter="data")
else:
    raise AssertionError(args)
""".replace("COMMIT", journey.KIT_COMMIT)
        .replace("MODEL_TAR", repr(handoff.MODEL_TAR))
        .replace("CONFIG_MEDIA", repr(handoff.CONFIG_MEDIA))
        .replace("MANIFEST_MEDIA", repr(handoff.MANIFEST_MEDIA))
        .replace("ARTIFACT_MEDIA", repr(handoff.ARTIFACT_MEDIA))
    )
    kit.chmod(0o700)
    return SimpleNamespace(
        request=request_path,
        kit=kit,
        kit_sha256=journey.digest(kit),
        python=Path(sys.executable),
        output=tmp_path / "journey",
        smoke_image="sha256:" + "a" * 64,
        container_llama="/bin/llama-completion",
        prompt="Hello",
        tokens=1,
        container_engine="podman",
        smoke_on_policy_rejection=False,
    )


@pytest.mark.parametrize("permit_smoke", [False, True])
def test_run_transfers_native_evidence_and_checks_real_recipient(
    real_native_journey, monkeypatch, permit_smoke
):
    from pathlib import Path

    args = real_native_journey
    args.smoke_on_policy_rejection = permit_smoke
    smoke_calls = []

    def smoke(request, decision, image, binary, logs, **options):
        # Native tree fixtures exercise transport and signed acceptance only.
        # Actual model inference is covered by separate GGUF runtime journeys.
        smoke_calls.append((request, decision, options))
        assert decision["technical_integrity_ok"]
        assert decision["envelope_authenticated"] and decision["receipt_authenticated"]
        assert decision["envelope_evidence_bound"]
        assert options["container_engine"] == "podman"
        return {"executed": True, "fixture_inference": True}

    monkeypatch.setattr(journey, "inference_smoke", smoke)
    result = journey.run(args)
    assert result["accepted"]["accepted"] is False  # Fixed fixture envelope expired.
    assert result["accepted"]["exit_code"] == 1
    assert {
        name: value["exit_code"] for name, value in result["scenarios"].items()
    } == {
        "repacked_same_contents": 1,
        "wrong_runtime": 2,
        "revoked_signer": 1,
        "altered_evidence": 2,
        "missing_package": 2,
        "candidate_changed": 2,
    }
    assert bool(smoke_calls) == permit_smoke
    if not permit_smoke:
        assert result["inference"] == {
            "executed": False,
            "reason": "recipient policy rejected",
        }
    assert journey.read_json(args.output / "result.json") == result
    assert args.output.stat().st_mode & 0o777 == 0o700
    request = journey.read_json(args.output / "recipient/recipient.json")
    original = journey.read_json(args.request)
    assert request["technical_anchors"] == original["technical_anchors"]
    for role, package in result["packages"].items():
        assert package["original"] != package["repacked"]
        assert request["sides"][role]["package_digest"] == package["original"]
        assert not (args.output / (role + "-context")).exists()
        publisher = (
            args.output
            / "publisher-store/storage/blobs/sha256"
            / package["original"][7:]
        )
        delivered = (
            args.output
            / "recipient/kit-store/storage/blobs/sha256"
            / package["original"][7:]
        )
        assert publisher.read_bytes() == delivered.read_bytes()
        assert publisher.stat().st_ino != delivered.stat().st_ino
        source = Path(original["sides"][role]["candidate"])
        copied = args.output / "recipient" / request["sides"][role]["candidate"]
        assert {p.name: p.read_bytes() for p in source.iterdir()} == {
            p.name: p.read_bytes() for p in copied.iterdir()
        }
    assert not list(args.output.rglob("*.withheld"))
    final = journey.read_json(args.output / "logs/before-inference.stdout")
    assert final["technical_integrity_ok"] and final["envelope_evidence_bound"]


@pytest.mark.parametrize("failure", ["unsupported", "empty_output", "mutated_model"])
def test_smoke_rejects_unusable_or_mutated_model(tmp_path, monkeypatch, failure):
    model = tmp_path / "model"
    model.mkdir()
    artifact = model / ("model.bin" if failure == "unsupported" else "model.gguf")
    artifact.write_bytes(b"GGUF fixture")
    request = tmp_path / "request.json"
    journey.write_json(
        request,
        {
            "sides": {
                "subject": {
                    "candidate": "model",
                    "artifact_file": artifact.name,
                    "content_digest": journey.digest(artifact),
                }
            }
        },
    )
    image = "sha256:" + "a" * 64

    def command(argv, *args, **kwargs):
        if "inspect" in argv:
            return json.dumps([{"Id": image}])
        if "--model" in argv:
            if failure == "mutated_model":
                artifact.write_bytes(b"replacement")
            return "answer" if failure == "mutated_model" else " \n"
        return "version"

    monkeypatch.setattr(journey, "command", command)
    message = {
        "unsupported": "requires a GGUF",
        "empty_output": "no generated text",
        "mutated_model": "changed during",
    }[failure]
    with pytest.raises((ValueError, RuntimeError), match=message):
        journey.inference_smoke(
            request,
            {"accepted": True},
            image,
            "/bin/llama",
            tmp_path,
            prompt="Hi",
            tokens=1,
        )


@pytest.mark.parametrize("member", ["../outside.gguf", "config.json", "weights.gguf"])
def test_run_rejects_unsafe_or_changed_source_before_pack(
    real_native_journey, monkeypatch, member
):
    from pathlib import Path

    args = real_native_journey
    request = journey.read_json(args.request)
    side = request["sides"]["baseline"]
    side["artifact_file"] = member
    if member == "weights.gguf":
        (Path(side["candidate"]) / member).write_bytes(b"changed source")
        side["content_digest"] = "sha256:" + "0" * 64
    journey.write_json(args.request, request)
    commands = []
    original_command = journey.command

    def record(argv, *values, **options):
        commands.append(argv)
        return original_command(argv, *values, **options)

    monkeypatch.setattr(journey, "command", record)
    with pytest.raises(ValueError):
        journey.run(args)
    assert commands == [[str(args.kit), "version"]]
    assert not (args.output / "result.json").exists()


@pytest.mark.parametrize(
    "failure",
    ["unchanged_repack", "contradictory", "invalid_exit", "unbound", "untrusted"],
)
def test_run_refuses_false_green_external_results(
    real_native_journey, monkeypatch, failure
):
    args = real_native_journey
    original_command = journey.command

    def command(argv, log, cwd, **options):
        if failure == "unchanged_repack" and "inspect" in argv:
            return json.dumps({"digest": "sha256:" + "a" * 64})
        if log.name == "recipient-decision":
            return json.dumps(
                {
                    "accepted": failure == "contradictory",
                    "exit_code": 3 if failure == "invalid_exit" else 1,
                    "technical_integrity_ok": failure != "untrusted",
                    "envelope_evidence_bound": failure != "unbound",
                }
            )
        return original_command(argv, log, cwd, **options)

    monkeypatch.setattr(journey, "command", command)
    with pytest.raises(RuntimeError, match="repack|contradicts|authentic bound"):
        journey.run(args)
    assert not (args.output / "result.json").exists()
    assert not (args.output / "logs/inference.command.json").exists()


@pytest.mark.parametrize("scenario", ["missing-package", "candidate-changed"])
def test_run_restores_delivered_bytes_when_negative_probe_fails(
    real_native_journey, monkeypatch, scenario
):
    from pathlib import Path

    args = real_native_journey
    original_command = journey.command

    def command(argv, log, cwd, **options):
        if log.name == scenario:
            raise RuntimeError("injected probe failure")
        return original_command(argv, log, cwd, **options)

    monkeypatch.setattr(journey, "command", command)
    with pytest.raises(RuntimeError, match="injected probe failure"):
        journey.run(args)
    source = journey.read_json(args.request)
    package = journey.read_json(args.output / "selected-packages.json")["subject"][
        "original"
    ]
    blob = args.output / "recipient/kit-store/storage/blobs/sha256" / package[7:]
    assert journey.digest(blob) == package
    original = Path(source["sides"]["subject"]["candidate"])
    delivered = args.output / "recipient/subject/model"
    assert {p.name: p.read_bytes() for p in original.iterdir()} == {
        p.name: p.read_bytes() for p in delivered.iterdir()
    }
    assert not list(args.output.rglob("*.withheld"))
    assert not (args.output / "result.json").exists()
