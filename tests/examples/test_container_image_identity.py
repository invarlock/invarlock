"""Engine output representation must not weaken example image bindings."""

import subprocess

import pytest

from examples.integrations import gguf_llama_cpp, launch, local_registry
from examples.integrations.evaluator_transaction import image_cleanup

IMAGE = "sha256:" + "a" * 64


@pytest.mark.parametrize("observed", [IMAGE, "a" * 64])
def test_gguf_inspection_normalizes_complete_engine_id(tmp_path, monkeypatch, observed):
    monkeypatch.setattr(
        launch,
        "_run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, observed + "\n", ""
        ),
    )
    assert (
        gguf_llama_cpp._inspect_image_id(
            tmp_path, container_engine="podman", image=IMAGE
        )
        == IMAGE
    )


@pytest.mark.parametrize(
    "observed",
    [
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
        "sha256:" + "a" * 63,
        "a" * 32 + " " + "a" * 32,
        "candidate:local",
        "a" * 64 + "\n" + "b" * 64,
    ],
)
def test_malformed_engine_ids_cannot_be_owned_or_removed(tmp_path, observed):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return observed

    with pytest.raises(RuntimeError):
        image_cleanup.record_owned_image_tag(
            run, "podman", "candidate:local", IMAGE, tmp_path
        )
    with pytest.raises(RuntimeError):
        image_cleanup.remove_owned_image_tags(
            run,
            "podman",
            tmp_path,
            [image_cleanup.OwnedImageTag("candidate:local", IMAGE)],
        )
    assert all(command[1:3] == ["image", "inspect"] for command in calls)


def test_bare_engine_id_preserves_cleanup_ownership(tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return "a" * 64 + "\n"

    owned = image_cleanup.record_owned_image_tag(
        run, "podman", "candidate:local", IMAGE, tmp_path
    )
    assert owned.image_id == IMAGE
    image_cleanup.remove_owned_image_tags(run, "podman", tmp_path, [owned])
    assert calls[-1] == ["podman", "image", "rm", "candidate:local"]
    calls.clear()

    def changed(command, **kwargs):
        calls.append(command)
        return "b" * 64

    with pytest.raises(RuntimeError, match="does not name the image built"):
        image_cleanup.record_owned_image_tag(
            changed, "podman", "candidate:local", IMAGE, tmp_path
        )
    calls.clear()
    with pytest.raises(RuntimeError, match="ownership changed"):
        image_cleanup.remove_owned_image_tags(changed, "podman", tmp_path, [owned])
    assert calls == [
        ["podman", "image", "inspect", "--format", "{{.Id}}", "candidate:local"]
    ]


@pytest.mark.parametrize(
    "identity,commit,bundle,accepted",
    [
        ("a" * 64, "c" * 40, "sha256:" + "d" * 64, True),
        ("b" * 64, "c" * 40, "sha256:" + "d" * 64, False),
        ("a" * 64, "e" * 40, "sha256:" + "d" * 64, False),
        ("a" * 64, "c" * 40, "sha256:" + "e" * 64, False),
    ],
)
def test_runtime_identity_normalization_preserves_source_binding(
    tmp_path, monkeypatch, identity, commit, bundle, accepted
):
    monkeypatch.setattr(
        launch,
        "_run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, "\t".join((identity, commit, bundle)), ""
        ),
    )

    def verify():
        launch._verify_runtime_image_identity(
            container_engine="podman",
            repository=tmp_path,
            runtime_image_id=IMAGE,
            source_commit="c" * 40,
            source_bundle_sha256="sha256:" + "d" * 64,
        )

    if accepted:
        verify()
    else:
        with pytest.raises(RuntimeError, match="not bound to the source"):
            verify()


@pytest.mark.parametrize("phase", ["body", "push", "tag"])
@pytest.mark.parametrize("inspection_prefix", ["sha256:", ""])
def test_registry_cleanup_never_removes_a_retargeted_tag(
    tmp_path, monkeypatch, phase, inspection_prefix
):
    calls = []
    observed_id = IMAGE
    endpoint = "127.0.0.1:49152"
    published = endpoint + "/runtime:" + "a" * 12
    canonical = endpoint + "/runtime@" + IMAGE
    monkeypatch.setattr(local_registry.secrets, "token_hex", lambda _: "fixed")

    def run(command, **kwargs):
        nonlocal observed_id
        calls.append(command)
        output = ""
        if command[1:3] == ["volume", "create"]:
            output = "invarlock-example-registry-fixed"
        elif command[1] == "run":
            output = "c" * 64
        elif command[1] == "port":
            output = endpoint
        elif command[1:3] == ["image", "inspect"]:
            output = inspection_prefix + observed_id.removeprefix("sha256:")
            if "RepoDigests" in command[4]:
                output += ' ["' + canonical + '"]'
        elif (phase == "push" and command[1] == "push") or (
            phase == "tag" and command[1:3] == ["image", "tag"]
        ):
            observed_id = "sha256:" + "b" * 64
            raise RuntimeError(phase + " failed")
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(local_registry, "_run", run)

    def publish():
        nonlocal observed_id
        with local_registry.published_local_image(
            repository=tmp_path,
            container_engine="podman",
            image=IMAGE,
            image_digest=IMAGE,
            repository_name="runtime",
        ) as reference:
            assert reference == canonical
            observed_id = "sha256:" + "b" * 64

    if phase == "body":
        publish()
    else:
        with pytest.raises(RuntimeError, match=phase + " failed"):
            publish()
    assert not any(
        command[1:3] in (["image", "remove"], ["image", "rm"])
        and command[-1] == published
        for command in calls
    )
    assert calls[-2][1:3] == ["container", "rm"]
    assert calls[-1][1:3] == ["volume", "rm"]
