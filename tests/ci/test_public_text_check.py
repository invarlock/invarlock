from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "checks" / "check_public_text.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_public_text", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def public_text():
    return _load_module()


@pytest.mark.parametrize(
    ("text", "rule_name"),
    [
        ("Run this on root@192.0.2.10.", "root_ssh_target"),
        ("Logs are under /Users/alice/project/run.log.", "absolute_home_path"),
        ("Use /root/runs/result.json.", "absolute_root_path"),
        ("Copy the .automation/session output.", "desktop_runtime_path"),
        ("Results are in /private/tmp/run-42/output.", "private_temp_path"),
        ("PATH=/private/tools:/usr/bin make verify", "full_path_environment"),
        ("api_key=abcdefghijklmnop", "credential_like_assignment"),
        ("Privacy gate passed for this report.", "review_process_status"),
        (
            "Checked by an automated reviewer before publishing.",
            "review_process_status",
        ),
        ("Adversarial\naudit completed.", "review_process_status"),
        ("Requires human approval.", "review_staffing_language"),
        ("Human judgment is excluded.", "review_staffing_language"),
        ("Open the human-readable report.", "review_staffing_language"),
        ("The human output is complete.", "review_staffing_language"),
        ("Human judgement is excluded.", "review_staffing_language"),
        ("It does not request human approval.", "review_staffing_language"),
        ("This is not a human-review request.", "review_staffing_language"),
        ("No human\nreview is required.", "review_staffing_language"),
        ("Remote validation passed.", "avoidable_remote_validation_claim"),
        ("Added the engineering backlog lane.", "planning_or_workspace_note"),
        ("Added worktree-aware remote launch handling.", "planning_or_workspace_note"),
        ("This is the compact evaluator release focus.", "planning_or_workspace_note"),
        ("The matrix is not a release breadth target.", "planning_or_workspace_note"),
        (
            "The catalog is reviewed rather than quota-driven.",
            "planning_or_workspace_note",
        ),
        ("Reserved for a future claim.", "planning_or_workspace_note"),
        (
            "Increasing to 600 records would add more execution cost.",
            "planning_or_workspace_note",
        ),
    ],
)
def test_detects_nonpublic_operational_language(
    public_text, text: str, rule_name: str
) -> None:
    findings = public_text.findings_for_text("sample.md", text)

    assert rule_name in {finding.rule.name for finding in findings}


def test_accepts_capability_focused_public_language(public_text) -> None:
    text = """\
`make verify` passed.
Strict CUDA/container validation passed on a CUDA-capable host.
Artifacts use repository-relative paths and placeholder digests.
The integration branch receives dependency updates.
Recorded-score provenance can name a human or judge source.
The descriptor field `uses_human_judgment` is false.
The required metric participates in the policy conjunction.
Reference-label studies retain their source attribution.
"""

    assert public_text.findings_for_text("sample.md", text) == []


def test_tracked_markdown_discovery_includes_every_repository_document(
    public_text,
) -> None:
    expected = {
        ROOT / raw_path.decode("utf-8")
        for raw_path in subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        if raw_path and Path(raw_path.decode("utf-8")).suffix.lower() == ".md"
    }

    discovered = set(public_text.tracked_markdown_paths(ROOT))

    assert discovered == expected
    assert ROOT / "CHANGELOG.md" in discovered


def test_repository_markdown_passes_the_public_text_check(public_text) -> None:
    paths = public_text.working_tree_markdown_paths(ROOT)

    assert public_text.scan_paths(paths, ROOT) == []


def test_external_file_findings_do_not_expose_the_absolute_path(
    public_text, tmp_path: Path
) -> None:
    document = tmp_path / "private-location.md"
    document.write_text("Privacy gate passed.\n", encoding="utf-8")

    findings = public_text.scan_paths([document], ROOT)

    assert len(findings) == 1
    assert findings[0].source == "<external-file>"
    assert str(tmp_path) not in findings[0].source


def test_cli_reports_findings_and_fails(tmp_path: Path) -> None:
    document = tmp_path / "report.md"
    sensitive_target = "root@example.invalid"
    document.write_text(f"Validated from {sensitive_target}.\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path), str(document)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "root_ssh_target" in result.stderr
    assert "Public text check failed: 1 finding(s)." in result.stderr
    assert sensitive_target not in result.stdout
    assert sensitive_target not in result.stderr


def test_main_scans_the_tracked_repository(public_text, capsys) -> None:
    expected_count = len(public_text.working_tree_markdown_paths(ROOT))

    result = public_text.main(["--root", str(ROOT)])

    captured = capsys.readouterr()
    assert result == 0
    assert f"Public text check passed for {expected_count} file(s)." in captured.out
    assert captured.err == ""


def test_main_fails_closed_without_a_git_repository(
    public_text, tmp_path: Path, capsys
) -> None:
    result = public_text.main(["--root", str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 2
    assert "CalledProcessError" in captured.err
    assert str(tmp_path) not in captured.err


@pytest.mark.parametrize("kind", ["missing", "directory", "dangling-symlink"])
def test_explicit_inputs_fail_closed(public_text, tmp_path, capsys, kind):
    path = tmp_path / "input.md"
    if kind == "directory":
        path.mkdir()
    elif kind == "dangling-symlink":
        path.symlink_to(tmp_path / "absent.md")
    assert public_text.main([str(path)]) == 2
    captured = capsys.readouterr()
    assert "could not run" in captured.err
    assert "passed" not in captured.out
    assert str(tmp_path) not in captured.err


def test_working_tree_inventory_uses_git_deletions_not_file_existence(
    public_text, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    calls = []

    def git(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == tmp_path
        if "--others" in command:
            assert "--exclude-standard" in command
            return SimpleNamespace(stdout=b"new.md\0new.MD\0")
        if "--deleted" in command:
            return SimpleNamespace(stdout=b"deleted.md\0")
        return SimpleNamespace(stdout=b"tracked.md\0deleted.md\0missing.md\0")

    monkeypatch.setattr(public_text.subprocess, "run", git)
    paths = public_text.working_tree_markdown_paths(tmp_path)
    assert paths == [
        tmp_path / name for name in ("missing.md", "new.MD", "new.md", "tracked.md")
    ]
    assert len(calls) == 3
    with pytest.raises(FileNotFoundError):
        public_text.scan_paths(paths, tmp_path)


def test_discovery_race_and_explicit_deleted_path_fail_closed(
    public_text, tmp_path, monkeypatch, capsys
):
    missing = tmp_path / "deleted.md"
    monkeypatch.setattr(
        public_text, "working_tree_markdown_paths", lambda root: [missing]
    )
    assert public_text.main(["--root", str(tmp_path)]) == 2
    assert public_text.main([str(missing)]) == 2
    assert "passed" not in capsys.readouterr().out


def vendored_document(root, text="A human-readable format.\n"):
    package = root / "native/vendor/example"
    package.mkdir(parents=True)
    document = package / "README.md"
    document.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(document.read_bytes()).hexdigest()
    provenance = {
        "name": "example",
        "version": "1.0.0",
        "origin": "external upstream dependency; not InvarLock implementation",
        "source_url": "https://static.crates.io/crates/example/example-1.0.0.crate",
        "archive_sha256": "a" * 64,
        "modified_upstream_files": [],
        "upstream_file_hashes": "upstream-files.sha256.json",
        "patched_file_hashes": "patched-files.sha256.json",
    }
    (package / "source-provenance.json").write_text(json.dumps(provenance))
    for key in ("upstream_file_hashes", "patched_file_hashes"):
        (package / provenance[key]).write_text(json.dumps({"README.md": digest}))
    return document, provenance


def test_unchanged_upstream_markdown_style_is_permitted(public_text, tmp_path):
    document, _ = vendored_document(tmp_path)
    assert public_text.scan_paths([document], tmp_path) == []
    # A text-only scan never assumes that the content is an upstream artifact.
    assert public_text.findings_for_text("README.md", document.read_text())


@pytest.mark.parametrize(
    "kind", ["modified", "local-document", "outside-vendor", "declared-modified"]
)
def test_local_or_changed_markdown_is_still_checked(public_text, tmp_path, kind):
    document, provenance = vendored_document(tmp_path)
    if kind == "modified":
        document.write_text(document.read_text() + "Local addition.\n")
    elif kind == "local-document":
        document = document.with_name("LOCAL.md")
        document.write_text("A human-readable format.\n")
    elif kind == "outside-vendor":
        local = tmp_path / "docs/README.md"
        local.parent.mkdir()
        local.write_bytes(document.read_bytes())
        document = local
    else:
        provenance["modified_upstream_files"] = ["README.md"]
        (document.parent / "source-provenance.json").write_text(json.dumps(provenance))
    assert {
        finding.rule.name for finding in public_text.scan_paths([document], tmp_path)
    } == {"review_staffing_language"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "A human-readable format. api_key=abcdefghijklmnop",
            "credential_like_assignment",
        ),
        (
            "A human-readable format. /Users/alice/private/report.md",
            "absolute_home_path",
        ),
        ("A human-readable format. root@example.invalid", "root_ssh_target"),
        ("A human-readable format. /private/tmp/private-report", "private_temp_path"),
        ("A human-readable format. Privacy gate passed.", "review_process_status"),
    ],
)
def test_unchanged_upstream_never_suppresses_sensitive_or_other_rules(
    public_text, tmp_path, text, expected
):
    document, _ = vendored_document(tmp_path, text)
    names = {
        finding.rule.name for finding in public_text.scan_paths([document], tmp_path)
    }
    assert expected in names
    assert "review_staffing_language" not in names


@pytest.mark.parametrize(
    "kind",
    [
        "absent",
        "invalid-json",
        "not-object",
        "missing-field",
        "bad-archive-hash",
        "wrong-upstream-hash",
        "wrong-patched-hash",
        "malformed-manifest",
        "escaping-manifest",
        "duplicate-provenance-key",
        "duplicate-manifest-key",
    ],
)
def test_malformed_or_inconsistent_vendor_provenance_fails_closed(
    public_text, tmp_path, kind
):
    document, provenance = vendored_document(tmp_path)
    path = document.parent / "source-provenance.json"
    if kind == "absent":
        path.unlink()
    elif kind == "invalid-json":
        path.write_text("{")
    elif kind == "not-object":
        path.write_text("[]")
    elif kind == "missing-field":
        del provenance["patched_file_hashes"]
        path.write_text(json.dumps(provenance))
    elif kind == "bad-archive-hash":
        provenance["archive_sha256"] = 123
        path.write_text(json.dumps(provenance))
    elif kind in ("wrong-upstream-hash", "wrong-patched-hash", "malformed-manifest"):
        key = (
            "patched_file_hashes"
            if kind == "wrong-patched-hash"
            else "upstream_file_hashes"
        )
        (document.parent / provenance[key]).write_text(
            "[]"
            if kind == "malformed-manifest"
            else json.dumps({"README.md": "f" * 64})
        )
    elif kind == "escaping-manifest":
        provenance["upstream_file_hashes"] = "../elsewhere.json"
        path.write_text(json.dumps(provenance))
    elif kind == "duplicate-provenance-key":
        path.write_text('{"name":"example",' + json.dumps(provenance)[1:])
    else:
        digest = hashlib.sha256(document.read_bytes()).hexdigest()
        (document.parent / provenance["upstream_file_hashes"]).write_text(
            '{"README.md":"' + digest + '","README.md":"' + digest + '"}'
        )
    assert {
        finding.rule.name for finding in public_text.scan_paths([document], tmp_path)
    } == {"review_staffing_language"}


@pytest.mark.parametrize("target", ["document", "provenance", "manifest"])
def test_vendor_symlinks_do_not_gain_style_exemptions(public_text, tmp_path, target):
    document, provenance = vendored_document(tmp_path)
    selected = {
        "document": document,
        "provenance": document.parent / "source-provenance.json",
        "manifest": document.parent / provenance["upstream_file_hashes"],
    }[target]
    moved = selected.with_name(selected.name + ".copy")
    selected.rename(moved)
    selected.symlink_to(moved)
    assert {
        finding.rule.name for finding in public_text.scan_paths([document], tmp_path)
    } == {"review_staffing_language"}


def test_explicit_local_symlink_does_not_gain_upstream_exemption(
    public_text, tmp_path, capsys
):
    document, _ = vendored_document(tmp_path)
    alias = tmp_path / "local.md"
    alias.symlink_to(document)
    assert public_text.main(["--root", str(tmp_path), str(alias)]) == 1
    assert "review_staffing_language" in capsys.readouterr().err
