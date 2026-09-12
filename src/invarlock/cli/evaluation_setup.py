"""Installed, no-execution starter material for evaluation setup."""

from __future__ import annotations

from importlib import resources

from invarlock.evaluation_records.templates import example_project
from invarlock.evidence_pack_contract import canonical_json_bytes

NATIVE_STARTER_FILES = ("request.yaml", "judge-policy.json", "cases.jsonl", "README.md")


def starter_artifacts(example: str) -> dict[str, bytes]:
    """Load native starter resources or build the synthetic captured examples."""
    if example == "native-judge":
        root = resources.files("invarlock").joinpath(
            "_data", "examples", "native-judge"
        )
        return {name: root.joinpath(name).read_bytes() for name in NATIVE_STARTER_FILES}
    if example not in {"classification", "extraction", "judge"}:
        raise ValueError(
            "example must be classification, extraction, judge (recorded ratings), or native-judge"
        )
    baseline, subject, policy = example_project(example)
    request = {
        "format_version": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            "baseline": {
                "path": "inputs/baseline.json",
                "adapter": "invarlock",
            },
            "subject": {
                "path": "inputs/subject.json",
                "adapter": "invarlock",
            },
            "policy": "policy.json",
        },
        "output": {"evidence": "artifacts/evidence"},
    }
    readme = (
        "# InvarLock captured evaluation example\n\n"
        "The JSON inputs are honest synthetic captured results. Replace them "
        "with independently produced records before relying on a decision.\n"
        "Run these commands from this directory:\n\n"
        "invarlock evaluate request.yaml --unsigned --json\n"
        "invarlock report artifacts/evidence --html artifacts/report.html "
        "--markdown artifacts/summary.md --junit artifacts/junit.xml --explain\n"
        "\nUnsigned reports are local, not independent verification or native "
        "acceptance. For a signed handoff use --signing-key, then verify with "
        "a recipient-owned --trust-profile (invarlock/trust-inputs-v2) and "
        "--receipt outside the pack. See docs/user-guide/captured-results.md.\n"
        "Add --fail-on-policy to evaluate for a local gate: 0 pass, 7 adverse "
        "decision, 2 input/work-budget failure. Report only a newly published "
        "pack. Commands default to text; --json emits captured evaluation/"
        "verification/report v2 results. Report output maps are requested_outputs "
        "and written_outputs, with failed_output and errors on write failure.\n"
    )

    if example == "judge":
        readme += (
            "\nThis starter compares recorded ratings. It does not call a judge. "
            "Use --example native-judge for native model answer generation followed "
            "by a configured judge measurement.\n"
        )
    return {
        "request.yaml": canonical_json_bytes(request),
        "inputs/baseline.json": canonical_json_bytes(baseline),
        "inputs/subject.json": canonical_json_bytes(subject),
        "policy.json": canonical_json_bytes(policy),
        "README.txt": readme.encode("utf-8"),
    }
