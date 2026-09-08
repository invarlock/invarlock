"""Exercise installed pipeline commands outside the source checkout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", default="invarlock-pipeline")
    parser.add_argument(
        "--pipeline-namespace",
        action="store_true",
        help="Use the pipeline subcommand when --cli names the core invarlock executable.",
    )
    args = parser.parse_args()
    executable = shutil.which(args.cli)
    if executable is None:
        raise SystemExit("Install the candidate wheel before running this example.")
    prefix = [executable, "pipeline"] if args.pipeline_namespace else [executable]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("INVARLOCK_PIPELINE_SIGNING_KEY", None)
    with tempfile.TemporaryDirectory(prefix="invarlock-pipeline-smoke-") as directory:
        root = Path(directory)

        def run(*arguments: str, expected: int = 0) -> str:
            result = subprocess.run(
                [*prefix, *arguments],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != expected:
                raise RuntimeError(
                    f"{arguments[0]} returned {result.returncode}, expected {expected}: "
                    f"{result.stdout}{result.stderr}"
                )
            return result.stdout.strip()

        run("keygen", "keys")
        for example in ("classification", "extraction", "judge"):
            run("init", example, "--example", example)
            project = f"{example}/pipeline.json"
            result = f"{example}/result"
            status = json.loads(
                run(
                    "compare",
                    project,
                    "--output",
                    result,
                    "--signing-key",
                    "keys/private.pem",
                )
            )
            assert status["decision"] == "pass"
            for name in (
                "evidence.json",
                "comparison.json",
                "report.html",
                "summary.md",
                "junit.xml",
            ):
                assert (root / result / name).stat().st_size > 0
            baseline = run("digest", f"{example}/baseline.json", "--run")
            candidate = run("digest", f"{example}/candidate.json", "--run")
            verification = json.loads(
                run(
                    "verify",
                    f"{result}/evidence.json",
                    "--public-key",
                    "keys/public.pem",
                    "--policy",
                    f"{example}/policy.json",
                    "--expected-baseline",
                    baseline,
                    "--expected-candidate",
                    candidate,
                )
            )
            assert verification["authenticated"] is True
            run("compare", project, "--output", result, expected=2)
            print(
                f"{example}: installed comparison, reports and independent verification pass"
            )
        evidence = root / "classification/result/evidence.json"
        original_evidence = evidence.read_bytes()
        regenerated = json.loads(run("report", str(evidence), "--output", "rendered"))
        assert regenerated["recorded_decision"] == "pass"
        assert regenerated["independent_verification"] == "not_performed"
        assert regenerated["signing"] == "signature_present_unverified"
        for name in ("report.html", "summary.md"):
            assert (root / "rendered" / name).read_bytes() == (
                root / "classification/result" / name
            ).read_bytes()
        assert evidence.read_bytes() == original_evidence
        human = run(
            "compare",
            "classification/pipeline.json",
            "--output",
            "human",
            "--output-format",
            "human",
            "--explain",
        )
        assert "Policy result: pass" in human
        assert "40/40 usable pairs, 0 missing" in human
        assert "Signing: Unsigned local evidence" in human
        assert "Independent verification: not performed" in human
        verified = run(
            "verify",
            str(evidence),
            "--public-key",
            "keys/public.pem",
            "--policy",
            "classification/policy.json",
            "--expected-baseline",
            run("digest", "classification/baseline.json", "--run"),
            "--expected-candidate",
            run("digest", "classification/candidate.json", "--run"),
            "--output-format",
            "human",
            "--explain",
        )
        assert "Independent verification: passed" in verified
        assert "40/40 usable pairs, 0 missing" in verified
        print("text summaries and rendering without independent verification pass")
        candidate_path = root / "judge/candidate.json"
        candidate_run = json.loads(candidate_path.read_text())
        for row in candidate_run["records"]:
            row["scores"]["quality"] = 0.1
        candidate_path.write_text(json.dumps(candidate_run))
        run("compare", "judge/pipeline.json", "--output", "regressed", expected=1)
        candidate_run["records"][0]["error"] = "upstream timeout"
        candidate_path.write_text(json.dumps(candidate_run))
        run("compare", "judge/pipeline.json", "--output", "incomplete", expected=3)
        print("regression, integration error and insufficient-evidence exit codes pass")


if __name__ == "__main__":
    main()
