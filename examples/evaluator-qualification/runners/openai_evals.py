"""Execute OpenAI Evals' upstream match-recording primitive."""

import importlib.metadata
import json
import os

os.environ.setdefault("OPENAI_API_KEY", "unused")

from evals.elsuite.modelgraded.classify_utils import MATCH_FNS  # noqa: E402
from runner_support import arguments, finish_deterministic, load_inputs  # noqa: E402


def source_revision() -> str:
    direct_url_text = importlib.metadata.distribution("evals").read_text(
        "direct_url.json"
    )
    if direct_url_text is None:
        raise ValueError("OpenAI Evals installation lacks direct source provenance")
    direct_url = json.loads(direct_url_text)
    revision = direct_url.get("vcs_info", {}).get("commit_id")
    if not isinstance(revision, str):
        raise ValueError("OpenAI Evals source revision is absent")
    return revision


def main() -> None:
    args = arguments()
    _, _, cases = load_inputs(args)
    expected_revision = (
        args.dependency_lock.read_text(encoding="utf-8").strip().rsplit("@", 1)[-1]
    )
    observed_revision = source_revision()
    if observed_revision != expected_revision:
        raise ValueError("OpenAI Evals source revision does not match the lock")
    scores: list[float] = []
    details = []
    exact_match = MATCH_FNS["exact"]
    for case in cases:
        score = float(exact_match(case["output"], case["reference"]))
        scores.append(score)
        details.append(
            {
                "matched": bool(score),
                "source_revision": observed_revision,
            }
        )
    finish_deterministic(
        args=args,
        entrypoint=("evals.elsuite.modelgraded.classify_utils.MATCH_FNS['exact']"),
        scores=scores,
        details=details,
    )


if __name__ == "__main__":
    main()
