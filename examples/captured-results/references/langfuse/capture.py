"""Replay retained Mistral 7B outputs through offline Langfuse experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ORIGINS = {
    "exact_match": "examples/hosted-service/references/mistral-7b-http",
    "normalized_nll": "examples/captured-results/references/mistral-7b-likelihood",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from langfuse import Evaluation, Langfuse

    spec = importlib.util.spec_from_file_location(
        "langfuse_export", ROOT / "examples/integrations/langfuse_export.py"
    )
    assert spec and spec.loader
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    args.output.mkdir(parents=True, exist_ok=False)
    client = Langfuse(
        public_key="offline-public", secret_key="offline-secret", tracing_enabled=False
    )
    reference = {
        "format": "invarlock/langfuse-reference-v1",
        "sdk_version": "4.14.1",
        "execution": (
            "offline Langfuse SDK experiments replaying retained model measurements; "
            "no new model or judge calls"
        ),
        "sources": {},
        "exports": {},
    }
    for metric, origin in ORIGINS.items():
        for side in ("baseline", "subject"):
            relative = f"{origin}/evidence/records/{side}.json"
            raw = (ROOT / relative).read_bytes()
            run = json.loads(raw)
            source_digest = "sha256:" + hashlib.sha256(raw).hexdigest()
            data = [
                {
                    "input": row["input"],
                    "expected_output": row["expected"],
                    "metadata": {
                        **row["metadata"],
                        "invarlock_id": row["id"],
                        "invarlock_retained_output": row["output"],
                        "invarlock_retained_source_sha256": source_digest,
                        **(
                            {
                                "invarlock_original_likelihood": row["likelihood"],
                                "invarlock_likelihood": {
                                    **row["likelihood"],
                                    "source": {"name": "langfuse", "version": "4.14.1"},
                                },
                            }
                            if "likelihood" in row
                            else {}
                        ),
                    },
                }
                for row in run["records"]
            ]

            def evaluate(*, output, expected_output, **_):
                return Evaluation(
                    name="exact_match",
                    value=output == expected_output,
                    data_type="BOOLEAN",
                )

            result = client.run_experiment(
                name=f"retained-mistral-7b-{metric}-{side}",
                data=data,
                task=lambda *, item, **_: item["metadata"]["invarlock_retained_output"],
                evaluators=[evaluate] if metric == "exact_match" else [],
                max_concurrency=1,
            )
            name = f"{metric}-{side}.json"
            exporter.write_experiment_export(
                result,
                args.output / name,
                expected_ids=[row["id"] for row in run["records"]],
            )
            reference["sources"][name] = {"path": relative, "sha256": source_digest}
            reference["exports"][name] = {
                "run_id": result.run_name,
                "record_count": len(data),
                "sha256": "sha256:"
                + hashlib.sha256((args.output / name).read_bytes()).hexdigest(),
                "artifact_digest": run["artifact_digest"],
                **(
                    {"service_identity": run["service_identity"]}
                    if "service_identity" in run
                    else {}
                ),
            }
    (args.output / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")


if __name__ == "__main__":
    main()
