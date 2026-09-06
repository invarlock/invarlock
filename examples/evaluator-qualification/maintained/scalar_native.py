"""Actual upstream scalar calls, with explicit current configurations and source facts."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any


def source_binding(module_name: str) -> dict[str, str]:
    module = importlib.import_module(module_name)
    return {
        "module_sha256": "sha256:"
        + hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    }


def build_scorer(
    provider: str, *, historical: bool = False
) -> tuple[Callable[[dict[str, str]], dict[str, Any]], dict[str, Any]]:
    if provider == "lm-evaluation-harness":
        import numpy as np
        from lm_eval.api.metrics import exact_match_hf_evaluate

        def score(case):
            outputs, references = [case["output"]], [case["reference"]]
            if not historical:
                outputs, references = (
                    np.asarray(outputs, dtype=object),
                    np.asarray(references, dtype=object),
                )
            return {
                "score": exact_match_hf_evaluate(
                    outputs,
                    references,
                    regexes_to_ignore=None,
                    ignore_case=False,
                    ignore_punctuation=False,
                    ignore_numbers=False,
                )["exact_match"]
            }

        return score, source_binding("lm_eval.api.metrics")
    if provider == "deepeval":
        from deepeval.metrics import ExactMatchMetric
        from deepeval.test_case import LLMTestCase

        def score(case):
            metric = ExactMatchMetric(threshold=1, verbose_mode=False)
            value = metric.measure(
                LLMTestCase(
                    input=case["input"],
                    actual_output=case["output"],
                    expected_output=case["reference"],
                ),
                _show_indicator=False,
                _log_metric_to_confident=False,
            )
            return {
                "score": value,
                "metric_score": metric.score,
                "successful": metric.is_successful(),
                "threshold": metric.threshold,
                "error": metric.error,
            }

        return score, source_binding(ExactMatchMetric.__module__)
    if provider == "ragas":
        from ragas.metrics.collections import ExactMatch

        metric = ExactMatch()

        def score(case):
            result = asyncio.run(
                metric.ascore(reference=case["reference"], response=case["output"])
            )
            return {"score": result.value}

        return score, source_binding(ExactMatch.__module__)
    if provider == "lighteval":
        from lighteval.metrics.metrics_sample import ExactMatches
        from lighteval.models.model_output import ModelResponse
        from lighteval.tasks.requests import Doc

        metric = ExactMatches(
            strip_strings=False,
            normalize_pred=None,
            normalize_gold=None,
            type_exact_match="full",
        )

        def score(case):
            document = Doc(
                task_name="invarlock-qualification",
                query=case["input"],
                choices=[case["reference"]],
                gold_index=0,
            )
            return {
                "score": metric.compute(
                    doc=document, model_response=ModelResponse(text=[case["output"]])
                )
            }

        return score, source_binding(ExactMatches.__module__)
    if provider == "hugging-face-evaluate":
        import evaluate

        metric = evaluate.load("exact_match")

        def score(case):
            return {
                "score": metric.compute(
                    predictions=[case["output"]],
                    references=[case["reference"]],
                    regexes_to_ignore=None,
                    ignore_case=False,
                    ignore_punctuation=False,
                    ignore_numbers=False,
                )["exact_match"]
            }

        module_path = Path(inspect.getfile(type(metric)))
        return score, {
            "module_sha256": "sha256:"
            + hashlib.sha256(module_path.read_bytes()).hexdigest()
        }
    if provider == "autoevals":
        from autoevals import ExactMatch

        metric = ExactMatch()

        def score(case):
            result = metric(output=case["output"], expected=case["reference"])
            return {"score": result.score, "name": result.name, "error": result.error}

        return score, source_binding(ExactMatch.__module__)
    if provider == "openevals":
        from openevals.exact import exact_match

        def score(case):
            result = exact_match(
                outputs=case["output"], reference_outputs=case["reference"]
            )
            return {"score": result["score"], "key": result["key"]}

        return score, source_binding("openevals.exact")
    if provider == "openai-evals":
        os.environ.setdefault("OPENAI_API_KEY", "unused")
        from evals.elsuite.modelgraded.classify_utils import MATCH_FNS

        def score(case):
            return {"score": MATCH_FNS["exact"](case["output"], case["reference"])}

        sources = source_binding("evals.elsuite.modelgraded.classify_utils")
        direct = importlib.metadata.distribution("evals").read_text("direct_url.json")
        sources["source_revision"] = (
            json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
        )
        return score, sources
    if provider == "arize-phoenix-evals":
        from phoenix.evals.metrics import exact_match

        def score(case):
            return {"score": exact_match(case["output"], case["reference"]).score}

        return score, source_binding("phoenix.evals.metrics.exact_match")
    if provider == "opik":
        from opik.evaluation.metrics import Equals

        metric = Equals(
            case_sensitive=not historical, name="equals_metric", track=False
        )

        def score(case):
            result = metric.score(output=case["output"], reference=case["reference"])
            return {"score": result.value, "name": result.name}

        return score, source_binding(Equals.__module__)
    if provider == "trulens":
        from runners.trulens_metric import exact_match
        from trulens.core import Metric

        metric = Metric(implementation=exact_match, name="exact_match")

        def score(case):
            return {"score": metric(case["output"], case["reference"])}

        return score, source_binding("runners.trulens_metric")
    raise ValueError("unsupported scalar evaluator")
