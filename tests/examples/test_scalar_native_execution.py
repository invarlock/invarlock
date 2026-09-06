from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/evaluator-qualification"
PROVIDERS = [
    "lm-evaluation-harness",
    "deepeval",
    "ragas",
    "lighteval",
    "hugging-face-evaluate",
    "autoevals",
    "openevals",
    "openai-evals",
    "arize-phoenix-evals",
    "opik",
    "trulens",
]


def load():
    spec = importlib.util.spec_from_file_location(
        "scalar_native_test", EXAMPLE / "maintained/scalar_native.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("historical", [False, True])
def test_explicit_native_configuration_preserves_each_call_input(
    monkeypatch, tmp_path, provider, historical
):
    module = load()
    calls = []
    configuration = []
    case = {"input": "question", "output": "a\0", "reference": "a", "record_id": "one"}

    class Metric:
        def __init__(self, **kwargs):
            configuration.append(kwargs)
            self.threshold = kwargs.get("threshold", 1)
            self.error = None

        def measure(self, test, **kwargs):
            assert kwargs == {
                "_show_indicator": False,
                "_log_metric_to_confident": False,
            }
            calls.append((test.actual_output, test.expected_output))
            self.score = float(test.actual_output == test.expected_output)
            return self.score

        def is_successful(self):
            return bool(self.score)

        async def ascore(self, *, response, reference):
            calls.append((response, reference))
            return SimpleNamespace(value=float(response == reference))

        def compute(self, **kwargs):
            if "doc" in kwargs:
                doc, response = kwargs["doc"], kwargs["model_response"]
                assert (
                    doc.gold_index == 0 and len(doc.choices) == len(response.text) == 1
                )
                values = (response.text[0], doc.choices[0])
                calls.append(values)
                return float(values[0] == values[1])
            values = (kwargs["predictions"][0], kwargs["references"][0])
            calls.append(values)
            assert (
                kwargs["ignore_case"]
                is kwargs["ignore_punctuation"]
                is kwargs["ignore_numbers"]
                is False
            )
            assert kwargs["regexes_to_ignore"] is None
            return {"exact_match": float(values[0] == values[1])}

        def __call__(self, *args, **kwargs):
            values = args if args else (kwargs["output"], kwargs["expected"])
            calls.append(values)
            score = float(values[0] == values[1])
            return (
                score
                if provider == "trulens"
                else SimpleNamespace(score=score, name="ExactMatch", error=None)
            )

        def score(self, *, output, reference):
            calls.append((output, reference))
            return SimpleNamespace(
                value=float(output == reference), name="equals_metric"
            )

    def lm(predictions, references, **kwargs):
        assert kwargs == {
            "regexes_to_ignore": None,
            "ignore_case": False,
            "ignore_punctuation": False,
            "ignore_numbers": False,
        }
        if historical:
            assert isinstance(predictions, list)
        else:
            assert predictions.dtype == object and references.dtype == object
        calls.append((predictions[0], references[0]))
        return {
            "exact_match": float(
                np.mean(np.asarray(predictions) == np.asarray(references))
            )
        }

    def named(*, outputs, reference_outputs):
        calls.append((outputs, reference_outputs))
        return {"key": "exact_match", "score": outputs == reference_outputs}

    def primitive(output, reference):
        calls.append((output, reference))
        return float(output == reference)

    mapping = {
        "lm_eval.api.metrics": {"exact_match_hf_evaluate": lm},
        "deepeval.metrics": {"ExactMatchMetric": Metric},
        "deepeval.test_case": {"LLMTestCase": SimpleNamespace},
        "ragas.metrics.collections": {"ExactMatch": Metric},
        "lighteval.metrics.metrics_sample": {"ExactMatches": Metric},
        "lighteval.models.model_output": {"ModelResponse": SimpleNamespace},
        "lighteval.tasks.requests": {"Doc": SimpleNamespace},
        "evaluate": {"load": lambda name: Metric() if name == "exact_match" else None},
        "autoevals": {"ExactMatch": Metric},
        "openevals.exact": {"exact_match": named},
        "evals.elsuite.modelgraded.classify_utils": {"MATCH_FNS": {"exact": primitive}},
        "phoenix.evals.metrics": {
            "exact_match": lambda output, reference: SimpleNamespace(
                score=primitive(output, reference)
            )
        },
        "opik.evaluation.metrics": {"Equals": Metric},
        "trulens.core": {"Metric": Metric},
        "runners.trulens_metric": {"exact_match": primitive},
    }
    for name, values in mapping.items():
        fake = ModuleType(name)
        fake.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, fake)
    monkeypatch.setattr(module, "source_binding", lambda _: {"module_sha256": "pinned"})
    source = tmp_path / "metric.py"
    source.write_text("captured module")
    monkeypatch.setattr(module.inspect, "getfile", lambda _: str(source))
    monkeypatch.setattr(
        module.importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(
            read_text=lambda _: json.dumps({"vcs_info": {"commit_id": "revision"}})
        ),
    )
    score, sources = module.build_scorer(provider, historical=historical)
    native = score(case)
    assert calls == [("a\0", "a")]
    assert native["score"] == (
        1.0 if provider == "lm-evaluation-harness" and historical else 0.0
    )
    assert sources["module_sha256"]
    if provider == "opik":
        assert configuration == [
            {"case_sensitive": not historical, "name": "equals_metric", "track": False}
        ]
    if provider == "lighteval":
        assert configuration == [
            {
                "strip_strings": False,
                "normalize_pred": None,
                "normalize_gold": None,
                "type_exact_match": "full",
            }
        ]
    if provider == "deepeval":
        assert configuration == [{"threshold": 1, "verbose_mode": False}]
    if provider == "openai-evals":
        assert sources["source_revision"] == "revision"
        monkeypatch.setattr(
            module.importlib.metadata,
            "distribution",
            lambda _: SimpleNamespace(read_text=lambda _: None),
        )
        assert module.build_scorer(provider)[1]["source_revision"] is None


def test_source_binding_hashes_actual_module_bytes_and_unknown_provider_fails(
    monkeypatch, tmp_path
):
    module = load()
    source = tmp_path / "module.py"
    source.write_bytes(b"module bytes\n")
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda _: SimpleNamespace(__file__=str(source)),
    )
    assert module.source_binding("metric") == {
        "module_sha256": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    }
    with pytest.raises(ValueError, match="unsupported"):
        module.build_scorer("unknown")
