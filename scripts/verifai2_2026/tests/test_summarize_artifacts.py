from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from scripts.verifai2_2026 import summarize_artifacts


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=True) + "\n", encoding="utf-8")


def test_is_artifact() -> None:
    assert summarize_artifacts._is_artifact(
        {"schema_version": "verifier_carrying_artifact.v1"}
    )
    assert not summarize_artifacts._is_artifact({"schema_version": "x"})
    assert not summarize_artifacts._is_artifact([])


def test_as_float_int_error_paths() -> None:
    assert summarize_artifacts._as_float("nope") is None
    assert summarize_artifacts._as_int("nope") is None


def test_load_evaluation_report_embedded(tmp_path: Path) -> None:
    art = {
        "schema_version": "verifier_carrying_artifact.v1",
        "guard_evidence": {
            "invarlock": {"evaluation_report": {"embedded": {"schema_version": "v1"}}}
        },
        "verifier_traces": [],
    }
    rep = summarize_artifacts._load_evaluation_report(art, strict=True)
    assert rep == {"schema_version": "v1"}


def test_load_evaluation_report_path(tmp_path: Path) -> None:
    rep_path = tmp_path / "evaluation.report.json"
    _write_json(rep_path, {"schema_version": "v1", "meta": {"model_id": "m"}})
    art = {
        "schema_version": "verifier_carrying_artifact.v1",
        "guard_evidence": {"invarlock": {"evaluation_report": {"path": str(rep_path)}}},
        "verifier_traces": [],
    }
    rep = summarize_artifacts._load_evaluation_report(art, strict=True)
    assert rep is not None
    assert rep["meta"]["model_id"] == "m"


def test_load_evaluation_report_non_strict_returns_none(tmp_path: Path) -> None:
    art = {
        "schema_version": "verifier_carrying_artifact.v1",
        "guard_evidence": {"invarlock": {"evaluation_report": {"path": ""}}},
        "verifier_traces": [],
    }
    assert summarize_artifacts._load_evaluation_report(art, strict=False) is None


def test_load_evaluation_report_strict_raises(tmp_path: Path) -> None:
    art = {
        "schema_version": "verifier_carrying_artifact.v1",
        "guard_evidence": {"invarlock": {"evaluation_report": {"path": "missing"}}},
        "verifier_traces": [],
    }
    with pytest.raises(ValueError, match="missing embedded evaluation report"):
        summarize_artifacts._load_evaluation_report(art, strict=True)


def test_trace_groups_preserves_order_and_filters_invalid() -> None:
    t1 = {
        "verifier": {"name": "mbpp"},
        "trace_contract": {"prompt_set": {"digest_sha256": "a"}},
    }
    t2 = {"verifier": {"name": "mbpp"}, "trace_contract": {"prompt_set": {}}}  # invalid
    t3 = {
        "verifier": {"name": "humaneval"},
        "trace_contract": {"prompt_set": {"digest_sha256": "b"}},
    }
    groups = summarize_artifacts._trace_groups([t1, t2, t3])
    assert [g["verifier_name"] for g in groups] == ["mbpp", "humaneval"]
    assert [g["prompt_digest"] for g in groups] == ["a", "b"]
    assert groups[0]["traces"] == [t1]


def test_extract_invarlock_features_none() -> None:
    assert summarize_artifacts._extract_invarlock_features(None) == {}


def test_rows_for_artifact_single_trace_per_verifier(tmp_path: Path) -> None:
    art_path = tmp_path / "artifact.json"
    art = {
        "schema_version": "verifier_carrying_artifact.v1",
        "guard_evidence": {
            "invarlock": {
                "evaluation_report": {
                    "embedded": {
                        "schema_version": "v1",
                        "meta": {"model_id": "demo", "profile": "dev", "adapter": "hf"},
                        "edit_name": "quant_rtn",
                        "primary_metric": {
                            "kind": "ppl_causal",
                            "ratio_vs_baseline": 1.01,
                        },
                        "validation": {"primary_metric_acceptable": False},
                        "invariants": {"status": "pass"},
                        "spectral": {
                            "summary": {"status": "ok", "stability_score": 0.9}
                        },
                        "rmt": {"status": "stable", "max_edge_ratio": 1.2},
                        "variance": {"enabled": False},
                    }
                }
            }
        },
        "verifier_traces": [
            {
                "schema_version": "verifier_trace.v1",
                "verifier": {"name": "mbpp"},
                "trace_contract": {
                    "prompt_set": {"digest_sha256": "d"},
                    "model": {"revision": "r0"},
                },
                "results": {"summary": {"n_total": 10, "n_pass": 2, "pass_rate": 0.2}},
            },
            {
                "schema_version": "verifier_trace.v1",
                "verifier": {"name": "humaneval"},
                "trace_contract": {
                    "prompt_set": {"digest_sha256": "e"},
                    "model": {"revision": "r0"},
                },
                "results": {"summary": {"n_total": 10, "n_pass": 1, "pass_rate": 0.1}},
            },
        ],
    }
    _write_json(art_path, art)

    rows = summarize_artifacts._rows_for_artifact(art_path, art, strict=True)
    assert len(rows) == 2
    r0 = rows[0]
    assert r0["verifier_name"] == "mbpp"
    assert r0["baseline_pass_rate"] == 0.2
    assert r0["edited_pass_rate"] is None
    assert r0["delta_pass_rate"] is None
    assert r0["primary_metric_ratio_vs_baseline"] == 1.01


def test_rows_for_artifact_paired_traces_delta(tmp_path: Path) -> None:
    art_path = tmp_path / "artifact.json"
    art = {
        "schema_version": "verifier_carrying_artifact.v1",
        "guard_evidence": {
            "invarlock": {
                "evaluation_report": {
                    "embedded": {"schema_version": "v1", "meta": {"model_id": "m"}}
                }
            }
        },
        "verifier_traces": [
            {
                "schema_version": "verifier_trace.v1",
                "verifier": {"name": "mbpp"},
                "trace_contract": {
                    "prompt_set": {"digest_sha256": "d"},
                    "model": {"revision": "baseline"},
                },
                "results": {
                    "summary": {"pass_rate": 0.1, "metric_name": "pass@10", "k": 10}
                },
            },
            {
                "schema_version": "verifier_trace.v1",
                "verifier": {"name": "mbpp"},
                "trace_contract": {
                    "prompt_set": {"digest_sha256": "d"},
                    "model": {"revision": "edited"},
                },
                "results": {
                    "summary": {"pass_rate": 0.25, "metric_name": "pass@10", "k": 10}
                },
            },
        ],
    }
    _write_json(art_path, art)
    rows = summarize_artifacts._rows_for_artifact(art_path, art, strict=True)
    assert len(rows) == 1
    row = rows[0]
    assert row["baseline_model_revision"] == "baseline"
    assert row["edited_model_revision"] == "edited"
    assert row["baseline_pass_rate"] == 0.1
    assert row["edited_pass_rate"] == 0.25
    assert row["delta_pass_rate"] == pytest.approx(0.15)
    assert row["verifier_metric_name"] == "pass@10"
    assert row["verifier_k"] == 10


def test_rows_for_artifact_no_valid_traces_returns_empty(tmp_path: Path) -> None:
    art_path = tmp_path / "artifact.json"
    art = {
        "schema_version": "verifier_carrying_artifact.v1",
        "guard_evidence": {
            "invarlock": {"evaluation_report": {"embedded": {"schema_version": "v1"}}}
        },
        "verifier_traces": [
            {"schema_version": "verifier_trace.v1"}
        ],  # missing name/digest
    }
    _write_json(art_path, art)
    rows = summarize_artifacts._rows_for_artifact(art_path, art, strict=True)
    assert rows == []


def test_write_csv_empty_and_duplicate_fields() -> None:
    buf = io.StringIO()
    summarize_artifacts._write_csv(buf, [])
    assert buf.getvalue() == ""

    buf = io.StringIO()
    rows = [{"a": 1, "b": 2}, {"a": 3, "c": 4}]
    summarize_artifacts._write_csv(buf, rows)
    text = buf.getvalue()
    assert text.splitlines()[0].startswith("a,")
    assert "c" in text.splitlines()[0]


def test_main_errors_no_candidates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert summarize_artifacts.main([str(tmp_path)]) == 2
    assert "No candidate" in capsys.readouterr().err


def test_main_errors_no_artifacts_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = tmp_path / "not_artifact.json"
    _write_json(p, {"schema_version": "nope"})
    assert summarize_artifacts.main([str(p)]) == 2
    assert "No verifier-carrying artifacts" in capsys.readouterr().err


def test_main_jsonl_stdout_skips_example_and_bad_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Should be ignored by the scanner.
    _write_json(
        tmp_path / "ignored.example.json",
        {
            "schema_version": "verifier_carrying_artifact.v1",
            "guard_evidence": {
                "invarlock": {
                    "evaluation_report": {"embedded": {"schema_version": "v1"}}
                }
            },
            "verifier_traces": [],
        },
    )

    # Malformed JSON should be skipped.
    (tmp_path / "bad.json").write_text("{not json}\n", encoding="utf-8")

    # One valid artifact should produce output rows to stdout.
    _write_json(
        tmp_path / "artifact.json",
        {
            "schema_version": "verifier_carrying_artifact.v1",
            "guard_evidence": {
                "invarlock": {
                    "evaluation_report": {"embedded": {"schema_version": "v1"}}
                }
            },
            "verifier_traces": [
                {
                    "schema_version": "verifier_trace.v1",
                    "verifier": {"name": "mbpp"},
                    "trace_contract": {
                        "prompt_set": {"digest_sha256": "d"},
                        "model": {"revision": "r"},
                    },
                    "results": {
                        "summary": {"n_total": 1, "n_pass": 1, "pass_rate": 1.0}
                    },
                }
            ],
        },
    )

    assert summarize_artifacts.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert '"verifier_name": "mbpp"' in out


def test_main_strict_propagates(tmp_path: Path) -> None:
    p = tmp_path / "artifact.json"
    _write_json(
        p,
        {
            "schema_version": "verifier_carrying_artifact.v1",
            "guard_evidence": {"invarlock": {"evaluation_report": {"path": "missing"}}},
            "verifier_traces": [],
        },
    )
    with pytest.raises(ValueError):
        summarize_artifacts.main([str(p), "--strict"])


def test_main_skips_artifact_on_exception_when_non_strict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # This artifact references an invalid evaluation report (malformed JSON),
    # so it should be skipped with a warning in non-strict mode.
    bad_rep = tmp_path / "evaluation.report.json"
    bad_rep.write_text("{not json}\n", encoding="utf-8")
    _write_json(
        tmp_path / "bad_artifact.json",
        {
            "schema_version": "verifier_carrying_artifact.v1",
            "guard_evidence": {
                "invarlock": {"evaluation_report": {"path": str(bad_rep)}}
            },
            "verifier_traces": [],
        },
    )

    # And a good artifact, so the overall run still succeeds.
    _write_json(
        tmp_path / "good_artifact.json",
        {
            "schema_version": "verifier_carrying_artifact.v1",
            "guard_evidence": {
                "invarlock": {
                    "evaluation_report": {"embedded": {"schema_version": "v1"}}
                }
            },
            "verifier_traces": [
                {
                    "schema_version": "verifier_trace.v1",
                    "verifier": {"name": "mbpp"},
                    "trace_contract": {
                        "prompt_set": {"digest_sha256": "d"},
                        "model": {"revision": "r"},
                    },
                    "results": {
                        "summary": {"n_total": 1, "n_pass": 0, "pass_rate": 0.0}
                    },
                }
            ],
        },
    )

    assert summarize_artifacts.main([str(tmp_path)]) == 0
    err = capsys.readouterr().err
    assert "Skipping artifact" in err


def test_main_csv_out(tmp_path: Path) -> None:
    p = tmp_path / "artifact.json"
    _write_json(
        p,
        {
            "schema_version": "verifier_carrying_artifact.v1",
            "guard_evidence": {
                "invarlock": {
                    "evaluation_report": {"embedded": {"schema_version": "v1"}}
                }
            },
            "verifier_traces": [
                {
                    "schema_version": "verifier_trace.v1",
                    "verifier": {"name": "mbpp"},
                    "trace_contract": {
                        "prompt_set": {"digest_sha256": "d"},
                        "model": {"revision": "r"},
                    },
                    "results": {
                        "summary": {"n_total": 1, "n_pass": 1, "pass_rate": 1.0}
                    },
                }
            ],
        },
    )
    out = tmp_path / "out.csv"
    assert summarize_artifacts.main([str(p), "--format", "csv", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("artifact_path,")
    assert ",mbpp," in text
