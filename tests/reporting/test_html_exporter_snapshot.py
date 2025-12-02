from __future__ import annotations

import html as html_mod
import re

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def _mk_report() -> dict:
    return {
        "meta": {
            "model_id": "gpt2",
            "adapter": "hf_gpt2",
            "device": "cpu",
            "seed": 42,
            "ts": "now",
            "auto": {"tier": "balanced"},
        },
        "data": {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "noop",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
                "sparsity": None,
                "bitwidth_map": None,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 50.0,
                "final": 49.0,
                "ratio_vs_baseline": 0.98,
                "display_ci": (0.97, 0.99),
            }
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def _extract_numbers(s: str) -> list[str]:
    return re.findall(r"[-+]?\d+(?:\.\d+)?", s)


def test_html_export_contains_same_numbers_as_markdown():
    # Import HTML exporter lazily to avoid circulars
    from invarlock.reporting.html import render_certificate_html

    cert = make_certificate(_mk_report(), _mk_report())
    md = render_certificate_markdown(cert)
    html = render_certificate_html(cert)

    nums_md = _extract_numbers(md)
    # Extract inner <pre> content and unescape
    m = re.search(r"<pre[^>]*>(.*?)</pre>", html, flags=re.DOTALL | re.IGNORECASE)
    assert m, "expected <pre> block in HTML output"
    inner = html_mod.unescape(m.group(1))
    nums_html = _extract_numbers(inner)

    assert nums_md == nums_html
