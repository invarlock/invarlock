import glob
import json
from pathlib import Path


def row(cert_path: str) -> dict:
    c = json.loads(Path(cert_path).read_text())
    pm = c.get("primary_metric", {}) or {}
    v = c.get("validation", {}) or {}
    return {
        "cert": cert_path,
        "kind": pm.get("kind"),
        "final": pm.get("final"),
        "ratio": pm.get("ratio_vs_baseline"),
        "estimated": pm.get("estimated", False),
        "counts_source": pm.get("counts_source"),
        "pm_ok": v.get("primary_metric_acceptable"),
        "overhead_ok": v.get("guard_overhead_acceptable"),
        "drift_ok": v.get("preview_final_drift_acceptable"),
        "ppl_ok": v.get("ppl_acceptable"),
        "overall": (all(v.values()) if isinstance(v, dict) and v else None),
    }


def main() -> None:
    paths = sorted(glob.glob("reports/**/evaluation.cert.json", recursive=True))
    for r in map(row, paths):
        print(r)


if __name__ == "__main__":
    main()
