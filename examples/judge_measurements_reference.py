"""Freeze outcome-blind K2 answer subsets and validate them without hosted calls."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import stat
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from invarlock.evaluation_comparison.comparison import make_run
from invarlock.evaluation_records.cases import case_set_digest, validate_run_case_set
from invarlock.evaluation_records.io import run_digest
from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes
from invarlock.judge_measurements.analysis import decode_analysis_policy
from invarlock.judge_measurements.contracts import (
    canonical_payload,
    measurement_plan_digest,
    render_judge_request,
    validate_measurement_plan,
)

FORMAT = "invarlock/k2-judge-answer-reference-v1"
SEED = "invarlock-k2-32b-judge-reference-v1"
WORKFLOWS = {
    "grounded_qa": ("answerable", "unanswerable"),
    "extraction": ("has_span", "empty_span"),
}
SIZES = {"pilot": 20, "human_review": 40}
ASSESSMENT = "remaining-tp2-remaining316-assessment-03"
PREPARATION = "remaining-tp2-four-roles-preparation-04/bundles"
PLANNED = "cluster-policy-companion/final-layout-04/grounded_qa-planned-cases.json"
MAX_BYTES = 64 * 1024 * 1024
MAX_BUNDLE_BYTES = 192 * 1024 * 1024
MODEL_REVISION = "466db5f23c8a7c96b0b320b688612ee6f4446a35"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read(path: Path, maximum: int = MAX_BYTES) -> bytes:
    return read_regular_file_bytes(path, label="reference input", max_bytes=maximum)


def obj(path: Path) -> Any:
    return parse_json_bytes(read(path), label="reference JSON")


def safe_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or len(name) > 4096 or not name or "\\" in name:
        raise ValueError("invalid reference path")
    path = Path(name)
    if path.is_absolute() or any(p in {"", ".", ".."} for p in name.split("/")):
        raise ValueError("reference paths must remain relative")
    return root / path


def rank(workflow: str, split: str, case_id: str) -> str:
    return sha(canonical_payload([SEED, workflow, split, case_id]))


def select(
    inventory: list[dict], workflow: str, sizes: dict[str, int] = SIZES
) -> dict[str, list[str]]:
    """Selection reads only IDs, strata and clusters, never outputs or scores."""
    if workflow not in WORKFLOWS or len(inventory) > 4000:
        raise ValueError("unsupported workflow or oversized inventory")
    seen: set[str] = set()
    for row in inventory:
        if set(row) != {"id", "metadata"} or row["id"] in seen:
            raise ValueError("inventory must contain unique outcome-free entries")
        seen.add(row["id"])
        metadata = row["metadata"]
        if set(metadata) != {
            "dataset",
            "family",
            "language",
            "slice",
            "source_cluster_id",
            "upstream_split",
            "workflow",
        }:
            raise ValueError("unexpected source metadata")
        if (
            metadata["workflow"] != workflow
            or metadata["slice"] not in WORKFLOWS[workflow]
        ):
            raise ValueError("inventory workflow/stratum mismatch")
        if not all(
            isinstance(v, str) and 0 < len(v) <= 4096 for v in metadata.values()
        ):
            raise ValueError("source metadata must be bounded strings")
        if not isinstance(row["id"], str) or not 0 < len(row["id"]) <= 128:
            raise ValueError("case ID must be bounded")
        if workflow == "grounded_qa" and metadata["language"] != "en":
            raise ValueError("only English QA belongs to this reference")
    used: set[str] = set()
    result: dict[str, list[str]] = {}
    for split in ("pilot", "final"):
        count: Counter = Counter()
        selected = []
        # Global rank order avoids assigning priority by stratum name.
        for row in sorted(
            inventory, key=lambda r: (rank(workflow, split, r["id"]), r["id"])
        ):
            cluster, stratum = (
                row["metadata"]["source_cluster_id"],
                row["metadata"]["slice"],
            )
            if cluster in used or (split == "pilot" and count[stratum] >= sizes[split]):
                continue
            selected.append(row["id"])
            used.add(cluster)
            count[stratum] += 1
        if split == "pilot" and any(
            count[s] != sizes[split] for s in WORKFLOWS[workflow]
        ):
            raise ValueError(
                f"infeasible frozen {workflow} {split} selection: {dict(count)}"
            )
        result[split] = sorted(selected)
    final_ids = set(result["final"])
    count = Counter()
    review = []
    for row in sorted(
        inventory, key=lambda r: (rank(workflow, "human_review", r["id"]), r["id"])
    ):
        stratum = row["metadata"]["slice"]
        if row["id"] in final_ids and count[stratum] < sizes["human_review"]:
            review.append(row["id"])
            count[stratum] += 1
    if any(count[s] != sizes["human_review"] for s in WORKFLOWS[workflow]):
        raise ValueError("infeasible human review selection")
    result["human_review"] = review
    return result


def complete(row: dict) -> bool:
    context = row.get("context", {})
    return (
        row.get("error") is None
        and isinstance(row.get("output"), str)
        and context.get("capture_present") is True
        and context.get("verified_complete") is True
        and context.get("native_result", {}).get("capture_complete") is True
        and context.get("native_result", {}).get("status") == "complete"
    )


def input_text(row: dict) -> str:
    messages = row["input"]
    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or set(messages[0]) != {"role", "content"}
    ):
        raise ValueError("expected exactly one unchanged source user message")
    if messages[0]["role"] != "user" or not isinstance(messages[0]["content"], str):
        raise ValueError("expected source user text")
    return messages[0]["content"]


def protocol_snapshot(value: dict) -> dict:
    model = value["model"]
    if model["repo"] != "IFM/K2-Horizon-32B" or model["revision"] != MODEL_REVISION:
        raise ValueError("unexpected source checkpoint")
    return {
        "model": {
            k: model[k]
            for k in (
                "key",
                "repo",
                "revision",
                "files",
                "artifact_digest",
                "template_kwargs",
                "eos_token_ids",
                "generation_config_sha256",
                "route",
                "native_parser",
                "native_parser_effort",
                "tensor_parallel",
                "expert_parallel",
            )
        },
        "generation": {
            k: value[k]
            for k in (
                "workload_sha256",
                "source_workload_sha256",
                "context_length",
                "maximum_output_tokens",
                "stop_policy",
                "content_policy",
                "input_policy",
            )
        },
    }


def capture_campaign(root: Path) -> dict:
    """Cross-check complete raw blocks, frozen planned QA and assessed endpoints."""
    sources: dict[str, dict] = {}

    def source(name: str) -> Any:
        raw = read(safe_path(root, name))
        sources[name] = {"sha256": sha(raw), "size_bytes": len(raw)}
        return parse_json_bytes(raw, label="campaign source")

    planned_value = source(PLANNED)
    planned = {row["id"]: row for row in planned_value["cases"]}
    if len(planned) != 4000:
        raise ValueError("expected 4000 unique planned QA cases")
    workflows = {}
    for workflow in WORKFLOWS:
        sides, endpoints, mappings, snapshots = {}, {}, {}, {}
        for role in ("A", "B"):
            rows, mapping = {}, {}
            for block in range(1, 17):
                job = (
                    f"k2-32b-{role.lower()}-{workflow.replace('_', '-')}-250-{block:02}"
                )
                source_name = f"{ASSESSMENT}/blocks/{job}/records.json"
                records = source(source_name)
                if not isinstance(records, list) or len(records) != 250:
                    raise ValueError("source block must retain all 250 records")
                protocol_name = f"{PREPARATION}/{job}/protocol.json"
                snapshots[protocol_name] = protocol_snapshot(source(protocol_name))
                for index, row in enumerate(records):
                    if row["id"] in rows:
                        raise ValueError("duplicate source record ID")
                    if (
                        row["metadata"]["workflow"] != workflow
                        or row["context"]["job_id"] != job
                    ):
                        raise ValueError("source block provenance mismatch")
                    rows[row["id"]] = row
                    mapping[row["id"]] = {
                        "path": source_name,
                        "row_index": index,
                        "row_sha256": sha(canonical_payload(row)),
                        "protocol_path": protocol_name,
                    }
            endpoint_name = (
                f"{ASSESSMENT}/endpoints/k2-32b-{role}-{workflow}-snapshot-01/run.json"
            )
            endpoint = source(endpoint_name)
            retained = {r["id"]: r for r in endpoint["records"]}
            if len(endpoint["records"]) != 4000 or retained != rows:
                raise ValueError(
                    "endpoint records do not exactly reproduce raw source blocks"
                )
            sides[role], mappings[role] = rows, mapping
            endpoints[role] = {
                "path": endpoint_name,
                "header": {
                    k: v for k, v in endpoint.items() if k not in {"records", "format"}
                },
            }
        if sides["A"].keys() != sides["B"].keys():
            raise ValueError("A/B membership differs")
        eligible = []
        for case_id, a in sides["A"].items():
            b = sides["B"][case_id]
            case = {k: a[k] for k in ("id", "input", "expected", "metadata")}
            if case != {k: b[k] for k in case}:
                raise ValueError("A/B planned input, reference or metadata differs")
            if workflow == "grounded_qa" and planned.get(case_id) != case:
                raise ValueError("raw QA differs from planned source case")
            if (
                complete(a)
                and complete(b)
                and (workflow != "grounded_qa" or a["metadata"]["language"] == "en")
            ):
                eligible.append({"id": case_id, "metadata": a["metadata"]})
        inventory = sorted(eligible, key=lambda row: row["id"])
        selection = select(inventory, workflow)
        selected = sorted(selection["pilot"] + selection["final"])
        workflows[workflow] = {
            "inventory": inventory,
            "raw": {
                role: [sides[role][case_id] for case_id in selected]
                for role in ("A", "B")
            },
            "mappings": {
                role: {case_id: mappings[role][case_id] for case_id in selected}
                for role in ("A", "B")
            },
            "endpoints": endpoints,
            "protocols": snapshots,
        }
    return {"sources": sources, "workflows": workflows}


def frozen_runs(
    workflow: str, split: str, raw: dict, ids: list[str], endpoints: dict
) -> dict:
    runs = {}
    for role, side in (("A", "baseline"), ("B", "subject")):
        records = {row["id"]: row for row in raw[role]}
        rows = []
        for case_id in ids:
            record = records[case_id]
            if not complete(record):
                raise ValueError("selected record is not complete")
            # Full raw context/scores remain in the source export, outside judge records.
            rows.append(
                {
                    "id": case_id,
                    "input": input_text(record),
                    "expected": record["expected"],
                    "metadata": record["metadata"],
                    "output": record["output"],
                    "scores": {},
                    "context": {},
                    "error": None,
                }
            )
        runs[side] = make_run(
            rows,
            source={"name": "K2 frozen answer reference", "version": "1"},
            run_id=f"k2-32b-{workflow}-{split}-{side}",
            artifact_digest=endpoints[role]["header"]["artifact_digest"],
            source_digest="sha256:" + sha(canonical_payload(raw[role])),
        )
    case_set = {
        "format": "invarlock/evaluation-case-set-v1",
        "cases": [
            {k: r[k] for k in ("id", "input", "expected", "metadata")}
            for r in runs["baseline"]["records"]
        ],
    }
    for run in runs.values():
        validate_run_case_set(run, case_set_digest(case_set))
    return {**runs, "case_set": case_set}


def bind_contracts(template: dict, runs: dict, split: str) -> dict:
    if set(template) != {"plan", "analysis_policy"}:
        raise ValueError(
            "explicit judge configuration requires plan and analysis_policy templates"
        )
    if set(template["plan"]) != {
        "rubric",
        "prompt",
        "judge",
        "parser",
        "scale",
        "schedule",
    }:
        raise ValueError("judge plan template has unexpected fields")
    if set(template["analysis_policy"]) != {"pilot", "final"}:
        raise ValueError("judge policy templates must cover pilot and final")
    plan = {
        "format": "invarlock/judge-measurement-plan-v1",
        "profile_id": "text-frozen-answer-v1",
        **copy.deepcopy(template["plan"]),
    }
    plan.update(
        case_set_sha256=case_set_digest(runs["case_set"]),
        baseline_run_sha256=run_digest(runs["baseline"]),
        subject_run_sha256=run_digest(runs["subject"]),
    )
    a = {row["id"]: row for row in runs["baseline"]["records"]}
    b = {row["id"]: row for row in runs["subject"]["records"]}
    plan["rubric"]["sha256"] = sha(plan["rubric"]["text"].encode())
    plan["sampling"] = {
        "basis": "curated_benchmark",
        "unit_weighting": "equal",
        "within_unit_weighting": "equal_cases",
        "case_units": [
            {
                "case_id": i,
                "unit_id": "unit-"
                + sha(a[i]["metadata"]["source_cluster_id"].encode()),
            }
            for i in sorted(a)
        ],
    }
    plan["schedule"]["expected_trials"] = 2 * len(a) * plan["schedule"]["repetitions"]
    plan["answer_bindings"] = [
        {
            "case_id": i,
            "baseline_answer_sha256": sha(a[i]["output"].encode()),
            "subject_answer_sha256": sha(b[i]["output"].encode()),
            "baseline_request_sha256": sha(
                render_judge_request(
                    plan, input_text=a[i]["input"], answer_text=a[i]["output"]
                )
            ),
            "subject_request_sha256": sha(
                render_judge_request(
                    plan, input_text=b[i]["input"], answer_text=b[i]["output"]
                )
            ),
        }
        for i in sorted(a)
    ]
    validate_measurement_plan(plan)
    policy = {
        **template["analysis_policy"][split],
        "plan_sha256": measurement_plan_digest(plan),
        "minimum_units": len(a),
    }
    decode_analysis_policy(policy, plan=plan)
    return {"plan": plan, "analysis_policy": policy}


def human_review(
    workflow: str,
    raw: dict,
    selection: dict,
    template: dict,
    stage: str = "final_validation",
) -> dict:
    if stage not in {"rubric_development", "final_validation"}:
        raise ValueError("unsupported blinded review stage")
    by_role = {role: {r["id"]: r for r in rows} for role, rows in raw.items()}
    rows = []
    chosen = (
        selection["pilot"]
        if stage == "rubric_development"
        else selection["human_review"]
    )
    for case_id in sorted(chosen, key=lambda i: rank(workflow, stage + "_order", i)):
        a, b = by_role["A"][case_id], by_role["B"][case_id]
        answers = [a["output"], b["output"]]
        if int(rank(workflow, stage + "_orientation", case_id), 16) % 2:
            answers.reverse()
        rows.append(
            {
                "review_id": "review-" + rank(workflow, stage + "_id", case_id),
                "input": input_text(a),
                "response_1": answers[0],
                "response_1_rating": None,
                "response_2": answers[1],
                "response_2_rating": None,
                "review_notes": None,
            }
        )
    rubric = template["plan"]["rubric"]
    scale = template["plan"]["scale"]
    return {
        "format": "invarlock/blinded-answer-review-v2",
        "workflow": workflow,
        "stage": stage,
        "instructions": "Assess each response independently using the frozen rubric. Enter exactly one allowed scale label in each response rating field. Use review_notes only for a brief rationale or ambiguity. No reference outcomes or model roles are supplied. Record judgments before obtaining the full source bundle.",
        "rubric": {
            "text": rubric["text"],
            "sha256": sha(rubric["text"].encode()),
        },
        "scale": copy.deepcopy(scale),
        "cases": rows,
    }


def derive(campaign: dict, templates: dict | None) -> dict[str, bytes]:
    if templates is not None and set(templates) != set(WORKFLOWS):
        raise ValueError("judge templates must explicitly cover both workflows")
    if set(campaign) != {"sources", "workflows"} or set(campaign["workflows"]) != set(
        WORKFLOWS
    ):
        raise ValueError("reference must contain both frozen workflows")
    output = {
        "sources.json": canonical_payload(campaign["sources"]),
        "study_status.json": canonical_payload(
            {
                "format": "invarlock/judge-reference-study-status-v1",
                "membership": "frozen_outcome_blind",
                "pilot_rubric_review": "not_recorded",
                "final_plan_status": "candidate_pending_pilot_review"
                if templates is not None
                else "not_constructed",
                "final_model_calls": 0,
            }
        ),
    }
    for workflow, value in campaign["workflows"].items():
        selection = select(value["inventory"], workflow)
        inventory_by_id = {row["id"]: row for row in value["inventory"]}
        selected = set(selection["pilot"] + selection["final"])
        for role in ("A", "B"):
            raw = value["raw"][role]
            if set(value["mappings"][role]) != selected:
                raise ValueError("raw mapping membership differs from selection")
            if len(raw) != len(selected) or {r["id"] for r in raw} != selected:
                raise ValueError("retained raw selection differs from frozen inventory")
            for row in raw:
                mapping = value["mappings"][role][row["id"]]
                if (
                    set(mapping) != {"path", "row_index", "row_sha256", "protocol_path"}
                    or type(mapping["row_index"]) is not int
                    or not 0 <= mapping["row_index"] < 250
                ):
                    raise ValueError("invalid raw row mapping")
                job = row["context"]["job_id"]
                if (
                    mapping["path"] != f"{ASSESSMENT}/blocks/{job}/records.json"
                    or mapping["protocol_path"] != f"{PREPARATION}/{job}/protocol.json"
                ):
                    raise ValueError("raw block provenance differs from mapping")
                if not job.startswith(
                    f"k2-32b-{role.lower()}-{workflow.replace('_', '-')}-250-"
                ):
                    raise ValueError("raw block role differs from mapping")
                if sha(canonical_payload(row)) != mapping["row_sha256"]:
                    raise ValueError("raw row mapping digest mismatch")
                if (
                    mapping["path"] not in campaign["sources"]
                    or mapping["protocol_path"] not in value["protocols"]
                ):
                    raise ValueError("raw row source or protocol missing")
                snapshot = value["protocols"][mapping["protocol_path"]]
                if (
                    snapshot["model"]["repo"] != "IFM/K2-Horizon-32B"
                    or snapshot["model"]["revision"] != MODEL_REVISION
                    or snapshot["generation"]["workload_sha256"]
                    != row["context"]["effective_workload_sha256"]
                ):
                    raise ValueError(
                        "selected raw workload or checkpoint differs from protocol"
                    )
                match = inventory_by_id[row["id"]]
                if row["metadata"] != match["metadata"]:
                    raise ValueError(
                        "selected metadata differs from eligible inventory"
                    )
        for name, content in value.items():
            output[f"{workflow}/{name}.json"] = canonical_payload(content)
        output[f"{workflow}/selection.json"] = canonical_payload(selection)
        if templates is not None:
            for stage in ("rubric_development", "final_validation"):
                output[f"human_review/{stage}/{workflow}.json"] = canonical_payload(
                    human_review(
                        workflow,
                        value["raw"],
                        selection,
                        templates[workflow],
                        stage,
                    )
                )
        for split in ("pilot", "final"):
            runs = frozen_runs(
                workflow, split, value["raw"], selection[split], value["endpoints"]
            )
            for side, run in runs.items():
                name = side if side == "case_set" else side + "_run"
                output[f"{workflow}/{split}/{name}.json"] = canonical_payload(run)
            if templates is not None:
                contracts = bind_contracts(templates[workflow], runs, split)
                if split == "final":
                    output[f"{workflow}/{split}/candidate_plan.json"] = (
                        canonical_payload(
                            {
                                "format": "invarlock/judge-candidate-plan-v1",
                                "status": "candidate_pending_pilot_review",
                                **contracts,
                            }
                        )
                    )
                else:
                    for name, value_ in contracts.items():
                        output[f"{workflow}/{split}/{name}.json"] = canonical_payload(
                            value_
                        )
    if templates is not None:
        output["judge_templates.json"] = canonical_payload(templates)
    return output


NOTICE_FILES = {
    "attribution/SGD-LICENSE.txt": "source-supplement-v2/sources/sgd/LICENSE.txt",
    "attribution/SGD-README.md": "source-supplement-v2/sources/sgd/README.md",
    "attribution/SQuAD-SOFTWARE-LICENSE.txt": "source-supplement-v2/sources/squad/LICENSE",
    "attribution/SQuAD-README.md": "source-supplement-v2/sources/squad/README.md",
}
ATTRIBUTION = """# Source attribution and changes

This reference contains frozen model answers and task prompts derived from
SQuAD 2.0 (Pranav Rajpurkar, Robin Jia and Percy Liang, *Know What You Don't
Know: Unanswerable Questions for SQuAD*, ACL 2018) and Schema-Guided Dialogue
(Abhinav Rastogi, Xiaoxue Zang, Srinivas Sunkara, Raghav Gupta and Pranav Khaitan,
*Towards Scalable Multi-Domain Conversational Agents: The Schema-Guided Dialogue
Dataset*, AAAI 2020). Dataset source URLs, revisions, sizes and hashes are retained
in source-attribution.json; unchanged publisher README and license files are
included beside this notice.

SGD source revision: e852981ae34990f4358979625854259302feaa78. SGD-derived data
is distributed under CC BY-SA 4.0; preserve the included license and attribution.
SQuAD includes Wikipedia-derived context. Preserve SQuAD and Wikipedia attribution
and applicable source terms. SQuAD-SOFTWARE-LICENSE.txt is the publisher's MIT
software license; it does not relicense all dataset text as MIT. Dataset material
is not relicensed under InvarLock's Apache-2.0 software license.

Changes: source dialogue schemas/current utterances and Wikipedia context/questions
were formatted as task prompts; K2 Horizon 32B generated frozen A/B answers.
This reference selects English QA and extraction source clusters with a published
hash rule, extracts the original single user-message text without repair, preserves
raw selected answers, metadata, original metrics and capture provenance separately,
and constructs new score-free evaluation records. It does not rerun generation,
change native scores, assert native runtime qualification, or imply endorsement by
model or dataset authors.
"""
README = """# K2 frozen-answer judge reference

This bundle freezes outcome-blind pilot and final subsets of an existing paired
K2 Horizon 32B campaign. It contains no new judge results or fabricated ratings.
Run the matching source checkout's helper with an installed InvarLock package:

```bash
python examples/judge_measurements_reference.py validate --bundle reference
```

Optional `--campaign-root` rechecks every source file and reconstructs extraction
from the original blocks and endpoints. Default validation is self-contained and
checks retained mappings, selection, source metadata, contract construction and
blinded review exports. It cannot authenticate omitted original block membership
from hashes alone. An independently obtained `--expected-sha256` manifest pin
protects the bundle boundary; a bundle cannot authorize itself.

Selection ranks compact sorted UTF-8 JSON [seed, workflow, split, case_id] using
SHA-256 with seed invarlock-k2-32b-judge-reference-v1. The pilot takes 20 cases per
stratum in rank order, skipping reused clusters. Final then takes one ranked case
from every remaining eligible cluster, without using answers or scores. No source
cluster appears twice across pilot and final within a workflow. QA is English
only; both sides must be complete. All excluded and eligible population counts
are reported from the retained inventory; full original inputs can be cross-checked
only when the optional campaign root is available.

Rubric-development review includes every pilot case (40 per workflow). Complete
this review before declaring a final rubric and plan immutable. Current final
plans are candidates pending that review: confirm the rubric unchanged or regenerate
plans before any final model call. Regeneration must preserve frozen final membership.
Final candidate_plan.json wraps the candidate plan and policy and is deliberately
not an executable judge-measurement-plan document. Only pilot/plan.json is ready
for the collection API at this stage.

The untouched final-validation review takes 40 cases per stratum from final with
its own hash ranking. Give reviewers only the appropriate
human_review/rubric_development/<workflow>.json or
human_review/final_validation/<workflow>.json. That file contains the frozen
rubric, allowed rating labels, empty rating and notes fields, anonymous review
IDs, unchanged input and two responses with independently hashed position order;
it exposes no native scores, source IDs or A/B role labels. Review order is also
independently ranked. Blinding is procedural: the full bundle and public
deterministic algorithm allow the study operator to recover the mapping. Freeze
human judgments before giving reviewers the full source bundle.

Judge plans and analysis policies, when supplied, are separately frozen per
workflow and split. They bind exact answers and rendered requests; repetitions
do not increase independent-unit counts. The final benchmark remains curated,
with no claim of traffic generalization. See attribution/ATTRIBUTION.md before
redistributing source-derived content.
"""


def notices(root: Path) -> tuple[dict[str, bytes], dict]:
    files = {
        name: read(safe_path(root, source), 1024 * 1024)
        for name, source in NOTICE_FILES.items()
    }
    files["attribution/ATTRIBUTION.md"] = ATTRIBUTION.encode()
    files["README.md"] = README.encode()
    manifest_name = "source-supplement-v2/source-manifest.json"
    raw = read(safe_path(root, manifest_name))
    source_manifest = parse_json_bytes(raw, label="dataset source manifest")
    filtered = [
        entry
        for entry in source_manifest["files"]
        if entry["dataset"] in {"sgd", "squad", "squad2"}
    ]
    if not filtered or not {"sgd", "squad"} <= {entry["dataset"] for entry in filtered}:
        raise ValueError("dataset attribution is incomplete")
    files["attribution/source-attribution.json"] = canonical_payload(filtered)
    pins = {
        source: {"sha256": sha(files[name]), "size_bytes": len(files[name])}
        for name, source in NOTICE_FILES.items()
    }
    pins[manifest_name] = {"sha256": sha(raw), "size_bytes": len(raw)}
    return files, pins


def build(campaign_root: Path, output: Path, templates: dict | None = None) -> dict:
    campaign = capture_campaign(campaign_root)
    notice_files, pins = notices(campaign_root)
    campaign["sources"].update(pins)
    files = {**derive(campaign, templates), **notice_files}
    if sum(map(len, files.values())) > MAX_BUNDLE_BYTES:
        raise ValueError("reference bundle exceeds total byte budget")
    manifest = {
        "format": FORMAT,
        "seed": SEED,
        "selection": "pilot-balanced-then-all-remaining-clusters-v1",
        "files": {
            name: {"sha256": sha(data), "size_bytes": len(data)}
            for name, data in sorted(files.items())
        },
        "new_model_calls": 0,
        "assurance": "retained-source-derivation-not-independent-authentication",
    }
    # All analysis and rendering completes before a new output directory is created.
    if output.exists() or output.is_symlink():
        raise ValueError("reference output must be new")
    output.mkdir(parents=False)
    for name, data in files.items():
        path = safe_path(output, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    with (output / "reference.json").open("xb") as stream:
        stream.write(canonical_payload(manifest))
    return validate_bundle(output)


def authenticated_bundle_files(bundle: Path, expected_sha256: str) -> dict[str, bytes]:
    """Return one content-pinned snapshot of a closed reference directory."""

    validate_bundle(bundle, expected_sha256=expected_sha256)
    manifest_bytes = read(bundle / "reference.json", 1024 * 1024)
    if sha(manifest_bytes) != expected_sha256:
        raise ValueError("independent reference manifest pin mismatch")
    manifest = parse_json_bytes(manifest_bytes, label="reference manifest")
    files = manifest["files"]
    retained: dict[str, bytes] = {}
    total = 0
    for name, pin in files.items():
        size = pin["size_bytes"]
        total += size
        if total > MAX_BUNDLE_BYTES:
            raise ValueError("reference exceeds total byte budget")
        data = read(safe_path(bundle, name), size)
        if len(data) != size or sha(data) != pin["sha256"]:
            raise ValueError("reference file digest or size mismatch")
        retained[name] = data
    # Recheck the closed directory after the snapshot. Later path changes cannot
    # alter the authenticated bytes returned to the caller.
    validate_bundle(bundle, expected_sha256=expected_sha256)
    return retained


def rebind_bundle(
    bundle: Path,
    output: Path,
    templates: dict,
    expected_sha256: str,
) -> dict:
    """Rebuild derived plans and review sheets over an authenticated frozen subset."""
    retained = authenticated_bundle_files(bundle, expected_sha256)

    def retained_json(name: str) -> Any:
        return parse_json_bytes(retained[name], label=name)

    campaign = {
        "sources": retained_json("sources.json"),
        "workflows": {
            workflow: {
                name: retained_json(f"{workflow}/{name}.json")
                for name in ("inventory", "raw", "mappings", "endpoints", "protocols")
            }
            for workflow in WORKFLOWS
        },
    }
    notice_names = set(NOTICE_FILES) | {
        "attribution/ATTRIBUTION.md",
        "attribution/source-attribution.json",
        "README.md",
    }
    files = {
        **derive(campaign, templates),
        **{name: retained[name] for name in notice_names},
    }
    if sum(map(len, files.values())) > MAX_BUNDLE_BYTES:
        raise ValueError("reference bundle exceeds total byte budget")
    rebound_manifest = {
        "format": FORMAT,
        "seed": SEED,
        "selection": "pilot-balanced-then-all-remaining-clusters-v1",
        "files": {
            name: {"sha256": sha(data), "size_bytes": len(data)}
            for name, data in sorted(files.items())
        },
        "new_model_calls": 0,
        "assurance": "retained-source-derivation-not-independent-authentication",
    }
    if output.exists() or output.is_symlink():
        raise ValueError("reference output must be new")
    output.mkdir(parents=False)
    for name, data in files.items():
        path = safe_path(output, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    with (output / "reference.json").open("xb") as stream:
        stream.write(canonical_payload(rebound_manifest))
    return validate_bundle(output)


def validate_bundle(
    bundle: Path, campaign_root: Path | None = None, expected_sha256: str | None = None
) -> dict:
    manifest_bytes = read(bundle / "reference.json", 1024 * 1024)
    if expected_sha256 is not None and sha(manifest_bytes) != expected_sha256:
        raise ValueError("independent reference manifest pin mismatch")
    manifest = parse_json_bytes(manifest_bytes, label="reference manifest")
    if (
        set(manifest)
        != {"format", "seed", "selection", "files", "new_model_calls", "assurance"}
        or manifest["format"] != FORMAT
        or manifest["seed"] != SEED
        or manifest["selection"] != "pilot-balanced-then-all-remaining-clusters-v1"
        or manifest["new_model_calls"] != 0
        or manifest["assurance"]
        != "retained-source-derivation-not-independent-authentication"
    ):
        raise ValueError("unsupported reference manifest")
    if not isinstance(manifest["files"], dict) or len(manifest["files"]) > 64:
        raise ValueError("unexpected reference inventory")
    files = {}
    total = 0
    for name, pin in manifest["files"].items():
        if (
            set(pin) != {"sha256", "size_bytes"}
            or type(pin["size_bytes"]) is not int
            or not 0 <= pin["size_bytes"] <= MAX_BYTES
        ):
            raise ValueError("invalid file pin")
        total += pin["size_bytes"]
        if total > MAX_BUNDLE_BYTES:
            raise ValueError("reference bundle exceeds total byte budget")
        data = read(safe_path(bundle, name), pin["size_bytes"])
        if len(data) != pin["size_bytes"] or sha(data) != pin["sha256"]:
            raise ValueError("reference file digest or size mismatch")
        files[name] = data
    actual = set()
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise ValueError("reference bundle must not contain symlinks")
        if path.is_file():
            actual.add(path.relative_to(bundle).as_posix())
    if actual != set(files) | {"reference.json"}:
        raise ValueError("reference contains unlisted or missing files")

    def json_file(name: str) -> Any:
        return parse_json_bytes(files[name], label=name)

    campaign = {
        "sources": json_file("sources.json"),
        "workflows": {
            workflow: {
                name: json_file(f"{workflow}/{name}.json")
                for name in ("inventory", "raw", "mappings", "endpoints", "protocols")
            }
            for workflow in WORKFLOWS
        },
    }
    templates = (
        json_file("judge_templates.json") if "judge_templates.json" in files else None
    )
    reproduced = derive(campaign, templates)
    if any(files.get(name) != value for name, value in reproduced.items()):
        raise ValueError("reference derivation does not reproduce retained artifacts")
    notice_names = set(NOTICE_FILES) | {
        "attribution/ATTRIBUTION.md",
        "attribution/source-attribution.json",
        "README.md",
    }
    if set(files) != set(reproduced) | notice_names:
        raise ValueError("unexpected derived artifact inventory")
    if (
        files["README.md"] != README.encode()
        or files["attribution/ATTRIBUTION.md"] != ATTRIBUTION.encode()
    ):
        raise ValueError("reference explanation or attribution was changed")
    for name, source in NOTICE_FILES.items():
        if campaign["sources"][source] != {
            "sha256": sha(files[name]),
            "size_bytes": len(files[name]),
        }:
            raise ValueError("retained attribution source pin mismatch")
    if campaign_root is not None:
        restored = capture_campaign(campaign_root)
        restored_notices, pins = notices(campaign_root)
        restored["sources"].update(pins)
        if restored != campaign or any(
            files[name] != value for name, value in restored_notices.items()
        ):
            raise ValueError("campaign source extraction differs from retained bundle")
    counts = {}
    for workflow, value in campaign["workflows"].items():
        selected = select(value["inventory"], workflow)
        by_id = {row["id"]: row for row in value["inventory"]}
        counts[workflow] = {
            "source_cases": 4000,
            "excluded_cases": 4000 - len(by_id),
            "eligible_cases": len(by_id),
            "eligible_clusters": len(
                {r["metadata"]["source_cluster_id"] for r in by_id.values()}
            ),
            **{
                split: {
                    "cases": len(ids),
                    "strata": dict(Counter(by_id[i]["metadata"]["slice"] for i in ids)),
                }
                for split, ids in selected.items()
            },
        }
    return {
        "ok": True,
        "manifest_sha256": sha(manifest_bytes),
        "campaign_cross_checked": campaign_root is not None,
        "independently_pinned": expected_sha256 is not None,
        "new_model_calls": 0,
        "counts": counts,
    }


MAX_ARCHIVE_BYTES = 32 * 1024 * 1024


def validate_archive(
    archive: Path, campaign_root: Path | None = None, expected_sha256: str | None = None
) -> dict:
    """Read a bounded ZIP transport without trusting archive paths or permissions."""
    payload = read(archive, MAX_ARCHIVE_BYTES)
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as packed:
            members = packed.infolist()
            names = [entry.filename for entry in members]
            if (
                len(names) > 65
                or len(names) != len(set(names))
                or "reference.json" not in names
            ):
                raise ValueError("unexpected archive inventory")
            if (
                sum(entry.file_size for entry in members)
                > MAX_BUNDLE_BYTES + 1024 * 1024
            ):
                raise ValueError("archive expansion exceeds total budget")
            with tempfile.TemporaryDirectory(
                prefix="invarlock-judge-reference-"
            ) as directory:
                root = Path(directory)
                for entry in members:
                    maximum = (
                        1024 * 1024 if entry.filename == "reference.json" else MAX_BYTES
                    )
                    if (
                        entry.file_size > maximum
                        or entry.is_dir()
                        or entry.flag_bits & 1
                        or stat.S_IFMT(entry.external_attr >> 16)
                        not in {0, stat.S_IFREG}
                        or entry.compress_type
                        not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    ):
                        raise ValueError("unsupported or oversized archive member")
                    target = safe_path(root, entry.filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as stream:
                        stream.write(packed.read(entry))
                return validate_bundle(root, campaign_root, expected_sha256)
    except zipfile.BadZipFile as exc:
        raise ValueError("invalid reference ZIP archive") from exc


def validate_reference(
    path: Path, campaign_root: Path | None = None, expected_sha256: str | None = None
) -> dict:
    return (
        validate_bundle(path, campaign_root, expected_sha256)
        if path.is_dir()
        else validate_archive(path, campaign_root, expected_sha256)
    )


def pack_bundle(bundle: Path, output: Path) -> dict:
    """Compress exactly validated files with fixed order, modes and timestamps."""
    verified = validate_bundle(bundle)
    manifest = obj(bundle / "reference.json")
    with zipfile.ZipFile(
        output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted([*manifest["files"], "reference.json"]):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = (stat.S_IFREG | 0o644) << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, read(safe_path(bundle, name)), compresslevel=9)
    checked = validate_archive(output, expected_sha256=verified["manifest_sha256"])
    payload = read(output, MAX_ARCHIVE_BYTES)
    checked["independently_pinned"] = False
    return {
        **checked,
        "archive_sha256": sha(payload),
        "archive_size_bytes": len(payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--campaign-root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--judge-templates", type=Path)
    check = commands.add_parser("validate")
    check.add_argument("--bundle", type=Path, required=True)
    check.add_argument("--campaign-root", type=Path)
    check.add_argument("--expected-sha256")
    pack = commands.add_parser("pack")
    pack.add_argument("--bundle", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    rebind = commands.add_parser("rebind")
    rebind.add_argument("--bundle", type=Path, required=True)
    rebind.add_argument("--output", type=Path, required=True)
    rebind.add_argument("--judge-templates", type=Path, required=True)
    rebind.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    try:
        result = (
            build(
                args.campaign_root,
                args.output,
                obj(args.judge_templates) if args.judge_templates else None,
            )
            if args.command == "build"
            else pack_bundle(args.bundle, args.output)
            if args.command == "pack"
            else rebind_bundle(
                args.bundle,
                args.output,
                obj(args.judge_templates),
                args.expected_sha256,
            )
            if args.command == "rebind"
            else validate_reference(
                args.bundle, args.campaign_root, args.expected_sha256
            )
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Reference validation failed: {exc}\n")
    print(canonical_payload(result).decode())


if __name__ == "__main__":
    main()
