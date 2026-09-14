from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from examples import judge_measurements_reference as ref


def inventory(workflow="grounded_qa", count=200):
    return [
        {
            "id": f"case-{i:04}",
            "metadata": {
                "dataset": "squad2" if workflow == "grounded_qa" else "sgd",
                "family": "fixture",
                "language": "en",
                "slice": ref.WORKFLOWS[workflow][i % 2],
                "source_cluster_id": f"cluster-{i:04}",
                "upstream_split": "test",
                "workflow": workflow,
            },
        }
        for i in range(count)
    ]


def campaign_fixture():
    result = {"sources": {}, "workflows": {}}
    for workflow in ref.WORKFLOWS:
        items = inventory(workflow)
        selected = ref.select(items, workflow)
        ids = set(selected["pilot"] + selected["final"])
        value = {
            "inventory": items,
            "raw": {},
            "mappings": {},
            "endpoints": {},
            "protocols": {},
        }
        for role in ("A", "B"):
            job = f"k2-32b-{role.lower()}-{workflow.replace('_', '-')}-250-01"
            path = f"{ref.ASSESSMENT}/blocks/{job}/records.json"
            protocol = f"{ref.PREPARATION}/{job}/protocol.json"
            endpoint = f"{ref.ASSESSMENT}/endpoints/k2-32b-{role}-{workflow}-snapshot-01/run.json"
            for name in (path, protocol, endpoint):
                result["sources"][name] = {"sha256": "1" * 64, "size_bytes": 10}
            value["protocols"][protocol] = {
                "model": {"repo": "IFM/K2-Horizon-32B", "revision": ref.MODEL_REVISION},
                "generation": {"workload_sha256": "2" * 64},
            }
            value["endpoints"][role] = {
                "path": endpoint,
                "header": {"artifact_digest": "sha256:" + "3" * 64},
            }
            value["raw"][role] = []
            value["mappings"][role] = {}
            for index, row in enumerate(items):
                if row["id"] not in ids:
                    continue
                record = {
                    **row,
                    "input": [{"role": "user", "content": f"Question {index}"}],
                    "expected": "reference",
                    "output": f"answer-{role}-{index}",
                    "scores": {"native_quality": 0.25},
                    "error": None,
                    "context": {
                        "capture_present": True,
                        "verified_complete": True,
                        "job_id": job,
                        "effective_workload_sha256": "2" * 64,
                        "native_result": {
                            "capture_complete": True,
                            "status": "complete",
                        },
                    },
                }
                value["raw"][role].append(record)
                value["mappings"][role][row["id"]] = {
                    "path": path,
                    "row_index": index,
                    "row_sha256": ref.sha(ref.canonical_payload(record)),
                    "protocol_path": protocol,
                }
        result["workflows"][workflow] = value
    return result


def test_selection_is_outcome_free_stable_and_cluster_disjoint():
    rows = inventory()
    selected = ref.select(rows, "grounded_qa")
    assert selected == ref.select(list(reversed(rows)), "grounded_qa")
    assert len(selected["pilot"]) == 40
    assert len(selected["final"]) == 160
    assert len(selected["human_review"]) == 80
    assert set(selected["human_review"]) <= set(selected["final"])
    assert not set(selected["pilot"]) & set(selected["final"])
    altered = copy.deepcopy(rows)
    altered[0]["native_score"] = 1
    with pytest.raises(ValueError, match="outcome-free"):
        ref.select(altered, "grounded_qa")


def test_duplicate_clusters_do_not_increase_final_units():
    rows = inventory()
    duplicate = copy.deepcopy(rows[0])
    duplicate["id"] = "alternative-case"
    result = ref.select([*rows, duplicate], "grounded_qa")
    assert len(result["pilot"]) + len(result["final"]) == 200


@pytest.mark.parametrize("alter", ["language", "duplicate", "too_small"])
def test_selection_rejects_ineligible_or_infeasible_population(alter):
    rows = inventory()
    if alter == "language":
        rows[0]["metadata"]["language"] = "es"
    elif alter == "duplicate":
        rows.append(rows[0])
    else:
        rows = rows[:30]
    with pytest.raises(ValueError):
        ref.select(rows, "grounded_qa")


def test_frozen_contracts_bind_answers_requests_and_cluster_units():
    campaign = campaign_fixture()
    templates = ref.obj(
        Path(__file__).parents[2]
        / "examples/judge-measurements/k2-judge-templates.json"
    )
    files = ref.derive(campaign, templates)
    candidate = json.loads(files["grounded_qa/final/candidate_plan.json"])
    assert candidate["status"] == "candidate_pending_pilot_review"
    plan, policy = candidate["plan"], candidate["analysis_policy"]
    run = json.loads(files["grounded_qa/final/baseline_run.json"])
    assert policy["minimum_units"] == 160
    assert plan["schedule"]["expected_trials"] == 960
    assert len({x["unit_id"] for x in plan["sampling"]["case_units"]}) == 160
    assert all(row["scores"] == {} and row["context"] == {} for row in run["records"])
    row, binding = run["records"][0], plan["answer_bindings"][0]
    assert binding["baseline_request_sha256"] == ref.sha(
        ref.render_judge_request(
            plan, input_text=row["input"], answer_text=row["output"]
        )
    )
    assert plan["judge"]["requested_model"] == "openai/gpt-5.6-sol"


def test_human_review_does_not_expose_roles_scores_or_case_ids():
    campaign = campaign_fixture()["workflows"]["grounded_qa"]
    selection = ref.select(campaign["inventory"], "grounded_qa")
    template = ref.obj(
        Path(__file__).parents[2]
        / "examples/judge-measurements/k2-judge-templates.json"
    )["grounded_qa"]
    review = ref.human_review("grounded_qa", campaign["raw"], selection, template)
    assert review["format"] == "invarlock/blinded-answer-review-v2"
    assert review["rubric"]["text"] == template["plan"]["rubric"]["text"]
    assert review["rubric"]["sha256"] == ref.sha(
        template["plan"]["rubric"]["text"].encode()
    )
    assert review["scale"] == template["plan"]["scale"]
    for row in review["cases"]:
        assert set(row) == {
            "review_id",
            "input",
            "response_1",
            "response_1_rating",
            "response_2",
            "response_2_rating",
            "review_notes",
        }
        assert row["review_id"].startswith("review-")
        assert row["response_1_rating"] is None
        assert row["response_2_rating"] is None
        assert row["review_notes"] is None
    # Fixture answers intentionally identify their origin so both orientations can be checked.
    assert {row["response_1"].split("-")[1] for row in review["cases"]} == {"A", "B"}


@pytest.mark.parametrize("alter", ["raw", "metadata", "mapping"])
def test_source_derivation_rejects_inconsistent_retained_records(alter):
    campaign = campaign_fixture()
    value = campaign["workflows"]["grounded_qa"]
    row = value["raw"]["A"][0]
    if alter == "raw":
        row["output"] = "different"
    elif alter == "metadata":
        value["inventory"][0]["metadata"] = {
            **value["inventory"][0]["metadata"],
            "family": "changed",
        }
    else:
        value["mappings"]["A"][row["id"]]["row_index"] = -1
    with pytest.raises(ValueError):
        ref.derive(campaign, None)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    campaign = campaign_fixture()
    notice_data = dict.fromkeys(ref.NOTICE_FILES, b"Retained source notice\n")
    notice_data.update(
        {
            "README.md": ref.README.encode(),
            "attribution/ATTRIBUTION.md": ref.ATTRIBUTION.encode(),
            "attribution/source-attribution.json": b"[]",
        }
    )
    pins = {
        source: {
            "sha256": ref.sha(notice_data[name]),
            "size_bytes": len(notice_data[name]),
        }
        for name, source in ref.NOTICE_FILES.items()
    }
    monkeypatch.setattr(ref, "capture_campaign", lambda root: copy.deepcopy(campaign))
    monkeypatch.setattr(
        ref, "notices", lambda root: (copy.deepcopy(notice_data), copy.deepcopy(pins))
    )
    output = tmp_path / "bundle"
    templates = ref.obj(
        Path(__file__).parents[2]
        / "examples/judge-measurements/k2-judge-templates.json"
    )
    result = ref.build(tmp_path, output, templates)
    return output, result


def test_self_contained_validation_never_needs_campaign(bundle, monkeypatch):
    path, result = bundle

    def forbidden(root):
        raise AssertionError("campaign access must be optional")

    monkeypatch.setattr(ref, "capture_campaign", forbidden)
    verified = ref.validate_bundle(path, expected_sha256=result["manifest_sha256"])
    assert verified["ok"] and verified["independently_pinned"]
    assert not verified["campaign_cross_checked"]
    with pytest.raises(ValueError, match="pin mismatch"):
        ref.validate_bundle(path, expected_sha256="0" * 64)


def test_rebind_reuses_only_an_authenticated_frozen_subset(
    bundle, tmp_path, monkeypatch
):
    path, result = bundle
    templates = ref.obj(
        Path(__file__).parents[2]
        / "examples/judge-measurements/k2-judge-templates.json"
    )
    original_selection = (path / "grounded_qa/selection.json").read_bytes()
    original_plan = (path / "grounded_qa/pilot/plan.json").read_bytes()
    templates["grounded_qa"]["plan"]["rubric"]["text"] += " Clarified."
    rebound = tmp_path / "rebound"
    verified = ref.rebind_bundle(
        path,
        rebound,
        templates,
        result["manifest_sha256"],
    )
    assert verified["ok"]
    assert (rebound / "grounded_qa/selection.json").read_bytes() == original_selection
    assert (rebound / "grounded_qa/pilot/plan.json").read_bytes() != original_plan
    assert (path / "grounded_qa/pilot/plan.json").read_bytes() == original_plan
    with pytest.raises(ValueError, match="pin mismatch"):
        ref.rebind_bundle(path, tmp_path / "rejected", templates, "0" * 64)

    original_validate = ref.validate_bundle
    calls = 0

    def replace_after_validation(*args, **kwargs):
        nonlocal calls
        result = original_validate(*args, **kwargs)
        calls += 1
        if calls == 1:
            endpoint = path / "grounded_qa/endpoints.json"
            changed = ref.obj(endpoint)
            changed["A"]["header"]["artifact_digest"] = "sha256:" + "9" * 64
            endpoint.write_bytes(ref.canonical_payload(changed))
        return result

    monkeypatch.setattr(ref, "validate_bundle", replace_after_validation)
    with pytest.raises(ValueError, match="digest or size mismatch"):
        ref.rebind_bundle(
            path,
            tmp_path / "substituted",
            templates,
            result["manifest_sha256"],
        )
    assert not (tmp_path / "substituted").exists()


def repin(bundle, name, data):
    (bundle / name).write_bytes(data)
    manifest = ref.obj(bundle / "reference.json")
    manifest["files"][name] = {"sha256": ref.sha(data), "size_bytes": len(data)}
    (bundle / "reference.json").write_bytes(ref.canonical_payload(manifest))


def test_blinded_review_substitution_fails_even_with_updated_file_hash(bundle):
    path, _ = bundle
    name = "human_review/final_validation/grounded_qa.json"
    changed = ref.obj(path / name)
    changed["cases"][0]["native_score"] = 1
    repin(path, name, ref.canonical_payload(changed))
    with pytest.raises(ValueError, match="derivation"):
        ref.validate_bundle(path)


def test_bundle_symlink_and_unlisted_files_are_rejected(bundle):
    path, _ = bundle
    (path / "extra.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="unlisted"):
        ref.validate_bundle(path)
    (path / "extra.txt").unlink()
    (path / "extra.txt").symlink_to(path / "README.md")
    with pytest.raises(ValueError, match="symlink"):
        ref.validate_bundle(path)


def test_optional_campaign_cross_check_reconstructs(bundle, tmp_path):
    path, _ = bundle
    assert ref.validate_bundle(path, campaign_root=tmp_path)["campaign_cross_checked"]


@pytest.fixture
def campaign_sources(tmp_path, monkeypatch):
    data = {}
    metadata = {w: inventory(w, 4000) for w in ref.WORKFLOWS}
    for rows in metadata.values():
        for i, row in enumerate(rows):
            row["metadata"]["source_cluster_id"] = f"cluster-{i % 200:04}"
    data[ref.PLANNED] = {
        "format": "invarlock/pipeline-case-set-v1",
        "cases": [
            {
                **r,
                "input": [{"role": "user", "content": r["id"]}],
                "expected": "reference",
            }
            for r in metadata["grounded_qa"]
        ],
    }
    for workflow in ref.WORKFLOWS:
        for role in ("A", "B"):
            full = []
            for block in range(1, 17):
                job = (
                    f"k2-32b-{role.lower()}-{workflow.replace('_', '-')}-250-{block:02}"
                )
                records = []
                for item in metadata[workflow][(block - 1) * 250 : block * 250]:
                    record = {
                        **item,
                        "input": [{"role": "user", "content": item["id"]}],
                        "expected": "reference",
                        "output": role,
                        "scores": {"native_quality": 0},
                        "error": None,
                        "context": {
                            "job_id": job,
                            "capture_present": True,
                            "verified_complete": True,
                            "effective_workload_sha256": "2" * 64,
                            "native_result": {
                                "capture_complete": True,
                                "status": "complete",
                            },
                        },
                    }
                    records.append(record)
                    full.append(record)
                data[f"{ref.ASSESSMENT}/blocks/{job}/records.json"] = records
                model = dict.fromkeys(
                    (
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
                )
                model.update(repo="IFM/K2-Horizon-32B", revision=ref.MODEL_REVISION)
                protocol = dict.fromkeys(
                    (
                        "source_workload_sha256",
                        "context_length",
                        "maximum_output_tokens",
                        "stop_policy",
                        "content_policy",
                        "input_policy",
                    ),
                    "fixture",
                )
                protocol.update(model=model, workload_sha256="2" * 64)
                data[f"{ref.PREPARATION}/{job}/protocol.json"] = protocol
            data[
                f"{ref.ASSESSMENT}/endpoints/k2-32b-{role}-{workflow}-snapshot-01/run.json"
            ] = {
                "format": "invarlock/pipeline-run-v1",
                "artifact_digest": "sha256:" + "3" * 64,
                "records": full,
            }

    def read(path, maximum=ref.MAX_BYTES):
        return ref.canonical_payload(data[path.relative_to(tmp_path).as_posix()])

    monkeypatch.setattr(ref, "read", read)
    return tmp_path, data


def test_full_campaign_extraction_and_outcome_independence(campaign_sources):
    root, data = campaign_sources
    original = ref.capture_campaign(root)
    assert len(original["sources"]) == 133
    for name, value in data.items():
        if name.endswith("records.json"):
            for row in value:
                row["scores"]["native_quality"] = 1
                row["output"] = "entirely different outcome"
    changed = ref.capture_campaign(root)
    for workflow in ref.WORKFLOWS:
        assert (
            original["workflows"][workflow]["inventory"]
            == changed["workflows"][workflow]["inventory"]
        )
        assert (
            original["workflows"][workflow]["mappings"]
            != changed["workflows"][workflow]["mappings"]
        )


@pytest.mark.parametrize(
    "alter",
    [
        "checkpoint",
        "planned_count",
        "block_count",
        "duplicate",
        "provenance",
        "endpoint",
        "membership",
        "paired_input",
        "planned_input",
    ],
)
def test_source_crosscheck_rejects_invalid_originals(campaign_sources, alter):
    root, data = campaign_sources
    block = data[f"{ref.ASSESSMENT}/blocks/k2-32b-a-grounded-qa-250-01/records.json"]
    if alter == "checkpoint":
        data[f"{ref.PREPARATION}/k2-32b-a-grounded-qa-250-01/protocol.json"]["model"][
            "repo"
        ] = "other/model"
    elif alter == "planned_count":
        data[ref.PLANNED]["cases"].pop()
    elif alter == "block_count":
        block.pop()
    elif alter == "duplicate":
        block[1]["id"] = block[0]["id"]
    elif alter == "provenance":
        block[0]["context"]["job_id"] = "wrong-block"
    elif alter == "endpoint":
        data[f"{ref.ASSESSMENT}/endpoints/k2-32b-A-grounded_qa-snapshot-01/run.json"][
            "records"
        ].pop()
    elif alter == "membership":
        block[0]["id"] = "different-case"
    elif alter == "paired_input":
        block[0]["expected"] = "changed"
    else:
        data[ref.PLANNED]["cases"][0]["expected"] = "changed"
    with pytest.raises(ValueError):
        ref.capture_campaign(root)


def test_capture_excludes_only_incomplete_or_non_english_pairs(campaign_sources):
    root, data = campaign_sources
    for role in ("a", "b"):
        row = data[
            f"{ref.ASSESSMENT}/blocks/k2-32b-{role}-grounded-qa-250-01/records.json"
        ][0]
        row["metadata"]["language"] = "es"
    data[ref.PLANNED]["cases"][0]["metadata"]["language"] = "es"
    data[f"{ref.ASSESSMENT}/blocks/k2-32b-a-grounded-qa-250-01/records.json"][1][
        "error"
    ] = "transport failure"
    captured = ref.capture_campaign(root)
    assert len(captured["workflows"]["grounded_qa"]["inventory"]) == 3998


def test_notice_sources_and_software_license_boundary(tmp_path):
    for source in ref.NOTICE_FILES.values():
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source license or README")
    path = tmp_path / "source-supplement-v2/source-manifest.json"
    path.write_text(
        json.dumps(
            {
                "files": [
                    {"dataset": "sgd"},
                    {"dataset": "squad"},
                    {"dataset": "unrelated"},
                ]
            }
        )
    )
    files, pins = ref.notices(tmp_path)
    assert len(pins) == 5
    assert len(json.loads(files["attribution/source-attribution.json"])) == 2
    assert (
        b"does not relicense all dataset text as MIT"
        in files["attribution/ATTRIBUTION.md"]
    )
    path.write_text('{"files": []}')
    with pytest.raises(ValueError, match="attribution"):
        ref.notices(tmp_path)


@pytest.mark.parametrize(
    "path", ["../escape", "/absolute", "a\\b", "a//b", "", "a/./b"]
)
def test_source_path_boundary(path, tmp_path):
    with pytest.raises(ValueError):
        ref.safe_path(tmp_path, path)


@pytest.mark.parametrize(
    "alter", ["workflow", "metadata_fields", "metadata_values", "id", "small_review"]
)
def test_additional_inventory_boundary(alter):
    rows = inventory()
    if alter == "workflow":
        rows[0]["metadata"]["workflow"] = "other"
    elif alter == "metadata_fields":
        rows[0]["metadata"]["unknown"] = "value"
    elif alter == "metadata_values":
        rows[0]["metadata"]["family"] = 1
    elif alter == "id":
        rows[0]["id"] = ""
    else:
        rows = rows[:60]
    with pytest.raises(ValueError):
        ref.select(rows, "grounded_qa")


@pytest.mark.parametrize("messages", [[], [{"role": "system", "content": "text"}]])
def test_single_user_message_profile(messages):
    with pytest.raises(ValueError):
        ref.input_text({"input": messages})


@pytest.mark.parametrize(
    "alter",
    [
        "format",
        "inventory",
        "pin",
        "size",
        "total",
        "missing",
        "extra",
        "readme",
        "notice",
        "campaign",
    ],
)
def test_validation_boundaries(bundle, monkeypatch, alter, tmp_path):
    path, _ = bundle
    manifest = ref.obj(path / "reference.json")
    if alter == "format":
        manifest["format"] = "other"
    elif alter == "inventory":
        manifest["files"] = []
    elif alter == "pin":
        manifest["files"]["README.md"]["size_bytes"] = -1
    elif alter == "size":
        manifest["files"]["README.md"]["sha256"] = "0" * 64
    elif alter == "total":
        monkeypatch.setattr(ref, "MAX_BUNDLE_BYTES", 1)
    elif alter == "missing":
        (path / "README.md").unlink()
    elif alter == "extra":
        repin(path, "unexpected.txt", b"extra")
        manifest = ref.obj(path / "reference.json")
    elif alter == "readme":
        repin(path, "README.md", b"altered explanation")
        manifest = ref.obj(path / "reference.json")
    elif alter == "notice":
        repin(path, "attribution/SGD-LICENSE.txt", b"altered notice")
        manifest = ref.obj(path / "reference.json")
    else:
        monkeypatch.setattr(ref, "capture_campaign", lambda root: {})
    (path / "reference.json").write_bytes(ref.canonical_payload(manifest))
    with pytest.raises((ValueError, OSError, KeyError)):
        ref.validate_bundle(
            path, campaign_root=tmp_path if alter == "campaign" else None
        )


def test_cli_build_and_validate_paths(bundle, monkeypatch, tmp_path, capsys):
    path, result = bundle
    monkeypatch.setattr("sys.argv", ["reference", "validate", "--bundle", str(path)])
    ref.main()
    assert json.loads(capsys.readouterr().out)["ok"]
    monkeypatch.setattr(
        "sys.argv",
        [
            "reference",
            "build",
            "--campaign-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "new"),
        ],
    )
    ref.main()
    assert json.loads(capsys.readouterr().out)["ok"]
    monkeypatch.setattr(
        "sys.argv",
        ["reference", "validate", "--bundle", str(path), "--expected-sha256", "wrong"],
    )
    with pytest.raises(SystemExit) as exc:
        ref.main()
    assert exc.value.code == 2


def test_existing_or_oversized_publication_rejected(bundle, monkeypatch, tmp_path):
    path, _ = bundle
    with pytest.raises(ValueError, match="new"):
        ref.build(tmp_path, path)
    monkeypatch.setattr(ref, "MAX_BUNDLE_BYTES", 1)
    with pytest.raises(ValueError, match="budget"):
        ref.build(tmp_path, tmp_path / "oversize")


def test_separate_pilot_review_and_candidate_final_plan_status():
    campaign = campaign_fixture()
    templates = ref.obj(
        Path(__file__).parents[2]
        / "examples/judge-measurements/k2-judge-templates.json"
    )
    files = ref.derive(campaign, templates)
    status = json.loads(files["study_status.json"])
    assert status["final_plan_status"] == "candidate_pending_pilot_review"
    pilot = json.loads(files["human_review/rubric_development/grounded_qa.json"])
    final = json.loads(files["human_review/final_validation/grounded_qa.json"])
    assert len(pilot["cases"]) == 40 and len(final["cases"]) == 80
    assert pilot["stage"] == "rubric_development"
    assert not {r["review_id"] for r in pilot["cases"]} & {
        r["review_id"] for r in final["cases"]
    }
    changed = copy.deepcopy(templates)
    changed["grounded_qa"]["plan"]["rubric"]["text"] += (
        " Additional pilot clarification."
    )
    regenerated = ref.derive(campaign, changed)
    assert (
        regenerated["grounded_qa/selection.json"] == files["grounded_qa/selection.json"]
    )
    assert (
        regenerated["grounded_qa/final/candidate_plan.json"]
        != files["grounded_qa/final/candidate_plan.json"]
    )
    with pytest.raises(ValueError, match="stage"):
        ref.human_review(
            "grounded_qa",
            campaign["workflows"]["grounded_qa"]["raw"],
            {},
            templates["grounded_qa"],
            "unsupported",
        )


@pytest.mark.parametrize(
    "alter",
    [
        "template",
        "plan",
        "policy",
        "workflow",
        "mappings",
        "raw_count",
        "provenance",
        "role",
        "protocol",
        "workload",
    ],
)
def test_derivation_closed_shapes(alter):
    campaign = campaign_fixture()
    templates = ref.obj(
        Path(__file__).parents[2]
        / "examples/judge-measurements/k2-judge-templates.json"
    )
    value = campaign["workflows"]["grounded_qa"]
    row = value["raw"]["A"][0]
    mapping = value["mappings"]["A"][row["id"]]
    if alter == "template":
        templates["grounded_qa"]["unknown"] = True
    elif alter == "plan":
        templates["grounded_qa"]["plan"]["unknown"] = True
    elif alter == "policy":
        templates["grounded_qa"]["analysis_policy"].pop("pilot")
    elif alter == "workflow":
        campaign["workflows"].pop("extraction")
    elif alter == "mappings":
        value["mappings"]["A"].pop(row["id"])
    elif alter == "raw_count":
        value["raw"]["A"].pop()
    elif alter == "provenance":
        mapping["path"] = "other/path"
    elif alter == "role":
        job = row["context"]["job_id"].replace("32b-a-", "32b-b-")
        row["context"]["job_id"] = job
        mapping["path"] = f"{ref.ASSESSMENT}/blocks/{job}/records.json"
        mapping["protocol_path"] = f"{ref.PREPARATION}/{job}/protocol.json"
    elif alter == "protocol":
        value["protocols"].pop(mapping["protocol_path"])
    else:
        value["protocols"][mapping["protocol_path"]]["generation"][
            "workload_sha256"
        ] = "other"
    with pytest.raises(ValueError):
        ref.derive(campaign, templates)


def test_incomplete_frozen_run_and_unknown_workflow_rejected():
    campaign = campaign_fixture()["workflows"]["grounded_qa"]
    row = campaign["raw"]["A"][0]
    row["error"] = "failure"
    with pytest.raises(ValueError, match="complete"):
        ref.frozen_runs(
            "grounded_qa", "pilot", campaign["raw"], [row["id"]], campaign["endpoints"]
        )
    with pytest.raises(ValueError, match="workflow"):
        ref.select([], "unsupported")
    with pytest.raises(ValueError, match="both workflows"):
        ref.derive(campaign_fixture(), {})


def test_archive_transport_is_deterministic_and_self_contained(
    bundle, tmp_path, monkeypatch, capsys
):
    path, result = bundle
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    packed = ref.pack_bundle(path, a)
    ref.pack_bundle(path, b)
    assert a.read_bytes() == b.read_bytes()
    assert packed["archive_sha256"] == ref.sha(a.read_bytes())
    assert not packed["independently_pinned"]
    assert ref.validate_reference(a, expected_sha256=result["manifest_sha256"])["ok"]
    monkeypatch.setattr(
        "sys.argv",
        [
            "reference",
            "pack",
            "--bundle",
            str(path),
            "--output",
            str(tmp_path / "c.zip"),
        ],
    )
    ref.main()
    assert json.loads(capsys.readouterr().out)["ok"]
    with pytest.raises(FileExistsError):
        ref.pack_bundle(path, a)


@pytest.mark.parametrize(
    "alter",
    [
        "invalid",
        "inventory",
        "traversal",
        "symlink",
        "directory",
        "compression",
        "member_size",
        "total_size",
    ],
)
def test_archive_rejects_unsafe_transport(tmp_path, monkeypatch, alter):
    import stat
    import zipfile

    path = tmp_path / "bad.zip"
    if alter == "invalid":
        path.write_bytes(b"not a zip")
    else:
        with zipfile.ZipFile(path, "w") as packed:
            if alter != "inventory":
                packed.writestr("reference.json", b"{}")
            name = (
                "../outside"
                if alter == "traversal"
                else "directory/"
                if alter == "directory"
                else "entry"
            )
            entry = zipfile.ZipInfo(name)
            if alter == "symlink":
                entry.create_system = 3
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            if alter == "compression":
                entry.compress_type = zipfile.ZIP_BZIP2
            packed.writestr(entry, b"content")
        if alter == "member_size":
            monkeypatch.setattr(ref, "MAX_BYTES", 1)
        if alter == "total_size":
            monkeypatch.setattr(ref, "MAX_BUNDLE_BYTES", -1024 * 1024)
    with pytest.raises(ValueError):
        ref.validate_archive(path)
    assert not (tmp_path / "outside").exists()


def test_final_candidate_cannot_be_loaded_as_an_executable_judge_plan():
    from invarlock.judge_measurements.contracts import validate_measurement_plan

    templates = ref.obj(
        Path(__file__).parents[2]
        / "examples/judge-measurements/k2-judge-templates.json"
    )
    files = ref.derive(campaign_fixture(), templates)
    assert "grounded_qa/final/plan.json" not in files
    assert "extraction/final/analysis_policy.json" not in files
    with pytest.raises(ValueError):
        validate_measurement_plan(
            json.loads(files["grounded_qa/final/candidate_plan.json"])
        )
    validate_measurement_plan(json.loads(files["grounded_qa/pilot/plan.json"]))


def test_retained_public_reference_archive_replays_offline():
    import zipfile

    package = (
        Path(__file__).parents[2] / "examples/judge-measurements/references/k2-32b"
    )
    published = ref.obj(package / "archive.json")
    archive = package / published["archive"]["path"]
    raw = ref.read(archive, ref.MAX_ARCHIVE_BYTES)
    assert len(raw) == published["archive"]["size_bytes"] < 10 * 1024 * 1024
    assert ref.sha(raw) == published["archive"]["sha256"]
    verified = ref.validate_reference(
        archive, expected_sha256=published["reference_manifest_sha256"]
    )
    assert verified["ok"] and not verified["campaign_cross_checked"]
    assert verified["counts"]["grounded_qa"]["final"]["cases"] == 422
    assert verified["counts"]["extraction"]["final"]["cases"] == 1288
    with zipfile.ZipFile(archive) as source:
        status = json.loads(source.read("study_status.json"))
        assert (
            status["final_plan_status"]
            == published["final_plan_status"]
            == "candidate_pending_pilot_review"
        )
        assert "grounded_qa/final/plan.json" not in source.namelist()
        for name in ("ATTRIBUTION.md", "SGD-LICENSE.txt", "SQuAD-SOFTWARE-LICENSE.txt"):
            assert source.read("attribution/" + name) == (package / name).read_bytes()
