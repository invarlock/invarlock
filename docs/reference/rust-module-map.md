# Rust module inventory

> **Reference**
>
> **Surface:** Target ownership for every current Python source module
>
> **Stability:** Experimental proposal; released behavior remains authoritative
>
> **Use this page when:** Locating a module in the proposed migration

Target ownership for all 141 Python modules at the fixed reference commit.
This is a design inventory, not a statement that these modules have been ported.
Mixed modules retain Python interfaces while portable semantics move to Rust.
Package initializers remain adapters. Non-Python schemas and packaged data retain
one authoritative source and require embedded-byte parity checks.

See the [architecture](rust-core-design.md) for boundaries and the
[comparison protocol](rust-comparison-protocol.md) for required evidence.
The machine-readable inventory is `experiments/rust/module-map.json`.

| Source module under `src/invarlock/` | Target ownership | Destination | Stage |
| --- | --- | --- | --- |
| `__init__.py` | Python | integration / facade / presentation | retained |
| `__main__.py` | Python | integration / facade / presentation | retained |
| `_bounded_subprocess.py` | Python | integration / facade / presentation | retained |
| `_optional_runtime_profiles.py` | Python | integration / facade / presentation | retained |
| `acceptance_attestation.py` | Rust with Python adapters | evidence.acceptance | 2–5 |
| `captured_contracts.py` | Rust with Python adapters | contracts.captured | 2–5 |
| `captured_evaluation.py` | Split | evidence.captured; Python API/capture wrapper | 3–6 |
| `captured_evidence_publication.py` | Rust with Python adapters | evidence.captured | 2–5 |
| `captured_normalization.py` | Rust with Python adapters | contracts.request | 2–5 |
| `captured_reporting.py` | Python | integration / facade / presentation | retained |
| `captured_verification.py` | Rust with Python adapters | evidence.captured | 2–5 |
| `cli/__init__.py` | Python | integration / facade / presentation | retained |
| `cli/app.py` | Python | integration / facade / presentation | retained |
| `cli/evaluation_setup.py` | Python | integration / facade / presentation | retained |
| `cli/evaluation_workflow.py` | Python | integration / facade / presentation | retained |
| `cli/runtime_profile.py` | Python | integration / facade / presentation | retained |
| `cli/verification_workflow.py` | Python | integration / facade / presentation | retained |
| `core/__init__.py` | Python | integration / facade / presentation | retained |
| `core/builtin_plugin_catalog.py` | Python | integration / facade / presentation | retained |
| `core/builtin_scorers.py` | Split | analysis.scoring; Python loading/resource adapters | 2–4 |
| `core/checkpoint_identity.py` | Split | contracts.identity; Python loading/resource adapters | 2–4 |
| `core/dataset_identity.py` | Split | contracts.identity; Python loading/resource adapters | 2–4 |
| `core/evaluation_request.py` | Split | contracts.request; Python loading/resource adapters | 2–4 |
| `core/registry.py` | Python | integration / facade / presentation | retained |
| `core/runtime_provider/__init__.py` | Split | contracts.runtime; Python protocol/value adapters | 4 |
| `core/runtime_provider/behavioral_observation.py` | Split | contracts.runtime; Python protocol/value adapters | 4 |
| `core/runtime_provider/behavioral_schedule.py` | Split | contracts.runtime; Python protocol/value adapters | 4 |
| `core/runtime_provider/request_bindings.py` | Split | contracts.runtime; Python protocol/value adapters | 4 |
| `core/runtime_provider/types.py` | Split | contracts.runtime; Python protocol/value adapters | 4 |
| `core/schedule_preparation.py` | Split | contracts.schedule; Python loading/resource adapters | 2–4 |
| `core/scorer_extension.py` | Split | contracts.scorer + analysis.scoring; Python loading/resource adapters | 2–4 |
| `core/scoring.py` | Split | analysis.scoring; Python loading/resource adapters | 2–4 |
| `diagnostics/__init__.py` | Python | integration / facade / presentation | retained |
| `diagnostics/observations.py` | Python | integration / facade / presentation | retained |
| `engine.py` | Split | evidence facade; Python API/capture wrapper | 3–6 |
| `evaluation_comparison/__init__.py` | Rust with Python adapters | analysis.comparison | 1–3 |
| `evaluation_comparison/capacity.py` | Rust with Python adapters | analysis.comparison | 1–3 |
| `evaluation_comparison/comparison.py` | Rust with Python adapters | analysis.comparison | 1–3 |
| `evaluation_comparison/likelihood.py` | Rust with Python adapters | analysis.comparison | 1–3 |
| `evaluation_oci.py` | Python | integration / facade / presentation | retained |
| `evaluation_record_contracts/__init__.py` | Rust with Python adapters | contracts.records | 2 |
| `evaluation_record_contracts/contracts.py` | Rust with Python adapters | contracts.records | 2 |
| `evaluation_record_contracts/validation_limits.py` | Rust with Python adapters | contracts.records | 2 |
| `evaluation_records/__init__.py` | Python | integration / facade / presentation | retained |
| `evaluation_records/adapters.py` | Python | integration / facade / presentation | retained |
| `evaluation_records/batch_integrations.py` | Python | integration / facade / presentation | retained |
| `evaluation_records/capture_facts.py` | Python | integration / facade / presentation | retained |
| `evaluation_records/cases.py` | Rust with Python adapters | contracts.records + filesystem | 2–3 |
| `evaluation_records/identity.py` | Rust with Python adapters | contracts.records + filesystem | 2–3 |
| `evaluation_records/integrations.py` | Python | integration / facade / presentation | retained |
| `evaluation_records/io.py` | Rust with Python adapters | contracts.records + filesystem | 2–3 |
| `evaluation_records/langfuse.py` | Python | integration / facade / presentation | retained |
| `evaluation_records/scalar_integrations.py` | Python | integration / facade / presentation | retained |
| `evaluation_records/templates.py` | Python | integration / facade / presentation | retained |
| `evaluation_run.py` | Python | integration / facade / presentation | retained |
| `evaluation_runtime.py` | Python | integration / facade / presentation | retained |
| `evaluation_side_worker.py` | Python | integration / facade / presentation | retained |
| `evaluation_transaction.py` | Python | integration / facade / presentation | retained |
| `evaluator_capture.py` | Split | contracts.records; Python API/capture wrapper | 3–6 |
| `evaluator_qualification.py` | Split | analysis.qualification; Python API/capture wrapper | 3–6 |
| `evaluator_qualification_cli.py` | Python | integration / facade / presentation | retained |
| `evidence_explanation.py` | Python | integration / facade / presentation | retained |
| `evidence_pack.py` | Rust with Python adapters | evidence.native | 2–5 |
| `evidence_pack_contract.py` | Rust with Python adapters | contracts.native + analysis.native | 2–5 |
| `evidence_pack_integrity.py` | Rust with Python adapters | evidence.crypto | 2–5 |
| `evidence_pack_json.py` | Rust with Python adapters | contracts.json | 2–5 |
| `evidence_pack_publication.py` | Rust with Python adapters | evidence.native | 2–5 |
| `evidence_pack_snapshot.py` | Rust with Python adapters | filesystem.snapshot | 2–5 |
| `evidence_pack_support.py` | Rust with Python adapters | evidence.errors | 2–5 |
| `evidence_pack_verification.py` | Rust with Python adapters | evidence.native | 2–5 |
| `evidence_receipt.py` | Rust with Python adapters | evidence.receipt | 2–5 |
| `evidence_reporting.py` | Python | integration / facade / presentation | retained |
| `evidence_sets/__init__.py` | Python | integration / facade / presentation | retained |
| `evidence_sets/cli.py` | Python | integration / facade / presentation | retained |
| `evidence_sets/contracts.py` | Rust with Python adapters | evidence.evidence_set | 5 |
| `evidence_sets/reporting.py` | Python | integration / facade / presentation | retained |
| `evidence_sets/verification.py` | Rust with Python adapters | evidence.evidence_set | 5 |
| `evidence_verification.py` | Split | evidence facade; Python API/capture wrapper | 3–6 |
| `exact_match_sensitivity.py` | Python | integration / facade / presentation | retained |
| `filesystem/__init__.py` | Rust with Python adapters | filesystem | 3–4 |
| `filesystem/atomic_directory.py` | Rust with Python adapters | filesystem | 3–4 |
| `filesystem/atomic_file.py` | Rust with Python adapters | filesystem | 3–4 |
| `filesystem/paths.py` | Rust with Python adapters | filesystem | 3–4 |
| `filesystem/staged_directory.py` | Rust with Python adapters | filesystem | 3–4 |
| `judge_evidence_types.py` | Rust with Python adapters | contracts.judge | 2–5 |
| `judge_measurement_types.py` | Rust with Python adapters | contracts.judge | 2–5 |
| `judge_measurements/__init__.py` | Python | integration / facade / presentation | retained |
| `judge_measurements/acceptance.py` | Split | evidence.judge; Python workflow/value adapters | 5 |
| `judge_measurements/analysis.py` | Split | analysis.judge; Python workflow/value adapters | 5 |
| `judge_measurements/captured_workflow.py` | Python | integration / facade / presentation | retained |
| `judge_measurements/cli_verification.py` | Python | integration / facade / presentation | retained |
| `judge_measurements/collector.py` | Python | integration / facade / presentation | retained |
| `judge_measurements/configured.py` | Python | integration / facade / presentation | retained |
| `judge_measurements/contracts.py` | Split | contracts.judge; Python workflow/value adapters | 5 |
| `judge_measurements/evidence.py` | Split | evidence.judge; Python workflow/value adapters | 5 |
| `judge_measurements/native_capture.py` | Split | contracts.judge; Python workflow/value adapters | 5 |
| `judge_measurements/native_recipe.py` | Split | contracts.judge; Python workflow/value adapters | 5 |
| `judge_measurements/native_workflow.py` | Python | integration / facade / presentation | retained |
| `judge_measurements/openai_compatible.py` | Python | integration / facade / presentation | retained |
| `judge_measurements/openai_compatible_contract.py` | Split | contracts.judge; Python workflow/value adapters | 5 |
| `judge_measurements/precision.py` | Split | analysis.precision; Python workflow/value adapters | 5 |
| `judge_measurements/reporting.py` | Python | integration / facade / presentation | retained |
| `judge_measurements/runner.py` | Python | integration / facade / presentation | retained |
| `judge_measurements/runtime_provider.py` | Python | integration / facade / presentation | retained |
| `judge_measurements/source_import.py` | Split | contracts.judge; Python workflow/value adapters | 5 |
| `judge_measurements/statistics.py` | Split | analysis.judge; Python workflow/value adapters | 5 |
| `judge_measurements/workflow.py` | Python | integration / facade / presentation | retained |
| `output_text_contract.py` | Rust with Python adapters | contracts.output_text | 2–5 |
| `paired_exact_match.py` | Rust with Python adapters | analysis.exact_match | 2–5 |
| `public_contracts.py` | Rust with Python adapters | contracts.schemas | 2–5 |
| `record_reporting.py` | Python | integration / facade / presentation | retained |
| `report_presentation.py` | Python | integration / facade / presentation | retained |
| `report_publication.py` | Python | integration / facade / presentation | retained |
| `runtime_behavior/__init__.py` | Python | integration / facade / presentation | retained |
| `runtime_behavior/transaction.py` | Python | integration / facade / presentation | retained |
| `runtime_import_authoring.py` | Rust with Python adapters | evidence.runtime | 2–5 |
| `runtime_manifest.py` | Rust with Python adapters | contracts.runtime | 2–5 |
| `runtime_provider_evidence.py` | Rust with Python adapters | evidence.runtime | 2–5 |
| `runtime_providers/__init__.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/_hf_safetensors_identity.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/_hf_transformers_identity.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/_tensorrt_llm_execution.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/_tensorrt_llm_inspection.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/gguf_identity.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/hf_transformers.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/hf_vision_text.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/hf_vision_text_conformance.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/llama_cpp.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/llama_cpp_conformance.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/llama_cpp_session.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/tensorrt_llm.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/tensorrt_llm_canary.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/tensorrt_llm_conformance.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/tensorrt_llm_identity.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/tensorrt_llm_runner.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/tensorrt_llm_session.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_security.py` | Python | integration / facade / presentation | retained |
| `runtime_security_helpers.py` | Python | integration / facade / presentation | retained |
| `runtime_verify.py` | Rust with Python adapters | evidence.runtime | 2–5 |
| `security.py` | Python | integration / facade / presentation | retained |
| `trust_inputs.py` | Rust with Python adapters | evidence.trust | 2–5 |
