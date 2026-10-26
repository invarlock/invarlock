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

The inventory assigns responsibilities, not a requirement to preserve Python's
internal module structure in Rust. The complete evidence core is the experiment's
target; individual ports are checkpoints. The separate Rust execution/control
layer targets portable runtime coordination, resource accounting and native command
workflows. Python model/SDK execution remains behind explicit adapters. These are
target assignments, not implemented capabilities; runtime execution is never a
dependency of the offline verifier.

See the [architecture](rust-core-design.md) for boundaries and the
[comparison protocol](rust-comparison-protocol.md) for required evidence.
The machine-readable inventory is `experiments/rust/module-map.json`.

| Source module under `src/invarlock/` | Target ownership | Destination | Stage |
| --- | --- | --- | --- |
| `__init__.py` | Python | integration / facade / presentation | retained |
| `__main__.py` | Python | integration / facade / presentation | retained |
| `_bounded_subprocess.py` | Split | execution.process; Python compatibility adapter | 6 |
| `_optional_runtime_profiles.py` | Python | integration / facade / presentation | retained |
| `acceptance_attestation.py` | Rust with Python adapters | evidence.acceptance | 2–5 |
| `captured_contracts.py` | Rust with Python adapters | contracts.captured | 2–5 |
| `captured_evaluation.py` | Split | evidence.captured; Python API/capture wrapper | 3–6 |
| `captured_evidence_publication.py` | Rust with Python adapters | evidence.captured | 2–5 |
| `captured_normalization.py` | Rust with Python adapters | contracts.request | 2–5 |
| `captured_reporting.py` | Split | commands.reports; Python value/presentation adapters | 6 |
| `captured_verification.py` | Rust with Python adapters | evidence.captured | 2–5 |
| `cli/__init__.py` | Python | integration / facade / presentation | retained |
| `cli/app.py` | Split | commands; Python entry-point adapters | 6 |
| `cli/evaluation_setup.py` | Split | commands; Python entry-point adapters | 6 |
| `cli/evaluation_workflow.py` | Split | commands; Python entry-point adapters | 6 |
| `cli/runtime_profile.py` | Split | commands; Python entry-point adapters | 6 |
| `cli/verification_workflow.py` | Split | commands; Python entry-point adapters | 6 |
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
| `evaluation_oci.py` | Split | execution.container; Python API adapters | 6 |
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
| `evaluation_run.py` | Split | execution.transaction; Python SDK/model adapters | 6 |
| `evaluation_runtime.py` | Split | execution.resources; Python SDK/model adapters | 6 |
| `evaluation_side_worker.py` | Split | contracts.worker + execution.worker; Python model worker | 6 |
| `evaluation_transaction.py` | Split | execution.transaction + evidence; Python API adapters | 6 |
| `evaluator_capture.py` | Split | contracts.records; Python API/capture wrapper | 3–6 |
| `evaluator_qualification.py` | Split | analysis.qualification; Python API/capture wrapper | 3–6 |
| `evaluator_qualification_cli.py` | Split | commands.qualification; Python entry-point adapters | 6 |
| `evidence_explanation.py` | Split | commands.reports; Python value/presentation adapters | 6 |
| `evidence_pack.py` | Rust with Python adapters | evidence.native | 2–5 |
| `evidence_pack_contract.py` | Rust with Python adapters | contracts.native + analysis.native | 2–5 |
| `evidence_pack_integrity.py` | Rust with Python adapters | evidence.crypto | 2–5 |
| `evidence_pack_json.py` | Rust with Python adapters | contracts.json | 2–5 |
| `evidence_pack_publication.py` | Rust with Python adapters | evidence.native | 2–5 |
| `evidence_pack_snapshot.py` | Rust with Python adapters | filesystem.snapshot | 2–5 |
| `evidence_pack_support.py` | Rust with Python adapters | evidence.errors | 2–5 |
| `evidence_pack_verification.py` | Rust with Python adapters | evidence.native | 2–5 |
| `evidence_receipt.py` | Rust with Python adapters | evidence.receipt | 2–5 |
| `evidence_reporting.py` | Split | commands.reports; Python value/presentation adapters | 6 |
| `evidence_sets/__init__.py` | Python | integration / facade / presentation | retained |
| `evidence_sets/cli.py` | Split | commands.evidence_sets; Python entry-point adapters | 6 |
| `evidence_sets/contracts.py` | Rust with Python adapters | evidence.evidence_set | 5 |
| `evidence_sets/reporting.py` | Split | commands.reports; Python value/presentation adapters | 6 |
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
| `judge_measurements/captured_workflow.py` | Split | commands.judge + evidence.judge; Python API adapters | 6 |
| `judge_measurements/cli_verification.py` | Split | commands.judge; Python entry-point adapters | 6 |
| `judge_measurements/collector.py` | Split | execution.collection + checkpoints; Python provider SDK adapters | 6 |
| `judge_measurements/configured.py` | Split | execution.configuration + contracts.judge; Python SDK adapters | 6 |
| `judge_measurements/contracts.py` | Split | contracts.judge; Python workflow/value adapters | 5 |
| `judge_measurements/evidence.py` | Split | evidence.judge; Python workflow/value adapters | 5 |
| `judge_measurements/native_capture.py` | Split | contracts.judge; Python workflow/value adapters | 5 |
| `judge_measurements/native_recipe.py` | Split | contracts.judge; Python workflow/value adapters | 5 |
| `judge_measurements/native_workflow.py` | Split | execution.workflow + evidence.judge; Python API adapters | 6 |
| `judge_measurements/openai_compatible.py` | Python | integration / facade / presentation | retained |
| `judge_measurements/openai_compatible_contract.py` | Split | contracts.judge; Python workflow/value adapters | 5 |
| `judge_measurements/precision.py` | Split | analysis.precision; Python workflow/value adapters | 5 |
| `judge_measurements/reporting.py` | Split | commands.reports; Python value/presentation adapters | 6 |
| `judge_measurements/runner.py` | Split | execution.admission + budgets + retries; Python provider SDK adapters | 6 |
| `judge_measurements/runtime_provider.py` | Python | integration / facade / presentation | retained |
| `judge_measurements/source_import.py` | Split | contracts.judge; Python workflow/value adapters | 5 |
| `judge_measurements/statistics.py` | Split | analysis.judge; Python workflow/value adapters | 5 |
| `judge_measurements/workflow.py` | Split | execution.workflow + evidence.judge; Python API adapters | 6 |
| `output_text_contract.py` | Rust with Python adapters | contracts.output_text | 2–5 |
| `paired_exact_match.py` | Rust with Python adapters | analysis.exact_match | 2–5 |
| `public_contracts.py` | Rust with Python adapters | contracts.schemas | 2–5 |
| `record_reporting.py` | Split | commands.reports; Python value/presentation adapters | 6 |
| `report_presentation.py` | Split | commands.reports; Python presentation adapters | 6 |
| `report_publication.py` | Split | commands.reports + filesystem.publication; Python API adapters | 6 |
| `runtime_behavior/__init__.py` | Python | integration / facade / presentation | retained |
| `runtime_behavior/transaction.py` | Split | execution.transaction; Python model adapters | 6 |
| `runtime_import_authoring.py` | Rust with Python adapters | evidence.runtime | 2–5 |
| `runtime_manifest.py` | Rust with Python adapters | contracts.runtime | 2–5 |
| `runtime_provider_evidence.py` | Rust with Python adapters | evidence.runtime | 2–5 |
| `runtime_providers/__init__.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/_hf_safetensors_identity.py` | Split | contracts.identity + safe inspection; Python format/SDK adapters where required | 6 |
| `runtime_providers/_hf_transformers_identity.py` | Split | contracts.identity + safe inspection; Python format/SDK adapters where required | 6 |
| `runtime_providers/_tensorrt_llm_execution.py` | Split | execution.provider process/session control; Python worker/SDK adapters | 6 |
| `runtime_providers/_tensorrt_llm_inspection.py` | Split | contracts.identity + safe inspection; Python format/SDK adapters where required | 6 |
| `runtime_providers/gguf_identity.py` | Split | contracts.identity + safe inspection; Python format/SDK adapters where required | 6 |
| `runtime_providers/hf_transformers.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/hf_vision_text.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/hf_vision_text_conformance.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/llama_cpp.py` | Split | execution.provider process/session control; Python worker/SDK adapters | 6 |
| `runtime_providers/llama_cpp_conformance.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/llama_cpp_session.py` | Split | execution.provider process/session control; Python worker/SDK adapters | 6 |
| `runtime_providers/tensorrt_llm.py` | Split | execution.provider process/session control; Python worker/SDK adapters | 6 |
| `runtime_providers/tensorrt_llm_canary.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/tensorrt_llm_conformance.py` | Python | Python provider execution; selected portable identities reviewed separately | retained |
| `runtime_providers/tensorrt_llm_identity.py` | Split | contracts.identity + safe inspection; Python format/SDK adapters where required | 6 |
| `runtime_providers/tensorrt_llm_runner.py` | Split | execution.provider process/session control; Python worker/SDK adapters | 6 |
| `runtime_providers/tensorrt_llm_session.py` | Split | execution.provider process/session control; Python worker/SDK adapters | 6 |
| `runtime_security.py` | Split | execution.enforcement; retained Python execution guards | 6 |
| `runtime_security_helpers.py` | Split | execution.policy; retained Python execution guards | 6 |
| `runtime_verify.py` | Rust with Python adapters | evidence.runtime | 2–5 |
| `security.py` | Split | execution.enforcement; retained Python socket guards | 6 |
| `trust_inputs.py` | Rust with Python adapters | evidence.trust | 2–5 |
