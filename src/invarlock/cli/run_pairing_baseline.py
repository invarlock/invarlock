"""Baseline schedule validation helpers for run command pairing flows."""

from __future__ import annotations

import hashlib
from array import array
from typing import Any

import typer
from rich.console import Console

_PAIRING_INT_ERRORS = (OverflowError, TypeError, ValueError)
_PAIRING_ASSIGNMENT_ERRORS = (KeyError, TypeError, ValueError)


class _BaselineScheduleValidator:
    def __init__(
        self,
        cfg: Any,
        pairing_schedule: dict[str, Any],
        baseline_report_data: dict[str, Any] | None,
        *,
        tokenizer_hash: str | None,
        resolved_loss_type: str,
        profile: str | None,
        baseline_path_str: str | None,
        console: Console | None,
        event_fn: Any | None,
        typed_failures: bool,
        canonical_dataset_id_fn: Any,
        tensor_or_list_to_ints_fn: Any,
        hash_sequences_fn: Any,
        invarlock_error_cls: Any,
    ) -> None:
        self.cfg = cfg
        self.pairing_schedule = pairing_schedule
        self.baseline_report_data = baseline_report_data
        self.tokenizer_hash = tokenizer_hash
        self.resolved_loss_type = resolved_loss_type
        self.profile = profile
        self.baseline_path_str = baseline_path_str
        self.console = console
        self.event_fn = event_fn
        self.typed_failures = typed_failures
        self.canonical_dataset_id_fn = canonical_dataset_id_fn
        self.tensor_or_list_to_ints_fn = tensor_or_list_to_ints_fn
        self.hash_sequences_fn = hash_sequences_fn
        self.invarlock_error_cls = invarlock_error_cls
        baseline_meta = (
            baseline_report_data.get("data")
            if isinstance(baseline_report_data, dict)
            else {}
        )
        self.baseline_meta = baseline_meta if isinstance(baseline_meta, dict) else {}

    def _profile_name(self) -> str:
        return (self.profile or "dev").strip().lower()

    def _emit(self, tag: str, message: str, emoji: str) -> None:
        if self.console is not None and self.event_fn is not None:
            self.event_fn(
                self.console,
                tag,
                message,
                emoji=emoji,
                profile=self.profile,
            )

    def _fail_schedule(self, reason: str) -> None:
        path = self.baseline_path_str or "baseline"
        prof = self._profile_name()
        message = f"PAIRING-EVIDENCE-MISSING: {path}: {reason}"
        shell_mode = self.console is not None and self.event_fn is not None
        if prof in {"ci", "release"} or self.typed_failures:
            raise self.invarlock_error_cls(code="E001", message=message)
        if not shell_mode:
            raise typer.Exit(1)
        self._emit(
            "FAIL",
            f"Baseline pairing schedule '{path}' is incompatible: {reason}",
            "❌",
        )
        raise typer.Exit(1)

    def _extract_meta(self, field: str, default: Any = None) -> Any:
        value = self.baseline_meta.get(field)
        return value if value is not None else default

    def _warn_or_fail_hash_mismatch(
        self,
        meta_key: str,
        expected_value: str,
        *,
        warn_message: str,
    ) -> None:
        baseline_value = self.baseline_meta.get(meta_key)
        if (
            isinstance(baseline_value, str)
            and baseline_value
            and baseline_value != expected_value
        ):
            prof = self._profile_name()
            if prof in {"ci", "release"}:
                self._fail_schedule(f"{meta_key} mismatch vs baseline report data")
            if self.console is not None and self.event_fn is not None:
                self.event_fn(
                    self.console,
                    "WARN",
                    warn_message,
                    emoji="⚠️",
                    profile=prof,
                )

    def _validate_dataset_identity(self) -> None:
        cfg_dataset = getattr(self.cfg.dataset, "provider", None)
        if cfg_dataset is None:
            cfg_dataset = getattr(self.cfg.dataset, "dataset", None)
        cfg_dataset = self.canonical_dataset_id_fn(cfg_dataset)
        baseline_dataset = self.canonical_dataset_id_fn(self._extract_meta("dataset"))
        if (
            baseline_dataset is not None
            and cfg_dataset is not None
            and baseline_dataset != cfg_dataset
        ):
            self._fail_schedule(
                f"dataset mismatch (baseline {baseline_dataset} vs config {cfg_dataset})"
            )

        cfg_split = getattr(self.cfg.dataset, "split", "validation")
        baseline_split = self._extract_meta("split")
        if (
            baseline_split is not None
            and cfg_split is not None
            and baseline_split != cfg_split
        ):
            self._fail_schedule(
                f"split mismatch (baseline {baseline_split} vs config {cfg_split})"
            )

    def _validate_text_geometry(self) -> None:
        cfg_seq_len = getattr(self.cfg.dataset, "seq_len", None)
        baseline_seq_len = self._extract_meta("seq_len")
        if (
            cfg_seq_len is not None
            and baseline_seq_len is not None
            and baseline_seq_len != cfg_seq_len
        ):
            self._fail_schedule(
                f"sequence length mismatch (baseline {baseline_seq_len} vs config {cfg_seq_len})"
            )

        cfg_stride = getattr(
            self.cfg.dataset,
            "stride",
            getattr(self.cfg.dataset, "seq_len", None),
        )
        baseline_stride = self._extract_meta("stride")
        if (
            baseline_stride is not None
            and cfg_stride is not None
            and baseline_stride != cfg_stride
        ):
            self._fail_schedule(
                f"stride mismatch (baseline {baseline_stride} vs config {cfg_stride})"
            )

    def _validate_tokenizer_hash(self) -> None:
        baseline_tokenizer_hash = self.baseline_meta.get("tokenizer_hash")
        if (
            baseline_tokenizer_hash
            and self.tokenizer_hash
            and baseline_tokenizer_hash != self.tokenizer_hash
        ):
            self._fail_schedule(
                "tokenizer hash mismatch between baseline and current configuration"
            )

    @staticmethod
    def _hash_strings(values: list[str]) -> str:
        return hashlib.blake2s(
            "||".join(values).encode("utf-8"),
            digest_size=16,
        ).hexdigest()

    @staticmethod
    def _hash_tokens(tokens: list[int]) -> bytes:
        if not tokens:
            return b""
        token_array = array("I", (int(token) & 0xFFFFFFFF for token in tokens))
        return hashlib.blake2b(token_array.tobytes(), digest_size=16).digest()

    @staticmethod
    def _hash_window_evidence(
        tokens: list[int], labels: list[int] | None = None
    ) -> bytes:
        if not tokens:
            return b""
        hasher = hashlib.blake2b(digest_size=16)
        token_array = array("I", (int(token) & 0xFFFFFFFF for token in tokens))
        hasher.update(token_array.tobytes())
        if labels is None:
            hasher.update(b"\x00")
            return hasher.digest()
        label_array = array("q", (int(label) for label in labels))
        hasher.update(b"\x01")
        hasher.update(label_array.tobytes())
        return hasher.digest()

    def _multimodal_arm_check(
        self,
        label: str,
        section: dict[str, Any],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        records_raw = section.get("records")
        records: list[dict[str, Any]] = []
        if isinstance(records_raw, list):
            for record in records_raw:
                if not isinstance(record, dict):
                    self._fail_schedule(f"{label} record is not an object")
                records.append(dict(record))

        example_ids_raw = section.get("example_ids")
        if isinstance(example_ids_raw, list) and example_ids_raw:
            example_ids = [str(value) for value in example_ids_raw]
        else:
            example_ids = [
                str(record.get("id") or record.get("example_id") or "")
                for record in records
            ]
        if not example_ids:
            self._fail_schedule(f"{label} missing example_ids")
        if records and len(records) != len(example_ids):
            self._fail_schedule(
                f"{label} coherence error: len(example_ids)={len(example_ids)} len(records)={len(records)}"
            )
        for idx, example_id in enumerate(example_ids):
            if not example_id:
                self._fail_schedule(
                    f"{label} example_ids contains empty id at index {idx}"
                )
        if records:
            for idx, record in enumerate(records):
                record_id = str(record.get("id") or record.get("example_id") or "")
                if record_id and record_id != example_ids[idx]:
                    self._fail_schedule(f"{label} record id mismatch at index {idx}")
        return example_ids, records

    def _validate_multimodal(
        self,
        preview: dict[str, Any],
        final: dict[str, Any],
    ) -> dict[str, Any]:
        preview_ids, _preview_records = self._multimodal_arm_check("preview", preview)
        final_ids, _final_records = self._multimodal_arm_check("final", final)

        if len(set(preview_ids)) != len(preview_ids):
            self._fail_schedule("duplicate example_ids detected in preview arm")
        if len(set(final_ids)) != len(final_ids):
            self._fail_schedule("duplicate example_ids detected in final arm")
        if set(preview_ids) & set(final_ids):
            self._fail_schedule("example_ids overlap between preview and final arms")

        preview_hash = self._hash_strings(preview_ids)
        final_hash = self._hash_strings(final_ids)
        dataset_hash = hashlib.blake2s(
            (preview_hash + final_hash).encode("utf-8"),
            digest_size=16,
        ).hexdigest()

        self._warn_or_fail_hash_mismatch(
            "preview_hash",
            preview_hash,
            warn_message="Baseline preview_hash mismatch; continuing in dev profile.",
        )
        self._warn_or_fail_hash_mismatch(
            "final_hash",
            final_hash,
            warn_message="Baseline final_hash mismatch; continuing in dev profile.",
        )
        self._warn_or_fail_hash_mismatch(
            "dataset_hash",
            dataset_hash,
            warn_message="Baseline dataset_hash mismatch; continuing in dev profile.",
        )

        self._validate_dataset_identity()

        baseline_prov = (
            self.baseline_report_data.get("provenance")
            if isinstance(self.baseline_report_data, dict)
            else {}
        )
        if not isinstance(baseline_prov, dict):
            baseline_prov = {}
        baseline_provider_digest = baseline_prov.get("provider_digest")
        if not isinstance(baseline_provider_digest, dict):
            baseline_provider_digest = {}

        effective_preview = len(preview_ids)
        effective_final = len(final_ids)
        dataset_meta = {
            key: self.baseline_meta.get(key)
            for key in (
                "dataset_hash",
                "preview_hash",
                "final_hash",
                "provider_kind",
                "provider_digest",
                "processor_sha256",
            )
            if self.baseline_meta.get(key) is not None
        }
        dataset_meta.setdefault("provider_kind", "vision_text")
        dataset_meta.setdefault("preview_hash", preview_hash)
        dataset_meta.setdefault("final_hash", final_hash)
        dataset_meta.setdefault("dataset_hash", dataset_hash)
        processor_sha = (
            preview.get("processor_sha256")
            or final.get("processor_sha256")
            or baseline_provider_digest.get("processor_sha256")
        )
        if isinstance(processor_sha, str) and processor_sha:
            dataset_meta["processor_sha256"] = processor_sha
        dataset_meta["loss_type"] = self.resolved_loss_type

        window_plan = self.baseline_meta.get("window_plan")
        if not isinstance(window_plan, dict):
            window_plan = {
                "profile": "vision_text",
                "requested_preview": effective_preview,
                "requested_final": effective_final,
                "actual_preview": effective_preview,
                "actual_final": effective_final,
                "coverage_ok": True,
            }

        return {
            "effective_preview": effective_preview,
            "effective_final": effective_final,
            "preview_count": effective_preview,
            "final_count": effective_final,
            "dataset_meta": dataset_meta,
            "window_plan": window_plan,
            "calibration_data": [],
        }

    def _arm_check(
        self,
        label: str,
        section: dict[str, Any],
    ) -> tuple[list[int], list[list[int]], list[list[int]] | None]:
        window_ids = section.get("window_ids")
        input_ids = section.get("input_ids")
        masks = section.get("attention_masks")
        if not isinstance(window_ids, list) or not isinstance(input_ids, list):
            self._fail_schedule(
                f"invalid {label} section: missing window_ids/input_ids"
            )
        if len(window_ids) != len(input_ids):
            self._fail_schedule(
                f"{label} coherence error: len(window_ids)={len(window_ids)} len(input_ids)={len(input_ids)}"
            )

        ids_int: list[int] = []
        seqs: list[list[int]] = []
        for idx, (wid, seq) in enumerate(zip(window_ids, input_ids, strict=False)):
            try:
                wid_int = int(wid)
            except _PAIRING_INT_ERRORS:
                self._fail_schedule(
                    f"{label} window_ids contains non-int at index {idx}"
                )
            ids_int.append(wid_int)
            seq_ints = self.tensor_or_list_to_ints_fn(seq)
            if not seq_ints:
                self._fail_schedule(f"{label} input_ids empty at index {idx}")
            seqs.append(seq_ints)

        masks_rows: list[list[int]] = []
        masks_missing = masks is None or masks == []
        if (
            isinstance(masks, list)
            and masks
            and len(seqs) == 1
            and not isinstance(masks[0], list)
        ):
            masks = [masks]

        if isinstance(masks, list) and masks:
            if len(masks) != len(seqs):
                self._fail_schedule(
                    f"{label} coherence error: len(attention_masks)={len(masks)} len(input_ids)={len(seqs)}"
                )
            for idx, (seq_ints, mask) in enumerate(zip(seqs, masks, strict=False)):
                if not isinstance(mask, list):
                    self._fail_schedule(
                        f"{label} attention_masks row is not a list at index {idx}"
                    )
                mask_ints = self.tensor_or_list_to_ints_fn(mask)
                if len(mask_ints) != len(seq_ints):
                    self._fail_schedule(
                        f"{label} attention_masks length mismatch at index {idx}"
                    )
                masks_rows.append(mask_ints)
        else:
            masks_missing = True
            masks_rows = [[1] * len(seq) for seq in seqs]

        if masks_missing:
            try:
                section["attention_masks"] = masks_rows
            except _PAIRING_ASSIGNMENT_ERRORS:
                pass

        labels = section.get("labels")
        label_rows: list[list[int]] | None = None
        if isinstance(labels, list) and labels:
            if len(labels) != len(seqs):
                self._fail_schedule(f"{label} labels length mismatch")
            label_rows = []
            for idx, row in enumerate(labels):
                row_ints = self.tensor_or_list_to_ints_fn(row)
                if len(row_ints) != len(seqs[idx]):
                    self._fail_schedule(
                        f"{label} labels length mismatch at index {idx}"
                    )
                label_rows.append(row_ints)

        for key in ("masked_token_counts", "actual_token_counts"):
            raw_counts = section.get(key)
            if raw_counts is not None and (
                not isinstance(raw_counts, list) or len(raw_counts) != len(seqs)
            ):
                self._fail_schedule(f"{label} {key} length mismatch")

        return ids_int, seqs, label_rows

    def _validate_text_hashes(
        self,
        preview_seqs: list[list[int]],
        final_seqs: list[list[int]],
        *,
        preview_labels: list[list[int]] | None = None,
        final_labels: list[list[int]] | None = None,
    ) -> None:
        use_preview_labels = preview_labels is not None and len(preview_labels) == len(
            preview_seqs
        )
        use_final_labels = final_labels is not None and len(final_labels) == len(
            final_seqs
        )
        preview_hashes = [
            self._hash_window_evidence(
                seq,
                preview_labels[idx] if use_preview_labels else None,
            )
            for idx, seq in enumerate(preview_seqs)
        ]
        final_hashes = [
            self._hash_window_evidence(
                seq,
                final_labels[idx] if use_final_labels else None,
            )
            for idx, seq in enumerate(final_seqs)
        ]
        if len(set(preview_hashes)) != len(preview_hashes):
            self._fail_schedule("duplicate token sequences detected in preview arm")
        if len(set(final_hashes)) != len(final_hashes):
            self._fail_schedule("duplicate token sequences detected in final arm")
        if set(preview_hashes) & set(final_hashes):
            self._fail_schedule("preview/final token sequence overlap detected")

        expected_preview_hash = self.hash_sequences_fn(preview_seqs)
        expected_final_hash = self.hash_sequences_fn(final_seqs)
        expected_dataset_hash = hashlib.blake2s(
            (expected_preview_hash + expected_final_hash).encode("utf-8"),
            digest_size=16,
        ).hexdigest()

        self._warn_or_fail_hash_mismatch(
            "preview_hash",
            expected_preview_hash,
            warn_message="Baseline preview_hash mismatch; continuing in dev profile.",
        )
        self._warn_or_fail_hash_mismatch(
            "final_hash",
            expected_final_hash,
            warn_message="Baseline final_hash mismatch; continuing in dev profile.",
        )
        self._warn_or_fail_hash_mismatch(
            "dataset_hash",
            expected_dataset_hash,
            warn_message="Baseline dataset_hash mismatch; continuing in dev profile.",
        )

    def _validate_text_schedule(
        self,
        preview: dict[str, Any],
        final: dict[str, Any],
    ) -> None:
        preview_ids, preview_seqs, preview_labels = self._arm_check("preview", preview)
        final_ids, final_seqs, final_labels = self._arm_check("final", final)

        if len(set(preview_ids)) != len(preview_ids):
            self._fail_schedule("duplicate window_ids detected in preview arm")
        if len(set(final_ids)) != len(final_ids):
            self._fail_schedule("duplicate window_ids detected in final arm")
        if set(preview_ids) & set(final_ids):
            self._fail_schedule("window_ids overlap between preview and final arms")

        self._validate_text_hashes(
            preview_seqs,
            final_seqs,
            preview_labels=preview_labels,
            final_labels=final_labels,
        )

    def _build_text_result(self) -> dict[str, Any]:
        baseline_preview = len(self.pairing_schedule["preview"].get("input_ids") or [])
        baseline_final = len(self.pairing_schedule["final"].get("input_ids") or [])
        cfg_preview = getattr(self.cfg.dataset, "preview_n", None)
        cfg_final = getattr(self.cfg.dataset, "final_n", None)
        if (
            cfg_preview is not None
            and baseline_preview is not None
            and baseline_preview != cfg_preview
        ) or (
            cfg_final is not None
            and baseline_final is not None
            and baseline_final != cfg_final
        ):
            self._emit(
                "WARN",
                (
                    "Adjusting evaluation window counts to match baseline schedule "
                    f"({baseline_preview}/{baseline_final})."
                ),
                "⚠️",
            )

        self._validate_text_geometry()
        self._validate_dataset_identity()
        self._validate_tokenizer_hash()

        effective_preview = int(baseline_preview)
        effective_final = int(baseline_final)
        dataset_meta = {
            key: self.baseline_meta.get(key)
            for key in (
                "tokenizer_hash",
                "tokenizer_name",
                "vocab_size",
                "bos_token",
                "eos_token",
                "pad_token",
                "add_prefix_space",
                "dataset_hash",
                "preview_hash",
                "final_hash",
                "preview_total_tokens",
                "final_total_tokens",
            )
            if self.baseline_meta.get(key) is not None
        }
        dataset_meta["loss_type"] = self.resolved_loss_type
        return {
            "effective_preview": effective_preview,
            "effective_final": effective_final,
            "preview_count": effective_preview,
            "final_count": effective_final,
            "dataset_meta": dataset_meta,
            "window_plan": self.baseline_meta.get("window_plan"),
            "calibration_data": [],
        }

    def validate(self) -> dict[str, Any]:
        try:
            preview = (
                self.pairing_schedule.get("preview")
                if isinstance(self.pairing_schedule, dict)
                else None
            )
            final = (
                self.pairing_schedule.get("final")
                if isinstance(self.pairing_schedule, dict)
                else None
            )
            if not isinstance(preview, dict) or not isinstance(final, dict):
                self._fail_schedule("missing preview/final evaluation_windows sections")

            multimodal_schedule = any(
                isinstance(section.get("example_ids"), list)
                or isinstance(section.get("records"), list)
                for section in (preview, final)
            )
            if multimodal_schedule:
                return self._validate_multimodal(preview, final)

            self._validate_text_schedule(preview, final)
        except self.invarlock_error_cls:
            raise
        except typer.Exit:
            raise
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self._fail_schedule(
                f"failed to validate baseline schedule integrity ({exc})"
            )
        return self._build_text_result()


def validate_and_harvest_baseline_schedule_impl(
    cfg: Any,
    pairing_schedule: dict[str, Any],
    baseline_report_data: dict[str, Any] | None,
    *,
    tokenizer_hash: str | None,
    resolved_loss_type: str,
    profile: str | None,
    baseline_path_str: str | None,
    console: Console | None,
    event_fn: Any | None,
    typed_failures: bool,
    canonical_dataset_id_fn: Any,
    tensor_or_list_to_ints_fn: Any,
    hash_sequences_fn: Any,
    invarlock_error_cls: Any,
) -> dict[str, Any]:
    """Validate baseline pairing compatibility and harvest dataset metadata."""
    validator = _BaselineScheduleValidator(
        cfg,
        pairing_schedule,
        baseline_report_data,
        tokenizer_hash=tokenizer_hash,
        resolved_loss_type=resolved_loss_type,
        profile=profile,
        baseline_path_str=baseline_path_str,
        console=console,
        event_fn=event_fn,
        typed_failures=typed_failures,
        canonical_dataset_id_fn=canonical_dataset_id_fn,
        tensor_or_list_to_ints_fn=tensor_or_list_to_ints_fn,
        hash_sequences_fn=hash_sequences_fn,
        invarlock_error_cls=invarlock_error_cls,
    )
    return validator.validate()
