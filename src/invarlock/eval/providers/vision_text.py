from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from invarlock.core.exceptions import DataError as _DataErr

from .base import EvaluationProvider
from .local_jsonl_shared import resolve_local_jsonl_files


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_answers(obj: dict[str, Any]) -> list[str]:
    answers = obj.get("answers")
    if isinstance(answers, list):
        values = [str(answer).strip() for answer in answers if str(answer).strip()]
        if values:
            return values
    answer = obj.get("answer")
    if isinstance(answer, str) and answer.strip():
        return [answer.strip()]
    raise _DataErr(
        code="E306",
        message="NO-SAMPLES: vision_text record is missing answer/answers",
    )


def _resolve_image_path(image_path: str, *, base_dir: Path) -> Path:
    candidate = Path(image_path).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    if not candidate.exists() or not candidate.is_file():
        raise _DataErr(
            code="E306",
            message=f"NO-SAMPLES: vision_text image file is missing ({candidate})",
        )
    return candidate


class VisionTextProvider(EvaluationProvider):
    name = "vision_text"

    def __init__(
        self,
        *,
        file: str | None = None,
        path: str | None = None,
        data_files: str | list[str] | None = None,
        max_samples: int = 0,
        items: list[dict[str, Any]] | None = None,
        transform_pipeline: str = "",
        seed: int | None = None,
    ) -> None:
        self.file = file
        self.path = path
        self.data_files = data_files
        self.max_samples = int(max_samples or 0)
        self._transform_pipeline = str(transform_pipeline or "")
        self._seed = int(seed) if seed is not None else None
        self._items_override = list(items or [])
        self._examples_cache: list[dict[str, Any]] | None = None

    def available_splits(self) -> list[str]:
        return ["validation"]

    def _resolve_files(self) -> list[Path]:
        if self._items_override:
            return []
        return resolve_local_jsonl_files(
            file=self.file,
            path=self.path,
            data_files=self.data_files,
        )

    def _load_examples(self) -> list[dict[str, Any]]:
        if self._examples_cache is not None:
            return list(self._examples_cache)

        examples: list[dict[str, Any]] = []
        if self._items_override:
            for index, raw in enumerate(self._items_override, start=1):
                if not isinstance(raw, dict):
                    continue
                rec_id = str(raw.get("id") or f"memory:{index}")
                prompt = str(raw.get("prompt") or "")
                answers = _normalize_answers(raw)
                image_bytes = raw.get("image_bytes")
                if isinstance(image_bytes, bytearray):
                    image_bytes = bytes(image_bytes)
                if not isinstance(image_bytes, bytes):
                    image_bytes = b""
                examples.append(
                    {
                        "id": rec_id,
                        "image_path": str(raw.get("image_path") or ""),
                        "prompt": prompt,
                        "answer": answers[0],
                        "answers": answers,
                        "image_sha256": _sha256_hex(image_bytes),
                        "prompt_sha256": _sha256_hex(prompt.encode("utf-8")),
                        "answer_sha256": _sha256_hex(
                            json.dumps(answers, ensure_ascii=True).encode("utf-8")
                        ),
                    }
                )
            self._examples_cache = examples
            return list(examples)

        files = self._resolve_files()
        if not files:
            raise _DataErr(
                code="E306",
                message=(
                    "NO-SAMPLES: vision_text produced no samples; check "
                    "file/path/data_files"
                ),
            )

        for file_path in files:
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError) as exc:
                raise _DataErr(
                    code="E306",
                    message=f"NO-SAMPLES: failed to read vision_text manifest ({exc})",
                ) from exc
            for line_no, raw_line in enumerate(lines, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise _DataErr(
                        code="E306",
                        message=(
                            "NO-SAMPLES: failed to parse vision_text manifest "
                            f"{file_path}:{line_no} ({exc})"
                        ),
                    ) from exc
                if not isinstance(obj, dict):
                    continue
                prompt = obj.get("prompt")
                image_path = obj.get("image_path")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise _DataErr(
                        code="E306",
                        message=(
                            "NO-SAMPLES: vision_text record is missing prompt "
                            f"({file_path}:{line_no})"
                        ),
                    )
                if not isinstance(image_path, str) or not image_path.strip():
                    raise _DataErr(
                        code="E306",
                        message=(
                            "NO-SAMPLES: vision_text record is missing image_path "
                            f"({file_path}:{line_no})"
                        ),
                    )
                answers = _normalize_answers(obj)
                resolved_image = _resolve_image_path(
                    image_path,
                    base_dir=file_path.parent,
                )
                image_bytes = resolved_image.read_bytes()
                rec_id = str(
                    obj.get("id") or f"{file_path.name}:{line_no}:{resolved_image.name}"
                )
                examples.append(
                    {
                        "id": rec_id,
                        "image_path": str(resolved_image),
                        "prompt": prompt.strip(),
                        "answer": answers[0],
                        "answers": answers,
                        "image_sha256": _sha256_hex(image_bytes),
                        "prompt_sha256": _sha256_hex(prompt.strip().encode("utf-8")),
                        "answer_sha256": _sha256_hex(
                            json.dumps(answers, ensure_ascii=True).encode("utf-8")
                        ),
                        "source_file": str(file_path),
                        "source_line": line_no,
                    }
                )
                if self.max_samples > 0 and len(examples) >= self.max_samples:
                    self._examples_cache = examples
                    return list(examples)

        if not examples:
            raise _DataErr(
                code="E306",
                message=(
                    "NO-SAMPLES: vision_text produced no samples; check "
                    "file/path/data_files"
                ),
            )

        self._examples_cache = examples
        return list(examples)

    def examples(
        self, split: str = "validation", **kwargs: Any
    ) -> list[dict[str, Any]]:
        del split, kwargs
        return self._load_examples()

    def pairing_schedule(self) -> list[str]:
        return sorted(str(item["id"]) for item in self._load_examples())

    def digest(self) -> dict[str, Any]:
        examples = self._load_examples()
        by_id = sorted(examples, key=lambda item: str(item["id"]))
        ids_sha256 = _sha256_hex("".join(item["id"] for item in by_id).encode("utf-8"))
        images_sha256 = _sha256_hex(
            "".join(item["image_sha256"] for item in by_id).encode("utf-8")
        )
        prompts_sha256 = _sha256_hex(
            "".join(item["prompt_sha256"] for item in by_id).encode("utf-8")
        )
        answers_sha256 = _sha256_hex(
            "".join(item["answer_sha256"] for item in by_id).encode("utf-8")
        )
        digest: dict[str, Any] = {
            "provider": "vision_text",
            "version": 2,
            "ids_sha256": ids_sha256,
            "images_sha256": images_sha256,
            "prompts_sha256": prompts_sha256,
            "answers_sha256": answers_sha256,
            "transform_pipeline": self._transform_pipeline,
        }
        if self._seed is not None:
            digest["seed"] = int(self._seed)
        return digest

    def batches(self, *, seed: int, batch_size: int) -> Iterable[dict[str, Any]]:
        del seed
        size = max(int(batch_size or 1), 1)
        examples = self._load_examples()
        for index in range(0, len(examples), size):
            chunk = examples[index : index + size]
            if len(chunk) == 1:
                yield dict(chunk[0])
            else:
                yield {"records": [dict(item) for item in chunk]}


__all__ = ["VisionTextProvider"]
