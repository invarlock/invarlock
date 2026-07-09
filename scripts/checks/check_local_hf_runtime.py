from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType


@dataclass(frozen=True)
class RuntimeImportFailure:
    module: str
    attr: str
    error: str


REQUIRED_IMPORTS = (
    ("tokenizers", "Tokenizer"),
    ("tokenizers.models", "WordLevel"),
    ("tokenizers.pre_tokenizers", "Whitespace"),
    ("transformers", "GPT2Config"),
    ("transformers", "GPT2LMHeadModel"),
    ("transformers", "PreTrainedTokenizerFast"),
)


def check_local_hf_runtime(
    import_module: Callable[[str], ModuleType] = importlib.import_module,
) -> list[RuntimeImportFailure]:
    failures: list[RuntimeImportFailure] = []
    for module_name, attr in REQUIRED_IMPORTS:
        try:
            module = import_module(module_name)
            getattr(module, attr)
        except Exception as exc:
            failures.append(
                RuntimeImportFailure(
                    module=module_name,
                    attr=attr,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return failures


def main() -> int:
    failures = check_local_hf_runtime()
    if not failures:
        print("Local HF runtime OK.")
        return 0

    print("Local HF runtime is unavailable or inconsistent.", file=sys.stderr)
    for failure in failures:
        print(
            f"- {failure.module}.{failure.attr}: {failure.error}",
            file=sys.stderr,
        )
    print("", file=sys.stderr)
    print("Refresh the workspace environment with:", file=sys.stderr)
    print("  make local-hf-env-refresh", file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "For hermetic smoke validation without changing the workspace env, run:",
        file=sys.stderr,
    )
    print("  make local-hf-pipeline-smoke-locked", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
