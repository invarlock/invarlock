from __future__ import annotations

import sys
import types
from importlib import import_module


def install_transformers_tokenizer_stub() -> None:
    """Install a tiny transformers tokenizer stub for import-only CLI tests."""
    try:
        import_module("transformers")
        return
    except (ImportError, ModuleNotFoundError):
        pass

    if "transformers" not in sys.modules:
        tr = types.ModuleType("transformers")

        class _Tok:
            pad_token = "<pad>"
            eos_token = "<eos>"

            def get_vocab(self) -> dict[str, int]:
                return {"<pad>": 0, "<eos>": 1}

        class _Auto:
            @staticmethod
            def from_pretrained(*_args: object, **_kwargs: object) -> _Tok:
                return _Tok()

        class _GPT2(_Auto):
            pass

        tr.AutoTokenizer = _Auto
        tr.GPT2Tokenizer = _GPT2
        sys.modules["transformers"] = tr

    if "transformers.tokenization_utils_base" not in sys.modules:
        sub = types.ModuleType("transformers.tokenization_utils_base")
        sub.PreTrainedTokenizerBase = object
        sys.modules["transformers.tokenization_utils_base"] = sub
