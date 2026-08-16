"""Optional Hugging Face vision-text runtime for InvarLock."""

__version__ = "0.15.0"

from .provider import HFVisionTextProvider, processor_contract_sha256

__all__ = ["HFVisionTextProvider", "__version__", "processor_contract_sha256"]
