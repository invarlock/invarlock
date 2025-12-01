"""Adapter-lite provenance extraction utilities.

Provides a tiny, versioned schema describing the adapter family and the
underlying library versions. This does not perform any edits; it only reads
environment and import metadata to annotate reports/certificates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version as pkg_version
from typing import Any


@dataclass
class AdapterProvenance:
    family: str
    library: str
    version: str | None
    supported: bool
    tested: list[str]
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FAMILY_MAP: dict[str, tuple[str, str, list[str]]] = {
    # name -> (family, library, tested_versions)
    "hf_gptq": ("gptq", "auto-gptq", []),
    "hf_awq": ("awq", "autoawq", []),
    "hf_bnb": ("bnb", "bitsandbytes", []),
    # ONNX stack (requires extras: invarlock[onnx])
    "hf_onnx": ("onnx", "onnxruntime", []),
}


def extract_adapter_provenance(adapter_name: str) -> AdapterProvenance:
    name = (adapter_name or "").strip().lower()
    family, library, tested = _FAMILY_MAP.get(name, ("hf", "transformers", []))

    ver: str | None
    try:
        ver = pkg_version(library)
        supported = True if (not tested or ver in tested) else False
        msg = (
            None
            if supported
            else f"Use Compare & Certify (BYOE); {library} version unsupported (tested: {tested})"
        )
    except Exception:  # Package not installed or version unknown
        ver = None
        supported = False
        msg = f"{library} not available; prefer Compare & Certify (BYOE) or install extras."

    return AdapterProvenance(
        family=family,
        library=library,
        version=ver,
        supported=supported,
        tested=tested,
        message=msg,
    )


__all__ = ["extract_adapter_provenance", "AdapterProvenance"]
