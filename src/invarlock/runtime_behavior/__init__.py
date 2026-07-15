"""Provider-neutral runtime behavioral execution and paired verification."""

from .contracts import (
    MAX_RUNTIME_BEHAVIORAL_SIDE_FILE_BYTES,
    RUNTIME_BEHAVIORAL_SIDE_CONFIG_FILENAME,
    RUNTIME_BEHAVIORAL_SIDE_CONFIG_FORMAT,
    RUNTIME_BEHAVIORAL_SIDE_REPORT_FILENAME,
    RUNTIME_BEHAVIORAL_SIDE_REPORT_FORMAT,
    RuntimeBehavioralRole,
    RuntimeBehaviorError,
    RuntimePairVerification,
    RuntimeSideBundle,
)
from .pair import verify_pair
from .side import run_side

__all__ = [
    "MAX_RUNTIME_BEHAVIORAL_SIDE_FILE_BYTES",
    "RUNTIME_BEHAVIORAL_SIDE_CONFIG_FILENAME",
    "RUNTIME_BEHAVIORAL_SIDE_CONFIG_FORMAT",
    "RUNTIME_BEHAVIORAL_SIDE_REPORT_FILENAME",
    "RUNTIME_BEHAVIORAL_SIDE_REPORT_FORMAT",
    "RuntimeBehavioralRole",
    "RuntimeBehaviorError",
    "RuntimePairVerification",
    "RuntimeSideBundle",
    "run_side",
    "verify_pair",
]
