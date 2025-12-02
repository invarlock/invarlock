"""
InvarLock Event Logger
==================

JSONL event logging for pipeline execution tracking.
Provides structured logging for analysis and debugging.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from .types import LogLevel

__all__ = ["EventLogger"]


class EventLogger:
    """
    JSONL event logger for InvarLock pipeline execution.

    Logs structured events to a JSONL file for analysis,
    debugging, and audit trails. Thread-safe and handles
    file rotation if needed.
    """

    SENSITIVE_KEYWORDS: Sequence[str] = (
        "token",
        "secret",
        "password",
        "passphrase",
        "api_key",
        "credential",
        "auth",
        "email",
    )

    def __init__(
        self,
        log_path: Path,
        auto_flush: bool = True,
        *,
        run_id: str | None = None,
        redact_keywords: Sequence[str] | None = None,
        max_string_length: int = 512,
    ):
        """
        Initialize event logger.

        Args:
            log_path: Path to JSONL log file
            auto_flush: Whether to flush after each write
            run_id: Optional run identifier to include in every log entry
            redact_keywords: Iterable of keywords whose values should be redacted
            max_string_length: Maximum length for logged strings before truncation
        """
        self.log_path = Path(log_path)
        self.auto_flush = auto_flush
        self._file: TextIO | None = None
        self._session_id = self._generate_session_id()
        self._run_id = run_id
        self._redact_keywords = tuple(
            keyword.lower() for keyword in (redact_keywords or self.SENSITIVE_KEYWORDS)
        )
        # Honor caller-provided limit; clamp to a small positive minimum
        self._max_string_length = max(1, int(max_string_length))

        # Ensure parent directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file for writing
        self._open_log_file()

        # Log session start
        session_start_payload = {
            "session_id": self._session_id,
            "log_path": str(self.log_path),
        }
        if self._run_id:
            session_start_payload["run_id"] = self._run_id
        self.log("logger", "session_start", LogLevel.INFO, session_start_payload)

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return f"session_{int(time.time())}"

    def _open_log_file(self) -> None:
        """Open the log file for writing."""
        try:
            self._file = open(self.log_path, "a", encoding="utf-8")
        except Exception as e:  # pragma: no cover - defensive guard
            raise OSError(f"Failed to open log file {self.log_path}: {e}") from e

    def log(
        self,
        component: str,
        operation: str,
        level: LogLevel,
        data: dict[str, Any] | None = None,
    ) -> None:
        """
        Log an event.

        Args:
            component: Component generating the event (e.g., "runner", "edit", "guard")
            operation: Operation being performed (e.g., "start", "complete", "error")
            level: Log level
            data: Optional additional data
        """
        if not self._file:
            return

        event: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self._session_id,
            "component": component,
            "operation": operation,
            "level": level.value,
        }

        if self._run_id:
            event["run_id"] = self._run_id

        if data:
            event["data"] = self._sanitize_data(data)

        try:
            json_line = json.dumps(event, default=self._json_serializer)
            self._file.write(json_line + "\n")

            if self.auto_flush:
                self._file.flush()

        except Exception as e:  # pragma: no cover - fallback to stderr
            import sys

            print(f"Event logging failed: {e}", file=sys.stderr)

    def _sanitize_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Sanitize data for JSON serialization.

        Removes non-serializable objects and large data structures,
        and redacts potential secrets.
        """

        def sanitize_value(key: str | None, value: Any) -> Any:
            key_lower = key.lower() if isinstance(key, str) else ""
            if key_lower and any(word in key_lower for word in self._redact_keywords):
                return "<redacted>"

            if isinstance(value, str):
                if any(word in value.lower() for word in self._redact_keywords):
                    return "<redacted>"
                if len(value) > self._max_string_length:
                    return f"<str len={len(value)}>"
                return value

            if isinstance(value, Mapping):
                return {
                    inner_key: sanitize_value(str(inner_key), inner_value)
                    for inner_key, inner_value in value.items()
                }

            # Common numeric array-like
            if hasattr(value, "tolist"):
                try:
                    return value.tolist()
                except Exception:
                    pass

            if isinstance(value, set | frozenset):
                try:
                    return list(value)
                except Exception:
                    pass

            if isinstance(value, Sequence) and not isinstance(
                value, str | bytes | bytearray
            ):
                return [sanitize_value(key, item) for item in value]

            if isinstance(value, bytes):
                return f"<bytes len={len(value)}>"
            # Preserve JSON-native scalars; coerce others to a placeholder string
            if value is None or isinstance(value, bool | int | float):
                return value
            return f"<{type(value).__name__}>"

        return {key: sanitize_value(key, value) for key, value in data.items()}

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for common non-serializable types."""
        if hasattr(obj, "tolist"):  # numpy arrays
            return obj.tolist()
        if hasattr(obj, "__dict__"):  # custom objects
            return str(obj)
        if isinstance(obj, set | frozenset):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        return str(obj)

    def log_error(
        self,
        component: str,
        operation: str,
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Convenience method for logging errors.

        Args:
            component: Component where error occurred
            operation: Operation that failed
            error: The exception that occurred
            context: Optional additional context
        """
        error_data: dict[str, Any] = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        if context:
            error_data["context"] = context

        self.log(component, operation, LogLevel.ERROR, error_data)

    def log_metric(
        self, component: str, metric_name: str, value: float, unit: str | None = None
    ) -> None:
        """
        Convenience method for logging metrics.

        Args:
            component: Component reporting the metric
            metric_name: Name of the metric
            value: Metric value
            unit: Optional unit (e.g., "seconds", "MB", "ratio")
        """
        metric_data = {"metric": metric_name, "value": value}

        if unit:
            metric_data["unit"] = unit

        self.log(component, "metric", LogLevel.INFO, metric_data)

    def log_checkpoint(
        self, component: str, checkpoint_id: str, operation: str
    ) -> None:
        """
        Convenience method for logging checkpoint operations.

        Args:
            component: Component handling the checkpoint
            checkpoint_id: Unique checkpoint identifier
            operation: Checkpoint operation ("create", "restore", "delete")
        """
        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "checkpoint_operation": operation,
        }

        self.log(component, "checkpoint", LogLevel.INFO, checkpoint_data)

    def close(self) -> None:
        """Close the log file."""
        if self._file:
            session_end_payload = {"session_id": self._session_id}
            if self._run_id:
                session_end_payload["run_id"] = self._run_id
            self.log("logger", "session_end", LogLevel.INFO, session_end_payload)

            try:
                self._file.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
            finally:
                self._file = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor - ensure file is closed."""
        if hasattr(self, "_file"):
            self.close()
