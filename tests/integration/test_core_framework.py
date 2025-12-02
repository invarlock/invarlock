"""
Core Framework Integration Tests
===============================

Tests for the core InvarLock framework functionality including:
- Plugin discovery and registration
- Event logging system
- Checkpoint management
- Guard orchestration
- End-to-end workflows
"""

import io
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from invarlock.core.api import GuardChain
from invarlock.core.checkpoint import CheckpointManager
from invarlock.core.events import EventLogger
from invarlock.core.registry import get_registry
from invarlock.core.types import GuardOutcome, LogLevel
from invarlock.guards.invariants import InvariantsGuard, check_all_invariants


class TestCoreFramework:
    """Test core framework functionality."""

    def test_plugin_discovery_system(self):
        """Test that the plugin discovery system works."""
        registry = get_registry()
        edits = registry.list_edits()
        assert "quant_rtn" in edits
        guards = registry.list_guards()
        assert "hello_guard" in guards

        quant_info = registry.get_plugin_info("quant_rtn", "edits")
        assert quant_info["available"] is True

        hello_guard = registry.get_plugin_info("hello_guard", "guards")
        assert hello_guard["available"] is True

        quant_edit = registry.get_edit("quant_rtn")
        assert quant_edit.name == "quant_rtn"

    def test_event_logging_system(self):
        """Test the event logging system."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "test_events.jsonl"

            with EventLogger(log_path) as logger:
                # Test basic logging
                logger.log(
                    "test_component", "test_operation", LogLevel.INFO, {"test": "data"}
                )

                # Test error logging
                try:
                    raise ValueError("Test error")
                except ValueError as e:
                    logger.log_error("test_component", "error_operation", e)

                # Test metric logging
                logger.log_metric("test_component", "test_metric", 42.0, "seconds")

                # Test checkpoint logging
                logger.log_checkpoint("test_component", "checkpoint_123", "create")

            # Verify log file was created and has content
            assert log_path.exists()
            assert log_path.stat().st_size > 0

            # Verify log content
            lines = log_path.read_text().strip().split("\n")
            assert len(lines) >= 4  # At least 4 events logged

    def test_event_logger_redaction_and_run_id(self):
        """Event logger should redact sensitive keys and include the run ID."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "redaction.jsonl"
            run_id = "unit-run-42"

            with EventLogger(log_path, run_id=run_id) as logger:
                logger.log(
                    "guard",
                    "start",
                    LogLevel.INFO,
                    {
                        "api_token": "super-secret",
                        "nested": {"password": "should-not-leak"},
                        "values": [{"secret": "value"}],
                        "message": "ok",
                    },
                )

            entries = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").strip().splitlines()
            ]
            payload = next(entry for entry in entries if entry["operation"] == "start")

            assert payload["run_id"] == run_id
            data = payload["data"]
            assert data["api_token"] == "<redacted>"
            assert data["nested"]["password"] == "<redacted>"
            assert data["values"][0]["secret"] == "<redacted>"

    def test_checkpoint_management(self):
        """Test checkpoint management functionality."""
        # Create a simple test model
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))

        with tempfile.TemporaryDirectory() as tmp_dir:
            Path(tmp_dir)
            manager = CheckpointManager()

            # Create mock adapter
            mock_adapter = Mock()
            mock_adapter.snapshot = Mock(return_value=b"mock_checkpoint_data")

            # Test checkpoint creation
            checkpoint_id = manager.create_checkpoint(model, mock_adapter)
            assert checkpoint_id is not None
            assert len(checkpoint_id) > 0

            # Test checkpoint listing - check internal storage
            assert len(manager.checkpoints) == 1
            assert checkpoint_id in manager.checkpoints

            # Modify model
            model[0].weight.clone()
            model[0].weight.data.fill_(999.0)

            # Test checkpoint restoration
            mock_adapter.restore = Mock()
            success = manager.restore_checkpoint(model, mock_adapter, checkpoint_id)
        assert success

        # Verify restore was called
        mock_adapter.restore.assert_called_once()

    def test_checkpoint_management_chunked(self, monkeypatch, tmp_path):
        """Checkpoint manager should support chunked snapshot mode."""

        class ChunkedAdapter:
            def snapshot(self, model):
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                return buffer.getvalue()

            def restore(self, model, blob):
                state = torch.load(io.BytesIO(blob), map_location="cpu")
                model.load_state_dict(state)

            def snapshot_chunked(self, model, prefix: str = "invarlock-snap-") -> str:
                path = Path(tempfile.mkdtemp(prefix=prefix))
                os.chmod(path, 0o700)
                torch.save(model.state_dict(), path / "state.pt")
                return str(path)

            def restore_chunked(self, model, path: str) -> None:
                state = torch.load(Path(path) / "state.pt", map_location="cpu")
                model.load_state_dict(state)

        monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "chunked")
        adapter = ChunkedAdapter()
        manager = CheckpointManager()
        model = nn.Linear(4, 4)

        checkpoint_id = manager.create_checkpoint(model, adapter)
        snapshot_meta = manager.checkpoints[checkpoint_id]
        assert snapshot_meta["mode"] == "chunked"
        snapshot_path = snapshot_meta["path"]
        assert os.path.isdir(snapshot_path)

        with torch.no_grad():
            model.weight.add_(1.0)

        restored = manager.restore_checkpoint(model, adapter, checkpoint_id)
        assert restored

        manager.cleanup()
        assert not os.path.exists(snapshot_path)

    def test_guard_system(self):
        """Test the guard system functionality."""
        # Create a simple test model
        model = nn.Linear(5, 3)

        # Test individual guard
        guard = InvariantsGuard()

        # Test guard preparation
        mock_adapter = Mock()
        result = guard.prepare(model, mock_adapter, None, {})
        assert result["ready"] is True

        # Test guard execution
        outcome = guard.finalize(model)
        assert isinstance(outcome, GuardOutcome)
        assert outcome.passed is True  # Model should pass basic invariants

        # Test guard chain
        chain = GuardChain([guard])

        # Execute guard chain using finalize_all
        chain_result = chain.finalize_all(model)
        assert isinstance(chain_result, list)
        assert len(chain_result) == 1
        assert chain_result[0].passed is True

    def test_invariants_checking(self):
        """Test the invariants checking system."""
        # Test with valid model
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

        outcome = check_all_invariants(model)
        assert isinstance(outcome, GuardOutcome)
        assert outcome.passed is True
        assert len(outcome.violations) == 0

        # Test with model containing NaN
        model[0].weight.data[0, 0] = float("nan")
        outcome = check_all_invariants(model)
        assert outcome.passed is False
        assert len(outcome.violations) > 0
        assert any("nan" in v.get("message", "").lower() for v in outcome.violations)


class TestFrameworkIntegration:
    """Test end-to-end framework integration."""

    def test_complete_workflow(self):
        """Test a complete workflow with all components."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Setup
            log_path = Path(tmp_dir) / "workflow.jsonl"
            checkpoint_dir = Path(tmp_dir) / "checkpoints"
            checkpoint_dir.mkdir()

            model = nn.Linear(4, 2)

            # Create components
            logger = EventLogger(log_path, auto_flush=True)
            checkpoint_manager = CheckpointManager()
            guard = InvariantsGuard()

            # Create mock adapter
            mock_adapter = Mock()
            mock_adapter.snapshot = Mock(return_value=b"mock_checkpoint_data")

            try:
                # Step 1: Log workflow start
                logger.log("workflow", "start", LogLevel.INFO, {"model_type": "Linear"})

                # Step 2: Create checkpoint
                checkpoint_id = checkpoint_manager.create_checkpoint(
                    model, mock_adapter
                )
                logger.log_checkpoint("workflow", checkpoint_id, "create")

                # Step 3: Prepare guards
                mock_adapter = Mock()
                guard_result = guard.prepare(model, mock_adapter, None, {})
                logger.log("guards", "prepare", LogLevel.INFO, guard_result)

                # Step 4: Run guards
                guard_outcome = guard.finalize(model)
                logger.log(
                    "guards",
                    "execute",
                    LogLevel.INFO,
                    {
                        "passed": guard_outcome.passed,
                        "violations": len(guard_outcome.violations),
                    },
                )

                # Step 5: Log completion
                logger.log("workflow", "complete", LogLevel.INFO, {"status": "success"})

                # Verify workflow completed successfully
                assert guard_outcome.passed is True
                assert checkpoint_id is not None

            finally:
                logger.close()

            # Verify log file contains all expected events
            log_content = log_path.read_text()
            assert "workflow" in log_content
            assert "start" in log_content
            assert "complete" in log_content
            assert checkpoint_id in log_content

    def test_error_handling_workflow(self):
        """Test workflow with error conditions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "error_workflow.jsonl"

            # Create a model with invalid state
            model = nn.Linear(3, 2)
            model.weight.data.fill_(float("inf"))  # Invalid weights

            logger = EventLogger(log_path, auto_flush=True)

            try:
                # Test error logging
                logger.log("test", "start", LogLevel.INFO)

                # Check invariants (should fail)
                outcome = check_all_invariants(model)

                if not outcome.passed:
                    logger.log(
                        "test",
                        "invariants_failed",
                        LogLevel.ERROR,
                        {"violations": len(outcome.violations)},
                    )

                logger.log("test", "complete", LogLevel.INFO)

                # Verify error was detected
                assert not outcome.passed
                assert len(outcome.violations) > 0

            finally:
                logger.close()

            # Verify error was logged
            log_content = log_path.read_text()
            assert "invariants_failed" in log_content

    def test_plugin_integration(self):
        """Test plugin system integration."""
        # Test that plugins can be discovered and instantiated
        registry = get_registry()

        info = registry.get_plugin_info("quant_rtn", "edits")
        assert info["available"] is True
        assert "module" in info
        assert "version" in info

        quant_edit = registry.get_edit("quant_rtn")
        assert quant_edit.name == "quant_rtn"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
