"""
Core API Coverage Tests
======================

Comprehensive tests to achieve 80%+ coverage for src/invarlock/core/api.py
Focuses on missing coverage areas in GuardChain and data classes.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from invarlock.core.api import (
    CalibrationData,
    DeviceType,
    Guard,
    GuardChain,
    MetricsDict,
    ModelAdapter,
    ModelEdit,
    ModelType,
    RunConfig,
    RunReport,
)


class TestGuardChainComprehensive:
    """Comprehensive tests for GuardChain to improve coverage."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock guards with different interfaces
        self.full_guard = Mock(spec=Guard)
        self.full_guard.name = "full_guard"
        self.full_guard.prepare = Mock(
            return_value={"ready": True, "baseline": "captured"}
        )
        self.full_guard.before_edit = Mock(return_value={"before": "result"})
        self.full_guard.after_edit = Mock(return_value={"after": "result"})
        self.full_guard.finalize = Mock(return_value=Mock(passed=True, action="none"))

        # Minimal guard with only validate method
        self.minimal_guard = Mock(spec=Guard)
        self.minimal_guard.name = "minimal_guard"
        # No prepare, before_edit, after_edit, finalize methods

        # Guard that returns None from hooks
        self.none_guard = Mock(spec=Guard)
        self.none_guard.name = "none_guard"
        self.none_guard.prepare = Mock(return_value={"ready": False})
        self.none_guard.before_edit = Mock(return_value=None)  # Returns None
        self.none_guard.after_edit = Mock(return_value=None)  # Returns None
        self.none_guard.finalize = Mock(return_value=Mock(passed=False, action="warn"))

        self.model = Mock()
        self.adapter = Mock(spec=ModelAdapter)
        self.calib = Mock()

    def test_prepare_all_with_mixed_guards(self):
        """Test prepare_all with guards that have and don't have prepare method."""
        # Mix of guards with and without prepare method
        chain = GuardChain([self.full_guard, self.minimal_guard], policy={"test": True})

        policy_config = {"policy": "test"}
        results = chain.prepare_all(self.model, self.adapter, self.calib, policy_config)

        # Should call prepare on full_guard
        self.full_guard.prepare.assert_called_once_with(
            self.model, self.adapter, self.calib, policy_config
        )

        # Should not call prepare on minimal_guard (doesn't have it)
        assert (
            not hasattr(self.minimal_guard, "prepare")
            or not self.minimal_guard.prepare.called
        )

        # Results should include both guards
        assert "full_guard" in results
        assert "minimal_guard" in results
        assert results["full_guard"] == {"ready": True, "baseline": "captured"}
        assert results["minimal_guard"] == {
            "ready": True
        }  # Default for guards without prepare

    def test_before_edit_all_with_none_returns(self):
        """Test before_edit_all with guards that return None vs actual results."""
        chain = GuardChain([self.full_guard, self.none_guard, self.minimal_guard])

        results = chain.before_edit_all(self.model)

        # Should call before_edit on guards that have it
        self.full_guard.before_edit.assert_called_once_with(self.model)
        self.none_guard.before_edit.assert_called_once_with(self.model)

        # Should only include non-None results
        assert len(results) == 1  # Only full_guard returns non-None
        assert results[0] == {"before": "result"}

    def test_after_edit_all_with_none_returns(self):
        """Test after_edit_all with guards that return None vs actual results."""
        chain = GuardChain([self.full_guard, self.none_guard, self.minimal_guard])

        results = chain.after_edit_all(self.model)

        # Should call after_edit on guards that have it
        self.full_guard.after_edit.assert_called_once_with(self.model)
        self.none_guard.after_edit.assert_called_once_with(self.model)

        # Should only include non-None results
        assert len(results) == 1  # Only full_guard returns non-None
        assert results[0] == {"after": "result"}

    def test_finalize_all_comprehensive(self):
        """Test finalize_all with different guard types."""
        chain = GuardChain([self.full_guard, self.none_guard, self.minimal_guard])

        results = chain.finalize_all(self.model)

        # Should call finalize on guards that have it
        self.full_guard.finalize.assert_called_once_with(self.model)
        self.none_guard.finalize.assert_called_once_with(self.model)

        # Should include all results (even None is appended for guards without finalize)
        assert len(results) == 2  # Only guards with finalize method

    def test_all_passed_with_failing_outcomes(self):
        """Test all_passed method with outcomes that fail."""
        chain = GuardChain([])

        # Test with all passing outcomes
        passing_outcomes = [Mock(passed=True), Mock(passed=True), Mock(passed=True)]
        assert chain.all_passed(passing_outcomes)

        # Test with one failing outcome
        failing_outcomes = [
            Mock(passed=True),
            Mock(passed=False),  # This one fails
            Mock(passed=True),
        ]
        assert not chain.all_passed(failing_outcomes)

        # Test with outcomes without passed attribute
        mixed_outcomes = [
            Mock(passed=True),
            Mock(spec=["other_attr"]),  # No passed attribute
            Mock(passed=True),
        ]
        assert chain.all_passed(mixed_outcomes)  # Treats missing as not failing

        # Test with empty outcomes
        assert chain.all_passed([])

    def test_get_worst_action_comprehensive(self):
        """Test get_worst_action method with different action types."""
        chain = GuardChain([])

        # Test with various action combinations
        outcomes_with_actions = [
            Mock(action="none"),
            Mock(action="warn"),
            Mock(action="rollback"),
        ]

        # Should import and use get_worst_action from types
        result = chain.get_worst_action(outcomes_with_actions)
        assert isinstance(result, str)

        # Test with outcomes without action attribute
        outcomes_no_actions = [
            Mock(spec=["other_attr"]),  # No action attribute
            Mock(spec=["other_attr"]),
        ]
        result = chain.get_worst_action(outcomes_no_actions)
        assert isinstance(result, str)

        # Test with empty outcomes
        result = chain.get_worst_action([])
        assert isinstance(result, str)

    def test_guard_chain_initialization(self):
        """Test GuardChain initialization with different parameters."""
        # Test with guards and policy
        guards = [self.full_guard, self.minimal_guard]
        policy = {"test_policy": "value"}
        chain = GuardChain(guards, policy)

        assert chain.guards == guards
        assert chain.policy == policy

        # Test with guards only
        chain2 = GuardChain(guards)
        assert chain2.guards == guards
        assert chain2.policy is None

        # Test with empty guards
        chain3 = GuardChain([])
        assert chain3.guards == []
        assert chain3.policy is None


class TestDataClassesCoverage:
    """Test data classes to improve coverage."""

    def test_run_config_comprehensive(self):
        """Test RunConfig with all field variations."""
        # Test default initialization
        config1 = RunConfig()
        assert config1.device == "auto"
        assert config1.max_pm_ratio == 1.5
        assert config1.event_path is None
        assert config1.checkpoint_interval == 0
        assert not config1.dry_run
        assert not config1.verbose
        assert config1.context == {}

        # Test custom initialization
        event_path = Path("/tmp/events.log")
        custom_context = {"test": "value"}
        config2 = RunConfig(
            device="cuda:0",
            max_pm_ratio=2.0,
            event_path=event_path,
            checkpoint_interval=100,
            dry_run=True,
            verbose=True,
            context=custom_context,
        )

        assert config2.device == "cuda:0"
        assert config2.max_pm_ratio == 2.0
        assert config2.event_path == event_path
        assert config2.checkpoint_interval == 100
        assert config2.dry_run
        assert config2.verbose
        assert config2.context == custom_context

        # Test that context is properly isolated between instances
        config1.context["new"] = "value"
        assert "new" not in config2.context

    def test_run_report_comprehensive(self):
        """Test RunReport with all field variations."""
        # Test default initialization
        report1 = RunReport()
        assert report1.meta == {}
        assert report1.edit == {}
        assert report1.guards == {}
        assert report1.metrics == {}
        assert report1.status == "pending"
        assert report1.error is None
        assert report1.context == {}

        # Test custom initialization
        meta_data = {"timestamp": "2024-01-01", "version": "1.0"}
        edit_data = {"operation": "sparsify", "sparsity": 0.5}
        guards_data = {"spectral": {"passed": True}, "rmt": {"passed": False}}
        metrics_data = {"ppl": 3.2, "accuracy": 0.95}
        context_data = {"model": "gpt2", "dataset": "wikitext"}

        report2 = RunReport(
            meta=meta_data,
            edit=edit_data,
            guards=guards_data,
            metrics=metrics_data,
            status="success",
            error="test error",
            context=context_data,
        )

        assert report2.meta == meta_data
        assert report2.edit == edit_data
        assert report2.guards == guards_data
        assert report2.metrics == metrics_data
        assert report2.status == "success"
        assert report2.error == "test error"
        assert report2.context == context_data

        # Test that default factory creates isolated instances
        report1.meta["new"] = "value"
        assert "new" not in report2.meta

        # Test different status values
        for status in ["pending", "success", "failed", "rollback"]:
            report = RunReport(status=status)
            assert report.status == status


class TestAbstractInterfacesCoverage:
    """Test abstract interfaces for better coverage."""

    def test_model_adapter_interface(self):
        """Test ModelAdapter abstract interface."""
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            ModelAdapter()

        # Test that subclass must implement all abstract methods
        class IncompleteAdapter(ModelAdapter):
            name = "incomplete"

        with pytest.raises(TypeError):
            IncompleteAdapter()

        # Test complete implementation
        class CompleteAdapter(ModelAdapter):
            name = "complete"

            def can_handle(self, model):
                return True

            def describe(self, model):
                return {"n_layer": 12, "heads_per_layer": 12}

            def snapshot(self, model):
                return b"snapshot_data"

            def restore(self, model, blob):
                pass

        adapter = CompleteAdapter()
        assert adapter.name == "complete"
        assert adapter.can_handle(Mock())
        assert adapter.describe(Mock()) == {"n_layer": 12, "heads_per_layer": 12}
        assert adapter.snapshot(Mock()) == b"snapshot_data"
        adapter.restore(Mock(), b"data")  # Should not raise

    def test_model_edit_interface(self):
        """Test ModelEdit abstract interface."""
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            ModelEdit()

        # Test complete implementation
        class CompleteEdit(ModelEdit):
            name = "complete_edit"

            def can_edit(self, model_desc):
                return model_desc.get("n_layer", 0) > 0

            def apply(self, model, adapter, **kwargs):
                return {"applied": True, "changes": kwargs}

        edit = CompleteEdit()
        assert edit.name == "complete_edit"
        assert edit.can_edit({"n_layer": 12})
        assert not edit.can_edit({"n_layer": 0})

        result = edit.apply(Mock(), Mock(), param1="value1", param2="value2")
        assert result["applied"]
        assert result["changes"] == {"param1": "value1", "param2": "value2"}

    def test_guard_interface(self):
        """Test Guard abstract interface."""
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            Guard()

        # Test complete implementation
        class CompleteGuard(Guard):
            name = "complete_guard"

            def validate(self, model, adapter, context):
                return {
                    "passed": context.get("should_pass", True),
                    "action": "none" if context.get("should_pass", True) else "warn",
                    "metrics": {"score": 0.95},
                }

        guard = CompleteGuard()
        assert guard.name == "complete_guard"

        result1 = guard.validate(Mock(), Mock(), {"should_pass": True})
        assert result1["passed"]
        assert result1["action"] == "none"

        result2 = guard.validate(Mock(), Mock(), {"should_pass": False})
        assert not result2["passed"]
        assert result2["action"] == "warn"


class TestTypeAliases:
    """Test type aliases for coverage."""

    def test_type_aliases_usage(self):
        """Test that type aliases work as expected."""
        # CalibrationData
        calib_data: CalibrationData = {"test": "data"}
        assert isinstance(calib_data, dict)

        calib_data2: CalibrationData = Mock()
        assert calib_data2 is not None

        # ModelType
        model: ModelType = Mock()
        assert model is not None

        # DeviceType
        device1: DeviceType = "cuda:0"
        device2: DeviceType = Mock()
        assert isinstance(device1, str)
        assert device2 is not None

        # MetricsDict
        metrics: MetricsDict = {
            "ppl": 3.2,
            "accuracy": 95,
            "model_name": "gpt2",
            "converged": True,
        }
        assert metrics["ppl"] == 3.2
        assert metrics["accuracy"] == 95
        assert metrics["model_name"] == "gpt2"
        assert metrics["converged"]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_guard_chain_edge_cases(self):
        """Test GuardChain with edge cases."""
        # Empty guard chain
        empty_chain = GuardChain([])

        results = empty_chain.prepare_all(Mock(), Mock(), Mock(), {})
        assert results == {}

        results = empty_chain.before_edit_all(Mock())
        assert results == []

        results = empty_chain.after_edit_all(Mock())
        assert results == []

        results = empty_chain.finalize_all(Mock())
        assert results == []

        assert empty_chain.all_passed([])

        # Chain with guards that have partial interfaces
        # Use spec to control which methods exist
        partial_guard = Mock(spec=["name", "before_edit"])
        partial_guard.name = "partial"
        partial_guard.before_edit = Mock(return_value="before_result")
        # No prepare, after_edit, finalize in spec

        chain = GuardChain([partial_guard])

        # prepare_all should handle missing prepare method
        prep_results = chain.prepare_all(Mock(), Mock(), Mock(), {})
        assert prep_results["partial"] == {"ready": True}

        # before_edit_all should call and include result
        before_results = chain.before_edit_all(Mock())
        assert before_results == ["before_result"]

        # after_edit_all should not call missing method
        after_results = chain.after_edit_all(Mock())
        assert after_results == []

        # finalize_all should not call missing method
        final_results = chain.finalize_all(Mock())
        assert final_results == []


if __name__ == "__main__":
    pytest.main([__file__])
