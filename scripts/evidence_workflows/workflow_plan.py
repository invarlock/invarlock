"""Typed workflow plan objects shared by evidence-running scripts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .workflow_state import WorkflowRunMetadata


@dataclass(frozen=True)
class WorkflowCommandStep:
    """A command planned for one lane of an evidence workflow."""

    name: str
    command: tuple[str, ...]
    log_mode: str = "a"
    output_path: Path | None = None
    retry_returncodes: tuple[int, ...] = ()
    retry_message: str | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "command": list(self.command),
        }
        if self.output_path is not None:
            payload["output_path"] = str(self.output_path)
        if self.retry_returncodes:
            payload["retry_returncodes"] = list(self.retry_returncodes)
        if self.retry_message:
            payload["retry_message"] = self.retry_message
        return payload


@dataclass(frozen=True)
class WorkflowLanePlan:
    """Renderer-neutral execution plan for one evidence lane."""

    slug: str
    lane_id: str
    model_id: str
    execution_mode: str
    lane_root: Path
    published_lane_root: Path
    report_path: Path
    verify_path: Path
    profile: str
    steps: tuple[WorkflowCommandStep, ...]
    resource_preflight: Mapping[str, object] | None = None
    prepared_preset: str | None = None

    @property
    def evaluate_step(self) -> WorkflowCommandStep:
        return self.step("evaluate")

    @property
    def verify_step(self) -> WorkflowCommandStep:
        return self.step("verify")

    def step(self, name: str) -> WorkflowCommandStep:
        for item in self.steps:
            if item.name == name:
                return item
        raise KeyError(f"workflow lane {self.slug!r} has no step {name!r}")

    def optional_step(self, name: str) -> WorkflowCommandStep | None:
        for item in self.steps:
            if item.name == name:
                return item
        return None

    def to_dry_run_entry(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "slug": self.slug,
            "lane_id": self.lane_id,
            "model_id": self.model_id,
            "execution_mode": self.execution_mode,
            "profile": self.profile,
            "resource_preflight": (
                dict(self.resource_preflight) if self.resource_preflight else None
            ),
            "steps": [step.to_payload() for step in self.steps],
            "evaluate": list(self.evaluate_step.command),
            "verify": list(self.verify_step.command),
        }
        prefetch = self.optional_step("prefetch")
        if prefetch is not None:
            payload["prefetch"] = list(prefetch.command)
        materialize = self.optional_step("materialize_dataset")
        if materialize is not None:
            payload["materialize_dataset"] = list(materialize.command)
        if self.prepared_preset is not None:
            payload["prepared_preset"] = self.prepared_preset
        return payload


@dataclass(frozen=True)
class WorkflowSweepPlan:
    """Resolved plan for an evidence workflow invocation."""

    metadata: WorkflowRunMetadata
    output_root: Path
    execution_root: Path
    lanes: tuple[WorkflowLanePlan, ...]

    def to_dry_run_payload(self) -> list[dict[str, object]]:
        return [lane.to_dry_run_entry() for lane in self.lanes]

    def to_manifest_payload(
        self,
        *,
        lane_entries: Sequence[Mapping[str, Any]],
        generated_at: str,
    ) -> dict[str, object]:
        payload = asdict(self.metadata)
        payload.update(
            {
                "generated_at": generated_at,
                "lanes": [dict(item) for item in lane_entries],
            }
        )
        return payload


__all__ = [
    "WorkflowCommandStep",
    "WorkflowLanePlan",
    "WorkflowSweepPlan",
]
