"""Checkpointing and resume helpers for the full experiment pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PIPELINE_VERSION = 1

PIPELINE_STAGES: tuple[str, ...] = (
    "train-oracle",
    "prepare-tournament-oracle",
    "train-tournament-players",
    "run-tournament",
    "analyze-tournament",
    "score-blunders",
)

STAGE_ALIASES: dict[str, str] = {
    "oracle": "train-oracle",
    "prepare-oracle": "prepare-tournament-oracle",
    "players": "train-tournament-players",
    "tournament": "run-tournament",
    "analyze": "analyze-tournament",
    "blunders": "score-blunders",
}


@dataclass(frozen=True, slots=True)
class PipelineState:
    version: int
    completed_stages: tuple[str, ...]
    updated_at: str
    metadata: dict[str, Any]

    @classmethod
    def empty(cls) -> PipelineState:
        return cls(version=PIPELINE_VERSION, completed_stages=(), updated_at="", metadata={})

    def is_complete(self, stage_id: str) -> bool:
        return stage_id in self.completed_stages


def normalize_stage_id(stage: str) -> str:
    """Return a canonical stage id, accepting short aliases."""
    text = stage.strip().lower()
    if not text:
        raise ValueError("stage id cannot be empty")
    return STAGE_ALIASES.get(text, text)


def parse_skip_stages(values: list[str]) -> set[str]:
    """Parse repeated or comma-separated ``--skip-stage`` values."""
    stages: set[str] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            normalized = normalize_stage_id(part)
            if normalized not in PIPELINE_STAGES:
                allowed = ", ".join(PIPELINE_STAGES)
                raise ValueError(f"unknown pipeline stage {part!r}; expected one of: {allowed}")
            stages.add(normalized)
    return stages


def pipeline_state_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "experiment_pipeline.json"


def load_pipeline_state(output_dir: str | Path) -> PipelineState:
    path = pipeline_state_path(output_dir)
    if not path.is_file():
        return PipelineState.empty()
    payload = json.loads(path.read_text(encoding="utf-8"))
    completed = payload.get("completed_stages", [])
    if not isinstance(completed, list):
        completed = []
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return PipelineState(
        version=int(payload.get("version", PIPELINE_VERSION)),
        completed_stages=tuple(str(stage) for stage in completed),
        updated_at=str(payload.get("updated_at", "")),
        metadata=metadata,
    )


def mark_stage_complete(
    output_dir: str | Path,
    stage_id: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Record that ``stage_id`` finished successfully."""
    if stage_id not in PIPELINE_STAGES:
        raise ValueError(f"unknown pipeline stage: {stage_id}")

    path = pipeline_state_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = load_pipeline_state(output_dir)
    completed = list(state.completed_stages)
    if stage_id not in completed:
        completed.append(stage_id)

    merged_metadata = dict(state.metadata)
    if metadata:
        merged_metadata[stage_id] = metadata

    payload = {
        "version": PIPELINE_VERSION,
        "completed_stages": completed,
        "updated_at": datetime.now(UTC).isoformat(),
        "metadata": merged_metadata,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def stages_to_skip(
    *,
    explicit_skip: set[str],
    resume_pipeline: bool,
    state: PipelineState,
) -> set[str]:
    """Return the set of stages that should not run."""
    skipped = set(explicit_skip)
    if resume_pipeline:
        skipped.update(state.completed_stages)
    return skipped
