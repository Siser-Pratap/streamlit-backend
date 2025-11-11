from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


@dataclass
class ProcessedArtifact:
    """Container for processed file bytes and metadata."""

    job_id: str
    filename: str
    content: bytes
    created_at: datetime


class InMemoryArtifactStore:
    """Simple in-memory store for processed artifacts."""

    def __init__(self) -> None:
        self._items: Dict[str, ProcessedArtifact] = {}

    def put(self, artifact: ProcessedArtifact) -> None:
        self._items[artifact.job_id] = artifact

    def get(self, job_id: str) -> Optional[ProcessedArtifact]:
        return self._items.get(job_id)

    def delete(self, job_id: str) -> None:
        if job_id in self._items:
            del self._items[job_id]


# Shared store instance used by the FastAPI application
artifact_store = InMemoryArtifactStore()
