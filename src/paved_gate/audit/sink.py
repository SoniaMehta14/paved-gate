from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

from paved_gate.audit.models import AuditRecord
from paved_gate.policy.models import AuditConfig


@runtime_checkable
class AuditSink(Protocol):
    async def write(self, record: AuditRecord) -> None: ...


class JsonlFileSink:
    """Append-only JSON Lines file. One record per line, serialised under a lock."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _append(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    async def write(self, record: AuditRecord) -> None:
        line = record.model_dump_json()
        async with self._lock:
            await asyncio.to_thread(self._append, line)


class StdoutSink:
    async def write(self, record: AuditRecord) -> None:
        sys.stdout.write(record.model_dump_json() + "\n")


class MemorySink:
    """Keeps records in memory; for tests and embedding in other telemetry pipelines."""

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    async def write(self, record: AuditRecord) -> None:
        self.records.append(record)


def sink_from_config(cfg: AuditConfig) -> AuditSink:
    if cfg.sink == "stdout":
        return StdoutSink()
    return JsonlFileSink(cfg.path)
