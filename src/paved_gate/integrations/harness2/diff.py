"""File level diff between a baseline harness and a composed harness."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Literal

from paved_gate.integrations.harness2.run_dir import Tree

ChangeKind = Literal["added", "removed", "modified"]


@dataclass(frozen=True)
class FileChange:
    path: str
    kind: ChangeKind
    before: str | None
    after: str | None
    lines_added: int
    lines_removed: int
    unified: str


def _change(path: str, before: str | None, after: str | None) -> FileChange:
    a = (before or "").splitlines(keepends=True)
    b = (after or "").splitlines(keepends=True)
    diff = list(difflib.unified_diff(a, b, f"baseline/{path}", f"adopted/{path}"))
    added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
    kind: ChangeKind = "added" if before is None else "removed" if after is None else "modified"
    unified = "".join(line if line.endswith("\n") else line + "\n" for line in diff)
    return FileChange(path, kind, before, after, added, removed, unified)


def diff_trees(baseline: Tree, adopted: Tree) -> list[FileChange]:
    return [
        _change(path, baseline.get(path), adopted.get(path))
        for path in sorted(baseline.keys() | adopted.keys())
        if baseline.get(path) != adopted.get(path)
    ]
