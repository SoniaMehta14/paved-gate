"""Read adopted harnesses out of a Harness² output directory.

Layout observed in the output of `harness2-demo` (harness2 commit 1cfe254):

    <output>/runs/<namespace>/<task>/harnesses/base/            baseline harness
    <output>/runs/<namespace>/<task>/harnesses/<label>/         every candidate, including the adopted one
    <output>/runs/<namespace>/<task>/selection/<sel>/selection.json   chosen_label, experience
    <output>/runs/<namespace>/<task>/reflection/<sel>/edits.md        reflection note on what each edit did
    <output>/runs/<namespace>/<task>/reflection/<sel>/deliverables/<label>/   task outputs as text
    <output>/library/.../harnesses/<task>/EXPERIENCE.md         the lesson retained for later tasks

Each namespace is one (benchmark, substrate, mode, k) setting, so one selection.json is one adoption.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

Tree = dict[str, str]

_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_INDEX = "INDEX.md"


class _Selection(BaseModel):
    """The fields of Harness² selection.json the gate relies on; the rest is ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    task_id: str
    mode: str
    k: int
    ok: bool
    chosen_label: str | None = None
    chosen_run_id: str | None = None
    chosen_is_base: bool = False
    experience: dict[str, str] | None = None


class RunDirError(ValueError):
    """The directory does not have the layout this reader expects."""


@dataclass(frozen=True)
class Adoption:
    task_id: str
    mode: str
    k: int
    namespace: str
    label: str
    is_base: bool
    task_dir: Path
    selection_path: Path
    baseline: Tree
    adopted: Tree
    baseline_outputs: Tree
    adopted_outputs: Tree
    reflection_note: Path | None
    lesson: Path | None
    experience: dict[str, str] | None


@dataclass(frozen=True)
class Skipped:
    selection_path: Path
    reason: str


def read_tree(root: Path) -> Tree:
    """Every file under `root` as text. Symlinks are recorded by target and never followed."""
    out: Tree = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if "__pycache__" in p.parts:
            continue
        if p.is_symlink():
            out[rel] = f"<symlink to {p.readlink()}>"
        elif p.is_file():
            out[rel] = p.read_text(encoding="utf-8", errors="replace")
    return out


def tree_sha256(tree: Tree) -> str:
    h = hashlib.sha256()
    for rel in sorted(tree):
        h.update(rel.encode() + b"\0" + tree[rel].encode() + b"\0")
    return "sha256:" + h.hexdigest()


def _outputs(deliverables: Path, labels: list[str]) -> Tree:
    out: Tree = {}
    for label in labels:
        d = deliverables / label
        if not d.is_dir():
            continue
        for rel, text in read_tree(d).items():
            if rel != _INDEX:
                out[f"{label}/{rel}"] = text
    return out


def _find_lesson(root: Path, run_id: str | None) -> Path | None:
    """The library EXPERIENCE.md written for this adoption: it names the adopted run id."""
    if not run_id:
        return None
    marker = f"(run `{run_id}`)"
    for p in sorted(root.rglob("EXPERIENCE.md")):
        if "library" in p.relative_to(root).parts and marker in p.read_text(encoding="utf-8", errors="replace"):
            return p
    return None


def _load(selection_path: Path, root: Path) -> Adoption | Skipped:
    try:
        sel = _Selection.model_validate_json(selection_path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        return Skipped(selection_path, f"unreadable selection.json: {exc.error_count()} validation errors")
    if not sel.ok or not sel.chosen_label:
        return Skipped(selection_path, "selection did not complete, nothing was adopted")
    if not _SAFE_LABEL.fullmatch(sel.chosen_label):
        return Skipped(selection_path, f"chosen_label {sel.chosen_label!r} is not a plain directory name")

    task_dir = selection_path.parents[2]
    harnesses = task_dir / "harnesses"
    base_dir, adopted_dir = harnesses / "base", harnesses / sel.chosen_label
    for d in (base_dir, adopted_dir):
        if not d.is_dir():
            raise RunDirError(f"{d} is missing; expected a harness directory next to {selection_path}")

    reflection = task_dir / "reflection" / selection_path.parent.name
    deliverables = reflection / "deliverables"
    base_labels = sorted(p.name for p in deliverables.glob("base*") if p.is_dir()) if deliverables.is_dir() else []
    note = reflection / "edits.md"
    return Adoption(
        task_id=sel.task_id,
        mode=sel.mode,
        k=sel.k,
        namespace=task_dir.parent.name,
        label=sel.chosen_label,
        is_base=sel.chosen_is_base,
        task_dir=task_dir,
        selection_path=selection_path,
        baseline=read_tree(base_dir),
        adopted=read_tree(adopted_dir),
        baseline_outputs=_outputs(deliverables, base_labels),
        adopted_outputs=_outputs(deliverables, [sel.chosen_label]),
        reflection_note=note if note.is_file() else None,
        lesson=_find_lesson(root, sel.chosen_run_id),
        experience=sel.experience,
    )


def discover(root: str | Path) -> list[Adoption | Skipped]:
    """Every adoption under a Harness² output directory (or any directory inside one)."""
    r = Path(root)
    if not r.is_dir():
        raise RunDirError(f"{r} is not a directory")
    found = sorted(r.rglob("selection/*/selection.json"))
    if not found:
        raise RunDirError(f"no selection/*/selection.json under {r}; is this a Harness² output directory?")
    return [_load(p, r) for p in found]
