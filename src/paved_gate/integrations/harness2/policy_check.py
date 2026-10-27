"""Check each changed file of a composed harness against the adoption policy."""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Literal

from paved_gate.integrations.harness2.config import Access, AdoptionPolicy, SkillTriggerFormat, TriggerMode
from paved_gate.integrations.harness2.diff import FileChange
from paved_gate.integrations.harness2.run_dir import Tree

Result = Literal["ok", "review", "reject"]
_RESULT: dict[Access, Result] = {"editable": "ok", "review": "review", "protected": "reject"}
_TRIGGER_RESULT: dict[TriggerMode, Result] = {"allow": "ok", "review": "review", "deny": "reject"}
_DESCRIPTION = re.compile(r"^description:\s*(.+)$", re.M)


@dataclass(frozen=True)
class FileResult:
    change: FileChange
    component: str | None
    access: Access
    source: str  # component:<name> | rule:<pattern> | unmapped | skill_trigger
    result: Result
    reason: str


@dataclass(frozen=True)
class TriggerLink:
    line: str
    skill: str | None
    skill_file: str | None
    skill_added: bool
    skill_policy_ok: bool
    accepted: bool
    reason: str


@dataclass(frozen=True)
class PolicyCheck:
    files: list[FileResult]
    triggers: list[TriggerLink]
    trigger_note: str | None = None  # why the skill trigger exemption did not apply, when it was considered


def component_of(policy: AdoptionPolicy, path: str) -> str | None:
    for name, comp in policy.components.items():
        if any(fnmatchcase(path, pat) for pat in comp.paths):
            return name
    return None


def _resolve(policy: AdoptionPolicy, change: FileChange) -> FileResult:
    path = change.path
    component = component_of(policy, path)
    for rule in policy.rules:
        if fnmatchcase(path, rule.pattern):
            why = f" ({rule.reason})" if rule.reason else ""
            reason = f"{path}: path rule {rule.pattern!r} makes it {rule.access}{why}"
            return FileResult(change, component, rule.access, f"rule:{rule.pattern}", _RESULT[rule.access], reason)
    if component is None:
        access = policy.unmapped_files
        reason = f"{path}: matches no component; unmapped files are {access}"
        return FileResult(change, None, access, "unmapped", _RESULT[access], reason)
    access = policy.components[component].access
    reason = {
        "editable": f"{path}: {component} is editable",
        "review": f"{path}: {component} changes need human review",
        "protected": f"{path}: {component} is protected",
    }[access]
    return FileResult(change, component, access, f"component:{component}", _RESULT[access], reason)


def _trigger_regex(fmt: SkillTriggerFormat) -> re.Pattern[str]:
    rx = re.escape(fmt.template)
    rx = rx.replace(re.escape("{desc}"), r"(?P<desc>.+)", 1)
    rx = rx.replace(re.escape("{name}"), r"(?P<name>[A-Za-z0-9_][A-Za-z0-9_.-]*)", 1)
    return re.compile(rx)


def skill_description(skill_md: str, name: str) -> str:
    """The text Harness² puts in a trigger line for a skill (harness2/core/tree.py `autotrigger`)."""
    m = _DESCRIPTION.search(skill_md[:600])
    return m.group(1).strip().rstrip(".") if m else f"the {name} workflow applies"


def _appended_lines(change: FileChange) -> list[str] | None:
    """Lines appended the way Harness² appends a trigger, or None if anything else changed.

    Harness² strips trailing whitespace from the file, then adds a newline and the line, so
    trailing blank lines of the baseline may disappear. Every other byte must be unchanged.
    """
    before, after = (change.before or "").rstrip(), change.after or ""
    if before:
        if not after.startswith(before + "\n"):
            return None
        after = after[len(before) + 1 :]
    lines = after.splitlines()
    return lines if lines else None


def _check_triggers(
    policy: AdoptionPolicy, change: FileChange, adopted: Tree, results: list[FileResult]
) -> tuple[list[TriggerLink], str | None]:
    """(links, None) when the change is only valid trigger lines, else (links, why not)."""
    fmt = policy.skill_trigger_format
    lines = _appended_lines(change)
    if lines is None:
        return [], "the change edits or removes existing text"
    by_path = {r.change.path: r for r in results}
    rx = _trigger_regex(fmt)
    links: list[TriggerLink] = []
    seen: set[str] = set()
    for line in lines:
        m = rx.fullmatch(line)
        if m is None:
            links.append(TriggerLink(line, None, None, False, False, False, "does not match the trigger template"))
            continue
        name = m["name"]
        skill_file = fmt.skill_file.format(name=name)
        added = skill_file in by_path and by_path[skill_file].change.kind == "added"
        skill_dir = posixpath.dirname(skill_file)
        skill_results = [r for p, r in by_path.items() if p == skill_file or p.startswith(skill_dir + "/")]
        policy_ok = added and all(r.result != "reject" for r in skill_results)
        accepted = False
        if name in seen:
            reason = f"names skill {name!r} more than once"
        elif not added:
            reason = f"names skill {name!r} but {skill_file} is not added in this edit"
        elif m["desc"] != skill_description(adopted.get(skill_file, ""), name):
            reason = f"text differs from the description in {skill_file}"
        elif not policy_ok:
            reason = f"skill {name!r} is rejected by the policy"
        else:
            reason = f"matches the template for skill {name!r} added in this edit"
            accepted = True
        seen.add(name)
        links.append(TriggerLink(line, name, skill_file, added, policy_ok, accepted, reason))
    bad = [link for link in links if not link.accepted]
    return links, (f"{len(bad)} of {len(links)} added lines are not valid skill trigger lines" if bad else None)


def check_policy(policy: AdoptionPolicy, changes: list[FileChange], adopted: Tree) -> PolicyCheck:
    results = [_resolve(policy, c) for c in changes]
    trigger_path = policy.skill_trigger_format.file
    idx = next((i for i, r in enumerate(results) if r.change.path == trigger_path), None)
    if idx is None or results[idx].result == "ok" or results[idx].change.kind == "removed":
        return PolicyCheck(results, [])

    current = results[idx]
    links, why_not = _check_triggers(policy, current.change, adopted, results)
    if why_not is not None:
        reason = f"{current.reason}; not a pure skill trigger addition: {why_not}"
        results[idx] = FileResult(
            current.change, current.component, current.access, current.source, current.result, reason
        )
        return PolicyCheck(results, links, f"not a pure skill trigger addition: {why_not}")

    mode = policy.skill_trigger_lines
    names = ", ".join(link.skill or "" for link in links)
    result = _TRIGGER_RESULT[mode]
    reason = f"{trigger_path}: only skill trigger lines for {names}; skill_trigger_lines is {mode}"
    results[idx] = FileResult(current.change, current.component, current.access, "skill_trigger", result, reason)
    return PolicyCheck(results, links)
