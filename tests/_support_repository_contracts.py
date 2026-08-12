from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TARGET_HEADER = re.compile(r"^(?P<targets>[^\t\s][^:]*)\s*:(?P<suffix>.*)$")
_TARGET_NAME = re.compile(r"^[A-Za-z0-9_.%/-]+$")
_TARGET_VARIABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*(?:\?|\+|:)?=")


@dataclass(frozen=True)
class MakeTarget:
    name: str
    declarations: tuple[str, ...]
    text: str

    @property
    def prerequisites(self) -> tuple[str, ...]:
        for declaration in self.declarations:
            suffix = declaration.split(":", 1)[1].strip()
            if not suffix or _TARGET_VARIABLE.match(suffix):
                continue
            return tuple(suffix.split("##", 1)[0].split())
        return ()


@dataclass(frozen=True)
class MakefileContract:
    text: str

    @classmethod
    def read(cls, path: Path) -> MakefileContract:
        return cls(path.read_text(encoding="utf-8"))

    def target(self, name: str) -> MakeTarget:
        lines = self.text.splitlines(keepends=True)
        matching_headers = [
            index for index, line in enumerate(lines) if name in _target_names(line)
        ]
        if not matching_headers:
            raise AssertionError(f"Make target {name!r} not found")

        start = matching_headers[0]
        end = len(lines)
        for index in range(start + 1, len(lines)):
            names = _target_names(lines[index])
            if names and name not in names:
                end = index
                break

        declarations = tuple(
            lines[index].rstrip("\n") for index in matching_headers if index < end
        )
        return MakeTarget(name, declarations, "".join(lines[start:end]))


def _target_names(line: str) -> tuple[str, ...]:
    match = _TARGET_HEADER.match(line.rstrip("\n"))
    if match is None:
        return ()
    names = tuple(match.group("targets").split())
    if not names or not all(_TARGET_NAME.fullmatch(name) for name in names):
        return ()
    return names
