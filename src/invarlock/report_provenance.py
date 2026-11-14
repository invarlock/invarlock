"""Complete local renderer identities and deterministic transport archives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def identity(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def renderer_identity(source_root: Path | None = None) -> dict[str, str]:
    root = source_root or Path(__file__).parent
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and p.suffix in {".py", ".cjs", ".json"}
        and "__pycache__" not in p.parts
    }


def write_archive(root: Path, output: Path, names: list[str]) -> None:
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
        for name in sorted(names):
            entry = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, (root / name).read_bytes())
