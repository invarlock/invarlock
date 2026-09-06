"""Derive an authenticated K2 runtime source without the optional Outlines backend."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import posixpath
import stat
import tarfile
from pathlib import Path, PurePosixPath

COMMIT = "392841f47cb7ef214601eeb528906a0abba02471"
PREFIX = f"sglang-{COMMIT}/"
ARCHIVE_SHA256 = "24656522ac57de8f67262b7f46ef23a08ff0c9225d55d63cdbb76f02b1f152f9"
ARCHIVE_URL = f"https://codeload.github.com/sgl-project/sglang/tar.gz/{COMMIT}"
DERIVED_VERSION = "0.0.0.dev0+invarlock.k2.392841f.1"
PATCH_HASHES = {
    "python/pyproject.toml": "79a8240d85e310935bb64995290efeeb1e69c1f61f466b19882c08ac544482be",
    "python/sglang/srt/constrained/outlines_backend.py": "ffef75513c79b201136730343165a9ee7444e159faa50d4c7f77bbbb95470dda",
    "python/sglang/srt/constrained/outlines_jump_forward.py": "766dcf56cc31da813f25f02ffad3ddee2acc23de550aae1c653f8017e3a1c77b",
}
BLOCKED_MODULES = tuple(name for name in PATCH_HASHES if name.endswith(".py"))
LINKS = {
    ".dockerignore": ".gitignore",
    "python/sglang/srt/mem_cache/cpp_radix_tree/.clang-format": "../../../kernels/aot/.clang-format",
    "sgl-model-gateway/LICENSE": "../LICENSE",
}
DEPENDENCY = b'  "outlines==0.1.11",\n'


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _path(name):
    if not name.startswith(PREFIX):
        raise ValueError("source member path has an unexpected root")
    relative = name[len(PREFIX) :]
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != relative
    ):
        raise ValueError("source member path is not canonical")
    return relative


def _read_archive(data):
    files, modes, links, names = {}, {}, {}, set()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 20000 or sum(m.size for m in members) > 256 * 1024 * 1024:
            raise ValueError("source archive exceeds expanded size or member bound")
        for member in members:
            if member.name.rstrip("/") == PREFIX.rstrip("/") and member.isdir():
                continue
            name = _path(member.name.rstrip("/") if member.isdir() else member.name)
            if name in names:
                raise ValueError("duplicate source member path")
            names.add(name)
            if member.isdir():
                continue
            if member.issym() and LINKS.get(name) == member.linkname:
                links[name] = member.linkname
            elif member.isfile():
                files[name] = archive.extractfile(member).read()
                modes[name] = 0o755 if member.mode & 0o111 else 0o644
            else:
                raise ValueError("unsupported source member type or symbolic link")
    for name, link in links.items():
        target = posixpath.normpath(posixpath.join(posixpath.dirname(name), link))
        if target not in files:
            raise ValueError("source symbolic link target is not a regular member")
        files[name], modes[name] = files[target], modes[target]
    return files, modes, links


def prepare(archive: Path, output: Path):
    """Authenticate first; change exactly three reviewed files in a fresh tree."""
    if output.exists():
        raise FileExistsError(output)
    descriptor = os.open(archive, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("source archive input must be a regular file")
        data = stream.read(64 * 1024 * 1024 + 1)
    if len(data) > 64 * 1024 * 1024 or _hash(data) != ARCHIVE_SHA256:
        raise ValueError("source archive identity differs from reviewed bytes")
    files, modes, links = _read_archive(data)
    for name, expected in PATCH_HASHES.items():
        if name not in files or _hash(files[name]) != expected:
            raise ValueError(f"reviewed file identity differs: {name}")
    before = {name: _hash(data) for name, data in sorted(files.items())}
    metadata = files["python/pyproject.toml"]
    if metadata.count(DEPENDENCY) != 1:
        raise ValueError("expected exactly one optional Outlines dependency")
    files["python/pyproject.toml"] = metadata.replace(DEPENDENCY, b"", 1)
    for name in BLOCKED_MODULES:
        # Retain the upstream license header; neither module imports dependencies.
        header = files[name].split(b'"""', 1)[0]
        files[name] = header + (
            b'"""Excluded optional Outlines backend in the restricted K2 runtime."""\n\n'
            b'raise RuntimeError("Outlines grammar and disk caching are unavailable in the restricted K2 runtime")\n'
        )
    after = {name: _hash(data) for name, data in sorted(files.items())}
    changed = {
        name: {"upstream_sha256": before[name], "derived_sha256": after[name]}
        for name in files
        if before[name] != after[name]
    }
    if set(changed) != set(PATCH_HASHES):
        raise ValueError("source transformation changed an unexpected file set")
    manifest = {
        "format": "invarlock/k2-source-derivation-v1",
        "status": "source_prepared_not_runtime_ready",
        "source_commit": COMMIT,
        "source_archive_url": ARCHIVE_URL,
        "source_archive_sha256": ARCHIVE_SHA256,
        "derived_distribution_version": DERIVED_VERSION,
        "changed_files": changed,
        "reified_symbolic_links": links,
        "upstream_files": before,
        "derived_files": after,
        "excluded_operations": ["Outlines grammar backend", "Outlines disk caching"],
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in files.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(payload)
        destination.chmod(modes[name])
    with (output / "source-derivation.json").open("x") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.archive, args.output)
    except (ValueError, OSError, tarfile.TarError) as error:
        parser.exit(2, f"K2 source derivation: {error}\n")
    print(
        json.dumps(
            {
                "source_commit": result["source_commit"],
                "changed_files": sorted(result["changed_files"]),
                "status": result["status"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
