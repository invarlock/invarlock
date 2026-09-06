"""Replay the selected Ubuntu package artifacts through signed release indexes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import os
import re
import stat
import subprocess
import tempfile
import urllib.request
from pathlib import Path

KEYRING_SHA256 = "80a36b0a6de2f69f49d2df75ef473ccde121e9e190b9ea01d20a4f63778d5c31"
RELEASES = {
    f"{host}_ubuntu_dists_{suite}_InRelease": f"https://{host}/ubuntu/dists/{suite}"
    for host, suite in (
        ("archive.ubuntu.com", "noble"),
        ("archive.ubuntu.com", "noble-updates"),
        ("archive.ubuntu.com", "noble-backports"),
        ("security.ubuntu.com", "noble-security"),
    )
}
COMPONENTS = ("main", "restricted", "universe", "multiverse")
INDEX_LIMIT = 256 * 1024 * 1024


def read_input(path, limit):
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("build input is not a bounded regular file")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("build input exceeds size bound")
    return data


def release_entries(data):
    entries, active = {}, False
    for line in data.decode().splitlines():
        if line == "SHA256:":
            active = True
            continue
        if not line.startswith(" "):
            active = False
        if active:
            matched = re.fullmatch(r" ([0-9a-f]{64})\s+([0-9]+) (\S+)", line)
            if matched is None:
                raise ValueError("invalid signed SHA256 entry")
            digest, size, name = matched.groups()
            if name in entries:
                raise ValueError("duplicate signed index path")
            entries[name] = (digest, int(size))
    if not entries:
        raise ValueError("signed release has no SHA256 index section")
    return entries


def verify_signature(release, keyring):
    if hashlib.sha256(keyring).hexdigest() != KEYRING_SHA256:
        raise ValueError("Ubuntu keyring differs from the independently pinned base")
    begin = b"-----BEGIN PGP SIGNED MESSAGE-----\n"
    signature = b"-----BEGIN PGP SIGNATURE-----"
    end = b"-----END PGP SIGNATURE-----\n"
    if (
        len(release) > 1024 * 1024
        or not release.startswith(begin)
        or not release.endswith(end)
        or release.count(begin) != 1
        or release.count(signature) != 1
        or release.count(end) != 1
    ):
        raise ValueError("release requires exactly one bounded clear-signed message")
    with tempfile.TemporaryDirectory(prefix="k2-apt-signature-") as temporary:
        root = Path(temporary)
        (root / "keyring.gpg").write_bytes(keyring)
        (root / "InRelease").write_bytes(release)
        checked = subprocess.run(
            [
                "gpgv",
                "--output",
                "-",
                "--homedir",
                str(root),
                "--keyring",
                str(root / "keyring.gpg"),
                str(root / "InRelease"),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        if len(checked.stdout) > 1024 * 1024:
            raise ValueError("verified release plaintext exceeds size bound")
        return checked.stdout


def signed_indexes(bundle):
    keyring = read_input(
        bundle / "package-indexes/ubuntu-archive-keyring.gpg", 1024 * 1024
    )
    for name, base in RELEASES.items():
        release = read_input(bundle / "repository-metadata" / name, 1024 * 1024)
        plaintext = verify_signature(release, keyring)
        fields = {}
        for line in plaintext.decode().splitlines():
            field, separator, value = line.partition(": ")
            if field in ("Origin", "Label", "Suite", "Codename"):
                if not separator or field in fields:
                    raise ValueError("ambiguous signed release identity")
                fields[field] = value
        if fields != {
            "Origin": "Ubuntu",
            "Label": "Ubuntu",
            "Suite": base.rsplit("/", 1)[1],
            "Codename": "noble",
        }:
            raise ValueError(
                "signed release identity differs from the expected Ubuntu suite"
            )
        entries = release_entries(plaintext)
        for component in COMPONENTS:
            relative = f"{component}/binary-amd64/Packages"
            raw, compressed = entries[relative], entries[relative + ".xz"]
            if raw[1] > INDEX_LIMIT or compressed[1] > 64 * 1024 * 1024:
                raise ValueError("signed index exceeds the declared size bound")
            filename = name.removesuffix("_InRelease") + f"_{component}_Packages.xz"
            url = f"{base}/{component}/binary-amd64/by-hash/SHA256/{compressed[0]}"
            yield (
                filename,
                url,
                raw,
                compressed,
                name,
                hashlib.sha256(release).hexdigest(),
            )


def decode_index(data, expected):
    if expected[1] > INDEX_LIMIT:
        raise ValueError("index size exceeds bound")
    chunks, total, streams = [], 0, 0
    while data:
        streams += 1
        if streams > 64:
            raise ValueError("too many concatenated index streams")
        decoder = lzma.LZMADecompressor(memlimit=64 * 1024 * 1024)
        chunk = decoder.decompress(data, max_length=expected[1] - total + 1)
        total += len(chunk)
        if not decoder.eof or total > expected[1]:
            raise ValueError("uncompressed package index identity differs")
        chunks.append(chunk)
        data = decoder.unused_data
    decoded = b"".join(chunks)
    if (
        len(decoded) != expected[1]
        or hashlib.sha256(decoded).hexdigest() != expected[0]
    ):
        raise ValueError("uncompressed package index identity differs")
    return decoded


def package_records(data):
    fields, size = {}, 0
    for raw in io.BytesIO(data):
        size += len(raw)
        if size > 1024 * 1024:
            raise ValueError("package record exceeds size bound")
        line = raw.decode().rstrip("\n")
        if not line:
            if fields:
                yield fields
            fields, size = {}, 0
        elif not line.startswith((" ", "\t")):
            name, separator, value = line.partition(":")
            if (
                not separator
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", name)
                or name in fields
            ):
                raise ValueError("malformed or duplicate package field")
            fields[name] = value.lstrip(" \t")
    if fields:
        yield fields


def verify(bundle, selected):
    """Return only identities replayed against the pinned Ubuntu signing keyring."""
    package_table = read_input(bundle / "deb-packages.tsv", 1024 * 1024)
    package_rows = package_table.decode().splitlines()
    requested = {}
    for line in package_rows:
        values = line.split("\t")
        if len(values) != 4 or values[0] not in selected or values[0] in requested:
            raise ValueError("OS package identity table is ambiguous")
        requested[values[0]] = tuple(values[1:])
    if set(requested) != set(selected):
        raise ValueError("OS package identity table differs from selected artifacts")
    wanted = set(requested.values())
    found, indexes = {}, []
    allowed = {"ubuntu-archive-keyring.gpg"}
    for name, _, raw, compressed, release, release_hash in signed_indexes(bundle):
        allowed.add(name)
        payload = read_input(bundle / "package-indexes" / name, 64 * 1024 * 1024)
        if (hashlib.sha256(payload).hexdigest(), len(payload)) != compressed:
            raise ValueError("compressed package index identity differs")
        records = package_records(decode_index(payload, raw))
        for record in records:
            identity = tuple(
                record.get(k) for k in ("Package", "Version", "Architecture")
            )
            if identity in wanted:
                digest, size = record.get("SHA256"), record.get("Size")
                if (
                    not isinstance(digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    or not isinstance(size, str)
                    or not size.isdecimal()
                ):
                    raise ValueError("selected signed package lacks a valid identity")
                found.setdefault(identity, []).append(
                    {
                        "sha256": digest,
                        "size_bytes": int(size),
                        "index": name,
                        "filename": record["Filename"],
                    }
                )
        indexes.append(
            {
                "path": name,
                "sha256": compressed[0],
                "uncompressed_sha256": raw[0],
                "release": release,
                "release_sha256": release_hash,
            }
        )
    if {p.name for p in (bundle / "package-indexes").iterdir()} != allowed:
        raise ValueError("unexpected retained package index artifacts")
    verified = []
    for name, identity in requested.items():
        data = selected[name]
        expected = hashlib.sha256(data).hexdigest(), len(data)
        matches = [
            entry
            for entry in found.get(identity, [])
            if (entry["sha256"], entry["size_bytes"]) == expected
        ]
        if not matches:
            raise ValueError(f"selected package lacks a signed index link: {name}")
        verified.append(
            {
                "artifact": name,
                "package": identity[0],
                "version": identity[1],
                "architecture": identity[2],
                **matches[0],
            }
        )
    return {
        "format": "invarlock/k2-ubuntu-artifact-replay-v1",
        "status": "signed_indexes_and_payloads_verified",
        "keyring_sha256": KEYRING_SHA256,
        "package_table_sha256": hashlib.sha256(package_table).hexdigest(),
        "indexes": indexes,
        "packages": verified,
    }


def fetch(bundle):
    """Fetch exact index hashes from already retained and authenticated releases."""
    for name, url, raw, compressed, _, _ in signed_indexes(bundle):
        destination = bundle / "package-indexes" / name
        if destination.exists():
            raise FileExistsError(destination)
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(compressed[1] + 1)
        if (hashlib.sha256(data).hexdigest(), len(data)) != compressed:
            raise ValueError("downloaded package index identity differs")
        decode_index(data, raw)
        with destination.open("xb") as stream:
            stream.write(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        fetch(args.bundle)
    except (
        ValueError,
        KeyError,
        OSError,
        subprocess.SubprocessError,
        lzma.LZMAError,
    ) as error:
        parser.exit(2, f"Ubuntu artifact replay: {error}\n")
    print(json.dumps({"status": "signed_indexes_fetched_not_package_replayed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
