#!/usr/bin/env python3
"""Generate separate Ed25519 keys for the runnable local example."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


def _fingerprint(key: ed25519.Ed25519PublicKey) -> str:
    raw = key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _write_key(path: Path) -> str:
    key = ed25519.Ed25519PrivateKey.generate()
    payload = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return _fingerprint(key.public_key())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "create independent evidence-signer and verifier Ed25519 keys for the "
            "offline example"
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".keys"),
        help="new directory for keys and public fingerprint files (default: .keys)",
    )
    parser.add_argument(
        "--role",
        choices=("both", "evidence-signer", "verifier"),
        default="both",
        help="key role to create (default: both)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    try:
        fingerprints: dict[str, str] = {}
        roles = ("evidence-signer", "verifier") if args.role == "both" else (args.role,)
        for role in roles:
            fingerprint = _write_key(output_dir / f"{role}.pem")
            (output_dir / f"{role}.fingerprint").write_text(
                fingerprint + "\n", encoding="ascii"
            )
            fingerprints[role] = fingerprint
    except BaseException:
        for child in output_dir.iterdir():
            child.unlink()
        output_dir.rmdir()
        raise

    for role, fingerprint in fingerprints.items():
        print(f"{role} key:         {output_dir / f'{role}.pem'}")
        print(f"{role} fingerprint: {fingerprint}")
    print("Distribute the fingerprint files, never the private PEM files.")


if __name__ == "__main__":
    main()
