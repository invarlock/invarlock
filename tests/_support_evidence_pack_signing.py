"""Test-only helpers for constructing signed evidence-pack fixtures.

The public package is verifier-only. Tests still need independently generated
signatures to exercise its trust boundary, so all private-key operations live
under the test tree and are excluded from wheels.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

SIGNATURE_FORMAT = "invarlock/evidence-pack-signature-v1"


def public_key_fingerprint(public_key: ed25519.Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def generate_signing_keypair(
    private_key_path: Path,
    *,
    public_key_path: Path,
) -> str:
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_key_path.parent.mkdir(parents=True, exist_ok=True)
    public_key_path.parent.mkdir(parents=True, exist_ok=True)
    private_key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    private_key_path.chmod(0o600)
    public_key_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return public_key_fingerprint(public_key)


def _load_private_key(path: Path) -> ed25519.Ed25519PrivateKey:
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, ed25519.Ed25519PrivateKey):
        raise TypeError("fixture signing key must be Ed25519")
    return key


def sign_manifest(
    manifest_path: Path,
    *,
    signing_key_path: Path,
    signature_path: Path | None = None,
) -> str:
    private_key = _load_private_key(signing_key_path)
    return _write_signature(
        manifest_path,
        private_key=private_key,
        signature_path=signature_path,
    )


def sign_manifest_ephemeral(
    manifest_path: Path,
    *,
    record_manifest_fingerprint: bool = True,
    signature_path: Path | None = None,
) -> str:
    private_key = ed25519.Ed25519PrivateKey.generate()
    fingerprint = public_key_fingerprint(private_key.public_key())
    if record_manifest_fingerprint:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("fixture manifest must be a JSON object")
        payload["signing_key_fingerprint"] = fingerprint
        manifest_path.write_text(
            json.dumps(payload, allow_nan=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _write_signature(
        manifest_path,
        private_key=private_key,
        signature_path=signature_path,
    )
    return fingerprint


def _write_signature(
    manifest_path: Path,
    *,
    private_key: ed25519.Ed25519PrivateKey,
    signature_path: Path | None,
) -> str:
    public_key = private_key.public_key()
    fingerprint = public_key_fingerprint(public_key)
    signature = private_key.sign(manifest_path.read_bytes())
    bundle = {
        "format": SIGNATURE_FORMAT,
        "algorithm": "ed25519",
        "signing_key_fingerprint": fingerprint,
        "public_key": {
            "encoding": "pem",
            "value": public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode("ascii"),
        },
        "signature": {
            "encoding": "base64",
            "value": base64.b64encode(signature).decode("ascii"),
        },
    }
    target = signature_path or manifest_path.with_name("manifest.signature.json")
    target.write_text(
        json.dumps(bundle, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return fingerprint


def main() -> int:
    parser = argparse.ArgumentParser(description="Sign an evidence-pack test fixture")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    sign_manifest_ephemeral(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
