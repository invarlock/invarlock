from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from invarlock.evidence_pack_integrity import (
    generate_signing_keypair,
    load_private_signing_key,
    public_key_fingerprint,
    sign_manifest,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sign an evidence-pack manifest with a package-native Ed25519 key."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest.json.",
    )
    parser.add_argument(
        "--signing-key",
        help="Optional Ed25519 private key PEM. When omitted, an ephemeral key is generated.",
    )
    parser.add_argument(
        "--signature-out",
        help="Optional output path for manifest.signature.json.",
    )
    parser.add_argument(
        "--generate-ephemeral",
        action="store_true",
        help="Generate an ephemeral Ed25519 key when --signing-key is omitted.",
    )
    return parser.parse_args(argv)


def _load_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must decode to a JSON object")
    return payload


def _sign_with_key(
    manifest_path: Path,
    *,
    signing_key_path: Path,
    signature_path: Path | None,
) -> str:
    fingerprint = public_key_fingerprint(
        load_private_signing_key(signing_key_path).public_key()
    )
    payload = _load_manifest(manifest_path)
    payload["signing_key_fingerprint"] = fingerprint
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sign_manifest(
        manifest_path,
        signing_key_path=signing_key_path,
        signature_path=signature_path,
    )
    return fingerprint


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = Path(args.manifest)
    signature_path = Path(args.signature_out) if args.signature_out else None

    if args.signing_key:
        fingerprint = _sign_with_key(
            manifest_path,
            signing_key_path=Path(args.signing_key),
            signature_path=signature_path,
        )
        print(fingerprint)
        return 0

    if not args.generate_ephemeral:
        print(
            "either --signing-key or --generate-ephemeral is required",
            file=sys.stderr,
        )
        return 2

    with TemporaryDirectory(prefix="invarlock-evidence-pack-signing-") as tmp_dir:
        private_key_path = Path(tmp_dir) / "ephemeral-signing-key.pem"
        public_key_path = Path(tmp_dir) / "ephemeral-signing-key.pub.pem"
        generate_signing_keypair(
            private_key_path,
            public_key_path=public_key_path,
        )
        fingerprint = _sign_with_key(
            manifest_path,
            signing_key_path=private_key_path,
            signature_path=signature_path,
        )
    print(fingerprint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
