"""Consistent Ed25519 verification at evidence and receipt trust boundaries."""

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

_FIELD = 2**255 - 19
_Y_MASK = 2**255 - 1
# The five distinct y coordinates of the eight cofactor-subgroup points.
# Ignore the x sign and reduce y so alternate encodings cannot bypass the guard.
# Coordinates: libsodium 1.0.20, ge25519_has_small_order:
# https://github.com/jedisct1/libsodium/blob/1.0.20/src/libsodium/crypto_core/ed25519/ref10/ed25519_ref10.c
_SMALL_ORDER_Y = frozenset(
    {
        0,
        1,
        _FIELD - 1,
        0x5FC536D880238B13933C6D305ACDFD5F098EFF289F4C345B027B2C28F95E826,
        0x7A03AC9277FDC74EC6CC392CFA53202A0F67100D760B3CBA4FD84D3D706A17C7,
    }
)


def _small_order(encoded: bytes) -> bool:
    return (int.from_bytes(encoded, "little") & _Y_MASK) % _FIELD in _SMALL_ORDER_Y


def verify_ed25519_signature(
    key: Ed25519PublicKey, signature: bytes, message: bytes
) -> None:
    """Reject small-order public/R points before the cryptographic verification.

    A trusted fingerprint does not make a degenerate public key a valid signer.
    The cryptographic library still verifies the full signature equation and
    scalar encoding. This guard adds no signing or secret-dependent arithmetic.
    """
    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    if len(signature) != 64 or _small_order(raw) or _small_order(signature[:32]):
        raise InvalidSignature
    key.verify(signature, message)
